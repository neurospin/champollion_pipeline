#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper script to generate sulcal embeddings per region.
Calls champollion/evaluate.py for each region in models_path.
"""

import argparse
import cProfile
import os
import pstats
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime
from io import StringIO
from os.path import abspath, dirname, exists, join
from pathlib import Path

from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.derivatives_layout import (
    DEFAULT_MASKS_VERSION,
    compute_models_cache_dir,
    compute_region_embeddings_dir,
    compute_run_log_derivatives_dir,
    compute_run_log_path,
)
from champollion_pipeline.models.cache import (
    MODEL_SET_FILE_NAME,
    ensure_models_cache_dir,
    read_model_set,
    write_json_atomic,
)
from champollion_pipeline.models.cache import ModelSet as ModelSet
from champollion_pipeline.models.cache import write_model_set as write_model_set
from champollion_pipeline.models.checkpoint import (
    DEFAULT_CHECKPOINT_EPOCH,
    check_region_weights,
    evaluation_model_dir,
    find_region_model_dirs,
    resolve_region_name,
    uses_best_weights_stand_in,
    weights_source,
)
from champollion_pipeline.models.fetch import HuggingFaceStrategy as HuggingFaceStrategy
from champollion_pipeline.models.fetch import InteractiveFallbackStrategy as InteractiveFallbackStrategy
from champollion_pipeline.models.fetch import LocalPathStrategy as LocalPathStrategy
from champollion_pipeline.models.fetch import ModelFetchStrategy as ModelFetchStrategy
from champollion_pipeline.models.fetch import RemoteArchiveStrategy as RemoteArchiveStrategy
from champollion_pipeline.models.fetch import fetch_model_set
from champollion_pipeline.process_setup import init_pipeline_process
from champollion_pipeline.utils.lib import CORTICAL_TILES_VERSION

# Add champollion to path for CKA imports
_SCRIPT_DIR = dirname(abspath(__file__))
PROVENANCE_FILE_NAME = "provenance.json"
_CHAMPOLLION_DIR = abspath(join(_SCRIPT_DIR, "..", "..", "external", "champollion_V1"))
if _CHAMPOLLION_DIR not in sys.path:
    sys.path.insert(0, _CHAMPOLLION_DIR)

from champollion.metrics.cka_coherence import test_models_coherence_from_directory  # noqa: E402


def _parse_non_negative_int(value: str) -> int:
    """argparse type: an integer >= 0 (Lightning numbers epochs from 0)."""
    try:
        number = int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {value!r}") from err
    if number < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {value!r}")
    return number


def read_pipeline_commit() -> str | None:
    """40-hex git commit of the champollion_pipeline source checkout, or None when git cannot report it."""
    repo_dir = dirname(abspath(__file__))
    try:
        result = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    if result.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", commit):
        return None
    return commit


class GenerateEmbeddings(ScriptBuilder):
    """Script for generating sulcal embeddings per region via champollion/evaluate.py."""

    def __init__(self):
        super().__init__(
            script_name="generate_embeddings",
            description="Generate sulcal embeddings per region using champollion/evaluate.py.",
        )
        (
            self.add_argument(
                "models_path", type=str, help="Path to the directory containing per-region model folders."
            )
            .add_argument(
                "datasets_root",
                type=str,
                help="Absolute path to the dataset root directory (e.g. /my/path/to/DATASET/).",
            )
            .add_flag("--overwrite", "Recompute embeddings that already exist on disk.")
            .add_flag("--cpu", "Force CPU usage (disable CUDA).")
            .add_flag("--profiling", "Enable Python profiling (cProfile).")
            .add_flag("--run-cka", "Run CKA coherence test after embeddings.")
            .add_flag("--no-cache", "Force re-extraction of archive (ignore cache).")
            .add_optional_argument(
                "--checkpoint-epoch",
                "Training epoch N (0-based) of the Lightning checkpoint epoch=N-step=*.ckpt to evaluate each region "
                "with. Default: 80.",
                default=DEFAULT_CHECKPOINT_EPOCH,
                type_=_parse_non_negative_int,
            )
            .add_optional_argument(
                "--cortical_version",
                f"Derivatives folder name (e.g. 'cortical_tiles-2027'). "
                f"Defaults to cortical_tiles-{CORTICAL_TILES_VERSION}.",
                default=f"cortical_tiles-{CORTICAL_TILES_VERSION}",
            )
            .add_flag(
                "--legacy",
                "Use deep_folding-2025 as the derivatives folder. Shorthand for --cortical_version deep_folding-2025.",
            )
            .add_optional_argument(
                "--model-version",
                "Model-set subfolder to download from the Hugging Face models repository (e.g. 'canonical_25'). "
                "Ignored when models_path is a local directory.",
                default=None,
            )
            .add_optional_argument(
                "--masks-version",
                "Deprecated alias of --model-version; same value and behaviour.",
                default=None,
            )
            .add_optional_argument(
                "--masks",
                "Cortical tiles mask version used as crops subdirectory (e.g. 'canonical_25'). "
                "Defaults to the --model-version value when given, else canonical_25.",
                default=None,
            )
            .add_flag(
                "--allow-version-mismatch",
                "Run even when --masks and --model-version differ (crops of one mask version evaluated with the "
                "model set of another); a warning is emitted.",
            )
            .add_argument(
                "--regions",
                type=str,
                nargs="+",
                default=None,
                help=(
                    "Restrict embedding generation to specific region names "
                    "(e.g. SCsylv_left FIPFIPPoCinf_right). "
                    "By default all regions found in models_path are processed."
                ),
            )
            .add_optional_argument(
                "--output",
                "Output base directory override. "
                "Defaults to {datasets_root}/derivatives/champollion_V1/{masks}/region_embeddings/.",
                default=None,
            )
            .add_optional_argument(
                "--subjects",
                "Deprecated and ignored. Each region's subjects file is read from "
                "{crops}/<crop>/mask/{L|R}skeleton_subject.csv next to its skeleton.",
                default=None,
            )
            .add_optional_argument(
                "--models-cache",
                "Models cache directory; relative paths resolve against the cwd. "
                "Defaults to <datasets_root>/derivatives/champollion_V1/models_cache.",
                default=None,
            )
        )
        self._models_source: dict | None = None

    def _make_regions_tmpdir(self, models_path: str) -> str:
        """Return a temp dir containing symlinks to only the requested region subdirs."""
        missing = [r for r in self.args.regions if not os.path.isdir(join(models_path, r))]
        if missing:
            raise FileNotFoundError(f"Regions not found in {models_path}: {', '.join(missing)}")
        tmpdir = tempfile.mkdtemp(prefix="champollion_regions_")
        for region in self.args.regions:
            os.symlink(join(models_path, region), join(tmpdir, region))
        return tmpdir

    def _get_derivatives_folder(self) -> str:
        """Return the derivatives folder to use, resolving --legacy and --cortical_version."""
        if getattr(self.args, "legacy", False):
            return "deep_folding-2025"
        return self.args.cortical_version

    def _get_crops_2mm_dir(self, datasets_root: str) -> str:
        """Return the 2mm crops directory, resolving --legacy and --masks."""
        derivatives = join(datasets_root, "derivatives", self._get_derivatives_folder())
        if getattr(self.args, "legacy", False):
            return join(derivatives, "crops", "2mm")
        return join(derivatives, "crops", self.args.masks, "2mm")

    def _patch_config_paths(self, config_path: str, target_folder: str) -> None:
        """Rewrite the derivatives folder in every YAML under config_path in-place."""

        pattern = re.compile(r"(derivatives/)([^/\n]+)(/crops(?:/[^/\n]+)*/2mm)")
        patched = 0
        for yaml_file in Path(config_path).rglob("*.yaml"):
            text = yaml_file.read_text()
            new_text = pattern.sub(lambda m: f"{m.group(1)}{target_folder}{m.group(3)}", text)
            if new_text != text:
                yaml_file.write_text(new_text)
                patched += 1
        if patched:
            print(f"Patched {patched} YAML file(s): derivatives folder → {target_folder}")

    def _compute_extract_dir(self) -> str:
        """Return the absolute models cache dir: --models-cache if given, else the dataset-tree default."""
        override = getattr(self.args, "models_cache", None)
        if override is not None:
            return abspath(override)
        return compute_models_cache_dir(self.args.datasets_root)

    def parse_args(self, args: list[str] | None = None) -> argparse.Namespace:
        """Parse args, then resolve the deprecated --masks-version alias into model_version."""
        parsed = super().parse_args(args)
        self._resolve_model_version_alias()
        self._resolve_masks_default()
        self._check_version_mismatch()
        self._check_local_model_set()
        return parsed

    def _resolve_model_version_alias(self) -> None:
        """Copy --masks-version into model_version with a FutureWarning; exit 2 if the two flags differ."""
        alias = self.args.masks_version
        if alias is None:
            return
        model_version = self.args.model_version
        if model_version is not None and model_version != alias:
            self.parser.error(
                f"--model-version {model_version!r} and --masks-version {alias!r} differ; --masks-version is a "
                "deprecated alias of --model-version, pass only --model-version"
            )
        warnings.warn(
            "--masks-version is deprecated; use --model-version instead "
            "(same value, the Hugging Face model-set subfolder).",
            FutureWarning,
            stacklevel=2,
        )
        self.args.model_version = alias

    def _resolve_masks_default(self) -> None:
        """Fill --masks when it was not given: the model version if set, else DEFAULT_MASKS_VERSION."""
        if self.args.masks is None:
            model_version = self.args.model_version
            self.args.masks = model_version if model_version is not None else DEFAULT_MASKS_VERSION

    def _check_version_mismatch(self) -> None:
        """Exit 2 when --masks and --model-version differ, unless --allow-version-mismatch (then warn)."""
        model_version, masks = self.args.model_version, self.args.masks
        if model_version is None or masks == model_version:
            return
        if not self.args.allow_version_mismatch:
            self.parser.error(
                f"--masks {masks!r} and --model-version {model_version!r} differ: --masks picks the input crops, "
                "--model-version the model set; pass --allow-version-mismatch to run with both"
            )
        warnings.warn(
            f"--masks {masks!r} and --model-version {model_version!r} differ; running anyway "
            f"(--allow-version-mismatch): crops {masks!r}, models {model_version!r}",
            UserWarning,
            stacklevel=2,
        )

    def _check_local_model_set(self) -> None:
        """Exit 2 when a local model set's model_set.json is invalid, or names other masks than --masks unless
        --allow-version-mismatch (then warn). A local directory without a marker: warn that it cannot be checked.
        Not a local directory: nothing."""
        models_dir = self.args.models_path
        if not os.path.isdir(models_dir):
            return
        if not os.path.isfile(join(models_dir, MODEL_SET_FILE_NAME)):
            masks = self.args.masks
            warnings.warn(
                f"{join(models_dir, MODEL_SET_FILE_NAME)} not found: the local model set cannot be checked against "
                f"--masks {masks!r}; running anyway with --masks {masks!r}",
                UserWarning,
                stacklevel=2,
            )
            return
        try:
            model_set = read_model_set(models_dir)
        except ValueError as err:
            self.parser.error(str(err))
        path, marker_masks, masks = join(models_dir, MODEL_SET_FILE_NAME), model_set.masks, self.args.masks
        if marker_masks == masks:
            return
        if not self.args.allow_version_mismatch:
            self.parser.error(
                f"{path} masks {marker_masks!r} and --masks {masks!r} differ: the local model set was trained on "
                f"{marker_masks!r} crops; pass --masks {marker_masks!r}, or --allow-version-mismatch to run with both"
            )
        warnings.warn(
            f"{path} masks {marker_masks!r} and --masks {masks!r} differ; running anyway (--allow-version-mismatch): "
            f"crops {masks!r}, models trained on {marker_masks!r}",
            UserWarning,
            stacklevel=2,
        )

    def fetch_models(self, models_path):
        """
        Fetch models using strategy pattern.

        Tries strategies in order:
        1. LocalPathStrategy - Use if path exists locally
        2. HuggingFaceStrategy - Try if it looks like HF repo ID
        3. RemoteArchiveStrategy - Try if it's a URL to an archive
        4. InteractiveFallbackStrategy - Ask user for local copy

        Args:
            models_path: Path, URL, or HF repo ID for the models

        Returns:
            Local path to the models
        """
        extract_to = ensure_models_cache_dir(self._compute_extract_dir())
        self._models_source = None
        fetched, self._models_source = fetch_model_set(
            models_path, extract_to, getattr(self.args, "model_version", None), getattr(self.args, "no_cache", False)
        )
        return fetched

    def _read_marker_model_version(self, fetched_models_path: str) -> str | None:
        """The model_version of the fetched model set's model_set.json, or None when absent or unreadable."""
        path = join(fetched_models_path, MODEL_SET_FILE_NAME)
        if not os.path.isfile(path):
            return None
        try:
            return read_model_set(fetched_models_path).model_version
        except (ValueError, OSError) as err:
            warnings.warn(
                f"Could not read {path} for provenance.json ({err}); model_version falls back to --model-version",
                UserWarning,
                stacklevel=2,
            )
            return None

    def _build_provenance(self, fetched_models_path: str) -> dict:
        """masks, model_version, checkpoint_epoch, models_source and pipeline_commit of this run, in that order."""
        model_version = self._read_marker_model_version(fetched_models_path)
        if model_version is None:
            model_version = getattr(self.args, "model_version", None)
        return {
            "masks": self.args.masks,
            "model_version": model_version,
            "checkpoint_epoch": self.args.checkpoint_epoch,
            "models_source": self._models_source,
            "pipeline_commit": read_pipeline_commit(),
        }

    def _write_provenance(self, output_base: str, provenance: dict) -> str:
        """Atomically write <output_base>/provenance.json and return its path; OSError fails the stage."""
        os.makedirs(output_base, exist_ok=True)
        path = join(output_base, PROVENANCE_FILE_NAME)
        write_json_atomic(path, provenance)
        print(f"Provenance written: {path}")
        return path

    def _validate_inputs(self):
        pass

    def default_log_path(self) -> str:
        """Run-log path of a bare --log: .log/embeddings_<stamp>.log in the dataset's champollion_V1/<masks> folder."""
        return compute_run_log_path(
            compute_run_log_derivatives_dir(
                "embeddings", self.args.datasets_root.rstrip("/"), masks_version=self.args.masks
            ),
            "embeddings",
            datetime.now().astimezone(),
        )

    def run(self):
        """Execute the embeddings pipeline script."""
        self._validate_inputs()
        if self.args.profiling:
            return self._run_with_profiling()
        return self._run_normal()

    def _run_with_profiling(self):
        """Run pipeline with cProfile enabled."""
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            result = self._run_normal()
        finally:
            profiler.disable()
            # Save profiling results
            stats = pstats.Stats(profiler)
            stats.sort_stats("cumulative")
            stats.dump_stats("embeddings_profile.prof")
            print("\nProfiling results saved to embeddings_profile.prof")
            print("View with: python -m pstats embeddings_profile.prof")
            # Print top 20 functions
            stream = StringIO()
            stats_print = pstats.Stats(profiler, stream=stream)
            stats_print.sort_stats("cumulative")
            stats_print.print_stats(20)
            print(stream.getvalue())

        return result

    def _run_normal(self):
        """Execute the embeddings pipeline script (normal mode)."""
        local_dir = os.getcwd()

        # Save original CUDA_VISIBLE_DEVICES value for restoration later
        original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        # Force CPU usage if requested
        if self.args.cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            print("CPU mode enabled: CUDA_VISIBLE_DEVICES set to empty string")

        try:
            return self._run_pipeline(local_dir)
        finally:
            # Restore original CUDA_VISIBLE_DEVICES value
            if original_cuda_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices

    def _run_pipeline(self, local_dir):
        """Run embeddings generation using champollion/evaluate.py per region."""
        original_models_path = self.args.models_path
        self.args.models_path = self.fetch_models(original_models_path)
        provenance = self._build_provenance(self.args.models_path)

        tmpdir = None
        if self.args.regions:
            tmpdir = self._make_regions_tmpdir(self.args.models_path)
            self.args.models_path = tmpdir

        script_dir = dirname(abspath(__file__))
        evaluate_script = abspath(
            join(script_dir, "..", "..", "external", "champollion_V1", "champollion", "evaluate.py")
        )

        datasets_root = self.args.datasets_root.rstrip("/")
        output_base = self.args.output or compute_region_embeddings_dir(datasets_root, self.args.masks)

        crops_2mm_dir = self._get_crops_2mm_dir(datasets_root)

        if self.args.subjects:
            warnings.warn(
                "--subjects is deprecated and ignored: each region uses the "
                "{side}skeleton_subject.csv next to its {side}skeleton.npy.",
                FutureWarning,
                stacklevel=2,
            )

        try:
            result = self._run_per_region(evaluate_script, crops_2mm_dir, output_base)
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir)
            self.args.models_path = original_models_path

        self._write_provenance(output_base, provenance)
        if self.args.run_cka:
            self._run_cka_test(output_base)

        return result

    def _find_crop_dir(self, crops_2mm_dir: str, region_model_name: str) -> str:
        """Map a model region name back to its crop directory name.

        Model dirs have dots stripped: crop 'SC-sylv.' → model 'SCsylv_left'.
        Search for a crop dir where dirname.replace('.', '') matches the base name.
        """
        base = region_model_name
        if base.endswith("_left"):
            base = base[:-5]
        elif base.endswith("_right"):
            base = base[:-6]

        if exists(crops_2mm_dir):
            for dname in os.listdir(crops_2mm_dir):
                if os.path.isdir(join(crops_2mm_dir, dname)):
                    if dname.replace(".", "") == base:
                        return dname
        return base

    def _run_per_region(self, evaluate_script: str, crops_2mm_dir: str, output_base: str) -> int:
        """Invoke champollion/evaluate.py for each region in models_path.

        Every region is attempted even after a failure. Returns the first non-zero
        evaluate code, or 0 when all evaluated regions succeed.
        """
        models_path = self.args.models_path
        epoch = self.args.checkpoint_epoch
        region_dirs = find_region_model_dirs(models_path)

        if not region_dirs:
            raise ValueError(f"No region model directories (containing logs/) found under {models_path}")
        check_region_weights(region_dirs, models_path, epoch)

        print(f"\nGenerating embeddings for {len(region_dirs)} regions → {output_base}")

        failed_regions: list[str] = []
        first_failure = 0
        for model_path in region_dirs:
            region = resolve_region_name(model_path, models_path)
            side = "L" if region.endswith("_left") else "R"
            crop_name = self._find_crop_dir(crops_2mm_dir, region)
            mask_dir = join(crops_2mm_dir, crop_name, "mask")
            skels_path = join(mask_dir, f"{side}skeleton.npy")
            subjects_path = join(mask_dir, f"{side}skeleton_subject.csv")
            saving_path = join(output_base, region, "full_embeddings.csv")

            if exists(saving_path) and not getattr(self.args, "overwrite", False):
                print(f"  [SKIP] {region} — already exists (--overwrite to recompute)")
                continue

            with evaluation_model_dir(model_path, epoch) as eval_model_path:
                os.makedirs(os.path.dirname(saving_path), exist_ok=True)
                # A recompute must never leave the previous run's output behind.
                if exists(saving_path):
                    os.remove(saving_path)

                cmd = [
                    sys.executable,
                    evaluate_script,
                    "-m",
                    eval_model_path,
                    "-sk",
                    skels_path,
                    "-i",
                    subjects_path,
                    "-s",
                    saving_path,
                ]

                weights = weights_source(model_path, epoch) or "<none found>"
                print(f"\n[Region {region}] weights: {weights}")
                if uses_best_weights_stand_in(model_path, epoch):
                    print(
                        f"[Region {region}] WARNING: no epoch={epoch} checkpoint; "
                        f"best_model_weights.pt stands in for epoch {epoch} (interim)"
                    )
                code = self.execute_command(cmd, shell=False)
            if code != 0:
                failed_regions.append(region)
                first_failure = first_failure or code
                if exists(saving_path):
                    os.remove(saving_path)

        if failed_regions:
            print(f"\n[FAILED] {len(failed_regions)} region(s) failed: {', '.join(failed_regions)}")
        return first_failure

    def _run_cka_test(self, output_base: str) -> None:
        """Run CKA coherence test on the generated embeddings."""
        print("\n" + "=" * 60)
        print("Running CKA Coherence Test")
        print("=" * 60)

        cka_output = join(output_base, "cka_results")
        print(f"Embeddings dir: {output_base}")
        print(f"CKA output: {cka_output}")

        try:
            test_models_coherence_from_directory(
                models_dir=output_base,
                embedding_filename="full_embeddings.csv",
                output_dir=cka_output,
                subject_column="ID",
            )
            print("CKA coherence test completed.")
        except Exception as e:
            print(f"Warning: CKA test failed - {e}")


def main():
    """Main entry point."""
    init_pipeline_process()
    script = GenerateEmbeddings()
    return script.main()


if __name__ == "__main__":
    exit(main())
