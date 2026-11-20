#!/usr/bin/env python3
"""
Wrapper script to generate sulcal embeddings per region.
Calls champollion/evaluate.py for each region in models_path.
"""

import argparse
import cProfile
import logging
import os
import pstats
import re
import shutil
import sys
import tempfile
import warnings
from datetime import datetime
from io import StringIO
from os.path import abspath, dirname, exists, join
from pathlib import Path

from champollion.metrics.cka_coherence import CKACoherenceTester, test_models_coherence_from_directory
from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline._stage_base import PARTIAL_STATUS
from champollion_pipeline.derivatives_layout import DEFAULT_MASKS_VERSION as DEFAULT_MASKS_VERSION
from champollion_pipeline.derivatives_layout import (
    compute_models_cache_dir,
    compute_region_embeddings_dir,
    compute_run_log_derivatives_dir,
    compute_run_log_path,
)
from champollion_pipeline.models.cache import MODEL_SET_FILE_NAME as MODEL_SET_FILE_NAME
from champollion_pipeline.models.cache import ModelSet as ModelSet
from champollion_pipeline.models.cache import ensure_models_cache_dir, write_json_atomic
from champollion_pipeline.models.cache import read_model_set as read_model_set
from champollion_pipeline.models.cache import write_model_set as write_model_set
from champollion_pipeline.models.checkpoint import (
    DEFAULT_CHECKPOINT_EPOCH,
    evaluation_model_dir,
    uses_best_weights_stand_in,
    weights_source,
)
from champollion_pipeline.models.embeddings_qc import (
    EMBEDDINGS_FILENAME,
    OUTPUT_QC_FILENAME,
    compute_crops_by_side,
    compute_excluded_subjects,
    compute_region_side,
    fetch_named_files_qc_rows,
    fetch_region_model_qc_rows,
    read_model_width,
    remove_subject_rows,
    write_model_widths,
)
from champollion_pipeline.models.epoch_spec import parse_epoch_specs
from champollion_pipeline.models.fetch import HuggingFaceStrategy as HuggingFaceStrategy
from champollion_pipeline.models.fetch import InteractiveFallbackStrategy as InteractiveFallbackStrategy
from champollion_pipeline.models.fetch import LocalPathStrategy as LocalPathStrategy
from champollion_pipeline.models.fetch import ModelFetchStrategy as ModelFetchStrategy
from champollion_pipeline.models.fetch import RemoteArchiveStrategy as RemoteArchiveStrategy
from champollion_pipeline.models.fetch import fetch_model_set
from champollion_pipeline.models.model_run import ModelRun, ModelRunRefused, resolve_model_run
from champollion_pipeline.models.version_options import (
    check_local_model_set,
    check_version_mismatch,
    find_marker_model_version,
    resolve_masks_default,
    resolve_model_version_alias,
)
from champollion_pipeline.process_setup import init_pipeline_process
from champollion_pipeline.utils.lib import CORTICAL_TILES_VERSION
from champollion_pipeline.utils.pipeline_commit import read_pipeline_commit
from champollion_pipeline.utils.qc_report import log_qc_exclusions, read_qc_tsv, write_qc_tsv

PROVENANCE_FILE_NAME = "provenance.json"
INPUT_QC_FILENAME = "input_qc.tsv"
FAILED_STATUS = 1

logger = logging.getLogger(__name__)


class _CheckpointEpochAction(argparse.Action):
    """Repeatable --checkpoint-epoch [<id>=]N: sets checkpoint_epoch (the default) and checkpoint_epochs_by_id.

    A value parse_epoch_specs refuses ends the parse with exit 2.
    """

    def __call__(self, parser, namespace, values, _option_string=None):
        held = [*getattr(namespace, "checkpoint_epoch_values", ()), values]
        try:
            default, by_id = parse_epoch_specs(held, DEFAULT_CHECKPOINT_EPOCH)
        except ValueError as err:
            raise argparse.ArgumentError(self, str(err)) from err
        namespace.checkpoint_epoch_values = held
        setattr(namespace, self.dest, default)
        namespace.checkpoint_epochs_by_id = by_id


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
            .add_argument(
                "--checkpoint-epoch",
                action=_CheckpointEpochAction,
                default=DEFAULT_CHECKPOINT_EPOCH,
                metavar="[ID=]N",
                help="Training epoch N (0-based) of the Lightning checkpoint epoch=N-step=*.ckpt to evaluate each "
                f"model with. Default: {DEFAULT_CHECKPOINT_EPOCH}. Repeatable: a bare N is the default epoch (at most "
                "one); ID=N sets the epoch of the models with model id ID (the folder path between the region folder "
                "and the model folder).",
            )
            .add_flag(
                "--per-model-folders",
                "Write the embeddings of every model with a model id to <output>/<region>/<model id>/, even a "
                "single model of its region (by default only a region holding two or more models does).",
            )
            .add_optional_argument(
                "--cortical_version",
                f"Derivatives folder name (e.g. 'cortical_tiles-2027'). "
                f"Defaults to cortical_tiles-{CORTICAL_TILES_VERSION}. Crops are read from "
                f"<datasets_root>/derivatives/<cortical_version>/crops/<masks>/2mm.",
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
                "Cortical tiles mask version used as crops subdirectory (e.g. 'canonical_25'): crops are read from "
                "<datasets_root>/derivatives/<cortical_version>/crops/<masks>/2mm. "
                "Default: the --model-version value when given; else, when models_path is a local folder, "
                "the name of that folder if it is a known Mask version (otherwise exit 2 with an ERROR, pass --masks); "
                f"{DEFAULT_MASKS_VERSION} only when models_path is not a local directory.",
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
        self._excluded_subjects: frozenset[str] = frozenset()
        self._run: ModelRun | None = None

    def _make_regions_tmpdir(self, models_path: str) -> str:
        """Return a temp dir containing symlinks to only the requested region subdirs."""
        missing = [r for r in self.args.regions if not os.path.isdir(join(models_path, r))]
        if missing:
            raise FileNotFoundError(
                f"Model folder(s) not found for {len(missing)} requested region(s): {', '.join(missing)} "
                f"(models_path searched: {models_path})"
            )
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
            logger.trace(f"Patched {patched} YAML file(s): derivatives folder → {target_folder}")

    def _compute_extract_dir(self) -> str:
        """Return the absolute models cache dir: --models-cache if given, else the dataset-tree default."""
        override = getattr(self.args, "models_cache", None)
        if override is not None:
            return abspath(override)
        return compute_models_cache_dir(self.args.datasets_root)

    def parse_args(self, args: list[str] | None = None) -> argparse.Namespace:
        """Parse args, then resolve the deprecated --masks-version alias into model_version."""
        parsed = super().parse_args(args)
        resolve_model_version_alias(self.args, self.parser)
        resolve_masks_default(self.args, self.parser)
        check_version_mismatch(self.args, self.parser)
        check_local_model_set(self.args, self.parser)
        return parsed

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

    def _build_provenance(self, fetched_models_path: str) -> dict:
        """masks, model_version, checkpoint_epoch, models_source and pipeline_commit of this run, in that order."""
        model_version = find_marker_model_version(fetched_models_path)
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
        logger.output(f"Provenance written: {path}")
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
            stats = pstats.Stats(profiler)
            stats.sort_stats("cumulative")
            stats.dump_stats("embeddings_profile.prof")
            logger.output(
                "Profiling results saved to embeddings_profile.prof "
                "(view with: python -m pstats embeddings_profile.prof)"
            )
            stream = StringIO()
            stats_print = pstats.Stats(profiler, stream=stream)
            stats_print.sort_stats("cumulative")
            stats_print.print_stats(20)
            logger.output(stream.getvalue())

        return result

    def _run_normal(self):
        """Execute the embeddings pipeline script (normal mode)."""
        local_dir = os.getcwd()

        original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        if self.args.cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            logger.trace("CPU mode enabled: CUDA_VISIBLE_DEVICES set to empty string")

        try:
            return self._run_pipeline(local_dir)
        finally:
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

        self._run = None
        cka_files = None
        try:
            input_status = self._run_input_qc(crops_2mm_dir, output_base)
            if input_status == FAILED_STATUS:
                return input_status
            result = self._run_per_region(evaluate_script, crops_2mm_dir, output_base)
            self._remove_excluded_rows(output_base)
            output_status = self._run_output_qc(output_base)
            if self._run is not None:
                cka_files = self._run.compute_cka_files(output_base)
                if self._run.lists_models():
                    provenance["models"] = self._run.compute_provenance_models()
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir)
            self.args.models_path = original_models_path
            self._run = None

        self._write_provenance(output_base, provenance)
        if self.args.run_cka:
            self._run_cka_test(output_base, cka_files)

        return result or output_status or input_status

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
            for dname in sorted(os.listdir(crops_2mm_dir)):
                if os.path.isdir(join(crops_2mm_dir, dname)) and dname.replace(".", "") == base:
                    return dname
        return base

    def _get_run(self) -> ModelRun:
        """The models of the run, resolved once per run; a refused models tree or epoch id exits 2."""
        if self._run is None:
            try:
                self._run = resolve_model_run(
                    self.args.models_path,
                    self.args.checkpoint_epoch,
                    getattr(self.args, "checkpoint_epochs_by_id", {}),
                    bool(getattr(self.args, "per_model_folders", False)),
                )
            except ModelRunRefused as err:
                self.parser.error(str(err))
        return self._run

    def _run_input_qc(self, crops_2mm_dir: str, output_base: str) -> int:
        """Write <output_base>/input_qc.tsv, keep its qc 0 subjects; return the status of _report_qc_rows."""
        regions = dict.fromkeys(entry.region for entry in self._get_run().entries)
        crops = {region: self._find_crop_dir(crops_2mm_dir, region) for region in regions}
        rows = fetch_region_model_qc_rows(crops_2mm_dir, compute_crops_by_side(crops))
        self._excluded_subjects = compute_excluded_subjects(rows)
        return self._report_qc_rows(rows, join(output_base, INPUT_QC_FILENAME), "input")

    def _run_output_qc(self, output_base: str) -> int:
        """Write <output_base>/output_qc.tsv over each model; return the status of _report_qc_rows."""
        models = self._get_run().entries
        folders = self._get_run().compute_output_folders(output_base)
        write_model_widths((entry.path, folders[entry.path]) for entry in models)
        files = [
            (entry.label, join(folders[entry.path], EMBEDDINGS_FILENAME), read_model_width(entry.path))
            for entry in models
        ]
        rows = fetch_named_files_qc_rows(files, read_qc_tsv(join(output_base, INPUT_QC_FILENAME)))
        return self._report_qc_rows(rows, join(output_base, OUTPUT_QC_FILENAME), "output")

    def _report_qc_rows(self, rows: list[dict[str, object]], qc_path: str, step: str) -> int:
        """Write and log ``rows``; return 0, PARTIAL_STATUS with a qc 0 row, or FAILED_STATUS with no qc 1 row."""
        write_qc_tsv(qc_path, rows)
        n_excluded = log_qc_exclusions(rows, logger, step)
        if n_excluded == len(rows):
            logger.fail(f"no subject passed the embeddings {step} QC, see {qc_path}")
            return FAILED_STATUS
        return PARTIAL_STATUS if n_excluded else 0

    def _remove_excluded_rows(self, output_base: str) -> None:
        """Drop the qc 0 subjects from the full_embeddings.csv of each model of the run, reused ones too."""
        if not self._excluded_subjects:
            return
        for folder in self._get_run().compute_output_folders(output_base).values():
            path = join(folder, EMBEDDINGS_FILENAME)
            if exists(path):
                remove_subject_rows(path, self._excluded_subjects)

    def _run_per_region(self, evaluate_script: str, crops_2mm_dir: str, output_base: str) -> int:
        """Invoke champollion/evaluate.py for each model in models_path, with the weights of its own epoch.

        Every model is attempted even after a failure. Returns the first non-zero
        evaluate code, or 0 when all evaluated models succeed.
        """
        run = self._get_run()
        models = run.entries
        folders = run.compute_output_folders(output_base)

        n_models = len(models)
        logger.trace(f"Generating embeddings for {n_models} models → {output_base}")

        failed_models: list[str] = []
        first_failure = 0
        for fold, entry in enumerate(models, start=1):
            region, label, model_path = entry.region, entry.label, entry.path
            epoch = run.epoch_of(entry)
            # Machine-read by champollion_sulcal_mcp (FOLD_RE): one record per model directory.
            logger.trace(f"Fold {fold}/{n_models}: {label}")
            side = compute_region_side(region)
            crop_name = self._find_crop_dir(crops_2mm_dir, region)
            mask_dir = join(crops_2mm_dir, crop_name, "mask")
            skels_path = join(mask_dir, f"{side}skeleton.npy")
            subjects_path = join(mask_dir, f"{side}skeleton_subject.csv")
            saving_path = join(folders[model_path], EMBEDDINGS_FILENAME)

            if exists(saving_path) and not getattr(self.args, "overwrite", False):
                logger.trace(f"[SKIP] {label} — already exists (--overwrite to recompute)")
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
                logger.trace(f"[Region {label}] weights: {weights}")
                if uses_best_weights_stand_in(model_path, epoch):
                    logger.warning(
                        f"[Region {label}] no epoch={epoch} checkpoint; "
                        f"best_model_weights.pt stands in for epoch {epoch} (interim)"
                    )
                code = self.execute_command(cmd, shell=False)
            if code != 0:
                failed_models.append(label)
                first_failure = first_failure or code
                if exists(saving_path):
                    os.remove(saving_path)

        if failed_models:
            logger.fail(f"[FAILED] {len(failed_models)} region(s) failed: {', '.join(failed_models)}")
        return first_failure

    def _run_cka_test(self, output_base: str, model_files: dict[str, str] | None = None) -> None:
        """Run CKA coherence test on the generated embeddings.

        ``model_files`` maps each model label to its embeddings file; the matrix then names each model by label.
        None searches output_base for full_embeddings.csv files, named by their parent folder.
        """
        cka_output = join(output_base, "cka_results")
        logger.trace(f"Running CKA coherence test: embeddings {output_base}, output {cka_output}")

        try:
            if model_files is None:
                test_models_coherence_from_directory(
                    models_dir=output_base,
                    embedding_filename="full_embeddings.csv",
                    output_dir=cka_output,
                    subject_column="ID",
                )
            else:
                found = {label: path for label, path in model_files.items() if exists(path)}
                tester = CKACoherenceTester(found, subject_column="ID")
                tester.run_full_analysis()
                tester.save_results(cka_output)
                tester.print_summary()
            logger.output("CKA coherence test completed.")
        except Exception as e:
            logger.warning(f"CKA test failed - {e}", exc_info=True)


def main():
    """Main entry point."""
    init_pipeline_process()
    script = GenerateEmbeddings()
    return script.main()


if __name__ == "__main__":
    exit(main())
