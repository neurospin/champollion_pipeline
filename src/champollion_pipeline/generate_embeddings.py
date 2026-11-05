#!/usr/bin/env python3
"""
Wrapper script to generate sulcal embeddings per region.
Calls champollion/evaluate.py for each region in models_path.
"""

import argparse
import cProfile
import csv
import logging
import os
import pstats
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime
from io import StringIO
from os.path import abspath, dirname, exists, join
from pathlib import Path

from champollion.metrics.cka_coherence import test_models_coherence_from_directory
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
from champollion_pipeline.models.version_options import (
    check_local_model_set,
    check_version_mismatch,
    find_marker_model_version,
    resolve_masks_default,
    resolve_model_version_alias,
)
from champollion_pipeline.process_setup import init_pipeline_process
from champollion_pipeline.utils.lib import CORTICAL_TILES_VERSION
from champollion_pipeline.utils.qc_report import (
    ID_COLUMN,
    QC_COLUMN,
    REASON_COLUMN,
    REASON_SEPARATOR,
    fetch_crop_qc_rows,
    write_qc_tsv,
)

PROVENANCE_FILE_NAME = "provenance.json"
INPUT_QC_FILENAME = "input_qc.tsv"
INPUT_QC_KINDS = ("skeleton",)  # evaluate.py reads no other crop kind
EMBEDDINGS_FILENAME = "full_embeddings.csv"
EMBEDDINGS_ID_COLUMN = "ID"
FAILED_STATUS = 1

logger = logging.getLogger(__name__)


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


def _compute_region_side(region: str) -> str:
    """Hemisphere side of a region model: L for *_left, else R."""
    return "L" if region.endswith("_left") else "R"


def fetch_region_model_qc_rows(
    crops_2mm_dir: str, crops_by_side: Mapping[str, Sequence[str]]
) -> list[dict[str, object]]:
    """One QC row per subject, sorted, over the crop folders of each side: qc 0 with the reasons of all its sides.

    The cohort is the union of the sides, so that a side whose files are wholly absent still shows its gap.
    """
    sides = sorted(crops_by_side)
    cohort = {
        row[ID_COLUMN]
        for side in sides
        for row in fetch_crop_qc_rows(crops_2mm_dir, crops_by_side[side], INPUT_QC_KINDS, sides=(side,))
    }
    reasons: dict[str, list[str]] = {subject: [] for subject in cohort}
    for side in sides:
        for row in fetch_crop_qc_rows(
            crops_2mm_dir, crops_by_side[side], INPUT_QC_KINDS, subjects=cohort, sides=(side,)
        ):
            if row[REASON_COLUMN]:
                reasons[row[ID_COLUMN]].append(row[REASON_COLUMN])
    return [
        {ID_COLUMN: subject, QC_COLUMN: 0 if found else 1, REASON_COLUMN: REASON_SEPARATOR.join(found)}
        for subject, found in sorted(reasons.items())
    ]


def _report_input_qc(rows: list[dict[str, object]], qc_path: str) -> int:
    """Log one ERROR per qc 0 row; return FAILED_STATUS, after one FAIL record, when no row has qc 1,
    PARTIAL_STATUS with a qc 0 row, else 0."""
    excluded = [row for row in rows if row[QC_COLUMN] == 0]
    for row in excluded:
        logger.error(f"input QC excluded {row[ID_COLUMN]}: {row[REASON_COLUMN]}")
    if len(excluded) == len(rows):
        logger.fail(f"no subject passed the embeddings input QC, see {qc_path}")
        return FAILED_STATUS
    return PARTIAL_STATUS if excluded else 0


def remove_subject_rows(csv_path: str, subjects: frozenset[str]) -> int:
    """Drop the rows of ``subjects`` from the embeddings CSV at ``csv_path``, atomically; return the dropped count.

    The header and the kept rows keep their bytes and the file its mode. An empty file, a header with no ID cell
    or a file holding none of ``subjects`` stays untouched and gives 0; a line too short to hold the ID cell is kept.
    """
    with open(csv_path, encoding="utf-8", newline="") as handle:
        lines = handle.readlines()
    header = next(csv.reader(lines[:1]), [])
    if EMBEDDINGS_ID_COLUMN not in header:
        return 0
    id_index = header.index(EMBEDDINGS_ID_COLUMN)
    kept = lines[:1]
    for line in lines[1:]:
        cells = next(csv.reader([line]), [])
        if len(cells) <= id_index or cells[id_index] not in subjects:
            kept.append(line)
    dropped = len(lines) - len(kept)
    if not dropped:
        return 0
    fd, tmp_path = tempfile.mkstemp(dir=dirname(csv_path), prefix=".embeddings_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.writelines(kept)
        shutil.copymode(csv_path, tmp_path)
        os.replace(tmp_path, csv_path)
    except BaseException:
        if exists(tmp_path):
            os.remove(tmp_path)
        raise
    return dropped


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
        self._excluded_subjects: frozenset[str] = frozenset()

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
        resolve_masks_default(self.args)
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
            # Save profiling results
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

        # Save original CUDA_VISIBLE_DEVICES value for restoration later
        original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        # Force CPU usage if requested
        if self.args.cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            logger.trace("CPU mode enabled: CUDA_VISIBLE_DEVICES set to empty string")

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
            input_status = self._run_input_qc(crops_2mm_dir, output_base)
            if input_status == FAILED_STATUS:
                return input_status
            result = self._run_per_region(evaluate_script, crops_2mm_dir, output_base)
            self._remove_excluded_rows(output_base)
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir)
            self.args.models_path = original_models_path

        self._write_provenance(output_base, provenance)
        if self.args.run_cka:
            self._run_cka_test(output_base)

        return result or input_status

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
                if os.path.isdir(join(crops_2mm_dir, dname)) and dname.replace(".", "") == base:
                    return dname
        return base

    def _find_checked_region_dirs(self) -> list[str]:
        """Region model dirs of the run; ValueError when there is none or when a region has no weights."""
        models_path = self.args.models_path
        region_dirs = find_region_model_dirs(models_path)
        if not region_dirs:
            raise ValueError(f"No region model directories (containing logs/) found under {models_path}")
        check_region_weights(region_dirs, models_path, self.args.checkpoint_epoch)
        return region_dirs

    def _compute_crops_by_side(self, region_dirs: list[str], crops_2mm_dir: str) -> dict[str, list[str]]:
        """Crop folder name of each region model, by the hemisphere side the model uses."""
        crops_by_side: dict[str, list[str]] = {}
        for model_path in region_dirs:
            region = resolve_region_name(model_path, self.args.models_path)
            crops_by_side.setdefault(_compute_region_side(region), []).append(
                self._find_crop_dir(crops_2mm_dir, region)
            )
        return crops_by_side

    def _run_input_qc(self, crops_2mm_dir: str, output_base: str) -> int:
        """Write <output_base>/input_qc.tsv, keep its qc 0 subjects for _remove_excluded_rows and return
        0, PARTIAL_STATUS with a qc 0 row, or FAILED_STATUS with no qc 1 row."""
        region_dirs = self._find_checked_region_dirs()
        rows = fetch_region_model_qc_rows(crops_2mm_dir, self._compute_crops_by_side(region_dirs, crops_2mm_dir))
        qc_path = join(output_base, INPUT_QC_FILENAME)
        write_qc_tsv(qc_path, rows)
        self._excluded_subjects = frozenset(row[ID_COLUMN] for row in rows if row[QC_COLUMN] == 0)
        return _report_input_qc(rows, qc_path)

    def _remove_excluded_rows(self, output_base: str) -> None:
        """Drop the qc 0 subjects from the full_embeddings.csv of each region model of the run, reused ones too."""
        if not self._excluded_subjects:
            return
        models_path = self.args.models_path
        for model_path in find_region_model_dirs(models_path):
            path = join(output_base, resolve_region_name(model_path, models_path), EMBEDDINGS_FILENAME)
            if exists(path):
                remove_subject_rows(path, self._excluded_subjects)

    def _run_per_region(self, evaluate_script: str, crops_2mm_dir: str, output_base: str) -> int:
        """Invoke champollion/evaluate.py for each region in models_path.

        Every region is attempted even after a failure. Returns the first non-zero
        evaluate code, or 0 when all evaluated regions succeed.
        """
        models_path = self.args.models_path
        epoch = self.args.checkpoint_epoch
        region_dirs = self._find_checked_region_dirs()

        n_regions = len(region_dirs)
        logger.trace(f"Generating embeddings for {n_regions} regions → {output_base}")

        failed_regions: list[str] = []
        first_failure = 0
        for fold, model_path in enumerate(region_dirs, start=1):
            region = resolve_region_name(model_path, models_path)
            # Machine-read by champollion_sulcal_mcp (FOLD_RE): one record per region directory.
            logger.trace(f"Fold {fold}/{n_regions}: {region}")
            side = _compute_region_side(region)
            crop_name = self._find_crop_dir(crops_2mm_dir, region)
            mask_dir = join(crops_2mm_dir, crop_name, "mask")
            skels_path = join(mask_dir, f"{side}skeleton.npy")
            subjects_path = join(mask_dir, f"{side}skeleton_subject.csv")
            saving_path = join(output_base, region, EMBEDDINGS_FILENAME)

            if exists(saving_path) and not getattr(self.args, "overwrite", False):
                logger.trace(f"[SKIP] {region} — already exists (--overwrite to recompute)")
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
                logger.trace(f"[Region {region}] weights: {weights}")
                if uses_best_weights_stand_in(model_path, epoch):
                    logger.warning(
                        f"[Region {region}] no epoch={epoch} checkpoint; "
                        f"best_model_weights.pt stands in for epoch {epoch} (interim)"
                    )
                code = self.execute_command(cmd, shell=False)
            if code != 0:
                failed_regions.append(region)
                first_failure = first_failure or code
                if exists(saving_path):
                    os.remove(saving_path)

        if failed_regions:
            logger.fail(f"[FAILED] {len(failed_regions)} region(s) failed: {', '.join(failed_regions)}")
        return first_failure

    def _run_cka_test(self, output_base: str) -> None:
        """Run CKA coherence test on the generated embeddings."""
        cka_output = join(output_base, "cka_results")
        logger.trace(f"Running CKA coherence test: embeddings {output_base}, output {cka_output}")

        try:
            test_models_coherence_from_directory(
                models_dir=output_base,
                embedding_filename="full_embeddings.csv",
                output_dir=cka_output,
                subject_column="ID",
            )
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
