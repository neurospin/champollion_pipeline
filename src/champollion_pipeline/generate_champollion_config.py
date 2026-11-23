#!/usr/bin/env python3
"""
Script to define and generate Champollion's configuration.

By default, YAMLs are written to the dataset's own derivatives tree:
``<D>/<dataset>/derivatives/champollion_V1/configs``, where ``<D>`` is the
parent of the ``<dataset>`` directory found in ``crop_path``.  Use ``--output``
to choose a different configs root; use ``--external-config`` to write the
dataset_localization YAML elsewhere (e.g. a writable path in read-only
container environments).  ``--champollion_loc`` is read-only by default.
"""

import argparse
import glob
import json
import logging
import os
from datetime import datetime
from os.path import abspath, dirname, exists, isfile, join
from typing import NamedTuple

import numpy as np
import yaml
from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.derivatives_layout import (
    DEFAULT_MASKS_VERSION,
    compute_champollion_configs_root,
    compute_region_model_name,
    compute_run_log_derivatives_dir,
    compute_run_log_path,
)
from champollion_pipeline.process_setup import init_pipeline_process
from champollion_pipeline.utils.cortical_tiles_config import INPUT_TYPES_DEFAULT, compute_versioned_crops_dir
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER, find_dataset_folder
from champollion_pipeline.utils.qc_report import (
    CROP_KIND_MASK,
    REASON_SEPARATOR,
    fetch_crop_qc_rows,
    fetch_grouped_crop_qc_rows,
    log_input_qc_exclusions,
    write_qc_tsv,
    write_run_qc_tsv,
)
from champollion_pipeline.utils.region_masks import (
    REGION_MASKS_OPTION,
    add_region_masks_option,
    find_unmapped_region_models,
    refuse_masks_with_region_masks,
    split_region_model_name,
)
from champollion_pipeline.utils.resource_copy import replace_with_writable_copy

logger = logging.getLogger(__name__)

# Get the script's directory for reliable path resolution
_SCRIPT_DIR = dirname(abspath(__file__))
_DEFAULT_CHAMPOLLION_LOC = abspath(join(_SCRIPT_DIR, "..", "..", "external", "champollion_V1"))
_LOCALIZATION_TEMPLATE = "# @package _global_\ndataset_folder: {dataset_folder}\n"
INPUT_QC_FILENAME = "input_qc.tsv"
INPUT_QC_KINDS = (*INPUT_TYPES_DEFAULT, CROP_KIND_MASK)
FAILED_STATUS = 1
PARTIAL_STATUS = 3  # value of _stage_base.PARTIAL_STATUS, not imported: that module chain imports torch
OUTPUT_QC_FILENAME = "output_qc.tsv"
DATASET_FOLDER_KEY = "dataset_folder"
DATASET_FOLDER_TOKEN = "${dataset_folder}"
CONFIG_SIDES = ("left", "right")
CROP_SIDES = {"left": "L", "right": "R"}  # hemisphere -> crop file side letter
CROPS_TEMPLATE_PATH = "TESTXX/crops/2mm"
_EXTERNAL_CROPS_WITH_MAP_ERROR = (
    f"--external_crops cannot be given together with {REGION_MASKS_OPTION}: the map names a crops/<version>/2mm "
    "folder of the dataset derivatives for each region model name; drop --external_crops or the map"
)
CONFIG_PATH_KEYS = (
    "numpy_all",
    "subjects_all",
    "foldlabel_all",
    "subjects_foldlabel_all",
    "train_val_csv_file",
    "cutin_mask_path",
)
CONFIG_REQUIRED_KEYS = ("dataset_name", *CONFIG_PATH_KEYS, "input_size")
REASON_MISSING_CONFIG = "missing_config"
REASON_UNPARSABLE_CONFIG = "unparsable_config"
REASON_MISSING_DATASET_FOLDER = "missing_dataset_folder"
REASON_MISSING_KEY_STEM = "missing_key:"
REASON_MISSING_PATH_STEM = "missing_path:"


def _list_region_folders(crop_path: str) -> list[str]:
    """Names of the directories directly under crop_path, sorted; [] when crop_path is not a directory."""
    if not os.path.isdir(crop_path):
        return []
    return sorted(name for name in os.listdir(crop_path) if os.path.isdir(join(crop_path, name)))


class ConfiguredModel(NamedTuple):
    """A region model name of the --region-masks map whose region has a folder in its Mask version crops dir."""

    name: str  # region model name, e.g. SC-sylv_left
    version: str  # Mask version the map gives it
    crops_dir: str  # <dataset derivatives>/crops/<version>/2mm
    region: str  # Sulcal region folder name, e.g. S.C.-sylv.
    side: str  # crop file side letter, L or R


def _list_crop_path_model_names(crop_path: str) -> list[str]:
    """Region model names of the region folders of crop_path: sorted folders, then left and right."""
    return [
        compute_region_model_name(region, side) for region in _list_region_folders(crop_path) for side in CONFIG_SIDES
    ]


def _group_configured_models(configured: list[ConfiguredModel]) -> list[tuple[str, str, list[str]]]:
    """Input QC groups (crops dir, side, regions sorted) of the configured models: by Mask version, then L, R."""
    groups: dict[tuple[str, str, str], set[str]] = {}
    for model in configured:
        groups.setdefault((model.version, model.side, model.crops_dir), set()).add(model.region)
    return [(crops_dir, side, sorted(regions)) for (_version, side, crops_dir), regions in sorted(groups.items())]


def _read_yaml_mapping(path: str) -> dict | None:
    """Mapping held by the YAML file at path; None when it cannot be read or holds no mapping."""
    try:
        with open(path, encoding="utf-8") as f:
            document = yaml.safe_load(f)
    except (OSError, ValueError, yaml.YAMLError):  # ValueError holds UnicodeDecodeError
        return None
    return document if isinstance(document, dict) else None


def _read_dataset_folder(localization_path: str) -> str | None:
    """Value of dataset_folder in the localization YAML at localization_path; None when it yields none."""
    mapping = _read_yaml_mapping(localization_path)
    return (mapping.get(DATASET_FOLDER_KEY) or None) if mapping else None


def _compute_mapping_reasons(mapping: dict, dataset_folder: str | None) -> list[str]:
    """Output QC reasons of a parsed region YAML; with no dataset_folder, its paths are not judged."""
    reasons = [f"{REASON_MISSING_KEY_STEM}{key}" for key in CONFIG_REQUIRED_KEYS if mapping.get(key) in (None, "")]
    if dataset_folder is None:
        return reasons
    for key in CONFIG_PATH_KEYS:
        value = mapping.get(key)
        if value in (None, ""):
            continue
        if not isfile(str(value).replace(DATASET_FOLDER_TOKEN, dataset_folder)):
            reasons.append(f"{REASON_MISSING_PATH_STEM}{key}")
    return reasons


def _compute_config_reasons(config_path: str, dataset_folder: str | None) -> list[str]:
    """Output QC reasons of the region YAML at config_path."""
    reasons = [REASON_MISSING_DATASET_FOLDER] if dataset_folder is None else []
    if not isfile(config_path):
        return [*reasons, REASON_MISSING_CONFIG]
    mapping = _read_yaml_mapping(config_path)
    if mapping is None:
        return [*reasons, REASON_UNPARSABLE_CONFIG]
    return [*reasons, *_compute_mapping_reasons(mapping, dataset_folder)]


def _fetch_output_qc_rows(model_names: list[str], dataset_loc: str, localization_path: str) -> list[dict[str, object]]:
    """One output QC row per region model name, in input order, check = region YAML file name."""
    dataset_folder = _read_dataset_folder(localization_path)
    rows: list[dict[str, object]] = []
    for model_name in model_names:
        name = f"{model_name}.yaml"
        reasons = _compute_config_reasons(join(dataset_loc, name), dataset_folder)
        rows.append({"check": name, "qc": 0 if reasons else 1, "reason": REASON_SEPARATOR.join(reasons)})
    return rows


def _report_output_qc(rows: list[dict[str, object]], qc_path: str) -> int:
    """Log one ERROR per qc 0 row and one FAIL when no config is usable; return the status."""
    lost = [row for row in rows if row["qc"] == 0]
    for row in lost:
        logger.error(f"output QC invalid config {row['check']}: {row['reason']}")
    if rows and len(lost) == len(rows):
        logger.fail(f"no generated config is usable, see {qc_path}")
        return FAILED_STATUS
    return PARTIAL_STATUS if lost else 0


class GenerateChampollionConfig(ScriptBuilder):
    """Script for generating Champollion configuration files."""

    def __init__(self):
        super().__init__(
            script_name="generate_champollion_config",
            description="Defining and generating Champollion's configuration.",
        )
        # Configure arguments using method chaining
        (
            self.add_argument("crop_path", help="Absolute path to crops path.", type=str)
            .add_required_argument("--dataset", "Name of the dataset.")
            .add_optional_argument(
                "--champollion_loc",
                "Path to the champollion_V1 checkout (default: external/champollion_V1). "
                "Read only: never written to unless --output / --external-config point inside it.",
                default=_DEFAULT_CHAMPOLLION_LOC,
            )
            .add_optional_argument(
                "--output",
                "Configs root; region YAMLs land at {output}/dataset/{dataset}/. "
                "Default: <D>/<dataset>/derivatives/champollion_V1/configs, where <D> is "
                "the parent of the <dataset> directory in crop_path.",
            )
            .add_optional_argument(
                "--external-config",
                "Where to write the dataset_localization YAML: an existing directory "
                "(file lands at {dir}/dataset_localization/{localization}.yaml) or a file path. "
                "Default: {configs root}/dataset_localization/, i.e. under --output if given, "
                "else under <D>/<dataset>/derivatives/champollion_V1/configs.",
                default=None,
            )
            .add_flag(
                "--external_crops",
                "Use crop_path as-is instead of deriving it from --dataset. "
                "Replaces the dataset/derivatives/... segment with the actual crop_path location. "
                "Requires --dataset to be set (used for config file naming).",
            )
            .add_optional_argument(
                "--masks",
                "Mask version tag (e.g. 'canonical_25'). Must match the value used when running run_cortical_tiles.",
                default=DEFAULT_MASKS_VERSION,
                type_=None,  # identity: argparse keeps the DeclaredMasksDefault mark of add_region_masks_option
            )
            .add_optional_argument(
                "--localization",
                "Name of the dataset_localization preset to write "
                "(e.g. 'local', 'jean-zay'). Must match the value used in "
                "train_champollion.py. Default: 'local'.",
                default="local",
            )
        )
        add_region_masks_option(self)

    def parse_args(self, args: list[str] | None = None) -> argparse.Namespace:
        """Parse the command line; exit 2 for --masks or --external_crops beside --region-masks.

        Afterwards args.masks is a plain str: the declared default when --masks was not given (the run label).
        """
        parsed = super().parse_args(args)
        refuse_masks_with_region_masks(self.parser, self.args)
        if self.args.region_mask_versions is not None and self.args.external_crops:
            self.parser.error(_EXTERNAL_CROPS_WITH_MAP_ERROR)
        return parsed

    def _refuse_unmapped_region_models(self) -> None:
        """Exit 2 naming each region model name of a region folder of crop_path that --region-masks leaves out."""
        unmapped = find_unmapped_region_models(
            self.args.region_mask_versions, _list_region_folders(self.args.crop_path)
        )
        if unmapped:
            self.parser.error(
                f"{REGION_MASKS_OPTION} gives no Mask version for {', '.join(unmapped)}, whose region folder lies "
                f"in {self.args.crop_path}; add a NAME=VERSION pair for each"
            )

    def _list_configured_models(self, dataset_folder: str) -> list[ConfiguredModel]:
        """The mapped region model names whose region has a folder in the crops dir of their Mask version.

        Ordered by region, then L before R, as the region folders of crop_path are without a map.
        """
        derivatives = join(dataset_folder, self.args.dataset, "derivatives", DERIVATIVES_FOLDER)
        configured = []
        for name, version in self.args.region_mask_versions.items():
            region, hemisphere = split_region_model_name(name)
            crops_dir = compute_versioned_crops_dir(derivatives, version)
            if os.path.isdir(join(crops_dir, region)):
                configured.append(ConfiguredModel(name, version, crops_dir, region, CROP_SIDES[hemisphere]))
        return sorted(configured, key=lambda model: (model.region, model.side))

    def _compute_crops_template_path(self, version: str) -> str:
        """Dataset-relative crops/<version>/2mm path that replaces TESTXX/crops/2mm in the template."""
        return f"{self.args.dataset}/derivatives/{DERIVATIVES_FOLDER}/crops/{version}/2mm"

    def _get_crop_size(self, crop_dir: str, side: str) -> tuple[int, int, int] | None:
        """Return (sizeX, sizeY, sizeZ) from .npy shape if present, else from .minf.

        .npy is preferred: its axis order matches what the DataLoader receives at
        runtime. .minf uses anatomical conventions that may differ (e.g. X/Y swapped).
        """
        for suffix in (f"{side}skeleton.npy", f"{side}label.npy"):
            npy_path = join(crop_dir, "mask", suffix)
            if exists(npy_path):
                shape = np.load(npy_path, mmap_mode="r").shape
                # shape: (N, Z, X, Y, channel=1) — template writes (1, Z, X, Y),
                # PaddingTensor.rotate_list then gives target (Z, X, Y, 1) = per-sample shape
                if len(shape) == 5:
                    return shape[1], shape[2], shape[3]
                # shape: (N, X, Y, Z) — template writes (1, X, Y, Z),
                # rotate_list gives (X, Y, Z) target = per-sample shape
                if len(shape) == 4:
                    return shape[1], shape[2], shape[3]
        minf_path = join(crop_dir, "mask", f"{side}mask_cropped.nii.gz.minf")
        if exists(minf_path):
            with open(minf_path) as f:
                raw = f.read().replace("attributes = ", "").replace("'", '"')
            info = json.loads(raw)
            return info["sizeX"], info["sizeY"], info["sizeZ"]
        return None

    def _write_region_config(
        self, template: str, crop_dir: str, crop_name: str, side: str, dataset_loc: str
    ) -> str | None:
        """Write the region dataset YAML of crop_name and side (L or R) from template; its path, None when skipped.

        Skipped when crop_dir holds no crop array or .minf to read the crop size from.
        """
        size = self._get_crop_size(crop_dir, side)
        if size is None:
            return None
        sx, sy, sz = size
        dataset_name = compute_region_model_name(crop_name, "left" if side == "L" else "right")
        filedata = (
            template.replace("REPLACE_CROP_NAME", crop_name)
            .replace("REPLACE_DATASET", dataset_name)
            .replace("REPLACE_SIDE", side)
            .replace("REPLACE_SIZEX", str(sx))
            .replace("REPLACE_SIZEY", str(sy))
            .replace("REPLACE_SIZEZ", str(sz))
        )
        result_file = join(dataset_loc, f"{dataset_name}.yaml")
        with open(result_file, "w") as f:
            f.write(filedata)
        logger.output(f"Dataset config written: {result_file}")
        return result_file

    def _create_dataset_configs(self, crop_path: str, dataset_loc: str, ref: str) -> None:
        """Inline replacement for create_dataset_config_files.py with .npy fallback."""
        crop_dirs = sorted(glob.glob(join(crop_path, "*")))
        skipped = []
        for crop_dir in crop_dirs:
            if not os.path.isdir(crop_dir):
                continue
            crop_name = os.path.basename(crop_dir)
            for side in ("L", "R"):
                if self._write_region_config(ref, crop_dir, crop_name, side, dataset_loc) is None:
                    skipped.append(f"{crop_name}/{side}")
        if skipped:
            logger.warning(f"Skipped {len(skipped)} sulci (no .minf or .npy found): {skipped}")

    def _create_mapped_dataset_configs(self, configured: list[ConfiguredModel], dataset_loc: str) -> None:
        """Write the region dataset YAML of each configured model from the packaged template and its crops dir."""
        with open(join(_SCRIPT_DIR, "resources", "reference.yaml"), encoding="utf-8") as f:
            template = f.read()
        skipped = []
        for model in configured:
            ref = template.replace(CROPS_TEMPLATE_PATH, self._compute_crops_template_path(model.version))
            crop_dir = join(model.crops_dir, model.region)
            if self._write_region_config(ref, crop_dir, model.region, model.side, dataset_loc) is None:
                skipped.append(model.name)
        if skipped:
            logger.warning(f"Skipped {len(skipped)} sulci (no .minf or .npy found): {skipped}")

    def _run_input_qc(self, configs_root: str, configured: list[ConfiguredModel] | None = None) -> int:
        """Write input_qc.tsv into configs_root; return 0, PARTIAL_STATUS or FAILED_STATUS.

        With configured models (a --region-masks run), the crops of each in its own crops dir; else crop_path's.
        """
        qc_path = join(configs_root, INPUT_QC_FILENAME)
        crop_path = self.args.crop_path
        if configured is None:
            rows = fetch_crop_qc_rows(crop_path, _list_region_folders(crop_path), INPUT_QC_KINDS)
        else:
            rows = fetch_grouped_crop_qc_rows(_group_configured_models(configured), INPUT_QC_KINDS)
        write_qc_tsv(qc_path, rows)
        n_excluded = log_input_qc_exclusions(rows, logger)
        if n_excluded == len(rows):
            logger.fail(f"no subject passed the config input QC, see {qc_path}")
            return FAILED_STATUS
        return PARTIAL_STATUS if n_excluded else 0

    def _run_output_qc(
        self,
        configs_root: str,
        dataset_loc: str,
        localization_path: str,
        configured: list[ConfiguredModel] | None = None,
    ) -> int:
        """Write output_qc.tsv into configs_root; return 0, PARTIAL_STATUS or FAILED_STATUS.

        One row per configured model (a --region-masks run), else per region folder of crop_path and side.
        """
        qc_path = join(configs_root, OUTPUT_QC_FILENAME)
        if configured is None:
            model_names = _list_crop_path_model_names(self.args.crop_path)
        else:
            model_names = [model.name for model in configured]
        rows = _fetch_output_qc_rows(model_names, dataset_loc, localization_path)
        write_run_qc_tsv(qc_path, rows)
        return _report_output_qc(rows, qc_path)

    def _validate_inputs(self):
        """Validate input paths."""
        if not exists(self.args.crop_path):
            raise ValueError(
                f"generate_champollion_config: Please input correct values. {self.args.crop_path} does not exist."
            )

    def _write_localization_yaml(self, dest_path: str, dataset_folder: str) -> None:
        """Write a dataset_localization YAML for the given environment."""
        os.makedirs(dirname(dest_path), exist_ok=True)
        with open(dest_path, "w") as f:
            f.write(_LOCALIZATION_TEMPLATE.format(dataset_folder=dataset_folder))
        logger.output(f"Localization config written: {dest_path}")

    def _find_checkout_dataset_configs(self, champollion_loc: str) -> str | None:
        """Return <champollion_loc>/champollion/configs/dataset/<dataset> if that dir exists, else None.

        Used only to print a notice: Hydra's primary config_path (the checkout's
        built-in configs) wins over hydra.searchpath (--config-dir), so stale
        in-checkout YAMLs from earlier default runs shadow the newly generated
        ones in training. O(1).
        """
        path = join(champollion_loc, "champollion", "configs", "dataset", self.args.dataset)
        return path if os.path.isdir(path) else None

    def default_log_path(self) -> str:
        """Run-log path of a bare --log: .log/config_<stamp>.log in the dataset's champollion_V1/<masks> folder."""
        return compute_run_log_path(
            compute_run_log_derivatives_dir(
                "config",
                join(find_dataset_folder(self.args.crop_path, self.args.dataset), self.args.dataset),
                masks_version=self.args.masks,
            ),
            "config",
            datetime.now().astimezone(),
        )

    def run(self):
        """Execute the champollion config generation script."""
        self._validate_inputs()
        if self.args.region_mask_versions is not None:
            self._refuse_unmapped_region_models()

        # Resolve champollion_loc to absolute path (read-only by default).
        champollion_loc = abspath(self.args.champollion_loc)

        # dataset_folder must be computed first — needed for configs_root default.
        dataset_folder = find_dataset_folder(self.args.crop_path, self.args.dataset)

        # Determine configs root: explicit --output wins, otherwise the dataset derivatives tree.
        configs_root = (
            abspath(self.args.output)
            if self.args.output
            else compute_champollion_configs_root(dataset_folder, self.args.dataset)
        )

        # Region YAMLs land at {configs_root}/dataset/{dataset}/ to match Hydra config-group layout.
        dataset_loc = join(configs_root, "dataset", self.args.dataset)

        configured = None if self.args.region_mask_versions is None else self._list_configured_models(dataset_folder)
        input_status = self._run_input_qc(configs_root, configured)
        if input_status == FAILED_STATUS:
            return input_status

        # Create dataset directory if it doesn't exist
        if not exists(dataset_loc):
            self.execute_command(["mkdir", "-p", dataset_loc], shell=False)

        # Always copy reference.yaml from template so re-runs regenerate cleanly
        reference_yaml_dest = join(dataset_loc, "reference.yaml")
        reference_yaml_src = join(_SCRIPT_DIR, "resources", "reference.yaml")
        replace_with_writable_copy(reference_yaml_src, reference_yaml_dest)

        my_lines = []
        with open(reference_yaml_dest) as f:
            for line in f.readlines():
                if self.args.external_crops:
                    # External crops: derive the full relative path from crop_path
                    # e.g. crop_path = /external/shared_project/TEST01/path/to/crops/crops/2mm
                    #      dataset_folder = /external/shared_project
                    #      => relative = TEST01/path/to/crops/crops/2mm
                    relative_path = os.path.relpath(self.args.crop_path, dataset_folder)
                    my_lines.append(line.replace(CROPS_TEMPLATE_PATH, relative_path))
                else:
                    # Standard: use the known derivatives folder structure including mask version
                    computed_path = self._compute_crops_template_path(self.args.masks)
                    my_lines.append(line.replace(CROPS_TEMPLATE_PATH, computed_path))

        with open(reference_yaml_dest, "w") as f:
            f.writelines(my_lines)

        logger.trace(f"generate_champollion_config.py/champollion_loc: {champollion_loc}")

        with open(reference_yaml_dest) as f:
            ref = f.read()
        if configured is None:
            self._create_dataset_configs(self.args.crop_path, dataset_loc, ref)
        else:
            self._create_mapped_dataset_configs(configured, dataset_loc)

        # Write the dataset_localization YAML for the requested environment.
        localization_name = f"{self.args.localization}.yaml"

        if self.args.external_config:
            localization_path = abspath(self.args.external_config)
            if os.path.isdir(localization_path):
                localization_path = join(localization_path, "dataset_localization", localization_name)
        else:
            # Default: follow configs_root (--output if given, else derivatives default).
            localization_path = join(configs_root, "dataset_localization", localization_name)
        self._write_localization_yaml(localization_path, dataset_folder)

        # Warn if stale in-checkout dataset configs exist; they are now only a fallback
        # for regions absent from configs_root, but their presence is worth flagging as clutter.
        stale = self._find_checkout_dataset_configs(champollion_loc)
        if stale:
            logger.warning(
                f"stale in-checkout dataset configs at {stale} are now a fallback "
                f"(only used for regions not found in {configs_root}); consider deleting them."
            )

        output_status = self._run_output_qc(configs_root, dataset_loc, localization_path, configured)
        return output_status or input_status  # input_status is 0 or 3 here: 1 and 3 of the output QC win


def main():
    """Main entry point."""
    init_pipeline_process()
    script = GenerateChampollionConfig()
    return script.main()


if __name__ == "__main__":
    exit(main())
