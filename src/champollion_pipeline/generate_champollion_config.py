#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to define and generate Champollion's configuration.

By default, YAMLs are written to the dataset's own derivatives tree:
``<D>/<dataset>/derivatives/champollion_V1/configs``, where ``<D>`` is the
parent of the ``<dataset>`` directory found in ``crop_path``.  Use ``--output``
to choose a different configs root; use ``--external-config`` to write the
dataset_localization YAML elsewhere (e.g. a writable path in read-only
container environments).  ``--champollion_loc`` is read-only by default.
"""

import glob
import json
import os
from os.path import abspath, dirname, exists, join

import numpy as np
from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.derivatives_layout import compute_champollion_configs_root, compute_region_model_name
from champollion_pipeline.process_setup import init_pipeline_process
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER, find_dataset_folder

# Get the script's directory for reliable path resolution
_SCRIPT_DIR = dirname(abspath(__file__))
_DEFAULT_CHAMPOLLION_LOC = abspath(join(_SCRIPT_DIR, "..", "..", "external", "champollion_V1"))
_LOCALIZATION_TEMPLATE = "# @package _global_\ndataset_folder: {dataset_folder}\n"


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
                default="canonical_25",
            )
            .add_optional_argument(
                "--localization",
                "Name of the dataset_localization preset to write "
                "(e.g. 'local', 'jean-zay'). Must match the value used in "
                "train_champollion.py. Default: 'local'.",
                default="local",
            )
        )

    def _get_crop_size(self, crop_dir: str, side: str) -> tuple[int, int, int] | None:
        """Return (sizeX, sizeY, sizeZ) from .npy shape if present, else from .minf.

        .npy is preferred: its axis order matches what the DataLoader receives at
        runtime. .minf uses anatomical conventions that may differ (e.g. X/Y swapped).
        """
        for suffix in (f"{side}skeleton.npy", f"{side}label.npy", f"{side}distbottom.npy"):
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
            with open(minf_path, "r") as f:
                raw = f.read().replace("attributes = ", "").replace("'", '"')
            info = json.loads(raw)
            return info["sizeX"], info["sizeY"], info["sizeZ"]
        return None

    def _create_dataset_configs(self, crop_path: str, dataset_loc: str, ref: str) -> None:
        """Inline replacement for create_dataset_config_files.py with .npy fallback."""
        crop_dirs = sorted(glob.glob(join(crop_path, "*")))
        skipped = []
        for crop_dir in crop_dirs:
            if not os.path.isdir(crop_dir):
                continue
            crop_name = os.path.basename(crop_dir)
            for side in ("L", "R"):
                size = self._get_crop_size(crop_dir, side)
                if size is None:
                    skipped.append(f"{crop_name}/{side}")
                    continue
                sx, sy, sz = size
                side_long = "left" if side == "L" else "right"
                dataset_name = compute_region_model_name(crop_name, side_long)
                filedata = (
                    ref.replace("REPLACE_CROP_NAME", crop_name)
                    .replace("REPLACE_DATASET", dataset_name)
                    .replace("REPLACE_SIDE", side)
                    .replace("REPLACE_SIZEX", str(sx))
                    .replace("REPLACE_SIZEY", str(sy))
                    .replace("REPLACE_SIZEZ", str(sz))
                )
                result_file = join(dataset_loc, f"{dataset_name}.yaml")
                with open(result_file, "w") as f:
                    f.write(filedata)
                print(result_file)
        if skipped:
            print(f"Skipped {len(skipped)} sulci (no .minf or .npy found): {skipped}")

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
        print(f"Localization config written: {dest_path}")

    def _find_checkout_dataset_configs(self, champollion_loc: str) -> str | None:
        """Return <champollion_loc>/champollion/configs/dataset/<dataset> if that dir exists, else None.

        Used only to print a notice: Hydra's primary config_path (the checkout's
        built-in configs) wins over hydra.searchpath (--config-dir), so stale
        in-checkout YAMLs from earlier default runs shadow the newly generated
        ones in training. O(1).
        """
        path = join(champollion_loc, "champollion", "configs", "dataset", self.args.dataset)
        return path if os.path.isdir(path) else None

    def run(self):
        """Execute the champollion config generation script."""
        self._validate_inputs()

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

        # Create dataset directory if it doesn't exist
        if not exists(dataset_loc):
            self.execute_command(["mkdir", "-p", dataset_loc], shell=False)

        # Always copy reference.yaml from template so re-runs regenerate cleanly
        reference_yaml_dest = join(dataset_loc, "reference.yaml")
        reference_yaml_src = join(dirname(dirname(_SCRIPT_DIR)), "reference.yaml")
        self.execute_command(["cp", reference_yaml_src, dataset_loc], shell=False)

        my_lines = []
        with open(reference_yaml_dest, "r") as f:
            for line in f.readlines():
                if self.args.external_crops:
                    # External crops: derive the full relative path from crop_path
                    # e.g. crop_path = /external/shared_project/TEST01/path/to/crops/crops/2mm
                    #      dataset_folder = /external/shared_project
                    #      => relative = TEST01/path/to/crops/crops/2mm
                    relative_path = os.path.relpath(self.args.crop_path, dataset_folder)
                    my_lines.append(line.replace("TESTXX/crops/2mm", relative_path))
                else:
                    # Standard: use the known derivatives folder structure including mask version
                    computed_path = f"{self.args.dataset}/derivatives/{DERIVATIVES_FOLDER}/crops/{self.args.masks}/2mm"
                    my_lines.append(line.replace("TESTXX/crops/2mm", computed_path))

        with open(reference_yaml_dest, "w") as f:
            f.writelines(my_lines)

        print(f"generate_champollion_config.py/champollion_loc: {champollion_loc}")

        with open(reference_yaml_dest, "r") as f:
            ref = f.read()
        self._create_dataset_configs(self.args.crop_path, dataset_loc, ref)
        result = 0

        # Write the dataset_localization YAML for the requested environment.
        localization_name = f"{self.args.localization}.yaml"

        if self.args.external_config:
            external_yaml = abspath(self.args.external_config)
            if os.path.isdir(external_yaml):
                external_yaml = join(external_yaml, "dataset_localization", localization_name)
            self._write_localization_yaml(external_yaml, dataset_folder)
        else:
            # Default: follow configs_root (--output if given, else derivatives default).
            default_yaml = join(configs_root, "dataset_localization", localization_name)
            self._write_localization_yaml(default_yaml, dataset_folder)

        # Warn if stale in-checkout dataset configs exist; they are now only a fallback
        # for regions absent from configs_root, but their presence is worth flagging as clutter.
        stale = self._find_checkout_dataset_configs(champollion_loc)
        if stale:
            print(
                f"Notice: stale in-checkout dataset configs at {stale} are now a fallback "
                f"(only used for regions not found in {configs_root}); consider deleting them."
            )

        return result


def main():
    """Main entry point."""
    init_pipeline_process()
    script = GenerateChampollionConfig()
    return script.build().print_args().run()


if __name__ == "__main__":
    exit(main())
