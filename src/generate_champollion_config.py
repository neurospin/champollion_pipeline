#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to define and generate Champollion's configuration.
"""

import glob
import json
import os
from os.path import abspath, dirname, exists, join

import numpy as np
from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER, find_dataset_folder

# Get the script's directory for reliable path resolution
_SCRIPT_DIR = dirname(abspath(__file__))
_DEFAULT_CHAMPOLLION_LOC = abspath(join(_SCRIPT_DIR, '..', 'external', 'champollion_V1'))


class GenerateChampollionConfig(ScriptBuilder):
    """Script for generating Champollion configuration files."""

    def __init__(self):
        super().__init__(
            script_name="generate_champollion_config",
            description="Defining and generating Champollion's configuration."
        )
        # Configure arguments using method chaining
        (self.add_argument("crop_path", help="Absolute path to crops path.", type=str)
         .add_required_argument("--dataset", "Name of the dataset.")
         .add_optional_argument("--champollion_loc", "Absolute path to Champollion binaries.",
                                default=_DEFAULT_CHAMPOLLION_LOC)
         .add_optional_argument("--output",
                                "Absolute path to desired output. Default is in Champollion_V1/config/dataset/")
         .add_optional_argument("--external-config",
                                "External path to write local.yaml (for read-only containers).",
                                default=None)
         .add_flag("--external_crops",
                   "Use crop_path as-is instead of deriving it from --dataset. "
                   "Replaces the dataset/derivatives/... segment with the actual crop_path location. "
                   "Requires --dataset to be set (used for config file naming).")
         .add_optional_argument("--masks",
                                "Mask version tag (e.g. 'canonical_25'). "
                                "Must match the value used when running run_cortical_tiles.",
                                default="canonical_25"))

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
                dataset_name = f"{crop_name.replace('.', '')}_{side_long}"
                filedata = (ref
                            .replace("REPLACE_CROP_NAME", crop_name)
                            .replace("REPLACE_DATASET", dataset_name)
                            .replace("REPLACE_SIDE", side)
                            .replace("REPLACE_SIZEX", str(sx))
                            .replace("REPLACE_SIZEY", str(sy))
                            .replace("REPLACE_SIZEZ", str(sz)))
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
                f"generate_champollion_config: Please input correct values. "
                f"{self.args.crop_path} does not exist."
            )

    def _handle_yaml_conf(self, conf_loc: str, dataset_folder: str,
                          output_loc: str | None = None):
        """Load and update the yaml configuration file.

        Args:
            conf_loc: Path to the source config file
            dataset_folder: Absolute path to the dataset folder root
            output_loc: Optional external output path. If None, writes to conf_loc.
                        Use this for read-only Apptainer containers.
        """
        lines = []

        with open(conf_loc, "r") as f:
            for line in f.readlines():
                if "dataset_folder" in line:
                    lines.append(
                        f"dataset_folder: {dataset_folder}\n"
                    )
                else:
                    lines.append(line)

        # Write to external location if specified, otherwise update in place
        dest_path = output_loc if output_loc else conf_loc
        os.makedirs(dirname(dest_path), exist_ok=True)

        with open(dest_path, "w") as f:
            f.writelines(lines)

        if output_loc:
            print(f"Config written to external path: {dest_path}")

    def run(self):
        """Execute the champollion config generation script."""
        self._validate_inputs()

        # Resolve champollion_loc to absolute path
        champollion_loc = abspath(self.args.champollion_loc)

        # Determine output location.
        # When --output is given it is treated as the configs root (parallel to
        # contrastive/configs/), so region YAMLs land at {output}/dataset/{dataset}/
        # to match the Hydra config-group layout expected by train_champollion.py.
        if self.args.output:
            dataset_loc = join(abspath(self.args.output), "dataset", self.args.dataset)
        else:
            dataset_loc = join(champollion_loc, "contrastive", "configs", "dataset", self.args.dataset)

        # Create dataset directory if it doesn't exist
        if not exists(dataset_loc):
            self.execute_command(["mkdir", "-p", dataset_loc], shell=False)

        # Always copy reference.yaml from template so re-runs regenerate cleanly
        reference_yaml_dest = join(dataset_loc, "reference.yaml")
        reference_yaml_src = join(dirname(_SCRIPT_DIR), "reference.yaml")
        self.execute_command(["cp", reference_yaml_src, dataset_loc], shell=False)

        dataset_folder = find_dataset_folder(self.args.crop_path, self.args.dataset)

        my_lines = []
        with open(reference_yaml_dest, 'r') as f:
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
                    computed_path = (
                        f"{self.args.dataset}/derivatives/{DERIVATIVES_FOLDER}"
                        f"/crops/{self.args.masks}/2mm"
                    )
                    my_lines.append(line.replace("TESTXX/crops/2mm", computed_path))

        with open(reference_yaml_dest, "w") as f:
            f.writelines(my_lines)

        print(f"generate_champollion_config.py/champollion_loc: {champollion_loc}")

        with open(reference_yaml_dest, "r") as f:
            ref = f.read()
        self._create_dataset_configs(self.args.crop_path, dataset_loc, ref)
        result = 0

        # Handle YAML configuration using absolute path
        local_yaml_path = join(champollion_loc, "contrastive", "configs", "dataset_localization", "local.yaml")

        # Support external config for read-only containers (e.g., Apptainer)
        if self.args.external_config:
            external_yaml = abspath(self.args.external_config)
            if os.path.isdir(external_yaml):
                # Mirror the built-in layout: dataset_localization/local.yaml
                external_yaml = join(external_yaml, "dataset_localization", os.path.basename(local_yaml_path))
            self._handle_yaml_conf(local_yaml_path, dataset_folder, external_yaml)
        else:
            self._handle_yaml_conf(local_yaml_path, dataset_folder)

        return result


def main():
    """Main entry point."""
    script = GenerateChampollionConfig()
    return script.build().print_args().run()


if __name__ == "__main__":
    exit(main())
