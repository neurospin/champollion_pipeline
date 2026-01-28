#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to define and generate Champollion's configuration.
"""

import os
from os.path import exists, join, abspath, dirname

from champollion_utils.script_builder import ScriptBuilder
from utils.lib import get_nth_parent_dir

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
                                default=None))

    def _validate_inputs(self):
        """Validate input paths."""
        if not exists(self.args.crop_path):
            raise ValueError(
                f"generate_champollion_config: Please input correct values. "
                f"{self.args.crop_path} does not exist."
            )

    def _handle_yaml_conf(self, conf_loc: str, crops_loc: str, output_loc: str | None = None):
        """Load and update the yaml configuration file.

        Args:
            conf_loc: Path to the source config file
            crops_loc: Path to crops location
            output_loc: Optional external output path. If None, writes to conf_loc.
                        Use this for read-only Apptainer containers.
        """
        lines = []

        with open(conf_loc, "r") as f:
            for line in f.readlines():
                if "dataset_folder" in line:
                    lines.append(
                        f"dataset_folder: {get_nth_parent_dir(crops_loc, 5)}\n"
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

        # Determine output location
        if self.args.output:
            dataset_loc = abspath(self.args.output)
        else:
            dataset_loc = join(champollion_loc, "contrastive", "configs", "dataset", self.args.dataset)

        # Create dataset directory if it doesn't exist
        if not exists(dataset_loc):
            self.execute_command(["mkdir", "-p", dataset_loc], shell=False)

        # Copy reference.yaml if it doesn't exist
        reference_yaml_dest = join(dataset_loc, "reference.yaml")
        if not exists(reference_yaml_dest):
            # reference.yaml is in the same directory as this script's src folder
            reference_yaml_src = join(_SCRIPT_DIR, "reference.yaml")
            self.execute_command(["cp", reference_yaml_src, dataset_loc], shell=False)

        # Update reference.yaml with dataset path (using absolute path)
        my_lines = []
        with open(reference_yaml_dest, 'r') as f:
            for line in f.readlines():
                computed_path = f"{self.args.dataset}/derivatives/deep_folding-2025"
                my_lines.append(line.replace("TESTXX", computed_path))

        with open(reference_yaml_dest, "w") as f:
            f.writelines(my_lines)

        print(f"generate_champollion_config.py/champollion_loc: {champollion_loc}")

        # Build command for create_dataset_config_files.py using absolute paths
        script_path = join(champollion_loc, "contrastive", "utils", "create_dataset_config_files.py")
        cmd = [
            "python3",
            script_path,
            "--path", dataset_loc,
            "--crop_path", self.args.crop_path
        ]

        result = self.execute_command(cmd, shell=False)

        # Handle YAML configuration using absolute path
        local_yaml_path = join(champollion_loc, "contrastive", "configs", "dataset_localization", "local.yaml")

        # Support external config for read-only containers (e.g., Apptainer)
        if self.args.external_config:
            external_yaml = abspath(self.args.external_config)
            self._handle_yaml_conf(local_yaml_path, self.args.crop_path, external_yaml)
        else:
            self._handle_yaml_conf(local_yaml_path, self.args.crop_path)

        return result


def main():
    """Main entry point."""
    script = GenerateChampollionConfig()
    return script.build().print_args().run()


if __name__ == "__main__":
    exit(main())
