#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to define and generate Champollion's configuration.
"""

from os import getcwd, chdir
from os.path import exists, join

from champollion_utils.src.champollion_utils.script_builder import ScriptBuilder
from utils.lib import get_nth_parent_dir


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
                                default=join(getcwd(), "../../champollion_V1/"))
        .add_optional_argument("--output", 
                               "Absolute path to desired output. Default is in Champollion_V1/config/dataset/"))

    def _validate_inputs(self):
        """Validate input paths."""
        if not exists(self.args.crop_path):
            raise ValueError(
                f"generate_champollion_config: Please input correct values. "
                f"{self.args.crop_path} does not exist."
            )

    def _handle_yaml_conf(self, conf_loc: str, crops_loc: str):
        """Load and update the yaml configuration file."""
        lines = []

        with open(conf_loc, "r") as f:
            for line in f.readlines():
                if "dataset_folder" in line:
                    lines.append(
                        f"dataset_folder: {get_nth_parent_dir(crops_loc, 5)}\n"
                    )
                else:
                    lines.append(line)

        with open(conf_loc, "w") as f:
            f.writelines(lines)

    def run(self):
        """Execute the champollion config generation script."""
        self._validate_inputs()

        local_dir = getcwd()

        output_loc: str = f"{self.args.champollion_loc}contrastive/configs/dataset/{self.args.dataset}" if not self.args.output else self.args.output

        dataset_loc = join(
            local_dir,
            output_loc
        )

        # Create dataset directory if it doesn't exist
        if not exists(dataset_loc):
            self.execute_command(["mkdir", "-p", dataset_loc], shell=False)

        # Copy reference.yaml if it doesn't exist
        if not exists(join(dataset_loc, "reference.yaml")):
            self.execute_command(["cp", "../reference.yaml", dataset_loc], shell=False)

        chdir(dataset_loc)

        # Update reference.yaml with dataset path
        my_lines = []
        with open("reference.yaml", 'r') as f:
            for line in f.readlines():
                computed_path = f"{self.args.dataset}/derivatives/deep_folding-2025"
                my_lines.append(line.replace("TESTXX", computed_path))

        with open("reference.yaml", "w") as f:
            f.writelines(my_lines)

        chdir(self.args.champollion_loc)
        print(f"generate_champollion_config.py/main/chdir: {getcwd()}")

        # Build command for create_dataset_config_files.py
        cmd = [
            "python3",
            "./contrastive/utils/create_dataset_config_files.py",
            "--path", dataset_loc,
            "--crop_path", self.args.crop_path
        ]

        result = self.execute_command(cmd, shell=False)

        # Handle YAML configuration
        self._handle_yaml_conf(
            "./contrastive/configs/dataset_localization/local.yaml",
            self.args.crop_path
        )

        chdir(local_dir)

        return result


def main():
    """Main entry point."""
    script = GenerateChampollionConfig()
    return script.build().print_args().run()


if __name__ == "__main__":
    exit(main())
