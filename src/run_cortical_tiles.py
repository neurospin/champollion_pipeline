#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate sulcal regions with cortical_tiles from Morphologist's graphs.
"""

import sys
import json
from os import getcwd, chdir
from os.path import abspath, dirname, join, exists
from joblib import cpu_count

from champollion_utils.script_builder import ScriptBuilder
from utils.lib import DERIVATIVES_FOLDER


class RunCorticalTiles(ScriptBuilder):
    """Script for running cortical_tiles to generate sulcal regions."""

    def __init__(self):
        super().__init__(
            script_name="run_cortical_tiles",
            description="Generating sulcal regions with cortical_tiles from Morphologist's graphs."
        )
        # Configure arguments using method chaining
        (self.add_argument("input", help="Absolute path to the directory containing subject folders "
                                        "(e.g., morphologist's output subjects directory).")
         .add_argument("output", help="Absolute path to the generated sulcal regions from cortical_tiles.")
         .add_optional_argument("--region-file", "Absolute path to the user's sulcal region's configuration file.")
         .add_required_argument("--path_to_graph", "Contains the sub-path that, for each subject, permits getting the sulcal graphs.")
         .add_required_argument("--path_sk_with_hull", "Contains the sub-path where to get the skeleton with hull.")
         .add_optional_argument("--sk_qc_path", "The path to the QC file if it exists.", default="")
         .add_optional_argument("--njobs", "Number of CPU cores allowed to use.", default=None, type_=int)
         .add_argument("--input-types", nargs="+", default=None,
                       help="Input types to generate (e.g. skeleton foldlabel extremities). "
                            "Default: all types.")
         .add_flag("--skip-distbottom",
                   "Skip distbottom generation (unused during inference)."))

    def run(self):
        """Execute the cortical_tiles script."""
        print(f"run_cortical_tiles.py/input: {self.args.input}")
        print(f"run_cortical_tiles.py/output: {self.args.output}")

        # Validate paths
        if not self.validate_paths([self.args.input, self.args.output]):
            raise ValueError("run_cortical_tiles.py: Please input valid paths.")

        # Convert input to absolute path
        input_abs = abspath(self.args.input)

        # Copy pipeline config template inside the input directory
        # (generate_sulcal_regions.py always looks for it at {path_dataset}/pipeline_loop_2mm.json)
        config_file_path: str = join(input_abs, "pipeline_loop_2mm.json")
        if not self.validate_paths([config_file_path]):
            source_config = abspath(join(
                dirname(__file__), '..', 'pipeline_loop_2mm.json'
            ))
            self.execute_command(
                ["cp", source_config, config_file_path],
                shell=False
            )

        # Set graphs_dir and output_dir in the pipeline JSON config.
        # generate_sulcal_regions.py derives both from path_dataset (-d arg)
        # when they are "$local". Since -d is now the subjects dir (not the
        # dataset root), we set them explicitly so outputs land in the right place.
        if exists(config_file_path):
            with open(config_file_path, 'r') as f:
                config = json.load(f)
            config['graphs_dir'] = input_abs
            config['output_dir'] = join(
                abspath(self.args.output), DERIVATIVES_FOLDER
            )
            with open(config_file_path, 'w') as f:
                json.dump(config, f, indent=3)

        # Set skip_distbottom in pipeline JSON if requested
        if self.args.skip_distbottom and exists(config_file_path):
            with open(config_file_path, 'r') as f:
                config = json.load(f)
            config['skip_distbottom'] = True
            with open(config_file_path, 'w') as f:
                json.dump(config, f, indent=3)

        # Prepare njobs
        if self.args.njobs is None:
            self.args.njobs = max(1, min(22, cpu_count() - 2))

        if self.args.njobs >= cpu_count():
            print(f"run_cortical_tiles.py: Warning - {self.args.njobs} jobs requested but only {cpu_count()} cores available.")

        # Build command to run cortical_tiles script directly
        # Get absolute paths
        script_path = abspath(join(
            dirname(__file__),
            '..', 'external', 'cortical_tiles', 'deep_folding', 'brainvisa', 'generate_sulcal_regions.py'
        ))

        # Get the directory where the script lives so we can run from there
        script_dir = dirname(script_path)
        current_dir = getcwd()

        cmd = [
            sys.executable,
            script_path,
            "-d", input_abs,
            "--path_to_graph", self.args.path_to_graph,
            "--path_sk_with_hull", self.args.path_sk_with_hull,
            "--njobs", str(self.args.njobs)
        ]

        # # Add optional arguments if provided
        # # If region_file is not provided, use the default one from champollion_pipeline
        # if self.args.region_file:
        #     cmd.extend(["--region-file", abspath(self.args.region_file)])
        # else:
        #     # Use default region file from champollion_pipeline directory
        #     default_region_file = abspath(join(
        #         dirname(__file__),
        #         '..', 'sulci_regions_champollion_V1.json'
        #     ))
        #     if exists(default_region_file):
        #         cmd.extend(["--region-file", default_region_file])

        if self.args.sk_qc_path:
            cmd.extend(["--sk_qc_path", self.args.sk_qc_path])

        if self.args.input_types:
            cmd.extend(["-y"] + self.args.input_types)

        # Change to script directory to run the command
        chdir(script_dir)
        result = self.execute_command(cmd, shell=False)
        # Change back to original directory
        chdir(current_dir)

        return result


def main():
    """Main entry point."""
    script = RunCorticalTiles()
    return script.build().print_args().run()


if __name__ == "__main__":
    exit(main())
