#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate sulcal regions with deep_folding from Morphologist's graphs.
"""

import sys
from os import getcwd, chdir, getcwd
from os.path import abspath, dirname, join
from joblib import cpu_count

from champollion_utils.src.champollion_utils.script_builder import ScriptBuilder


class RunDeepFolding(ScriptBuilder):
    """Script for running deep_folding to generate sulcal regions."""

    def __init__(self):
        super().__init__(
            script_name="run_deep_folding",
            description="Generating sulcal regions with deep_folding from Morphologist's graphs."
        )
        # Configure arguments using method chaining
        (self.add_argument("input", help="Absolute path to Morphologist's graphs.")
         .add_argument("output", help="Absolute path to the generated sulcal regions from deep_folding.")
         .add_optional_argument("--region-file", "Absolute path to the user's sulcal region's configuration file.")
         .add_required_argument("--path_to_graph", "Contains the sub-path that, for each subject, permits getting the sulcal graphs.")
         .add_required_argument("--path_sk_with_hull", "Contains the sub-path where to get the skeleton with hull.")
         .add_optional_argument("--sk_qc_path", "The path to the QC file if it exists.", default="")
         .add_optional_argument("--njobs", "Number of CPU cores allowed to use.", default=None, type_=int))

    def run(self):
        """Execute the deep_folding script."""
        print(f"run_deep_folding.py/input: {self.args.input}")
        print(f"run_deep_folding.py/output: {self.args.output}")

        # Validate paths
        if not self.validate_paths([self.args.input, self.args.output]):
            raise ValueError("run_deep_folding.py: Please input valid paths.")
        
        # If not exist in the current dataset copy config file
        config_file_path: str = join(self.args.input, "pipeline_loop_2mm.json")
        if not self.validate_paths([config_file_path]):
            self.execute_command(["cp", "../pipeline_loop_2mm.json", config_file_path], shell=False)

        # Prepare njobs
        if self.args.njobs is None:
            self.args.njobs = min(22, cpu_count() - 2)

        if self.args.njobs >= cpu_count():
            print(f"run_deep_folding.py: Warning - {self.args.njobs} jobs requested but only {cpu_count()} cores available.")

        # Build command to run deep_folding script directly
        # Get absolute paths
        script_path = abspath(join(
            dirname(__file__),
            '..', '..', 'deep_folding', 'deep_folding', 'brainvisa', 'generate_sulcal_regions.py'
        ))

        # Convert input/output to absolute paths since we'll be changing directory
        input_abs = abspath(self.args.input)

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

        # Change to script directory to run the command
        chdir(script_dir)
        result = self.execute_command(cmd, shell=False)
        # Change back to original directory
        chdir(current_dir)

        return result


def main():
    """Main entry point."""
    script = RunDeepFolding()
    return script.build().print_args().run()


if __name__ == "__main__":
    exit(main())
