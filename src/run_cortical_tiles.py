#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate sulcal regions with cortical_tiles from Morphologist's graphs.
"""

import sys
import json
import re
from glob import glob
from os import getcwd, chdir
from os.path import abspath, dirname, join, exists
from joblib import cpu_count

from champollion_utils.script_builder import ScriptBuilder


class RunCorticalTiles(ScriptBuilder):
    """Script for running cortical_tiles to generate sulcal regions."""

    def __init__(self):
        super().__init__(
            script_name="run_cortical_tiles",
            description="Generating sulcal regions with cortical_tiles from Morphologist's graphs."
        )
        # Configure arguments using method chaining
        (self.add_argument("input", help="Absolute path to Morphologist's graphs.")
         .add_argument("output", help="Absolute path to the generated sulcal regions from cortical_tiles.")
         .add_optional_argument("--region-file", "Absolute path to the user's sulcal region's configuration file.")
         .add_required_argument("--path_to_graph", "Contains the sub-path that, for each subject, permits getting the sulcal graphs.")
         .add_required_argument("--path_sk_with_hull", "Contains the sub-path where to get the skeleton with hull.")
         .add_optional_argument("--sk_qc_path", "The path to the QC file if it exists.", default="")
         .add_optional_argument("--njobs", "Number of CPU cores allowed to use.", default=None, type_=int))

    def run(self):
        """Execute the cortical_tiles script."""
        print(f"run_cortical_tiles.py/input: {self.args.input}")
        print(f"run_cortical_tiles.py/output: {self.args.output}")

        # Validate paths
        if not self.validate_paths([self.args.input, self.args.output]):
            raise ValueError("run_cortical_tiles.py: Please input valid paths.")

        # Convert input to absolute path
        input_abs = abspath(self.args.input)

        # If not exist in the current dataset copy config file
        config_file_path: str = join(input_abs, "pipeline_loop_2mm.json")
        if not self.validate_paths([config_file_path]):
            source_config = abspath(join(dirname(__file__), '..', 'pipeline_loop_2mm.json'))
            self.execute_command(["cp", source_config, config_file_path], shell=False)

        # Fix graphs_dir in pipeline_loop_2mm.json
        # The config expects graphs_dir to point to where the morphologist subjects are located
        # This is typically input_path/derivatives/morphologist-X.Y
        if exists(config_file_path):
            with open(config_file_path, 'r') as f:
                config = json.load(f)

            graphs_dir = config.get('graphs_dir', '')

            # Check if graphs_dir points to morphologist directory
            # Pattern matches /morphologist-X.Y where X.Y is version number
            morphologist_pattern = r'/morphologist-\d+\.\d+$'

            # If graphs_dir doesn't end with morphologist-X.Y, find it
            if not re.search(morphologist_pattern, graphs_dir):
                # Look for morphologist directory
                morpho_path = join(input_abs, 'derivatives', 'morphologist-*')
                morphologist_dirs = glob(morpho_path)

                if morphologist_dirs:
                    # Use the first morphologist directory found
                    morphologist_path = morphologist_dirs[0]
                    config['graphs_dir'] = morphologist_path

                    # Write back the corrected config
                    with open(config_file_path, 'w') as f:
                        json.dump(config, f, indent=3)

                    print(f"run_cortical_tiles.py: Corrected graphs_dir "
                          f"from {graphs_dir} to {morphologist_path}")
                else:
                    print(f"run_cortical_tiles.py: Warning - "
                          f"no morphologist directory found in "
                          f"{input_abs}/derivatives/")

        # Prepare njobs
        if self.args.njobs is None:
            self.args.njobs = min(22, cpu_count() - 2)

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
