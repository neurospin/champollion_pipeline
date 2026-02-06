#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate graphs with morphologist from the user's raw data.
"""

from os import getcwd, chdir, listdir
from os.path import isfile, join, splitext, basename

from champollion_utils.script_builder import ScriptBuilder


class GenerateMorphologistGraphs(ScriptBuilder):
    """Script for generating graphs with morphologist."""

    def __init__(self):
        super().__init__(
            script_name="morphologist_graphs_generator",
            description="Generating graphs with morphologist from the user raw data."
        )
        # Configure arguments using method chaining
        (self.add_argument("input", help="Absolute path to the user's raw data.")
         .add_argument("output", help="Absolute path to the generated graphs from morphologist. "
                                      "Morphologist will create a $output/derivatives/morphologist-6.0/ "
                                      "directory for output generations.")
         .add_flag("--parallel", "Enable parallel processing using Soma-Workflow (--swf).")
         .add_flag("--enable-sulcal-recognition",
                   "Enable sulcal recognition (adds 10-20 min/subject, disabled by default for embeddings pipeline)."))

    def _get_input_files(self):
        """Get list of valid input files."""
        # List of allowed extensions for files as raw data for the pipeline
        LIST_OF_EXTENSIONS = [".nii.gz", ".nii", ".gz"]

        input_files = [
            f for f in listdir(self.args.input)
            if isfile(join(self.args.input, f))
            and splitext(basename(f))[1] in LIST_OF_EXTENSIONS
        ]

        return input_files

    def run(self):
        """Execute the morphologist script."""
        print(f"Current working directory: {getcwd()}")

        # Validate paths
        if not self.validate_paths([self.args.input, self.args.output]):
            raise ValueError(
                "generate_morphologist_graphs.py: Please input valid paths."
            )

        input_files = self._get_input_files()

        local_dir = getcwd()
        chdir(self.args.input)

        # Build command
        cmd = [
            "morphologist-cli",
            *input_files,
            self.args.output,
            "--",
            "--of",
            "morphologist-auto-nonoverlap-1.0"
        ]

        # Add parallel processing flag (must come before process parameters)
        if self.args.parallel:
            cmd.append("--swf")

        # Skip sulcal recognition by default (process parameters come last)
        if not self.args.enable_sulcal_recognition:
            cmd.append("SulciRecognition.selected=0")

        result = self.execute_command(cmd, shell=True)

        chdir(local_dir)

        return result


def main():
    """Main entry point."""
    script = GenerateMorphologistGraphs()
    return script.build().print_args().run()


if __name__ == "__main__":
    exit(main())
