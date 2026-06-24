#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate sulcal regions with cortical_tiles from Morphologist's graphs.

Output structure
----------------
Sulcal region crops are written to::

    {output}/cortical_tiles-{VERSION}/crops/{voxel_size}mm/{region}/

where ``{voxel_size}mm`` is the voxel resolution (e.g. ``2mm``) and
``{VERSION}`` is the cortical_tiles release year (e.g. ``2026``).

When a mask version tag is provided via ``--masks`` (e.g. ``canonical_25``),
it overrides the ``masks_version`` field in the pipeline JSON config so that
the correct labelled masks are used during region extraction.
"""

import json
import shutil
import sys
from os import chdir, getcwd
from os.path import abspath, dirname, exists, isdir, join

from champollion_utils.script_builder import ScriptBuilder
from joblib import cpu_count

from champollion_pipeline.utils.cortical_tiles_config import CorticalTilesConfigFactory, versioned_crops_exist
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER


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
         .add_required_argument(
             "--path_to_graph",
             "Contains the sub-path that, for each subject, permits getting the sulcal graphs.")
         .add_required_argument("--path_sk_with_hull", "Contains the sub-path where to get the skeleton with hull.")
         .add_optional_argument("--sk_qc_path", "The path to the QC file if it exists.", default="")
         .add_optional_argument("--njobs", "Number of CPU cores allowed to use.", default=None, type_=int)
         .add_argument("--input-types", nargs="+", default=None,
                       help="Input types to generate (e.g. skeleton foldlabel extremities). "
                            "Default: all types.")
         .add_flag("--skip-distbottom",
                   "Skip distbottom generation (unused during inference).")
         .add_argument("--regions", nargs="+", default=None,
                       help="Restrict processing to these sulcal regions "
                            "(space-separated). Default: all 28 regions.")
         .add_optional_argument(
             "--masks",
             "Mask version tag (e.g. 'canonical_25'). Overrides "
             "masks_version in the pipeline JSON config.",
             default=None)
         .add_flag("--overwrite",
                   "Re-generate crops even if they already exist for this mask version."))

    def _preflight_check(self, output_abs: str, config_path: str) -> bool:
        """Check for existing crops and migrate legacy flat structure if needed.

        Returns False (caller should abort) when crops for the requested mask
        version already exist and --overwrite was not passed.
        """
        existing_cfg = CorticalTilesConfigFactory.from_pipeline_json(config_path)
        requested_cfg = CorticalTilesConfigFactory.from_args(self.args)

        vox_str = f"{int(requested_cfg.out_voxel_size)}mm"
        derivatives_abs = join(output_abs, DERIVATIVES_FOLDER)

        # Migrate legacy crops written before versioned path was introduced.
        # Old structure: crops/2mm/  →  New structure: crops/{masks_version}/2mm/
        legacy_crops = join(derivatives_abs, "crops", vox_str)
        if isdir(legacy_crops) and existing_cfg is not None:
            old_version = existing_cfg.masks_version
            target = join(derivatives_abs, "crops", old_version, vox_str)
            if not isdir(target):
                print(f"Migrating legacy crops to versioned path: crops/{old_version}/...")
                shutil.move(legacy_crops, target)
            else:
                if not self.args.overwrite:
                    print(
                        f"Legacy crops found at crops/{vox_str}/ but versioned crops already exist "
                        f"at crops/{old_version}/{vox_str}/. Please resolve manually or use --overwrite "
                        "to remove both and regenerate."
                    )
                    return False
                print(
                    f"--overwrite: removing legacy crops/{vox_str}/ and existing crops/{old_version}/{vox_str}/..."
                )
                shutil.rmtree(legacy_crops)
                shutil.rmtree(target)

        if versioned_crops_exist(derivatives_abs, requested_cfg.masks_version, requested_cfg.out_voxel_size):
            if not self.args.overwrite:
                print(
                    f"Crops for mask version '{requested_cfg.masks_version}' already exist. "
                    "Use --overwrite to regenerate."
                )
                return False
            print(f"Overwriting existing crops for '{requested_cfg.masks_version}'.")
        return True

    def run(self):
        """Execute the cortical_tiles script."""
        print(f"run_cortical_tiles.py/input: {self.args.input}")
        print(f"run_cortical_tiles.py/output: {self.args.output}")

        # Validate paths
        if not self.validate_paths([self.args.input, self.args.output]):
            raise ValueError("run_cortical_tiles.py: Please input valid paths.")

        # Convert input to absolute path
        input_abs = abspath(self.args.input)

        output_abs = abspath(self.args.output)
        config_file_path: str = join(output_abs, "pipeline_loop_2mm.json")

        if not self._preflight_check(output_abs, config_file_path):
            return 0

        # Copy pipeline config template into the output (derivatives) directory.
        # generate_sulcal_regions.py reads it from {path_dataset}/pipeline_loop_2mm.json,
        # and we pass output as -d so the config is never inside the subjects directory
        # (which would cause generate_skeletons.py to list it as a subject).
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
            config['path_to_graph'] = self.args.path_to_graph
            config['path_to_skeleton_with_hull'] = self.args.path_sk_with_hull
            config['masks_version'] = self.args.masks if self.args.masks else 'canonical_25'
            config['skel_qc_path'] = self.args.sk_qc_path if self.args.sk_qc_path else ""
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
            print(
                f"run_cortical_tiles.py: Warning - {self.args.njobs} jobs requested "
                f"but only {cpu_count()} cores available."
            )

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
            "-d", abspath(self.args.output),
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

        if self.args.regions:
            cmd.extend(["-r"] + self.args.regions)

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
