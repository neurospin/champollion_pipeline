#!/usr/bin/env python3
"""
Stand-alone script to compare two cortical_tiles outputs.

Scans each output directory for mask files matching
    crops/2mm/*/mask/<pattern>
and for every matched pair computes one or both of:

  diff         – voxel-level change counts:
                   changed : total voxels where values differ
                   added   : voxels absent in set_a but present in set_b
                   removed : voxels present in set_a but absent in set_b

  wasserstein  – approximate 3-D Earth Mover's Distance (in voxels) via
                 axis-marginal projections (same method as compare_masks.py)

Select the metric with --metric diff|wasserstein|both.
Results are written to a JSON file grouped into voxel-count buckets
(step controlled by --bucket_step).

Usage:
    python compare_cortical_tiles.py \\
        --set_a /path/to/run_a/crops/2mm \\
        --set_b /path/to/run_b/crops/2mm \\
        --output diff_report.json \\
        --pattern "*mask_skeleton.nii.gz" \\
        --metric both \\
        --bucket_step 1
"""

import sys

from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.process_setup import init_pipeline_process
from compare import bucket_label, find_masks, load_mask_vol, run_subcommand, voxel_diff, wasserstein_distance

__all__ = [
    "CompareCorticalTiles",
    "bucket_label",
    "find_masks",
    "load_mask_vol",
    "main",
    "voxel_diff",
    "wasserstein_distance",
]


class CompareCorticalTiles(ScriptBuilder):
    """Compare two cortical_tiles outputs with selectable metric(s)."""

    def __init__(self):
        super().__init__(
            script_name="compare_cortical_tiles",
            description=(
                "Compare two cortical_tiles outputs (crops/2mm directories) "
                "using voxel diff, Wasserstein distance, or both."
            ),
        )
        (
            self.add_required_argument(
                "--set_a", "Path to the first crops/2mm directory (e.g. .../cortical_tiles-2025/crops/2mm)."
            )
            .add_required_argument("--set_b", "Path to the second crops/2mm directory.")
            .add_optional_argument("--output", "Output JSON file path.", default="cortical_tiles_diff_report.json")
            .add_optional_argument(
                "--pattern",
                "Glob pattern for mask files inside each region's mask/ folder. "
                "Use '*mask_cropped.nii.gz' or '*mask_foldlabel.nii.gz' for other types.",
                default="*mask_skeleton.nii.gz",
            )
            .add_argument(
                "--metric",
                choices=["diff", "wasserstein", "both"],
                default="diff",
                help="Comparison metric: voxel diff, Wasserstein distance, or both. Default: diff.",
            )
            .add_optional_argument(
                "--bucket_step",
                "Bucket width for the summary table. In voxels (int) for diff; "
                "in voxel-distance (float) for Wasserstein. Default: 10.",
                default=10,
                type_=float,
            )
        )

    def run(self) -> int:
        names = ("set_a", "set_b", "output", "pattern", "metric", "bucket_step")
        return run_subcommand("cortical_tiles", {name: getattr(self.args, name) for name in names})


def main():
    init_pipeline_process()
    script = CompareCorticalTiles()
    return script.main()


if __name__ == "__main__":
    sys.exit(main())
