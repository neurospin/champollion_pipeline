#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stand-alone script to compare two sets of sulcal masks.

Loads each matched pair with PyAIMS and computes one or both of:

  wasserstein  – approximate 3-D Earth Mover's Distance (in voxels) via
                 axis-marginal projections, grouped into bucket steps.
                 The distance is reported in voxels: for 2 mm masks, 1 vox ≈ 2 mm.

  diff         – voxel-level change counts:
                   changed : total voxels where values differ
                   added   : voxels absent in set_a but present in set_b
                   removed : voxels present in set_a but absent in set_b

Select the metric with --metric wasserstein|diff|both.

Usage:
    python compare_masks.py --set_a /path/to/mask/canonical_25/2.0 \\
        --set_b /path/to/mask/corrected_canonical_26_1/2.0 \\
        --output diff_report.json \\
        --metric both \\
        --bucket_step 1
"""

import argparse
import sys

from compare import bucket_label, find_masks, load_mask_vol, run_subcommand, voxel_diff, wasserstein_distance

__all__ = ["bucket_label", "find_masks", "load_mask_vol", "main", "parse_args", "voxel_diff", "wasserstein_distance"]


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Compare two sets of sulcal masks and report deviations.")
    parser.add_argument(
        "--set_a", required=True, help="Path to the first mask set directory (e.g. .../canonical_25/2.0)."
    )
    parser.add_argument("--set_b", required=True, help="Path to the second mask set directory.")
    parser.add_argument(
        "--output", default="mask_diff_report.json", help="Output JSON file path. Default: mask_diff_report.json."
    )
    parser.add_argument(
        "--metric",
        choices=["wasserstein", "diff", "both"],
        default="wasserstein",
        help="Comparison metric. Default: wasserstein.",
    )
    parser.add_argument(
        "--bucket_step", type=float, default=1.0, help="Bucket width for the summary table (in voxels). Default: 1."
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return run_subcommand(
        "masks",
        {
            "set_a": args.set_a,
            "set_b": args.set_b,
            "output": args.output,
            "metric": args.metric,
            "bucket_step": args.bucket_step,
        },
    )


if __name__ == "__main__":
    sys.exit(main())
