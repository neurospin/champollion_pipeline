#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stand-alone script to compare two sets of sulcal masks.

Loads each matched pair with PyAIMS, converts to numpy probability maps
(integer counts, one value per voxel), computes the approximate 3-D
Wasserstein (Earth Mover's) distance via axis-marginal projections, then
writes a JSON file grouping mask names by distance bucket (1-voxel steps).

The distance is reported in voxels: for 2 mm masks, 1 vox ≈ 2 mm.

Usage:
    python compare_masks.py --set_a /path/to/mask/canonical_25/2.0 \\
        --set_b /path/to/mask/corrected_canonical_26_1/2.0 \\
        --output diff_report.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance as _wasserstein_1d

try:
    from soma import aims
except ImportError:
    print("ERROR: PyAIMS (soma.aims) is not available in this environment.")
    sys.exit(1)


_BUCKET_STEP = 1  # voxels


def load_mask_vol(path: str) -> np.ndarray:
    """Read a NIfTI mask with PyAIMS and return a 3-D float64 array (x, y, z).

    The raw integer counts (0 = absent, k = k subjects) are preserved as-is
    so they can serve as probability weights in the Wasserstein computation.
    """
    vol = aims.read(path)
    return np.asarray(vol, dtype=np.float64).squeeze()


def wasserstein_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Approximate 3-D Wasserstein distance (in voxels) via axis-marginals.

    Each 3-D probability map is projected onto each spatial axis (summing
    over the other two axes).  The 1-D Wasserstein distances along X, Y, Z
    are combined as the Euclidean norm, giving an approximation of the true
    3-D Earth Mover's Distance.  Returns 0.0 when both maps are empty.
    """
    sum_a = a.sum()
    sum_b = b.sum()
    if sum_a == 0.0 and sum_b == 0.0:
        return 0.0

    d_sq = 0.0
    for axis in range(3):
        other = tuple(i for i in range(3) if i != axis)
        proj_a = a.sum(axis=other)
        proj_b = b.sum(axis=other)
        s_a = proj_a.sum()
        s_b = proj_b.sum()
        if s_a == 0.0 or s_b == 0.0:
            continue
        positions = np.arange(len(proj_a), dtype=np.float64)
        d = float(_wasserstein_1d(positions, positions,
                                  proj_a / s_a, proj_b / s_b))
        d_sq += d * d
    return float(np.sqrt(d_sq))


def bucket_label(dist: float) -> str:
    """Return the 5-voxel-step bucket label for a given distance value."""
    low = int(dist // _BUCKET_STEP) * _BUCKET_STEP
    high = low + _BUCKET_STEP
    return f"{low}-{high}vox"


def find_masks(directory: Path) -> dict:
    """Return {relative_path: absolute_path} for final masks under directory.

    Only matches files at exactly one subdirectory deep (side/sulcus.nii.gz),
    ignoring intermediate per-subject files (side/sulcus/subject.nii.gz).
    Uses the relative path as key so side is included and cross-sulcus subject
    filename collisions cannot occur.
    """
    return {
        str(p.relative_to(directory)): str(p)
        for p in sorted(directory.glob("*/*.nii.gz"))
    }


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Compare two sets of sulcal masks and report deviations.")
    parser.add_argument(
        "--set_a", required=True,
        help="Path to the first mask set directory "
             "(e.g. .../canonical_25/2.0).")
    parser.add_argument(
        "--set_b", required=True,
        help="Path to the second mask set directory.")
    parser.add_argument(
        "--output", default="mask_diff_report.json",
        help="Output JSON file path. Default: mask_diff_report.json.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    dir_a = Path(args.set_a)
    dir_b = Path(args.set_b)

    if not dir_a.is_dir():
        print(f"ERROR: set_a directory not found: {dir_a}")
        sys.exit(1)
    if not dir_b.is_dir():
        print(f"ERROR: set_b directory not found: {dir_b}")
        sys.exit(1)

    masks_a = find_masks(dir_a)
    masks_b = find_masks(dir_b)

    common = sorted(set(masks_a) & set(masks_b))
    only_a = sorted(set(masks_a) - set(masks_b))
    only_b = sorted(set(masks_b) - set(masks_a))

    print(f"Masks in set_a:      {len(masks_a)}")
    print(f"Masks in set_b:      {len(masks_b)}")
    print(f"Common (compared):   {len(common)}")
    print(f"Only in set_a:       {len(only_a)}")
    print(f"Only in set_b:       {len(only_b)}")

    # --- compare common masks ---
    buckets: dict[str, list] = {}
    deviations: dict[str, float] = {}

    for name in common:
        arr_a = load_mask_vol(masks_a[name])
        arr_b = load_mask_vol(masks_b[name])
        dist = wasserstein_distance(arr_a, arr_b)
        deviations[name] = round(dist, 3)
        label = bucket_label(dist)
        buckets.setdefault(label, []).append(name)

    # Sort buckets by their lower bound
    sorted_buckets = dict(
        sorted(buckets.items(), key=lambda kv: int(kv[0].split("-")[0]))
    )

    # --- build report ---
    report = {
        "set_a": str(dir_a),
        "set_b": str(dir_b),
        "summary": {
            "total_common": len(common),
            "only_in_set_a": only_a,
            "only_in_set_b": only_b,
        },
        "deviations_by_bucket": sorted_buckets,
        "deviations_per_mask": dict(
            sorted(deviations.items(), key=lambda kv: -kv[1])
        ),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to: {out_path}")

    # Print quick summary
    if sorted_buckets:
        print("\nDeviation buckets:")
        for label, names in sorted_buckets.items():
            print(f"  {label:10s}  {len(names):3d} mask(s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
