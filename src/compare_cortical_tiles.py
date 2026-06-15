#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import json
import sys
from pathlib import Path

import numpy as np
from champollion_utils.script_builder import ScriptBuilder
from scipy.stats import wasserstein_distance as _wasserstein_1d

try:
    from soma import aims
except ImportError:
    print("ERROR: PyAIMS (soma.aims) is not available in this environment.")
    sys.exit(1)


def load_mask_vol(path: str) -> np.ndarray:
    """Read a NIfTI mask with PyAIMS and return a 3-D float64 array (x, y, z)."""
    vol = aims.read(path)
    return np.asarray(vol, dtype=np.float64).squeeze()


def voxel_diff(a: np.ndarray, b: np.ndarray) -> dict:
    """Count voxels that changed between two mask volumes.

    Returns:
      changed  – total voxels where a[i] != b[i]
      added    – voxels that went from 0 in a to nonzero in b
      removed  – voxels that went from nonzero in a to 0 in b
    """
    a_nz = a != 0
    b_nz = b != 0
    return {
        "changed": int(np.sum(a != b)),
        "added": int(np.sum(~a_nz & b_nz)),
        "removed": int(np.sum(a_nz & ~b_nz)),
    }


def wasserstein_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Approximate 3-D Wasserstein distance (in voxels) via axis-marginals.

    Projects each 3-D map onto X, Y, Z axes, computes 1-D Wasserstein
    distances, then combines as Euclidean norm. Returns 0.0 when both empty.
    """
    if a.sum() == 0.0 and b.sum() == 0.0:
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


def bucket_label(value: float, step: float) -> str:
    low = (value // step) * step
    high = low + step
    if step == int(step):
        return f"{int(low)}-{int(high)}vox"
    return f"{low:.2f}-{high:.2f}vox"


def find_masks(directory: Path, pattern: str) -> dict:
    """Return {relative_path: absolute_path} for masks matching pattern.

    Scans directory/*/mask/<pattern> (one region level + mask/ subdir).
    The relative path (e.g. S.CINGULATE.LEFT/mask/Lmask_skeleton.nii.gz)
    is used as the match key so region + side are both captured.
    """
    return {
        str(p.relative_to(directory)): str(p)
        for p in sorted(directory.glob(f"*/mask/{pattern}"))
    }


class CompareCorticalTiles(ScriptBuilder):
    """Compare two cortical_tiles outputs with selectable metric(s)."""

    def __init__(self):
        super().__init__(
            script_name="compare_cortical_tiles",
            description=(
                "Compare two cortical_tiles outputs (crops/2mm directories) "
                "using voxel diff, Wasserstein distance, or both."
            )
        )
        (self.add_required_argument(
             "--set_a",
             "Path to the first crops/2mm directory "
             "(e.g. .../cortical_tiles-2025/crops/2mm).")
         .add_required_argument(
             "--set_b",
             "Path to the second crops/2mm directory.")
         .add_optional_argument(
             "--output",
             "Output JSON file path.",
             default="cortical_tiles_diff_report.json")
         .add_optional_argument(
             "--pattern",
             "Glob pattern for mask files inside each region's mask/ folder. "
             "Use '*mask_cropped.nii.gz' or '*mask_foldlabel.nii.gz' for other types.",
             default="*mask_skeleton.nii.gz")
         .add_argument(
             "--metric",
             choices=["diff", "wasserstein", "both"],
             default="diff",
             help="Comparison metric: voxel diff, Wasserstein distance, or both. "
                  "Default: diff.")
         .add_optional_argument(
             "--bucket_step",
             "Bucket width for the summary table. In voxels (int) for diff; "
             "in voxel-distance (float) for Wasserstein. Default: 10.",
             default=10, type_=float))

    def run(self) -> int:
        dir_a = Path(self.args.set_a)
        dir_b = Path(self.args.set_b)

        if not self.validate_paths([str(dir_a), str(dir_b)]):
            return 1

        masks_a = find_masks(dir_a, self.args.pattern)
        masks_b = find_masks(dir_b, self.args.pattern)

        common = sorted(set(masks_a) & set(masks_b))
        only_a = sorted(set(masks_a) - set(masks_b))
        only_b = sorted(set(masks_b) - set(masks_a))

        print(f"Masks in set_a:      {len(masks_a)}")
        print(f"Masks in set_b:      {len(masks_b)}")
        print(f"Common (compared):   {len(common)}")
        print(f"Only in set_a:       {len(only_a)}")
        print(f"Only in set_b:       {len(only_b)}")

        use_diff = self.args.metric in ("diff", "both")
        use_wass = self.args.metric in ("wasserstein", "both")
        step = self.args.bucket_step

        diff_buckets: dict[str, list] = {}
        wass_buckets: dict[str, list] = {}
        diffs: dict[str, dict] = {}
        distances: dict[str, float] = {}

        for name in common:
            arr_a = load_mask_vol(masks_a[name])
            arr_b = load_mask_vol(masks_b[name])

            if arr_a.shape != arr_b.shape:
                print(f"WARNING: shape mismatch for {name}: "
                      f"{arr_a.shape} vs {arr_b.shape}, skipping.")
                continue

            if use_diff:
                d = voxel_diff(arr_a, arr_b)
                diffs[name] = d
                lbl = bucket_label(d["changed"], step)
                diff_buckets.setdefault(lbl, []).append(name)

            if use_wass:
                dist = round(wasserstein_distance(arr_a, arr_b), 3)
                distances[name] = dist
                lbl = bucket_label(dist, step)
                wass_buckets.setdefault(lbl, []).append(name)

        def sort_buckets(b):
            return dict(sorted(b.items(), key=lambda kv: float(kv[0].split("-")[0])))

        report = {
            "set_a": str(dir_a),
            "set_b": str(dir_b),
            "mask_pattern": self.args.pattern,
            "metric": self.args.metric,
            "summary": {
                "total_common": len(common),
                "only_in_set_a": only_a,
                "only_in_set_b": only_b,
            },
        }

        if use_diff:
            report["diff_by_bucket"] = sort_buckets(diff_buckets)
            report["diffs_per_mask"] = dict(
                sorted(diffs.items(), key=lambda kv: -kv[1]["changed"])
            )

        if use_wass:
            report["wasserstein_by_bucket"] = sort_buckets(wass_buckets)
            report["wasserstein_per_mask"] = dict(
                sorted(distances.items(), key=lambda kv: -kv[1])
            )

        out_path = Path(self.args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nReport written to: {out_path}")

        if use_diff and diff_buckets:
            print(f"\nDiff buckets (step={step}):")
            for lbl, names in sort_buckets(diff_buckets).items():
                print(f"  {lbl:15s}  {len(names):3d} mask(s)")
            unchanged = sum(1 for d in diffs.values() if d["changed"] == 0)
            print(f"Unchanged masks: {unchanged}/{len(diffs)}")

        if use_wass and wass_buckets:
            print(f"\nWasserstein buckets (step={step}):")
            for lbl, names in sort_buckets(wass_buckets).items():
                print(f"  {lbl:15s}  {len(names):3d} mask(s)")

        return 0


def main():
    script = CompareCorticalTiles()
    return script.main()


if __name__ == "__main__":
    sys.exit(main())
