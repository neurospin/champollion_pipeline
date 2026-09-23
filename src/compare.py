#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified comparison script for sulcal data.

Subcommands
-----------
  masks           Compare two sets of sulcal NIfTI masks (side/sulcus.nii.gz).
  cortical_tiles  Compare two cortical_tiles outputs (region/mask/<pattern>).
  databases       Compare sulcal labeling between two graph annotation campaigns.

Mask metrics (masks / cortical_tiles)
--------------------------------------
  wasserstein  – approximate 3-D Earth Mover's Distance via axis-marginal
                 projections (in voxels; for 2 mm masks, 1 vox ≈ 2 mm).
  diff         – voxel-level change counts: changed / added / removed.
  both         – compute and report both metrics.

Usage
-----
    python compare.py masks --set_a /path/to/mask/canonical_25/2.0 \\
        --set_b /path/to/mask/corrected/2.0 --metric both

    python compare.py cortical_tiles \\
        --set_a /path/to/run_a/crops/2mm \\
        --set_b /path/to/run_b/crops/2mm \\
        --pattern "*mask_skeleton.nii.gz" --metric diff

    python compare.py databases \\
        --labeled_subjects_dir /neurospin/.../manually_labeled/pclean/all \\
        --path_to_graph_a t1mri/t1/default_analysis/folds/3.3/base2018_manual \\
        --path_to_graph_b t1mri/t1/default_analysis/folds/3.3/base2018b_manual \\
        --label_a base2018 --label_b base2018b
"""

import argparse
import csv
import json
import os
import sys
from os.path import abspath, dirname, join
from pathlib import Path

import numpy as np
from champollion_utils.script_builder import ScriptBuilder
from scipy.stats import wasserstein_distance as _wasserstein_1d

try:
    from soma import aims
except ImportError:
    print("ERROR: PyAIMS (soma.aims) is not available in this environment.")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Shared NIfTI mask helpers
# --------------------------------------------------------------------------- #

def load_mask_vol(path: str) -> np.ndarray:
    """Read a NIfTI mask with PyAIMS and return a 3-D float64 array (x, y, z).

    Raw integer counts are preserved as-is for use as Wasserstein weights.
    """
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

    Projects each 3-D map onto X, Y, Z, computes 1-D Wasserstein distances,
    then combines as Euclidean norm. Returns 0.0 when both maps are empty.
    """
    if a.sum() == 0.0 and b.sum() == 0.0:
        return 0.0
    d_sq = 0.0
    for axis in range(3):
        other = tuple(i for i in range(3) if i != axis)
        proj_a = a.sum(axis=other)
        proj_b = b.sum(axis=other)
        s_a, s_b = proj_a.sum(), proj_b.sum()
        if s_a == 0.0 or s_b == 0.0:
            continue
        positions = np.arange(len(proj_a), dtype=np.float64)
        d = float(_wasserstein_1d(positions, positions, proj_a / s_a, proj_b / s_b))
        d_sq += d * d
    return float(np.sqrt(d_sq))


def bucket_label(value: float, step: float) -> str:
    low = (value // step) * step
    high = low + step
    if step == int(step):
        return f"{int(low)}-{int(high)}vox"
    return f"{low:.2f}-{high:.2f}vox"


def sort_buckets(b: dict) -> dict:
    return dict(sorted(b.items(), key=lambda kv: float(kv[0].split("-")[0])))


def visualise_mask_diffs(diffs: dict, masks_a: dict, masks_b: dict) -> None:
    """Open an interactive Anatomist session for changed mask pairs.

    For each of the 5 most-changed mask pairs, opens three Axial windows:
      - A: set_a volume  → grey palette   (B-W LINEAR)
      - B: set_b volume  → violet palette  (VIOLET-lfusion)
      - XOR: diff volume → red palette    (RED TEMPERATURE, binary 0/1)

    Spawns a fresh Python subprocess so Anatomist gets a clean QApplication
    with no pre-existing Qt state from the calling process (e.g. VS Code).
    XOR volumes are written to temp NIfTI files passed to the subprocess.
    """
    import json
    import subprocess
    import sys
    import tempfile

    changed = {n: d for n, d in diffs.items() if d["changed"] > 0}
    if not changed:
        print("No changed masks to visualise.")
        return

    top5 = sorted(changed.items(), key=lambda kv: -kv[1]["changed"])[:5]
    print(f"\nOpening Anatomist for {len(changed)} changed mask(s) "
          f"(showing up to 5 most changed)…")

    tmp_files: list = []
    entries: list = []

    for name, _info in top5:
        path_a = masks_a[name]
        path_b = masks_b[name]

        vol_a_aims = aims.read(path_a)
        arr_a = np.asarray(vol_a_aims, dtype=np.float64).squeeze()
        arr_b = np.asarray(aims.read(path_b), dtype=np.float64).squeeze()
        xor = (arr_a != arr_b).astype(np.int16)
        if xor.ndim == 3:
            xor = xor[..., np.newaxis]
        vol_xor = aims.Volume(xor)
        vol_xor.copyHeaderFrom(vol_a_aims.header())
        tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
        tmp.close()
        aims.write(vol_xor, tmp.name)
        tmp_files.append(tmp.name)
        entries.append({"path_a": path_a, "path_b": path_b, "path_xor": tmp.name,
                        "name": name})

    _VIEWER = """\
import json, sys
import anatomist.direct.api as ana
from soma.qt_gui.qt_backend import Qt

entries = json.loads(sys.argv[1])
a = ana.Anatomist()
block = a.createWindowsBlock(3)
alive = []
for e in entries:
    va  = a.loadObject(e['path_a'])
    vb  = a.loadObject(e['path_b'])
    vx  = a.loadObject(e['path_xor'])
    va.setPalette('B-W LINEAR')
    vb.setPalette('VIOLET-lfusion')
    vx.setPalette('RED TEMPERATURE')
    wa = a.createWindow('Axial', block=block)
    wb = a.createWindow('Axial', block=block)
    wx = a.createWindow('Axial', block=block)
    wa.addObjects([va]); wb.addObjects([vb]); wx.addObjects([vx])
    alive.extend([va, vb, vx, wa, wb, wx])
print('Anatomist ready. Close all windows to exit.')
qt_app = Qt.QApplication.instance()
if qt_app is not None:
    qt_app.exec_()
"""

    try:
        subprocess.run(
            [sys.executable, "-c", _VIEWER, json.dumps(entries)],
            check=False,
        )
    finally:
        for f in tmp_files:
            Path(f).unlink(missing_ok=True)


def save_xor_vol(ref_path: str, a: np.ndarray, b: np.ndarray, out_path: str) -> None:
    """Write a NIfTI XOR image (1 = voxel differs, 0 = same) to out_path.

    Voxel size and transformation are copied from ref_path so the output
    sits in the same space as the input masks.
    """
    ref_vol = aims.read(ref_path)
    xor = (a != b).astype(np.int16)
    if xor.ndim == 3:
        xor = xor[..., np.newaxis]
    out_vol = aims.Volume(xor)
    out_vol.copyHeaderFrom(ref_vol.header())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    aims.write(out_vol, out_path)


# --------------------------------------------------------------------------- #
# Database-mode module-level workers (must be picklable for joblib)
# --------------------------------------------------------------------------- #

def _get_subject_voxel_counts(sub, brainvisa_dir):
    """Worker: load one graph and return (subject_name, {sulcus: voxel_count})."""
    import glob as _glob
    import sys as _sys
    if brainvisa_dir not in _sys.path:
        _sys.path.insert(0, brainvisa_dir)
    from soma import aims  # noqa: PLC0415

    matches = _glob.glob(join(sub['dir'], sub['graph_file']))
    if not matches:
        return sub['subject'], None

    graph = aims.read(matches[0])
    counts: dict = {}
    for vertex in graph.vertices():
        name = vertex.get('name')
        if name is None:
            continue
        n = 0
        for bucket_name in ('aims_ss', 'aims_bottom', 'aims_other'):
            bucket = vertex.get(bucket_name)
            if bucket is not None:
                n += len(list(bucket[0].keys()))
        counts[name] = counts.get(name, 0) + n
    return sub['subject'], counts


def _mask_stats(mask_dir: str, brainvisa_dir: str) -> dict:
    """Return {relative_path: (max, sum, nonzero)} for all masks in mask_dir."""
    import glob as _g
    import sys as _s
    if brainvisa_dir not in _s.path:
        _s.path.insert(0, brainvisa_dir)
    from soma import aims  # noqa: PLC0415

    stats = {}
    for path in sorted(_g.glob(join(mask_dir, "*/*.nii.gz"))):
        rel = os.path.relpath(path, mask_dir)
        arr = np.asarray(aims.read(path))
        stats[rel] = (int(arr.max()), int(arr.sum()), int(np.count_nonzero(arr)))
    return stats


# --------------------------------------------------------------------------- #
# Unified script class
# --------------------------------------------------------------------------- #

class Compare(ScriptBuilder):
    """Unified comparison tool for sulcal masks, cortical tiles, and databases."""

    def __init__(self):
        super().__init__(
            script_name="compare",
            description="Compare sulcal masks, cortical_tiles outputs, or graph databases.",
        )
        subparsers = self.parser.add_subparsers(
            dest="mode", required=True, metavar="MODE",
            description="Choose a comparison mode.")

        # ── Shared parent for mask-comparison arguments ────────────────────
        mask_parent = argparse.ArgumentParser(add_help=False)
        mask_parent.add_argument(
            "--set_a", required=True,
            help="Path to the first set directory.")
        mask_parent.add_argument(
            "--set_b", required=True,
            help="Path to the second set directory.")
        mask_parent.add_argument(
            "--output", default="comparison_report.json",
            help="Output JSON file path. Default: comparison_report.json.")
        mask_parent.add_argument(
            "--metric", choices=["wasserstein", "diff", "both"], default="diff",
            help="Comparison metric. Default: diff.")
        mask_parent.add_argument(
            "--bucket_step", type=float, default=1.0,
            help="Bucket width for the summary table (in voxels). Default: 1.")
        mask_parent.add_argument(
            "--xor_dir", default=None,
            help="Optional output directory for per-mask XOR NIfTI images "
                 "(1 = voxel differs, 0 = same). Files are written with the "
                 "same relative path as the input masks.")
        mask_parent.add_argument(
            "--visualisation", action="store_true", default=False,
            help="Open an interactive Anatomist session showing all changed "
                 "mask triplets (set_a=grey, set_b=violet, XOR=white) fused "
                 "in Axial/Sagittal/Coronal views. Requires Anatomist.")

        # ── masks subcommand ───────────────────────────────────────────────
        subparsers.add_parser(
            "masks",
            parents=[mask_parent],
            help="Compare two sets of sulcal NIfTI masks (side/sulcus.nii.gz).",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # ── cortical_tiles subcommand ──────────────────────────────────────
        tiles_p = subparsers.add_parser(
            "cortical_tiles",
            parents=[mask_parent],
            help="Compare two cortical_tiles outputs (crops/2mm directory).",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        tiles_p.add_argument(
            "--pattern", default="*mask_skeleton.nii.gz",
            help="Glob pattern for mask files inside each region's mask/ folder.")

        # ── databases subcommand ───────────────────────────────────────────
        db_p = subparsers.add_parser(
            "databases",
            help="Compare sulcal labeling between two graph annotation campaigns.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        db_p.add_argument(
            "--labeled_subjects_dir", required=True,
            help="Root directory containing subject subdirectories.")
        db_p.add_argument(
            "--path_to_graph_a", required=True,
            help="Relative sub-path for campaign A "
                 "(e.g. t1mri/t1/default_analysis/folds/3.3/base2018_manual).")
        db_p.add_argument(
            "--path_to_graph_b", required=True,
            help="Relative sub-path for campaign B.")
        db_p.add_argument("--label_a", default="A", help="Name for campaign A.")
        db_p.add_argument("--label_b", default="B", help="Name for campaign B.")
        db_p.add_argument(
            "--side", default="both",
            help="Hemisphere side: L, R, or both.")
        db_p.add_argument(
            "--masks_a", default=None,
            help="Mask directory for campaign A. Optional.")
        db_p.add_argument(
            "--masks_b", default=None,
            help="Mask directory for campaign B. Optional.")
        db_p.add_argument(
            "--output", default="db_comparison.csv",
            help="Output CSV file path.")
        db_p.add_argument(
            "--njobs", type=int, default=None,
            help="Parallel workers. Default: cpu_count - 2 (max 22).")

    # ---------------------------------------------------------------------- #
    # Dispatch
    # ---------------------------------------------------------------------- #

    def run(self) -> int:
        if self.args.mode == "masks":
            return self._run_masks()
        if self.args.mode == "cortical_tiles":
            return self._run_cortical_tiles()
        if self.args.mode == "databases":
            return self._run_databases()
        print(f"ERROR: unknown mode '{self.args.mode}'")
        return 1

    # ---------------------------------------------------------------------- #
    # Mask comparison modes
    # ---------------------------------------------------------------------- #

    def _find_masks(self, directory: Path, glob_pattern: str) -> dict:
        return {
            str(p.relative_to(directory)): str(p.resolve())
            for p in sorted(directory.glob(glob_pattern))
        }

    def _run_masks(self) -> int:
        return self._compare_nifti_masks("*/*.nii.gz")

    def _run_cortical_tiles(self) -> int:
        return self._compare_nifti_masks(f"*/mask/{self.args.pattern}")

    def _compare_nifti_masks(self, glob_pattern: str) -> int:
        dir_a = Path(self.args.set_a)
        dir_b = Path(self.args.set_b)

        if not self.validate_paths([str(dir_a), str(dir_b)]):
            return 1

        masks_a = self._find_masks(dir_a, glob_pattern)
        masks_b = self._find_masks(dir_b, glob_pattern)

        common = sorted(set(masks_a) & set(masks_b))
        only_a = sorted(set(masks_a) - set(masks_b))
        only_b = sorted(set(masks_b) - set(masks_a))

        print(f"Masks in set_a:      {len(masks_a)}")
        print(f"Masks in set_b:      {len(masks_b)}")
        print(f"Common (compared):   {len(common)}")
        print(f"Only in set_a:       {len(only_a)}")
        print(f"Only in set_b:       {len(only_b)}")

        use_wass = self.args.metric in ("wasserstein", "both")
        use_diff = self.args.metric in ("diff", "both")
        step = self.args.bucket_step

        wass_buckets: dict[str, list] = {}
        diff_buckets: dict[str, list] = {}
        distances: dict[str, float] = {}
        diffs: dict[str, dict] = {}

        for name in common:
            arr_a = load_mask_vol(masks_a[name])
            arr_b = load_mask_vol(masks_b[name])

            if arr_a.shape != arr_b.shape:
                print(f"WARNING: shape mismatch for {name}: "
                      f"{arr_a.shape} vs {arr_b.shape}, skipping.")
                continue

            if use_wass:
                dist = wasserstein_distance(arr_a, arr_b)
                distances[name] = round(dist, 3)
                wass_buckets.setdefault(bucket_label(dist, step), []).append(name)

            if use_diff:
                d = voxel_diff(arr_a, arr_b)
                diffs[name] = d
                diff_buckets.setdefault(bucket_label(d["changed"], step), []).append(name)

            if self.args.xor_dir:
                out_xor = str(Path(self.args.xor_dir) / name)
                save_xor_vol(masks_a[name], arr_a, arr_b, out_xor)

        report = {
            "mode": self.args.mode,
            "set_a": str(dir_a),
            "set_b": str(dir_b),
            "metric": self.args.metric,
            "summary": {
                "total_common": len(common),
                "only_in_set_a": only_a,
                "only_in_set_b": only_b,
            },
        }
        if self.args.mode == "cortical_tiles":
            report["mask_pattern"] = self.args.pattern

        if use_wass:
            report["wasserstein_by_bucket"] = sort_buckets(wass_buckets)
            report["wasserstein_per_mask"] = dict(
                sorted(distances.items(), key=lambda kv: -kv[1]))

        if use_diff:
            report["diff_by_bucket"] = sort_buckets(diff_buckets)
            report["diffs_per_mask"] = dict(
                sorted(diffs.items(), key=lambda kv: -kv[1]["changed"]))

        if self.args.xor_dir:
            report["xor_dir"] = str(self.args.xor_dir)

        out_path = Path(self.args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to: {out_path}")
        if self.args.xor_dir:
            print(f"XOR images written to: {self.args.xor_dir}")

        if use_wass and wass_buckets:
            print(f"\nWasserstein buckets (step={step}):")
            for lbl, names in sort_buckets(wass_buckets).items():
                print(f"  {lbl:15s}  {len(names):3d} mask(s)")

        if use_diff and diff_buckets:
            print(f"\nDiff buckets (step={step}):")
            for lbl, names in sort_buckets(diff_buckets).items():
                print(f"  {lbl:15s}  {len(names):3d} mask(s)")
            unchanged = sum(1 for d in diffs.values() if d["changed"] == 0)
            print(f"Unchanged masks: {unchanged}/{len(diffs)}")

        if self.args.visualisation:
            # Use diff counts if available, otherwise synthesise from wasserstein results
            vis_diffs = diffs if use_diff else {
                n: {"changed": 1, "added": 0, "removed": 0}
                for n in distances if distances[n] > 0
            }
            visualise_mask_diffs(vis_diffs, masks_a, masks_b)

        return 0

    # ---------------------------------------------------------------------- #
    # Database comparison mode
    # ---------------------------------------------------------------------- #

    def _load_database(self, path_to_graph, sides, subjects_dir,
                       njobs, brainvisa_dir) -> dict:
        from deep_folding.brainvisa.utils.subjects import get_all_subjects_as_dictionary
        from joblib import Parallel, delayed

        all_data: dict = {}
        for side in sides:
            pattern = '%(subject)s/' + path_to_graph + '/%(side)s%(subject)s*.arg'
            subjects = get_all_subjects_as_dictionary([subjects_dir], [pattern], side)
            print(f"    [{side}] {len(subjects)} subjects found, "
                  f"loading with {njobs} worker(s)…")

            results = Parallel(n_jobs=njobs, prefer='processes')(
                delayed(_get_subject_voxel_counts)(sub, brainvisa_dir)
                for sub in subjects
            )
            n_ok = 0
            for sub_name, counts in results:
                if counts is None:
                    print(f"    [{side}] WARNING: no graph for {sub_name}")
                    continue
                all_data.setdefault(sub_name, {}).update(counts)
                n_ok += 1
            print(f"    [{side}] Done ({n_ok}/{len(subjects)}).")
        return all_data

    def _run_databases(self) -> int:
        from joblib import cpu_count

        brainvisa_dir = abspath(join(
            dirname(__file__), '..', 'external', 'cortical_tiles',
            'deep_folding', 'brainvisa'
        ))
        if brainvisa_dir not in sys.path:
            sys.path.insert(0, brainvisa_dir)

        sides = ["L", "R"] if self.args.side == "both" else [self.args.side]
        njobs = self.args.njobs or max(1, min(22, cpu_count() - 2))
        la, lb = self.args.label_a, self.args.label_b

        print(f"\nLoading campaign A ({la}): {self.args.path_to_graph_a}")
        data_a = self._load_database(
            self.args.path_to_graph_a, sides,
            self.args.labeled_subjects_dir, njobs, brainvisa_dir)

        print(f"\nLoading campaign B ({lb}): {self.args.path_to_graph_b}")
        data_b = self._load_database(
            self.args.path_to_graph_b, sides,
            self.args.labeled_subjects_dir, njobs, brainvisa_dir)

        subs_a, subs_b = set(data_a), set(data_b)
        print(f"\nSubjects in {la} only:  {len(subs_a - subs_b)}")
        print(f"Subjects in {lb} only:  {len(subs_b - subs_a)}")
        print(f"Subjects in both:        {len(subs_a & subs_b)}")

        all_sulci = sorted(
            {s for d in data_a.values() for s in d} |
            {s for d in data_b.values() for s in d}
        )

        rows = []
        for sulcus in all_sulci:
            counts_a = [data_a[s][sulcus] for s in sorted(subs_a) if sulcus in data_a[s]]
            counts_b = [data_b[s][sulcus] for s in sorted(subs_b) if sulcus in data_b[s]]
            n_a, n_b = len(counts_a), len(counts_b)
            pct_a = 100.0 * n_a / len(subs_a) if subs_a else 0
            pct_b = 100.0 * n_b / len(subs_b) if subs_b else 0
            vpsa = np.mean(counts_a) if counts_a else 0.0
            vpsb = np.mean(counts_b) if counts_b else 0.0
            rows.append({
                "sulcus": sulcus,
                f"N_{la}": n_a, f"N_{lb}": n_b,
                f"pct_{la}": round(pct_a, 1), f"pct_{lb}": round(pct_b, 1),
                f"vox_per_subject_{la}": round(vpsa, 1),
                f"vox_per_subject_{lb}": round(vpsb, 1),
                "vox_ratio_B_over_A": round(vpsb / vpsa, 3) if vpsa > 0 else None,
            })

        # Optional mask stats
        mask_a_stats, mask_b_stats = {}, {}
        if self.args.masks_a and os.path.isdir(self.args.masks_a):
            print(f"\nLoading mask stats from {la}: {self.args.masks_a}")
            mask_a_stats = _mask_stats(abspath(self.args.masks_a), brainvisa_dir)
        if self.args.masks_b and os.path.isdir(self.args.masks_b):
            print(f"Loading mask stats from {lb}: {self.args.masks_b}")
            mask_b_stats = _mask_stats(abspath(self.args.masks_b), brainvisa_dir)

        if mask_a_stats or mask_b_stats:
            all_mask_keys = sorted(set(mask_a_stats) | set(mask_b_stats))
            mask_lookup: dict = {}
            for key in all_mask_keys:
                sulcus_name = os.path.basename(key).replace(".nii.gz", "")
                sa, sb = mask_a_stats.get(key), mask_b_stats.get(key)
                n_sa = len(subs_a) or 1
                n_sb = len(subs_b) or 1
                mask_lookup.setdefault(sulcus_name, {}).update({
                    f"mask_max_{la}": sa[0] if sa else None,
                    f"mask_max_{lb}": sb[0] if sb else None,
                    f"mask_sum_per_sub_{la}": round(sa[1] / n_sa, 1) if sa else None,
                    f"mask_sum_per_sub_{lb}": round(sb[1] / n_sb, 1) if sb else None,
                })
            for row in rows:
                row.update(mask_lookup.get(row["sulcus"], {}))

        output_path = abspath(self.args.output)
        os.makedirs(dirname(output_path) or ".", exist_ok=True)
        if rows:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        print(f"\nSulci with biggest voxel-density difference ({lb}/{la}):\n")
        print(f"  {'Sulcus':<45}  {f'N({la})':>7}  {f'N({lb})':>7}  "
              f"{f'vox/sub({la})':>12}  {f'vox/sub({lb})':>12}  {'ratio':>6}")
        print("  " + "-" * 100)
        sortable = [r for r in rows if r.get("vox_ratio_B_over_A") is not None]
        for row in sorted(sortable,
                          key=lambda r: abs(r["vox_ratio_B_over_A"] - 1.0),
                          reverse=True)[:30]:
            print(f"  {row['sulcus']:<45}  "
                  f"{row[f'N_{la}']:>7}  {row[f'N_{lb}']:>7}  "
                  f"{row[f'vox_per_subject_{la}']:>12.1f}  "
                  f"{row[f'vox_per_subject_{lb}']:>12.1f}  "
                  f"{row['vox_ratio_B_over_A']:>6.3f}")

        print(f"\nCSV written to: {output_path}")
        return 0


def main():
    script = Compare()
    return script.main()


if __name__ == "__main__":
    sys.exit(main())
