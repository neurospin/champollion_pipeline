#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compare sulcal labeling between two graph annotation campaigns
(e.g. base2018_manual vs base2018b_manual) and optionally two mask sets.

For each subject found in either campaign, opens the corresponding graph file
and reports the number of voxels labeled per sulcus.  Outputs a CSV summary
with per-sulcus coverage (% subjects) and voxel density (voxels/subject), and
optionally extends the comparison with mask-level statistics.

Usage
-----
    pixi run compare-databases \\
      --labeled_subjects_dir /neurospin/.../manually_labeled/pclean/all \\
      --path_to_graph_a t1mri/t1/default_analysis/folds/3.3/base2018_manual \\
      --path_to_graph_b t1mri/t1/default_analysis/folds/3.3/base2018b_manual \\
      --label_a base2018 --label_b base2018b \\
      --masks_a external/cortical_tiles/data/mask/canonical_25/2mm \\
      --masks_b /path/to/sulci_masks_2018/2.0 \\
      --output /tmp/db_comparison.csv
"""

import csv
import os
import sys
from os.path import abspath, dirname, join

import numpy as np

from champollion_utils.script_builder import ScriptBuilder


# --------------------------------------------------------------------------- #
# Parallel workers (must be module-level for joblib pickling)
# --------------------------------------------------------------------------- #

def _get_subject_voxel_counts(sub, brainvisa_dir):
    """Worker: load one graph and return (subject_name, {sulcus: voxel_count}).

    Counts voxels across aims_ss, aims_bottom, aims_other buckets (native space).
    Returns (sub_name, None) if no graph file found.
    """
    import glob as _glob
    import sys as _sys

    if brainvisa_dir not in _sys.path:
        _sys.path.insert(0, brainvisa_dir)
    from soma import aims  # noqa: PLC0415

    matches = _glob.glob(_join(sub['dir'], sub['graph_file']))
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


def _join(*args):
    from os.path import join as _j
    return _j(*args)


# --------------------------------------------------------------------------- #
# Mask statistics helper
# --------------------------------------------------------------------------- #

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
        nz = int(np.count_nonzero(arr))
        stats[rel] = (int(arr.max()), int(arr.sum()), nz)
    return stats


# --------------------------------------------------------------------------- #
# Main script class
# --------------------------------------------------------------------------- #

class CompareDatabases(ScriptBuilder):
    """Compare sulcal labeling between two graph annotation campaigns."""

    def __init__(self):
        super().__init__(
            script_name="compare_databases",
            description=(
                "Compare sulcal labeling coverage and voxel density between "
                "two graph annotation campaigns, with optional mask comparison."
            ),
        )
        (self
         .add_required_argument(
             "--labeled_subjects_dir",
             "Root directory containing subject subdirectories.")
         .add_required_argument(
             "--path_to_graph_a",
             "Relative sub-path for campaign A "
             "(e.g. t1mri/t1/default_analysis/folds/3.3/base2018_manual).")
         .add_required_argument(
             "--path_to_graph_b",
             "Relative sub-path for campaign B "
             "(e.g. t1mri/t1/default_analysis/folds/3.3/base2018b_manual).")
         .add_optional_argument("--label_a", "Name for campaign A.", default="A")
         .add_optional_argument("--label_b", "Name for campaign B.", default="B")
         .add_optional_argument(
             "--side",
             "Hemisphere side: L, R, or both.",
             default="both")
         .add_optional_argument(
             "--masks_a",
             "Mask directory for campaign A (e.g. canonical_25/2mm). "
             "Optional — skipped if not provided.",
             default=None)
         .add_optional_argument(
             "--masks_b",
             "Mask directory for campaign B. Optional.",
             default=None)
         .add_optional_argument(
             "--output",
             "Output CSV file path.",
             default="./db_comparison.csv")
         .add_optional_argument(
             "--njobs",
             "Parallel workers. Default: cpu_count - 2 (max 22).",
             default=None, type_=int))

    # ---------------------------------------------------------------------- #

    def _load_database(self, path_to_graph, sides, subjects_dir,
                       njobs, brainvisa_dir) -> dict:
        """Return {subject: {sulcus: voxel_count}} for all subjects found."""
        from joblib import Parallel, delayed
        from deep_folding.brainvisa.utils.subjects import get_all_subjects_as_dictionary

        all_data: dict = {}
        for side in sides:
            pattern = (
                '%(subject)s/' + path_to_graph + '/%(side)s%(subject)s*.arg'
            )
            subjects = get_all_subjects_as_dictionary(
                [subjects_dir], [pattern], side
            )
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

    # ---------------------------------------------------------------------- #

    def run(self):
        from joblib import cpu_count
        from deep_folding.brainvisa.utils.subjects import get_all_subjects_as_dictionary

        brainvisa_dir = abspath(join(
            dirname(__file__),
            '..', 'external', 'cortical_tiles', 'deep_folding', 'brainvisa'
        ))
        if brainvisa_dir not in sys.path:
            sys.path.insert(0, brainvisa_dir)

        sides = ["L", "R"] if self.args.side == "both" else [self.args.side]
        njobs = self.args.njobs or max(1, min(22, cpu_count() - 2))
        la, lb = self.args.label_a, self.args.label_b

        # ── Load both databases ────────────────────────────────────────────
        print(f"\nLoading campaign A ({la}): {self.args.path_to_graph_a}")
        data_a = self._load_database(
            self.args.path_to_graph_a, sides,
            self.args.labeled_subjects_dir, njobs, brainvisa_dir)

        print(f"\nLoading campaign B ({lb}): {self.args.path_to_graph_b}")
        data_b = self._load_database(
            self.args.path_to_graph_b, sides,
            self.args.labeled_subjects_dir, njobs, brainvisa_dir)

        # ── Subject sets ───────────────────────────────────────────────────
        subs_a = set(data_a)
        subs_b = set(data_b)
        subs_both = subs_a & subs_b
        print(f"\nSubjects in {la} only:  {len(subs_a - subs_b)}")
        print(f"Subjects in {lb} only:  {len(subs_b - subs_a)}")
        print(f"Subjects in both:        {len(subs_both)}")

        # ── Per-sulcus stats ───────────────────────────────────────────────
        all_sulci = sorted(
            {s for d in data_a.values() for s in d} |
            {s for d in data_b.values() for s in d}
        )

        rows = []
        for sulcus in all_sulci:
            counts_a = [data_a[s][sulcus] for s in sorted(subs_a) if sulcus in data_a[s]]
            counts_b = [data_b[s][sulcus] for s in sorted(subs_b) if sulcus in data_b[s]]

            n_a = len(counts_a)
            n_b = len(counts_b)
            pct_a = 100.0 * n_a / len(subs_a) if subs_a else 0
            pct_b = 100.0 * n_b / len(subs_b) if subs_b else 0
            vpsa = np.mean(counts_a) if counts_a else 0.0
            vpsb = np.mean(counts_b) if counts_b else 0.0

            rows.append({
                "sulcus": sulcus,
                f"N_{la}": n_a,
                f"N_{lb}": n_b,
                f"pct_{la}": round(pct_a, 1),
                f"pct_{lb}": round(pct_b, 1),
                f"vox_per_subject_{la}": round(vpsa, 1),
                f"vox_per_subject_{lb}": round(vpsb, 1),
                "vox_ratio_B_over_A": round(vpsb / vpsa, 3) if vpsa > 0 else None,
            })

        # ── Optional mask stats ────────────────────────────────────────────
        mask_a_stats = {}
        mask_b_stats = {}
        if self.args.masks_a and os.path.isdir(self.args.masks_a):
            print(f"\nLoading mask stats from {la}: {self.args.masks_a}")
            mask_a_stats = _mask_stats(abspath(self.args.masks_a), brainvisa_dir)
        if self.args.masks_b and os.path.isdir(self.args.masks_b):
            print(f"Loading mask stats from {lb}: {self.args.masks_b}")
            mask_b_stats = _mask_stats(abspath(self.args.masks_b), brainvisa_dir)

        if mask_a_stats or mask_b_stats:
            all_mask_keys = sorted(set(mask_a_stats) | set(mask_b_stats))
            mask_lookup = {}
            for key in all_mask_keys:
                # key is like "L/OCCIPITAL_left.nii.gz"
                sulcus_name = os.path.basename(key).replace(".nii.gz", "")
                mask_lookup.setdefault(sulcus_name, {})[
                    "mask_max_" + la] = mask_a_stats.get(key, (None,))[0]
                mask_lookup[sulcus_name][
                    "mask_max_" + lb] = mask_b_stats.get(key, (None,))[0]
                n_subs_a_mask = len(subs_a) or 1
                n_subs_b_mask = len(subs_b) or 1
                sa = mask_a_stats.get(key)
                sb = mask_b_stats.get(key)
                mask_lookup[sulcus_name]["mask_sum_per_sub_" + la] = (
                    round(sa[1] / n_subs_a_mask, 1) if sa else None)
                mask_lookup[sulcus_name]["mask_sum_per_sub_" + lb] = (
                    round(sb[1] / n_subs_b_mask, 1) if sb else None)

            for row in rows:
                extra = mask_lookup.get(row["sulcus"], {})
                row.update(extra)

        # ── Write CSV ──────────────────────────────────────────────────────
        output_path = abspath(self.args.output)
        out_dir = dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if rows:
            fieldnames = list(rows[0].keys())
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        # ── Print summary (sulci with biggest vox_ratio difference) ────────
        print(f"\nSulci with biggest voxel-density difference ({lb}/{la}):\n")
        print(f"  {'Sulcus':<45}  {f'N({la})':>7}  {f'N({lb})':>7}  "
              f"{f'vox/sub({la})':>12}  {f'vox/sub({lb})':>12}  {'ratio':>6}")
        print("  " + "-" * 100)
        sortable = [r for r in rows if r.get("vox_ratio_B_over_A") is not None]
        for row in sorted(sortable, key=lambda r: abs(r["vox_ratio_B_over_A"] - 1.0), reverse=True)[:30]:
            print(f"  {row['sulcus']:<45}  "
                  f"{row[f'N_{la}']:>7}  "
                  f"{row[f'N_{lb}']:>7}  "
                  f"{row[f'vox_per_subject_{la}']:>12.1f}  "
                  f"{row[f'vox_per_subject_{lb}']:>12.1f}  "
                  f"{row['vox_ratio_B_over_A']:>6.3f}")

        print(f"\nCSV written to: {output_path}")
        return 0


def main():
    script = CompareDatabases()
    return script.build().print_args().run()


if __name__ == "__main__":
    raise SystemExit(main())
