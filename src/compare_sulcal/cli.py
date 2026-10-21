"""Command-line front end of compare.py: the Compare script, run_subcommand and main."""

import argparse
import csv
import json
import math
import os
import sys
from os.path import abspath, dirname, join
from pathlib import Path

import numpy as np
from champollion_utils.script_builder import ScriptBuilder
from joblib import Parallel, cpu_count, delayed

from compare_sulcal import brainvisa, crops, databases, embeddings, masks, viewer

_SIDE_CHOICES = ("L", "R", "both")


_EMB_BASE_COLUMNS = ["region", "side", "subject", "knn_overlap"]


_AIMS_MODES = frozenset({"masks", "cortical_tiles", "databases", "crops"})


def _positive_finite_float(value: str) -> float:
    """argparse type: parse value as a finite float greater than zero."""
    try:
        number = float(value)
    except ValueError:
        number = math.nan
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive finite number, got {value!r}")
    return number


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
            dest="mode", required=True, metavar="MODE", description="Choose a comparison mode."
        )

        # ── Shared parent for mask-comparison arguments ────────────────────
        mask_parent = argparse.ArgumentParser(add_help=False)
        mask_parent.add_argument("--set_a", required=True, help="Path to the first set directory.")
        mask_parent.add_argument("--set_b", required=True, help="Path to the second set directory.")
        mask_parent.add_argument(
            "--output", default="comparison_report.json", help="Output JSON file path. Default: comparison_report.json."
        )
        mask_parent.add_argument(
            "--metric",
            choices=["wasserstein", "diff", "both"],
            default="diff",
            help="Comparison metric. Default: diff.",
        )
        mask_parent.add_argument(
            "--bucket_step",
            type=_positive_finite_float,
            default=1.0,
            help="Bucket width for the summary table, in voxels; a finite number > 0. Default: 1.",
        )
        mask_parent.add_argument(
            "--xor_dir",
            default=None,
            help="Optional output directory for per-mask XOR NIfTI images "
            "(1 = voxel differs, 0 = same). Files are written with the "
            "same relative path as the input masks.",
        )
        mask_parent.add_argument(
            "--visualisation",
            action="store_true",
            default=False,
            help="Open an interactive Anatomist viewer on the 5 most changed masks, ranked by "
            "changed voxels (--metric diff/both) or by Wasserstein distance (--metric "
            "wasserstein). Each mask is shown as a fusion of set_a (grey), set_b (violet) "
            "and their XOR (red) in linked Axial, Sagittal and Coronal windows. XOR files "
            "come from --xor_dir when given, otherwise from a temporary directory removed "
            "afterwards. Requires Anatomist and a usable display (DISPLAY or "
            "WAYLAND_DISPLAY set, QT_QPA_PLATFORM not offscreen); skipped otherwise. Exits 1 "
            "if the viewer fails.",
        )

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
            "--pattern",
            default="*mask_skeleton.nii.gz",
            help="Glob pattern for mask files inside each region's mask/ folder.",
        )

        # ── crops subcommand ──────────────────────────────────────────────
        crops_p = subparsers.add_parser(
            "crops",
            help="Compare two cortical_tiles crops/2mm directories (.npy crop sets).",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        crops_p.add_argument("--set_a", required=True, help="Path to the first crops/2mm directory.")
        crops_p.add_argument("--set_b", required=True, help="Path to the second crops/2mm directory.")
        crops_p.add_argument(
            "--output", default="crops_comparison", help="Output directory. Default: crops_comparison."
        )
        crops_p.add_argument(
            "--input_type",
            choices=crops._CROPS_INPUT_TYPES,
            default="skeleton",
            help="Reads mask/{side}{input_type}.npy and _subject.csv. Default: skeleton.",
        )
        crops_p.add_argument(
            "--xor_dir",
            default=None,
            help="Optional directory for per-subject XOR files of the top_lost subjects.",
        )
        crops_p.add_argument(
            "--regions",
            nargs="+",
            default=None,
            help="Region names to compare. Default: intersection of region dirs in both sets.",
        )
        crops_p.add_argument(
            "--side",
            choices=_SIDE_CHOICES,
            default="both",
            help="Hemisphere side(s) to compare. Default: both.",
        )
        crops_p.add_argument("--top_k", type=int, default=5, help="Length of top_lost list. Default: 5.")
        crops_p.add_argument("--njobs", type=int, default=1, help="Joblib workers over region/side pairs. Default: 1.")

        # ── databases subcommand ───────────────────────────────────────────
        db_p = subparsers.add_parser(
            "databases",
            help="Compare sulcal labeling between two graph annotation campaigns.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        db_p.add_argument(
            "--labeled_subjects_dir", required=True, help="Root directory containing subject subdirectories."
        )
        db_p.add_argument(
            "--path_to_graph_a",
            required=True,
            help="Relative sub-path for campaign A (e.g. t1mri/t1/default_analysis/folds/3.3/base2018_manual).",
        )
        db_p.add_argument("--path_to_graph_b", required=True, help="Relative sub-path for campaign B.")
        db_p.add_argument("--label_a", default="A", help="Name for campaign A.")
        db_p.add_argument("--label_b", default="B", help="Name for campaign B.")
        db_p.add_argument(
            "--side", choices=_SIDE_CHOICES, default="both", help="Hemisphere side(s) to compare: L, R or both."
        )
        db_p.add_argument("--masks_a", default=None, help="Mask directory for campaign A. Optional.")
        db_p.add_argument("--masks_b", default=None, help="Mask directory for campaign B. Optional.")
        db_p.add_argument("--output", default="db_comparison.csv", help="Output CSV file path.")
        db_p.add_argument("--njobs", type=int, default=None, help="Parallel workers. Default: cpu_count - 2 (max 22).")

        # ── embeddings subcommand ──────────────────────────────────────────
        emb_p = subparsers.add_parser(
            "embeddings",
            help="Compare two champollion_V1 embedding directories per subject, region and side.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        emb_p.add_argument("--set_a", required=True, help="Path to the first embeddings directory.")
        emb_p.add_argument("--set_b", required=True, help="Path to the second embeddings directory.")
        emb_p.add_argument(
            "--output", default="embeddings_comparison", help="Output directory. Default: embeddings_comparison."
        )
        emb_p.add_argument(
            "--regions", nargs="+", default=None, help="Region names to compare. Default: regions in both sets."
        )
        emb_p.add_argument(
            "--side", choices=_SIDE_CHOICES, default="both", help="Hemisphere side(s) to compare. Default: both."
        )
        emb_p.add_argument("--k", type=int, default=15, help="k for kNN overlap. Default: 15.")
        emb_p.add_argument("--njobs", type=int, default=1, help="Joblib workers over region/side pairs. Default: 1.")
        emb_p.add_argument("--umap_a", default=None, help="UMAP reference directory for set A (.pkl files).")
        emb_p.add_argument("--umap_b", default=None, help="UMAP reference directory for set B (.pkl files).")
        emb_p.add_argument("--covariate", default=None, help="CSV file with per-subject covariate values.")
        emb_p.add_argument("--covariate_column", default=None, help="Column name in --covariate CSV.")

    # ---------------------------------------------------------------------- #
    # Dispatch
    # ---------------------------------------------------------------------- #

    def run(self) -> int:
        if brainvisa.aims is None and self.args.mode in _AIMS_MODES:
            print(f"{brainvisa._AIMS_UNAVAILABLE_MESSAGE} ({brainvisa._AIMS_IMPORT_ERROR})", file=sys.stderr)
            return 1
        if self.args.mode == "masks":
            return self._run_masks()
        if self.args.mode == "cortical_tiles":
            return self._run_cortical_tiles()
        if self.args.mode == "databases":
            return self._run_databases()
        if self.args.mode == "crops":
            return self._run_crops()
        if self.args.mode == "embeddings":
            return self._run_embeddings()
        print(f"ERROR: unknown mode '{self.args.mode}'")
        return 1

    # ---------------------------------------------------------------------- #
    # Embeddings comparison mode
    # ---------------------------------------------------------------------- #

    def _run_embeddings(self) -> int:
        """Run the embeddings comparison subcommand."""
        set_a = Path(self.args.set_a)
        set_b = Path(self.args.set_b)
        for flag, path in (("--set_a", set_a), ("--set_b", set_b)):
            if not path.is_dir():
                print(f"ERROR: {flag} is not a directory: {path}", file=sys.stderr)
                return 1
        if (self.args.umap_a is None) != (self.args.umap_b is None):
            print("ERROR: --umap_a and --umap_b must be given together.", file=sys.stderr)
            return 1
        if (self.args.covariate is None) != (self.args.covariate_column is None):
            print("ERROR: --covariate and --covariate_column must be given together.", file=sys.stderr)
            return 1
        if self.args.k < 1:
            print("ERROR: --k must be a positive integer.", file=sys.stderr)
            return 1

        covariate_map = None
        if self.args.covariate is not None:
            covariate_map = embeddings._load_covariate_csv(self.args.covariate, self.args.covariate_column)
            if isinstance(covariate_map, str):
                print(f"ERROR: {covariate_map}", file=sys.stderr)
                return 1

        out_dir = Path(self.args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        sides = ["L", "R"] if self.args.side == "both" else [self.args.side]
        pairs = embeddings._list_embedding_pairs(set_a, set_b, self.args.regions, sides)
        options = {
            "k": self.args.k,
            "umap_a": self.args.umap_a,
            "umap_b": self.args.umap_b,
            "covariate": covariate_map,
            "covariate_column": self.args.covariate_column,
            "output": str(out_dir),
            "explicit": self.args.regions is not None,
        }
        results = Parallel(n_jobs=self.args.njobs)(
            delayed(embeddings._compare_embeddings_region_side)(set_a, set_b, region, side, options)
            for region, side in pairs
        )

        columns = list(_EMB_BASE_COLUMNS)
        if self.args.umap_a is not None:
            columns += embeddings._EMB_UMAP_COLUMNS
        if covariate_map is not None:
            columns.append(embeddings._EMB_COVARIATE_COLUMN)

        all_rows: list[dict] = []
        regions_summary: dict = {}
        skipped_list: list[dict] = []
        umap_skipped: list[dict] = []
        for result in results:
            key = result["key"]
            if result["skipped"] is not None:
                skipped_list.append(result["skipped"])
                print(f"SKIP  {key}: {result['skipped']['reason']}")
            umap_skipped.extend(result["umap_skipped"])
            for entry in result["umap_skipped"]:
                print(f"UMAP SKIP  {key}: {entry['reason']}")
            if result["summary"] is not None:
                regions_summary[key] = result["summary"]
                s = result["summary"]
                print(f"  {key}: n={s['n_subjects']} cka={s['cka']:.4f} knn_overlap_mean={s['knn_overlap_mean']:.4f}")
            all_rows.extend(result["rows"])

        with open(out_dir / "per_subject.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(all_rows)

        summary = {
            "set_a": str(set_a.resolve()),
            "set_b": str(set_b.resolve()),
            "umap_a": self.args.umap_a,
            "umap_b": self.args.umap_b,
            "covariate": self.args.covariate,
            "covariate_column": self.args.covariate_column,
            "k": self.args.k,
            "regions": regions_summary,
            "skipped": skipped_list,
            "umap_skipped": umap_skipped,
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(embeddings._normalize_json_value(summary), f, indent=2, allow_nan=False)
        print(f"Wrote {out_dir / 'per_subject.csv'} ({len(all_rows)} rows) and {out_dir / 'summary.json'}")
        return 0

    # ---------------------------------------------------------------------- #
    # Crops comparison mode
    # ---------------------------------------------------------------------- #

    def _run_crops(self) -> int:
        """Run the crops comparison subcommand."""
        set_a = Path(self.args.set_a)
        set_b = Path(self.args.set_b)
        if not set_a.is_dir():
            print(f"ERROR: --set_a is not a directory: {set_a}", file=sys.stderr)
            return 1
        if not set_b.is_dir():
            print(f"ERROR: --set_b is not a directory: {set_b}", file=sys.stderr)
            return 1

        out_dir = Path(self.args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        sides = ["L", "R"] if self.args.side == "both" else [self.args.side]
        regions = crops._list_crop_regions(set_a, set_b, self.args.regions)
        explicit = self.args.regions is not None
        xor_dir = Path(self.args.xor_dir) if self.args.xor_dir else None

        pairs = [(region, side) for region in regions for side in sides]

        results = Parallel(n_jobs=self.args.njobs)(
            delayed(crops._compare_region_side)(
                set_a, set_b, region, side, self.args.input_type, self.args.top_k, explicit, xor_dir=xor_dir
            )
            for region, side in pairs
        )

        all_rows: list[dict] = []
        regions_summary: dict = {}
        skipped_list: list[dict] = []

        for result in results:
            key = result["key"]
            region_name = result["region"]
            side_name = result["side"]
            if result["skipped"] is not None:
                skipped_list.append(result["skipped"])
                print(f"SKIP  {key}: {result['skipped']['reason']}")
            if result["summary"] is not None:
                regions_summary[key] = result["summary"]
                s = result["summary"]
                print(
                    f"  {key}: n={s['n_subjects_compared']} "
                    f"changed={s['subjects_changed']} "
                    f"pct_lost_mean={s['pct_lost_mean']:.4f} "
                    f"dice_min={s['dice_min']:.4f}"
                )
            for row in result["rows"]:
                all_rows.append(
                    {
                        "region": region_name,
                        "side": side_name,
                        "subject": row["subject"],
                        "n_a": row["n_a"],
                        "n_b": row["n_b"],
                        "kept": row["kept"],
                        "lost": row["lost"],
                        "gained": row["gained"],
                        "pct_lost": round(row["pct_lost"], 6),
                        "dice": round(row["dice"], 6),
                        "shift": "inf" if math.isinf(row["shift"]) else round(row["shift"], 6),
                    }
                )

        csv_path = out_dir / "per_subject.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=crops._CROPS_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(all_rows)

        summary = {
            "set_a": str(set_a.resolve()),
            "set_b": str(set_b.resolve()),
            "input_type": self.args.input_type,
            "regions": regions_summary,
            "skipped": skipped_list,
        }
        json_path = out_dir / "summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, allow_nan=False)

        print(f"Wrote {csv_path} and {json_path}")
        if xor_dir is not None:
            print(f"XOR files written under: {xor_dir}")
        return 0

    # ---------------------------------------------------------------------- #
    # Mask comparison modes
    # ---------------------------------------------------------------------- #

    def _run_masks(self) -> int:
        return self._compare_nifti_masks("*/*.nii.gz")

    def _run_cortical_tiles(self) -> int:
        return self._compare_nifti_masks(f"*/mask/{self.args.pattern}")

    def _compare_nifti_masks(self, glob_pattern: str) -> int:
        dir_a = Path(self.args.set_a)
        dir_b = Path(self.args.set_b)

        if not self.validate_paths([str(dir_a), str(dir_b)]):
            return 1

        masks_a = masks.find_masks(dir_a, glob_pattern)
        masks_b = masks.find_masks(dir_b, glob_pattern)

        common = sorted(set(masks_a) & set(masks_b))
        only_a = sorted(set(masks_a) - set(masks_b))
        only_b = sorted(set(masks_b) - set(masks_a))

        print(f"Masks in set_a:      {len(masks_a)}")
        print(f"Masks in set_b:      {len(masks_b)}")
        print(f"Common:              {len(common)}")
        print(f"Only in set_a:       {len(only_a)}")
        print(f"Only in set_b:       {len(only_b)}")

        use_wass = self.args.metric in ("wasserstein", "both")
        use_diff = self.args.metric in ("diff", "both")
        step = self.args.bucket_step

        wass_buckets, diff_buckets, distances, diffs, skipped_shape_mismatch = _score_common_masks(
            common, masks_a, masks_b, use_wass, use_diff, step, self.args.xor_dir
        )

        total_compared = len(common) - len(skipped_shape_mismatch)
        print(f"Compared:            {total_compared}")
        print(f"Skipped (shape mismatch): {len(skipped_shape_mismatch)}")

        report = {
            "mode": self.args.mode,
            "set_a": str(dir_a),
            "set_b": str(dir_b),
            "metric": self.args.metric,
            "summary": {
                "total_common": len(common),
                "total_compared": total_compared,
                "skipped_shape_mismatch": skipped_shape_mismatch,
                "only_in_set_a": only_a,
                "only_in_set_b": only_b,
            },
        }
        if self.args.mode == "cortical_tiles":
            report["mask_pattern"] = self.args.pattern

        if use_wass:
            report["wasserstein_by_bucket"] = masks.sort_buckets(wass_buckets)
            report["wasserstein_per_mask"] = {
                name: masks._normalize_distance_for_json(d)
                for name, d in sorted(distances.items(), key=lambda kv: -kv[1])
            }

        if use_diff:
            report["diff_by_bucket"] = masks.sort_buckets(diff_buckets)
            report["diffs_per_mask"] = dict(sorted(diffs.items(), key=lambda kv: -kv[1]["changed"]))

        if self.args.xor_dir:
            report["xor_dir"] = str(self.args.xor_dir)

        out_path = Path(self.args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, allow_nan=False)
        print(f"\nReport written to: {out_path}")
        if self.args.xor_dir:
            print(f"XOR images written to: {self.args.xor_dir}")

        if use_wass and wass_buckets:
            print(f"\nWasserstein buckets (step={step}):")
            _print_bucket_counts(wass_buckets)

        if use_diff and diff_buckets:
            print(f"\nDiff buckets (step={step}):")
            _print_bucket_counts(diff_buckets)
            unchanged = sum(1 for d in diffs.values() if d["changed"] == 0)
            print(f"Unchanged masks: {unchanged}/{len(diffs)}")

        if not self.args.visualisation:
            return 0
        scores = {name: d["changed"] for name, d in diffs.items()} if use_diff else distances
        return viewer.visualise_mask_diffs(scores, masks_a, masks_b, xor_dir=self.args.xor_dir)

    # ---------------------------------------------------------------------- #
    # Database comparison mode
    # ---------------------------------------------------------------------- #

    def _load_database(self, path_to_graph, sides, subjects_dir, njobs, brainvisa_dir) -> dict:
        if brainvisa.get_all_subjects_as_dictionary is None:
            raise ImportError(
                "The databases subcommand requires cortical_tiles (pixi environment with the cortical-tiles feature)"
            ) from brainvisa._CORTICAL_TILES_IMPORT_ERROR

        all_data: dict = {}
        for side in sides:
            pattern = "%(subject)s/" + path_to_graph + "/%(side)s%(subject)s*.arg"
            subjects = brainvisa.get_all_subjects_as_dictionary([subjects_dir], [pattern], side)
            print(f"    [{side}] {len(subjects)} subjects found, loading with {njobs} worker(s)…")

            results = Parallel(n_jobs=njobs, prefer="processes")(
                delayed(databases._get_subject_voxel_counts)(sub, brainvisa_dir) for sub in subjects
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
        brainvisa_dir = abspath(
            join(dirname(__file__), "..", "..", "external", "cortical_tiles", "cortical_tiles", "brainvisa")
        )
        if brainvisa_dir not in sys.path:
            sys.path.insert(0, brainvisa_dir)

        sides = ["L", "R"] if self.args.side == "both" else [self.args.side]
        njobs = self.args.njobs or max(1, min(22, cpu_count() - 2))
        la, lb = self.args.label_a, self.args.label_b

        print(f"\nLoading campaign A ({la}): {self.args.path_to_graph_a}")
        data_a = self._load_database(
            self.args.path_to_graph_a, sides, self.args.labeled_subjects_dir, njobs, brainvisa_dir
        )

        print(f"\nLoading campaign B ({lb}): {self.args.path_to_graph_b}")
        data_b = self._load_database(
            self.args.path_to_graph_b, sides, self.args.labeled_subjects_dir, njobs, brainvisa_dir
        )

        subs_a, subs_b = set(data_a), set(data_b)
        print(f"\nSubjects in {la} only:  {len(subs_a - subs_b)}")
        print(f"Subjects in {lb} only:  {len(subs_b - subs_a)}")
        print(f"Subjects in both:        {len(subs_a & subs_b)}")

        all_sulci = sorted({s for d in data_a.values() for s in d} | {s for d in data_b.values() for s in d})

        rows = []
        for sulcus in all_sulci:
            counts_a = [data_a[s][sulcus] for s in sorted(subs_a) if sulcus in data_a[s]]
            counts_b = [data_b[s][sulcus] for s in sorted(subs_b) if sulcus in data_b[s]]
            n_a, n_b = len(counts_a), len(counts_b)
            pct_a = 100.0 * n_a / len(subs_a) if subs_a else 0
            pct_b = 100.0 * n_b / len(subs_b) if subs_b else 0
            vpsa = np.mean(counts_a) if counts_a else 0.0
            vpsb = np.mean(counts_b) if counts_b else 0.0
            rows.append(
                {
                    "sulcus": sulcus,
                    f"N_{la}": n_a,
                    f"N_{lb}": n_b,
                    f"pct_{la}": round(pct_a, 1),
                    f"pct_{lb}": round(pct_b, 1),
                    f"vox_per_subject_{la}": round(vpsa, 1),
                    f"vox_per_subject_{lb}": round(vpsb, 1),
                    "vox_ratio_B_over_A": round(vpsb / vpsa, 3) if vpsa > 0 else None,
                }
            )

        # Optional mask stats
        mask_a_stats, mask_b_stats = {}, {}
        if self.args.masks_a and os.path.isdir(self.args.masks_a):
            print(f"\nLoading mask stats from {la}: {self.args.masks_a}")
            mask_a_stats = databases._mask_stats(abspath(self.args.masks_a), brainvisa_dir)
        if self.args.masks_b and os.path.isdir(self.args.masks_b):
            print(f"Loading mask stats from {lb}: {self.args.masks_b}")
            mask_b_stats = databases._mask_stats(abspath(self.args.masks_b), brainvisa_dir)

        if mask_a_stats or mask_b_stats:
            all_mask_keys = sorted(set(mask_a_stats) | set(mask_b_stats))
            mask_lookup: dict = {}
            for key in all_mask_keys:
                sulcus_name = os.path.basename(key).replace(".nii.gz", "")
                sa, sb = mask_a_stats.get(key), mask_b_stats.get(key)
                n_sa = len(subs_a) or 1
                n_sb = len(subs_b) or 1
                mask_lookup.setdefault(sulcus_name, {}).update(
                    {
                        f"mask_max_{la}": sa[0] if sa else None,
                        f"mask_max_{lb}": sb[0] if sb else None,
                        f"mask_sum_per_sub_{la}": round(sa[1] / n_sa, 1) if sa else None,
                        f"mask_sum_per_sub_{lb}": round(sb[1] / n_sb, 1) if sb else None,
                    }
                )
            for row in rows:
                row.update(mask_lookup.get(row["sulcus"], {}))

        if not rows:
            print(f"\nNo sulcus rows found for {la} vs {lb}; no CSV written.")
            return 0

        output_path = abspath(self.args.output)
        os.makedirs(dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nSulci with biggest voxel-density difference ({lb}/{la}):\n")
        print(
            f"  {'Sulcus':<45}  {f'N({la})':>7}  {f'N({lb})':>7}  "
            f"{f'vox/sub({la})':>12}  {f'vox/sub({lb})':>12}  {'ratio':>6}"
        )
        print("  " + "-" * 100)
        sortable = [r for r in rows if r.get("vox_ratio_B_over_A") is not None]
        for row in sorted(sortable, key=lambda r: abs(r["vox_ratio_B_over_A"] - 1.0), reverse=True)[:30]:
            print(
                f"  {row['sulcus']:<45}  "
                f"{row[f'N_{la}']:>7}  {row[f'N_{lb}']:>7}  "
                f"{row[f'vox_per_subject_{la}']:>12.1f}  "
                f"{row[f'vox_per_subject_{lb}']:>12.1f}  "
                f"{row['vox_ratio_B_over_A']:>6.3f}"
            )

        print(f"\nCSV written to: {output_path}")
        return 0


def run_subcommand(mode: str, options: dict[str, object]) -> int:
    """Run the Compare subcommand mode with options, as `python compare.py <mode> --<name>=<value> ...` would.

    An option whose value is None is left out (the subcommand default applies). Every other value is passed as
    the single token f"--{name}={value}" (the `=` form, so a value starting with '-' is never read as an option),
    so Compare's own parser types and choices validate it. No update check and no argument print (the calling
    script owns both).
    """
    argv = [mode] + [f"--{name}={value}" for name, value in options.items() if value is not None]
    script = Compare()
    script.parse_args(argv)
    return script.run()


def main():
    script = Compare()
    return script.main()


def _print_bucket_counts(buckets: "dict[str, list]") -> None:
    """Print one `  <label>  <n> mask(s)` line per bucket, in masks.sort_buckets order."""
    for lbl, names in masks.sort_buckets(buckets).items():
        print(f"  {lbl:15s}  {len(names):3d} mask(s)")


def _score_common_masks(
    common: "list[str]",
    masks_a: "dict[str, str]",
    masks_b: "dict[str, str]",
    use_wass: bool,
    use_diff: bool,
    step: float,
    xor_dir: "str | None",
) -> "tuple[dict[str, list], dict[str, list], dict[str, float], dict[str, dict], dict[str, dict[str, list[int]]]]":
    """Score each common mask pair; returns (wass_buckets, diff_buckets, distances, diffs, skipped_shape_mismatch)."""
    wass_buckets: dict[str, list] = {}
    diff_buckets: dict[str, list] = {}
    distances: dict[str, float] = {}
    diffs: dict[str, dict] = {}
    skipped_shape_mismatch: dict[str, dict[str, list[int]]] = {}

    for name in common:
        arr_a = masks.load_mask_vol(masks_a[name])
        arr_b = masks.load_mask_vol(masks_b[name])

        if arr_a.shape != arr_b.shape:
            print(f"WARNING: shape mismatch for {name}: {arr_a.shape} vs {arr_b.shape}, skipping.")
            skipped_shape_mismatch[name] = {
                "shape_a": [int(n) for n in arr_a.shape],
                "shape_b": [int(n) for n in arr_b.shape],
            }
            continue

        if use_wass:
            dist = masks.wasserstein_distance(arr_a, arr_b)
            is_one_side_empty = math.isinf(dist)
            distances[name] = dist if is_one_side_empty else round(dist, 3)
            bucket = masks.ONE_SIDE_EMPTY_BUCKET if is_one_side_empty else masks.bucket_label(dist, step)
            wass_buckets.setdefault(bucket, []).append(name)

        if use_diff:
            d = masks.voxel_diff(arr_a, arr_b)
            diffs[name] = d
            diff_buckets.setdefault(masks.bucket_label(d["changed"], step), []).append(name)

        if xor_dir:
            out_xor = str(Path(xor_dir) / name)
            masks.save_xor_vol(masks_a[name], arr_a, arr_b, out_xor)

    return wass_buckets, diff_buckets, distances, diffs, skipped_shape_mismatch
