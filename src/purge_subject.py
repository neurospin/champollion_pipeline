#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove all cortical_tiles derivatives for a single subject.

Handles:
  - Per-subject .nii.gz crop files in Rcrops/, Rlabels/, Rextremities/, Rdistbottom/
  - Per-subject subdirectories under skeletons/, foldlabels/, extremities/, transforms/
  - Aggregated .npy arrays (Rskeleton.npy, Rlabel.npy, ...) filtered by subject CSV
"""

import shutil
from os.path import exists, join
from pathlib import Path

import numpy as np
import pandas as pd

from champollion_utils.script_builder import ScriptBuilder


class PurgeSubject(ScriptBuilder):
    """Remove all cortical_tiles derivatives for one subject."""

    def __init__(self):
        super().__init__(
            script_name="purge_subject",
            description="Remove all cortical_tiles derivatives for a single subject.",
        )
        (self.add_argument(
            "derivatives",
            help="Path to the cortical_tiles derivatives directory "
                 "(e.g. data/mydata/derivatives/cortical_tiles-2026/).")
         .add_required_argument(
            "--subject",
            "Subject ID to remove (e.g. sub-123456).")
         .add_flag("--dry-run", "Print what would be deleted without deleting."))

    def _log(self, msg):
        prefix = "[dry-run] " if self.args.dry_run else ""
        print(f"{prefix}{msg}")

    def _remove_file(self, path):
        self._log(f"Delete file: {path}")
        if not self.args.dry_run:
            Path(path).unlink()

    def _remove_dir(self, path):
        self._log(f"Delete dir:  {path}")
        if not self.args.dry_run:
            shutil.rmtree(path)

    def _purge_per_subject_dirs(self, derivatives):
        """Remove {subject}/ subdirectories under skeletons/, foldlabels/, etc."""
        top_level_dirs = ["skeletons", "foldlabels", "extremities", "transforms", "distmaps"]
        for dirname in top_level_dirs:
            subj_dir = join(derivatives, dirname, self.args.subject)
            if exists(subj_dir):
                self._remove_dir(subj_dir)

    def _purge_per_subject_crop_files(self, crops_2mm):
        """Remove per-subject .nii.gz files inside Rcrops/, Rlabels/, etc."""
        subject = self.args.subject
        for region_dir in Path(crops_2mm).iterdir():
            if not region_dir.is_dir():
                continue
            mask_dir = region_dir / "mask"
            if not mask_dir.exists():
                continue
            for subdir in mask_dir.iterdir():
                if not subdir.is_dir():
                    continue
                for f in subdir.iterdir():
                    stem = f.name
                    for ext in (".nii.gz", ".nii", ".gz", ".minf"):
                        if stem.endswith(ext):
                            stem = stem[: -len(ext)]
                            break
                    if stem == subject or stem.startswith(f"{subject}_"):
                        self._remove_file(str(f))

    def _purge_aggregated_arrays(self, crops_2mm):
        """Remove subject row from aggregated .npy arrays and their subject CSVs."""
        subject = self.args.subject
        found_in_any = False

        for region_dir in Path(crops_2mm).iterdir():
            if not region_dir.is_dir():
                continue
            for mask_dir in [region_dir / "mask", region_dir]:
                if not mask_dir.exists():
                    continue
                for csv_path in sorted(mask_dir.glob("*_subject.csv")):
                    npy_path = csv_path.parent / (csv_path.name.replace("_subject.csv", ".npy"))
                    if not npy_path.exists():
                        continue

                    subjects_df = pd.read_csv(csv_path)
                    col = subjects_df.columns[0]
                    subjects_df[col] = subjects_df[col].astype(str)
                    mask = subjects_df[col] == str(subject)

                    if not mask.any():
                        continue

                    found_in_any = True
                    idx = subjects_df.index[mask].tolist()
                    self._log(
                        f"Filter row(s) {idx} from array: {npy_path.name} "
                        f"(region {region_dir.name})"
                    )

                    if not self.args.dry_run:
                        arr = np.load(str(npy_path), allow_pickle=True)
                        keep = ~mask.values
                        arr_filtered = arr[keep]
                        np.save(str(npy_path), arr_filtered)

                        subjects_filtered = subjects_df[keep].reset_index(drop=True)
                        subjects_filtered.to_csv(str(csv_path), index=False)

        if not found_in_any:
            print(f"Subject '{subject}' not found in any aggregated array.")

    def run(self):
        derivatives = self.args.derivatives
        if not exists(derivatives):
            raise ValueError(f"Derivatives directory not found: {derivatives}")

        crops_2mm = join(derivatives, "crops", "2mm")
        subject = self.args.subject

        print(f"Purging subject '{subject}' from: {derivatives}")

        self._purge_per_subject_dirs(derivatives)
        if exists(crops_2mm):
            self._purge_per_subject_crop_files(crops_2mm)
            self._purge_aggregated_arrays(crops_2mm)
        else:
            print("No crops/2mm directory found — skipping crop purge.")

        action = "Would remove" if self.args.dry_run else "Removed"
        print(f"\nDone. {action} all data for subject '{subject}'.")
        return 0


def main():
    script = PurgeSubject()
    return script.build().print_args().run()


if __name__ == "__main__":
    exit(main())
