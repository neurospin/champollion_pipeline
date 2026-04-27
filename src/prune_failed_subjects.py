#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove cortical_tiles outputs for subjects that failed QC.

Scans the cortical_tiles derivatives directory, reads a QC TSV/CSV file, and
deletes all files belonging to subjects with qc==0 or absent from the QC file.
This brings the output directory to the same state as if --sk_qc_path had been
passed to run_cortical_tiles.py from the start.
"""

import os
from os.path import join, exists
from pathlib import Path

import pandas as pd

from champollion_utils.script_builder import ScriptBuilder
from utils.lib import DERIVATIVES_FOLDER


class PruneFailedSubjects(ScriptBuilder):
    """Remove cortical_tiles outputs for QC-failing subjects."""

    def __init__(self):
        super().__init__(
            script_name="prune_failed_subjects",
            description="Remove cortical_tiles outputs for subjects that failed QC.",
        )
        (
            self.add_argument(
                "output",
                help="Absolute path to the cortical_tiles output directory "
                "(the directory passed as --output to run_cortical_tiles.py).",
            )
            .add_required_argument(
                "--qc",
                "Path to QC TSV/CSV file with 'participant_id' and 'qc' columns "
                "(same format as --sk_qc_path in run_cortical_tiles.py). "
                "Subjects with qc==0 or absent from this file will be removed.",
            )
            .add_flag("--dry-run", "Print files that would be deleted without deleting them.")
        )

    def _discover_subjects(self, crops_dir: str) -> set:
        """Return all subject IDs found across region subdirs of crops_dir."""
        subjects = set()
        for region_dir in Path(crops_dir).iterdir():
            if not region_dir.is_dir():
                continue
            for f in region_dir.iterdir():
                if f.is_file():
                    # Strip compound extensions like .nii.gz
                    stem = f.name
                    for ext in (".nii.gz", ".nii", ".gz"):
                        if stem.endswith(ext):
                            stem = stem[: -len(ext)]
                            break
                    subjects.add(stem)
        return subjects

    def _read_passing_subjects(self, qc_path: str) -> set:
        """Return set of subject IDs with qc != 0."""
        sep = "\t" if qc_path.endswith(".tsv") else ","
        qc_file = pd.read_csv(qc_path, sep=sep)
        qc_file["participant_id"] = qc_file["participant_id"].astype(str)
        passing = qc_file[qc_file["qc"] != 0]["participant_id"].tolist()
        return set(passing)

    def run(self):
        """Execute the prune operation."""
        derivatives_dir = join(self.args.output, DERIVATIVES_FOLDER)
        if not exists(derivatives_dir):
            raise ValueError(
                f"Derivatives directory not found: {derivatives_dir}\n"
                f"Expected {DERIVATIVES_FOLDER}/ inside {self.args.output}"
            )

        crops_dir = join(derivatives_dir, "crops", "2mm")
        if not exists(crops_dir):
            raise ValueError(
                f"Crops directory not found: {crops_dir}\n"
                "Run run_cortical_tiles.py before pruning."
            )

        all_subjects = self._discover_subjects(crops_dir)
        if not all_subjects:
            print("No subjects found in crops directory. Nothing to prune.")
            return 0

        passing = self._read_passing_subjects(self.args.qc)
        to_remove = all_subjects - passing

        if not to_remove:
            print(f"All {len(all_subjects)} subjects pass QC. Nothing to prune.")
            return 0

        print(f"Subjects to remove ({len(to_remove)}): {sorted(to_remove)}")
        print(f"Subjects kept ({len(passing & all_subjects)}): {sorted(passing & all_subjects)}")

        deleted = 0
        for root, _dirs, files in os.walk(derivatives_dir):
            for fname in files:
                stem = fname
                for ext in (".nii.gz", ".nii", ".gz", ".csv", ".json", ".trm"):
                    if stem.endswith(ext):
                        stem = stem[: -len(ext)]
                        break
                if stem in to_remove:
                    fpath = join(root, fname)
                    if self.args.dry_run:
                        print(f"[dry-run] Would delete: {fpath}")
                    else:
                        os.remove(fpath)
                    deleted += 1

        action = "Would delete" if self.args.dry_run else "Deleted"
        print(f"\nSummary: {len(to_remove)} subjects removed, {action.lower()} {deleted} files.")
        return 0


def main():
    script = PruneFailedSubjects()
    return script.build().print_args().run()


if __name__ == "__main__":
    exit(main())
