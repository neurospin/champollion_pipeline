#!/usr/bin/env python3
"""
Remove all cortical_tiles derivatives for a single subject.

Handles:
  - Per-subject .nii.gz crop files in Rcrops/, Rlabels/, Rextremities/, Rdistbottom/
  - Per-subject subdirectories under skeletons/, foldlabels/, extremities/, transforms/
  - Aggregated .npy arrays (Rskeleton.npy, Rlabel.npy, ...) filtered by subject CSV
"""

import logging
import os
import shutil
from datetime import datetime
from os.path import exists, isdir, join
from pathlib import Path

import numpy as np
import pandas as pd
from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.derivatives_layout import DEFAULT_MASKS_VERSION, compute_run_log_path
from champollion_pipeline.process_setup import init_pipeline_process
from champollion_pipeline.utils.cortical_tiles_config import CROPS_DIRNAME, compute_versioned_crops_dir
from champollion_pipeline.utils.subject_name import subject_name_problem

logger = logging.getLogger(__name__)

PER_SUBJECT_STAGE_DIRS = ["skeletons", "foldlabels", "extremities", "transforms", "distmaps"]


class PurgeSubject(ScriptBuilder):
    """Remove all cortical_tiles derivatives for one subject."""

    def __init__(self):
        super().__init__(
            script_name="purge_subject",
            description="Remove all cortical_tiles derivatives for a single subject.",
        )
        (
            self.add_argument(
                "derivatives",
                help="Path to the cortical_tiles derivatives directory "
                "(e.g. data/mydata/derivatives/cortical_tiles-2026/).",
            )
            .add_required_argument("--subject", "Subject ID to remove (e.g. sub-123456).")
            .add_flag("--dry-run", "Print what would be deleted without deleting.")
            .add_optional_argument(
                "--masks",
                "Mask version of the crops to purge: crops/<masks>/2mm under the derivatives directory "
                "(default canonical_25). Files under other crops/<version>/ trees are left untouched.",
                default=DEFAULT_MASKS_VERSION,
            )
        )

    def _log(self, msg: str) -> None:
        """Log one deletion action at OUTPUT; the message starts with "[dry-run] " under --dry-run."""
        prefix = "[dry-run] " if self.args.dry_run else ""
        logger.output(f"{prefix}{msg}")

    def _remove_file(self, path):
        self._log(f"Delete file: {path}")
        if not self.args.dry_run:
            Path(path).unlink()

    def _remove_dir(self, path):
        self._log(f"Delete dir:  {path}")
        if not self.args.dry_run:
            shutil.rmtree(path)

    def _per_subject_dirs(self, derivatives):
        """Return (stage dir, {subject}/ subdirectory) pairs that exist under skeletons/, foldlabels/, etc."""
        pairs = []
        for dirname in PER_SUBJECT_STAGE_DIRS:
            stage_dir = join(derivatives, dirname)
            subj_dir = join(stage_dir, self.args.subject)
            if exists(subj_dir):
                pairs.append((stage_dir, subj_dir))
        return pairs

    def _per_subject_crop_files(self, crops_2mm):
        """Return the per-subject .nii.gz files inside Rcrops/, Rlabels/, etc."""
        subject = self.args.subject
        files = []
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
                        files.append(f)
        return files

    def _purge_per_subject_dirs(self, derivatives):
        """Remove {subject}/ subdirectories under skeletons/, foldlabels/, etc."""
        for _stage_dir, subj_dir in self._per_subject_dirs(derivatives):
            self._remove_dir(subj_dir)

    def _purge_per_subject_crop_files(self, crops_2mm):
        """Remove per-subject .nii.gz files inside Rcrops/, Rlabels/, etc."""
        for f in self._per_subject_crop_files(crops_2mm):
            self._remove_file(str(f))

    def _array_csvs(self, crops_2mm):
        """Yield (region dir, subject CSV, paired .npy) for every aggregated array with a subject CSV."""
        for region_dir in Path(crops_2mm).iterdir():
            if not region_dir.is_dir():
                continue
            for mask_dir in [region_dir / "mask", region_dir]:
                if not mask_dir.exists():
                    continue
                for csv_path in sorted(mask_dir.glob("*_subject.csv")):
                    npy_path = csv_path.parent / (csv_path.name.replace("_subject.csv", ".npy"))
                    if npy_path.exists():
                        yield region_dir, csv_path, npy_path

    def _has_array_rows(self, crops_2mm):
        """True when some subject CSV of an aggregated array lists the subject."""
        for _region_dir, csv_path, _npy_path in self._array_csvs(crops_2mm):
            subjects_df = pd.read_csv(csv_path)
            if (subjects_df[subjects_df.columns[0]].astype(str) == str(self.args.subject)).any():
                return True
        return False

    def _plan(self, derivatives, crops_2mm):
        """Return (per-subject dirs, crop files, has array rows) without touching anything."""
        dirs = self._per_subject_dirs(derivatives)
        if not exists(crops_2mm):
            return dirs, [], False
        return dirs, self._per_subject_crop_files(crops_2mm), self._has_array_rows(crops_2mm)

    def _purge_aggregated_arrays(self, crops_2mm):
        """Remove subject row from aggregated .npy arrays and their subject CSVs."""
        subject = self.args.subject
        found_in_any = False

        for region_dir, csv_path, npy_path in self._array_csvs(crops_2mm):
            subjects_df = pd.read_csv(csv_path)
            col = subjects_df.columns[0]
            subjects_df[col] = subjects_df[col].astype(str)
            mask = subjects_df[col] == str(subject)

            if not mask.any():
                continue

            found_in_any = True
            idx = subjects_df.index[mask].tolist()
            self._log(f"Filter row(s) {idx} from array: {npy_path.name} (region {region_dir.name})")

            if not self.args.dry_run:
                arr = np.load(str(npy_path), allow_pickle=True)
                keep = ~mask.values
                arr_filtered = arr[keep]
                np.save(str(npy_path), arr_filtered)

                subjects_filtered = subjects_df[keep].reset_index(drop=True)
                subjects_filtered.to_csv(str(csv_path), index=False)

        if not found_in_any:
            logger.warning(f"Subject '{subject}' not found in any aggregated array.")

    def default_log_path(self) -> str:
        """Run-log path of a bare --log: <dataset>/derivatives/.log/purge_subject_<stamp>.log."""
        return compute_run_log_path(
            str(Path(self.args.derivatives).parent), self.script_name, datetime.now().astimezone()
        )

    def _versioned_crops_dir(self, derivatives: str) -> str:
        """crops/<masks>/2mm under derivatives; ValueError, before any deletion, for an unusable --masks."""
        masks = self.args.masks
        reason = subject_name_problem(masks)
        if reason is not None:
            raise ValueError(f"Invalid --masks {masks!r}: {reason}.")
        return compute_versioned_crops_dir(derivatives, masks)

    def _warn_missing_crops_dir(self, derivatives: str, crops_dir: str) -> None:
        """One WARNING naming the missing crops/<masks>/2mm, with a hint when a legacy crops/2mm exists."""
        message = f"No crops directory for --masks {self.args.masks}: {crops_dir} not found, skipping crop purge."
        legacy = join(derivatives, CROPS_DIRNAME, "2mm")
        if isdir(legacy):
            message += (
                f" A legacy unversioned crops directory exists at {legacy}: rerun run_cortical_tiles.py "
                f"to move it under crops/<masks_version>/2mm."
            )
        logger.warning(message)

    def run(self):
        derivatives = self.args.derivatives
        if not exists(derivatives):
            raise ValueError(f"Derivatives directory not found: {derivatives}")

        crops_dir = self._versioned_crops_dir(derivatives)
        subject = self.args.subject

        reason = subject_name_problem(subject)
        if reason is not None:
            logger.error(f"Refusing --subject {subject!r}: {reason}.")
            return 1

        dirs, crop_files, has_array_rows = self._plan(derivatives, crops_dir)
        if not dirs and not crop_files and not has_array_rows:
            logger.error(
                f"Refusing --subject {subject!r}: no per-subject directory, crop file "
                f"or subject CSV row in {derivatives}."
            )
            return 1

        outside = False
        for stage_dir, subj_dir in dirs:
            if not os.path.realpath(subj_dir).startswith(os.path.realpath(stage_dir) + os.sep):
                logger.error(
                    f"Refusing --subject {subject!r}: {subj_dir} resolves to "
                    f"{os.path.realpath(subj_dir)}, outside {stage_dir}."
                )
                outside = True
        if outside:
            return 1

        logger.trace(f"Purging subject '{subject}' from: {derivatives}")

        for _stage_dir, subj_dir in dirs:
            self._remove_dir(subj_dir)
        if exists(crops_dir):
            for f in crop_files:
                self._remove_file(str(f))
            self._purge_aggregated_arrays(crops_dir)
        else:
            self._warn_missing_crops_dir(derivatives, crops_dir)

        action = "Would remove" if self.args.dry_run else "Removed"
        logger.output(f"Done. {action} all data for subject '{subject}'.")
        return 0


def main():
    init_pipeline_process()
    script = PurgeSubject()
    return script.main()


if __name__ == "__main__":
    exit(main())
