#!/usr/bin/env python3
"""
Remove cortical_tiles outputs for subjects that failed QC.

Scans the cortical_tiles derivatives directory, reads a QC TSV/CSV file, and
deletes all files belonging to subjects with qc==0 or absent from the QC file.
This brings the output directory to the same state as if --sk_qc_path had been
passed to run_cortical_tiles.py from the start.
"""

import logging
import os
import re
from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime
from os.path import exists, isdir, join
from pathlib import Path, PurePath

import numpy as np
import pandas as pd
from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.derivatives_layout import DEFAULT_MASKS_VERSION, compute_run_log_path
from champollion_pipeline.process_setup import init_pipeline_process
from champollion_pipeline.utils.cortical_tiles_config import CROPS_DIRNAME, compute_versioned_crops_dir
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from champollion_pipeline.utils.subject_name import subject_name_problem

logger = logging.getLogger(__name__)

CROP_SUFFIX = "_cropped_skeleton.nii.gz"
CROP_SIDES = ("L", "R")
SUBJECT_CSV_SUFFIX = "_subject.csv"


def _file_belongs_to(rel_path: PurePath, subjects: Collection[str]) -> bool:
    """True when a subject ID is a whole directory of rel_path or a whole '_'/'.' token of its name.

    '-' is not a delimiter, so sub-10 never matches sub-1.
    """
    dirs = set(rel_path.parent.parts)
    for subject in subjects:
        if subject in dirs:
            return True
        if re.search(r"(?<![^_.])" + re.escape(subject) + r"(?![^_.])", rel_path.name):
            return True
    return False


def _in_masks_scope(rel_path: PurePath, masks: str) -> bool:
    """True for a file outside crops/; for a file under crops/, True only when it lies under crops/<masks>/."""
    parts = rel_path.parts
    if parts[0] != CROPS_DIRNAME:
        return True
    return len(parts) > 2 and parts[1] == masks


def _crop_ids(side_dir: Path) -> set[str]:
    """<id> of each regular file <id>_cropped_skeleton.nii.gz of side_dir (empty when side_dir is missing)."""
    if not side_dir.is_dir():
        return set()
    return {
        f.name[: -len(CROP_SUFFIX)]
        for f in side_dir.iterdir()
        if f.is_file() and f.name.endswith(CROP_SUFFIX) and len(f.name) > len(CROP_SUFFIX)
    }


@dataclass(frozen=True)
class AggregatePlan:
    """Rows of one <region>/mask/<name>_subject.csv and its paired <name>.npy to filter out."""

    csv_path: Path
    npy_path: Path
    removed: tuple[int, ...]
    n_rows: int


def _paired_npy(csv_path: Path) -> Path:
    """<name>.npy beside <name>_subject.csv."""
    return csv_path.with_name(csv_path.name[: -len(SUBJECT_CSV_SUFFIX)] + ".npy")


def _read_subject_csv(csv_path: Path) -> pd.DataFrame:
    """Subject CSV as exact strings (no NA parsing); an empty file reads as no column."""
    try:
        return pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _npy_rows(npy_path: Path) -> int:
    """Row count of an array file read without pickle; ValueError for a 0-d array."""
    arr = np.load(npy_path, mmap_mode="r", allow_pickle=False)
    if arr.ndim == 0:
        raise ValueError("0-d array")
    return arr.shape[0]


def _plan_pair(csv_path: Path, subjects: Collection[str]) -> tuple[AggregatePlan | None, str | None]:
    """Plan of one CSV/npy pair (None when no row matches or no npy), or its input-check problem."""
    npy_path = _paired_npy(csv_path)
    if not npy_path.is_file():
        return None, None
    df = _read_subject_csv(csv_path)
    if df.shape[1] == 0:
        return None, f"{csv_path}: no subject column"
    removed = tuple(int(i) for i in np.flatnonzero(df.iloc[:, 0].isin(list(subjects)).to_numpy()))
    if not removed:
        return None, None
    n_rows = len(df)
    try:
        arr_rows = _npy_rows(npy_path)
    except (OSError, ValueError) as exc:
        return None, f"{npy_path}: unreadable array ({exc})"
    if arr_rows != n_rows:
        return None, f"{npy_path}: {arr_rows} array rows but {n_rows} subject rows in {csv_path.name}"
    return AggregatePlan(csv_path=csv_path, npy_path=npy_path, removed=removed, n_rows=n_rows), None


def plan_aggregate_rows(crops_dir: str, subjects: Collection[str]) -> tuple[list[AggregatePlan], list[str]]:
    """Read-only. Plans (removed non-empty), sorted by csv_path, and one problem text per invalid pair."""
    plans: list[AggregatePlan] = []
    problems: list[str] = []
    for csv_path in sorted(Path(crops_dir).glob(f"*/mask/*{SUBJECT_CSV_SUFFIX}")):
        plan, problem = _plan_pair(csv_path, subjects)
        if problem is not None:
            problems.append(problem)
        elif plan is not None:
            plans.append(plan)
    return plans, problems


def _replace_via_tmp(path: Path, write, mode: str) -> None:
    """Write through an open handle on the sibling <name>.tmp, then os.replace it onto path."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, mode, **({"newline": ""} if "b" not in mode else {})) as fh:
        write(fh)
    os.replace(tmp, path)


def filter_aggregate_rows(plan: AggregatePlan) -> str | None:
    """Rewrite npy then csv without plan.removed rows, order kept; re-read both; problem text or None."""
    removed = list(plan.removed)
    arr = np.load(plan.npy_path, allow_pickle=False)
    df = _read_subject_csv(plan.csv_path)
    pruned = set(df.iloc[removed, 0])
    kept = np.delete(arr, removed, axis=0)
    df_kept = df.drop(index=removed)
    _replace_via_tmp(plan.npy_path, lambda fh: np.save(fh, kept), "wb")
    _replace_via_tmp(plan.csv_path, lambda fh: df_kept.to_csv(fh, index=False), "w")

    expected = plan.n_rows - len(removed)
    arr_rows = _npy_rows(plan.npy_path)
    df_after = _read_subject_csv(plan.csv_path)
    if not arr_rows == len(df_after) == expected:
        return (
            f"{plan.npy_path}: {arr_rows} array rows and {len(df_after)} subject rows after filtering, "
            f"expected {expected}"
        )
    if df_after.shape[1] and df_after.iloc[:, 0].isin(list(pruned)).any():
        return f"{plan.csv_path}: still lists a pruned subject after filtering"
    return None


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
            .add_optional_argument(
                "--masks",
                "Mask version of the crops to prune: crops/<masks>/2mm under the derivatives directory "
                "(default canonical_25). Files under other crops/<version>/ trees are left untouched.",
                default=DEFAULT_MASKS_VERSION,
            )
        )

    def _discover_subjects(self, crops_dir: str) -> set[str]:
        """IDs of crops_dir/<region>/mask/<side>crops/<id>_cropped_skeleton.nii.gz, side in CROP_SIDES."""
        subjects: set[str] = set()
        for region_dir in Path(crops_dir).iterdir():
            if not region_dir.is_dir():
                continue
            for side in CROP_SIDES:
                subjects |= _crop_ids(region_dir / "mask" / f"{side}crops")
        return subjects

    def _read_passing_subjects(self, qc_path: str) -> set:
        """Return set of subject IDs with qc != 0."""
        sep = "\t" if qc_path.endswith(".tsv") else ","
        qc_file = pd.read_csv(qc_path, sep=sep)
        qc_file["participant_id"] = qc_file["participant_id"].astype(str)
        passing = qc_file[qc_file["qc"] != 0]["participant_id"].tolist()
        return set(passing)

    def default_log_path(self) -> str:
        """Run-log path of a bare --log: <output>/.log/prune_failed_subjects_<stamp>.log (<output> = derivatives)."""
        return compute_run_log_path(self.args.output, self.script_name, datetime.now().astimezone())

    def _resolve_crops_dir(self, derivatives_dir: str) -> str:
        """crops/<masks>/2mm under derivatives_dir; ValueError, before any deletion, when it cannot be used."""
        if not exists(derivatives_dir):
            raise ValueError(
                f"Derivatives directory not found: {derivatives_dir}\n"
                f"Expected {DERIVATIVES_FOLDER}/ inside {self.args.output}"
            )
        masks = self.args.masks
        reason = subject_name_problem(masks)
        if reason is not None:
            raise ValueError(f"Invalid --masks {masks!r}: {reason}.")
        crops_dir = compute_versioned_crops_dir(derivatives_dir, masks)
        if not isdir(crops_dir):
            message = f"Crops directory for --masks {masks} not found: {crops_dir}"
            legacy = join(derivatives_dir, CROPS_DIRNAME, "2mm")
            if isdir(legacy):
                message += (
                    f"; a legacy unversioned crops directory exists at {legacy}: rerun run_cortical_tiles.py on "
                    f"{self.args.output} to move it under crops/<masks_version>/2mm"
                )
            raise ValueError(message)
        return crops_dir

    def _delete_subject_files(self, derivatives_dir: str, to_remove: set[str]) -> int:
        """Delete (or list under --dry-run) the in-scope files of to_remove; return their count (P3/P4)."""
        deleted = 0
        for root, _dirs, files in os.walk(derivatives_dir):
            for fname in files:
                rel = PurePath(root, fname).relative_to(derivatives_dir)
                if _in_masks_scope(rel, self.args.masks) and _file_belongs_to(rel, to_remove):
                    fpath = join(root, fname)
                    if self.args.dry_run:
                        logger.output(f"[dry-run] Would delete: {fpath}")
                    else:
                        os.remove(fpath)
                        logger.output(f"Removed: {fpath}")
                    deleted += 1
        return deleted

    def run(self):
        """Execute the prune operation."""
        derivatives_dir = join(self.args.output, DERIVATIVES_FOLDER)
        crops_dir = self._resolve_crops_dir(derivatives_dir)

        all_subjects = self._discover_subjects(crops_dir)
        passing_subjects = self._read_passing_subjects(self.args.qc)
        to_remove = all_subjects - passing_subjects

        if not to_remove:
            logger.output(f"All {len(all_subjects)} subjects pass QC. Nothing to prune.")
            return 0

        logger.output(f"Subjects to remove ({len(to_remove)}): {sorted(to_remove)}")

        plans, problems = plan_aggregate_rows(crops_dir, to_remove)
        if not self._check_aggregates(problems):
            return 1

        deleted = self._delete_subject_files(derivatives_dir, to_remove)
        aggregates_ok = self._prune_aggregates(plans)

        action = "Would delete" if self.args.dry_run else "Deleted"
        logger.output(f"Summary: {len(to_remove)} subjects removed, {action.lower()} {deleted} files.")
        return 0 if aggregates_ok else 1

    def _check_aggregates(self, problems: list[str]) -> bool:
        """A1 per input-check problem; True when there is none."""
        for problem in problems:
            logger.error(f"Refusing to prune: {problem}.")
        return not problems

    def _prune_aggregates(self, plans: list[AggregatePlan]) -> bool:
        """P6 per plan under --dry-run; else filter each plan (P7, or A2 on a failed check). False after an A2."""
        ok = True
        for plan in plans:
            if self.args.dry_run:
                logger.output(
                    f"[dry-run] Would filter rows {list(plan.removed)} from: {plan.npy_path} and {plan.csv_path.name}"
                )
                continue
            problem = filter_aggregate_rows(plan)
            if problem is not None:
                logger.error(f"Aggregate check failed after filtering: {problem}.")
                ok = False
            else:
                logger.output(f"Filtered rows {list(plan.removed)} from: {plan.npy_path} and {plan.csv_path.name}")
        return ok


def main():
    init_pipeline_process()
    script = PruneFailedSubjects()
    return script.main()


if __name__ == "__main__":
    exit(main())
