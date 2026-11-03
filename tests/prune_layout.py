"""Real cortical_tiles layout fixtures for the prune_failed_subjects tests.

cortical_tiles writes one crop per subject to
crops/<masks>/2mm/<region>/mask/<side>crops/<id>_cropped_skeleton.nii.gz (the subject
ID is the file name minus CROP_SUFFIX, kept whole) and stacks each crop dir into
crops/<masks>/2mm/<region>/mask/<side><name>.npy, whose rows follow
crops/<masks>/2mm/<region>/mask/<side><name>_subject.csv (column Subject).
<masks> is the mask version (MASKS, canonical_25, unless a helper is given another);
crops/2mm without a version is the legacy layout run_cortical_tiles migrates away.
Resampled skeletons sit at skeletons/2mm/<side>/<side>resampled_skeleton_<id>.nii.gz.

Shared by tests/test_prune_failed_subjects*.py (COMP-LOGGING-BDRABCZUK-36893E603579,
section 8). Synthetic subject IDs and trees only.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from champollion_pipeline.prune_failed_subjects import PruneFailedSubjects
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER

CROP_SUFFIX = "_cropped_skeleton.nii.gz"
MASKS = "canonical_25"


def crops_2mm(base: Path, masks: str = MASKS) -> Path:
    """<base>/<derivatives>/crops/<masks>/2mm, the directory prune discovers subjects in."""
    return base / DERIVATIVES_FOLDER / "crops" / masks / "2mm"


def legacy_crops_2mm(base: Path) -> Path:
    """<base>/<derivatives>/crops/2mm, the unversioned layout written before mask versions."""
    return base / DERIVATIVES_FOLDER / "crops" / "2mm"


def touch(path: Path) -> Path:
    """Create an empty file and its parent directories; return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def crop_file(base: Path, region: str, side: str, subject: str, masks: str = MASKS) -> Path:
    """Path of crops/<masks>/2mm/<region>/mask/<side>crops/<subject>_cropped_skeleton.nii.gz (not created)."""
    return crops_2mm(base, masks) / region / "mask" / f"{side}crops" / f"{subject}{CROP_SUFFIX}"


def make_crop_files(base: Path, region: str, side: str, subjects: list[str], masks: str = MASKS) -> list[Path]:
    """Create the per-subject crop files of one region and side; return their paths."""
    return [touch(crop_file(base, region, side, subject, masks)) for subject in subjects]


def skeleton_file(base: Path, side: str, subject: str) -> Path:
    """Path of skeletons/2mm/<side>/<side>resampled_skeleton_<subject>.nii.gz (not created)."""
    return base / DERIVATIVES_FOLDER / "skeletons" / "2mm" / side / f"{side}resampled_skeleton_{subject}.nii.gz"


def make_aggregate(
    base: Path, region: str, name: str, subjects: list[str], masks: str = MASKS
) -> tuple[Path, Path, np.ndarray]:
    """Write mask/<name>.npy (row i filled with i + 1) and mask/<name>_subject.csv; return paths and array."""
    mask = crops_2mm(base, masks) / region / "mask"
    mask.mkdir(parents=True, exist_ok=True)
    arr = np.stack([np.full((2, 3, 2, 1), i + 1, dtype=np.float32) for i in range(len(subjects))])
    npy = mask / f"{name}.npy"
    csv = mask / f"{name}_subject.csv"
    np.save(npy, arr)
    pd.DataFrame({"Subject": subjects}).to_csv(csv, index=False)
    return npy, csv, arr


def run_prune(base: Path, qc_rows: list[tuple[str, int]], dry_run: bool = False, masks: str | None = None) -> int:
    """Write <base>/qc.tsv from (participant_id, qc) rows, run PruneFailedSubjects on base; return its code.

    masks=None passes no --masks (the parser default applies).
    """
    qc = base / "qc.tsv"
    qc.write_text("participant_id\tqc\n" + "\n".join(f"{s}\t{q}" for s, q in qc_rows))
    script = PruneFailedSubjects()
    argv = [str(base), "--qc", str(qc)] + (["--dry-run"] if dry_run else [])
    script.args = script.parse_args(argv + (["--masks", masks] if masks is not None else []))
    return script.run()
