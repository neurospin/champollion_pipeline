"""Skeleton check of the stage 2 input QC, and the crop_qc.tsv that keeps its exclusions out of the crop step.

Holds the one definition of the skeleton file pattern shared by the pre-check of main.py and the input QC. main.py loads
this module, so it imports no toolchain package: aims.read comes in as the ``read_volume`` argument.
"""

import glob
import json
import logging
import os
from collections.abc import Callable, Mapping
from os.path import exists, isfile, join

import champollion_utils.logger  # noqa: F401 - installs Logger.trace/output/fail

from champollion_pipeline.utils.qc_report import REASON_SEPARATOR, write_qc_tsv

logger = logging.getLogger(__name__)

SKELETON_SIDES = ("L", "R")
REASON_MISSING_SKELETON_STEM = "missing_skeleton_"  # + side
REASON_UNUSABLE_SKELETON_STEM = "unusable_skeleton_"  # + side
CROP_QC_FILENAME = "crop_qc.tsv"


def compute_skeleton_pattern(side: str) -> str:
    """File name pattern of the skeleton of hemisphere ``side``, ``*`` as a wildcard."""
    return f"{side}skeleton*.nii.gz"


def list_skeleton_files(subject_dir: str, folders: list[str], side: str) -> list[str]:
    """Skeleton files of ``side`` in ``folders``, relative to ``subject_dir``, each folder's files sorted."""
    files = []
    for folder in folders:
        pattern = join(glob.escape(join(subject_dir, folder)), compute_skeleton_pattern(side))
        files.extend(match for match in sorted(glob.glob(pattern)) if isfile(match))
    return files


def has_nonzero_voxel(path: str, read_volume: Callable[[str], object]) -> bool:
    """True when the volume at ``path`` holds a non-zero voxel; a file ``read_volume`` refuses holds none."""
    try:
        volume = read_volume(path)
    except Exception as error:  # REQ-PRECHECK-BDRABCZUK-BE65CA1049B5: whatever aims.read raises
        logger.trace(f"aims.read refused {path}: {error}")
        return False
    return bool(volume.np.any())


def compute_skeleton_reasons(subject_dir: str, folders: list[str], read_volume: Callable[[str], object]) -> list[str]:
    """Input QC reasons of a subject's skeletons, L before R, at most one per side."""
    reasons = []
    for side in SKELETON_SIDES:
        files = list_skeleton_files(subject_dir, folders, side)
        if not files:
            reasons.append(f"{REASON_MISSING_SKELETON_STEM}{side}")
        elif not any(has_nonzero_voxel(path, read_volume) for path in files):
            reasons.append(f"{REASON_UNUSABLE_SKELETON_STEM}{side}")
    return reasons


def has_skeleton_reason(row: Mapping[str, object]) -> bool:
    """True when the reason cell of an input_qc.tsv ``row`` holds a skeleton reason."""
    stems = (REASON_MISSING_SKELETON_STEM, REASON_UNUSABLE_SKELETON_STEM)
    return any(reason.startswith(stems) for reason in str(row["reason"]).split(REASON_SEPARATOR))


def save_crop_qc(derivatives_dir: str, rows: list[dict[str, object]]) -> str:
    """Write crop_qc.tsv with the qc 1 ``rows`` when one row holds a skeleton reason; return its path, else ""."""
    if not any(has_skeleton_reason(row) for row in rows):
        return ""
    path = join(derivatives_dir, CROP_QC_FILENAME)
    write_qc_tsv(path, [row for row in rows if row["qc"] == 1])
    return path


def save_config_skel_qc(config_path: str, crop_qc_abs: str) -> None:
    """Set skel_qc_path of the pipeline JSON config, on a run that skips the crop step.

    Only this key changes: _preflight_check reads the stored masks_version. The file is replaced, not opened for
    writing, because an older run may have left a config with no write bit.
    """
    if not crop_qc_abs or not exists(config_path):
        return
    with open(config_path) as f:
        config = json.load(f)
    config["skel_qc_path"] = crop_qc_abs
    tmp_path = f"{config_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(config, f, indent=3)
    os.replace(tmp_path, config_path)
