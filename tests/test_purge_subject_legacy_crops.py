#!/usr/bin/env python3
"""
Pin tests: purge_subject keeps deleting legacy distbottom and extremities leftovers.

Stage 2 stops writing distbottom and extremities crops (epic TASK-NODISTB-BDRABCZUK-5DDC4636377C),
but derivatives written before that change still hold them. These tests build a synthetic
crops/<masks>/2mm tree in the cortical_tiles layout (<region>/mask/<side>distbottom/,
<side>extremities/, <side>distbottom.npy + _subject.csv, ...) and check that PurgeSubject.run
still removes the subject from each of them, so the guard survives the removal of the code
that wrote them.
"""

import numpy as np
import pandas as pd
import pytest

from champollion_pipeline.derivatives_layout import DEFAULT_MASKS_VERSION
from champollion_pipeline.purge_subject import PurgeSubject

SUBJECT = "sub-01"
OTHER = "sub-02"
REGION = "S.C.-sylv."
LEGACY_INPUT_TYPES = ("distbottom", "extremities")
SIDES = ("L", "R")


def _run_purge(derivatives) -> int:
    """Run PurgeSubject on derivatives for SUBJECT with the default --masks."""
    script = PurgeSubject()
    script.args = script.parse_args([str(derivatives), "--subject", SUBJECT])
    return script.run()


def _mask_dir(derivatives):
    """crops/<default masks>/2mm/<REGION>/mask under derivatives, created."""
    mask_dir = derivatives / "crops" / DEFAULT_MASKS_VERSION / "2mm" / REGION / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    return mask_dir


def _make_legacy_crop(mask_dir, side: str, input_type: str, subject: str):
    """<side><input_type>/<subject>_cropped_<input_type>.nii.gz, as cortical_tiles wrote it."""
    crop_dir = mask_dir / f"{side}{input_type}"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop = crop_dir / f"{subject}_cropped_{input_type}.nii.gz"
    crop.write_bytes(b"crop")
    return crop


def _make_legacy_array(mask_dir, side: str, input_type: str, subjects: list[str]):
    """<side><input_type>.npy with one row per subject, row i filled with i, and its _subject.csv."""
    arr = np.stack([np.full((1, 2, 2, 2), i, dtype=np.int16) for i in range(len(subjects))])
    npy = mask_dir / f"{side}{input_type}.npy"
    csv = mask_dir / f"{side}{input_type}_subject.csv"
    np.save(str(npy), arr)
    pd.DataFrame({"Subject": subjects}).to_csv(str(csv), index=False)
    return npy, csv


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("input_type", LEGACY_INPUT_TYPES)
def test_purge_deletes_legacy_crop_file(tmp_path, input_type, side):
    """A legacy <side>distbottom / <side>extremities crop file of the subject is deleted; another subject's stays."""
    mask_dir = _mask_dir(tmp_path)
    mine = _make_legacy_crop(mask_dir, side, input_type, SUBJECT)
    theirs = _make_legacy_crop(mask_dir, side, input_type, OTHER)

    assert _run_purge(tmp_path) == 0

    assert not mine.exists()
    assert theirs.read_bytes() == b"crop"


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("input_type", LEGACY_INPUT_TYPES)
def test_purge_removes_subject_row_from_legacy_array(tmp_path, input_type, side):
    """The subject's row leaves a legacy <side>distbottom / <side>extremities .npy array; the other rows stay."""
    mask_dir = _mask_dir(tmp_path)
    npy, _csv = _make_legacy_array(mask_dir, side, input_type, [OTHER, SUBJECT, "sub-03"])

    assert _run_purge(tmp_path) == 0

    arr = np.load(str(npy))
    assert arr.shape == (2, 1, 2, 2, 2)
    assert arr[0].max() == 0
    assert arr[1].max() == 2


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("input_type", LEGACY_INPUT_TYPES)
def test_purge_removes_subject_from_legacy_subject_csv(tmp_path, input_type, side):
    """The subject leaves a legacy <side>distbottom / <side>extremities _subject.csv; others stay in order."""
    mask_dir = _mask_dir(tmp_path)
    _npy, csv = _make_legacy_array(mask_dir, side, input_type, [OTHER, SUBJECT, "sub-03"])

    assert _run_purge(tmp_path) == 0

    assert pd.read_csv(str(csv))["Subject"].astype(str).tolist() == [OTHER, "sub-03"]


def test_purge_deletes_legacy_extremities_stage_directory(tmp_path):
    """<derivatives>/extremities/<subject>/ is deleted by PurgeSubject.run; another subject's directory stays."""
    mine = tmp_path / "extremities" / SUBJECT
    theirs = tmp_path / "extremities" / OTHER
    for subject_dir in (mine, theirs):
        subject_dir.mkdir(parents=True)
        (subject_dir / "Rextremities.nii.gz").write_bytes(b"vol")

    assert _run_purge(tmp_path) == 0

    assert not mine.exists()
    assert (theirs / "Rextremities.nii.gz").read_bytes() == b"vol"
