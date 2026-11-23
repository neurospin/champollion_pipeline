#!/usr/bin/env python3
"""
Grouped crop QC of qc_report (TASK-MODELVER-BDRABCZUK-037E4DCBA29C): fetch_grouped_crop_qc_rows over crop groups
(crops folder, hemisphere side, Sulcal regions) laid in two Mask version folders. Synthetic fixtures only.
"""

from pathlib import Path

import pytest

from champollion_pipeline.utils.qc_report import fetch_crop_qc_rows, fetch_grouped_crop_qc_rows
from tests.stage2_layout import add_crops, crop_array, crop_csv

_SYLVIAN = "S.C.-sylv."
_STS = "S.T.s."
_SUBJECT = "sub-01"
_OTHER_SUBJECT = "sub-02"
_COHORT = (_SUBJECT, _OTHER_SUBJECT)
_KINDS = ("skeleton",)


@pytest.fixture
def version_dirs(temp_dir):
    """The 2mm crops folders of two Mask versions."""
    return tuple(str(Path(temp_dir) / "crops" / version / "2mm") for version in ("canonical_25", "canonical_26"))


def _gap(side, region, kind="skeleton"):
    """Reason of a crop gap."""
    return f"missing_crop:{kind}:{side}:{region}"


def _qc(rows):
    """participant_id -> (qc, reason)."""
    return {row["participant_id"]: (row["qc"], row["reason"]) for row in rows}


# REQ-MODELVER-BDRABCZUK-467F5322F4B0
def test_missing_crop_of_one_group_fails_the_subject(version_dirs):
    first, second = version_dirs
    add_crops(first, _COHORT, [_SYLVIAN], sides="L")
    add_crops(second, [_SUBJECT], [_STS], sides="R")

    rows = fetch_grouped_crop_qc_rows([(first, "L", [_SYLVIAN]), (second, "R", [_STS])], _KINDS, _COHORT)

    assert _qc(rows) == {_SUBJECT: (1, ""), _OTHER_SUBJECT: (0, _gap("R", _STS))}


def test_subject_complete_in_each_group_is_kept(version_dirs):
    first, second = version_dirs
    add_crops(first, _COHORT, [_SYLVIAN], sides="L")
    add_crops(second, _COHORT, [_STS], sides="R")

    rows = fetch_grouped_crop_qc_rows([(first, "L", [_SYLVIAN]), (second, "R", [_STS])], _KINDS)

    assert _qc(rows) == {_SUBJECT: (1, ""), _OTHER_SUBJECT: (1, "")}


def _remove_left_sts_array(crops_dir):
    crop_array(crops_dir, _STS, "L", "skeleton").unlink()


def _unlist_other_subject_right(crops_dir):
    crop_csv(crops_dir, _SYLVIAN, "R", "skeleton").write_text(f"Subject\n{_SUBJECT}\n")


def _keep_everything(crops_dir):
    del crops_dir


def _reasons(rows):
    """participant_id -> (qc, set of reasons): reason order follows the groups (REQ-MODELVER-BDRABCZUK-AF2B8B314013)."""
    return {row["participant_id"]: (row["qc"], set(filter(None, row["reason"].split(";")))) for row in rows}


# REQ-MODELVER-BDRABCZUK-D80FD6D3FF36
@pytest.mark.parametrize("change", [_keep_everything, _remove_left_sts_array, _unlist_other_subject_right])
@pytest.mark.parametrize("subjects", [None, (*_COHORT, "sub-03")])
def test_one_group_per_side_equals_the_crop_qc(temp_dir, change, subjects):
    regions = [_SYLVIAN, _STS]
    add_crops(temp_dir, _COHORT, regions)
    change(temp_dir)

    grouped = fetch_grouped_crop_qc_rows([(temp_dir, "L", regions), (temp_dir, "R", regions)], _KINDS, subjects)

    assert _reasons(grouped) == _reasons(fetch_crop_qc_rows(temp_dir, regions, _KINDS, subjects))


# REQ-MODELVER-BDRABCZUK-8EAF4A82AB4F
def test_cohort_is_the_union_of_the_group_subject_lists(version_dirs):
    first, second = version_dirs
    add_crops(first, [_SUBJECT], [_SYLVIAN], sides="L")
    add_crops(second, [_OTHER_SUBJECT], [_STS], sides="R")

    rows = fetch_grouped_crop_qc_rows([(first, "L", [_SYLVIAN]), (second, "R", [_STS])], _KINDS)

    assert [row["participant_id"] for row in rows] == [_SUBJECT, _OTHER_SUBJECT]


def test_given_subjects_are_the_cohort(version_dirs):
    first, second = version_dirs
    add_crops(first, _COHORT, [_SYLVIAN], sides="L")
    add_crops(second, _COHORT, [_STS], sides="R")

    rows = fetch_grouped_crop_qc_rows([(first, "L", [_SYLVIAN]), (second, "R", [_STS])], _KINDS, ["sub-03"])

    assert _qc(rows) == {"sub-03": (0, f"{_gap('L', _SYLVIAN)};{_gap('R', _STS)}")}


# REQ-MODELVER-BDRABCZUK-AF2B8B314013
def test_reasons_follow_group_order_once_each(version_dirs):
    first, second = version_dirs
    add_crops(first, _COHORT, [_SYLVIAN], sides="L")
    add_crops(second, _COHORT, [_STS], sides="R")
    groups = [(second, "R", [_STS]), (first, "L", [_SYLVIAN]), (second, "R", [_STS])]

    rows = fetch_grouped_crop_qc_rows(groups, _KINDS, ["sub-03"])

    assert _qc(rows) == {"sub-03": (0, f"{_gap('R', _STS)};{_gap('L', _SYLVIAN)}")}
