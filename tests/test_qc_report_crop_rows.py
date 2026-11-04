#!/usr/bin/env python3
"""
Shared crop-completeness check of qc_report: fetch_crop_qc_rows over a synthetic crops folder.
Consumers: run_cortical_tiles output QC, then the generate_champollion_config, generate_embeddings
and train_champollion input QC. Synthetic fixtures only.
"""

import pytest

from champollion_pipeline.utils.qc_report import fetch_crop_qc_rows
from tests.stage2_layout import add_crops, crop_array, crop_csv

_REGION = "S.C.-sylv."
_OTHER_REGION = "S.T.s."
_SUBJECT = "sub-01"
_OTHER_SUBJECT = "sub-02"
_COHORT = (_SUBJECT, _OTHER_SUBJECT)


@pytest.fixture
def cohort_crops(temp_dir):
    """Crops folder holding the skeleton crops of the cohort in ``_REGION``."""
    add_crops(temp_dir, _COHORT, [_REGION])
    return temp_dir


def _gap(kind, side, region=_REGION):
    """Reason of a crop gap."""
    return f"missing_crop:{kind}:{side}:{region}"


def _rows(crops_dir, subjects=_COHORT, regions=(_REGION,), kinds=("skeleton",), **options):
    """Rows of fetch_crop_qc_rows, keyed by participant_id."""
    return {row["participant_id"]: row for row in fetch_crop_qc_rows(crops_dir, regions, kinds, subjects, **options)}


def _assert_each_subject_fails(crops_dir, reason, **options):
    """Each subject of the cohort has qc 0 with ``reason``."""
    rows = _rows(crops_dir, **options)

    for subject in _COHORT:
        assert (rows[subject]["qc"], rows[subject]["reason"]) == (0, reason)


class TestCompleteSubject:
    def test_subject_with_each_crop_file_is_kept(self, temp_dir):
        regions = (_REGION, _OTHER_REGION)
        kinds = ("skeleton", "foldlabel", "extremities", "mask")
        add_crops(temp_dir, _COHORT, regions, kinds)

        rows = _rows(temp_dir, regions=regions, kinds=kinds)

        assert {subject: (row["qc"], row["reason"]) for subject, row in rows.items()} == {
            _SUBJECT: (1, ""),
            _OTHER_SUBJECT: (1, ""),
        }

    def test_subject_of_a_partly_listed_cohort_is_kept(self, temp_dir):
        add_crops(temp_dir, [_SUBJECT], [_REGION])

        rows = _rows(temp_dir)

        assert (rows[_SUBJECT]["qc"], rows[_SUBJECT]["reason"]) == (1, "")
        assert rows[_OTHER_SUBJECT]["qc"] == 0


class TestArrayFile:
    def test_absent_array_fails_each_subject(self, cohort_crops):
        crop_array(cohort_crops, _REGION, "L", "skeleton").unlink()

        _assert_each_subject_fails(cohort_crops, _gap("skeleton", "L"))

    def test_zero_byte_array_fails_each_subject(self, cohort_crops):
        crop_array(cohort_crops, _REGION, "L", "skeleton").write_bytes(b"")

        _assert_each_subject_fails(cohort_crops, _gap("skeleton", "L"))

    def test_gap_on_one_side_names_that_side(self, cohort_crops):
        crop_array(cohort_crops, _REGION, "R", "skeleton").unlink()

        assert _rows(cohort_crops)[_SUBJECT]["reason"] == _gap("skeleton", "R")


class TestSubjectCsv:
    def test_subject_left_out_of_the_csv(self, cohort_crops):
        add_crops(cohort_crops, [_SUBJECT], [_REGION], sides="L")

        rows = _rows(cohort_crops)

        assert (rows[_OTHER_SUBJECT]["qc"], rows[_OTHER_SUBJECT]["reason"]) == (0, _gap("skeleton", "L"))
        assert rows[_SUBJECT]["qc"] == 1

    def test_absent_csv_fails_each_subject(self, cohort_crops):
        crop_csv(cohort_crops, _REGION, "R", "skeleton").unlink()

        _assert_each_subject_fails(cohort_crops, _gap("skeleton", "R"))

    def test_csv_without_a_subject_column_fails_each_subject(self, cohort_crops):
        crop_csv(cohort_crops, _REGION, "L", "skeleton").write_text(f"ID\n{_SUBJECT}\n{_OTHER_SUBJECT}\n")

        _assert_each_subject_fails(cohort_crops, _gap("skeleton", "L"))

    def test_subject_left_out_of_one_region_only(self, temp_dir):
        regions = (_REGION, _OTHER_REGION)
        add_crops(temp_dir, _COHORT, regions)
        add_crops(temp_dir, [_SUBJECT], [_OTHER_REGION], sides="R")

        rows = _rows(temp_dir, regions=regions)

        assert rows[_OTHER_SUBJECT]["reason"] == _gap("skeleton", "R", _OTHER_REGION)


class TestMaskKind:
    def test_mask_array_alone_is_enough(self, temp_dir):
        add_crops(temp_dir, _COHORT, [_REGION], ("mask",))

        rows = _rows(temp_dir, kinds=("mask",))

        assert (rows[_SUBJECT]["qc"], rows[_SUBJECT]["reason"]) == (1, "")

    def test_absent_mask_array_fails_each_subject(self, temp_dir):
        add_crops(temp_dir, _COHORT, [_REGION], ("mask",), sides="L")

        _assert_each_subject_fails(temp_dir, _gap("mask", "R"), kinds=("mask",))


class TestFoldlabelKind:
    def test_label_files_satisfy_the_foldlabel_kind(self, temp_dir):
        add_crops(temp_dir, _COHORT, [_REGION], ("foldlabel",))

        assert crop_array(temp_dir, _REGION, "L", "foldlabel").name == "Llabel.npy"
        assert _rows(temp_dir, kinds=("foldlabel",))[_SUBJECT]["qc"] == 1

    def test_absent_label_array_names_the_foldlabel_kind(self, temp_dir):
        add_crops(temp_dir, _COHORT, [_REGION], ("foldlabel",))
        crop_array(temp_dir, _REGION, "L", "foldlabel").unlink()

        assert _rows(temp_dir, kinds=("foldlabel",))[_SUBJECT]["reason"] == _gap("foldlabel", "L")


class TestSides:
    def test_gap_on_a_side_left_out_is_ignored(self, cohort_crops):
        crop_array(cohort_crops, _REGION, "L", "skeleton").unlink()

        rows = _rows(cohort_crops, sides=("R",))

        assert (rows[_SUBJECT]["qc"], rows[_SUBJECT]["reason"]) == (1, "")

    def test_gap_on_a_requested_side_is_reported(self, cohort_crops):
        crop_array(cohort_crops, _REGION, "R", "skeleton").unlink()

        assert _rows(cohort_crops, sides=("R",))[_SUBJECT]["reason"] == _gap("skeleton", "R")

    def test_sides_default_to_both(self, temp_dir):
        add_crops(temp_dir, _COHORT, [_REGION], ("mask",), sides="")

        assert _rows(temp_dir, kinds=("mask",))[_SUBJECT]["reason"] == ";".join(_gap("mask", side) for side in "LR")


class TestSeveralReasons:
    def test_reasons_follow_region_side_kind_order(self, temp_dir):
        regions = (_OTHER_REGION, _REGION)
        kinds = ("skeleton", "mask")

        rows = _rows(temp_dir, regions=regions, kinds=kinds)

        expected = [_gap(kind, side, region) for region in regions for side in "LR" for kind in kinds]
        assert rows[_SUBJECT]["reason"] == ";".join(expected)

    def test_array_and_csv_gaps_give_one_reason(self, temp_dir):
        add_crops(temp_dir, _COHORT, [_REGION])
        crop_array(temp_dir, _REGION, "L", "skeleton").unlink()
        crop_csv(temp_dir, _REGION, "L", "skeleton").unlink()

        assert _rows(temp_dir)[_SUBJECT]["reason"] == _gap("skeleton", "L")


class TestSubjectSet:
    def test_given_subjects_get_one_row_each(self, temp_dir):
        add_crops(temp_dir, [_SUBJECT], [_REGION])

        rows = fetch_crop_qc_rows(temp_dir, [_REGION], ["skeleton"], [_OTHER_SUBJECT, _SUBJECT, "sub-03"])

        assert sorted(row["participant_id"] for row in rows) == [_SUBJECT, _OTHER_SUBJECT, "sub-03"]

    def test_subject_outside_the_collection_gets_no_row(self, temp_dir):
        add_crops(temp_dir, _COHORT, [_REGION])

        assert list(_rows(temp_dir, subjects=[_SUBJECT])) == [_SUBJECT]

    def test_none_takes_the_subjects_of_the_csvs(self, temp_dir):
        add_crops(temp_dir, _COHORT, [_REGION])

        rows = _rows(temp_dir, subjects=None)

        assert sorted(rows) == sorted(_COHORT)
        assert {row["qc"] for row in rows.values()} == {1}

    def test_none_reports_a_subject_left_out_of_one_csv(self, temp_dir):
        regions = (_REGION, _OTHER_REGION)
        add_crops(temp_dir, _COHORT, [_REGION])
        add_crops(temp_dir, [_SUBJECT], [_OTHER_REGION])

        rows = _rows(temp_dir, subjects=None, regions=regions)

        expected = ";".join(_gap("skeleton", side, _OTHER_REGION) for side in "LR")
        assert (rows[_OTHER_SUBJECT]["qc"], rows[_OTHER_SUBJECT]["reason"]) == (0, expected)

    def test_none_ignores_a_region_not_requested(self, temp_dir):
        add_crops(temp_dir, [_SUBJECT], [_REGION])
        add_crops(temp_dir, ["sub-09"], [_OTHER_REGION])

        assert list(_rows(temp_dir, subjects=None)) == [_SUBJECT]

    def test_rows_are_sorted_by_participant_id(self, temp_dir):
        add_crops(temp_dir, ["sub-03", _SUBJECT, _OTHER_SUBJECT], [_REGION])

        rows = fetch_crop_qc_rows(temp_dir, [_REGION], ["skeleton"], ["sub-03", _OTHER_SUBJECT, _SUBJECT])

        assert [row["participant_id"] for row in rows] == [_SUBJECT, _OTHER_SUBJECT, "sub-03"]


class TestBids:
    def test_session_keyed_cell_matches_its_participant(self, temp_dir):
        add_crops(temp_dir, [f"{_SUBJECT}_ses-1_run-2"], [_REGION])

        rows = _rows(temp_dir, subjects=[_SUBJECT], is_bids=True)

        assert (rows[_SUBJECT]["qc"], rows[_SUBJECT]["reason"]) == (1, "")

    def test_several_scans_of_a_participant_give_one_row(self, temp_dir):
        add_crops(temp_dir, [f"{_SUBJECT}_ses-1", f"{_SUBJECT}_ses-2"], [_REGION])

        assert list(_rows(temp_dir, subjects=None, is_bids=True)) == [_SUBJECT]

    def test_without_is_bids_the_whole_cell_is_the_id(self, temp_dir):
        add_crops(temp_dir, ["subj_01"], [_REGION])

        assert list(_rows(temp_dir, subjects=None)) == ["subj_01"]
        assert _rows(temp_dir, subjects=["subj"])["subj"]["qc"] == 0
