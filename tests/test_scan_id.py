#!/usr/bin/env python3
"""
Tests for ``champollion_pipeline.file_indexer.ScanId``.

ScanId is the unit of work of the scan-centric streaming mode. It follows the
Morphologist layout strictly: graphs live at
``<subjects_dir>/<subject>/t1mri/<acquisition>/default_analysis/folds/...``,
so a scan is one subject plus one acquisition folder name kept verbatim
(``default_acquisition``, ``ses-V1``, or a BIDS key such as ``ses-1_run-1``),
never split into session and run.

Its string form is the embeddings CSV ID and matches the sequential
cortical_tiles subject identifier exactly: the subject alone in the classic
format, the subject plus the ses-/acq-/run- entities found in the path with
--bids (cortical_tiles generate_skeletons.get_skeleton_filename, mirrored here).

Covers REQ-STREAMING-BDRABCZUK-9AB08EE49DFD (three fields, acquisition verbatim),
REQ-STREAMING-BDRABCZUK-A1A589AC7DB7 (bids defaults to False),
REQ-STREAMING-BDRABCZUK-C9BE1D914854 (acquisition mandatory),
REQ-STREAMING-BDRABCZUK-2CAAB8E6C69A (path_prefix),
REQ-STREAMING-BDRABCZUK-8B899B3A9B10 (frozen fields),
REQ-STREAMING-BDRABCZUK-FA8B330C15BF (equality and hash),
REQ-STREAMING-BDRABCZUK-E4B5FD8209E6 (ordering),
REQ-STREAMING-BDRABCZUK-B18FBDA4F9A8 (str, classic format),
REQ-STREAMING-BDRABCZUK-AFC4B66C3DC6 (str, BIDS format) and
REQ-STREAMING-BDRABCZUK-22A200032839 (package-level export).
"""

import dataclasses

import pytest

from champollion_pipeline import file_indexer
from champollion_pipeline.file_indexer.scan_id import ScanId


@pytest.mark.unit
class TestScanIdFields:
    """REQ-STREAMING-BDRABCZUK-9AB08EE49DFD, -A1A589AC7DB7 and -C9BE1D914854."""

    def test_exactly_subject_acquisition_bids(self):
        assert [field.name for field in dataclasses.fields(ScanId)] == ["subject", "acquisition", "bids"]

    @pytest.mark.parametrize("acquisition", ["default_acquisition", "ses-V1", "ses-1_run-1", "ses-1_acq-T1w_run-2"])
    def test_acquisition_kept_verbatim(self, acquisition):
        scan = ScanId("sub-01", acquisition)
        assert dataclasses.astuple(scan) == ("sub-01", acquisition, False)

    def test_bids_defaults_to_false(self):
        assert ScanId("sub-01", "ses-1_run-1").bids is False

    def test_acquisition_is_required(self):
        with pytest.raises(TypeError):
            ScanId("sub-01")


@pytest.mark.unit
class TestScanIdPathPrefix:
    """REQ-STREAMING-BDRABCZUK-2CAAB8E6C69A: '<subject>/t1mri/<acquisition>'."""

    @pytest.mark.parametrize(
        ("subject", "acquisition", "expected"),
        [
            ("sub-01", "ses-1_run-1", "sub-01/t1mri/ses-1_run-1"),
            ("subject_A", "default_acquisition", "subject_A/t1mri/default_acquisition"),
            ("sub-01", "ses-V1", "sub-01/t1mri/ses-V1"),
        ],
    )
    def test_path_prefix(self, subject, acquisition, expected):
        assert ScanId(subject, acquisition).path_prefix == expected


@pytest.mark.unit
class TestScanIdStrClassic:
    """REQ-STREAMING-BDRABCZUK-B18FBDA4F9A8: bids False -> the subject alone, whatever the acquisition."""

    @pytest.mark.parametrize(
        ("subject", "acquisition"),
        [
            ("subject_A", "default_acquisition"),
            ("sub-01", "ses-V1"),
            ("sub-01", "ses-1_run-1"),
        ],
    )
    def test_subject_alone(self, subject, acquisition):
        assert str(ScanId(subject, acquisition, bids=False)) == subject

    def test_two_acquisitions_of_one_subject_share_the_id(self):
        """Accepted caveat, as in the sequential pipeline."""
        assert str(ScanId("sub-01", "default_acquisition")) == str(ScanId("sub-01", "ses-V1"))


@pytest.mark.unit
class TestScanIdStrBids:
    """REQ-STREAMING-BDRABCZUK-AFC4B66C3DC6: subject, then '_ses-X', '_acq-Y', '_run-Z' found in path_prefix."""

    @pytest.mark.parametrize(
        ("acquisition", "expected"),
        [
            ("ses-1_run-1", "sub-01_ses-1_run-1"),
            ("ses-1_acq-T1w_run-2", "sub-01_ses-1_acq-T1w_run-2"),
            ("ses-V1", "sub-01_ses-V1"),
            ("acq-T1w", "sub-01_acq-T1w"),
            ("run-1_ses-2", "sub-01_ses-2_run-1"),
            ("ses-1_rec-norm_run-1", "sub-01_ses-1_run-1"),
            ("default_acquisition", "sub-01"),
        ],
    )
    def test_entities_in_fixed_order(self, acquisition, expected):
        assert str(ScanId("sub-01", acquisition, bids=True)) == expected


@pytest.mark.unit
class TestScanIdValueSemantics:
    """REQ-STREAMING-BDRABCZUK-8B899B3A9B10 and -FA8B330C15BF."""

    @pytest.mark.parametrize("field", ["subject", "acquisition"])
    def test_fields_are_frozen(self, field):
        scan = ScanId("sub-01", "ses-1_run-1")
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(scan, field, "changed")

    @pytest.mark.parametrize("scan_args", [("sub-01", "default_acquisition"), ("sub-01", "ses-1_run-1")])
    def test_equal_values_equal_hash(self, scan_args):
        first, second = ScanId(*scan_args), ScanId(*scan_args)
        assert first is not second
        assert first == second
        assert hash(first) == hash(second)

    def test_bids_flag_does_not_change_identity(self):
        first, second = ScanId("sub-01", "ses-1_run-1", bids=True), ScanId("sub-01", "ses-1_run-1")
        assert first == second
        assert hash(first) == hash(second)

    def test_different_acquisition_not_equal(self):
        assert ScanId("sub-01", "ses-1_run-1") != ScanId("sub-01", "ses-1_run-2")


@pytest.mark.unit
class TestScanIdOrdering:
    """REQ-STREAMING-BDRABCZUK-E4B5FD8209E6: subject, then acquisition, as strings."""

    def test_sorted_order(self):
        scans = [
            ScanId("sub-02", "default_acquisition"),
            ScanId("sub-01", "ses-2_run-1"),
            ScanId("sub-01", "ses-1_run-2"),
            ScanId("sub-01", "default_acquisition"),
            ScanId("sub-01", "ses-1_run-1"),
        ]
        assert [dataclasses.astuple(scan) for scan in sorted(scans)] == [
            ("sub-01", "default_acquisition", False),
            ("sub-01", "ses-1_run-1", False),
            ("sub-01", "ses-1_run-2", False),
            ("sub-01", "ses-2_run-1", False),
            ("sub-02", "default_acquisition", False),
        ]


@pytest.mark.unit
def test_package_exports_scan_id():
    """REQ-STREAMING-BDRABCZUK-22A200032839: ``file_indexer.ScanId`` is the scan_id class."""
    assert file_indexer.ScanId is ScanId
