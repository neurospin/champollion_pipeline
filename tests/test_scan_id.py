#!/usr/bin/env python3
"""
Tests for ``champollion_pipeline.file_indexer.ScanId``.

ScanId is the unit of work of the scan-centric streaming mode: one subject,
plus the BIDS session and acquisition when the database has them. Its string
form names per-scan log and embedding folders, so for a BIDS scan without
acquisition it must equal the stage-1 ``--bids`` participant id
(``sub-<label>`` then ``_ses-<label>``).

Covers REQ-STREAMING-BDRABCZUK-932C0A773718 (path_prefix),
REQ-STREAMING-BDRABCZUK-5A0B08634E45 (str without acquisition),
REQ-STREAMING-BDRABCZUK-C7FC55C0F216 (str with acquisition suffix),
REQ-STREAMING-BDRABCZUK-39BDD4DA073E (frozen fields),
REQ-STREAMING-BDRABCZUK-E9A8B59033D0 (equality and hash),
REQ-STREAMING-BDRABCZUK-56B2F5790F3B (ordering, None first) and
REQ-STREAMING-BDRABCZUK-22A200032839 (package-level export).
"""

import dataclasses

import pytest

from champollion_pipeline import file_indexer
from champollion_pipeline.file_indexer.scan_id import ScanId


def _fields(scan: ScanId) -> tuple:
    return (scan.subject, scan.session, scan.acquisition)


@pytest.mark.unit
class TestScanIdPathPrefix:
    """REQ-STREAMING-BDRABCZUK-932C0A773718: subject, then '/<session>' when set; acquisition ignored."""

    @pytest.mark.parametrize(
        ("scan_args", "expected"),
        [
            (("sub-001",), "sub-001"),
            (("sub-001", None, "acq-T1w"), "sub-001"),
            (("sub-001", "ses-01"), "sub-001/ses-01"),
            (("sub-001", "ses-01", "acq-T1w"), "sub-001/ses-01"),
        ],
    )
    def test_path_prefix(self, scan_args, expected):
        assert ScanId(*scan_args).path_prefix == expected


@pytest.mark.unit
class TestScanIdStr:
    """REQ-STREAMING-BDRABCZUK-5A0B08634E45: subject, then '_<session>' when set (no acquisition)."""

    def test_subject_only(self):
        assert str(ScanId("sub-001")) == "sub-001"

    def test_non_bids_subject_name(self):
        assert str(ScanId("subject_A")) == "subject_A"

    def test_subject_and_session_match_stage1_bids_participant_id(self):
        assert str(ScanId("sub-01", "ses-01")) == "sub-01_ses-01"


@pytest.mark.unit
class TestScanIdStrWithAcquisition:
    """REQ-STREAMING-BDRABCZUK-C7FC55C0F216: subject, then '_<session>' when set, then '_<acquisition>'."""

    @pytest.mark.parametrize(
        ("scan_args", "expected"),
        [
            (("sub-01", "ses-01", "acq-T1w"), "sub-01_ses-01_acq-T1w"),
            (("sub-01", None, "acq-T1w"), "sub-01_acq-T1w"),
        ],
    )
    def test_acquisition_suffix(self, scan_args, expected):
        assert str(ScanId(*scan_args)) == expected


@pytest.mark.unit
class TestScanIdValueSemantics:
    """REQ-STREAMING-BDRABCZUK-39BDD4DA073E and -E9A8B59033D0."""

    @pytest.mark.parametrize("field", ["subject", "session", "acquisition"])
    def test_fields_are_frozen(self, field):
        scan = ScanId("sub-01", "ses-01", "acq-T1w")
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(scan, field, "changed")

    @pytest.mark.parametrize(
        "scan_args",
        [("sub-01",), ("sub-01", "ses-01"), ("sub-01", None, "acq-T1w"), ("sub-01", "ses-01", "acq-T1w")],
    )
    def test_equal_values_equal_hash(self, scan_args):
        first, second = ScanId(*scan_args), ScanId(*scan_args)
        assert first is not second
        assert first == second
        assert hash(first) == hash(second)

    def test_different_values_not_equal(self):
        assert ScanId("sub-01", "ses-01") != ScanId("sub-01", "ses-02")


@pytest.mark.unit
class TestScanIdOrdering:
    """REQ-STREAMING-BDRABCZUK-56B2F5790F3B: subject, session, acquisition; None before any label."""

    def test_sorted_order(self):
        scans = [
            ScanId("sub-02"),
            ScanId("sub-01", "ses-02"),
            ScanId("sub-01", "ses-01", "acq-b"),
            ScanId("sub-01", "ses-01"),
            ScanId("sub-01", None, "acq-a"),
            ScanId("sub-01", "ses-01", "acq-a"),
            ScanId("sub-01"),
        ]
        assert [_fields(scan) for scan in sorted(scans)] == [
            ("sub-01", None, None),
            ("sub-01", None, "acq-a"),
            ("sub-01", "ses-01", None),
            ("sub-01", "ses-01", "acq-a"),
            ("sub-01", "ses-01", "acq-b"),
            ("sub-01", "ses-02", None),
            ("sub-02", None, None),
        ]


@pytest.mark.unit
def test_package_exports_scan_id():
    """REQ-STREAMING-BDRABCZUK-22A200032839: ``file_indexer.ScanId`` is the scan_id class."""
    assert file_indexer.ScanId is ScanId
