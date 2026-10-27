#!/usr/bin/env python3
"""
Tests for ``champollion_pipeline.file_indexer.pipeline_checks``.

SubjectEligibilityChecker indexes a subjects directory and answers, per scan,
whether the files a stage needs are present. The streaming arrival barrier
polls it through the public API (list_scans, scan_has_pattern, refresh), and
the sequential pre-stage report uses check(). Paths and patterns come from the
user's configuration, so they are checked, never trusted.

Synthetic folders only (tmp_path); no real subject data.
"""

import json
import logging
import re
from pathlib import Path

import pytest
from champollion_utils.logger import OUTPUT

from champollion_pipeline import file_indexer
from champollion_pipeline.file_indexer.btree import BTree
from champollion_pipeline.file_indexer.pipeline_checks import (
    SubjectEligibilityChecker,
    SubjectEligibilityReport,
)
from champollion_pipeline.file_indexer.scan_id import ScanId

pytestmark = pytest.mark.unit  # noqa: V107


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _records(caplog, level: int) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == level]


# ---------------------------------------------------------------------------
# Scan listing without BIDS
# ---------------------------------------------------------------------------


def test_list_scans_non_bids_one_scan_per_subject_folder(tmp_path):
    """REQ-STREAMING-BDRABCZUK-9B4C5A0703A3: one subject-only ScanId per folder, sorted."""
    for name in ["subB", "subA", "sub-10"]:
        (tmp_path / name / "ses-01").mkdir(parents=True)
    _touch(tmp_path / "README.txt")  # a file directly under subjects_dir is not a scan

    checker = SubjectEligibilityChecker(str(tmp_path), [])

    assert checker.list_scans() == [ScanId("sub-10"), ScanId("subA"), ScanId("subB")]


# ---------------------------------------------------------------------------
# Scan listing with BIDS
# ---------------------------------------------------------------------------


def test_list_scans_bids_sessions_from_path_to_graph(tmp_path):
    """REQ-STREAMING-BDRABCZUK-CBDDED39FBE8: one ScanId per path_to_graph match, session from ses-."""
    for sub, ses in [("sub-002", "ses-01"), ("sub-001", "ses-02"), ("sub-001", "ses-01")]:
        (tmp_path / sub / ses / "t1mri" / "folds").mkdir(parents=True)

    checker = SubjectEligibilityChecker(str(tmp_path), [], bids=True, path_to_graph="ses-*/t1mri/folds")

    assert checker.list_scans() == [
        ScanId("sub-001", "ses-01"),
        ScanId("sub-001", "ses-02"),
        ScanId("sub-002", "ses-01"),
    ]


def test_list_scans_bids_acquisitions_from_path_to_graph(tmp_path):
    """REQ-STREAMING-BDRABCZUK-CBDDED39FBE8: acquisition from acq-, duplicates collapsed."""
    ses = tmp_path / "sub-001" / "ses-01" / "t1mri"
    for acq in ["acq-T1w", "acq-MPRAGE"]:
        _touch(ses / acq / "folds" / "Lsub.arg")
        _touch(ses / acq / "folds" / "Rsub.arg")  # second match of the same scan

    checker = SubjectEligibilityChecker(str(tmp_path), [], bids=True, path_to_graph="ses-*/t1mri/acq-*/folds/*.arg")

    assert checker.list_scans() == [
        ScanId("sub-001", "ses-01", "acq-MPRAGE"),
        ScanId("sub-001", "ses-01", "acq-T1w"),
    ]


def test_list_scans_bids_falls_back_to_session_folders(tmp_path):
    """REQ-STREAMING-BDRABCZUK-06A514178952: no path_to_graph match -> one ScanId per ses- folder."""
    (tmp_path / "sub-001" / "ses-01" / "t1mri" / "folds").mkdir(parents=True)
    (tmp_path / "sub-002" / "ses-02").mkdir(parents=True)
    (tmp_path / "sub-002" / "ses-01").mkdir(parents=True)
    (tmp_path / "sub-002" / "anat").mkdir(parents=True)  # not a ses- folder

    checker = SubjectEligibilityChecker(str(tmp_path), [], bids=True, path_to_graph="ses-*/t1mri/folds")

    assert checker.list_scans() == [
        ScanId("sub-001", "ses-01"),
        ScanId("sub-002", "ses-01"),
        ScanId("sub-002", "ses-02"),
    ]


def test_list_scans_bids_falls_back_to_subject_only(tmp_path):
    """REQ-STREAMING-BDRABCZUK-073C6D88F453: no match and no ses- folder -> subject-only ScanId."""
    (tmp_path / "sub-001" / "anat").mkdir(parents=True)

    checker = SubjectEligibilityChecker(str(tmp_path), [], bids=True, path_to_graph="ses-*/t1mri/folds")

    assert checker.list_scans() == [ScanId("sub-001")]


def test_list_scans_bids_skips_non_subject_folders(tmp_path):
    """REQ-STREAMING-BDRABCZUK-1978C19F68DA: folders not starting with sub- are not scans."""
    for name in ["sub-001", "derivatives", "code", "sourcedata"]:
        (tmp_path / name).mkdir()

    checker = SubjectEligibilityChecker(str(tmp_path), [], bids=True)

    assert checker.list_scans() == [ScanId("sub-001")]


# ---------------------------------------------------------------------------
# Hidden entries
# ---------------------------------------------------------------------------


def test_hidden_folder_is_not_a_scan(tmp_path):
    """REQ-STREAMING-BDRABCZUK-09D7FFE201DF: a dot-folder under subjects_dir is not listed."""
    (tmp_path / "sub-001").mkdir()
    (tmp_path / ".snapshot").mkdir()

    checker = SubjectEligibilityChecker(str(tmp_path), [])

    assert checker.list_scans() == [ScanId("sub-001")]


def test_hidden_file_does_not_match(tmp_path):
    """REQ-STREAMING-BDRABCZUK-09D7FFE201DF: dot-files and files in dot-folders never match."""
    _touch(tmp_path / "sub-001" / "graphs" / ".Rsub.arg")
    _touch(tmp_path / "sub-001" / ".cache" / "Rsub.arg")

    checker = SubjectEligibilityChecker(str(tmp_path), [])

    assert checker.scan_has_pattern(ScanId("sub-001"), "graphs/*.arg") is False
    assert checker.scan_has_pattern(ScanId("sub-001"), ".cache/Rsub.arg") is False


# ---------------------------------------------------------------------------
# Missing subjects_dir
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["absent", "regular_file"])
def test_list_scans_missing_subjects_dir_warns_and_returns_empty(tmp_path, caplog, kind):
    """REQ-STREAMING-BDRABCZUK-F79B400747C3: no directory -> [] and one WARNING naming it."""
    subjects_dir = tmp_path / "subjects"
    if kind == "regular_file":
        _touch(subjects_dir)
    caplog.set_level(logging.DEBUG)

    checker = SubjectEligibilityChecker(str(subjects_dir), ["graphs/*.arg"])
    scans = checker.list_scans()

    assert scans == []
    warnings = _records(caplog, logging.WARNING)
    assert len(warnings) == 1
    assert str(subjects_dir) in warnings[0].getMessage()


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------


def test_scan_has_pattern_exact_path(tmp_path):
    """REQ-STREAMING-BDRABCZUK-6CAD6F696D4E: a pattern without wildcard matches one exact path."""
    _touch(tmp_path / "sub-001" / "ses-01" / "graphs" / "Rsub.arg")
    checker = SubjectEligibilityChecker(str(tmp_path), [])
    scan = ScanId("sub-001", "ses-01")

    assert checker.scan_has_pattern(scan, "graphs/Rsub.arg") is True
    assert checker.scan_has_pattern(scan, "graphs/Lsub.arg") is False


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("graphs/R*.arg", True),
        ("graphs/R?ub.arg", True),
        ("graphs/[LR]sub.arg", True),
        ("graphs/[!R]sub.arg", False),
        ("skel/*.nii.gz", False),
    ],
)
def test_scan_has_pattern_wildcards(tmp_path, pattern, expected):
    """REQ-STREAMING-BDRABCZUK-6CAD6F696D4E: fnmatch wildcards *, ? and [seq]."""
    _touch(tmp_path / "sub-001" / "graphs" / "Rsub.arg")
    checker = SubjectEligibilityChecker(str(tmp_path), [])

    assert checker.scan_has_pattern(ScanId("sub-001"), pattern) is expected


def test_scan_has_pattern_ignores_other_scans(tmp_path):
    """REQ-STREAMING-BDRABCZUK-6CAD6F696D4E: only files under the scan's own path_prefix count."""
    _touch(tmp_path / "sub-10" / "graphs" / "Rsub.arg")  # shares the text prefix "sub-1"
    _touch(tmp_path / "sub-2" / "ses-02" / "graphs" / "Rsub.arg")
    checker = SubjectEligibilityChecker(str(tmp_path), [])

    assert checker.scan_has_pattern(ScanId("sub-1"), "graphs/*.arg") is False
    assert checker.scan_has_pattern(ScanId("sub-2", "ses-01"), "graphs/*.arg") is False
    assert checker.scan_has_pattern(ScanId("sub-2", "ses-02"), "graphs/*.arg") is True


# ---------------------------------------------------------------------------
# Index refresh
# ---------------------------------------------------------------------------


def test_refresh_sees_files_created_after_first_query(tmp_path):
    """REQ-STREAMING-BDRABCZUK-19C641F39A2F: refresh() picks up a file that arrived later."""
    (tmp_path / "sub-001").mkdir()
    checker = SubjectEligibilityChecker(str(tmp_path), [])
    scan = ScanId("sub-001")
    assert checker.scan_has_pattern(scan, "skel/*.npy") is False

    _touch(tmp_path / "sub-001" / "skel" / "S.C.-sylv._right.npy")
    checker.refresh()

    assert checker.scan_has_pattern(scan, "skel/*.npy") is True


def test_refresh_forgets_removed_files(tmp_path):
    """REQ-STREAMING-BDRABCZUK-19C641F39A2F: refresh() drops a file removed since the last index."""
    target = tmp_path / "sub-001" / "skel" / "S.C.-sylv._right.npy"
    _touch(target)
    checker = SubjectEligibilityChecker(str(tmp_path), [])
    scan = ScanId("sub-001")
    assert checker.scan_has_pattern(scan, "skel/*.npy") is True

    target.unlink()
    checker.refresh()

    assert checker.scan_has_pattern(scan, "skel/*.npy") is False


# ---------------------------------------------------------------------------
# Eligibility check
# ---------------------------------------------------------------------------


def test_check_lists_eligible_scans(tmp_path):
    """REQ-STREAMING-BDRABCZUK-9C039B129810: eligible = scans matching each pattern, sorted."""
    for sub in ["sub-003", "sub-001", "sub-002"]:
        _touch(tmp_path / sub / "graphs" / "Rsub.arg")
        _touch(tmp_path / sub / "skel" / "Rsub.nii.gz")
    _touch(tmp_path / "sub-004" / "graphs" / "Rsub.arg")  # skeleton missing

    report = SubjectEligibilityChecker(str(tmp_path), ["graphs/R*.arg", "skel/R*.nii.gz"], stage_name="tiles").check()

    assert isinstance(report, SubjectEligibilityReport)
    assert report.eligible == [ScanId("sub-001"), ScanId("sub-002"), ScanId("sub-003")]
    assert report.stage_name == "tiles"
    assert report.input_dir == str(tmp_path)


def test_check_reindexes_files_created_since_last_call(tmp_path):
    """REQ-STREAMING-BDRABCZUK-9C039B129810: check() judges on the files present when it runs."""
    (tmp_path / "sub-001").mkdir()
    checker = SubjectEligibilityChecker(str(tmp_path), ["graphs/*.arg"])
    assert checker.check().eligible == []

    _touch(tmp_path / "sub-001" / "graphs" / "Rsub.arg")

    assert checker.check().eligible == [ScanId("sub-001")]


def test_check_empty_patterns_makes_each_scan_eligible(tmp_path):
    """REQ-STREAMING-BDRABCZUK-9C039B129810: no required pattern -> each listed scan is eligible."""
    (tmp_path / "sub-002").mkdir()
    (tmp_path / "sub-001").mkdir()

    report = SubjectEligibilityChecker(str(tmp_path), []).check()

    assert report.eligible == [ScanId("sub-001"), ScanId("sub-002")]
    assert report.ineligible == {}


def test_check_maps_ineligible_scan_to_missing_patterns(tmp_path):
    """REQ-STREAMING-BDRABCZUK-E9404986A01E: ineligible scan -> its missing patterns, in given order."""
    patterns = ["skel/*.nii.gz", "graphs/*.arg", "labels/*.arg"]
    _touch(tmp_path / "sub-ok" / "skel" / "R.nii.gz")
    _touch(tmp_path / "sub-ok" / "graphs" / "R.arg")
    _touch(tmp_path / "sub-ok" / "labels" / "R.arg")
    _touch(tmp_path / "sub-bad" / "graphs" / "R.arg")

    report = SubjectEligibilityChecker(str(tmp_path), patterns).check()

    assert report.ineligible == {ScanId("sub-bad"): ["skel/*.nii.gz", "labels/*.arg"]}
    assert ScanId("sub-ok") in report.eligible


def test_check_saves_index_json_in_new_folder(tmp_path):
    """REQ-STREAMING-BDRABCZUK-62B33A216CFF: the index is saved as BTree.to_dict() JSON."""
    subjects = tmp_path / "subjects"
    _touch(subjects / "sub-001" / "graphs" / "Rsub.arg")
    index_path = tmp_path / "reports" / "pre" / "index.json"

    SubjectEligibilityChecker(str(subjects), ["graphs/*.arg"]).check(save_index_to=str(index_path))

    tree = BTree.from_dict(json.loads(index_path.read_text()))
    assert [key for key, _ in tree] == ["sub-001/graphs/Rsub.arg"]


# ---------------------------------------------------------------------------
# Report logging
# ---------------------------------------------------------------------------


def _report() -> SubjectEligibilityReport:
    return SubjectEligibilityReport(
        stage_name="cortical_tiles",
        input_dir="/data/subjects",
        eligible=[ScanId("sub-001"), ScanId("sub-002")],
        ineligible={ScanId("sub-003", "ses-01"): ["graphs/R*.arg", "skel/R*.nii.gz"]},
    )


def test_log_summary_output_record(caplog):
    """REQ-STREAMING-BDRABCZUK-4D7FF4E1847F: one OUTPUT record with stage, dir, total, eligible."""
    caplog.set_level(logging.DEBUG)

    _report().log_summary()

    outputs = _records(caplog, OUTPUT)
    assert len(outputs) == 1
    message = outputs[0].getMessage()
    assert "cortical_tiles" in message
    assert "/data/subjects" in message
    assert re.search(r"\b3\b", message), "total scan count missing"
    assert re.search(r"\b2\b", message), "eligible scan count missing"


def test_log_summary_warns_on_ineligible_scans(caplog):
    """REQ-STREAMING-BDRABCZUK-B18CC0140E15: one WARNING naming each ineligible scan and its patterns."""
    report = _report()
    report.ineligible[ScanId("sub-004")] = ["graphs/R*.arg"]
    caplog.set_level(logging.DEBUG)

    report.log_summary()

    warnings = _records(caplog, logging.WARNING)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    for scan, missing in report.ineligible.items():
        assert str(scan) in message
        for pattern in missing:
            assert pattern in message


def test_log_summary_no_warning_when_all_eligible(caplog):
    """REQ-STREAMING-BDRABCZUK-B18CC0140E15: no ineligible scan -> no WARNING."""
    report = SubjectEligibilityReport("cortical_tiles", "/data/subjects", [ScanId("sub-001")], {})
    caplog.set_level(logging.DEBUG)

    report.log_summary()

    assert _records(caplog, OUTPUT), "summary record missing"
    assert _records(caplog, logging.WARNING) == []


# ---------------------------------------------------------------------------
# Constructor input checks (never trust user input)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", ["/etc/passwd", "/sub-001/graphs/*.arg"])
def test_constructor_rejects_absolute_pattern(tmp_path, pattern):
    """REQ-STREAMING-BDRABCZUK-CFDD2C1A400A: an absolute pattern raises ValueError."""
    with pytest.raises(ValueError, match=re.escape(pattern)):
        SubjectEligibilityChecker(str(tmp_path), ["graphs/*.arg", pattern])


@pytest.mark.parametrize("pattern", ["../sub-002/graphs/*.arg", "graphs/../../x.arg", ".."])
def test_constructor_rejects_parent_segment_pattern(tmp_path, pattern):
    """REQ-STREAMING-BDRABCZUK-CFDD2C1A400A: a '..' path segment raises ValueError."""
    with pytest.raises(ValueError, match=re.escape(pattern)):
        SubjectEligibilityChecker(str(tmp_path), [pattern])


def test_constructor_accepts_dots_inside_a_name(tmp_path):
    """REQ-STREAMING-BDRABCZUK-CFDD2C1A400A: '..' inside a file name is not a path segment."""
    _touch(tmp_path / "sub-001" / "graphs" / "R..arg")

    checker = SubjectEligibilityChecker(str(tmp_path), ["graphs/R..arg"])

    assert checker.check().eligible == [ScanId("sub-001")]


def test_constructor_rejects_string_patterns(tmp_path):
    """REQ-STREAMING-BDRABCZUK-5B4BF83CA4E0: a bare str for required_patterns raises TypeError."""
    with pytest.raises(TypeError):
        SubjectEligibilityChecker(str(tmp_path), "graphs/*.arg")


# ---------------------------------------------------------------------------
# Package export
# ---------------------------------------------------------------------------


def test_package_exports_checker():
    """REQ-STREAMING-BDRABCZUK-B3E8FAD249FC: file_indexer exports SubjectEligibilityChecker."""
    assert file_indexer.SubjectEligibilityChecker is SubjectEligibilityChecker
