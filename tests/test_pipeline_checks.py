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


ACQ = "default_acquisition"


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _scan_dir(root: Path, subject: str, acquisition: str = ACQ) -> Path:
    """Morphologist's per-scan folder: <subjects_dir>/<subject>/t1mri/<acquisition>."""
    return root / subject / "t1mri" / acquisition


def _records(caplog, level: int) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == level]


# ---------------------------------------------------------------------------
# Scan listing in the Morphologist layout (both formats)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bids", [False, True])
def test_list_scans_one_scan_per_t1mri_acquisition_folder(tmp_path, bids):
    """REQ-STREAMING-BDRABCZUK-93331ADFDB36: one ScanId per <subject>/t1mri/<acquisition> folder, sorted."""
    for subject, acquisition in [
        ("sub-02", ACQ),
        ("sub-01", "ses-2"),
        ("sub-01", "ses-1"),
        ("sub-10", ACQ),
    ]:
        _touch(_scan_dir(tmp_path, subject, acquisition) / "default_analysis" / "folds" / "3.1" / "Lsub.arg")
    _touch(tmp_path / "README.txt")  # a file directly under subjects_dir is not a scan
    _touch(tmp_path / "sub-02" / "t1mri" / "notes.txt")  # a file under t1mri is not an acquisition

    checker = SubjectEligibilityChecker(str(tmp_path), [], bids=bids)

    assert checker.list_scans() == [
        ScanId("sub-01", "ses-1"),
        ScanId("sub-01", "ses-2"),
        ScanId("sub-02", ACQ),
        ScanId("sub-10", ACQ),
    ]


@pytest.mark.parametrize("bids", [False, True])
@pytest.mark.parametrize("acquisition", ["ses-1_run-1", "ses-V1_acq-T1w_run-2", "acq-MPRAGE", ACQ])
def test_list_scans_keeps_acquisition_folder_verbatim(tmp_path, bids, acquisition):
    """REQ-STREAMING-BDRABCZUK-93331ADFDB36: the acquisition folder name is never split (no ses- parsing)."""
    _scan_dir(tmp_path, "sub-01", acquisition).mkdir(parents=True)

    scans = SubjectEligibilityChecker(str(tmp_path), [], bids=bids).list_scans()

    assert [(scan.subject, scan.acquisition) for scan in scans] == [("sub-01", acquisition)]


def test_list_scans_ignores_path_to_graph(tmp_path):
    """REQ-STREAMING-BDRABCZUK-93331ADFDB36: scans come from the t1mri/<acquisition> folders, not path_to_graph."""
    _scan_dir(tmp_path, "sub-01", "ses-1").mkdir(parents=True)

    checker = SubjectEligibilityChecker(str(tmp_path), [], bids=True, path_to_graph="ses-*/anat/folds/*.arg")

    assert checker.list_scans() == [ScanId("sub-01", "ses-1")]


@pytest.mark.parametrize(
    ("bids", "expected_str"),
    [(False, "sub-01"), (True, "sub-01_ses-1_run-1")],
)
def test_list_scans_sets_bids_flag(tmp_path, bids, expected_str):
    """REQ-STREAMING-BDRABCZUK-4F5581B160B3: each listed ScanId carries the checker's bids value."""
    _scan_dir(tmp_path, "sub-01", "ses-1_run-1").mkdir(parents=True)
    _scan_dir(tmp_path, "sub-02", ACQ).mkdir(parents=True)

    scans = SubjectEligibilityChecker(str(tmp_path), [], bids=bids).list_scans()

    assert len(scans) == 2
    assert all(scan.bids is bids for scan in scans)
    assert str(scans[0]) == expected_str


@pytest.mark.parametrize("bids", [False, True])
@pytest.mark.parametrize(
    "layout",
    ["no_t1mri", "empty_t1mri", "file_only_t1mri", "session_folder_above_t1mri"],
)
def test_list_scans_subject_without_acquisition_folder_warns(tmp_path, caplog, bids, layout):
    """REQ-STREAMING-BDRABCZUK-4A7E3E59BB55: no t1mri/<acquisition> folder -> no ScanId, one WARNING naming it."""
    _scan_dir(tmp_path, "sub-01").mkdir(parents=True)
    bad = tmp_path / "sub-02"
    if layout == "no_t1mri":
        (bad / "anat").mkdir(parents=True)
    elif layout == "empty_t1mri":
        (bad / "t1mri").mkdir(parents=True)
    elif layout == "file_only_t1mri":
        _touch(bad / "t1mri" / "sub-02.nii.gz")
    else:  # a ses- folder is not a Morphologist level: the subject has no t1mri/<acquisition> folder
        (bad / "ses-1" / "t1mri" / ACQ).mkdir(parents=True)
    caplog.set_level(logging.DEBUG)

    scans = SubjectEligibilityChecker(str(tmp_path), [], bids=bids).list_scans()

    assert [scan.subject for scan in scans] == ["sub-01"]
    warnings = _records(caplog, logging.WARNING)
    assert len(warnings) == 1
    assert "sub-02" in warnings[0].getMessage()


def test_list_scans_warns_once_per_subject_without_acquisition_folder(tmp_path, caplog):
    """REQ-STREAMING-BDRABCZUK-4A7E3E59BB55: each subject folder without a scan gets its own WARNING."""
    _scan_dir(tmp_path, "subA").mkdir(parents=True)
    (tmp_path / "subB").mkdir()
    (tmp_path / "subC" / "t1mri").mkdir(parents=True)
    caplog.set_level(logging.DEBUG)

    scans = SubjectEligibilityChecker(str(tmp_path), []).list_scans()

    assert scans == [ScanId("subA", ACQ)]
    messages = [r.getMessage() for r in _records(caplog, logging.WARNING)]
    assert len(messages) == 2
    assert sum("subB" in m for m in messages) == 1
    assert sum("subC" in m for m in messages) == 1


def test_list_scans_warns_once_per_checker_instance(tmp_path, caplog):
    """REQ-STREAMING-BDRABCZUK-4A7E3E59BB55: later calls on one checker stay silent; a new checker warns again."""
    _scan_dir(tmp_path, "subA").mkdir(parents=True)
    (tmp_path / "subB").mkdir()
    caplog.set_level(logging.DEBUG)
    checker = SubjectEligibilityChecker(str(tmp_path), [])

    first = checker.list_scans()
    second = checker.list_scans()

    assert first == second == [ScanId("subA", ACQ)]
    warnings = _records(caplog, logging.WARNING)
    assert len(warnings) == 1
    assert "subB" in warnings[0].getMessage()

    caplog.clear()
    assert SubjectEligibilityChecker(str(tmp_path), []).list_scans() == [ScanId("subA", ACQ)]
    warnings = _records(caplog, logging.WARNING)
    assert len(warnings) == 1
    assert "subB" in warnings[0].getMessage()


def test_list_scans_bids_skips_non_subject_folders(tmp_path, caplog):
    """REQ-STREAMING-BDRABCZUK-1978C19F68DA: with bids, folders not starting with sub- are skipped, unwarned."""
    _scan_dir(tmp_path, "sub-001").mkdir(parents=True)
    _scan_dir(tmp_path, "derivatives").mkdir(parents=True)  # Morphologist-like, still not a subject
    for name in ["code", "sourcedata"]:
        (tmp_path / name).mkdir()
    caplog.set_level(logging.DEBUG)

    checker = SubjectEligibilityChecker(str(tmp_path), [], bids=True)

    assert checker.list_scans() == [ScanId("sub-001", ACQ)]
    assert _records(caplog, logging.WARNING) == []


# ---------------------------------------------------------------------------
# Hidden entries
# ---------------------------------------------------------------------------


def test_hidden_folder_is_not_a_scan(tmp_path):
    """REQ-STREAMING-BDRABCZUK-09D7FFE201DF: dot-folders as subject or acquisition are not listed."""
    _scan_dir(tmp_path, "sub-001").mkdir(parents=True)
    _scan_dir(tmp_path, ".snapshot").mkdir(parents=True)
    _scan_dir(tmp_path, "sub-001", ".tmp_acquisition").mkdir(parents=True)

    checker = SubjectEligibilityChecker(str(tmp_path), [])

    assert checker.list_scans() == [ScanId("sub-001", ACQ)]


def test_hidden_file_does_not_match(tmp_path):
    """REQ-STREAMING-BDRABCZUK-09D7FFE201DF: dot-files and files in dot-folders never match."""
    scan_dir = _scan_dir(tmp_path, "sub-001")
    _touch(scan_dir / "graphs" / ".Rsub.arg")
    _touch(scan_dir / ".cache" / "Rsub.arg")

    checker = SubjectEligibilityChecker(str(tmp_path), [])

    assert checker.scan_has_pattern(ScanId("sub-001", ACQ), "graphs/*.arg") is False
    assert checker.scan_has_pattern(ScanId("sub-001", ACQ), ".cache/Rsub.arg") is False


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
    _touch(_scan_dir(tmp_path, "sub-001", "ses-01") / "graphs" / "Rsub.arg")
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
    _touch(_scan_dir(tmp_path, "sub-001") / "graphs" / "Rsub.arg")
    checker = SubjectEligibilityChecker(str(tmp_path), [])

    assert checker.scan_has_pattern(ScanId("sub-001", ACQ), pattern) is expected


def test_scan_has_pattern_ignores_other_scans(tmp_path):
    """REQ-STREAMING-BDRABCZUK-6CAD6F696D4E: only files under the scan's own path_prefix count."""
    _touch(_scan_dir(tmp_path, "sub-10") / "graphs" / "Rsub.arg")  # shares the text prefix "sub-1"
    _touch(_scan_dir(tmp_path, "sub-2", "ses-02") / "graphs" / "Rsub.arg")
    _touch(_scan_dir(tmp_path, "sub-3", "ses-1_run-10") / "graphs" / "Rsub.arg")  # shares "ses-1_run-1"
    checker = SubjectEligibilityChecker(str(tmp_path), [])

    assert checker.scan_has_pattern(ScanId("sub-1", ACQ), "graphs/*.arg") is False
    assert checker.scan_has_pattern(ScanId("sub-2", "ses-01"), "graphs/*.arg") is False
    assert checker.scan_has_pattern(ScanId("sub-2", "ses-02"), "graphs/*.arg") is True
    assert checker.scan_has_pattern(ScanId("sub-3", "ses-1_run-1"), "graphs/*.arg") is False


@pytest.mark.parametrize("bids", [False, True])
def test_scan_has_pattern_resolves_under_t1mri_acquisition(tmp_path, bids):
    """REQ-STREAMING-BDRABCZUK-6CAD6F696D4E: the pattern is composed under <subject>/t1mri/<acquisition>."""
    _touch(tmp_path / "sub-01" / "ses-1" / "graphs" / "Rsub.arg")  # old subject/session layout: not the scan
    _touch(tmp_path / "sub-01" / "graphs" / "Rsub.arg")  # subject level: not the scan
    _touch(_scan_dir(tmp_path, "sub-01", "ses-1") / "skel" / "Rsub.nii.gz")
    checker = SubjectEligibilityChecker(str(tmp_path), [], bids=bids)
    scan = ScanId("sub-01", "ses-1", bids)

    assert checker.scan_has_pattern(scan, "graphs/*.arg") is False
    assert checker.scan_has_pattern(scan, "skel/*.nii.gz") is True


# ---------------------------------------------------------------------------
# Index refresh
# ---------------------------------------------------------------------------


def test_refresh_sees_files_created_after_first_query(tmp_path):
    """REQ-STREAMING-BDRABCZUK-19C641F39A2F: refresh() picks up a file that arrived later."""
    scan_dir = _scan_dir(tmp_path, "sub-001")
    scan_dir.mkdir(parents=True)
    checker = SubjectEligibilityChecker(str(tmp_path), [])
    scan = ScanId("sub-001", ACQ)
    assert checker.scan_has_pattern(scan, "skel/*.npy") is False

    _touch(scan_dir / "skel" / "S.C.-sylv._right.npy")
    checker.refresh()

    assert checker.scan_has_pattern(scan, "skel/*.npy") is True


def test_refresh_forgets_removed_files(tmp_path):
    """REQ-STREAMING-BDRABCZUK-19C641F39A2F: refresh() drops a file removed since the last index."""
    target = _scan_dir(tmp_path, "sub-001") / "skel" / "S.C.-sylv._right.npy"
    _touch(target)
    checker = SubjectEligibilityChecker(str(tmp_path), [])
    scan = ScanId("sub-001", ACQ)
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
        _touch(_scan_dir(tmp_path, sub) / "graphs" / "Rsub.arg")
        _touch(_scan_dir(tmp_path, sub) / "skel" / "Rsub.nii.gz")
    _touch(_scan_dir(tmp_path, "sub-004") / "graphs" / "Rsub.arg")  # skeleton missing

    report = SubjectEligibilityChecker(str(tmp_path), ["graphs/R*.arg", "skel/R*.nii.gz"], stage_name="tiles").check()

    assert isinstance(report, SubjectEligibilityReport)
    assert report.eligible == [ScanId("sub-001", ACQ), ScanId("sub-002", ACQ), ScanId("sub-003", ACQ)]
    assert report.stage_name == "tiles"
    assert report.input_dir == str(tmp_path)


def test_check_judges_each_acquisition_separately(tmp_path):
    """REQ-STREAMING-BDRABCZUK-9C039B129810: two acquisitions of one subject are two scans, judged apart."""
    _touch(_scan_dir(tmp_path, "sub-01", "ses-1") / "graphs" / "Rsub.arg")
    _scan_dir(tmp_path, "sub-01", "ses-2").mkdir(parents=True)

    report = SubjectEligibilityChecker(str(tmp_path), ["graphs/*.arg"], bids=True).check()

    assert report.eligible == [ScanId("sub-01", "ses-1")]
    assert report.ineligible == {ScanId("sub-01", "ses-2"): ["graphs/*.arg"]}


def test_check_reindexes_files_created_since_last_call(tmp_path):
    """REQ-STREAMING-BDRABCZUK-9C039B129810: check() judges on the files present when it runs."""
    _scan_dir(tmp_path, "sub-001").mkdir(parents=True)
    checker = SubjectEligibilityChecker(str(tmp_path), ["graphs/*.arg"])
    assert checker.check().eligible == []

    _touch(_scan_dir(tmp_path, "sub-001") / "graphs" / "Rsub.arg")

    assert checker.check().eligible == [ScanId("sub-001", ACQ)]


def test_check_empty_patterns_makes_each_scan_eligible(tmp_path):
    """REQ-STREAMING-BDRABCZUK-9C039B129810: no required pattern -> each listed scan is eligible."""
    _scan_dir(tmp_path, "sub-002").mkdir(parents=True)
    _scan_dir(tmp_path, "sub-001").mkdir(parents=True)

    report = SubjectEligibilityChecker(str(tmp_path), []).check()

    assert report.eligible == [ScanId("sub-001", ACQ), ScanId("sub-002", ACQ)]
    assert report.ineligible == {}


def test_check_maps_ineligible_scan_to_missing_patterns(tmp_path):
    """REQ-STREAMING-BDRABCZUK-E9404986A01E: ineligible scan -> its missing patterns, in given order."""
    patterns = ["skel/*.nii.gz", "graphs/*.arg", "labels/*.arg"]
    ok_dir = _scan_dir(tmp_path, "sub-ok")
    _touch(ok_dir / "skel" / "R.nii.gz")
    _touch(ok_dir / "graphs" / "R.arg")
    _touch(ok_dir / "labels" / "R.arg")
    _touch(_scan_dir(tmp_path, "sub-bad") / "graphs" / "R.arg")

    report = SubjectEligibilityChecker(str(tmp_path), patterns).check()

    assert report.ineligible == {ScanId("sub-bad", ACQ): ["skel/*.nii.gz", "labels/*.arg"]}
    assert ScanId("sub-ok", ACQ) in report.eligible


def test_check_saves_index_json_in_new_folder(tmp_path):
    """REQ-STREAMING-BDRABCZUK-62B33A216CFF: the index is saved as BTree.to_dict() JSON."""
    subjects = tmp_path / "subjects"
    _touch(_scan_dir(subjects, "sub-001") / "graphs" / "Rsub.arg")
    index_path = tmp_path / "reports" / "pre" / "index.json"

    SubjectEligibilityChecker(str(subjects), ["graphs/*.arg"]).check(save_index_to=str(index_path))

    tree = BTree.from_dict(json.loads(index_path.read_text()))
    assert [key for key, _ in tree] == [f"sub-001/t1mri/{ACQ}/graphs/Rsub.arg"]


# ---------------------------------------------------------------------------
# Report logging
# ---------------------------------------------------------------------------


def _report() -> SubjectEligibilityReport:
    return SubjectEligibilityReport(
        stage_name="cortical_tiles",
        input_dir="/data/subjects",
        eligible=[ScanId("sub-001", ACQ), ScanId("sub-002", ACQ)],
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
    report.ineligible[ScanId("sub-004", ACQ)] = ["graphs/R*.arg"]
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
    report = SubjectEligibilityReport("cortical_tiles", "/data/subjects", [ScanId("sub-001", ACQ)], {})
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
    _touch(_scan_dir(tmp_path, "sub-001") / "graphs" / "R..arg")

    checker = SubjectEligibilityChecker(str(tmp_path), ["graphs/R..arg"])

    assert checker.check().eligible == [ScanId("sub-001", ACQ)]


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
