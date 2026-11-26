"""SubjectEligibilityChecker.list_scans lists subjects through the shared non-subject rule."""

import logging
from pathlib import Path

import pytest

from champollion_pipeline.file_indexer import pipeline_checks
from champollion_pipeline.file_indexer.pipeline_checks import SubjectEligibilityChecker
from champollion_pipeline.file_indexer.scan_id import ScanId

pytestmark = pytest.mark.unit  # noqa: V107

ACQ = "default_acquisition"


def _scan_dir(root: Path, subject: str) -> Path:
    return root / subject / "t1mri" / ACQ


def test_qc_folder_without_t1mri_is_no_scan_and_no_warning(tmp_path, caplog):
    """REQ-INPUTKIND-BDRABCZUK-E2569E9B4669: a qc folder holding no t1mri folder is left out, unwarned."""
    _scan_dir(tmp_path, "sub-001").mkdir(parents=True)
    (tmp_path / "qc").mkdir()
    (tmp_path / "qc" / "report.tsv").write_text("x")
    caplog.set_level(logging.DEBUG)

    scans = SubjectEligibilityChecker(str(tmp_path), []).list_scans()

    assert scans == [ScanId("sub-001", ACQ)]
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_qc_folder_with_t1mri_and_no_graph_is_no_scan_and_no_warning(tmp_path, caplog):
    """REQ-INPUTKIND-BDRABCZUK-E2569E9B4669 (layout-free revision): a t1mri folder without a .arg file is no graph."""
    _scan_dir(tmp_path, "sub-001").mkdir(parents=True)
    (_scan_dir(tmp_path, "sub-001") / "Lsub-001.arg").write_text("x")
    _scan_dir(tmp_path, "qc").mkdir(parents=True)
    (tmp_path / "qc" / "report.tsv").write_text("x")
    caplog.set_level(logging.DEBUG)

    scans = SubjectEligibilityChecker(str(tmp_path), []).list_scans()

    assert scans == [ScanId("sub-001", ACQ)]
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_subjects_come_from_the_shared_lister(tmp_path, monkeypatch):
    """REQ-INPUTKIND-BDRABCZUK-B98700AD6446: list_scans takes the subject folders from list_morphologist_subjects."""
    _scan_dir(tmp_path, "subB").mkdir(parents=True)
    _scan_dir(tmp_path, "subA").mkdir(parents=True)
    calls = []

    def fake_lister(subjects_dir):
        calls.append(str(subjects_dir))
        return ["subA"]

    monkeypatch.setattr(pipeline_checks, "list_morphologist_subjects", fake_lister)

    scans = SubjectEligibilityChecker(str(tmp_path), []).list_scans()

    assert calls == [str(tmp_path)]
    assert scans == [ScanId("subA", ACQ)]


def test_unreadable_subjects_folder_keeps_the_oserror(tmp_path, monkeypatch):
    """REQ-INPUTKIND-BDRABCZUK-887AE48F139B: an OSError from listing the subjects folder reaches the caller."""

    def failing_lister(subjects_dir):
        raise PermissionError(13, "denied", str(subjects_dir))

    monkeypatch.setattr(pipeline_checks, "list_morphologist_subjects", failing_lister)

    with pytest.raises(OSError):
        SubjectEligibilityChecker(str(tmp_path), []).list_scans()
