"""LOGGING convention for prune_failed_subjects.py (TASK-LOGGING-BDRABCZUK-1761ECDBE832).

PruneFailedSubjects reports through the champollion_pipeline.prune_failed_subjects logger
instead of print(). The listings and the summary are results (OUTPUT, on stdout), so the
champollion_sulcal_mcp job log keeps them.

REQ-LOGGING-BDRABCZUK-F71343498890: the IDs of the subjects to remove are logged at OUTPUT.
REQ-LOGGING-BDRABCZUK-769B1F0E5F85: without --dry-run, each deleted file is logged at OUTPUT after Removed:.
REQ-LOGGING-BDRABCZUK-0DDCDCD2FD48: the Summary: line with the subject and file counts is logged at OUTPUT.
(REQ-LOGGING-BDRABCZUK-09E839CCB79E and -728A1E8B1120: tests/test_prune_failed_subjects.py,
TestRun::test_no_subjects_in_crops_returns_early and ::test_dry_run_output_mentions_would_delete.
REQ-LOGGING-BDRABCZUK-C96C8DAAF703 and -22ABEF2F2CBD: tests/test_logging_convention.py, through
LOG_CONVERTED_MODULES.)

Synthetic subject IDs and a synthetic derivatives tree only.
"""

import re
from pathlib import Path

import pytest

from champollion_pipeline.prune_failed_subjects import PruneFailedSubjects
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from tests.log_assertions import OUTPUT

pytestmark = pytest.mark.unit  # noqa: V107

LOGGER_NAME = "champollion_pipeline.prune_failed_subjects"


def _tree(tmp_path: Path, subjects: list[str], qc_rows: list[tuple[str, int]]) -> Path:
    """Crops of region_a for ``subjects`` and a QC TSV of ``qc_rows``; return the QC path."""
    region_dir = tmp_path / DERIVATIVES_FOLDER / "crops" / "2mm" / "region_a"
    region_dir.mkdir(parents=True)
    for subject in subjects:
        (region_dir / f"{subject}.nii.gz").touch()
    qc = tmp_path / "qc.tsv"
    qc.write_text("participant_id\tqc\n" + "\n".join(f"{s}\t{q}" for s, q in qc_rows))
    return qc


def _run(tmp_path: Path, qc: Path, dry_run: bool = False) -> None:
    script = PruneFailedSubjects()
    script.args = script.parse_args([str(tmp_path), "--qc", str(qc)] + (["--dry-run"] if dry_run else []))
    assert script.run() == 0


def test_subjects_to_remove_logged_at_output(tmp_path, pipeline_caplog):
    qc = _tree(tmp_path, ["sub-01", "sub-02", "sub-03"], [("sub-01", 1), ("sub-02", 0)])

    _run(tmp_path, qc, dry_run=True)

    records = [
        record
        for record in pipeline_caplog.records
        if record.levelno == OUTPUT and "sub-02" in record.getMessage() and "sub-03" in record.getMessage()
    ]
    assert records, [(record.levelname, record.getMessage()) for record in pipeline_caplog.records]
    assert records[0].name == LOGGER_NAME


def test_removed_file_logged_at_output(tmp_path, pipeline_caplog):
    qc = _tree(tmp_path, ["sub-01", "sub-02"], [("sub-01", 1), ("sub-02", 0)])
    removed = tmp_path / DERIVATIVES_FOLDER / "crops" / "2mm" / "region_a" / "sub-02.nii.gz"

    _run(tmp_path, qc)

    assert not removed.exists()
    record = pipeline_caplog.assert_logged(OUTPUT, f"Removed: {removed}")
    assert record.name == LOGGER_NAME


def test_summary_logged_at_output(tmp_path, pipeline_caplog):
    qc = _tree(tmp_path, ["sub-01", "sub-02"], [("sub-01", 1), ("sub-02", 0)])
    skeleton_dir = tmp_path / DERIVATIVES_FOLDER / "skeletons" / "2mm" / "region_a"
    skeleton_dir.mkdir(parents=True)
    (skeleton_dir / "sub-02.nii.gz").touch()

    _run(tmp_path, qc)

    # One subject removed, two of its files deleted.
    record = pipeline_caplog.assert_logged(OUTPUT, "Summary:")
    summary = record.getMessage()
    assert re.search(r"Summary:\D*\b1\b\D+\b2\b", summary), summary
    assert record.name == LOGGER_NAME
