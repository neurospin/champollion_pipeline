"""LOGGING convention for generate_labelling_qc.py (TASK-LOGGING-BDRABCZUK-940DB0BFBA89).

GenerateLabellingQC reports its summary through the champollion_pipeline.generate_labelling_qc
logger instead of print().

REQ-LOGGING-BDRABCZUK-27D3F29D9692: the row total, qc=1 and qc=0 counts are logged at OUTPUT.
REQ-LOGGING-BDRABCZUK-7620BB6E838A: each distinct non-empty reason and its row count are logged at OUTPUT.
(REQ-LOGGING-BDRABCZUK-C96C8DAAF703 and -22ABEF2F2CBD: tests/test_logging_convention.py, through
LOG_CONVERTED_MODULES.)
"""

import re
from pathlib import Path

import pytest

from champollion_pipeline.generate_labelling_qc import GenerateLabellingQC
from tests.log_assertions import OUTPUT

pytestmark = pytest.mark.unit  # noqa: V107

LOGGER_NAME = "champollion_pipeline.generate_labelling_qc"
SESSION = "deepcnn_session_auto"
CLASSIC_PATH = "t1mri/ses-V1/default_analysis/folds/3.1"


def _subject(subjects: Path, subject: str, sides: tuple[str, ...]) -> None:
    """One subject whose labelling session folder holds the labelled graphs of ``sides``."""
    session_dir = subjects / subject / CLASSIC_PATH / SESSION
    session_dir.mkdir(parents=True)
    for side in sides:
        (session_dir / f"{side}{subject}_{SESSION}.arg").write_text("graph")


def _output_messages(capture, text: str) -> list[str]:
    """Messages of the OUTPUT records of the generate_labelling_qc logger that contain ``text``, each once."""
    return [
        record.getMessage()
        for record in capture.records
        if record.name == LOGGER_NAME and record.levelno == OUTPUT and text in record.getMessage()
    ]


def _run(tmp_path: Path) -> None:
    """Three subjects: s01 labelled, s02 and s03 missing their right hemisphere graph."""
    subjects = tmp_path / "subjects"
    subjects.mkdir()
    _subject(subjects, "s01", ("L", "R"))
    _subject(subjects, "s02", ("L",))
    _subject(subjects, "s03", ("L",))
    script = GenerateLabellingQC()
    script.parse_args([str(subjects), str(tmp_path / "qc" / "labelling_qc.tsv"), "--path_to_graph", CLASSIC_PATH])
    assert script.run() == 0


def test_qc_totals_logged_at_output(tmp_path, pipeline_caplog):
    _run(tmp_path)

    totals = _output_messages(pipeline_caplog, "Total:")
    assert len(totals) == 1, totals
    assert re.search(r"Total:\s*3\b.*qc=1:\s*1\b.*qc=0:\s*2\b", totals[0]), totals[0]


def test_reason_count_logged_at_output(tmp_path, pipeline_caplog):
    _run(tmp_path)

    reasons = _output_messages(pipeline_caplog, "missing_R")
    assert len(reasons) == 1, reasons
    assert re.search(r"missing_R\D*\b2\b", reasons[0]), reasons[0]
