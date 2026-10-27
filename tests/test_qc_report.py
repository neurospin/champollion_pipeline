#!/usr/bin/env python3
"""
Shared per-subject QC report (src/champollion_pipeline/utils/qc_report.py).

One tab-separated file per QC step: participant_id, qc (1 keep, 0 exclude),
optional extra columns, reason (empty when kept). The format stays readable by
cortical_tiles --sk_qc_path / prune_failed_subjects (participant_id, qc) and
matches generate_labelling_qc's column order.

Requirements (TASK-NTUI-BDRABCZUK-EF103D1CCE4E):
- REQ-NTUI-BDRABCZUK-A8325C762870 writer columns
- REQ-NTUI-BDRABCZUK-6D5EF9527726 writer refuses invalid rows
- REQ-NTUI-BDRABCZUK-54DD787A2C1E reader records
- REQ-NTUI-BDRABCZUK-36495992618F reader without reason column
- REQ-NTUI-BDRABCZUK-0384C824FAA2 reader refuses invalid files
- REQ-NTUI-BDRABCZUK-42D979F31779 summary counts
- REQ-NTUI-BDRABCZUK-F81C30964784 no toolchain import
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from champollion_pipeline.generate_labelling_qc import QC_COLUMNS
from champollion_pipeline.utils import qc_report

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def _lines(path: Path) -> list:
    return path.read_text().splitlines()


def _write_text(path: Path, lines: list) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# REQ-NTUI-BDRABCZUK-A8325C762870: writer columns
# ---------------------------------------------------------------------------


def test_write_header_and_rows(tmp_path):
    out = tmp_path / "qc.tsv"
    rows = [
        {"participant_id": "sub-01", "qc": 1, "reason": ""},
        {"participant_id": "sub-02", "qc": 0, "reason": "missing_L"},
        {"participant_id": "sub-03", "qc": 1},
    ]
    qc_report.write_qc_tsv(out, rows)
    assert _lines(out) == [
        "participant_id\tqc\treason",
        "sub-01\t1\t",
        "sub-02\t0\tmissing_L",
        "sub-03\t1\t",
    ]


def test_write_matches_labelling_qc_column_order(tmp_path):
    out = tmp_path / "qc.tsv"
    rows = [{"participant_id": "s01", "qc": 1, "labelled_L": 1, "labelled_R": 1, "reason": ""}]
    qc_report.write_qc_tsv(out, rows, extra_columns=("labelled_L", "labelled_R"))
    assert tuple(_lines(out)[0].split("\t")) == QC_COLUMNS
    assert _lines(out)[1] == "s01\t1\t1\t1\t"


def test_write_bool_qc_as_digit(tmp_path):
    out = tmp_path / "qc.tsv"
    rows = [
        {"participant_id": "a", "qc": True, "reason": ""},
        {"participant_id": "b", "qc": False, "reason": "bad"},
    ]
    qc_report.write_qc_tsv(out, rows)
    assert [line.split("\t")[1] for line in _lines(out)[1:]] == ["1", "0"]


# ---------------------------------------------------------------------------
# REQ-NTUI-BDRABCZUK-6D5EF9527726: writer refuses invalid rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param([{"qc": 1, "reason": ""}], id="no_participant_id"),
        pytest.param([{"participant_id": "", "qc": 1, "reason": ""}], id="empty_participant_id"),
        pytest.param(
            [{"participant_id": "s01", "qc": 1}, {"participant_id": "s01", "qc": 0, "reason": "x"}],
            id="repeated_participant_id",
        ),
        pytest.param([{"participant_id": "s01", "qc": 2}], id="qc_2"),
        pytest.param([{"participant_id": "s01", "qc": "yes"}], id="qc_text"),
        pytest.param([{"participant_id": "s01"}], id="qc_missing"),
    ],
)
def test_write_rejects_invalid_rows(tmp_path, rows):
    out = tmp_path / "qc.tsv"
    with pytest.raises(ValueError):
        qc_report.write_qc_tsv(out, rows)
    assert not out.exists()


# ---------------------------------------------------------------------------
# REQ-NTUI-BDRABCZUK-54DD787A2C1E: reader records
# ---------------------------------------------------------------------------


def test_read_records(tmp_path):
    path = _write_text(
        tmp_path / "qc.tsv",
        ["participant_id\tqc\tlabelled_L\treason", "s01\t1\t1\t", "s02\t0\t0\tmissing_L"],
    )
    assert qc_report.read_qc_tsv(path) == [
        {"participant_id": "s01", "qc": 1, "labelled_L": "1", "reason": ""},
        {"participant_id": "s02", "qc": 0, "labelled_L": "0", "reason": "missing_L"},
    ]


def test_read_keeps_leading_zeros(tmp_path):
    path = _write_text(tmp_path / "qc.tsv", ["participant_id\tqc\treason", "0001\t1\t", "1e5\t0\tx"])
    records = qc_report.read_qc_tsv(path)
    assert [r["participant_id"] for r in records] == ["0001", "1e5"]
    assert all(type(r["qc"]) is int for r in records)


def test_round_trip(tmp_path):
    out = tmp_path / "qc.tsv"
    rows = [
        {"participant_id": "007", "qc": 1, "reason": ""},
        {"participant_id": "sub-02", "qc": 0, "reason": "no_session_dir"},
    ]
    qc_report.write_qc_tsv(out, rows)
    assert qc_report.read_qc_tsv(out) == rows


# ---------------------------------------------------------------------------
# REQ-NTUI-BDRABCZUK-36495992618F: reader without reason column
# ---------------------------------------------------------------------------


def test_read_sk_qc_format_without_reason(tmp_path):
    path = _write_text(tmp_path / "sk_qc.tsv", ["participant_id\tqc", "s01\t1", "s02\t0"])
    assert qc_report.read_qc_tsv(path) == [
        {"participant_id": "s01", "qc": 1, "reason": ""},
        {"participant_id": "s02", "qc": 0, "reason": ""},
    ]


# ---------------------------------------------------------------------------
# REQ-NTUI-BDRABCZUK-0384C824FAA2: reader refuses invalid files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lines",
    [
        pytest.param(["subject\tqc", "s01\t1"], id="no_participant_id_column"),
        pytest.param(["participant_id\treason", "s01\t"], id="no_qc_column"),
        pytest.param(["participant_id\tqc", "s01\t1", "s01\t0"], id="repeated_participant_id"),
        pytest.param(["participant_id\tqc", "s01\t2"], id="qc_2"),
        pytest.param(["participant_id\tqc", "s01\t"], id="qc_empty"),
        pytest.param(["participant_id\tqc", "s01\tok"], id="qc_text"),
    ],
)
def test_read_rejects_invalid_files(tmp_path, lines):
    path = _write_text(tmp_path / "bad_qc.tsv", lines)
    with pytest.raises(ValueError, match=str(path)):
        qc_report.read_qc_tsv(path)


# ---------------------------------------------------------------------------
# REQ-NTUI-BDRABCZUK-42D979F31779: summary counts
# ---------------------------------------------------------------------------


def test_summarise_counts():
    records = [
        {"participant_id": "a", "qc": 1, "reason": ""},
        {"participant_id": "b", "qc": 0, "reason": "missing_L"},
        {"participant_id": "c", "qc": 0, "reason": "missing_L"},
        {"participant_id": "d", "qc": 0, "reason": "no_session_dir"},
        {"participant_id": "e", "qc": 0, "reason": ""},
    ]
    assert qc_report.summarise_qc(records) == {
        "n_subjects": 5,
        "n_excluded": 4,
        "excluded_per_reason": {"missing_L": 2, "no_session_dir": 1},
    }


# ---------------------------------------------------------------------------
# REQ-NTUI-BDRABCZUK-F81C30964784: no toolchain import
# ---------------------------------------------------------------------------


def test_write_and_read_without_toolchain(tmp_path):
    """conftest stubs soma in-process, so this runs in a child where the toolchain is blocked."""
    out = tmp_path / "qc.tsv"
    code = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(_SRC_DIR)!r})
        for name in ("soma", "soma.aims", "cortical_tiles", "champollion_V1"):
            sys.modules[name] = None
        import champollion_utils.script_builder as _sb
        _sb.check_for_updates = lambda *a, **k: None
        from champollion_pipeline.utils import qc_report
        qc_report.write_qc_tsv({str(out)!r}, [{{"participant_id": "s01", "qc": 0, "reason": "x"}}])
        print(qc_report.read_qc_tsv({str(out)!r}))
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str([{"participant_id": "s01", "qc": 0, "reason": "x"}])
