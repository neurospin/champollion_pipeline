#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV output reporting of the src/compare.py databases subcommand.

Requirements: REQ-COMPARE-78 (no sulcus loaded: stdout carries a
"No sulcus rows found" message naming both campaign labels), REQ-COMPARE-79
(no sulcus loaded: no "CSV written to" line), REQ-COMPARE-80 (no sulcus
loaded: no file is created at --output), REQ-COMPARE-81 (no sulcus loaded:
exit status 0), REQ-COMPARE-82 (at least one sulcus loaded: the CSV is
written and stdout names its absolute --output path).

``Compare._load_database`` is replaced by a fake serving in-memory
{subject: {sulcus: voxel_count}} dicts, so neither soma nor graph files are
needed; PyAIMS is stubbed in conftest.py.
"""

import csv
from os.path import abspath

import pytest

from compare import Compare

_LABEL_A = "campaignAlpha"
_LABEL_B = "campaignBeta"
_GRAPH_A = "graph_a"
_GRAPH_B = "graph_b"
_NO_ROWS = "No sulcus rows found"
_CSV_WRITTEN = "CSV written to"

# Two ways a campaign pair yields no sulcus row: no subject at all, or
# subjects whose graphs carry no labelled sulcus.
_EMPTY_CASES = {
    "no_subjects": ({}, {}),
    "subjects_without_sulci": ({"sub-01": {}}, {"sub-02": {}}),
}

_NON_EMPTY = (
    {"sub-01": {"S.C._left": 100, "F.C.M._left": 40}},
    {"sub-01": {"S.C._left": 120}},
)


def _run_databases(monkeypatch, data_a, data_b, output):
    """Run the databases subcommand with a fake loader; return its exit status."""
    by_graph = {_GRAPH_A: data_a, _GRAPH_B: data_b}

    def _fake_load_database(_self, path_to_graph, _sides, _subjects_dir, _njobs, _brainvisa_dir):
        return {sub: dict(counts) for sub, counts in by_graph[path_to_graph].items()}

    monkeypatch.setattr(Compare, "_load_database", _fake_load_database)
    script = Compare()
    script.args = script.parse_args(
        [
            "databases",
            "--labeled_subjects_dir",
            "subjects",
            "--path_to_graph_a",
            _GRAPH_A,
            "--path_to_graph_b",
            _GRAPH_B,
            "--label_a",
            _LABEL_A,
            "--label_b",
            _LABEL_B,
            "--njobs",
            "1",
            "--output",
            str(output),
        ]
    )
    return script.run()


@pytest.fixture(params=sorted(_EMPTY_CASES))
def empty_case(request):
    return _EMPTY_CASES[request.param]


@pytest.mark.unit
class TestDatabasesEmptyPrintsNoRowsMessage:
    """REQ-COMPARE-78: no sulcus loaded -> stdout message "No sulcus rows found" naming both labels."""

    def test_no_rows_message_names_both_labels(self, empty_case, tmp_path, monkeypatch, capsys):
        _run_databases(monkeypatch, *empty_case, tmp_path / "db.csv")

        out_lines = capsys.readouterr().out.splitlines()
        message_lines = [line for line in out_lines if _NO_ROWS in line]
        assert message_lines, f"no stdout line contains {_NO_ROWS!r}"
        assert any(_LABEL_A in line and _LABEL_B in line for line in message_lines)


@pytest.mark.unit
class TestDatabasesEmptyOmitsCsvWrittenLine:
    """REQ-COMPARE-79: no sulcus loaded -> stdout carries no "CSV written to" line."""

    def test_csv_written_line_absent(self, empty_case, tmp_path, monkeypatch, capsys):
        _run_databases(monkeypatch, *empty_case, tmp_path / "db.csv")

        assert _CSV_WRITTEN not in capsys.readouterr().out


@pytest.mark.unit
class TestDatabasesEmptyCreatesNoFile:
    """REQ-COMPARE-80: no sulcus loaded -> no file is created at the --output path."""

    def test_output_file_not_created(self, empty_case, tmp_path, monkeypatch):
        output = tmp_path / "db.csv"

        _run_databases(monkeypatch, *empty_case, output)

        assert not output.exists()


@pytest.mark.unit
class TestDatabasesEmptyExitStatus:
    """REQ-COMPARE-81: no sulcus loaded -> exit status 0."""

    def test_exit_status_is_zero(self, empty_case, tmp_path, monkeypatch):
        assert _run_databases(monkeypatch, *empty_case, tmp_path / "db.csv") == 0


@pytest.mark.unit
class TestDatabasesNonEmptyReportsWrittenPath:
    """REQ-COMPARE-82: at least one sulcus loaded -> CSV written, stdout names its absolute --output path."""

    def test_csv_file_written_with_one_row_per_sulcus(self, tmp_path, monkeypatch):
        output = tmp_path / "db.csv"

        assert _run_databases(monkeypatch, *_NON_EMPTY, output) == 0

        with output.open(newline="") as f:
            sulci = [row["sulcus"] for row in csv.DictReader(f)]
        assert sorted(sulci) == ["F.C.M._left", "S.C._left"]

    def test_csv_written_line_names_absolute_output_path(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        relative_output = "out/db.csv"

        _run_databases(monkeypatch, *_NON_EMPTY, relative_output)

        expected = f"{_CSV_WRITTEN}: {abspath(relative_output)}"
        assert expected in capsys.readouterr().out.splitlines()
        assert (tmp_path / relative_output).is_file()
