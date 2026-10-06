#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for generate_labelling_qc.py (TASK-199).

The script walks a Morphologist subjects directory and writes a QC TSV
flagging subjects whose labelling session folder is missing, empty, or
lacks one hemisphere's labelled graph -- the graphs cortical_tiles'
remove_ventricle needs to build the whole-brain volume.

Every check builds a synthetic Morphologist tree in ``tmp_path``; no real
graph is ever read (the script is a pure filesystem check).

Requirements:
    REQ-LABELQC-01  TSV file, one sorted row per subject subdirectory, column order
    REQ-LABELQC-02  qc = 1 when both hemispheres are labelled, else 0
    REQ-LABELQC-03  labelled_L / labelled_R = True / False
    REQ-LABELQC-04  classic layout graph path
    REQ-LABELQC-05  BIDS layout graph path ('*' wildcard, single match)
    REQ-LABELQC-06  reason codes and their precedence
    REQ-LABELQC-07  default labelling session deepcnn_session_auto
    REQ-LABELQC-08  --njobs option, output independent of its value
    REQ-LABELQC-09  runs where soma.aims cannot be imported
"""

import csv
import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from cortical_tiles.brainvisa.utils.subjects import select_good_qc

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"

SESSION = "deepcnn_session_auto"
CLASSIC_PATH = "t1mri/ses-V1/default_analysis/folds/3.1"
BIDS_PATH = "t1mri/*/default_analysis/folds/3.1"
BIDS_KEYS = "ses-1_run-1"
COLUMNS = ["participant_id", "qc", "labelled_L", "labelled_R", "reason"]
MODULE = "champollion_pipeline.generate_labelling_qc"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_script():
    """Instantiate the script class.

    Resolved per test (not a module-level import) so a missing module fails
    each test on its own instead of aborting collection of the whole suite.
    """
    return importlib.import_module(MODULE).GenerateLabellingQC()


def _graph_dir(subjects: Path, subject: str, path_to_graph: str = CLASSIC_PATH) -> Path:
    return subjects / subject / path_to_graph


def _make_subject(
    subjects: Path,
    subject: str,
    sides=("L", "R"),
    session: str = SESSION,
    path_to_graph: str = CLASSIC_PATH,
    with_graph_dir: bool = True,
    with_session_dir: bool = True,
    extra_files=(),
) -> Path:
    """Build one subject's Morphologist tree and return its session folder.

    ``sides`` lists the hemispheres whose labelled graph is written as
    ``<S><subject>_<session>.arg`` inside the session folder.
    """
    subject_dir = subjects / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = _graph_dir(subjects, subject, path_to_graph)
    session_dir = graph_dir / session
    if not with_graph_dir:
        return session_dir
    graph_dir.mkdir(parents=True, exist_ok=True)
    if not with_session_dir:
        return session_dir
    session_dir.mkdir(parents=True, exist_ok=True)
    for side in sides:
        (session_dir / f"{side}{subject}_{session}.arg").write_text("graph")
    for name in extra_files:
        (session_dir / name).write_text("x")
    return session_dir


def _run(subjects: Path, output: Path, *extra: str) -> list[dict]:
    """Run the script on ``subjects`` and return the TSV rows as dicts."""
    script = _new_script()
    script.parse_args([str(subjects), str(output), "--path_to_graph", CLASSIC_PATH, *extra])
    script.run()
    return _read_rows(output)


def _run_bids(subjects: Path, output: Path, *extra: str) -> list[dict]:
    script = _new_script()
    script.parse_args([str(subjects), str(output), "--path_to_graph", BIDS_PATH, "--bids", *extra])
    script.run()
    return _read_rows(output)


def _read_rows(output: Path) -> list[dict]:
    with open(output, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _row(rows: list[dict], subject: str) -> dict:
    matches = [r for r in rows if r["participant_id"] == subject]
    assert len(matches) == 1, f"expected exactly one row for {subject}, got {rows}"
    return matches[0]


@pytest.fixture
def subjects(tmp_path) -> Path:
    path = tmp_path / "subjects"
    path.mkdir()
    return path


@pytest.fixture
def output(tmp_path) -> Path:
    return tmp_path / "qc" / "labelling_qc.tsv"


# ---------------------------------------------------------------------------
# REQ-LABELQC-01: output file shape
# ---------------------------------------------------------------------------


class TestOutputFile:
    def test_header_columns_in_order(self, subjects, output):
        _make_subject(subjects, "s01")
        _run(subjects, output)
        header = output.read_text().splitlines()[0]
        assert header.split("\t") == COLUMNS

    def test_one_row_per_subject_subdirectory_sorted(self, subjects, output):
        for name in ("s03", "s01", "s02"):
            _make_subject(subjects, name)
        # Plain files in the subjects directory are not subjects.
        (subjects / "s04.minf").write_text("x")
        (subjects / "notes.txt").write_text("x")
        rows = _run(subjects, output)
        assert [r["participant_id"] for r in rows] == ["s01", "s02", "s03"]


# ---------------------------------------------------------------------------
# REQ-LABELQC-02: qc value
# ---------------------------------------------------------------------------


class TestQcValue:
    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"sides": ("L", "R")}, "1"),
            ({"sides": ("L",)}, "0"),
            ({"sides": ("R",)}, "0"),
            ({"sides": ()}, "0"),
            ({"with_session_dir": False}, "0"),
            ({"with_graph_dir": False}, "0"),
        ],
        ids=["both", "only_L", "only_R", "neither", "no_session_dir", "no_graph_dir"],
    )
    def test_qc_value(self, subjects, output, kwargs, expected):
        _make_subject(subjects, "s01", **kwargs)
        rows = _run(subjects, output)
        assert _row(rows, "s01")["qc"] == expected

    def test_qc_file_filters_through_cortical_tiles_select_good_qc(self, subjects, output):
        """The file is a valid --sk_qc_path: cortical_tiles keeps only qc=1 subjects."""
        _make_subject(subjects, "good")
        _make_subject(subjects, "half", sides=("L",))
        _make_subject(subjects, "none", with_session_dir=False)
        _run(subjects, output)
        assert output.suffix == ".tsv"
        kept = select_good_qc(["good", "half", "none"], str(output))
        assert kept == ["good"]


# ---------------------------------------------------------------------------
# REQ-LABELQC-03: labelled_L / labelled_R
# ---------------------------------------------------------------------------


class TestLabelledColumns:
    @pytest.mark.parametrize(
        "sides, expected_l, expected_r",
        [
            (("L", "R"), "True", "True"),
            (("L",), "True", "False"),
            (("R",), "False", "True"),
            ((), "False", "False"),
        ],
        ids=["both", "only_L", "only_R", "neither"],
    )
    def test_labelled_columns(self, subjects, output, sides, expected_l, expected_r):
        _make_subject(subjects, "s01", sides=sides, extra_files=("other.txt",))
        row = _row(_run(subjects, output), "s01")
        assert (row["labelled_L"], row["labelled_R"]) == (expected_l, expected_r)


# ---------------------------------------------------------------------------
# REQ-LABELQC-04: classic layout path
# ---------------------------------------------------------------------------


class TestClassicLayout:
    def test_graph_found_at_classic_path(self, subjects, output):
        session_dir = _make_subject(subjects, "s01", sides=())
        (session_dir / f"Ls01_{SESSION}.arg").write_text("g")
        (session_dir / f"Rs01_{SESSION}.arg").write_text("g")
        row = _row(_run(subjects, output), "s01")
        assert (row["labelled_L"], row["labelled_R"], row["qc"]) == ("True", "True", "1")

    def test_graph_with_wrong_name_is_not_counted(self, subjects, output):
        """A file not named <S><subject>_<session>.arg is not that subject's graph."""
        session_dir = _make_subject(subjects, "s01", sides=())
        (session_dir / f"Ls01_{SESSION}.arg").write_text("g")
        (session_dir / f"Rother_{SESSION}.arg").write_text("g")
        (session_dir / "Rs01.arg").write_text("g")
        row = _row(_run(subjects, output), "s01")
        assert (row["labelled_L"], row["labelled_R"], row["qc"]) == ("True", "False", "0")


# ---------------------------------------------------------------------------
# REQ-LABELQC-05: BIDS layout path
# ---------------------------------------------------------------------------


class TestBidsLayout:
    def _bids_subject(self, subjects: Path, subject: str, sides) -> Path:
        concrete = BIDS_PATH.replace("*", BIDS_KEYS)
        return _make_subject(subjects, subject, sides=sides, path_to_graph=concrete)

    def test_graph_found_through_wildcard_path(self, subjects, output):
        self._bids_subject(subjects, "sub-01", ("L", "R"))
        session_dir = _graph_dir(subjects, "sub-01", BIDS_PATH.replace("*", BIDS_KEYS)) / SESSION
        assert (session_dir / f"Lsub-01_{SESSION}.arg").exists()
        row = _row(_run_bids(subjects, output), "sub-01")
        assert (row["labelled_L"], row["labelled_R"], row["qc"], row["reason"]) == ("True", "True", "1", "")

    def test_bids_missing_right_hemisphere(self, subjects, output):
        self._bids_subject(subjects, "sub-01", ("L",))
        self._bids_subject(subjects, "sub-02", ("L", "R"))
        rows = _run_bids(subjects, output)
        r1 = _row(rows, "sub-01")
        assert (r1["labelled_L"], r1["labelled_R"], r1["qc"], r1["reason"]) == ("True", "False", "0", "missing_R")
        assert _row(rows, "sub-02")["qc"] == "1"


# ---------------------------------------------------------------------------
# REQ-LABELQC-06: reason codes
# ---------------------------------------------------------------------------


class TestReason:
    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"with_graph_dir": False}, "no_session_dir"),
            ({"with_session_dir": False}, "no_session_dir"),
            ({"sides": ()}, "empty_session_dir"),
            ({"sides": (), "extra_files": ("other.txt",)}, "missing_both"),
            ({"sides": ("R",)}, "missing_L"),
            ({"sides": ("L",)}, "missing_R"),
            ({"sides": ("L", "R")}, ""),
        ],
        ids=[
            "no_graph_dir",
            "no_session_dir",
            "empty_session_dir",
            "missing_both",
            "missing_L",
            "missing_R",
            "ok",
        ],
    )
    def test_reason_code(self, subjects, output, kwargs, expected):
        _make_subject(subjects, "s01", **kwargs)
        assert _row(_run(subjects, output), "s01")["reason"] == expected


# ---------------------------------------------------------------------------
# REQ-LABELQC-07: labelling session
# ---------------------------------------------------------------------------


class TestLabellingSession:
    def test_default_is_deepcnn_session_auto(self, subjects, output):
        script = _new_script()
        args = script.parse_args([str(subjects), str(output), "--path_to_graph", CLASSIC_PATH])
        assert args.labelling_session == "deepcnn_session_auto"

    def test_default_session_folder_is_checked(self, subjects, output):
        _make_subject(subjects, "s01", session="deepcnn_session_auto")
        _make_subject(subjects, "s02", session="0_auto")
        rows = _run(subjects, output)
        assert _row(rows, "s01")["qc"] == "1"
        assert _row(rows, "s02")["reason"] == "no_session_dir"

    def test_custom_session_is_honoured(self, subjects, output):
        _make_subject(subjects, "s01", session="deepcnn_session_auto")
        _make_subject(subjects, "s02", session="0_auto")
        rows = _run(subjects, output, "--labelling_session", "0_auto")
        assert _row(rows, "s01")["reason"] == "no_session_dir"
        assert _row(rows, "s02")["qc"] == "1"


# ---------------------------------------------------------------------------
# REQ-LABELQC-08: --njobs
# ---------------------------------------------------------------------------


class TestNjobs:
    def test_njobs_option_parsed_as_int(self, subjects, output):
        script = _new_script()
        args = script.parse_args([str(subjects), str(output), "--path_to_graph", CLASSIC_PATH, "--njobs", "3"])
        assert args.njobs == 3

    def test_output_identical_across_njobs(self, subjects, tmp_path):
        _make_subject(subjects, "s01")
        _make_subject(subjects, "s02", sides=("L",))
        _make_subject(subjects, "s03", with_session_dir=False)
        _make_subject(subjects, "s04", sides=())
        _make_subject(subjects, "s05", sides=("R",))
        out1 = tmp_path / "qc_1.tsv"
        out2 = tmp_path / "qc_2.tsv"
        _run(subjects, out1, "--njobs", "1")
        _run(subjects, out2, "--njobs", "2")
        assert out1.read_text() == out2.read_text()


# ---------------------------------------------------------------------------
# REQ-LABELQC-09: no soma.aims
# ---------------------------------------------------------------------------


class TestWithoutAims:
    def test_qc_file_written_when_soma_aims_unavailable(self, subjects, output, tmp_path):
        """conftest stubs soma in-process, so this runs in a child where soma is blocked."""
        _make_subject(subjects, "s01")
        _make_subject(subjects, "s02", sides=("L",))
        code = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(_SRC_DIR)!r})
            sys.modules["soma"] = None
            sys.modules["soma.aims"] = None
            import champollion_utils.script_builder as _sb
            _sb.check_for_updates = lambda *a, **k: None
            from champollion_pipeline.generate_labelling_qc import GenerateLabellingQC
            script = GenerateLabellingQC()
            script.parse_args([{str(subjects)!r}, {str(output)!r}, "--path_to_graph", {CLASSIC_PATH!r}])
            script.run()
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], cwd=str(tmp_path), capture_output=True, text=True, timeout=120
        )
        assert proc.returncode == 0, proc.stderr
        rows = _read_rows(output)
        assert [(r["participant_id"], r["qc"]) for r in rows] == [("s01", "1"), ("s02", "0")]
