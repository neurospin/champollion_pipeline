#!/usr/bin/env python3
"""
Labelled-graph pre-check tests of the whole-brain step of run_cortical_tiles.py
(moved out of test_run_cortical_tiles.py; the shared helpers are imported from it).
"""

import logging
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from champollion_pipeline.utils.qc_report import read_qc_tsv
from tests.test_run_cortical_tiles import (  # noqa: F401
    _GRAPH_SUBPATH,
    _LOGGER_NAME,
    _WILDCARD_SESSION,
    _make_subject,
    _run_whole_brain_step,
    _run_wildcard_step,
    _stub_whole_brain_functions,  # noqa: V103 - autouse fixture, as in the source module
)
from tests.test_run_cortical_tiles import (
    _RemoveVentricleRecorder as _BaseVentricleRecorder,
)


class _RemoveVentricleRecorder(_BaseVentricleRecorder):
    """Recorder that also calls ``on_call`` (no argument) at the start of every call."""

    def __init__(self, on_call=None):
        super().__init__()
        self.on_call = on_call

    def __call__(self, *args, **kwargs):
        if self.on_call:
            self.on_call()
        return super().__call__(*args, **kwargs)


def _prepare_whole_brain_step(temp_dir, subjects, extra_args=None, *, path_to_graph=_GRAPH_SUBPATH, make_subject=None):
    """Return a parsed RunCorticalTiles with ``subjects`` laid out on disk
    (``{subject: [sessions holding a labelled graph]}``).

    ``make_subject(input_dir, output_dir, subject, value)`` lays out one subject
    (default ``_make_subject``); ``path_to_graph`` is the --path_to_graph value.
    """
    root = Path(temp_dir)
    input_dir, output_dir = root / "input", root / "output"
    for directory in (input_dir, output_dir):
        directory.mkdir()
    lay_out = make_subject or _make_subject
    for subject, value in subjects.items():
        lay_out(input_dir, output_dir, subject, value)

    argv = [str(input_dir), str(output_dir), "--path_to_graph", path_to_graph]
    argv += ["--path_sk_with_hull", "skeleton/path", *(extra_args or [])]
    script = RunCorticalTiles()
    script.parse_args(argv)
    return script


def _execute_whole_brain_step(script, recorder, on_crop=None):
    """Run ``script`` with the crop subprocess and fusion mocked and ``remove_ventricle``
    replaced by ``recorder``; ``on_crop`` is called with no argument at the crop subprocess.

    Returns the Python warnings emitted during run() and run()'s return value.
    """

    def crop_succeeds(*_args, **_kwargs):
        if on_crop:
            on_crop()
        return 0

    with (
        warnings.catch_warnings(record=True) as caught,
        patch.object(script, "validate_paths", return_value=True),
        patch.object(script, "execute_command", side_effect=crop_succeeds),
        patch.object(script, "_generate_mask_npys"),
        patch("champollion_pipeline.run_cortical_tiles.add_left_and_right_volumes.add_left_and_right_volumes"),
        patch("champollion_pipeline.run_cortical_tiles.remove_ventricle.remove_ventricle", side_effect=recorder),
    ):
        warnings.simplefilter("always")
        result = script.run()

    return [str(w.message) for w in caught], result


_LABELLING_QC_FILE = "labelling_qc.tsv"
_BIDS_PATH_TO_GRAPH = "*/t1mri/default_acquisition/default_analysis/folds/3.1"


def _labelling_qc_path(temp_dir):
    """Where run() writes the labelling QC report of a ``_run_whole_brain_step`` run."""
    return Path(temp_dir) / "output" / DERIVATIVES_FOLDER / _LABELLING_QC_FILE


def _labelling_qc_rows(temp_dir):
    """Rows of the labelling QC report, keyed by participant_id."""
    return {row["participant_id"]: row for row in read_qc_tsv(_labelling_qc_path(temp_dir))}


def _records_at(level, pipeline_caplog):
    """Messages of the run_cortical_tiles logger records at exactly ``level``."""
    return [r.getMessage() for r in pipeline_caplog.records if r.name == _LOGGER_NAME and r.levelno == level]


def _make_labelled_subject(input_dir, output_dir, subject, graphs):
    """Lay out one fused subject whose labelled graphs are ``graphs``,
    ``{labelling session: sides holding a graph}`` (``"LR"``, ``"R"``, ``""``: an
    empty session folder), under ``<input>/<subject>/<path_to_graph>/``.
    ``graphs`` None leaves a subject folder with no fused volume.
    """
    if graphs is None:
        (Path(input_dir) / subject).mkdir()
        return
    _make_subject(input_dir, output_dir, subject, [])
    for session, sides in graphs.items():
        session_dir = Path(input_dir) / subject / _GRAPH_SUBPATH / session
        session_dir.mkdir(parents=True)
        for side in sides:
            (session_dir / f"{side}{subject}_{session}.arg").touch()


def _make_bids_volume(input_dir, output_dir, fused_id, sides):
    """Lay out the fused volume ``<folder>_<keys>`` of a BIDS subject: its labelled
    graphs sit at ``<input>/<folder>/<path_to_graph, * replaced by keys>/<session>/``,
    named after the folder alone, as cortical_tiles' get_labelled_graph reads them.
    """
    _make_subject(input_dir, output_dir, fused_id, [])
    folder, keys = fused_id.split("_", 1)
    session_dir = Path(input_dir) / folder / _BIDS_PATH_TO_GRAPH.replace("*", keys) / _WILDCARD_SESSION
    session_dir.mkdir(parents=True)
    for side in sides:
        (session_dir / f"{side}{folder}_{_WILDCARD_SESSION}.arg").touch()


def _run_labelled_step(temp_dir, subjects, extra_args=None):
    """``_run_whole_brain_step`` with ``subjects`` given as ``{subject: graphs}`` of ``_make_labelled_subject``."""
    return _run_whole_brain_step(temp_dir, subjects, extra_args, make_subject=_make_labelled_subject)


_BOTH_SIDES = {"deepcnn_session_auto": "LR"}


class TestLabellingQcPreCheck:
    """Labelled-graph pre-check of the whole-brain step: labelling_qc.tsv, then the
    per-subject report. With no usable subject the whole-brain volume is left out
    and the stage keeps the crop result (user decision 2026-11-04, "Report, keep
    stage going"). Assertions on run()'s value use a crop step that returns 0,
    so they do not depend on how run() maps other crop exit codes.
    """

    def test_report_exists_at_first_remove_ventricle_call(self, temp_dir):
        report_seen = []

        script = _prepare_whole_brain_step(temp_dir, {"subj01": _BOTH_SIDES}, make_subject=_make_labelled_subject)
        recorder = _RemoveVentricleRecorder(lambda: report_seen.append(_labelling_qc_path(temp_dir).is_file()))
        _execute_whole_brain_step(script, recorder)

        assert report_seen[:1] == [True]

    def test_report_is_written_after_the_crop_step(self, temp_dir):
        report_seen = []

        script = _prepare_whole_brain_step(temp_dir, {"subj01": _BOTH_SIDES}, make_subject=_make_labelled_subject)
        _execute_whole_brain_step(
            script,
            _RemoveVentricleRecorder(),
            on_crop=lambda: report_seen.append(_labelling_qc_path(temp_dir).exists()),
        )

        assert report_seen == [False]
        assert _labelling_qc_path(temp_dir).is_file()

    def test_one_row_per_fused_subject(self, temp_dir):
        _run_labelled_step(temp_dir, {"subj01": _BOTH_SIDES, "subj02": {}, "subj03": None})

        assert sorted(_labelling_qc_rows(temp_dir)) == ["subj01", "subj02"]

    def test_report_columns(self, temp_dir):
        _run_labelled_step(temp_dir, {"subj01": _BOTH_SIDES})

        header = _labelling_qc_path(temp_dir).read_text().splitlines()[0]
        assert header.split("\t") == ["participant_id", "qc", "labelled_L", "labelled_R", "reason"]

    @pytest.mark.parametrize(
        ("extra_args", "graphs", "expected"),
        [
            pytest.param([], {"deepcnn_session_auto": "LR"}, (1, "True", "True", ""), id="first_candidate"),
            pytest.param([], {"default_session_auto": "LR"}, (1, "True", "True", ""), id="second_candidate"),
            pytest.param(
                [],
                {"deepcnn_session_auto": "L", "default_session_auto": "LR"},
                (0, "True", "False", "missing_R"),
                id="first_existing_candidate_decides",
            ),
            pytest.param([], {"deepcnn_session_auto": "R"}, (0, "False", "True", "missing_L"), id="missing_left"),
            pytest.param([], {"deepcnn_session_auto": ""}, (0, "False", "False", "empty_session_dir"), id="empty"),
            pytest.param([], {"0_auto": "LR"}, (0, "False", "False", "no_session_dir"), id="non_candidate_session"),
            pytest.param([], {}, (0, "False", "False", "no_session_dir"), id="no_session"),
            pytest.param(
                ["--labelling_session", "0_auto"], {"0_auto": "LR"}, (1, "True", "True", ""), id="explicit_session"
            ),
            pytest.param(
                ["--labelling_session", "0_auto"],
                {"deepcnn_session_auto": "LR"},
                (0, "False", "False", "no_session_dir"),
                id="explicit_session_absent",
            ),
        ],
    )
    def test_row_values_follow_the_labelling_qc_rules(self, temp_dir, extra_args, graphs, expected):
        # subj01 passes under each session, so the row checked is never the only usable one.
        healthy = {"deepcnn_session_auto": "LR", "0_auto": "LR"}
        _run_labelled_step(temp_dir, {"subj01": healthy, "subj02": graphs}, extra_args)

        row = _labelling_qc_rows(temp_dir)["subj02"]
        assert (row["qc"], row["labelled_L"], row["labelled_R"], row["reason"]) == expected

    def test_wildcard_path_to_graph_resolves_to_the_matched_folder(self, temp_dir):
        _run_wildcard_step(temp_dir, {"subj01": ["acq1"], "subj02": ["acq2"]})

        rows = _labelling_qc_rows(temp_dir)
        assert {subject: row["qc"] for subject, row in rows.items()} == {"subj01": 1, "subj02": 1}

    @pytest.mark.parametrize("acquisitions", [[], ["acq1", "acq2"]], ids=["no_folder", "several_folders"])
    def test_unresolved_wildcard_gives_qc_0(self, temp_dir, acquisitions):
        _run_wildcard_step(temp_dir, {"subj01": ["acq1"], "subj02": acquisitions})

        row = _labelling_qc_rows(temp_dir)["subj02"]
        assert row["qc"] == 0
        assert row["reason"]

    def test_bids_rows_follow_the_graph_of_each_fused_volume(self, temp_dir):
        _run_whole_brain_step(
            temp_dir,
            {"sub-01_ses-1": "LR", "sub-01_ses-2": "L", "sub-02_ses-1": "LR"},
            ["--bids"],
            path_to_graph=_BIDS_PATH_TO_GRAPH,
            make_subject=_make_bids_volume,
        )

        rows = _labelling_qc_rows(temp_dir)
        assert {subject: row["qc"] for subject, row in rows.items()} == {
            "sub-01_ses-1": 1,
            "sub-01_ses-2": 0,
            "sub-02_ses-1": 1,
        }

    def test_one_error_record_per_failing_subject(self, temp_dir, pipeline_caplog):
        _run_labelled_step(temp_dir, {"subj01": _BOTH_SIDES, "subj02": {"deepcnn_session_auto": "L"}, "subj03": {}})

        errors = _records_at(logging.ERROR, pipeline_caplog)
        assert len(errors) == 2
        assert [m for m in errors if "subj02" in m and "missing_R" in m]
        assert [m for m in errors if "subj03" in m and "no_session_dir" in m]

    @pytest.mark.parametrize(
        "graphs", [{}, {"deepcnn_session_auto": ""}, {"deepcnn_session_auto": "L"}], ids=["none", "empty", "one_side"]
    )
    def test_run_returns_zero_when_no_subject_is_usable(self, temp_dir, graphs):
        _, _, result = _run_labelled_step(temp_dir, {"subj01": graphs, "subj02": graphs})

        assert result == 0

    @pytest.mark.parametrize(
        "graphs", [{"deepcnn_session_auto": ""}, {"deepcnn_session_auto": "L"}], ids=["empty", "one_side"]
    )
    def test_no_ventricle_removal_when_no_subject_is_usable(self, temp_dir, graphs):
        recorder, _, _ = _run_labelled_step(temp_dir, {"subj01": graphs, "subj02": graphs})

        assert recorder.sessions_by_subject == {}

    def test_one_warning_names_the_report_when_whole_brain_is_left_out(self, temp_dir, pipeline_caplog):
        _run_labelled_step(temp_dir, {"subj01": {}, "subj02": {}})

        report = str(_labelling_qc_path(temp_dir))
        warned = [m for m in _records_at(logging.WARNING, pipeline_caplog) if report in m]
        assert len(warned) == 1
        assert "whole-brain" in warned[0].lower()
        assert "left out" in warned[0].lower()
