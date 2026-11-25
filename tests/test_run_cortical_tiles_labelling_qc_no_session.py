#!/usr/bin/env python3
"""
Whole-brain step of run_cortical_tiles.py when stage 1 ran without sulcal
recognition: no subject has a labelling session folder, so the step logs one
WARNING instead of one ERROR per subject (user decision 2026-11-19).
"""

import logging

from tests.stage2_layout import _stub_whole_brain_functions  # noqa: F401, V103 - autouse fixture
from tests.test_run_cortical_tiles_labelling_qc import (
    _BOTH_SIDES,
    _labelling_qc_path,
    _labelling_qc_rows,
    _records_at,
    _run_labelled_step,
)

_FLAG = "--enable-sulcal-recognition"


class TestNoLabellingSession:
    def test_no_error_and_one_warning_naming_the_flag_when_no_subject_has_a_session(self, temp_dir, pipeline_caplog):
        _run_labelled_step(temp_dir, {"subj01": {}, "subj02": {}, "subj03": {}})

        assert _records_at(logging.ERROR, pipeline_caplog) == []
        warned = _records_at(logging.WARNING, pipeline_caplog)
        assert len(warned) == 1
        assert _FLAG in warned[0]
        assert "whole-brain" in warned[0].lower()
        assert "sulcal labels" in warned[0]

    def test_the_one_warning_lists_the_excluded_subject_ids(self, temp_dir, pipeline_caplog):
        _run_labelled_step(temp_dir, {"subj01": {}, "subj02": {}, "subj03": {}})

        warned = _records_at(logging.WARNING, pipeline_caplog)
        assert len(warned) == 1
        assert all(subject in warned[0] for subject in ("subj01", "subj02", "subj03"))
        assert _records_at(logging.ERROR, pipeline_caplog) == []

    def test_report_is_still_written_with_no_session_dir_rows(self, temp_dir):
        _run_labelled_step(temp_dir, {"subj01": {}, "subj02": {}})

        assert _labelling_qc_path(temp_dir).read_text().splitlines() == [
            "participant_id\tqc\tlabelled_L\tlabelled_R\treason",
            "subj01\t0\tFalse\tFalse\tno_session_dir",
            "subj02\t0\tFalse\tFalse\tno_session_dir",
        ]

    def test_mixed_run_logs_an_error_only_for_the_failing_subject(self, temp_dir, pipeline_caplog):
        _run_labelled_step(temp_dir, {"subj01": _BOTH_SIDES, "subj02": {}})

        errors = _records_at(logging.ERROR, pipeline_caplog)
        assert len(errors) == 1
        assert "subj02" in errors[0] and "no_session_dir" in errors[0]
        assert [m for m in _records_at(logging.WARNING, pipeline_caplog) if _FLAG in m] == []

    def test_session_folder_without_graphs_keeps_its_errors(self, temp_dir, pipeline_caplog):
        _run_labelled_step(temp_dir, {"subj01": {"deepcnn_session_auto": ""}, "subj02": {}})

        errors = _records_at(logging.ERROR, pipeline_caplog)
        assert len(errors) == 2
        assert sorted(_labelling_qc_rows(temp_dir)) == ["subj01", "subj02"]
