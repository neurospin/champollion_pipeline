#!/usr/bin/env python3
"""
Stage 6 exit-status tests for generate_snapshots.py
"""

import logging
import os
from unittest.mock import patch

import pytest

from tests.log_assertions import FAIL
from tests.test_generate_snapshots import (
    _LOAD_UMAP_MODEL,
    _RENDER_SULCAL,
    QC_FILE,
    _embeddings_csv,
    _morphologist_subject,
    _qc_rows,
    _RecordingUmapModel,
    _run_snapshots,
    _umap_reference,
)

# --- Stage 6 exit status (user decision 2026-11-04, "Fail only if nothing usable") ---


def _sulcal_args(temp_dir, kept=(), excluded=()):
    """Morphologist subjects, ``excluded`` ones lacking the left graph; return the --sulcal-only arguments."""
    morph = os.path.join(temp_dir, "morphologist")
    for subject in kept:
        _morphologist_subject(morph, subject)
    for subject in excluded:
        _morphologist_subject(morph, subject, graphs=("R",), meshes=("R",))
    return ["--output_dir", os.path.join(temp_dir, "out"), "--morphologist_dir", morph, "--sulcal-only"]


def _umap_all_excluded_args(temp_dir):
    """One embeddings CSV whose every row holds an empty cell; return the --umap-only arguments."""
    embeddings = os.path.join(temp_dir, "embeddings")
    _embeddings_csv(embeddings, "S.Or.", "left", [("sub-01", "", 0.2), ("sub-02", 0.3, "")])
    reference = os.path.join(temp_dir, "reference")
    _umap_reference(reference, "S.Or.", "left")
    return [
        "--output_dir",
        os.path.join(temp_dir, "out"),
        "--embeddings_dir",
        embeddings,
        "--reference_data_dir",
        reference,
        "--umap-only",
    ]


def _run_with_renders_patched(args):
    """Run GenerateSnapshots with the sulcal render and the UMAP model patched; return its exit value."""
    with (
        patch(_RENDER_SULCAL, side_effect=lambda graph_path, out, *a, **k: out),
        patch(_LOAD_UMAP_MODEL, return_value=_RecordingUmapModel()),
    ):
        return _run_snapshots(args)


def _messages_at(capture, level):
    """Messages of the pipeline records captured at exactly ``level``."""
    return [record.getMessage() for record in capture.records if record.levelno == level]


class TestSnapshotsExitPolicy:
    """Bad subjects are reported per subject; the stage fails only when input_qc.tsv keeps no subject."""

    @pytest.mark.parametrize(
        "build_args",
        [lambda temp_dir: _sulcal_args(temp_dir, excluded=("sub-01", "sub-02")), _umap_all_excluded_args],
        ids=["sulcal", "umap"],
    )
    def test_every_subject_excluded_returns_1(self, temp_dir, build_args):
        """REQ-NTUI-BDRABCZUK-65FF209DE4D8."""
        result = _run_with_renders_patched(build_args(temp_dir))
        rows = _qc_rows(os.path.join(temp_dir, "out"))
        assert sorted(row["qc"] for row in rows.values()) == [0, 0]
        assert result == 1

    def test_partial_exclusion_returns_0(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-B5B9D4243D2C."""
        result = _run_with_renders_patched(_sulcal_args(temp_dir, kept=("sub-01",), excluded=("sub-02", "sub-03")))
        rows = _qc_rows(os.path.join(temp_dir, "out"))
        assert sorted(row["qc"] for row in rows.values()) == [0, 0, 1]
        assert result == 0

    def test_one_error_record_per_excluded_subject(self, temp_dir, pipeline_caplog):
        """REQ-NTUI-BDRABCZUK-03AB88C9072B: one ERROR record per qc 0 row, holding its participant_id and reason."""
        _run_with_renders_patched(_sulcal_args(temp_dir, kept=("sub-01",), excluded=("sub-02", "sub-03")))
        messages = _messages_at(pipeline_caplog, logging.ERROR)
        assert len(messages) == 2, messages
        for subject in ("sub-02", "sub-03"):
            naming = [m for m in messages if subject in m and "missing_sulcal_graph_left" in m]
            assert len(naming) == 1, messages

    def test_no_error_record_when_no_subject_excluded(self, temp_dir, pipeline_caplog):
        """REQ-NTUI-BDRABCZUK-03AB88C9072B: no qc 0 row, no ERROR record."""
        _run_with_renders_patched(_sulcal_args(temp_dir, kept=("sub-01",)))
        assert _messages_at(pipeline_caplog, logging.ERROR) == []

    def test_every_subject_excluded_emits_one_fail_record(self, temp_dir, pipeline_caplog):
        """REQ-NTUI-BDRABCZUK-E237F2F0A8FE."""
        _run_with_renders_patched(_sulcal_args(temp_dir, excluded=("sub-01",)))
        messages = _messages_at(pipeline_caplog, FAIL)
        assert len(messages) == 1, messages
        assert os.path.join(temp_dir, "out", QC_FILE) in messages[0]

    def test_partial_exclusion_emits_no_fail_record(self, temp_dir, pipeline_caplog):
        """REQ-NTUI-BDRABCZUK-E237F2F0A8FE: a qc 1 row left, no FAIL record."""
        _run_with_renders_patched(_sulcal_args(temp_dir, kept=("sub-01",), excluded=("sub-02",)))
        assert _messages_at(pipeline_caplog, FAIL) == []
