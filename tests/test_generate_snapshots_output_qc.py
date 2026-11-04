#!/usr/bin/env python3
"""
Stage 6 output QC tests for generate_snapshots.py: output_qc.tsv, one row per expected snapshot file.
"""

import csv
import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.generate_snapshots import discover_umap_pairs, find_completed_regions
from tests.log_assertions import FAIL
from tests.test_generate_snapshots import (
    _RENDER_SULCAL,
    _RENDER_TILES,
    _RENDER_UMAP,
    _all_steps_args,
    _embeddings_csv,
    _morphologist_subject,
    _run_snapshots,
    _umap_reference,
)

OUTPUT_QC_FILE = "output_qc.tsv"
PNG_BYTES = b"\x89PNG"
SULCAL_LEFT = "sulcal_graph_left.png"
SULCAL_RIGHT = "sulcal_graph_right.png"
TILES_LEFT = "tiles_masks_left.png"
TILES_RIGHT = "tiles_masks_right.png"
UMAP_PLOT = "umap_S.Or._left.png"
ALL_STEPS_FILES = (SULCAL_LEFT, SULCAL_RIGHT, TILES_LEFT, UMAP_PLOT)


class _Renders:
    """Render stand-ins writing each snapshot file, except those named ``gone`` (none) or ``hollow`` (zero bytes)."""

    def __init__(self, gone=(), hollow=()):
        self.gone = set(gone)
        self.hollow = set(hollow)

    def _write(self, path):
        name = os.path.basename(path)
        if name not in self.gone:
            Path(path).write_bytes(b"" if name in self.hollow else PNG_BYTES)
        return path

    def sulcal(self, graph_path, out, *args, **kwargs):
        return self._write(out)

    def tiles(self, crops_dir, out, size, **kwargs):
        base = os.path.splitext(out)[0]
        completed = find_completed_regions(crops_dir)
        return [self._write(f"{base}_{hemi}.png") for hemi in ("left", "right") if completed[hemi]]

    def umap(self, embeddings_dir, reference_dir, out, size, **kwargs):
        base = os.path.splitext(out)[0]
        pairs = discover_umap_pairs(embeddings_dir, reference_dir, regions=kwargs.get("regions"))
        return [self._write(f"{base}_{region}_{hemi}.png") for _, _, _, region, hemi in pairs]


def _run_stage(args, gone=(), hollow=(), raising=None):
    """Run GenerateSnapshots with the render stand-ins (``raising``: the render target to fail); return its status."""
    renders = _Renders(gone, hollow)
    effects = {_RENDER_SULCAL: renders.sulcal, _RENDER_TILES: renders.tiles, _RENDER_UMAP: renders.umap}
    if raising:
        effects[raising] = RuntimeError("synthetic render failure")
    with (
        patch(_RENDER_SULCAL, side_effect=effects[_RENDER_SULCAL]),
        patch(_RENDER_TILES, side_effect=effects[_RENDER_TILES]),
        patch(_RENDER_UMAP, side_effect=effects[_RENDER_UMAP]),
    ):
        return _run_snapshots(args)


def _report_path(temp_dir):
    return os.path.join(temp_dir, "out", OUTPUT_QC_FILE)


def _rows(temp_dir):
    """Read output_qc.tsv of the run, keyed by check; the cells stay text."""
    with open(_report_path(temp_dir), encoding="utf-8", newline="") as handle:
        return {row["check"]: row for row in csv.DictReader(handle, delimiter="\t")}


def _messages_at(capture, level):
    """Messages of the pipeline records captured at exactly ``level``."""
    return [record.getMessage() for record in capture.records if record.levelno == level]


def _output_args(temp_dir):
    return ["--output_dir", os.path.join(temp_dir, "out")]


def _sulcal_args(temp_dir, kept=("sub-01",), excluded=()):
    """Morphologist subjects, ``excluded`` ones lacking the left graph; return the --sulcal-only arguments."""
    morph = os.path.join(temp_dir, "morphologist")
    os.makedirs(morph, exist_ok=True)
    for subject in kept:
        _morphologist_subject(morph, subject)
    for subject in excluded:
        _morphologist_subject(morph, subject, graphs=("R",), meshes=("R",))
    return [*_output_args(temp_dir), "--morphologist_dir", morph, "--sulcal-only"]


def _umap_args(temp_dir, subjects=("sub-01", "sub-02"), model=True):
    """One S.Or. left embeddings CSV holding ``subjects``; return the --umap-only arguments."""
    embeddings = os.path.join(temp_dir, "embeddings")
    _embeddings_csv(embeddings, "S.Or.", "left", [(subject, 0.1, 0.2) for subject in subjects])
    reference = os.path.join(temp_dir, "reference")
    _umap_reference(reference, "S.Or.", "left", model=model)
    return [*_output_args(temp_dir), "--embeddings_dir", embeddings, "--reference_data_dir", reference, "--umap-only"]


def _tiles_args(temp_dir, masks=("L", "R")):
    """One S.Or. crop region with the given mask sides; return the --tiles-only arguments."""
    crops = Path(temp_dir) / "cortical_tiles" / "crops" / "2mm"
    (crops / "S.Or." / "mask").mkdir(parents=True)
    for side in masks:
        (crops / "S.Or." / "mask" / f"{side}mask_skeleton.nii.gz").touch()
    return [*_output_args(temp_dir), "--cortical_tiles_dir", str(crops.parent.parent), "--tiles-only"]


def _second_acquisition_left_graph(temp_dir, subject="sub-01"):
    """Add a left sulcal graph of ``subject`` in a wk40 acquisition, making the left hemisphere ambiguous."""
    folds = Path(temp_dir) / "morphologist" / subject / "t1mri" / "wk40" / "default_analysis" / "folds"
    folds.mkdir(parents=True)
    (folds / f"Lsulci_{subject}.arg").touch()


class TestOutputQcReport:
    """output_qc.tsv is a per-run report: check, qc, reason, one row per expected snapshot file."""

    def test_report_holds_the_run_columns(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-E9393E117269."""
        _run_stage(_all_steps_args(temp_dir))
        header = Path(_report_path(temp_dir)).read_text().splitlines()[0].split("\t")
        assert header == ["check", "qc", "reason"]

    def test_one_row_per_expected_file(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-EB04153CD98F."""
        _run_stage(_all_steps_args(temp_dir))
        assert sorted(_rows(temp_dir)) == sorted(ALL_STEPS_FILES)

    def test_no_row_without_expected_file(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-EB04153CD98F: a --tiles-only run with no crops directory expects no file."""
        _run_stage([*_output_args(temp_dir), "--tiles-only"])
        assert _rows(temp_dir) == {}

    def test_complete_run_rows_hold_qc_1_and_no_reason(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-F0C2431435D3."""
        _run_stage(_all_steps_args(temp_dir))
        assert {(row["qc"], row["reason"]) for row in _rows(temp_dir).values()} == {("1", "")}


class TestExpectedSulcalFiles:
    """The sulcal step expects one file per unambiguous hemisphere of the rendered subject."""

    def test_both_hemispheres_expected(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-70F8EA6C643F."""
        _run_stage(_sulcal_args(temp_dir))
        assert sorted(_rows(temp_dir)) == [SULCAL_LEFT, SULCAL_RIGHT]

    def test_only_the_rendered_subject_counts(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-70F8EA6C643F: a second kept subject adds no row."""
        _run_stage(_sulcal_args(temp_dir, kept=("sub-01", "sub-02")))
        assert sorted(_rows(temp_dir)) == [SULCAL_LEFT, SULCAL_RIGHT]

    def test_acquisition_tags_the_expected_name(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-70F8EA6C643F: --acquisition puts its name in the expected file name."""
        _run_stage([*_sulcal_args(temp_dir), "--acquisition", "default_acquisition"])
        assert sorted(_rows(temp_dir)) == [
            "sulcal_graph_default_acquisition_left.png",
            "sulcal_graph_default_acquisition_right.png",
        ]

    def test_ambiguous_hemisphere_gives_no_row(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-70F8EA6C643F: a hemisphere with two graphs has no expected file."""
        args = _sulcal_args(temp_dir)
        _second_acquisition_left_graph(temp_dir)
        _run_stage(args)
        assert sorted(_rows(temp_dir)) == [SULCAL_RIGHT]

    def test_unselected_step_gives_no_row(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-70F8EA6C643F: --umap-only leaves the sulcal step out."""
        _run_stage([*_all_steps_args(temp_dir), "--umap-only"])
        assert sorted(_rows(temp_dir)) == [UMAP_PLOT]


class TestExpectedUmapFiles:
    """The UMAP step expects one file per discovered pair with a kept subject."""

    def test_pair_expected(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-0C529A38ED11."""
        _run_stage(_umap_args(temp_dir))
        assert sorted(_rows(temp_dir)) == [UMAP_PLOT]

    def test_csv_without_reference_gives_no_row(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-0C529A38ED11: discover_umap_pairs leaves out a CSV lacking a reference file."""
        _run_stage(_umap_args(temp_dir, model=False))
        assert _rows(temp_dir) == {}

    def test_csv_left_out_by_region_filter_gives_no_row(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-0C529A38ED11: a region outside --umap_region expects no plot."""
        args = _umap_args(temp_dir)
        _embeddings_csv(
            os.path.join(temp_dir, "embeddings"), "F.I.P.", "left", [("sub-01", 0.1, 0.2), ("sub-02", 1, 2)]
        )
        _umap_reference(os.path.join(temp_dir, "reference"), "F.I.P.", "left")
        _run_stage([*args, "--umap_region", "S.Or."])
        assert sorted(_rows(temp_dir)) == [UMAP_PLOT]

    def test_csv_without_kept_subject_gives_no_row(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-0C529A38ED11: the input QC excludes the only CSV subject, a sulcal subject stays."""
        args = [a for a in _umap_args(temp_dir, subjects=("sub-09",)) if a != "--umap-only"]
        _embeddings_csv(os.path.join(temp_dir, "embeddings"), "S.Or.", "left", [("sub-09", "", 0.2)])
        morph = os.path.join(temp_dir, "morphologist")
        _morphologist_subject(morph, "sub-01")
        _run_stage([*args, "--morphologist_dir", morph])
        assert sorted(_rows(temp_dir)) == [SULCAL_LEFT, SULCAL_RIGHT]

    def test_unselected_step_gives_no_row(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-0C529A38ED11: --sulcal-only leaves the UMAP step out."""
        _run_stage([*_all_steps_args(temp_dir), "--sulcal-only"])
        assert sorted(_rows(temp_dir)) == [SULCAL_LEFT, SULCAL_RIGHT]


class TestExpectedTilesFiles:
    """The tiles step expects one file per hemisphere with a completed crop region."""

    def test_both_hemispheres_expected(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-F23D5DFAC952."""
        _run_stage(_tiles_args(temp_dir))
        assert sorted(_rows(temp_dir)) == [TILES_LEFT, TILES_RIGHT]

    def test_hemisphere_without_completed_region_gives_no_row(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-F23D5DFAC952: no right mask, so no right file is expected."""
        _run_stage(_tiles_args(temp_dir, masks=("L",)))
        assert sorted(_rows(temp_dir)) == [TILES_LEFT]

    def test_unselected_step_gives_no_row(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-F23D5DFAC952: --umap-only leaves the tiles step out."""
        _run_stage([*_all_steps_args(temp_dir), "--umap-only"])
        assert TILES_LEFT not in _rows(temp_dir)


class TestSnapshotGaps:
    """The rows come from the disk, not from what the renders returned."""

    @pytest.mark.parametrize("name", ALL_STEPS_FILES)
    def test_absent_file(self, temp_dir, name):
        """REQ-NTUI-BDRABCZUK-D6731AB50B1D."""
        _run_stage(_all_steps_args(temp_dir), gone=[name])
        row = _rows(temp_dir)[name]
        assert (row["qc"], row["reason"]) == ("0", "missing_snapshot")

    @pytest.mark.parametrize("name", ALL_STEPS_FILES)
    def test_zero_byte_file(self, temp_dir, name):
        """REQ-NTUI-BDRABCZUK-5B85A8FBB7A7."""
        _run_stage(_all_steps_args(temp_dir), hollow=[name])
        row = _rows(temp_dir)[name]
        assert (row["qc"], row["reason"]) == ("0", "empty_snapshot")

    @pytest.mark.parametrize(
        "target,names",
        [
            (_RENDER_SULCAL, (SULCAL_LEFT, SULCAL_RIGHT)),
            (_RENDER_TILES, (TILES_LEFT,)),
            (_RENDER_UMAP, (UMAP_PLOT,)),
        ],
        ids=["sulcal", "tiles", "umap"],
    )
    def test_render_exception_counts_as_absent(self, temp_dir, target, names):
        """REQ-NTUI-BDRABCZUK-D6731AB50B1D: a render that raised leaves its expected files absent."""
        _run_stage(_all_steps_args(temp_dir), raising=target)
        rows = _rows(temp_dir)
        for name in names:
            assert (rows[name]["qc"], rows[name]["reason"]) == ("0", "missing_snapshot")

    def test_other_rows_keep_qc_1(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-F0C2431435D3: a gap in one file leaves the other rows passing."""
        _run_stage(_all_steps_args(temp_dir), gone=[UMAP_PLOT])
        rows = _rows(temp_dir)
        assert rows[UMAP_PLOT]["qc"] == "0"
        for name in (SULCAL_LEFT, SULCAL_RIGHT, TILES_LEFT):
            assert (rows[name]["qc"], rows[name]["reason"]) == ("1", "")


class TestOutputQcLog:
    """Each failing row of output_qc.tsv gets one ERROR record."""

    def test_one_error_record_per_qc_0_row(self, temp_dir, pipeline_caplog):
        """REQ-NTUI-BDRABCZUK-5E3B954EEEF0."""
        _run_stage(_all_steps_args(temp_dir), gone=[TILES_LEFT], hollow=[UMAP_PLOT])
        assert _rows(temp_dir)[TILES_LEFT]["qc"] == "0"
        messages = _messages_at(pipeline_caplog, logging.ERROR)
        assert len(messages) == 2, messages
        assert len([m for m in messages if TILES_LEFT in m and "missing_snapshot" in m]) == 1, messages
        assert len([m for m in messages if UMAP_PLOT in m and "empty_snapshot" in m]) == 1, messages

    def test_no_error_record_on_a_complete_run(self, temp_dir, pipeline_caplog):
        """REQ-NTUI-BDRABCZUK-5E3B954EEEF0: no qc 0 row, no ERROR record."""
        _run_stage(_all_steps_args(temp_dir))
        assert sorted(_rows(temp_dir)) == sorted(ALL_STEPS_FILES)
        assert _messages_at(pipeline_caplog, logging.ERROR) == []


class TestEverySnapshotLost:
    """A run with expected snapshots and no usable one fails (user decisions 2026-11-04)."""

    def test_every_snapshot_lost_returns_1(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-B77A1CEE4057."""
        result = _run_stage(_all_steps_args(temp_dir), gone=ALL_STEPS_FILES)
        assert {row["qc"] for row in _rows(temp_dir).values()} == {"0"}
        assert result == 1

    def test_every_snapshot_empty_returns_1(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-B77A1CEE4057: a zero-byte file is lost too."""
        result = _run_stage(_all_steps_args(temp_dir), hollow=ALL_STEPS_FILES)
        assert {row["qc"] for row in _rows(temp_dir).values()} == {"0"}
        assert result == 1

    def test_partial_exclusion_with_every_snapshot_lost_returns_1(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-B77A1CEE4057: one subject excluded, the kept subject loses every snapshot."""
        args = _sulcal_args(temp_dir, kept=("sub-01",), excluded=("sub-02",))
        result = _run_stage(args, gone=[SULCAL_LEFT, SULCAL_RIGHT])
        assert {row["qc"] for row in _rows(temp_dir).values()} == {"0"}
        assert result == 1

    def test_tiles_only_every_snapshot_lost_returns_1(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-B77A1CEE4057: the --tiles-only exemption does not cover lost tiles."""
        result = _run_stage(_tiles_args(temp_dir), gone=[TILES_LEFT, TILES_RIGHT])
        assert {row["qc"] for row in _rows(temp_dir).values()} == {"0"}
        assert result == 1

    def test_every_snapshot_lost_emits_one_fail_record(self, temp_dir, pipeline_caplog):
        """REQ-NTUI-BDRABCZUK-BD2EF323017F."""
        _run_stage(_all_steps_args(temp_dir), gone=ALL_STEPS_FILES)
        messages = _messages_at(pipeline_caplog, FAIL)
        assert len(messages) == 1, messages
        assert _report_path(temp_dir) in messages[0]

    def test_tiles_only_every_snapshot_lost_emits_one_fail_record(self, temp_dir, pipeline_caplog):
        """REQ-NTUI-BDRABCZUK-BD2EF323017F."""
        _run_stage(_tiles_args(temp_dir), gone=[TILES_LEFT, TILES_RIGHT])
        messages = _messages_at(pipeline_caplog, FAIL)
        assert len(messages) == 1, messages
        assert _report_path(temp_dir) in messages[0]

    def test_usable_snapshot_emits_no_fail_record(self, temp_dir, pipeline_caplog):
        """REQ-NTUI-BDRABCZUK-BD2EF323017F: a qc 1 row left, no FAIL record."""
        _run_stage(_all_steps_args(temp_dir), gone=[SULCAL_LEFT, TILES_LEFT, UMAP_PLOT])
        assert _rows(temp_dir)[SULCAL_RIGHT]["qc"] == "1"
        assert _messages_at(pipeline_caplog, FAIL) == []


class TestPartialSuccess:
    """Excluded subjects or lost snapshots beside usable output return 3, as stage 2 does for a partial run."""

    def test_partial_exclusion_with_complete_snapshots_returns_3(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-9A3DE0C40474."""
        result = _run_stage(_sulcal_args(temp_dir, kept=("sub-01",), excluded=("sub-02",)))
        assert {row["qc"] for row in _rows(temp_dir).values()} == {"1"}
        assert result == 3

    def test_partial_exclusion_without_expected_file_returns_3(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-9A3DE0C40474: no UMAP reference, so output_qc.tsv holds no row."""
        args = _umap_args(temp_dir, model=False)
        rows = [("sub-01", 0.1, 0.2), ("sub-02", "", 0.2)]
        _embeddings_csv(os.path.join(temp_dir, "embeddings"), "S.Or.", "left", rows)
        result = _run_stage(args)
        assert _rows(temp_dir) == {}
        assert result == 3

    def test_partial_exclusion_with_a_lost_snapshot_returns_3(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-9A3DE0C40474."""
        args = _sulcal_args(temp_dir, kept=("sub-01",), excluded=("sub-02",))
        result = _run_stage(args, gone=[SULCAL_LEFT])
        assert _rows(temp_dir)[SULCAL_RIGHT]["qc"] == "1"
        assert result == 3

    def test_some_snapshots_lost_returns_3(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-60BF5C53D2AD."""
        result = _run_stage(_all_steps_args(temp_dir), gone=[SULCAL_LEFT, TILES_LEFT, UMAP_PLOT])
        assert _rows(temp_dir)[SULCAL_RIGHT]["qc"] == "1"
        assert result == 3

    def test_every_subject_excluded_with_a_lost_snapshot_returns_1(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-60BF5C53D2AD: without a qc 1 input row the input QC failure stands."""
        args = [a for a in _sulcal_args(temp_dir, kept=(), excluded=("sub-01",)) if a != "--sulcal-only"]
        tiles = [a for a in _tiles_args(temp_dir) if a != "--tiles-only"]
        result = _run_stage([*args, *tiles[2:]], gone=[TILES_LEFT])
        assert _rows(temp_dir)[TILES_RIGHT]["qc"] == "1"
        assert result == 1

    def test_tiles_only_some_tiles_lost_returns_3(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-D1D35D36AEAF."""
        result = _run_stage(_tiles_args(temp_dir), gone=[TILES_LEFT])
        assert _rows(temp_dir)[TILES_RIGHT]["qc"] == "1"
        assert result == 3


class TestCompleteRun:
    """Every kept subject and every expected snapshot usable: the stage returns 0."""

    def test_complete_run_returns_0(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-D5BECAF3CDFA."""
        result = _run_stage(_all_steps_args(temp_dir))
        assert {row["qc"] for row in _rows(temp_dir).values()} == {"1"}
        assert result == 0

    def test_no_expected_file_returns_0(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-10C6D01FC822: kept subjects, no UMAP reference, so no expected file."""
        result = _run_stage(_umap_args(temp_dir, model=False))
        assert _rows(temp_dir) == {}
        assert result == 0

    def test_tiles_only_complete_returns_0(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-A9F3A3FF7023."""
        result = _run_stage(_tiles_args(temp_dir))
        assert {row["qc"] for row in _rows(temp_dir).values()} == {"1"}
        assert result == 0

    def test_tiles_only_no_expected_file_returns_0(self, temp_dir):
        """REQ-NTUI-BDRABCZUK-A9F3A3FF7023: row N0T, a --tiles-only run with no crops directory."""
        result = _run_stage([*_output_args(temp_dir), "--tiles-only"])
        assert _rows(temp_dir) == {}
        assert result == 0
