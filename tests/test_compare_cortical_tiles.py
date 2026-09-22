#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for src/compare_cortical_tiles.py

PyAIMS is stubbed in conftest.py; here ``compare_cortical_tiles.aims`` is
replaced by a fake reader serving small in-memory numpy volumes.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import compare_cortical_tiles
from compare_cortical_tiles import (
    CompareCorticalTiles,
    bucket_label,
    find_masks,
    load_mask_vol,
    voxel_diff,
    wasserstein_distance,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeAims:
    """Minimal ``soma.aims`` stand-in mapping file paths to numpy volumes."""

    def __init__(self, volumes: dict):
        self.volumes = volumes

    def read(self, path):
        return self.volumes[str(path)]


def _point_volume(index, shape=(4, 4, 4), value=1.0):
    vol = np.zeros(shape, dtype=np.float64)
    vol[index] = value
    return vol


def _make_tile_tree(root: Path, regions: list, filename="Lmask_skeleton.nii.gz") -> Path:
    for region in regions:
        path = root / region / "mask" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return root


def _make_script(argv: list) -> CompareCorticalTiles:
    script = CompareCorticalTiles()
    script.args = script.parse_args(argv)
    return script


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestLoadMaskVol:
    def test_squeezes_trailing_axis(self, monkeypatch):
        vol = np.ones((2, 2, 2, 1), dtype=np.int16)
        monkeypatch.setattr(compare_cortical_tiles, "aims", _FakeAims({"/m.nii.gz": vol}))
        arr = load_mask_vol("/m.nii.gz")
        assert arr.shape == (2, 2, 2)
        assert arr.dtype == np.float64


class TestVoxelDiff:
    def test_identical_volumes(self):
        a = np.ones((2, 2, 2))
        assert voxel_diff(a, a.copy()) == {"changed": 0, "added": 0, "removed": 0}

    def test_added_and_removed(self):
        a = np.array([[[1.0, 0.0]]])
        b = np.array([[[0.0, 1.0]]])
        assert voxel_diff(a, b) == {"changed": 2, "added": 1, "removed": 1}


class TestWassersteinDistance:
    def test_both_empty_is_zero(self):
        z = np.zeros((3, 3, 3))
        assert wasserstein_distance(z, z.copy()) == 0.0

    def test_single_axis_shift(self):
        assert wasserstein_distance(_point_volume((0, 0, 0)), _point_volume((3, 0, 0))) == pytest.approx(3.0)

    def test_one_empty_map_skips_axes(self):
        assert wasserstein_distance(_point_volume((1, 1, 1)), np.zeros((4, 4, 4))) == 0.0


class TestBucketLabel:
    def test_integer_step(self):
        assert bucket_label(15.0, 10.0) == "10-20vox"

    def test_fractional_step(self):
        assert bucket_label(1.25, 0.5) == "1.00-1.50vox"


class TestFindMasks:
    def test_matches_region_mask_layout(self, tmp_path):
        _make_tile_tree(tmp_path, ["S.C.LEFT", "S.C.RIGHT"])
        found = find_masks(tmp_path, "*mask_skeleton.nii.gz")
        assert sorted(found) == [
            "S.C.LEFT/mask/Lmask_skeleton.nii.gz",
            "S.C.RIGHT/mask/Lmask_skeleton.nii.gz",
        ]

    def test_pattern_filters_out_other_mask_types(self, tmp_path):
        _make_tile_tree(tmp_path, ["S.C.LEFT"], filename="Lmask_skeleton.nii.gz")
        _make_tile_tree(tmp_path, ["S.C.LEFT"], filename="Lmask_foldlabel.nii.gz")
        assert list(find_masks(tmp_path, "*mask_foldlabel.nii.gz")) == ["S.C.LEFT/mask/Lmask_foldlabel.nii.gz"]

    def test_empty_directory(self, tmp_path):
        assert find_masks(tmp_path, "*.nii.gz") == {}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestArgumentParsing:
    def test_defaults(self):
        script = _make_script(["--set_a", "a", "--set_b", "b"])
        assert script.args.output == "cortical_tiles_diff_report.json"
        assert script.args.pattern == "*mask_skeleton.nii.gz"
        assert script.args.metric == "diff"
        assert script.args.bucket_step == 10.0

    def test_explicit_values(self):
        script = _make_script(
            ["--set_a", "a", "--set_b", "b", "--metric", "both", "--bucket_step", "2", "--pattern", "*.nii.gz"]
        )
        assert script.args.metric == "both"
        assert script.args.bucket_step == 2.0
        assert script.args.pattern == "*.nii.gz"

    def test_invalid_metric_rejected(self):
        script = CompareCorticalTiles()
        with pytest.raises(SystemExit):
            script.parse_args(["--set_a", "a", "--set_b", "b", "--metric", "nope"])

    def test_set_b_required(self):
        script = CompareCorticalTiles()
        with pytest.raises(SystemExit):
            script.parse_args(["--set_a", "a"])


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


@pytest.fixture
def tile_sets(tmp_path, monkeypatch):
    """Two crops/2mm trees with one shared region plus one exclusive each."""
    dir_a = _make_tile_tree(tmp_path / "a", ["SHARED", "ONLY_A"])
    dir_b = _make_tile_tree(tmp_path / "b", ["SHARED", "ONLY_B"])

    volumes = {
        str(dir_a / "SHARED" / "mask" / "Lmask_skeleton.nii.gz"): _point_volume((0, 0, 0)),
        str(dir_b / "SHARED" / "mask" / "Lmask_skeleton.nii.gz"): _point_volume((2, 0, 0)),
        str(dir_a / "ONLY_A" / "mask" / "Lmask_skeleton.nii.gz"): np.zeros((4, 4, 4)),
        str(dir_b / "ONLY_B" / "mask" / "Lmask_skeleton.nii.gz"): np.zeros((4, 4, 4)),
    }
    monkeypatch.setattr(compare_cortical_tiles, "aims", _FakeAims(volumes))
    return dir_a, dir_b


class TestRun:
    def test_returns_one_when_path_missing(self, tmp_path):
        script = _make_script(["--set_a", str(tmp_path / "nope"), "--set_b", str(tmp_path)])
        assert script.run() == 1

    def test_diff_report(self, tile_sets, tmp_path):
        dir_a, dir_b = tile_sets
        out = tmp_path / "reports" / "diff.json"
        script = _make_script(
            ["--set_a", str(dir_a), "--set_b", str(dir_b), "--output", str(out), "--bucket_step", "1"]
        )
        assert script.run() == 0
        report = json.loads(out.read_text())
        assert report["metric"] == "diff"
        assert report["mask_pattern"] == "*mask_skeleton.nii.gz"
        assert report["summary"]["total_common"] == 1
        assert report["summary"]["only_in_set_a"] == ["ONLY_A/mask/Lmask_skeleton.nii.gz"]
        assert report["diffs_per_mask"]["SHARED/mask/Lmask_skeleton.nii.gz"]["changed"] == 2
        assert "wasserstein_by_bucket" not in report

    def test_wasserstein_report(self, tile_sets, tmp_path):
        dir_a, dir_b = tile_sets
        out = tmp_path / "wass.json"
        script = _make_script(
            [
                "--set_a",
                str(dir_a),
                "--set_b",
                str(dir_b),
                "--output",
                str(out),
                "--metric",
                "wasserstein",
                "--bucket_step",
                "1",
            ]
        )
        assert script.run() == 0
        report = json.loads(out.read_text())
        assert report["wasserstein_per_mask"]["SHARED/mask/Lmask_skeleton.nii.gz"] == pytest.approx(2.0)
        assert "diffs_per_mask" not in report

    def test_both_metrics_and_console_summary(self, tile_sets, tmp_path, capsys):
        dir_a, dir_b = tile_sets
        out = tmp_path / "both.json"
        script = _make_script(
            [
                "--set_a",
                str(dir_a),
                "--set_b",
                str(dir_b),
                "--output",
                str(out),
                "--metric",
                "both",
                "--bucket_step",
                "1",
            ]
        )
        assert script.run() == 0
        report = json.loads(out.read_text())
        assert "diff_by_bucket" in report
        assert "wasserstein_by_bucket" in report
        printed = capsys.readouterr().out
        assert "Diff buckets (step=1.0)" in printed
        assert "Wasserstein buckets (step=1.0)" in printed
        assert "Unchanged masks: 0/1" in printed

    def test_shape_mismatch_is_skipped(self, tmp_path, monkeypatch, capsys):
        dir_a = _make_tile_tree(tmp_path / "a", ["SHARED"])
        dir_b = _make_tile_tree(tmp_path / "b", ["SHARED"])
        volumes = {
            str(dir_a / "SHARED" / "mask" / "Lmask_skeleton.nii.gz"): np.zeros((4, 4, 4)),
            str(dir_b / "SHARED" / "mask" / "Lmask_skeleton.nii.gz"): np.zeros((2, 2, 2)),
        }
        monkeypatch.setattr(compare_cortical_tiles, "aims", _FakeAims(volumes))
        out = tmp_path / "r.json"
        script = _make_script(["--set_a", str(dir_a), "--set_b", str(dir_b), "--output", str(out)])
        assert script.run() == 0
        assert "shape mismatch" in capsys.readouterr().out
        assert json.loads(out.read_text())["diffs_per_mask"] == {}


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_runs_full_pipeline(self, tile_sets, tmp_path, monkeypatch):
        dir_a, dir_b = tile_sets
        out = tmp_path / "main.json"
        monkeypatch.setattr(
            sys,
            "argv",
            ["compare_cortical_tiles.py", "--set_a", str(dir_a), "--set_b", str(dir_b), "--output", str(out)],
        )
        monkeypatch.setattr("champollion_utils.script_builder.check_for_updates", lambda *a, **k: None, raising=False)
        assert compare_cortical_tiles.main() == 0
        assert out.is_file()
