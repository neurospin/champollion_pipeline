#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for src/compare_masks.py

PyAIMS is stubbed in conftest.py; here ``compare_masks.aims`` is replaced by a
fake reader that serves small in-memory numpy volumes, so no NIfTI file is
ever decoded.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import compare_masks
from compare_masks import bucket_label, find_masks, load_mask_vol, parse_args, voxel_diff, wasserstein_distance

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


def _make_mask_tree(root: Path, names: list) -> Path:
    """Create empty side/sulcus.nii.gz files under root."""
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return root


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestLoadMaskVol:
    def test_squeezes_trailing_axis(self, monkeypatch):
        vol = np.ones((2, 2, 2, 1), dtype=np.int16)
        monkeypatch.setattr(compare_masks, "aims", _FakeAims({"/m.nii.gz": vol}))
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

    def test_value_change_without_add_or_remove(self):
        a = np.array([[[1.0, 1.0]]])
        b = np.array([[[2.0, 1.0]]])
        assert voxel_diff(a, b) == {"changed": 1, "added": 0, "removed": 0}


class TestWassersteinDistance:
    def test_both_empty_is_zero(self):
        z = np.zeros((3, 3, 3))
        assert wasserstein_distance(z, z.copy()) == 0.0

    def test_identical_maps_is_zero(self):
        a = _point_volume((1, 1, 1))
        assert wasserstein_distance(a, a.copy()) == pytest.approx(0.0)

    def test_single_axis_shift(self):
        a = _point_volume((0, 0, 0))
        b = _point_volume((2, 0, 0))
        assert wasserstein_distance(a, b) == pytest.approx(2.0)

    def test_diagonal_shift_combines_axes(self):
        a = _point_volume((0, 0, 0))
        b = _point_volume((1, 1, 0))
        assert wasserstein_distance(a, b) == pytest.approx(np.sqrt(2.0))

    def test_one_empty_map_skips_axes(self):
        a = _point_volume((1, 1, 1))
        b = np.zeros((4, 4, 4))
        assert wasserstein_distance(a, b) == 0.0


class TestBucketLabel:
    def test_integer_step(self):
        assert bucket_label(3.7, 1.0) == "3-4vox"

    def test_integer_step_ten(self):
        assert bucket_label(23.0, 10.0) == "20-30vox"

    def test_fractional_step(self):
        assert bucket_label(0.7, 0.5) == "0.50-1.00vox"


class TestFindMasks:
    def test_returns_relative_to_absolute_map(self, tmp_path):
        _make_mask_tree(tmp_path, ["L/S.C.nii.gz", "R/S.C.nii.gz"])
        found = find_masks(tmp_path)
        assert sorted(found) == ["L/S.C.nii.gz", "R/S.C.nii.gz"]
        assert found["L/S.C.nii.gz"] == str(tmp_path / "L" / "S.C.nii.gz")

    def test_ignores_deeper_per_subject_files(self, tmp_path):
        _make_mask_tree(tmp_path, ["L/S.C.nii.gz", "L/S.C/sub-01.nii.gz"])
        assert list(find_masks(tmp_path)) == ["L/S.C.nii.gz"]

    def test_empty_directory(self, tmp_path):
        assert find_masks(tmp_path) == {}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self):
        args = parse_args(["--set_a", "a", "--set_b", "b"])
        assert args.output == "mask_diff_report.json"
        assert args.metric == "wasserstein"
        assert args.bucket_step == 1.0

    def test_explicit_values(self):
        args = parse_args(
            ["--set_a", "a", "--set_b", "b", "--metric", "both", "--bucket_step", "5", "--output", "o.json"]
        )
        assert args.metric == "both"
        assert args.bucket_step == 5.0
        assert args.output == "o.json"

    def test_invalid_metric_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--set_a", "a", "--set_b", "b", "--metric", "nope"])

    def test_set_a_required(self):
        with pytest.raises(SystemExit):
            parse_args(["--set_b", "b"])


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


@pytest.fixture
def mask_sets(tmp_path, monkeypatch):
    """Two mask sets with one shared mask plus one exclusive mask each."""
    dir_a = _make_mask_tree(tmp_path / "a", ["L/shared.nii.gz", "L/only_a.nii.gz"])
    dir_b = _make_mask_tree(tmp_path / "b", ["L/shared.nii.gz", "L/only_b.nii.gz"])

    volumes = {
        str(dir_a / "L" / "shared.nii.gz"): _point_volume((0, 0, 0)),
        str(dir_b / "L" / "shared.nii.gz"): _point_volume((2, 0, 0)),
        str(dir_a / "L" / "only_a.nii.gz"): np.zeros((4, 4, 4)),
        str(dir_b / "L" / "only_b.nii.gz"): np.zeros((4, 4, 4)),
    }
    monkeypatch.setattr(compare_masks, "aims", _FakeAims(volumes))
    return dir_a, dir_b


class TestMain:
    def test_missing_set_a_exits(self, tmp_path, capsys):
        (tmp_path / "b").mkdir()
        with pytest.raises(SystemExit) as exc:
            compare_masks.main(["--set_a", str(tmp_path / "nope"), "--set_b", str(tmp_path / "b")])
        assert exc.value.code == 1
        assert "set_a directory not found" in capsys.readouterr().out

    def test_missing_set_b_exits(self, tmp_path, capsys):
        (tmp_path / "a").mkdir()
        with pytest.raises(SystemExit) as exc:
            compare_masks.main(["--set_a", str(tmp_path / "a"), "--set_b", str(tmp_path / "nope")])
        assert exc.value.code == 1
        assert "set_b directory not found" in capsys.readouterr().out

    def test_wasserstein_report(self, mask_sets, tmp_path):
        dir_a, dir_b = mask_sets
        out = tmp_path / "reports" / "wass.json"
        rc = compare_masks.main(
            ["--set_a", str(dir_a), "--set_b", str(dir_b), "--output", str(out), "--metric", "wasserstein"]
        )
        assert rc == 0
        report = json.loads(out.read_text())
        assert report["metric"] == "wasserstein"
        assert report["summary"]["total_common"] == 1
        assert report["summary"]["only_in_set_a"] == ["L/only_a.nii.gz"]
        assert report["summary"]["only_in_set_b"] == ["L/only_b.nii.gz"]
        assert report["wasserstein_per_mask"]["L/shared.nii.gz"] == pytest.approx(2.0)
        assert report["wasserstein_by_bucket"] == {"2-3vox": ["L/shared.nii.gz"]}
        assert "diff_by_bucket" not in report

    def test_diff_report(self, mask_sets, tmp_path):
        dir_a, dir_b = mask_sets
        out = tmp_path / "diff.json"
        rc = compare_masks.main(
            ["--set_a", str(dir_a), "--set_b", str(dir_b), "--output", str(out), "--metric", "diff"]
        )
        assert rc == 0
        report = json.loads(out.read_text())
        assert report["diffs_per_mask"]["L/shared.nii.gz"] == {"changed": 2, "added": 1, "removed": 1}
        assert "wasserstein_per_mask" not in report

    def test_both_metrics_and_console_summary(self, mask_sets, tmp_path, capsys):
        dir_a, dir_b = mask_sets
        out = tmp_path / "both.json"
        rc = compare_masks.main(
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
        assert rc == 0
        report = json.loads(out.read_text())
        assert "wasserstein_by_bucket" in report
        assert "diff_by_bucket" in report
        printed = capsys.readouterr().out
        assert "Wasserstein buckets (step=1.0)" in printed
        assert "Diff buckets (step=1.0)" in printed
        assert "Unchanged masks: 0/1" in printed

    def test_fractional_bucket_step_sorts(self, mask_sets, tmp_path):
        dir_a, dir_b = mask_sets
        out = tmp_path / "frac.json"
        compare_masks.main(
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
                "0.5",
            ]
        )
        report = json.loads(out.read_text())
        assert list(report["wasserstein_by_bucket"]) == ["2.00-2.50vox"]
