"""Tests for value-only voxel changes in src/compare.py (TASK-179).

A non-binary mask whose nonzero support is identical in both sets but whose
values differ used to show up only as "changed", indistinguishable from a
geometry change. voxel_diff now reports those voxels as "value_changed".
"""

import json
from pathlib import Path

import numpy as np

import compare
from compare import Compare, voxel_diff


class _FakeAims:
    """Minimal ``soma.aims`` stand-in serving numpy volumes by path."""

    def __init__(self, objects):
        self.objects = dict(objects)

    def read(self, path):
        return self.objects[str(path)]


def _make_script(argv: list) -> Compare:
    script = Compare()
    script.args = script.parse_args(argv)
    return script


def _value_only_pair():
    """Two 4x4x4 masks: same three nonzero voxels, two of them with new values."""
    a = np.zeros((4, 4, 4))
    a[0, 0, 0], a[1, 1, 1], a[2, 2, 2] = 1.0, 2.0, 3.0
    b = a.copy()
    b[0, 0, 0], b[1, 1, 1] = 5.0, 7.0
    return a, b


def _install_pair(tmp_path: Path, monkeypatch, rel: str, a: np.ndarray, b: np.ndarray):
    dirs = []
    objects = {}
    for side, arr in (("a", a), ("b", b)):
        path = tmp_path / side / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        objects[str(path.resolve())] = arr
        dirs.append(tmp_path / side)
    monkeypatch.setattr(compare, "aims", _FakeAims(objects))
    return dirs[0], dirs[1]


def _run_masks(tmp_path: Path, monkeypatch, a: np.ndarray, b: np.ndarray):
    dir_a, dir_b = _install_pair(tmp_path, monkeypatch, "L/S.F.sup._left.nii.gz", a, b)
    out = tmp_path / "report.json"
    script = _make_script(["masks", "--set_a", str(dir_a), "--set_b", str(dir_b), "--output", str(out)])
    assert script.run() == 0
    return json.loads(out.read_text())


class TestReqCompare86ValueChangedCount:
    """[REQ-COMPARE-86](REQUIREMENTS.md#req-compare-86) value_changed counts same-support value changes."""

    def test_value_only_change_is_counted(self):
        a, b = _value_only_pair()
        assert voxel_diff(a, b)["value_changed"] == 2

    def test_scaled_volume_counts_whole_support(self):
        a, _ = _value_only_pair()
        assert voxel_diff(a, 2 * a)["value_changed"] == 3

    def test_geometry_change_is_not_counted(self):
        a = np.array([[[1.0, 0.0, 4.0]]])
        b = np.array([[[0.0, 3.0, 4.0]]])
        assert voxel_diff(a, b)["value_changed"] == 0

    def test_identical_volumes_count_zero(self):
        a, _ = _value_only_pair()
        assert voxel_diff(a, a.copy()) == {"changed": 0, "added": 0, "removed": 0, "value_changed": 0}


class TestReqCompare87ChangedInvariant:
    """[REQ-COMPARE-87](REQUIREMENTS.md#req-compare-87) changed == added + removed + value_changed."""

    def test_mixed_volume_partitions_changed(self):
        a = np.array([[[0.0, 1.0, 2.0, 3.0, 0.0, 4.0]]])
        b = np.array([[[1.0, 0.0, 5.0, 3.0, 0.0, 6.0]]])
        assert voxel_diff(a, b) == {"changed": 4, "added": 1, "removed": 1, "value_changed": 2}

    def test_seeded_random_volumes_partition_changed(self):
        rng = np.random.default_rng(179)
        a = rng.integers(0, 4, size=(6, 6, 6)).astype(np.float64)
        b = rng.integers(0, 4, size=(6, 6, 6)).astype(np.float64)
        d = voxel_diff(a, b)
        assert d["value_changed"] > 0
        assert d["changed"] == d["added"] + d["removed"] + d["value_changed"]


class TestReqCompare88ReportValueChanged:
    """[REQ-COMPARE-88](REQUIREMENTS.md#req-compare-88) diffs_per_mask entries carry value_changed."""

    def test_masks_mode_entry_carries_value_changed(self, tmp_path, monkeypatch):
        a, b = _value_only_pair()
        report = _run_masks(tmp_path, monkeypatch, a, b)
        assert report["diffs_per_mask"]["L/S.F.sup._left.nii.gz"] == {
            "changed": 2,
            "added": 0,
            "removed": 0,
            "value_changed": 2,
        }

    def test_cortical_tiles_mode_entry_carries_value_changed(self, tmp_path, monkeypatch):
        rel = "REGION/mask/Lmask_skeleton.nii.gz"
        a, b = _value_only_pair()
        dir_a, dir_b = _install_pair(tmp_path, monkeypatch, rel, a, b)
        out = tmp_path / "tiles.json"
        script = _make_script(["cortical_tiles", "--set_a", str(dir_a), "--set_b", str(dir_b), "--output", str(out)])
        assert script.run() == 0
        assert json.loads(out.read_text())["diffs_per_mask"][rel]["value_changed"] == 2


class TestReqCompare89BucketFromChanged:
    """[REQ-COMPARE-89](REQUIREMENTS.md#req-compare-89) value-only masks are bucketed by total changed."""

    def test_value_only_mask_bucketed_by_changed(self, tmp_path, monkeypatch):
        a, b = _value_only_pair()
        report = _run_masks(tmp_path, monkeypatch, a, b)
        assert report["diff_by_bucket"] == {"2-3vox": ["L/S.F.sup._left.nii.gz"]}


class TestReqCompare90UnchangedCount:
    """[REQ-COMPARE-90](REQUIREMENTS.md#req-compare-90) value-only masks are not reported as unchanged."""

    def test_value_only_mask_not_counted_unchanged(self, tmp_path, monkeypatch, capsys):
        a, b = _value_only_pair()
        _run_masks(tmp_path, monkeypatch, a, b)
        assert "Unchanged masks: 0/1" in capsys.readouterr().out
