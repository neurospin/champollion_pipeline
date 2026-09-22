#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for src/compare.py (unified masks / cortical_tiles / databases tool).

PyAIMS is stubbed in conftest.py.  Here ``compare.aims`` is replaced by a fake
reader serving small in-memory numpy volumes, and the database mode's BrainVISA
dependencies are injected as stub modules, so nothing touches real data.
"""

import csv
import json
import sys
import types
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import compare
from compare import (
    Compare,
    _get_subject_voxel_counts,
    _mask_stats,
    bucket_label,
    load_mask_vol,
    save_xor_vol,
    sort_buckets,
    visualise_mask_diffs,
    voxel_diff,
    wasserstein_distance,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeVolume:
    """Stand-in for an aims Volume carrying a header."""

    def __init__(self, array, header=None):
        self.array = np.asarray(array)
        self._header = header if header is not None else {"voxel_size": [2.0, 2.0, 2.0]}
        self.copied_header = None

    def header(self):
        return self._header

    def copyHeaderFrom(self, header):  # noqa: N802 - mirrors the PyAIMS API
        self.copied_header = header

    def __array__(self, dtype=None):
        return self.array.astype(dtype) if dtype is not None else self.array


class _FakeAims:
    """Fake ``soma.aims`` recording writes and serving registered volumes."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.written = []

    def read(self, path):
        return self.objects[str(path)]

    def Volume(self, array):  # noqa: N802 - mirrors the PyAIMS API
        return _FakeVolume(array)

    def write(self, vol, path):
        self.written.append((vol, str(path)))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()


def _point_volume(index, shape=(4, 4, 4), value=1.0):
    vol = np.zeros(shape, dtype=np.float64)
    vol[index] = value
    return vol


def _make_tree(root: Path, rel_paths: list) -> Path:
    for rel in rel_paths:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return root


def _make_script(argv: list) -> Compare:
    script = Compare()
    script.args = script.parse_args(argv)
    return script


# ---------------------------------------------------------------------------
# Shared mask helpers
# ---------------------------------------------------------------------------


class TestLoadMaskVol:
    def test_squeezes_and_casts(self, monkeypatch):
        monkeypatch.setattr(compare, "aims", _FakeAims({"/m.nii.gz": np.ones((2, 2, 2, 1), dtype=np.int16)}))
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
        assert wasserstein_distance(_point_volume((0, 0, 0)), _point_volume((2, 0, 0))) == pytest.approx(2.0)

    def test_diagonal_shift(self):
        assert wasserstein_distance(_point_volume((0, 0, 0)), _point_volume((1, 1, 0))) == pytest.approx(np.sqrt(2.0))

    def test_one_empty_map_skips_axes(self):
        assert wasserstein_distance(_point_volume((1, 1, 1)), np.zeros((4, 4, 4))) == 0.0


class TestBucketLabel:
    def test_integer_step(self):
        assert bucket_label(3.7, 1.0) == "3-4vox"

    def test_fractional_step(self):
        assert bucket_label(0.7, 0.5) == "0.50-1.00vox"


class TestSortBuckets:
    def test_sorts_numerically_not_lexically(self):
        buckets = {"10-11vox": ["b"], "2-3vox": ["a"]}
        assert list(sort_buckets(buckets)) == ["2-3vox", "10-11vox"]


# ---------------------------------------------------------------------------
# save_xor_vol
# ---------------------------------------------------------------------------


class TestSaveXorVol:
    def test_writes_binary_xor_with_reference_header(self, tmp_path, monkeypatch):
        ref = _FakeVolume(np.zeros((2, 2, 2)), header={"voxel_size": [2.0, 2.0, 2.0]})
        fake = _FakeAims({"/ref.nii.gz": ref})
        monkeypatch.setattr(compare, "aims", fake)

        a = np.array([[[1.0, 0.0]], [[0.0, 0.0]]])
        b = np.array([[[0.0, 0.0]], [[0.0, 0.0]]])
        out = tmp_path / "xor" / "L" / "m.nii.gz"
        save_xor_vol("/ref.nii.gz", a, b, str(out))

        assert out.is_file()
        vol, path = fake.written[0]
        assert path == str(out)
        assert vol.array.shape[-1] == 1
        assert int(vol.array.sum()) == 1
        assert vol.copied_header == {"voxel_size": [2.0, 2.0, 2.0]}


# ---------------------------------------------------------------------------
# visualise_mask_diffs
# ---------------------------------------------------------------------------


class TestVisualiseMaskDiffs:
    def test_no_changed_masks_returns_early(self, monkeypatch, capsys):
        run = MagicMock()
        monkeypatch.setattr("subprocess.run", run)
        visualise_mask_diffs({"m": {"changed": 0}}, {}, {})
        assert "No changed masks to visualise." in capsys.readouterr().out
        run.assert_not_called()

    def test_spawns_viewer_for_top_changed_masks(self, tmp_path, monkeypatch):
        names = [f"m{i}.nii.gz" for i in range(7)]
        objects = {}
        masks_a, masks_b = {}, {}
        for i, name in enumerate(names):
            pa, pb = f"/a/{name}", f"/b/{name}"
            objects[pa] = _FakeVolume(_point_volume((0, 0, 0)))
            objects[pb] = _FakeVolume(_point_volume((1, 0, 0)))
            masks_a[name], masks_b[name] = pa, pb
        fake = _FakeAims(objects)
        monkeypatch.setattr(compare, "aims", fake)
        run = MagicMock()
        monkeypatch.setattr("subprocess.run", run)

        diffs = {name: {"changed": i + 1} for i, name in enumerate(names)}
        visualise_mask_diffs(diffs, masks_a, masks_b)

        run.assert_called_once()
        entries = json.loads(run.call_args.args[0][3])
        assert len(entries) == 5
        assert entries[0]["name"] == "m6.nii.gz"
        # Temporary XOR volumes are cleaned up afterwards.
        for entry in entries:
            assert not Path(entry["path_xor"]).exists()


# ---------------------------------------------------------------------------
# Database-mode module-level workers
# ---------------------------------------------------------------------------


class _FakeVertex:
    def __init__(self, name, bucket_sizes):
        self._data = {"name": name}
        for bucket_name, size in bucket_sizes.items():
            self._data[bucket_name] = [{f"v{i}": 1 for i in range(size)}]

    def get(self, key):
        return self._data.get(key)


class _FakeGraph:
    def __init__(self, vertices):
        self._vertices = vertices

    def vertices(self):
        return self._vertices


@pytest.fixture
def fake_soma_aims(monkeypatch):
    def _install(objects):
        fake = _FakeAims(objects)
        monkeypatch.setattr(sys.modules["soma"], "aims", fake)
        return fake

    return _install


class TestGetSubjectVoxelCounts:
    def test_returns_none_when_no_graph_matches(self, tmp_path):
        sub = {"subject": "sub-01", "dir": str(tmp_path), "graph_file": "L*.arg"}
        assert _get_subject_voxel_counts(sub, str(tmp_path)) == ("sub-01", None)

    def test_sums_buckets_per_sulcus(self, tmp_path, fake_soma_aims):
        graph_path = tmp_path / "Lsub-01.arg"
        graph_path.touch()
        fake_soma_aims(
            {
                str(graph_path): _FakeGraph(
                    [
                        _FakeVertex("S.C._left", {"aims_ss": 2, "aims_bottom": 1}),
                        _FakeVertex("S.C._left", {"aims_other": 3}),
                        _FakeVertex(None, {"aims_ss": 5}),
                    ]
                )
            }
        )
        sub = {"subject": "sub-01", "dir": str(tmp_path), "graph_file": "L*.arg"}
        assert _get_subject_voxel_counts(sub, str(tmp_path)) == ("sub-01", {"S.C._left": 6})


class TestMaskStats:
    def test_collects_max_sum_and_nonzero(self, tmp_path, fake_soma_aims):
        mask_dir = tmp_path / "masks"
        (mask_dir / "L").mkdir(parents=True)
        path = mask_dir / "L" / "OCCIPITAL_left.nii.gz"
        path.touch()
        fake_soma_aims({str(path): np.array([[[0, 2, 3]]])})
        assert _mask_stats(str(mask_dir), str(tmp_path)) == {"L/OCCIPITAL_left.nii.gz": (3, 5, 2)}

    def test_empty_directory(self, tmp_path, fake_soma_aims):
        fake_soma_aims({})
        assert _mask_stats(str(tmp_path), str(tmp_path)) == {}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestArgumentParsing:
    def test_mode_is_required(self):
        script = Compare()
        with pytest.raises(SystemExit):
            script.parse_args([])

    def test_masks_defaults(self):
        script = _make_script(["masks", "--set_a", "a", "--set_b", "b"])
        assert script.args.mode == "masks"
        assert script.args.output == "comparison_report.json"
        assert script.args.metric == "diff"
        assert script.args.bucket_step == 1.0
        assert script.args.xor_dir is None
        assert script.args.visualisation is False

    def test_cortical_tiles_adds_pattern(self):
        script = _make_script(["cortical_tiles", "--set_a", "a", "--set_b", "b"])
        assert script.args.pattern == "*mask_skeleton.nii.gz"

    def test_databases_defaults(self):
        script = _make_script(
            ["databases", "--labeled_subjects_dir", "d", "--path_to_graph_a", "a", "--path_to_graph_b", "b"]
        )
        assert script.args.label_a == "A"
        assert script.args.side == "both"
        assert script.args.output == "db_comparison.csv"
        assert script.args.njobs is None

    def test_invalid_metric_rejected(self):
        script = Compare()
        with pytest.raises(SystemExit):
            script.parse_args(["masks", "--set_a", "a", "--set_b", "b", "--metric", "nope"])


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_masks_mode(self, monkeypatch):
        script = _make_script(["masks", "--set_a", "a", "--set_b", "b"])
        monkeypatch.setattr(Compare, "_run_masks", lambda self: 42)
        assert script.run() == 42

    def test_cortical_tiles_mode(self, monkeypatch):
        script = _make_script(["cortical_tiles", "--set_a", "a", "--set_b", "b"])
        monkeypatch.setattr(Compare, "_run_cortical_tiles", lambda self: 43)
        assert script.run() == 43

    def test_databases_mode(self, monkeypatch):
        script = _make_script(
            ["databases", "--labeled_subjects_dir", "d", "--path_to_graph_a", "a", "--path_to_graph_b", "b"]
        )
        monkeypatch.setattr(Compare, "_run_databases", lambda self: 44)
        assert script.run() == 44

    def test_unknown_mode_returns_one(self, capsys):
        script = Compare()
        script.args = Namespace(mode="bogus")
        assert script.run() == 1
        assert "unknown mode 'bogus'" in capsys.readouterr().out


class TestFindMasks:
    def test_returns_relative_to_absolute_map(self, tmp_path):
        _make_tree(tmp_path, ["L/S.C.nii.gz", "R/S.C.nii.gz"])
        script = _make_script(["masks", "--set_a", "a", "--set_b", "b"])
        found = script._find_masks(tmp_path, "*/*.nii.gz")
        assert sorted(found) == ["L/S.C.nii.gz", "R/S.C.nii.gz"]
        assert found["L/S.C.nii.gz"] == str((tmp_path / "L" / "S.C.nii.gz").resolve())


# ---------------------------------------------------------------------------
# masks / cortical_tiles comparison
# ---------------------------------------------------------------------------


@pytest.fixture
def mask_sets(tmp_path, monkeypatch):
    dir_a = _make_tree(tmp_path / "a", ["L/shared.nii.gz", "L/only_a.nii.gz"])
    dir_b = _make_tree(tmp_path / "b", ["L/shared.nii.gz", "L/only_b.nii.gz"])
    objects = {
        str((dir_a / "L" / "shared.nii.gz").resolve()): _FakeVolume(_point_volume((0, 0, 0))),
        str((dir_b / "L" / "shared.nii.gz").resolve()): _FakeVolume(_point_volume((2, 0, 0))),
        str((dir_a / "L" / "only_a.nii.gz").resolve()): _FakeVolume(np.zeros((4, 4, 4))),
        str((dir_b / "L" / "only_b.nii.gz").resolve()): _FakeVolume(np.zeros((4, 4, 4))),
    }
    monkeypatch.setattr(compare, "aims", _FakeAims(objects))
    return dir_a, dir_b


class TestCompareNiftiMasks:
    def test_returns_one_when_path_missing(self, tmp_path):
        script = _make_script(["masks", "--set_a", str(tmp_path / "nope"), "--set_b", str(tmp_path)])
        assert script.run() == 1

    def test_diff_report(self, mask_sets, tmp_path):
        dir_a, dir_b = mask_sets
        out = tmp_path / "reports" / "diff.json"
        script = _make_script(["masks", "--set_a", str(dir_a), "--set_b", str(dir_b), "--output", str(out)])
        assert script.run() == 0
        report = json.loads(out.read_text())
        assert report["mode"] == "masks"
        assert report["metric"] == "diff"
        assert report["summary"]["total_common"] == 1
        assert report["summary"]["only_in_set_a"] == ["L/only_a.nii.gz"]
        assert report["diffs_per_mask"]["L/shared.nii.gz"] == {"changed": 2, "added": 1, "removed": 1}
        assert "mask_pattern" not in report
        assert "wasserstein_per_mask" not in report

    def test_wasserstein_report(self, mask_sets, tmp_path):
        dir_a, dir_b = mask_sets
        out = tmp_path / "wass.json"
        script = _make_script(
            ["masks", "--set_a", str(dir_a), "--set_b", str(dir_b), "--output", str(out), "--metric", "wasserstein"]
        )
        assert script.run() == 0
        report = json.loads(out.read_text())
        assert report["wasserstein_per_mask"]["L/shared.nii.gz"] == pytest.approx(2.0)
        assert "diffs_per_mask" not in report

    def test_both_metrics_and_console_summary(self, mask_sets, tmp_path, capsys):
        dir_a, dir_b = mask_sets
        out = tmp_path / "both.json"
        script = _make_script(
            ["masks", "--set_a", str(dir_a), "--set_b", str(dir_b), "--output", str(out), "--metric", "both"]
        )
        assert script.run() == 0
        printed = capsys.readouterr().out
        assert "Wasserstein buckets (step=1.0)" in printed
        assert "Diff buckets (step=1.0)" in printed
        assert "Unchanged masks: 0/1" in printed

    def test_shape_mismatch_is_skipped(self, tmp_path, monkeypatch, capsys):
        dir_a = _make_tree(tmp_path / "a", ["L/shared.nii.gz"])
        dir_b = _make_tree(tmp_path / "b", ["L/shared.nii.gz"])
        monkeypatch.setattr(
            compare,
            "aims",
            _FakeAims(
                {
                    str((dir_a / "L" / "shared.nii.gz").resolve()): np.zeros((4, 4, 4)),
                    str((dir_b / "L" / "shared.nii.gz").resolve()): np.zeros((2, 2, 2)),
                }
            ),
        )
        out = tmp_path / "r.json"
        script = _make_script(["masks", "--set_a", str(dir_a), "--set_b", str(dir_b), "--output", str(out)])
        assert script.run() == 0
        assert "shape mismatch" in capsys.readouterr().out
        assert json.loads(out.read_text())["diffs_per_mask"] == {}

    def test_xor_dir_writes_volumes(self, mask_sets, tmp_path):
        dir_a, dir_b = mask_sets
        out = tmp_path / "xor.json"
        xor_dir = tmp_path / "xors"
        script = _make_script(
            [
                "masks",
                "--set_a",
                str(dir_a),
                "--set_b",
                str(dir_b),
                "--output",
                str(out),
                "--xor_dir",
                str(xor_dir),
            ]
        )
        assert script.run() == 0
        assert json.loads(out.read_text())["xor_dir"] == str(xor_dir)
        assert (xor_dir / "L" / "shared.nii.gz").is_file()

    def test_cortical_tiles_mode_uses_pattern(self, tmp_path, monkeypatch):
        rel = "REGION/mask/Lmask_skeleton.nii.gz"
        dir_a = _make_tree(tmp_path / "a", [rel])
        dir_b = _make_tree(tmp_path / "b", [rel])
        monkeypatch.setattr(
            compare,
            "aims",
            _FakeAims(
                {
                    str((dir_a / rel).resolve()): _point_volume((0, 0, 0)),
                    str((dir_b / rel).resolve()): _point_volume((1, 0, 0)),
                }
            ),
        )
        out = tmp_path / "tiles.json"
        script = _make_script(["cortical_tiles", "--set_a", str(dir_a), "--set_b", str(dir_b), "--output", str(out)])
        assert script.run() == 0
        report = json.loads(out.read_text())
        assert report["mode"] == "cortical_tiles"
        assert report["mask_pattern"] == "*mask_skeleton.nii.gz"
        assert list(report["diffs_per_mask"]) == [rel]

    def test_visualisation_uses_diff_counts(self, mask_sets, tmp_path, monkeypatch):
        dir_a, dir_b = mask_sets
        seen = {}
        monkeypatch.setattr(compare, "visualise_mask_diffs", lambda diffs, ma, mb: seen.update({"diffs": diffs}))
        script = _make_script(
            [
                "masks",
                "--set_a",
                str(dir_a),
                "--set_b",
                str(dir_b),
                "--output",
                str(tmp_path / "v.json"),
                "--visualisation",
            ]
        )
        assert script.run() == 0
        assert seen["diffs"]["L/shared.nii.gz"]["changed"] == 2

    def test_visualisation_synthesises_from_wasserstein(self, mask_sets, tmp_path, monkeypatch):
        dir_a, dir_b = mask_sets
        seen = {}
        monkeypatch.setattr(compare, "visualise_mask_diffs", lambda diffs, ma, mb: seen.update({"diffs": diffs}))
        script = _make_script(
            [
                "masks",
                "--set_a",
                str(dir_a),
                "--set_b",
                str(dir_b),
                "--output",
                str(tmp_path / "v.json"),
                "--metric",
                "wasserstein",
                "--visualisation",
            ]
        )
        assert script.run() == 0
        assert seen["diffs"] == {"L/shared.nii.gz": {"changed": 1, "added": 0, "removed": 0}}


# ---------------------------------------------------------------------------
# databases mode
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_deep_folding(monkeypatch):
    def _install(subjects):
        subjects_mod = types.ModuleType("cortical_tiles.brainvisa.utils.subjects")
        subjects_mod.get_all_subjects_as_dictionary = MagicMock(return_value=subjects)
        for name, mod in [
            ("cortical_tiles", types.ModuleType("cortical_tiles")),
            ("cortical_tiles.brainvisa", types.ModuleType("cortical_tiles.brainvisa")),
            ("cortical_tiles.brainvisa.utils", types.ModuleType("cortical_tiles.brainvisa.utils")),
            ("cortical_tiles.brainvisa.utils.subjects", subjects_mod),
        ]:
            monkeypatch.setitem(sys.modules, name, mod)
        return subjects_mod

    return _install


def _db_argv(tmp_path, **overrides):
    argv = [
        "databases",
        "--labeled_subjects_dir",
        str(tmp_path),
        "--path_to_graph_a",
        "folds/3.3/a",
        "--path_to_graph_b",
        "folds/3.3/b",
    ]
    for key, value in overrides.items():
        argv += [f"--{key}", str(value)]
    return argv


class TestLoadDatabase:
    def test_aggregates_subjects_over_sides(self, tmp_path, stub_deep_folding, monkeypatch):
        subjects_mod = stub_deep_folding([{"subject": "sub-01", "dir": "d", "graph_file": "g"}])
        monkeypatch.setattr(compare, "_get_subject_voxel_counts", lambda sub, bv: (sub["subject"], {"S.C.": 7}))
        script = _make_script(_db_argv(tmp_path))
        data = script._load_database("folds/3.3/a", ["L", "R"], str(tmp_path), 1, str(tmp_path))
        assert data == {"sub-01": {"S.C.": 7}}
        assert subjects_mod.get_all_subjects_as_dictionary.call_count == 2

    def test_warns_when_subject_has_no_graph(self, tmp_path, stub_deep_folding, monkeypatch, capsys):
        stub_deep_folding([{"subject": "sub-99", "dir": "d", "graph_file": "g"}])
        monkeypatch.setattr(compare, "_get_subject_voxel_counts", lambda sub, bv: ("sub-99", None))
        script = _make_script(_db_argv(tmp_path))
        assert script._load_database("p", ["L"], str(tmp_path), 1, str(tmp_path)) == {}
        assert "no graph for sub-99" in capsys.readouterr().out


@pytest.fixture
def two_campaigns(monkeypatch):
    data_a = {"sub-01": {"S.C.": 100}, "sub-02": {"S.C.": 200, "F.C.M.": 50}}
    data_b = {"sub-02": {"S.C.": 400, "F.C.M.": 50}, "sub-03": {"S.C.": 300}}
    calls = iter([data_a, data_b])
    monkeypatch.setattr(Compare, "_load_database", lambda self, *a, **k: next(calls))
    return data_a, data_b


class TestRunDatabases:
    def test_writes_csv_with_per_sulcus_rows(self, tmp_path, two_campaigns):
        out = tmp_path / "out" / "db.csv"
        script = _make_script(_db_argv(tmp_path, output=out))
        assert script.run() == 0
        with open(out, newline="") as f:
            rows = {r["sulcus"]: r for r in csv.DictReader(f)}
        assert sorted(rows) == ["F.C.M.", "S.C."]
        assert rows["S.C."]["vox_per_subject_A"] == "150.0"
        assert rows["S.C."]["vox_per_subject_B"] == "350.0"
        assert float(rows["S.C."]["vox_ratio_B_over_A"]) == pytest.approx(2.333, abs=1e-3)

    def test_reports_subject_set_overlap(self, tmp_path, two_campaigns, capsys):
        script = _make_script(_db_argv(tmp_path, output=tmp_path / "db.csv"))
        script.run()
        printed = capsys.readouterr().out
        assert "Subjects in A only:  1" in printed
        assert "Subjects in both:        1" in printed

    def test_single_side_and_njobs_are_forwarded(self, tmp_path, monkeypatch):
        seen = []

        def _record(self, path_to_graph, sides, subjects_dir, njobs, brainvisa_dir):
            seen.append((sides, njobs))
            return {}

        monkeypatch.setattr(Compare, "_load_database", _record)
        script = _make_script(_db_argv(tmp_path, side="L", njobs=3, output=tmp_path / "db.csv"))
        assert script.run() == 0
        assert seen == [(["L"], 3), (["L"], 3)]

    def test_njobs_defaults_to_cpu_budget(self, tmp_path, monkeypatch):
        seen = []

        def _record(self, path_to_graph, sides, subjects_dir, njobs, brainvisa_dir):
            seen.append(njobs)
            return {}

        monkeypatch.setattr(Compare, "_load_database", _record)
        script = _make_script(_db_argv(tmp_path, output=tmp_path / "db.csv"))
        script.run()
        assert 1 <= seen[0] <= 22

    def test_no_csv_written_when_no_sulci(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Compare, "_load_database", lambda self, *a, **k: {})
        out = tmp_path / "empty.csv"
        script = _make_script(_db_argv(tmp_path, output=out))
        assert script.run() == 0
        assert not out.exists()

    def test_mask_stats_columns_are_merged(self, tmp_path, two_campaigns, monkeypatch):
        masks_a = tmp_path / "masks_a"
        masks_b = tmp_path / "masks_b"
        masks_a.mkdir()
        masks_b.mkdir()
        stats = iter(
            [
                {"L/S.C..nii.gz": (10, 400, 40), "L/F.C.M..nii.gz": (1, 40, 4)},
                {"L/S.C..nii.gz": (20, 800, 80), "L/F.C.M..nii.gz": (2, 80, 8)},
            ]
        )
        monkeypatch.setattr(compare, "_mask_stats", lambda d, bv: next(stats))
        out = tmp_path / "masked.csv"
        script = _make_script(_db_argv(tmp_path, output=out, masks_a=masks_a, masks_b=masks_b))
        assert script.run() == 0
        with open(out, newline="") as f:
            rows = {r["sulcus"]: r for r in csv.DictReader(f)}
        assert rows["S.C."]["mask_max_A"] == "10"
        assert rows["S.C."]["mask_sum_per_sub_B"] == "400.0"

    def test_mask_dirs_ignored_when_absent(self, tmp_path, two_campaigns, monkeypatch):
        called = MagicMock()
        monkeypatch.setattr(compare, "_mask_stats", called)
        script = _make_script(_db_argv(tmp_path, output=tmp_path / "nomask.csv", masks_a=tmp_path / "nope"))
        assert script.run() == 0
        called.assert_not_called()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_runs_masks_mode(self, mask_sets, tmp_path, monkeypatch):
        dir_a, dir_b = mask_sets
        out = tmp_path / "main.json"
        monkeypatch.setattr(
            sys,
            "argv",
            ["compare.py", "masks", "--set_a", str(dir_a), "--set_b", str(dir_b), "--output", str(out)],
        )
        monkeypatch.setattr("champollion_utils.script_builder.check_for_updates", lambda *a, **k: None, raising=False)
        assert compare.main() == 0
        assert out.is_file()
