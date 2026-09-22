#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for src/compare_databases.py

Both heavy dependencies are replaced: ``soma.aims`` (stubbed in conftest.py) is
patched with a fake reader, and ``deep_folding.brainvisa.utils.subjects`` is
injected as a stub module so no BrainVISA installation is needed.
"""

import csv
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

import compare_databases
from compare_databases import CompareDatabases, _get_subject_voxel_counts, _join, _mask_stats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeAims:
    """Fake ``soma.aims`` returning pre-registered graphs / volumes by path."""

    def __init__(self, objects: dict):
        self.objects = objects

    def read(self, path):
        return self.objects[str(path)]


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
def fake_aims(monkeypatch):
    """Replace the ``aims`` attribute on the stubbed ``soma`` module."""

    def _install(objects):
        fake = _FakeAims(objects)
        monkeypatch.setattr(sys.modules["soma"], "aims", fake)
        return fake

    return _install


@pytest.fixture
def stub_deep_folding(monkeypatch):
    """Inject a stub ``deep_folding.brainvisa.utils.subjects`` module."""

    def _install(subjects):
        subjects_mod = types.ModuleType("deep_folding.brainvisa.utils.subjects")
        subjects_mod.get_all_subjects_as_dictionary = MagicMock(return_value=subjects)
        for name, mod in [
            ("deep_folding", types.ModuleType("deep_folding")),
            ("deep_folding.brainvisa", types.ModuleType("deep_folding.brainvisa")),
            ("deep_folding.brainvisa.utils", types.ModuleType("deep_folding.brainvisa.utils")),
            ("deep_folding.brainvisa.utils.subjects", subjects_mod),
        ]:
            monkeypatch.setitem(sys.modules, name, mod)
        return subjects_mod

    return _install


def _make_script(argv: list) -> CompareDatabases:
    script = CompareDatabases()
    script.args = script.parse_args(argv)
    return script


def _base_argv(tmp_path, **overrides):
    argv = [
        "--labeled_subjects_dir",
        str(tmp_path),
        "--path_to_graph_a",
        "folds/3.3/base2018_manual",
        "--path_to_graph_b",
        "folds/3.3/base2018b_manual",
    ]
    for key, value in overrides.items():
        argv += [f"--{key}", str(value)]
    return argv


# ---------------------------------------------------------------------------
# Module-level workers
# ---------------------------------------------------------------------------


class TestJoin:
    def test_joins_path_components(self):
        assert _join("a", "b", "c.txt") == "a/b/c.txt"


class TestGetSubjectVoxelCounts:
    def test_returns_none_when_no_graph_matches(self, tmp_path):
        sub = {"subject": "sub-01", "dir": str(tmp_path), "graph_file": "L*.arg"}
        assert _get_subject_voxel_counts(sub, str(tmp_path)) == ("sub-01", None)

    def test_sums_buckets_per_sulcus(self, tmp_path, fake_aims):
        graph_path = tmp_path / "Lsub-01.arg"
        graph_path.touch()
        graph = _FakeGraph(
            [
                _FakeVertex("S.C._left", {"aims_ss": 3, "aims_bottom": 2, "aims_other": 1}),
                _FakeVertex("S.C._left", {"aims_ss": 4}),
                _FakeVertex("F.C.M._left", {"aims_ss": 5}),
            ]
        )
        fake_aims({str(graph_path): graph})
        sub = {"subject": "sub-01", "dir": str(tmp_path), "graph_file": "L*.arg"}
        name, counts = _get_subject_voxel_counts(sub, str(tmp_path))
        assert name == "sub-01"
        assert counts == {"S.C._left": 10, "F.C.M._left": 5}

    def test_unnamed_vertices_are_ignored(self, tmp_path, fake_aims):
        graph_path = tmp_path / "Lsub-02.arg"
        graph_path.touch()
        fake_aims({str(graph_path): _FakeGraph([_FakeVertex(None, {"aims_ss": 9})])})
        sub = {"subject": "sub-02", "dir": str(tmp_path), "graph_file": "L*.arg"}
        assert _get_subject_voxel_counts(sub, str(tmp_path)) == ("sub-02", {})

    def test_brainvisa_dir_added_to_syspath(self, tmp_path, fake_aims):
        marker = str(tmp_path / "bv_dir_databases")
        sub = {"subject": "sub-03", "dir": str(tmp_path), "graph_file": "NOPE*.arg"}
        _get_subject_voxel_counts(sub, marker)
        assert marker in sys.path
        sys.path.remove(marker)


class TestMaskStats:
    def test_collects_max_sum_and_nonzero(self, tmp_path, fake_aims):
        mask_dir = tmp_path / "masks"
        (mask_dir / "L").mkdir(parents=True)
        path = mask_dir / "L" / "OCCIPITAL_left.nii.gz"
        path.touch()
        fake_aims({str(path): np.array([[[0, 2, 3]]])})
        stats = _mask_stats(str(mask_dir), str(tmp_path))
        assert stats == {"L/OCCIPITAL_left.nii.gz": (3, 5, 2)}

    def test_empty_directory(self, tmp_path, fake_aims):
        fake_aims({})
        assert _mask_stats(str(tmp_path), str(tmp_path)) == {}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestArgumentParsing:
    def test_defaults(self, tmp_path):
        script = _make_script(_base_argv(tmp_path))
        assert script.args.label_a == "A"
        assert script.args.label_b == "B"
        assert script.args.side == "both"
        assert script.args.masks_a is None
        assert script.args.masks_b is None
        assert script.args.output == "./db_comparison.csv"
        assert script.args.njobs is None

    def test_explicit_values(self, tmp_path):
        script = _make_script(_base_argv(tmp_path, label_a="x", label_b="y", side="L", njobs=3))
        assert script.args.label_a == "x"
        assert script.args.side == "L"
        assert script.args.njobs == 3

    def test_labeled_subjects_dir_required(self):
        script = CompareDatabases()
        with pytest.raises(SystemExit):
            script.parse_args(["--path_to_graph_a", "a", "--path_to_graph_b", "b"])


# ---------------------------------------------------------------------------
# _load_database
# ---------------------------------------------------------------------------


class TestLoadDatabase:
    def test_aggregates_subjects_over_sides(self, tmp_path, stub_deep_folding, monkeypatch):
        subjects_mod = stub_deep_folding([{"subject": "sub-01", "dir": "d", "graph_file": "g"}])
        monkeypatch.setattr(
            compare_databases,
            "_get_subject_voxel_counts",
            lambda sub, bv: (sub["subject"], {"S.C.": 7}),
        )
        script = _make_script(_base_argv(tmp_path))
        data = script._load_database("folds/3.3/a", ["L", "R"], str(tmp_path), 1, str(tmp_path))
        assert data == {"sub-01": {"S.C.": 7}}
        assert subjects_mod.get_all_subjects_as_dictionary.call_count == 2

    def test_pattern_includes_graph_subpath(self, tmp_path, stub_deep_folding, monkeypatch):
        subjects_mod = stub_deep_folding([])
        monkeypatch.setattr(compare_databases, "_get_subject_voxel_counts", lambda sub, bv: (None, None))
        script = _make_script(_base_argv(tmp_path))
        script._load_database("folds/3.3/a", ["L"], str(tmp_path), 1, str(tmp_path))
        patterns = subjects_mod.get_all_subjects_as_dictionary.call_args.args[1]
        assert patterns == ["%(subject)s/folds/3.3/a/%(side)s%(subject)s*.arg"]

    def test_warns_when_subject_has_no_graph(self, tmp_path, stub_deep_folding, monkeypatch, capsys):
        stub_deep_folding([{"subject": "sub-99", "dir": "d", "graph_file": "g"}])
        monkeypatch.setattr(compare_databases, "_get_subject_voxel_counts", lambda sub, bv: ("sub-99", None))
        script = _make_script(_base_argv(tmp_path))
        assert script._load_database("p", ["L"], str(tmp_path), 1, str(tmp_path)) == {}
        assert "no graph for sub-99" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


@pytest.fixture
def two_campaigns(monkeypatch):
    """Patch _load_database to return two fixed campaigns (A then B)."""
    data_a = {"sub-01": {"S.C.": 100}, "sub-02": {"S.C.": 200, "F.C.M.": 50}}
    data_b = {"sub-02": {"S.C.": 400, "F.C.M.": 50}, "sub-03": {"S.C.": 300}}
    calls = iter([data_a, data_b])
    monkeypatch.setattr(CompareDatabases, "_load_database", lambda self, *a, **k: next(calls))
    return data_a, data_b


class TestRun:
    def test_writes_csv_with_per_sulcus_rows(self, tmp_path, two_campaigns):
        out = tmp_path / "out" / "db.csv"
        script = _make_script(_base_argv(tmp_path, output=out, label_a="A", label_b="B"))
        assert script.run() == 0

        with open(out, newline="") as f:
            rows = {r["sulcus"]: r for r in csv.DictReader(f)}
        assert sorted(rows) == ["F.C.M.", "S.C."]
        assert rows["S.C."]["N_A"] == "2"
        assert rows["S.C."]["N_B"] == "2"
        assert rows["S.C."]["vox_per_subject_A"] == "150.0"
        assert rows["S.C."]["vox_per_subject_B"] == "350.0"
        assert float(rows["S.C."]["vox_ratio_B_over_A"]) == pytest.approx(2.333, abs=1e-3)

    def test_reports_subject_set_overlap(self, tmp_path, two_campaigns, capsys):
        script = _make_script(_base_argv(tmp_path, output=tmp_path / "db.csv"))
        script.run()
        printed = capsys.readouterr().out
        assert "Subjects in A only:  1" in printed
        assert "Subjects in B only:  1" in printed
        assert "Subjects in both:        1" in printed

    def test_single_side_is_honoured(self, tmp_path, monkeypatch):
        seen = []

        def _record(self, path_to_graph, sides, subjects_dir, njobs, brainvisa_dir):
            seen.append(sides)
            return {}

        monkeypatch.setattr(CompareDatabases, "_load_database", _record)
        script = _make_script(_base_argv(tmp_path, side="L", output=tmp_path / "db.csv"))
        assert script.run() == 0
        assert seen == [["L"], ["L"]]

    def test_njobs_override_is_forwarded(self, tmp_path, monkeypatch):
        seen = []

        def _record(self, path_to_graph, sides, subjects_dir, njobs, brainvisa_dir):
            seen.append(njobs)
            return {}

        monkeypatch.setattr(CompareDatabases, "_load_database", _record)
        script = _make_script(_base_argv(tmp_path, njobs=4, output=tmp_path / "db.csv"))
        script.run()
        assert seen == [4, 4]

    def test_njobs_defaults_to_cpu_budget(self, tmp_path, monkeypatch):
        seen = []

        def _record(self, path_to_graph, sides, subjects_dir, njobs, brainvisa_dir):
            seen.append(njobs)
            return {}

        monkeypatch.setattr(CompareDatabases, "_load_database", _record)
        script = _make_script(_base_argv(tmp_path, output=tmp_path / "db.csv"))
        script.run()
        assert 1 <= seen[0] <= 22

    def test_no_csv_written_when_no_sulci(self, tmp_path, monkeypatch):
        monkeypatch.setattr(CompareDatabases, "_load_database", lambda self, *a, **k: {})
        out = tmp_path / "empty.csv"
        script = _make_script(_base_argv(tmp_path, output=out))
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
        monkeypatch.setattr(compare_databases, "_mask_stats", lambda d, bv: next(stats))

        out = tmp_path / "masked.csv"
        script = _make_script(
            _base_argv(tmp_path, output=out, masks_a=masks_a, masks_b=masks_b, label_a="A", label_b="B")
        )
        assert script.run() == 0

        with open(out, newline="") as f:
            rows = {r["sulcus"]: r for r in csv.DictReader(f)}
        assert rows["S.C."]["mask_max_A"] == "10"
        assert rows["S.C."]["mask_max_B"] == "20"
        assert rows["S.C."]["mask_sum_per_sub_A"] == "200.0"
        assert rows["S.C."]["mask_sum_per_sub_B"] == "400.0"

    def test_mask_dirs_ignored_when_absent(self, tmp_path, two_campaigns, monkeypatch):
        called = MagicMock()
        monkeypatch.setattr(compare_databases, "_mask_stats", called)
        out = tmp_path / "nomask.csv"
        script = _make_script(_base_argv(tmp_path, output=out, masks_a=tmp_path / "nope"))
        assert script.run() == 0
        called.assert_not_called()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_builds_and_runs(self, tmp_path, two_campaigns, monkeypatch):
        out = tmp_path / "main.csv"
        monkeypatch.setattr(sys, "argv", ["compare_databases.py"] + _base_argv(tmp_path, output=out))
        monkeypatch.setattr("champollion_utils.script_builder.check_for_updates", lambda *a, **k: None, raising=False)
        assert compare_databases.main() == 0
        assert out.is_file()
