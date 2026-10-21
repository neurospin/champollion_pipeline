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
from unittest.mock import MagicMock

import pytest

import compare
import compare_databases
from compare_databases import CompareDatabases

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# run()
# ---------------------------------------------------------------------------


@pytest.fixture  # noqa: V103
def two_campaigns(monkeypatch):
    """Patch compare.Compare._load_database (the single implementation) to return two fixed campaigns (A then B)."""
    data_a = {"sub-01": {"S.C.": 100}, "sub-02": {"S.C.": 200, "F.C.M.": 50}}
    data_b = {"sub-02": {"S.C.": 400, "F.C.M.": 50}, "sub-03": {"S.C.": 300}}
    calls = iter([data_a, data_b])
    monkeypatch.setattr(compare.Compare, "_load_database", lambda self, *a, **k: next(calls))
    return data_a, data_b


class TestRun:
    @pytest.mark.usefixtures("two_campaigns")
    def test_writes_csv_with_per_sulcus_rows(self, tmp_path):
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

    @pytest.mark.usefixtures("two_campaigns")
    def test_reports_subject_set_overlap(self, tmp_path, capsys):
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

        monkeypatch.setattr(compare.Compare, "_load_database", _record)
        script = _make_script(_base_argv(tmp_path, side="L", output=tmp_path / "db.csv"))
        assert script.run() == 0
        assert seen == [["L"], ["L"]]

    def test_njobs_override_is_forwarded(self, tmp_path, monkeypatch):
        seen = []

        def _record(self, path_to_graph, sides, subjects_dir, njobs, brainvisa_dir):
            seen.append(njobs)
            return {}

        monkeypatch.setattr(compare.Compare, "_load_database", _record)
        script = _make_script(_base_argv(tmp_path, njobs=4, output=tmp_path / "db.csv"))
        script.run()
        assert seen == [4, 4]

    def test_njobs_defaults_to_cpu_budget(self, tmp_path, monkeypatch):
        seen = []

        def _record(self, path_to_graph, sides, subjects_dir, njobs, brainvisa_dir):
            seen.append(njobs)
            return {}

        monkeypatch.setattr(compare.Compare, "_load_database", _record)
        script = _make_script(_base_argv(tmp_path, output=tmp_path / "db.csv"))
        script.run()
        assert 1 <= seen[0] <= 22

    def test_no_csv_written_when_no_sulci(self, tmp_path, monkeypatch):
        monkeypatch.setattr(compare.Compare, "_load_database", lambda self, *a, **k: {})
        out = tmp_path / "empty.csv"
        script = _make_script(_base_argv(tmp_path, output=out))
        assert script.run() == 0
        assert not out.exists()

    @pytest.mark.usefixtures("two_campaigns")
    def test_mask_stats_columns_are_merged(self, tmp_path, monkeypatch):
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
        monkeypatch.setattr(compare, "_mask_stats", lambda d, _bv: next(stats))

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

    @pytest.mark.usefixtures("two_campaigns")
    def test_mask_dirs_ignored_when_absent(self, tmp_path, monkeypatch):
        called = MagicMock()
        monkeypatch.setattr(compare, "_mask_stats", called)
        out = tmp_path / "nomask.csv"
        script = _make_script(_base_argv(tmp_path, output=out, masks_a=tmp_path / "nope"))
        assert script.run() == 0
        called.assert_not_called()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    @pytest.mark.usefixtures("two_campaigns")
    def test_main_builds_and_runs(self, tmp_path, monkeypatch):
        out = tmp_path / "main.csv"
        monkeypatch.setattr(sys, "argv", ["compare_databases.py"] + _base_argv(tmp_path, output=out))
        monkeypatch.setattr("champollion_utils.script_builder.check_for_updates", lambda *a, **k: None, raising=False)
        assert compare_databases.main() == 0
        assert out.is_file()
