#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for src/run_cka.py

The CKA subprocess is never launched: ``execute_command`` is replaced by a
recorder so that only path discovery and pair-matching logic is exercised.
"""

import sys

import pytest

import run_cka
from run_cka import CKA_MODULE, RunCKA

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_script(argv: list, record: list | None = None) -> RunCKA:
    """Build a RunCKA with parsed args and a non-executing execute_command."""
    script = RunCKA()
    script.args = script.parse_args(argv)
    calls = record if record is not None else []
    script.executed = calls

    def _fake_execute(cmd, shell=False):
        calls.append(cmd)
        return 0

    script.execute_command = _fake_execute
    return script


def _touch(path, text="ID,dim0\nsub-01,1.0\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestArgumentParsing:
    def test_required_arguments(self, tmp_path):
        script = _make_script(["--path_a", "a", "--path_b", "b", "--output_dir", str(tmp_path)])
        assert script.args.path_a == "a"
        assert script.args.path_b == "b"
        assert script.args.output_dir == str(tmp_path)

    def test_defaults(self, tmp_path):
        script = _make_script(["--path_a", "a", "--path_b", "b", "--output_dir", str(tmp_path)])
        assert script.args.name_a == "A"
        assert script.args.name_b == "B"
        assert script.args.subpath_a == "full_embeddings.csv"
        assert script.args.subpath_b == "full_embeddings.csv"
        assert script.args.subject_column == "ID"
        assert script.args.region is None

    def test_output_dir_is_required(self):
        script = RunCKA()
        with pytest.raises(SystemExit):
            script.parse_args(["--path_a", "a", "--path_b", "b"])


# ---------------------------------------------------------------------------
# _cka_cmd / _run_pairs
# ---------------------------------------------------------------------------


class TestCkaCommand:
    def test_command_shape(self, tmp_path):
        script = _make_script(
            [
                "--path_a",
                "a",
                "--path_b",
                "b",
                "--output_dir",
                str(tmp_path),
                "--name_a",
                "trained",
                "--name_b",
                "ref",
                "--subject_column",
                "Subject",
            ]
        )
        cmd = script._cka_cmd("/x/a.csv", "/x/b.csv", "/out")
        assert cmd == [
            sys.executable,
            "-m",
            CKA_MODULE,
            "trained:/x/a.csv",
            "ref:/x/b.csv",
            "--output-dir",
            "/out",
            "--subject-column",
            "Subject",
        ]


class TestRunPairs:
    def test_counts_every_pair(self, tmp_path):
        script = _make_script(["--path_a", "a", "--path_b", "b", "--output_dir", str(tmp_path)])
        pairs = [("r1", "a1", "b1"), ("r2", "a2", "b2")]
        assert script._run_pairs(pairs) == 2
        assert len(script.executed) == 2

    def test_warns_but_continues_on_failure(self, tmp_path, capsys):
        script = _make_script(["--path_a", "a", "--path_b", "b", "--output_dir", str(tmp_path)])
        script.execute_command = lambda cmd, shell=False: 3
        assert script._run_pairs([("r1", "a1", "b1")]) == 1
        assert "CKA failed for r1" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Single-pair mode
# ---------------------------------------------------------------------------


class TestSinglePairMode:
    def test_runs_once_on_two_files(self, tmp_path):
        a = _touch(tmp_path / "a.csv")
        b = _touch(tmp_path / "b.csv")
        out = tmp_path / "out"
        script = _make_script(["--path_a", str(a), "--path_b", str(b), "--output_dir", str(out)])
        assert script.run() == 0
        assert len(script.executed) == 1
        assert out.is_dir()

    def test_missing_path_a_raises(self, tmp_path):
        b = _touch(tmp_path / "b.csv")
        script = _make_script(
            ["--path_a", str(tmp_path / "missing.csv"), "--path_b", str(b), "--output_dir", str(tmp_path / "o")]
        )
        with pytest.raises(FileNotFoundError, match="--path_a not found"):
            script.run()

    def test_missing_path_b_raises(self, tmp_path):
        a = _touch(tmp_path / "a.csv")
        script = _make_script(
            ["--path_a", str(a), "--path_b", str(tmp_path / "missing.csv"), "--output_dir", str(tmp_path / "o")]
        )
        with pytest.raises(FileNotFoundError, match="--path_b not found"):
            script.run()


class TestMixedFileDirectory:
    def test_file_a_directory_b_raises(self, tmp_path):
        a = _touch(tmp_path / "a.csv")
        d = tmp_path / "dir_b"
        d.mkdir()
        script = _make_script(["--path_a", str(a), "--path_b", str(d), "--output_dir", str(tmp_path / "o")])
        with pytest.raises(NotADirectoryError, match="--path_a must be a directory"):
            script.run()

    def test_directory_a_file_b_raises(self, tmp_path):
        d = tmp_path / "dir_a"
        (d / "sub").mkdir(parents=True)
        b = _touch(tmp_path / "b.csv")
        script = _make_script(["--path_a", str(d), "--path_b", str(b), "--output_dir", str(tmp_path / "o")])
        with pytest.raises(NotADirectoryError, match="--path_b must be a directory"):
            script.run()


# ---------------------------------------------------------------------------
# Nested-directory mode
# ---------------------------------------------------------------------------


def _nested_tree(root, regions, subpath="full_embeddings.csv"):
    for region in regions:
        _touch(root / region / subpath)
    return root


class TestNestedMode:
    def test_pairs_every_common_region(self, tmp_path):
        a = _nested_tree(tmp_path / "a", ["SC_left", "SC_right"])
        b = _nested_tree(tmp_path / "b", ["SC_left", "SC_right"])
        script = _make_script(["--path_a", str(a), "--path_b", str(b), "--output_dir", str(tmp_path / "o")])
        assert script.run() == 0
        assert len(script.executed) == 2

    def test_skips_region_missing_in_a(self, tmp_path, capsys):
        a = tmp_path / "a"
        (a / "SC_left").mkdir(parents=True)  # directory but no CSV
        _touch(a / "SC_right" / "full_embeddings.csv")
        b = _nested_tree(tmp_path / "b", ["SC_left", "SC_right"])
        script = _make_script(["--path_a", str(a), "--path_b", str(b), "--output_dir", str(tmp_path / "o")])
        assert script.run() == 0
        assert "[skip] SC_left" in capsys.readouterr().out
        assert len(script.executed) == 1

    def test_skips_region_missing_in_b(self, tmp_path, capsys):
        a = _nested_tree(tmp_path / "a", ["SC_left", "SC_right"])
        b = _nested_tree(tmp_path / "b", ["SC_right"])
        (tmp_path / "b" / "SC_left").mkdir(parents=True)
        script = _make_script(["--path_a", str(a), "--path_b", str(b), "--output_dir", str(tmp_path / "o")])
        assert script.run() == 0
        assert "[skip] SC_left" in capsys.readouterr().out
        assert len(script.executed) == 1

    def test_region_filter_selects_one(self, tmp_path):
        a = _nested_tree(tmp_path / "a", ["SC_left", "SC_right"])
        b = _nested_tree(tmp_path / "b", ["SC_left", "SC_right"])
        script = _make_script(
            ["--path_a", str(a), "--path_b", str(b), "--output_dir", str(tmp_path / "o"), "--region", "SC_left"]
        )
        assert script.run() == 0
        assert len(script.executed) == 1

    def test_unknown_region_raises(self, tmp_path):
        a = _nested_tree(tmp_path / "a", ["SC_left"])
        b = _nested_tree(tmp_path / "b", ["SC_left"])
        script = _make_script(
            ["--path_a", str(a), "--path_b", str(b), "--output_dir", str(tmp_path / "o"), "--region", "NOPE"]
        )
        with pytest.raises(ValueError, match="not found under"):
            script.run()

    def test_custom_subpaths_are_used(self, tmp_path):
        a = _nested_tree(tmp_path / "a", ["SC_left"], subpath="run_a/emb.csv")
        b = _nested_tree(tmp_path / "b", ["SC_left"], subpath="run_b/emb.csv")
        script = _make_script(
            [
                "--path_a",
                str(a),
                "--path_b",
                str(b),
                "--output_dir",
                str(tmp_path / "o"),
                "--subpath_a",
                "run_a/emb.csv",
                "--subpath_b",
                "run_b/emb.csv",
            ]
        )
        assert script.run() == 0
        assert script.executed[0][3].endswith("run_a/emb.csv")

    def test_no_matching_pairs_raises(self, tmp_path):
        a = tmp_path / "a"
        (a / "SC_left").mkdir(parents=True)
        b = tmp_path / "b"
        (b / "SC_left").mkdir(parents=True)
        script = _make_script(["--path_a", str(a), "--path_b", str(b), "--output_dir", str(tmp_path / "o")])
        with pytest.raises(RuntimeError, match="No matching embedding pairs"):
            script.run()


# ---------------------------------------------------------------------------
# Flat-directory mode
# ---------------------------------------------------------------------------


class TestFlatMode:
    def test_matches_identical_basenames(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        for d in (a, b):
            _touch(d / "SC_left_embeddings.csv")
            _touch(d / "SC_right_embeddings.csv")
        script = _make_script(["--path_a", str(a), "--path_b", str(b), "--output_dir", str(tmp_path / "o")])
        assert script.run() == 0
        assert len(script.executed) == 2

    def test_reports_unmatched_files_in_a(self, tmp_path, capsys):
        a, b = tmp_path / "a", tmp_path / "b"
        _touch(a / "SC_left_embeddings.csv")
        _touch(a / "extra_embeddings.csv")
        _touch(b / "SC_left_embeddings.csv")
        script = _make_script(["--path_a", str(a), "--path_b", str(b), "--output_dir", str(tmp_path / "o")])
        assert script.run() == 0
        assert "1 file(s) in path_a have no match" in capsys.readouterr().out

    def test_region_filter_on_basename(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        for d in (a, b):
            _touch(d / "SC_left_embeddings.csv")
            _touch(d / "FIP_right_embeddings.csv")
        script = _make_script(
            ["--path_a", str(a), "--path_b", str(b), "--output_dir", str(tmp_path / "o"), "--region", "FIP"]
        )
        assert script.run() == 0
        assert len(script.executed) == 1

    def test_region_filter_without_match_raises(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        for d in (a, b):
            _touch(d / "SC_left_embeddings.csv")
        script = _make_script(
            ["--path_a", str(a), "--path_b", str(b), "--output_dir", str(tmp_path / "o"), "--region", "NOPE"]
        )
        with pytest.raises(ValueError, match="No CSV matching region"):
            script.run()

    def test_models_cache_subdir_does_not_trigger_nested_mode(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        for d in (a, b):
            _touch(d / "SC_left_embeddings.csv")
            (d / "models_cache").mkdir(parents=True, exist_ok=True)
        script = _make_script(["--path_a", str(a), "--path_b", str(b), "--output_dir", str(tmp_path / "o")])
        assert script.run() == 0
        assert len(script.executed) == 1


class TestFlatModeModelIdFallback:
    def test_matches_by_embedded_model_id(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        _touch(a / "SC_left_name07-58-00--111_embeddings.csv")
        _touch(b / "other_naming_name07-58-00--111_embeddings.csv")
        script = _make_script(["--path_a", str(a), "--path_b", str(b), "--output_dir", str(tmp_path / "o")])
        assert script.run() == 0
        assert len(script.executed) == 1

    def test_reports_unmatched_model_ids(self, tmp_path, capsys):
        a, b = tmp_path / "a", tmp_path / "b"
        _touch(a / "SC_left_name07-58-00--111_embeddings.csv")
        _touch(a / "SC_right_name08-00-00--222_embeddings.csv")
        _touch(b / "other_name07-58-00--111_embeddings.csv")
        script = _make_script(["--path_a", str(a), "--path_b", str(b), "--output_dir", str(tmp_path / "o")])
        assert script.run() == 0
        assert "no model-ID match" in capsys.readouterr().out

    def test_region_filter_on_model_id_match(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        _touch(a / "SC_left_name07-58-00--111_embeddings.csv")
        _touch(b / "x_name07-58-00--111_embeddings.csv")
        script = _make_script(
            ["--path_a", str(a), "--path_b", str(b), "--output_dir", str(tmp_path / "o"), "--region", "SC_left"]
        )
        assert script.run() == 0
        assert len(script.executed) == 1

    def test_region_filter_without_model_id_match_raises(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        _touch(a / "SC_left_name07-58-00--111_embeddings.csv")
        _touch(b / "x_name07-58-00--111_embeddings.csv")
        script = _make_script(
            ["--path_a", str(a), "--path_b", str(b), "--output_dir", str(tmp_path / "o"), "--region", "NOPE"]
        )
        with pytest.raises(ValueError, match="No CSV matching region"):
            script.run()

    def test_no_common_model_ids_raises(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        _touch(a / "SC_left_name07-58-00--111_embeddings.csv")
        _touch(b / "SC_left_name09-00-00--999_embeddings.csv")
        script = _make_script(["--path_a", str(a), "--path_b", str(b), "--output_dir", str(tmp_path / "o")])
        with pytest.raises(RuntimeError, match="No matching embedding pairs"):
            script.run()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_runs_single_pair(self, tmp_path, monkeypatch):
        a = _touch(tmp_path / "a.csv")
        b = _touch(tmp_path / "b.csv")
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_cka.py", "--path_a", str(a), "--path_b", str(b), "--output_dir", str(tmp_path / "o")],
        )
        monkeypatch.setattr("champollion_utils.script_builder.check_for_updates", lambda *x, **k: None, raising=False)
        monkeypatch.setattr(RunCKA, "execute_command", lambda self, cmd, shell=False: 0)
        assert run_cka.main() == 0
