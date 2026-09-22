#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for src/generate_umap_reference.py

The module imports ``get_model_paths`` from the champollion_V1 submodule at
import time, and ``umap`` lazily inside ``generate_umap_reference()``.  Both are
stubbed here so the tests run without the submodule and without fitting a real
UMAP model.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# The champollion_V1 submodule may not be checked out; stub its helper module
# before importing the script under test.
sys.modules.setdefault("put_together_embeddings_files", MagicMock())

import generate_umap_reference  # noqa: E402
from generate_umap_reference import GenerateUmapReference, _parse_region_hemi  # noqa: E402

generate_umap_reference_fn = generate_umap_reference.generate_umap_reference

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeReducer:
    """Stand-in for ``umap.UMAP`` that projects onto the first two columns."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, X):
        return np.asarray(X)[:, :2]


@pytest.fixture
def fake_umap(monkeypatch):
    module = MagicMock()
    module.UMAP = _FakeReducer
    monkeypatch.setitem(sys.modules, "umap", module)
    return module


@pytest.fixture
def no_disk_writes(monkeypatch):
    """Capture joblib.dump / np.save instead of writing artefacts."""
    dumps, saves = [], []
    monkeypatch.setattr(generate_umap_reference.joblib, "dump", lambda obj, path: dumps.append((obj, path)))
    monkeypatch.setattr(generate_umap_reference.np, "save", lambda path, arr: saves.append((path, arr)))
    return dumps, saves


def _make_model_dir(models_dir, region_hemi, subpath, rows=3, dims=4, run="run1"):
    """Create ``models_dir/region_hemi/run/subpath`` with a small CSV."""
    model_path = models_dir / region_hemi / run
    csv_path = model_path / subpath
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = "ID," + ",".join(f"d{i}" for i in range(dims))
    lines = [header] + [f"sub-{r:02d}," + ",".join(str(float(r + i)) for i in range(dims)) for r in range(rows)]
    csv_path.write_text("\n".join(lines) + "\n")
    return str(model_path)


def _patch_model_paths(monkeypatch, paths):
    monkeypatch.setattr(generate_umap_reference, "get_model_paths", lambda d: list(paths))


# ---------------------------------------------------------------------------
# _parse_region_hemi
# ---------------------------------------------------------------------------


class TestParseRegionHemi:
    def test_left_suffix(self):
        assert _parse_region_hemi("FColl-SRh_left") == ("FColl-SRh", "left")

    def test_right_suffix(self):
        assert _parse_region_hemi("CINGULATE_right") == ("CINGULATE", "right")

    def test_region_name_containing_underscores(self):
        assert _parse_region_hemi("S.T.s.ter.pf.or._left") == ("S.T.s.ter.pf.or.", "left")

    def test_unrecognised_suffix(self):
        assert _parse_region_hemi("models_cache") == (None, None)


# ---------------------------------------------------------------------------
# generate_umap_reference()
# ---------------------------------------------------------------------------


class TestGenerateUmapReference:
    def test_returns_empty_when_no_models(self, tmp_path, monkeypatch, capsys):
        _patch_model_paths(monkeypatch, [])
        result = generate_umap_reference_fn(str(tmp_path), "emb.csv", str(tmp_path / "out"))
        assert result == []
        assert "No model directories found" in capsys.readouterr().out

    def test_returns_empty_when_no_reference_csv(self, tmp_path, monkeypatch, capsys):
        models_dir = tmp_path / "models"
        model_path = models_dir / "SC_left" / "run1"
        model_path.mkdir(parents=True)
        _patch_model_paths(monkeypatch, [str(model_path)])
        result = generate_umap_reference_fn(str(models_dir), "missing.csv", str(tmp_path / "out"))
        assert result == []
        assert "Reference CSV not found" in capsys.readouterr().out

    def test_skips_unrecognised_directory_names(self, tmp_path, monkeypatch, capsys):
        models_dir = tmp_path / "models"
        model_path = models_dir / "models_cache" / "run1"
        model_path.mkdir(parents=True)
        _patch_model_paths(monkeypatch, [str(model_path)])
        generate_umap_reference_fn(str(models_dir), "emb.csv", str(tmp_path / "out"))
        assert "Skipping unrecognised dir name: models_cache" in capsys.readouterr().out

    def test_reports_unreadable_csv(self, tmp_path, monkeypatch, capsys):
        models_dir = tmp_path / "models"
        model_path = _make_model_dir(models_dir, "SC_left", "emb.csv")
        _patch_model_paths(monkeypatch, [model_path])
        monkeypatch.setattr("pandas.read_csv", MagicMock(side_effect=ValueError("broken")), raising=True)
        result = generate_umap_reference_fn(str(models_dir), "emb.csv", str(tmp_path / "out"))
        assert result == []
        assert "Could not load" in capsys.readouterr().out

    def test_fits_and_saves_one_roi(self, tmp_path, monkeypatch, fake_umap, no_disk_writes):
        dumps, saves = no_disk_writes
        models_dir = tmp_path / "models"
        out_dir = tmp_path / "out"
        model_path = _make_model_dir(models_dir, "SC_left", "emb.csv", rows=5, dims=4)
        _patch_model_paths(monkeypatch, [model_path])

        result = generate_umap_reference_fn(
            str(models_dir), "emb.csv", str(out_dir), n_neighbors=7, min_dist=0.2, random_state=1
        )

        assert result == [("SC", "left")]
        assert out_dir.is_dir()
        assert len(dumps) == 1 and dumps[0][1].endswith("umap_SC_left.pkl")
        assert len(saves) == 1 and saves[0][0].endswith("umap_SC_left_coords.npy")
        assert saves[0][1].shape == (5, 2)
        assert dumps[0][0].kwargs == {"n_neighbors": 7, "min_dist": 0.2, "random_state": 1}

    def test_concatenates_runs_of_the_same_roi(self, tmp_path, monkeypatch, fake_umap, no_disk_writes):
        _, saves = no_disk_writes
        models_dir = tmp_path / "models"
        paths = [
            _make_model_dir(models_dir, "SC_left", "emb.csv", rows=3, run="run1"),
            _make_model_dir(models_dir, "SC_left", "emb.csv", rows=2, run="run2"),
        ]
        _patch_model_paths(monkeypatch, paths)

        result = generate_umap_reference_fn(str(models_dir), "emb.csv", str(tmp_path / "out"))

        assert result == [("SC", "left")]
        assert saves[0][1].shape == (5, 2)

    def test_handles_both_hemispheres(self, tmp_path, monkeypatch, fake_umap, no_disk_writes):
        models_dir = tmp_path / "models"
        paths = [
            _make_model_dir(models_dir, "SC_left", "emb.csv"),
            _make_model_dir(models_dir, "SC_right", "emb.csv"),
        ]
        _patch_model_paths(monkeypatch, paths)
        result = generate_umap_reference_fn(str(models_dir), "emb.csv", str(tmp_path / "out"))
        assert result == [("SC", "left"), ("SC", "right")]

    def test_existing_artefacts_are_skipped(self, tmp_path, monkeypatch, fake_umap, no_disk_writes):
        dumps, _ = no_disk_writes
        models_dir = tmp_path / "models"
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "umap_SC_left.pkl").touch()
        (out_dir / "umap_SC_left_coords.npy").touch()
        _patch_model_paths(monkeypatch, [_make_model_dir(models_dir, "SC_left", "emb.csv")])

        result = generate_umap_reference_fn(str(models_dir), "emb.csv", str(out_dir))

        assert result == [("SC", "left")]
        assert dumps == []

    def test_overwrite_regenerates_existing_artefacts(self, tmp_path, monkeypatch, fake_umap, no_disk_writes):
        dumps, _ = no_disk_writes
        models_dir = tmp_path / "models"
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "umap_SC_left.pkl").touch()
        (out_dir / "umap_SC_left_coords.npy").touch()
        _patch_model_paths(monkeypatch, [_make_model_dir(models_dir, "SC_left", "emb.csv")])

        generate_umap_reference_fn(str(models_dir), "emb.csv", str(out_dir), overwrite=True)

        assert len(dumps) == 1


# ---------------------------------------------------------------------------
# GenerateUmapReference (ScriptBuilder wrapper)
# ---------------------------------------------------------------------------


def _make_script(argv: list) -> GenerateUmapReference:
    script = GenerateUmapReference()
    script.args = script.parse_args(argv)
    return script


class TestArgumentParsing:
    def test_defaults(self):
        script = _make_script(["--models_dir", "m", "--reference_subpath", "emb.csv"])
        assert script.args.output_dir == "reference_data/"
        assert script.args.n_neighbors == 15
        assert script.args.min_dist == 0.1
        assert script.args.random_state == 42
        assert script.args.overwrite is False

    def test_explicit_values(self):
        script = _make_script(
            [
                "--models_dir",
                "m",
                "--reference_subpath",
                "emb.csv",
                "--output_dir",
                "out/",
                "--n_neighbors",
                "30",
                "--min_dist",
                "0.5",
                "--random_state",
                "7",
                "--overwrite",
            ]
        )
        assert script.args.n_neighbors == 30
        assert script.args.min_dist == 0.5
        assert script.args.random_state == 7
        assert script.args.overwrite is True

    def test_models_dir_is_required(self):
        script = GenerateUmapReference()
        with pytest.raises(SystemExit):
            script.parse_args(["--reference_subpath", "emb.csv"])


class TestRun:
    def test_forwards_arguments(self, monkeypatch):
        captured = {}

        def _fake(**kwargs):
            captured.update(kwargs)
            return [("SC", "left")]

        monkeypatch.setattr(generate_umap_reference, "generate_umap_reference", _fake)
        script = _make_script(
            ["--models_dir", "m", "--reference_subpath", "emb.csv", "--output_dir", "o", "--n_neighbors", "9"]
        )
        assert script.run() == 0
        assert captured == {
            "models_dir": "m",
            "reference_subpath": "emb.csv",
            "output_dir": "o",
            "n_neighbors": 9,
            "min_dist": 0.1,
            "random_state": 42,
            "overwrite": False,
        }

    def test_overwrite_banner(self, monkeypatch, capsys):
        monkeypatch.setattr(generate_umap_reference, "generate_umap_reference", lambda **k: [])
        script = _make_script(["--models_dir", "m", "--reference_subpath", "e.csv", "--overwrite"])
        assert script.run() == 0
        out = capsys.readouterr().out
        assert "Overwrite mode   : ON" in out
        assert "Done: 0 UMAP reference model(s) available." in out


class TestMain:
    def test_main_builds_and_runs(self, monkeypatch):
        monkeypatch.setattr(generate_umap_reference, "generate_umap_reference", lambda **k: [])
        monkeypatch.setattr(
            sys, "argv", ["generate_umap_reference.py", "--models_dir", "m", "--reference_subpath", "e.csv"]
        )
        monkeypatch.setattr("champollion_utils.script_builder.check_for_updates", lambda *a, **k: None, raising=False)
        assert generate_umap_reference.main() == 0
