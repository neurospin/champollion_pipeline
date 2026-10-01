#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for src/generate_umap_reference.py using the REAL
``get_model_paths`` from the champollion_V1 submodule (REQ-UMAPREF-01/02).

``tests/test_generate_umap_reference.py`` replaces ``get_model_paths`` with a
lambda, so the path-stripping logic in ``generate_umap_reference()`` was never
exercised against what the real discovery helper returns.  Here the helper is
loaded straight from its source file (bypassing any ``sys.modules`` stub that
another test module may have installed) and injected into the module under
test.  Only ``umap`` is stubbed, so no real UMAP model is fitted.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_HELPER_FILE = (
    Path(__file__).resolve().parents[1]
    / "external"
    / "champollion_V1"
    / "champollion"
    / "utils"
    / "put_together_embeddings_files.py"
)

if not _HELPER_FILE.is_file():  # pragma: no cover - depends on submodule checkout
    pytest.skip(f"champollion_V1 submodule not checked out: {_HELPER_FILE}", allow_module_level=True)

import generate_umap_reference  # noqa: E402

REFERENCE_SUBPATH = "ref_embeddings/full_embeddings.csv"


def _load_real_get_model_paths():
    spec = importlib.util.spec_from_file_location("_real_put_together_embeddings_files", _HELPER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_model_paths


class _FakeReducer:
    """Stand-in for ``umap.UMAP`` that projects onto the first two columns."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, X):
        return np.asarray(X)[:, :2]


@pytest.fixture  # noqa: V103
def real_get_model_paths(monkeypatch):
    fn = _load_real_get_model_paths()
    assert Path(fn.__code__.co_filename).resolve() == _HELPER_FILE
    monkeypatch.setattr(generate_umap_reference, "get_model_paths", fn)
    return fn


@pytest.fixture  # noqa: V103
def fake_umap(monkeypatch):
    module = MagicMock()
    module.UMAP = _FakeReducer
    monkeypatch.setitem(sys.modules, "umap", module)
    return module


def _write_csv(path, rows=3, dims=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "ID," + ",".join(f"d{i}" for i in range(dims))
    lines = [header] + [f"sub-{r:02d}," + ",".join(str(float(r + i)) for i in range(dims)) for r in range(rows)]
    path.write_text("\n".join(lines) + "\n")


def _make_model(model_dir, rows=3, with_csv=True):
    """Mark ``model_dir`` as a trained model and (optionally) give it a reference CSV."""
    (model_dir / ".hydra").mkdir(parents=True, exist_ok=True)
    (model_dir / ".hydra" / "config.yaml").write_text("model: test\n")
    if with_csv:
        _write_csv(model_dir / REFERENCE_SUBPATH, rows=rows)


@pytest.fixture
def models_tree(tmp_path):
    """A models dir mixing every case that the real discovery helper can return.

    models/
      README.txt                                  file at the root
      SC_left/run1/                               model, depth 2
      FColl-SRh_right/group_a/seed_3/run7/        model, nested depth 4 (5 rows)
      FColl-SRh_right/group_b/run2/               second run of the same ROI (2 rows)
      embeddings/run1/                            model, unrecognised first component
      CINGULATE_left/scratch/                     NOT a model (no .hydra), has CSV
      STs_right/run1/                             model WITHOUT the reference CSV
    """
    models = tmp_path / "models"
    models.mkdir()
    (models / "README.txt").write_text("not a directory\n")
    _make_model(models / "SC_left" / "run1", rows=3)
    _make_model(models / "FColl-SRh_right" / "group_a" / "seed_3" / "run7", rows=5)
    _make_model(models / "FColl-SRh_right" / "group_b" / "run2", rows=2)
    _make_model(models / "embeddings" / "run1", rows=3)
    _write_csv(models / "CINGULATE_left" / "scratch" / REFERENCE_SUBPATH, rows=3)
    _make_model(models / "STs_right" / "run1", with_csv=False)
    return models


EXPECTED_PAIRS = [("FColl-SRh", "right"), ("SC", "left")]


class TestRealModelDiscovery:
    @pytest.mark.usefixtures("real_get_model_paths", "fake_umap")
    def test_returns_pairs_of_recognised_models_at_any_depth(self, tmp_path, models_tree):
        out_dir = tmp_path / "out"

        result = generate_umap_reference.generate_umap_reference(str(models_tree), REFERENCE_SUBPATH, str(out_dir))

        assert result == EXPECTED_PAIRS
        assert sorted(p.name for p in out_dir.iterdir()) == [
            "umap_FColl-SRh_right.pkl",
            "umap_FColl-SRh_right_coords.npy",
            "umap_SC_left.pkl",
            "umap_SC_left_coords.npy",
        ]
        # Both runs of the nested ROI are concatenated (5 + 2 rows).
        assert np.load(out_dir / "umap_FColl-SRh_right_coords.npy").shape == (7, 2)
        assert np.load(out_dir / "umap_SC_left_coords.npy").shape == (3, 2)

    @pytest.mark.usefixtures("real_get_model_paths", "fake_umap")
    def test_trailing_slash_on_models_dir_gives_same_pairs(self, tmp_path, models_tree):
        without_slash = generate_umap_reference.generate_umap_reference(
            str(models_tree), REFERENCE_SUBPATH, str(tmp_path / "out_a")
        )
        with_slash = generate_umap_reference.generate_umap_reference(
            str(models_tree) + "/", REFERENCE_SUBPATH, str(tmp_path / "out_b")
        )

        assert with_slash == without_slash == EXPECTED_PAIRS
