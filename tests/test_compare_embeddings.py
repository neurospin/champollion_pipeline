#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
``compare.py embeddings``: per-subject comparison of two embedding versions.

Each set is a champollion_V1 embeddings directory holding, per region and
hemisphere, ``<region>_<hemi>_embeddings.csv`` (``ID`` column + one column per
embedding dimension, ``hemi`` = ``left`` / ``right``, ``--side`` L / R).
Optional UMAP reference directories hold ``umap_<region>_<hemi>.pkl`` (+
``umap_<region>_<hemi>_coords.npy``), loaded through the module-level
``compare.load_umap_model(path)`` helper, which these tests replace with a fake
returning a deterministic linear "UMAP" (no umap-learn, no real pickle).

Pinned module-level helpers (public names the implementation must provide):

* ``compare.linear_cka(x, y) -> float`` - centred linear CKA of two
  ``(n_subjects, n_features)`` matrices with rows paired.
* ``compare.knn_overlap(emb_a, emb_b, k) -> np.ndarray`` - shape
  ``(n_subjects,)``; fraction of each row's k nearest rows in ``emb_a``
  (Euclidean, self excluded) also among its k nearest rows in ``emb_b``.
* ``compare.procrustes_align(source, target) -> np.ndarray`` - ``source``
  mapped onto ``target`` by translation, rotation/reflection and isotropic
  scale (least squares, rows paired).
* ``compare.load_umap_model(path)`` - returns an object with ``.transform(X)``.
* ``Compare._run_embeddings`` - the subcommand's entry point.

Output contract pinned here:

* ``per_subject.csv`` columns: region, side, subject, knn_overlap; plus
  umap_a_x, umap_a_y, umap_b_x, umap_b_y, displacement when --umap_a/--umap_b
  are given (empty cells for a region without UMAP model); plus covariate when
  --covariate is given.
* ``summary.json``: ``regions["<region>/<side>"]`` with n_subjects, only_in_a,
  only_in_b, cka, knn_overlap_mean, knn_overlap_median and, with a covariate,
  spearman_covariate_knn_change / spearman_covariate_displacement; top-level
  ``umap_skipped`` list of {region, side, reason} with reason
  ``missing_umap_a`` / ``missing_umap_b``.
* ``<output>/umap_<region>_<side>.png`` per compared region/side when UMAP refs
  are given.

Requirements: REQ-COMPARE-54 .. REQ-COMPARE-70 (TASK-204). Documentation and
static checks (REQ-COMPARE-71..73) live in test_compare_embeddings_docs.py.
All data is synthetic and small. No nibabel.
"""

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

import compare
from compare import Compare

_R1 = "SCall-SsP-SintraCing"
_R2 = "FColl-SRh"
_R3 = "OCCIPITAL"
_HEMI = {"L": "left", "R": "right"}
_DIM = 6
_UMAP_COLUMNS = ["umap_a_x", "umap_a_y", "umap_b_x", "umap_b_y", "displacement"]


# --------------------------------------------------------------------------- #
# Synthetic data helpers
# --------------------------------------------------------------------------- #


def _subjects(n, prefix="sub-"):
    return [f"{prefix}{i:03d}" for i in range(n)]


def _random_orthogonal(dim, seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(dim, dim)))
    return q


def _write_embeddings(root, region, side, subjects, matrix):
    """Write ``<root>/<region>_<hemi>_embeddings.csv`` (ID + dim columns)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{region}_{_HEMI[side]}_embeddings.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID"] + [f"dim{i + 1}" for i in range(matrix.shape[1])])
        for subject, row in zip(subjects, matrix):
            writer.writerow([subject] + [repr(float(v)) for v in row])
    return path


def _run(tmp_path, *extra, set_a=None, set_b=None):
    out = tmp_path / "out"
    script = Compare()
    script.args = script.parse_args(
        [
            "embeddings",
            "--set_a",
            str(set_a or tmp_path / "a"),
            "--set_b",
            str(set_b or tmp_path / "b"),
            "--output",
            str(out),
            "--njobs",
            "1",
            *extra,
        ]
    )
    assert script.run() == 0
    return out


def _rows(out):
    with open(out / "per_subject.csv", newline="") as f:
        return list(csv.DictReader(f))


def _summary(out):
    return json.loads((out / "summary.json").read_text())


def _region_rows(rows, region=_R1, side="L"):
    return {r["subject"]: r for r in rows if r["region"] == region and r["side"] == side}


def _ref_knn_sets(x, k):
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return [set(np.argsort(row, kind="stable")[:k].tolist()) for row in d]


def _ref_knn_overlap(a, b, k):
    na, nb = _ref_knn_sets(a, k), _ref_knn_sets(b, k)
    return np.array([len(sa & sb) / k for sa, sb in zip(na, nb)])


def _ref_cka(x, y):
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)
    num = np.linalg.norm(yc.T @ xc, "fro") ** 2
    return num / (np.linalg.norm(xc.T @ xc, "fro") * np.linalg.norm(yc.T @ yc, "fro"))


# --------------------------------------------------------------------------- #
# Fake UMAP models
# --------------------------------------------------------------------------- #

# Real path of each placeholder .pkl -> fake model. Reset by the autouse fixture.
_MODELS = {}


class _FakeUmap:
    """Deterministic linear stand-in for a fitted UMAP: X[:, :2] @ m + t."""

    def __init__(self, m=None, t=(0.0, 0.0)):
        self.m = np.eye(2) if m is None else np.asarray(m, dtype=float)
        self.t = np.asarray(t, dtype=float)

    def transform(self, x):
        return np.asarray(x, dtype=float)[:, :2] @ self.m + self.t


def _fake_load_umap_model(path):
    return _MODELS[os.path.realpath(str(path))]


@pytest.fixture(autouse=True)  # noqa: V103 - autouse fixture, used by pytest
def fake_umap(monkeypatch):
    _MODELS.clear()
    monkeypatch.setattr(compare, "load_umap_model", _fake_load_umap_model, raising=False)
    yield
    _MODELS.clear()


def _write_umap(root, region, side, model):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    pkl = root / f"umap_{region}_{_HEMI[side]}.pkl"
    pkl.write_bytes(b"placeholder")
    np.save(root / f"umap_{region}_{_HEMI[side]}_coords.npy", np.zeros((3, 2)))
    _MODELS[os.path.realpath(str(pkl))] = model
    return pkl


# Similarity transform applied by the fake set-B model: scale 3, rotation 40
# degrees composed with a reflection, translation (5, -2).
_THETA = np.deg2rad(40.0)
_ROT_REFL = np.array([[np.cos(_THETA), -np.sin(_THETA)], [np.sin(_THETA), np.cos(_THETA)]]) @ np.diag([1.0, -1.0])
_SIM_B = _FakeUmap(m=3.0 * _ROT_REFL, t=(5.0, -2.0))


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def paired_sets(tmp_path):
    """R1/L: 20 common subjects, B rows reversed, plus 2 only in A, 3 only in B."""
    rng = np.random.default_rng(0)
    common = _subjects(20)
    emb_a = rng.normal(size=(20, _DIM))
    emb_b = emb_a @ _random_orthogonal(_DIM, 1) * 2.0 + rng.normal(scale=0.4, size=(20, _DIM))
    only_a = _subjects(2, "onlyA-")
    only_b = _subjects(3, "onlyB-")
    _write_embeddings(tmp_path / "a", _R1, "L", common + only_a, np.vstack([emb_a, rng.normal(size=(2, _DIM))]))
    order = list(range(19, -1, -1))
    _write_embeddings(
        tmp_path / "b",
        _R1,
        "L",
        [common[i] for i in order] + only_b,
        np.vstack([emb_b[order], rng.normal(size=(3, _DIM))]),
    )
    return {"subjects": common, "a": emb_a, "b": emb_b}


@pytest.fixture
def umap_sets(tmp_path, paired_sets):
    """paired_sets plus UMAP refs for R1/L: identity model for A, similarity for B."""
    _write_umap(tmp_path / "ua", _R1, "L", _FakeUmap())
    _write_umap(tmp_path / "ub", _R1, "L", _SIM_B)
    return paired_sets


def _umap_args(tmp_path):
    return ("--umap_a", str(tmp_path / "ua"), "--umap_b", str(tmp_path / "ub"))


def _write_covariate(path, values):
    """Plain ``subject,<column>`` CSV, rows in reversed order plus a stranger."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "pct_lost"])
        writer.writerow(["stranger-999", "123.0"])
        for subject, value in reversed(list(values.items())):
            writer.writerow([subject, repr(float(value))])
    return path


def _covariate_args(path):
    return ("--covariate", str(path), "--covariate_column", "pct_lost")


# --------------------------------------------------------------------------- #
# REQ-COMPARE-54  pairing by ID, one row per common subject
# --------------------------------------------------------------------------- #


class TestEmbeddingsSubjectPairing:
    """REQ-COMPARE-54."""

    def test_one_row_per_common_subject(self, tmp_path, paired_sets):
        rows = _rows(_run(tmp_path))
        assert sorted(r["subject"] for r in rows) == sorted(paired_sets["subjects"])
        assert {(r["region"], r["side"]) for r in rows} == {(_R1, "L")}

    def test_pairs_rows_by_id_not_row_order(self, tmp_path):
        rng = np.random.default_rng(3)
        subjects = _subjects(12)
        emb = rng.normal(size=(12, _DIM))
        _write_embeddings(tmp_path / "a", _R1, "L", subjects, emb)
        _write_embeddings(tmp_path / "b", _R1, "L", subjects[::-1], emb[::-1])
        out = _run(tmp_path, "--k", "3")
        assert all(float(r["knn_overlap"]) == pytest.approx(1.0) for r in _rows(out))
        assert _summary(out)["regions"][f"{_R1}/L"]["cka"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# REQ-COMPARE-55  pairing counts
# --------------------------------------------------------------------------- #


class TestEmbeddingsPairingCounts:
    """REQ-COMPARE-55."""

    def test_reports_n_subjects_only_in_a_only_in_b(self, tmp_path, paired_sets):
        entry = _summary(_run(tmp_path))["regions"][f"{_R1}/L"]
        assert (entry["n_subjects"], entry["only_in_a"], entry["only_in_b"]) == (20, 2, 3)


# --------------------------------------------------------------------------- #
# REQ-COMPARE-56  default region selection and --side
# --------------------------------------------------------------------------- #


@pytest.fixture  # noqa: V103 - requested via usefixtures
def region_sets(tmp_path):
    rng = np.random.default_rng(4)
    subjects = _subjects(10)
    for region, side in [(_R1, "L"), (_R1, "R"), (_R2, "L")]:
        _write_embeddings(tmp_path / "a", region, side, subjects, rng.normal(size=(10, _DIM)))
    for region, side in [(_R1, "L"), (_R1, "R"), (_R3, "L")]:
        _write_embeddings(tmp_path / "b", region, side, subjects, rng.normal(size=(10, _DIM)))


@pytest.mark.usefixtures("region_sets")
class TestEmbeddingsRegionSelection:
    """REQ-COMPARE-56."""

    def test_default_compares_regions_present_in_both_sets(self, tmp_path):
        out = _run(tmp_path, "--k", "3")
        assert set(_summary(out)["regions"]) == {f"{_R1}/L", f"{_R1}/R"}
        assert {(r["region"], r["side"]) for r in _rows(out)} == {(_R1, "L"), (_R1, "R")}

    @pytest.mark.parametrize("side", ["L", "R"])
    def test_side_selects_hemisphere(self, tmp_path, side):
        out = _run(tmp_path, "--k", "3", "--side", side)
        assert set(_summary(out)["regions"]) == {f"{_R1}/{side}"}


# --------------------------------------------------------------------------- #
# REQ-COMPARE-57  linear CKA
# --------------------------------------------------------------------------- #


class TestEmbeddingsCka:
    """REQ-COMPARE-57."""

    def test_linear_cka_is_one_under_orthogonal_transform(self):
        x = np.random.default_rng(5).normal(size=(30, _DIM))
        assert compare.linear_cka(x, x @ _random_orthogonal(_DIM, 6)) == pytest.approx(1.0)

    def test_linear_cka_is_invariant_to_isotropic_scaling(self):
        rng = np.random.default_rng(7)
        x, y = rng.normal(size=(30, _DIM)), rng.normal(size=(30, 4))
        assert compare.linear_cka(x, 3.5 * y) == pytest.approx(compare.linear_cka(x, y))

    def test_linear_cka_is_centred(self):
        rng = np.random.default_rng(8)
        x, y = rng.normal(size=(30, _DIM)), rng.normal(size=(30, 4))
        assert compare.linear_cka(x + 10.0, y - 4.0) == pytest.approx(_ref_cka(x, y))

    def test_summary_cka_of_id_paired_matrices(self, tmp_path, paired_sets):
        entry = _summary(_run(tmp_path))["regions"][f"{_R1}/L"]
        assert entry["cka"] == pytest.approx(_ref_cka(paired_sets["a"], paired_sets["b"]), abs=1e-6)


# --------------------------------------------------------------------------- #
# REQ-COMPARE-58  per-subject kNN overlap
# --------------------------------------------------------------------------- #


class TestEmbeddingsKnnOverlap:
    """REQ-COMPARE-58."""

    def test_knn_overlap_helper_known_case(self):
        # 1-D points, k=1. Counting a point as its own neighbour would give 1.0.
        a = np.array([[0.0], [1.0], [10.0], [3.0]])
        b = np.array([[0.0], [5.0], [10.0], [1.0]])
        result = compare.knn_overlap(a, b, 1)
        # A nn: 0->1, 1->0, 2->3, 3->1 ; B nn: 0->3, 1->3, 2->1, 3->0
        assert np.asarray(result).tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
        assert np.asarray(compare.knn_overlap(a, a, 2)).tolist() == pytest.approx([1.0] * 4)

    def test_per_subject_overlap_over_paired_subjects(self, tmp_path, paired_sets):
        rows = _region_rows(_rows(_run(tmp_path, "--k", "4")))
        expected = _ref_knn_overlap(paired_sets["a"], paired_sets["b"], 4)
        got = [float(rows[s]["knn_overlap"]) for s in paired_sets["subjects"]]
        assert got == pytest.approx(expected.tolist(), abs=1e-6)
        assert 0.0 < expected.mean() < 1.0  # non-trivial fixture


# --------------------------------------------------------------------------- #
# REQ-COMPARE-59  --k default
# --------------------------------------------------------------------------- #


class TestEmbeddingsKDefault:
    """REQ-COMPARE-59."""

    def test_k_defaults_to_15(self, tmp_path):
        args = Compare().parse_args(["embeddings", "--set_a", str(tmp_path), "--set_b", str(tmp_path)])
        assert args.k == 15


# --------------------------------------------------------------------------- #
# REQ-COMPARE-60  summary mean / median overlap
# --------------------------------------------------------------------------- #


class TestEmbeddingsKnnSummary:
    """REQ-COMPARE-60."""

    def test_mean_and_median_of_per_subject_overlap(self, tmp_path, paired_sets):
        out = _run(tmp_path, "--k", "4")
        values = [float(r["knn_overlap"]) for r in _rows(out)]
        entry = _summary(out)["regions"][f"{_R1}/L"]
        assert entry["knn_overlap_mean"] == pytest.approx(float(np.mean(values)), abs=1e-6)
        assert entry["knn_overlap_median"] == pytest.approx(float(np.median(values)), abs=1e-6)


# --------------------------------------------------------------------------- #
# REQ-COMPARE-61  umap_a coordinates
# --------------------------------------------------------------------------- #


class TestEmbeddingsUmapA:
    """REQ-COMPARE-61."""

    def test_umap_a_columns_are_set_a_projection(self, tmp_path, umap_sets):
        _MODELS.clear()
        model_a = _FakeUmap(m=[[2.0, 1.0], [0.0, -1.0]], t=(0.5, 0.25))
        _write_umap(tmp_path / "ua", _R1, "L", model_a)
        _write_umap(tmp_path / "ub", _R1, "L", _SIM_B)
        rows = _region_rows(_rows(_run(tmp_path, "--k", "4", *_umap_args(tmp_path))))
        expected = model_a.transform(umap_sets["a"])
        for subject, (x, y) in zip(umap_sets["subjects"], expected):
            assert float(rows[subject]["umap_a_x"]) == pytest.approx(x, abs=1e-5)
            assert float(rows[subject]["umap_a_y"]) == pytest.approx(y, abs=1e-5)


# --------------------------------------------------------------------------- #
# REQ-COMPARE-62  umap_b Procrustes-aligned onto umap_a
# --------------------------------------------------------------------------- #


class TestEmbeddingsProcrustes:
    """REQ-COMPARE-62."""

    def test_procrustes_align_recovers_similarity_transform(self):
        target = np.random.default_rng(9).normal(size=(15, 2))
        source = _SIM_B.transform(target)
        assert np.allclose(compare.procrustes_align(source, target), target, atol=1e-8)

    def test_procrustes_align_is_least_squares_for_noisy_source(self):
        rng = np.random.default_rng(10)
        target = rng.normal(size=(40, 2))
        source = _SIM_B.transform(target + rng.normal(scale=0.05, size=(40, 2)))
        aligned = np.asarray(compare.procrustes_align(source, target))
        assert aligned.shape == target.shape
        # Mean and isotropic scale are matched to the target, not copied from source.
        assert np.allclose(aligned.mean(axis=0), target.mean(axis=0), atol=1e-8)
        assert np.linalg.norm(aligned - target) < 0.1 * np.linalg.norm(target - target.mean(axis=0))

    def test_umap_b_aligned_onto_umap_a(self, tmp_path):
        rng = np.random.default_rng(11)
        subjects = _subjects(16)
        emb = rng.normal(size=(16, _DIM))
        _write_embeddings(tmp_path / "a", _R1, "L", subjects, emb)
        _write_embeddings(tmp_path / "b", _R1, "L", subjects[::-1], emb[::-1])
        _write_umap(tmp_path / "ua", _R1, "L", _FakeUmap())
        _write_umap(tmp_path / "ub", _R1, "L", _SIM_B)
        rows = _region_rows(_rows(_run(tmp_path, "--k", "3", *_umap_args(tmp_path))))
        for subject, (x, y) in zip(subjects, emb[:, :2]):
            assert float(rows[subject]["umap_b_x"]) == pytest.approx(x, abs=1e-5)
            assert float(rows[subject]["umap_b_y"]) == pytest.approx(y, abs=1e-5)


# --------------------------------------------------------------------------- #
# REQ-COMPARE-63  displacement
# --------------------------------------------------------------------------- #


class TestEmbeddingsDisplacement:
    """REQ-COMPARE-63."""

    def test_displacement_is_euclidean_distance(self, tmp_path, umap_sets):
        rows = _rows(_run(tmp_path, "--k", "4", *_umap_args(tmp_path)))
        assert rows
        displacements = []
        for r in rows:
            ax, ay, bx, by = (float(r[c]) for c in _UMAP_COLUMNS[:4])
            displacements.append(float(r["displacement"]))
            assert displacements[-1] == pytest.approx(np.hypot(ax - bx, ay - by), abs=1e-5)
        assert max(displacements) > 0.0


# --------------------------------------------------------------------------- #
# REQ-COMPARE-64 / 65  missing UMAP model for one region
# --------------------------------------------------------------------------- #


@pytest.fixture(params=["a", "b"])
def missing_umap(request, tmp_path):
    """R1/L and R2/L compared; R2/L lacks its .pkl in umap_<param> only."""
    rng = np.random.default_rng(12)
    subjects = _subjects(10)
    for region in (_R1, _R2):
        _write_embeddings(tmp_path / "a", region, "L", subjects, rng.normal(size=(10, _DIM)))
        _write_embeddings(tmp_path / "b", region, "L", subjects, rng.normal(size=(10, _DIM)))
    _write_umap(tmp_path / "ua", _R1, "L", _FakeUmap())
    _write_umap(tmp_path / "ub", _R1, "L", _SIM_B)
    if request.param == "a":
        _write_umap(tmp_path / "ub", _R2, "L", _SIM_B)
    else:
        _write_umap(tmp_path / "ua", _R2, "L", _FakeUmap())
    return request.param


class TestEmbeddingsMissingUmapModel:
    """REQ-COMPARE-64."""

    def test_region_still_has_cka_and_knn(self, tmp_path, missing_umap):
        out = _run(tmp_path, "--k", "3", "--side", "L", *_umap_args(tmp_path))
        entry = _summary(out)["regions"][f"{_R2}/L"]
        assert isinstance(entry["cka"], float)
        assert isinstance(entry["knn_overlap_mean"], float)
        rows = _region_rows(_rows(out), region=_R2)
        assert len(rows) == 10
        assert all(r["knn_overlap"] != "" for r in rows.values())

    def test_umap_cells_empty(self, tmp_path, missing_umap):
        out = _run(tmp_path, "--k", "3", "--side", "L", *_umap_args(tmp_path))
        rows = _rows(out)
        for r in _region_rows(rows, region=_R2).values():
            assert [r[c] for c in _UMAP_COLUMNS] == [""] * 5
        for r in _region_rows(rows, region=_R1).values():
            assert all(r[c] != "" for c in _UMAP_COLUMNS)


class TestEmbeddingsMissingUmapSkip:
    """REQ-COMPARE-65."""

    def test_region_listed_under_umap_skipped(self, tmp_path, missing_umap):
        out = _run(tmp_path, "--k", "3", "--side", "L", *_umap_args(tmp_path))
        assert _summary(out)["umap_skipped"] == [{"region": _R2, "side": "L", "reason": f"missing_umap_{missing_umap}"}]


# --------------------------------------------------------------------------- #
# REQ-COMPARE-66  side-by-side UMAP PNG
# --------------------------------------------------------------------------- #


class TestEmbeddingsUmapPlot:
    """REQ-COMPARE-66."""

    def test_one_png_per_compared_region_side(self, tmp_path, umap_sets):
        out = _run(tmp_path, "--k", "4", *_umap_args(tmp_path))
        pngs = sorted(p.name for p in out.glob("*.png"))
        assert pngs == [f"umap_{_R1}_L.png"]
        assert (out / pngs[0]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_no_png_without_umap_refs(self, tmp_path, paired_sets):
        out = _run(tmp_path, "--k", "4")
        assert list(out.glob("*.png")) == []


# --------------------------------------------------------------------------- #
# REQ-COMPARE-67  covariate matched by subject ID
# --------------------------------------------------------------------------- #


class TestEmbeddingsCovariate:
    """REQ-COMPARE-67."""

    def test_covariate_matched_by_subject_id(self, tmp_path, paired_sets):
        values = {s: 0.5 * i for i, s in enumerate(paired_sets["subjects"])}
        cov = _write_covariate(tmp_path / "cov.csv", values)
        rows = _region_rows(_rows(_run(tmp_path, "--k", "4", *_covariate_args(cov))))
        assert {s: float(r["covariate"]) for s, r in rows.items()} == pytest.approx(values)


# --------------------------------------------------------------------------- #
# REQ-COMPARE-68 / 69  Spearman correlations
# --------------------------------------------------------------------------- #


def _covariate_values(subjects, seed):
    rng = np.random.default_rng(seed)
    return {s: float(v) for s, v in zip(subjects, rng.uniform(0.0, 50.0, size=len(subjects)))}


class TestEmbeddingsSpearmanKnn:
    """REQ-COMPARE-68."""

    def test_spearman_covariate_knn_change(self, tmp_path, paired_sets):
        values = _covariate_values(paired_sets["subjects"], 13)
        cov = _write_covariate(tmp_path / "cov.csv", values)
        out = _run(tmp_path, "--k", "4", *_covariate_args(cov))
        rows = _region_rows(_rows(out))
        subjects = paired_sets["subjects"]
        expected = spearmanr(
            [values[s] for s in subjects], [1.0 - float(rows[s]["knn_overlap"]) for s in subjects]
        ).statistic
        entry = _summary(out)["regions"][f"{_R1}/L"]
        assert entry["spearman_covariate_knn_change"] == pytest.approx(float(expected), abs=1e-6)


class TestEmbeddingsSpearmanDisplacement:
    """REQ-COMPARE-69."""

    def test_spearman_covariate_displacement(self, tmp_path, umap_sets):
        values = _covariate_values(umap_sets["subjects"], 14)
        cov = _write_covariate(tmp_path / "cov.csv", values)
        out = _run(tmp_path, "--k", "4", *_umap_args(tmp_path), *_covariate_args(cov))
        rows = _region_rows(_rows(out))
        subjects = umap_sets["subjects"]
        expected = spearmanr(
            [values[s] for s in subjects], [float(rows[s]["displacement"]) for s in subjects]
        ).statistic
        entry = _summary(out)["regions"][f"{_R1}/L"]
        assert entry["spearman_covariate_displacement"] == pytest.approx(float(expected), abs=1e-6)


# --------------------------------------------------------------------------- #
# REQ-COMPARE-70  strict JSON
# --------------------------------------------------------------------------- #


def _reject_constant(name):
    raise ValueError(f"non-strict JSON constant {name}")


class TestEmbeddingsStrictJson:
    """REQ-COMPARE-70."""

    def test_undefined_spearman_written_as_null(self, tmp_path, paired_sets):
        # A constant covariate makes the Spearman correlation undefined (NaN).
        cov = _write_covariate(tmp_path / "cov.csv", {s: 7.0 for s in paired_sets["subjects"]})
        out = _run(tmp_path, "--k", "4", *_covariate_args(cov))
        summary = json.loads((out / "summary.json").read_text(), parse_constant=_reject_constant)
        assert summary["regions"][f"{_R1}/L"]["spearman_covariate_knn_change"] is None
