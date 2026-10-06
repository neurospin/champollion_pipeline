#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
``compare.py crops``: per-subject comparison of two cortical_tiles crop sets.

Each set is a ``crops/2mm`` directory holding, per region,
``<region>/mask/{side}{input_type}.npy`` (shape ``(n_subjects, x, y, z, 1)``),
``<region>/mask/{side}{input_type}_subject.csv`` (one ``Subject`` column, row
order matching the ``.npy``) and ``<region>/mask/{side}mask_cropped.nii.gz``,
whose PyAIMS header (``aims.Finder().check(path)`` + ``.header()``) locates the
crop: ``voxel_size``, ``referentials`` and ``transformations`` (16 row-major
floats each, storage-order mm -> referential). AIMS storage order is the
``.npy`` order, so with a shared referential, voxel size ``vs`` and diagonal
+-1 rotation ``R``, set B voxel ``j`` coincides with set A voxel
``j + R (t_b - t_a) / vs``.

Requirements: REQ-COMPARE-09 (per_subject.csv rows and columns),
REQ-COMPARE-10 (pairing by subject ID), REQ-COMPARE-11 (voxel counts),
REQ-COMPARE-12 (pct_lost, dice, shift), REQ-COMPARE-15 (summary.json fields),
REQ-COMPARE-16 (--regions / --side), REQ-COMPARE-18 (header alignment),
REQ-COMPARE-19 (contradicting headers skipped), REQ-COMPARE-20 (headerless
shape mismatch skipped), REQ-COMPARE-21 (headerless equal shapes compared
directly), REQ-COMPARE-23/24/25 (skip ``reason`` for header skips),
REQ-COMPARE-26 (input missing in one set), REQ-COMPARE-27/28 (input missing in
both sets), REQ-COMPARE-29 (subject CSV mismatch), REQ-COMPARE-30 (no common
subjects). REQ-COMPARE-22 (no soma.aims) lives in test_compare_without_aims.py.

All fixtures are synthetic: crops are written with numpy, ``mask_cropped.nii.gz``
is an empty placeholder file, and ``compare.aims`` is replaced by a fake whose
``Finder`` serves the header registered for each placeholder path. No nibabel.
"""

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

import compare
from compare import Compare

_CSV_COLUMNS = ["region", "side", "subject", "n_a", "n_b", "kept", "lost", "gained", "pct_lost", "dice", "shift"]

_REGION = "S.C.-sylv."
_OTHER_REGION = "F.I.P."
_SHAPE = (4, 4, 4)
_VS = 2.0
_REFERENTIAL = "Talairach-MNI template-SPM"
_MINUS_I = ((-1, 0, 0), (0, -1, 0), (0, 0, -1))
_PLUS_I = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


# --------------------------------------------------------------------------- #
# Fake PyAIMS header reader
# --------------------------------------------------------------------------- #

# Real path of each placeholder mask_cropped.nii.gz -> header dict served by
# the fake Finder. Reset for every test by the autouse ``fake_aims`` fixture.
_HEADERS = {}


def _key(path):
    return os.path.realpath(str(path))


class _FakeFinder:
    """Minimal ``aims.Finder``: check() then header() of the last checked path."""

    def __init__(self):
        self._header = None

    def check(self, path):
        self._header = _HEADERS.get(_key(path))
        return self._header is not None

    def header(self):
        return self._header


class _FakeAims:
    Finder = _FakeFinder


@pytest.fixture(autouse=True)
def fake_aims(monkeypatch):
    _HEADERS.clear()
    monkeypatch.setattr(compare, "aims", _FakeAims())
    yield
    _HEADERS.clear()


def _header(translation=(0.0, 0.0, 0.0), rotation=_MINUS_I, vs=_VS, referentials=(_REFERENTIAL,)):
    """PyAIMS-like header: one storage-mm -> referential transformation per referential."""
    vs3 = vs if isinstance(vs, tuple) else (vs, vs, vs)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    transformation = [float(x) for x in matrix.ravel()]
    return {
        "voxel_size": [float(v) for v in vs3] + [1.0],
        "referentials": list(referentials),
        "transformations": [transformation for _ in referentials],
    }


_DEFAULT = object()


def _write_crops(root, region, side, subjects, voxels, shape, header=_DEFAULT, input_type="skeleton"):
    """Write one region/side of a synthetic crop set with numpy.

    ``voxels`` is one iterable of ``.npy`` (x, y, z) indices per subject, each
    set to 1. ``header`` is the dict the fake Finder serves for the
    ``{side}mask_cropped.nii.gz`` placeholder (default: ``_header()``);
    ``None`` writes no placeholder; ``"unreadable"`` writes the placeholder but
    registers no header, so ``Finder.check`` returns False.
    """
    mask_dir = Path(root) / region / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((len(subjects), *shape, 1), dtype=np.int16)
    for i, subject_voxels in enumerate(voxels):
        for x, y, z in subject_voxels:
            arr[i, x, y, z, 0] = 1
    np.save(mask_dir / f"{side}{input_type}.npy", arr)
    with open(mask_dir / f"{side}{input_type}_subject.csv", "w", newline="") as f:
        f.write("Subject\n")
        for s in subjects:
            f.write(f"{s}\n")
    if header is _DEFAULT:
        header = _header()
    if header is None:
        return
    placeholder = mask_dir / f"{side}mask_cropped.nii.gz"
    placeholder.touch()
    if header != "unreadable":
        _HEADERS[_key(placeholder)] = header


def _run_crops(tmp_path, *extra):
    out = tmp_path / "out"
    script = Compare()
    script.args = script.parse_args(
        [
            "crops",
            "--set_a",
            str(tmp_path / "a"),
            "--set_b",
            str(tmp_path / "b"),
            "--output",
            str(out),
            "--njobs",
            "1",
            *extra,
        ]
    )
    assert script.run() == 0
    return out


def _read_rows(out):
    with open(out / "per_subject.csv", newline="") as f:
        return list(csv.DictReader(f))


def _read_header(out):
    with open(out / "per_subject.csv", newline="") as f:
        return next(csv.reader(f))


def _read_summary(out):
    return json.loads((out / "summary.json").read_text())


def _row(rows, subject, region=_REGION, side="L"):
    matches = [r for r in rows if r["subject"] == subject and r["region"] == region and r["side"] == side]
    assert len(matches) == 1, f"expected one row for {region}/{side}/{subject}, got {matches}"
    return matches[0]


def _bool_volume(shape, voxels):
    vol = np.zeros(shape, dtype=bool)
    for v in voxels:
        vol[v] = True
    return vol


# Metrics fixture: one region/side (S.C.-sylv./L), same shape and header in both sets.
#   s1: A {(0,0,0),(1,1,1),(2,2,2),(3,3,3)}  B {(0,0,0),(1,1,1),(0,3,0)}
#       -> n_a 4, n_b 3, kept 2, lost 2, gained 1, pct_lost 50, dice 4/7
#   s3: A {(1,2,3)}  B {}  -> emptied: n_a 1, n_b 0, lost 1, pct_lost 100, dice 0, shift inf
#   s5: A {}  B {}         -> pct_lost 0, dice 1, shift 0
#   s2 only in A, s4 only in B. B rows are in a different order than A rows.
_S1_A = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
_S1_B = [(0, 0, 0), (1, 1, 1), (0, 3, 0)]
_S3_A = [(1, 2, 3)]


@pytest.fixture
def metrics_sets(tmp_path):
    _write_crops(
        tmp_path / "a",
        _REGION,
        "L",
        ["s1", "s2", "s3", "s5"],
        [_S1_A, [(0, 1, 0)], _S3_A, []],
        _SHAPE,
    )
    _write_crops(
        tmp_path / "b",
        _REGION,
        "L",
        ["s3", "s4", "s1", "s5"],
        [[], [(2, 2, 2)], _S1_B, []],
        _SHAPE,
    )
    return tmp_path


def _run_metrics(metrics_sets):
    return _run_crops(metrics_sets, "--top_k", "2")


# --------------------------------------------------------------------------- #
# REQ-COMPARE-09: per_subject.csv rows and columns
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestCropsPerSubjectCsv:
    """REQ-COMPARE-09: one per_subject.csv row per compared subject, fixed columns."""

    def test_header_lists_columns_in_order(self, metrics_sets):
        metrics_out = _run_metrics(metrics_sets)
        assert _read_header(metrics_out) == _CSV_COLUMNS

    def test_one_row_per_compared_subject(self, metrics_sets):
        metrics_out = _run_metrics(metrics_sets)
        rows = _read_rows(metrics_out)

        assert sorted((r["region"], r["side"], r["subject"]) for r in rows) == [
            (_REGION, "L", "s1"),
            (_REGION, "L", "s3"),
            (_REGION, "L", "s5"),
        ]

    def test_input_type_label_reads_label_npy(self, tmp_path):
        # skeleton and label files hold different volumes; only label must be read.
        for root, label_voxels in ((tmp_path / "a", [(0, 0, 0), (1, 0, 0)]), (tmp_path / "b", [(0, 0, 0)])):
            _write_crops(root, _REGION, "L", ["s1"], [[(3, 3, 3)]], _SHAPE, input_type="skeleton")
            _write_crops(root, _REGION, "L", ["s1"], [label_voxels], _SHAPE, input_type="label")

        row = _row(_read_rows(_run_crops(tmp_path, "--input_type", "label")), "s1")

        assert (int(row["n_a"]), int(row["n_b"]), int(row["kept"])) == (2, 1, 1)


# --------------------------------------------------------------------------- #
# REQ-COMPARE-10: pairing by subject ID
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestCropsSubjectPairing:
    """REQ-COMPARE-10: crops are paired by subject ID, not by row position."""

    def test_pairs_crops_by_subject_id_not_row_order(self, metrics_sets):
        metrics_out = _run_metrics(metrics_sets)
        # s1 is row 0 in set A and row 2 in set B; row-order pairing would give n_b 0.
        row = _row(_read_rows(metrics_out), "s1")

        assert (int(row["n_a"]), int(row["n_b"]), int(row["kept"])) == (4, 3, 2)

    def test_subjects_in_one_set_only_are_not_compared(self, metrics_sets):
        metrics_out = _run_metrics(metrics_sets)
        subjects = {r["subject"] for r in _read_rows(metrics_out)}

        assert "s2" not in subjects
        assert "s4" not in subjects


# --------------------------------------------------------------------------- #
# REQ-COMPARE-11: voxel counts
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestCropsVoxelCounts:
    """REQ-COMPARE-11: n_a, n_b, kept, lost, gained count nonzero voxels."""

    def test_counts_for_partially_overlapping_subject(self, metrics_sets):
        metrics_out = _run_metrics(metrics_sets)
        row = _row(_read_rows(metrics_out), "s1")

        assert {k: int(row[k]) for k in ("n_a", "n_b", "kept", "lost", "gained")} == {
            "n_a": 4,
            "n_b": 3,
            "kept": 2,
            "lost": 2,
            "gained": 1,
        }

    def test_counts_for_emptied_subject(self, metrics_sets):
        metrics_out = _run_metrics(metrics_sets)
        row = _row(_read_rows(metrics_out), "s3")

        assert {k: int(row[k]) for k in ("n_a", "n_b", "kept", "lost", "gained")} == {
            "n_a": 1,
            "n_b": 0,
            "kept": 0,
            "lost": 1,
            "gained": 0,
        }


# --------------------------------------------------------------------------- #
# REQ-COMPARE-12: pct_lost, dice, shift
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestCropsDerivedMetrics:
    """REQ-COMPARE-12: pct_lost, dice and shift definitions, edge cases included."""

    def test_partial_overlap_metrics(self, metrics_sets):
        metrics_out = _run_metrics(metrics_sets)
        row = _row(_read_rows(metrics_out), "s1")
        expected_shift = compare.wasserstein_distance(
            _bool_volume(_SHAPE, _S1_A).astype(np.float64), _bool_volume(_SHAPE, _S1_B).astype(np.float64)
        )

        assert float(row["pct_lost"]) == pytest.approx(50.0, abs=1e-3)
        assert float(row["dice"]) == pytest.approx(4.0 / 7.0, abs=1e-3)
        assert expected_shift > 0.0
        assert float(row["shift"]) == pytest.approx(expected_shift, abs=1e-3)

    def test_emptied_subject_metrics(self, metrics_sets):
        metrics_out = _run_metrics(metrics_sets)
        row = _row(_read_rows(metrics_out), "s3")

        assert float(row["pct_lost"]) == pytest.approx(100.0)
        assert float(row["dice"]) == pytest.approx(0.0)
        assert float(row["shift"]) == float("inf")

    def test_both_empty_subject_metrics(self, metrics_sets):
        metrics_out = _run_metrics(metrics_sets)
        row = _row(_read_rows(metrics_out), "s5")

        assert float(row["pct_lost"]) == pytest.approx(0.0)
        assert float(row["dice"]) == pytest.approx(1.0)
        assert float(row["shift"]) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# REQ-COMPARE-18: alignment from the PyAIMS header transformations
# --------------------------------------------------------------------------- #

# Different shapes. A: (6,6,6), B: (4,3,2), both R = -I, vs 2 mm.
# t_a = (6,-22,84) (IXI canonical_25), t_b = t_a - 2 mm * (1,1,4), so
# offset = R (t_b - t_a) / vs = (t_a - t_b) / 2 = (1,1,4):
# B .npy voxel (i,j,k) coincides with A .npy voxel (i+1, j+1, k+4).
# A zero offset, or the opposite sign, would not match them up.
_ALIGN_SHAPE_A = (6, 6, 6)
_ALIGN_SHAPE_B = (4, 3, 2)
_ALIGN_T_A = (6.0, -22.0, 84.0)
_ALIGN_OFFSET = [1, 1, 4]
_ALIGN_T_B = tuple(t - _VS * o for t, o in zip(_ALIGN_T_A, _ALIGN_OFFSET))  # (4,-24,76)


@pytest.fixture
def aligned_sets(tmp_path):
    # A: (1,1,4) <-> B (0,0,0), (4,3,5) <-> B (3,2,1), plus (0,0,0) outside B's footprint.
    _write_crops(
        tmp_path / "a",
        _REGION,
        "R",
        ["s1"],
        [[(1, 1, 4), (4, 3, 5), (0, 0, 0)]],
        _ALIGN_SHAPE_A,
        header=_header(translation=_ALIGN_T_A),
    )
    _write_crops(
        tmp_path / "b",
        _REGION,
        "R",
        ["s1"],
        [[(0, 0, 0), (3, 2, 1)]],
        _ALIGN_SHAPE_B,
        header=_header(translation=_ALIGN_T_B),
    )
    return tmp_path


def _counts(row):
    return {k: int(row[k]) for k in ("n_a", "n_b", "kept", "lost", "gained")}


@pytest.mark.unit
class TestCropsHeaderAlignment:
    """REQ-COMPARE-18: set B voxel j is compared with set A voxel j + R (t_b - t_a) / vs."""

    def test_different_shapes_align_through_header_offset(self, aligned_sets):
        row = _row(_read_rows(_run_crops(aligned_sets)), "s1", side="R")

        assert _counts(row) == {"n_a": 3, "n_b": 2, "kept": 2, "lost": 1, "gained": 0}

    def test_same_shape_shifted_translation_aligns(self, tmp_path):
        # IXI case: R = -I, t_a = (6,-22,84), t_b = (4,-22,84) -> offset (1,0,0):
        # B (0,0,0) coincides with A (1,0,0).
        _write_crops(tmp_path / "a", _REGION, "L", ["s1"], [[(1, 0, 0)]], _SHAPE, header=_header((6.0, -22.0, 84.0)))
        _write_crops(tmp_path / "b", _REGION, "L", ["s1"], [[(0, 0, 0)]], _SHAPE, header=_header((4.0, -22.0, 84.0)))

        out = _run_crops(tmp_path)
        row = _row(_read_rows(out), "s1")

        assert {k: int(row[k]) for k in ("kept", "lost", "gained")} == {"kept": 1, "lost": 0, "gained": 0}
        assert _read_summary(out)["regions"][f"{_REGION}/L"]["alignment_offset_vox"] == [1, 0, 0]

    def test_offset_follows_rotation_sign(self, tmp_path):
        # R = +I: t_b = t_a + 2 mm on x -> offset +(1,0,0) (with R = -I it would be -1).
        _write_crops(
            tmp_path / "a", _REGION, "L", ["s1"], [[(1, 0, 0)]], _SHAPE, header=_header((0.0, 0.0, 0.0), _PLUS_I)
        )
        _write_crops(
            tmp_path / "b", _REGION, "L", ["s1"], [[(0, 0, 0)]], _SHAPE, header=_header((2.0, 0.0, 0.0), _PLUS_I)
        )

        out = _run_crops(tmp_path)
        row = _row(_read_rows(out), "s1")

        assert {k: int(row[k]) for k in ("kept", "lost", "gained")} == {"kept": 1, "lost": 0, "gained": 0}
        assert _read_summary(out)["regions"][f"{_REGION}/L"]["alignment_offset_vox"] == [1, 0, 0]

    def test_voxels_outside_overlap_counted_on_union_grid(self, tmp_path):
        # R = -I, t_b = t_a + 2 mm on x -> offset (-1,0,0): B (1,0,0) <-> A (0,0,0),
        # B (0,0,0) <-> A (-1,0,0), outside A's crop but inside the union grid.
        _write_crops(tmp_path / "a", _REGION, "L", ["s1"], [[(0, 0, 0)]], _SHAPE, header=_header((0.0, 0.0, 0.0)))
        _write_crops(
            tmp_path / "b", _REGION, "L", ["s1"], [[(0, 0, 0), (1, 0, 0)]], _SHAPE, header=_header((2.0, 0.0, 0.0))
        )

        row = _row(_read_rows(_run_crops(tmp_path)), "s1")

        assert _counts(row) == {"n_a": 1, "n_b": 2, "kept": 1, "lost": 0, "gained": 1}


# --------------------------------------------------------------------------- #
# REQ-COMPARE-19: contradicting headers -> skipped, whatever the shapes
# --------------------------------------------------------------------------- #

_ROT_Z_90 = ((0, -1, 0), (1, 0, 0), (0, 0, 1))

# (header A, header B) per case. Equal (4,4,4) shapes on purpose: a skip here
# cannot come from a shape check.
_CONTRADICTING_HEADERS = {
    "referential_differs": (_header(), _header(referentials=("Scanner-based anatomical coordinates",))),
    "voxel_size_differs": (_header(), _header(vs=1.0)),
    "non_axis_aligned": (_header(rotation=_ROT_Z_90), _header(rotation=_ROT_Z_90)),
    "transformations_differ": (_header(rotation=_MINUS_I), _header(rotation=_PLUS_I)),
    "non_integer_offset": (_header((0.0, 0.0, 0.0)), _header((1.0, 0.0, 0.0))),
}


@pytest.fixture(params=sorted(_CONTRADICTING_HEADERS))
def contradicting_sets(request, tmp_path):
    """S.C.-sylv./L with contradicting headers next to a comparable F.I.P./L."""
    header_a, header_b = _CONTRADICTING_HEADERS[request.param]
    _write_crops(tmp_path / "a", _REGION, "L", ["s1"], [[(0, 0, 0)]], _SHAPE, header=header_a)
    _write_crops(tmp_path / "b", _REGION, "L", ["s1"], [[(0, 0, 0)]], _SHAPE, header=header_b)
    for root in (tmp_path / "a", tmp_path / "b"):
        _write_crops(root, _OTHER_REGION, "L", ["s1"], [[(1, 1, 1)]], _SHAPE)
    return tmp_path


def _assert_listed_under_skipped(out):
    skipped = _read_summary(out)["skipped"]
    assert any(e.get("region") == _REGION and e.get("side") == "L" for e in skipped), skipped


def _assert_not_compared(out):
    rows = _read_rows(out)
    summary = _read_summary(out)
    assert [r for r in rows if r["region"] == _REGION] == []
    assert f"{_REGION}/L" not in summary["regions"]
    assert f"{_OTHER_REGION}/L" in summary["regions"]


@pytest.mark.unit
class TestCropsSkipsContradictingHeaders:
    """REQ-COMPARE-19: contradicting mask_cropped headers put the region/side under skipped."""

    def test_region_listed_under_skipped(self, contradicting_sets):
        _assert_listed_under_skipped(_run_crops(contradicting_sets))

    def test_skipped_region_is_not_compared(self, contradicting_sets):
        _assert_not_compared(_run_crops(contradicting_sets))

    def test_reason_names_condition(self, request, contradicting_sets):
        """REQ-COMPARE-25: each case breaks exactly one condition; its key is the expected reason."""
        expected = request.node.callspec.params["contradicting_sets"]

        assert _skipped_entry(_run_crops(contradicting_sets))["reason"] == expected


# --------------------------------------------------------------------------- #
# REQ-COMPARE-20 / REQ-COMPARE-21: no usable header in one set
# --------------------------------------------------------------------------- #

_NO_TRANSFORMATION = {k: v for k, v in _header().items() if k != "transformations"}

# (header A, header B) per case; see _write_crops for None / "unreadable".
_HEADERLESS = {
    "a_no_mask_cropped": (None, _DEFAULT),
    "b_no_mask_cropped": (_DEFAULT, None),
    "b_unreadable_header": (_DEFAULT, "unreadable"),
    "b_no_transformation": (_DEFAULT, _NO_TRANSFORMATION),
}


def _write_headerless(tmp_path, case, shape_b):
    header_a, header_b = _HEADERLESS[case]
    _write_crops(tmp_path / "a", _REGION, "L", ["s1"], [[(0, 0, 0), (1, 1, 1)]], _SHAPE, header=header_a)
    _write_crops(tmp_path / "b", _REGION, "L", ["s1"], [[(0, 0, 0)]], shape_b, header=header_b)
    for root in (tmp_path / "a", tmp_path / "b"):
        _write_crops(root, _OTHER_REGION, "L", ["s1"], [[(1, 1, 1)]], _SHAPE)
    return tmp_path


@pytest.fixture(params=sorted(_HEADERLESS))
def headerless_mismatched_sets(request, tmp_path):
    return _write_headerless(tmp_path, request.param, (3, 3, 3))


@pytest.fixture(params=sorted(_HEADERLESS))
def headerless_equal_sets(request, tmp_path):
    return _write_headerless(tmp_path, request.param, _SHAPE)


@pytest.mark.unit
class TestCropsSkipsHeaderlessShapeMismatch:
    """REQ-COMPARE-20: no usable header in one set and different shapes -> skipped."""

    def test_region_listed_under_skipped(self, headerless_mismatched_sets):
        _assert_listed_under_skipped(_run_crops(headerless_mismatched_sets))

    def test_skipped_region_is_not_compared(self, headerless_mismatched_sets):
        _assert_not_compared(_run_crops(headerless_mismatched_sets))


@pytest.mark.unit
class TestCropsHeaderlessEqualShapes:
    """REQ-COMPARE-21: no usable header in one set and equal shapes -> direct comparison, null offset."""

    def test_compared_voxel_for_voxel(self, headerless_equal_sets):
        row = _row(_read_rows(_run_crops(headerless_equal_sets)), "s1")

        assert _counts(row) == {"n_a": 2, "n_b": 1, "kept": 1, "lost": 1, "gained": 0}

    def test_alignment_offset_is_null(self, headerless_equal_sets):
        summary = _read_summary(_run_crops(headerless_equal_sets))

        assert f"{_REGION}/L" in summary["regions"], summary
        assert summary["regions"][f"{_REGION}/L"]["alignment_offset_vox"] is None


# --------------------------------------------------------------------------- #
# REQ-COMPARE-15: summary.json per region/side fields
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestCropsSummary:
    """REQ-COMPARE-15: per region/side summary.json entry."""

    def test_region_side_entry_values(self, metrics_sets):
        metrics_out = _run_metrics(metrics_sets)
        entry = _read_summary(metrics_out)["regions"][f"{_REGION}/L"]

        # Compared subjects: s1 (pct_lost 50, dice 4/7), s3 (100, 0), s5 (0, 1).
        assert entry["subjects_changed"] == 2
        assert entry["pct_lost_mean"] == pytest.approx(50.0, abs=1e-3)
        assert entry["pct_lost_p95"] == pytest.approx(95.0, abs=1e-3)
        assert entry["pct_lost_max"] == pytest.approx(100.0, abs=1e-3)
        assert entry["dice_mean"] == pytest.approx((4.0 / 7.0 + 0.0 + 1.0) / 3.0, abs=1e-3)
        assert entry["dice_min"] == pytest.approx(0.0, abs=1e-3)
        assert entry["subjects_emptied"] == 1
        assert entry["crop_shape_a"] == list(_SHAPE)
        assert entry["crop_shape_b"] == list(_SHAPE)
        assert "alignment_offset_vox" in entry

    def test_top_lost_lists_top_k_subjects_by_pct_lost(self, metrics_sets):
        metrics_out = _run_metrics(metrics_sets)
        entry = _read_summary(metrics_out)["regions"][f"{_REGION}/L"]

        assert [e["subject"] for e in entry["top_lost"]] == ["s3", "s1"]

    def test_alignment_offset_reported(self, aligned_sets):
        aligned_out = _run_crops(aligned_sets)
        entry = _read_summary(aligned_out)["regions"][f"{_REGION}/R"]

        assert entry["alignment_offset_vox"] == _ALIGN_OFFSET
        assert entry["crop_shape_a"] == list(_ALIGN_SHAPE_A)
        assert entry["crop_shape_b"] == list(_ALIGN_SHAPE_B)


# --------------------------------------------------------------------------- #
# REQ-COMPARE-16: --regions / --side selection
# --------------------------------------------------------------------------- #


@pytest.fixture
def two_region_sets(tmp_path):
    """S.C.-sylv. and F.I.P. on both sides in both sets, plus ONLY_A in set A only."""
    for root in (tmp_path / "a", tmp_path / "b"):
        for region in (_REGION, _OTHER_REGION):
            for side in ("L", "R"):
                _write_crops(root, region, side, ["s1"], [[(0, 0, 0)]], _SHAPE)
    _write_crops(tmp_path / "a", "ONLY_A", "L", ["s1"], [[(0, 0, 0)]], _SHAPE)
    return tmp_path


@pytest.mark.unit
class TestCropsRegionSideSelection:
    """REQ-COMPARE-16: --regions and --side restrict what is compared."""

    def test_regions_and_side_restrict_comparison(self, two_region_sets):
        out = _run_crops(two_region_sets, "--regions", _REGION, "--side", "L")

        assert {(r["region"], r["side"]) for r in _read_rows(out)} == {(_REGION, "L")}
        assert set(_read_summary(out)["regions"]) == {f"{_REGION}/L"}

    def test_defaults_compare_both_sides_of_regions_in_both_sets(self, two_region_sets):
        out = _run_crops(two_region_sets)

        assert set(_read_summary(out)["regions"]) == {
            f"{_REGION}/L",
            f"{_REGION}/R",
            f"{_OTHER_REGION}/L",
            f"{_OTHER_REGION}/R",
        }


# --------------------------------------------------------------------------- #
# REQ-COMPARE-23 / 24: reason of headerless skips (REQ-COMPARE-25 sits with REQ-COMPARE-19)
# --------------------------------------------------------------------------- #


def _skipped_entry(out, region=_REGION, side="L"):
    entries = [e for e in _read_summary(out)["skipped"] if e.get("region") == region and e.get("side") == side]
    assert len(entries) == 1, f"expected one skipped entry for {region}/{side}, got {_read_summary(out)['skipped']}"
    return entries[0]


@pytest.fixture(params=["a_no_mask_cropped", "b_no_mask_cropped", "b_unreadable_header"])
def no_mask_cropped_sets(request, tmp_path):
    return _write_headerless(tmp_path, request.param, (3, 3, 3))


@pytest.mark.unit
class TestCropsSkipReasonHeaderAbsent:
    """REQ-COMPARE-23 / REQ-COMPARE-24: reason of a headerless shape-mismatch skip."""

    def test_reason_is_no_mask_cropped(self, no_mask_cropped_sets):
        """REQ-COMPARE-23: missing or unreadable mask_cropped header -> no_mask_cropped."""
        assert _skipped_entry(_run_crops(no_mask_cropped_sets))["reason"] == "no_mask_cropped"

    def test_reason_is_no_transformation(self, tmp_path):
        """REQ-COMPARE-24: both headers readable, one without transformations -> no_transformation."""
        sets = _write_headerless(tmp_path, "b_no_transformation", (3, 3, 3))

        assert _skipped_entry(_run_crops(sets))["reason"] == "no_transformation"


# --------------------------------------------------------------------------- #
# REQ-COMPARE-26: input missing in one set
# --------------------------------------------------------------------------- #

# case -> (set lacking the file, file suffix removed, expected reason)
_MISSING_IN_ONE = {
    "a_no_npy": ("a", "skeleton.npy", "missing_in_a"),
    "a_no_csv": ("a", "skeleton_subject.csv", "missing_in_a"),
    "b_no_npy": ("b", "skeleton.npy", "missing_in_b"),
    "b_no_csv": ("b", "skeleton_subject.csv", "missing_in_b"),
}


@pytest.fixture(params=sorted(_MISSING_IN_ONE))
def missing_in_one_set(request, tmp_path):
    """S.C.-sylv./L lacks one input file in one set (region dir kept), next to a comparable F.I.P./L."""
    lacking, suffix, reason = _MISSING_IN_ONE[request.param]
    for root in (tmp_path / "a", tmp_path / "b"):
        _write_crops(root, _REGION, "L", ["s1"], [[(0, 0, 0)]], _SHAPE)
        _write_crops(root, _OTHER_REGION, "L", ["s1"], [[(1, 1, 1)]], _SHAPE)
    (tmp_path / lacking / _REGION / "mask" / f"L{suffix}").unlink()
    return tmp_path, reason


@pytest.mark.unit
class TestCropsSkipsInputMissingInOneSet:
    """REQ-COMPARE-26: input in one set only -> skipped with missing_in_a / missing_in_b."""

    def test_listed_under_skipped_with_reason(self, missing_in_one_set):
        sets, reason = missing_in_one_set

        assert _skipped_entry(_run_crops(sets))["reason"] == reason

    def test_region_side_is_not_compared(self, missing_in_one_set):
        sets, _ = missing_in_one_set

        _assert_not_compared(_run_crops(sets))


# --------------------------------------------------------------------------- #
# REQ-COMPARE-27 / 28: input missing in both sets
# --------------------------------------------------------------------------- #

_ABSENT_REGION = "S.T.s."


@pytest.mark.unit
class TestCropsInputMissingInBothSets:
    """REQ-COMPARE-27: explicit region -> missing_in_a_and_b; REQ-COMPARE-28: discovered -> silent."""

    @pytest.mark.parametrize("layout", ["no_region_dir", "empty_mask_dir"])
    def test_explicit_region_listed_as_missing_in_a_and_b(self, tmp_path, layout):
        for root in (tmp_path / "a", tmp_path / "b"):
            _write_crops(root, _OTHER_REGION, "L", ["s1"], [[(1, 1, 1)]], _SHAPE)
            if layout == "empty_mask_dir":
                (root / _ABSENT_REGION / "mask").mkdir(parents=True)

        out = _run_crops(tmp_path, "--regions", _ABSENT_REGION, _OTHER_REGION, "--side", "L")

        assert _skipped_entry(out, region=_ABSENT_REGION)["reason"] == "missing_in_a_and_b"
        assert f"{_OTHER_REGION}/L" in _read_summary(out)["regions"]

    def test_discovered_region_side_ignored_silently(self, tmp_path):
        # S.C.-sylv. exists in both sets with L only; R is missing in both.
        for root in (tmp_path / "a", tmp_path / "b"):
            _write_crops(root, _REGION, "L", ["s1"], [[(0, 0, 0)]], _SHAPE)

        summary = _read_summary(_run_crops(tmp_path))

        assert [e for e in summary["skipped"] if e.get("region") == _REGION and e.get("side") == "R"] == []
        assert f"{_REGION}/R" not in summary["regions"]
        assert f"{_REGION}/L" in summary["regions"]


# --------------------------------------------------------------------------- #
# REQ-COMPARE-29: _subject.csv row count != .npy subject count
# --------------------------------------------------------------------------- #


def _rewrite_subject_csv(root, subjects, region=_REGION, side="L", input_type="skeleton"):
    with open(Path(root) / region / "mask" / f"{side}{input_type}_subject.csv", "w", newline="") as f:
        f.write("Subject\n")
        for s in subjects:
            f.write(f"{s}\n")


@pytest.fixture(params=["a_extra_row", "b_missing_row"])
def subject_csv_mismatch_sets(request, tmp_path):
    """S.C.-sylv./L: two .npy subjects per set, one set's CSV row count wrong; F.I.P./L comparable."""
    for root in (tmp_path / "a", tmp_path / "b"):
        _write_crops(root, _REGION, "L", ["s1", "s2"], [[(0, 0, 0)], [(1, 1, 1)]], _SHAPE)
        _write_crops(root, _OTHER_REGION, "L", ["s1"], [[(1, 1, 1)]], _SHAPE)
    if request.param == "a_extra_row":
        _rewrite_subject_csv(tmp_path / "a", ["s1", "s2", "s3"])
    else:
        _rewrite_subject_csv(tmp_path / "b", ["s1"])
    return tmp_path


@pytest.mark.unit
class TestCropsSkipsSubjectCsvMismatch:
    """REQ-COMPARE-29: CSV rows != .npy first dimension (either set) -> subject_csv_mismatch."""

    def test_listed_under_skipped_with_reason(self, subject_csv_mismatch_sets):
        assert _skipped_entry(_run_crops(subject_csv_mismatch_sets))["reason"] == "subject_csv_mismatch"

    def test_region_side_is_not_compared(self, subject_csv_mismatch_sets):
        _assert_not_compared(_run_crops(subject_csv_mismatch_sets))


# --------------------------------------------------------------------------- #
# REQ-COMPARE-30: no subject ID shared by the two sets
# --------------------------------------------------------------------------- #


@pytest.fixture
def no_common_subject_sets(tmp_path):
    """S.C.-sylv./L: A holds s1, s2 and B holds s3, s4; F.I.P./L comparable."""
    _write_crops(tmp_path / "a", _REGION, "L", ["s1", "s2"], [[(0, 0, 0)], [(1, 1, 1)]], _SHAPE)
    _write_crops(tmp_path / "b", _REGION, "L", ["s3", "s4"], [[(0, 0, 0)], [(1, 1, 1)]], _SHAPE)
    for root in (tmp_path / "a", tmp_path / "b"):
        _write_crops(root, _OTHER_REGION, "L", ["s1"], [[(1, 1, 1)]], _SHAPE)
    return tmp_path


@pytest.mark.unit
class TestCropsSkipsNoCommonSubjects:
    """REQ-COMPARE-30: empty subject ID intersection -> no_common_subjects."""

    def test_listed_under_skipped_with_reason(self, no_common_subject_sets):
        assert _skipped_entry(_run_crops(no_common_subject_sets))["reason"] == "no_common_subjects"

    def test_region_side_is_not_compared(self, no_common_subject_sets):
        _assert_not_compared(_run_crops(no_common_subject_sets))
