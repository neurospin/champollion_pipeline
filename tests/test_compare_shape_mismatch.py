#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report visibility of common masks that src/compare.py skips because the two
mask volumes differ in shape.

Requirements: REQ-COMPARE-07 (report lists each skipped mask with both
shapes under ``summary.skipped_shape_mismatch``), REQ-COMPARE-08
(``summary.total_compared`` counts only the masks actually compared).

Scope: reporting only. Aligning differently-shaped crops is TASK-194.

PyAIMS is stubbed in conftest.py; ``compare.aims`` is replaced here by a fake
reader serving small in-memory numpy volumes.
"""

import json

import numpy as np
import pytest

import compare
from compare import Compare

_SHAPE_A = (4, 4, 4)
_SHAPE_B = (2, 3, 2)

_MODES = {
    # mode -> (relative path of the shape-mismatch mask, relative path of a matched-shape mask)
    "masks": ("L/mismatch.nii.gz", "L/shared.nii.gz"),
    "cortical_tiles": ("REGION/mask/Lmask_skeleton.nii.gz", "OTHER/mask/Lmask_skeleton.nii.gz"),
}


def _point_volume(shape, index):
    vol = np.zeros(shape, dtype=np.float64)
    vol[index] = 1.0
    return vol


class _FakeAims:
    """Fake ``soma.aims`` serving registered numpy volumes by resolved path."""

    def __init__(self, objects):
        self.objects = dict(objects)

    def read(self, path):
        return self.objects[str(path)]


def _run_compare(mode, tmp_path, monkeypatch, volumes_by_rel):
    """Run compare.py on two sets; volumes_by_rel maps rel path -> (vol_a, vol_b)."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    objects = {}
    for rel, (vol_a, vol_b) in volumes_by_rel.items():
        for root, vol in ((dir_a, vol_a), (dir_b, vol_b)):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            objects[str(path.resolve())] = vol
    monkeypatch.setattr(compare, "aims", _FakeAims(objects))

    out = tmp_path / "report.json"
    script = Compare()
    script.args = script.parse_args(
        [mode, "--set_a", str(dir_a), "--set_b", str(dir_b), "--output", str(out), "--metric", "both"]
    )
    assert script.run() == 0
    return json.loads(out.read_text())


@pytest.fixture(params=sorted(_MODES))
def mode(request):
    return request.param


@pytest.fixture
def mismatch_report(mode, tmp_path, monkeypatch):
    """Report for one shape-mismatch common mask plus one matched-shape common mask."""
    mismatch_rel, shared_rel = _MODES[mode]
    report = _run_compare(
        mode,
        tmp_path,
        monkeypatch,
        {
            mismatch_rel: (_point_volume(_SHAPE_A, (0, 0, 0)), _point_volume(_SHAPE_B, (0, 0, 0))),
            shared_rel: (_point_volume(_SHAPE_A, (0, 0, 0)), _point_volume(_SHAPE_A, (2, 0, 0))),
        },
    )
    return report, mismatch_rel, shared_rel


class TestReportListsShapeMismatch:
    """REQ-COMPARE-07."""

    def test_skipped_mask_listed_with_both_shapes(self, mismatch_report):
        report, mismatch_rel, shared_rel = mismatch_report
        skipped = report["summary"]["skipped_shape_mismatch"]
        assert skipped == {mismatch_rel: {"shape_a": list(_SHAPE_A), "shape_b": list(_SHAPE_B)}}
        assert shared_rel not in skipped


class TestReportCountsComparedMasks:
    """REQ-COMPARE-08."""

    def test_total_compared_excludes_shape_mismatch(self, mismatch_report):
        report, _mismatch_rel, _shared_rel = mismatch_report
        summary = report["summary"]
        assert summary["total_compared"] == 1
        assert summary["total_common"] == 2

    def test_total_compared_equals_total_common_without_mismatch(self, mode, tmp_path, monkeypatch):
        mismatch_rel, shared_rel = _MODES[mode]
        report = _run_compare(
            mode,
            tmp_path,
            monkeypatch,
            {
                rel: (_point_volume(_SHAPE_A, (0, 0, 0)), _point_volume(_SHAPE_A, (1, 0, 0)))
                for rel in (mismatch_rel, shared_rel)
            },
        )
        assert report["summary"]["total_compared"] == 2
        assert report["summary"]["total_common"] == 2
