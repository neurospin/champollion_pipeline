#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wasserstein behaviour of src/compare.py when exactly one mask of a pair is empty.

An emptied (or newly created) mask is the most extreme change a mask can
undergo, so it must never be reported as an unchanged (0.0) distance.

Requirements: REQ-COMPARE-04 (function returns +inf), REQ-COMPARE-05 (report
records null per mask, keeping the JSON standard), REQ-COMPARE-06 (report
buckets the mask under ``one_side_empty``).

PyAIMS is stubbed in conftest.py; ``compare.aims`` is replaced here by a fake
reader serving small in-memory numpy volumes.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import compare
from compare import Compare, wasserstein_distance

_SHAPE = (4, 4, 4)


def _point_volume(index, value=1.0):
    vol = np.zeros(_SHAPE, dtype=np.float64)
    vol[index] = value
    return vol


def _empty_volume():
    return np.zeros(_SHAPE, dtype=np.float64)


class _FakeAims:
    """Fake ``soma.aims`` serving registered numpy volumes by resolved path."""

    def __init__(self, objects):
        self.objects = dict(objects)

    def read(self, path):
        return self.objects[str(path)]


def _reject_non_standard_constant(token):
    raise ValueError(f"non-standard JSON constant in report: {token}")


def _load_strict_json(path: Path) -> dict:
    """Parse the report, failing on Infinity / -Infinity / NaN tokens."""
    return json.loads(path.read_text(), parse_constant=_reject_non_standard_constant)


# (volume in set_a, volume in set_b) for the one-side-empty mask
_ONE_SIDE_EMPTY_CASES = {
    "emptied": (lambda: _point_volume((1, 1, 1)), _empty_volume),
    "created": (_empty_volume, lambda: _point_volume((1, 1, 1))),
}

_MODES = {
    # mode -> (relative path of the one-side-empty mask, relative path of a normal shared mask)
    "masks": ("L/gone.nii.gz", "L/shared.nii.gz"),
    "cortical_tiles": ("REGION/mask/Lmask_skeleton.nii.gz", "OTHER/mask/Lmask_skeleton.nii.gz"),
}


@pytest.fixture(params=sorted(_MODES))
def mode(request):
    return request.param


@pytest.fixture(params=sorted(_ONE_SIDE_EMPTY_CASES))
def one_side_empty_report(request, mode, tmp_path, monkeypatch):
    """Run compare.py on two sets holding one one-side-empty mask and one shifted mask."""
    make_a, make_b = _ONE_SIDE_EMPTY_CASES[request.param]
    empty_rel, shared_rel = _MODES[mode]
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    objects = {}
    for root, empty_vol, shared_vol in (
        (dir_a, make_a(), _point_volume((0, 0, 0))),
        (dir_b, make_b(), _point_volume((2, 0, 0))),
    ):
        for rel, vol in ((empty_rel, empty_vol), (shared_rel, shared_vol)):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            objects[str(path.resolve())] = vol
    monkeypatch.setattr(compare, "aims", _FakeAims(objects))

    out = tmp_path / "report.json"
    script = Compare()
    script.args = script.parse_args(
        [mode, "--set_a", str(dir_a), "--set_b", str(dir_b), "--output", str(out), "--metric", "wasserstein"]
    )
    assert script.run() == 0
    return out, empty_rel, shared_rel


class TestWassersteinDistanceOneSideEmpty:
    """REQ-COMPARE-04."""

    def test_emptied_mask_is_positive_infinity(self):
        assert wasserstein_distance(_point_volume((1, 1, 1)), _empty_volume()) == math.inf

    def test_created_mask_is_positive_infinity(self):
        assert wasserstein_distance(_empty_volume(), _point_volume((1, 1, 1))) == math.inf


class TestReportPerMaskOneSideEmpty:
    """REQ-COMPARE-05."""

    def test_per_mask_value_is_null(self, one_side_empty_report):
        out, empty_rel, shared_rel = one_side_empty_report
        report = _load_strict_json(out)
        per_mask = report["wasserstein_per_mask"]
        assert empty_rel in per_mask
        assert per_mask[empty_rel] is None
        assert per_mask[shared_rel] == pytest.approx(2.0)


class TestReportBucketOneSideEmpty:
    """REQ-COMPARE-06."""

    def test_mask_listed_under_one_side_empty_bucket(self, one_side_empty_report):
        out, empty_rel, shared_rel = one_side_empty_report
        buckets = _load_strict_json(out)["wasserstein_by_bucket"]
        assert buckets.get("one_side_empty") == [empty_rel]
        assert not any(empty_rel in names for key, names in buckets.items() if key != "one_side_empty")
        assert any(shared_rel in names for key, names in buckets.items() if key != "one_side_empty")
