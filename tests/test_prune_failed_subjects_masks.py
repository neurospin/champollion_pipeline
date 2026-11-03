#!/usr/bin/env python3
"""
prune_failed_subjects.py on the versioned crops root crops/<masks>/2mm.

run_cortical_tiles writes its crops under
<output>/cortical_tiles-<version>/crops/<masks>/2mm (e.g. crops/canonical_25/2mm)
and moves the legacy unversioned crops/2mm away (DEF-PATTERNS-BDRABCZUK-36F2AACA0EBF).

- REQ (masks default): with no --masks the parser yields masks canonical_25.
- REQ (explicit masks): --masks <M> is kept as given.
- REQ (missing root): an absent crops/<M>/2mm raises a ValueError naming that path,
  a legacy crops/2mm being no fallback.
- REQ (no deletion): an absent crops/<M>/2mm deletes no file under <output>.
- REQ (legacy hint): with crops/<M>/2mm absent and crops/2mm present, the ValueError
  names crops/2mm.
- REQ (mask scope): with --masks <M>, no file under crops/<V> for a <V> other than <M>
  is deleted (crops/2mm included).
- REQ (invalid masks): a --masks value that is empty, '.', '..' or holds a path
  separator raises a ValueError before any deletion.
- Invariant (regression over the subject-removal scripts): with --masks <M>,
  crops/<M>/2mm is the crops directory.

All subject IDs and mask version names are synthetic.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from champollion_pipeline.prune_failed_subjects import PruneFailedSubjects
from tests.prune_layout import (
    CROP_SUFFIX,
    MASKS,
    crops_2mm,
    legacy_crops_2mm,
    make_aggregate,
    make_crop_files,
    run_prune,
    skeleton_file,
    touch,
)

OTHER_MASKS = "masks_v_test"


def _parse(tmp_path: Path, *extra: str):
    return PruneFailedSubjects().parse_args([str(tmp_path), "--qc", "/some/qc.tsv", *extra])


def _files_under(base: Path) -> set[str]:
    """Relative paths of every regular file under base, qc.tsv excluded."""
    return {str(p.relative_to(base)) for p in base.rglob("*") if p.is_file() and p.name != "qc.tsv"}


def _legacy_crop_file(base: Path, region: str, subject: str) -> Path:
    return legacy_crops_2mm(base) / region / "mask" / "Rcrops" / f"{subject}{CROP_SUFFIX}"


def _crop_names(base: Path, region: str, masks: str = MASKS) -> set[str]:
    return {f.name for f in (crops_2mm(base, masks) / region / "mask" / "Rcrops").iterdir()}


# ---------------------------------------------------------------------------
# --masks argument
# ---------------------------------------------------------------------------


class TestMasksArgument:
    def test_no_masks_yields_canonical_25(self, tmp_path):
        assert _parse(tmp_path).masks == "canonical_25"

    def test_explicit_masks_kept(self, tmp_path):
        assert _parse(tmp_path, "--masks", OTHER_MASKS).masks == OTHER_MASKS


# ---------------------------------------------------------------------------
# Invariant: crops/<M>/2mm is the crops directory (regression of the defect)
# ---------------------------------------------------------------------------


class TestVersionedCropsRoot:
    def test_default_masks_prunes_failed_subject_under_canonical_25(self, tmp_path):
        """The defect: on crops/canonical_25/2mm alone prune raised 'crops/2mm directory not found'."""
        make_crop_files(tmp_path, "region_a", "R", ["sub-01", "sub-02"])

        assert run_prune(tmp_path, [("sub-01", 1), ("sub-02", 0)]) == 0
        assert _crop_names(tmp_path, "region_a") == {f"sub-01{CROP_SUFFIX}"}

    def test_explicit_masks_prunes_failed_subject_under_that_version(self, tmp_path):
        make_crop_files(tmp_path, "region_a", "R", ["sub-01", "sub-02"], masks=OTHER_MASKS)
        failed_skel = touch(skeleton_file(tmp_path, "R", "sub-02"))

        assert run_prune(tmp_path, [("sub-01", 1), ("sub-02", 0)], masks=OTHER_MASKS) == 0
        assert _crop_names(tmp_path, "region_a", OTHER_MASKS) == {f"sub-01{CROP_SUFFIX}"}
        assert not failed_skel.exists()

    def test_aggregate_rows_filtered_under_versioned_root(self, tmp_path):
        subjects = ["sub-01", "sub-02", "sub-03"]
        make_crop_files(tmp_path, "region_a", "R", subjects, masks=OTHER_MASKS)
        npy, csv, arr = make_aggregate(tmp_path, "region_a", "Rskeleton", subjects, masks=OTHER_MASKS)

        assert run_prune(tmp_path, [("sub-01", 1), ("sub-02", 0), ("sub-03", 1)], masks=OTHER_MASKS) == 0
        assert pd.read_csv(csv, dtype=str)["Subject"].tolist() == ["sub-01", "sub-03"]
        np.testing.assert_array_equal(np.load(npy), arr[[0, 2]])


# ---------------------------------------------------------------------------
# Absent crops/<M>/2mm
# ---------------------------------------------------------------------------


class TestMissingVersionedCrops:
    def test_error_names_missing_versioned_path(self, tmp_path):
        make_crop_files(tmp_path, "region_a", "R", ["sub-01"])
        missing = crops_2mm(tmp_path, OTHER_MASKS)

        with pytest.raises(ValueError, match=re.escape(str(missing))):
            run_prune(tmp_path, [("sub-01", 1)], masks=OTHER_MASKS)

    def test_legacy_crops_2mm_is_not_a_fallback(self, tmp_path):
        touch(_legacy_crop_file(tmp_path, "region_a", "sub-01"))
        missing = crops_2mm(tmp_path)

        with pytest.raises(ValueError, match=re.escape(str(missing))):
            run_prune(tmp_path, [("sub-01", 1)])

    def test_no_file_deleted_with_legacy_layout_only(self, tmp_path):
        touch(_legacy_crop_file(tmp_path, "region_a", "sub-01"))
        touch(_legacy_crop_file(tmp_path, "region_a", "sub-02"))
        touch(skeleton_file(tmp_path, "R", "sub-02"))
        before = _files_under(tmp_path)

        with pytest.raises(ValueError):
            run_prune(tmp_path, [("sub-01", 1), ("sub-02", 0)])
        assert _files_under(tmp_path) == before

    def test_error_names_legacy_crops_2mm(self, tmp_path):
        touch(_legacy_crop_file(tmp_path, "region_a", "sub-01"))
        legacy = legacy_crops_2mm(tmp_path)

        with pytest.raises(ValueError, match=re.escape(str(legacy))):
            run_prune(tmp_path, [("sub-01", 1)])


# ---------------------------------------------------------------------------
# Deletion confined to crops/<masks>/ (user decision 2026-11-03)
# ---------------------------------------------------------------------------


class TestMasksScope:
    def test_other_mask_version_files_kept(self, tmp_path):
        make_crop_files(tmp_path, "region_a", "R", ["sub-01", "sub-02"])
        other = make_crop_files(tmp_path, "region_a", "R", ["sub-01", "sub-02"], masks=OTHER_MASKS)

        assert run_prune(tmp_path, [("sub-01", 1), ("sub-02", 0)]) == 0
        assert _crop_names(tmp_path, "region_a") == {f"sub-01{CROP_SUFFIX}"}
        assert all(path.exists() for path in other)

    def test_legacy_crops_2mm_files_kept(self, tmp_path):
        make_crop_files(tmp_path, "region_a", "R", ["sub-01", "sub-02"])
        legacy = touch(_legacy_crop_file(tmp_path, "region_a", "sub-02"))

        assert run_prune(tmp_path, [("sub-01", 1), ("sub-02", 0)]) == 0
        assert _crop_names(tmp_path, "region_a") == {f"sub-01{CROP_SUFFIX}"}
        assert legacy.exists()


# ---------------------------------------------------------------------------
# --masks naming more or less than one directory
# ---------------------------------------------------------------------------


def _assert_refused_before_deletion(base: Path, masks: str) -> None:
    """Build crops where --masks <masks> would resolve, then expect a ValueError and no file deleted."""
    make_crop_files(base, "region_a", "R", ["sub-01", "sub-02"], masks=masks)
    touch(skeleton_file(base, "R", "sub-02"))
    before = _files_under(base)

    with pytest.raises(ValueError):
        run_prune(base, [("sub-01", 1), ("sub-02", 0)], masks=masks)
    assert _files_under(base) == before


class TestInvalidMasks:
    @pytest.mark.parametrize("masks", ["nested/canonical_25", "../canonical_25"])
    def test_path_separator_raises_before_deletion(self, tmp_path, masks):
        _assert_refused_before_deletion(tmp_path, masks)

    @pytest.mark.parametrize("masks", ["", ".", ".."])
    def test_dot_or_empty_raises_before_deletion(self, tmp_path, masks):
        _assert_refused_before_deletion(tmp_path, masks)
