#!/usr/bin/env python3
"""
Exact subject-ID matching in prune_failed_subjects.py.

A file belongs to a pruned subject only when the subject ID equals a whole
directory component of its path (REQ-PATTERNS-BDRABCZUK-D2DE7EF33859) or a
whole filename token bounded by ``_``, ``.`` or a name boundary
(REQ-PATTERNS-BDRABCZUK-D37175240D7D). A QC-passing subject whose ID merely
contains the pruned ID keeps its files (REQ-PATTERNS-BDRABCZUK-742BCBF9CDAF).

All subject IDs are synthetic.
"""

from pathlib import Path

from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from tests.prune_layout import CROP_SUFFIX, crop_file, crops_2mm, make_crop_files, run_prune, skeleton_file, touch

FAILED = "sub-1"
SIBLINGS = ("sub-10", "sub-1a")


def _derivatives(base: Path) -> Path:
    return base / DERIVATIVES_FOLDER


def _make_crops(base: Path, subjects: tuple[str, ...]) -> None:
    """Crop files discovery reads: crops/<masks>/2mm/region_a/mask/Rcrops/<subject>_cropped_skeleton.nii.gz."""
    make_crop_files(base, "region_a", "R", list(subjects))


def _crop_names(base: Path) -> list[str]:
    """Sorted names of the files left in crops/<masks>/2mm/region_a/mask/Rcrops."""
    return sorted(f.name for f in (crops_2mm(base) / "region_a" / "mask" / "Rcrops").iterdir() if f.is_file())


def _prune_failed_with_siblings(base: Path) -> None:
    _make_crops(base, (FAILED, *SIBLINGS))
    run_prune(base, [(FAILED, 0)] + [(s, 1) for s in SIBLINGS])


# ---------------------------------------------------------------------------
# REQ-PATTERNS-BDRABCZUK-D2DE7EF33859: directory component equal to the ID
# ---------------------------------------------------------------------------


class TestPathComponentMatch:
    def test_file_under_directory_named_after_failed_id_is_deleted(self, tmp_path):
        target = touch(_derivatives(tmp_path) / "transforms" / "R" / FAILED / "transform.trm")
        _prune_failed_with_siblings(tmp_path)
        assert not target.exists()


# ---------------------------------------------------------------------------
# REQ-PATTERNS-BDRABCZUK-D37175240D7D: delimited filename token equal to the ID
# ---------------------------------------------------------------------------


class TestFilenameTokenMatch:
    def test_failed_id_as_leading_token_is_deleted(self, tmp_path):
        target = touch(crop_file(tmp_path, "region_a", "R", FAILED))
        _prune_failed_with_siblings(tmp_path)
        assert not target.exists()

    def test_failed_id_as_trailing_token_is_deleted(self, tmp_path):
        target = touch(skeleton_file(tmp_path, "R", FAILED))
        _prune_failed_with_siblings(tmp_path)
        assert not target.exists()


# ---------------------------------------------------------------------------
# REQ-PATTERNS-BDRABCZUK-742BCBF9CDAF: siblings whose ID contains the pruned ID
# ---------------------------------------------------------------------------


class TestSiblingIdsKept:
    def test_sibling_crop_files_are_kept(self, tmp_path):
        _prune_failed_with_siblings(tmp_path)
        assert _crop_names(tmp_path) == [f"{s}{CROP_SUFFIX}" for s in sorted(SIBLINGS)]

    def test_sibling_token_files_are_kept(self, tmp_path):
        deriv = _derivatives(tmp_path)
        kept = []
        for sibling in SIBLINGS:
            kept.append(touch(crop_file(tmp_path, "region_a", "R", sibling)))
            kept.append(touch(skeleton_file(tmp_path, "R", sibling)))
            kept.append(touch(deriv / "transforms" / "R" / f"R_transform_to_ICBM2009c_{sibling}.trm"))
        _prune_failed_with_siblings(tmp_path)
        assert [p for p in kept if not p.exists()] == []

    def test_sibling_directory_is_kept(self, tmp_path):
        kept = [
            touch(_derivatives(tmp_path) / "transforms" / "R" / sibling / f"R_transform_{sibling}.trm")
            for sibling in SIBLINGS
        ]
        _prune_failed_with_siblings(tmp_path)
        assert [p for p in kept if not p.exists()] == []

    def test_numeric_sibling_id_is_kept(self, tmp_path):
        failed, siblings = "100002", ("1000021", "2100002")
        _make_crops(tmp_path, (failed, *siblings))
        kept = [touch(skeleton_file(tmp_path, "R", s)) for s in siblings]
        run_prune(tmp_path, [(failed, 0)] + [(s, 1) for s in siblings])

        assert _crop_names(tmp_path) == [f"{s}{CROP_SUFFIX}" for s in sorted(siblings)]
        assert [p for p in kept if not p.exists()] == []
