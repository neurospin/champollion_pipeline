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

from champollion_pipeline.prune_failed_subjects import PruneFailedSubjects
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER

FAILED = "sub-1"
SIBLINGS = ("sub-10", "sub-1a")


def _derivatives(base: Path) -> Path:
    return base / DERIVATIVES_FOLDER


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _make_crops(base: Path, subjects: tuple[str, ...]) -> None:
    """Crop files the subject discovery reads: crops/2mm/<region>/<subject>.nii.gz."""
    for subject in subjects:
        _touch(_derivatives(base) / "crops" / "2mm" / "region_a" / f"{subject}.nii.gz")


def _run_prune(base: Path, qc_rows: list[tuple[str, int]]) -> None:
    qc = base / "qc.tsv"
    qc.write_text("participant_id\tqc\n" + "\n".join(f"{s}\t{q}" for s, q in qc_rows))
    script = PruneFailedSubjects()
    script.args = script.parse_args([str(base), "--qc", str(qc)])
    script.run()


def _prune_failed_with_siblings(base: Path) -> None:
    _make_crops(base, (FAILED, *SIBLINGS))
    _run_prune(base, [(FAILED, 0)] + [(s, 1) for s in SIBLINGS])


# ---------------------------------------------------------------------------
# REQ-PATTERNS-BDRABCZUK-D2DE7EF33859: directory component equal to the ID
# ---------------------------------------------------------------------------


class TestPathComponentMatch:
    def test_file_under_directory_named_after_failed_id_is_deleted(self, tmp_path):
        target = _touch(_derivatives(tmp_path) / "transforms" / "R" / FAILED / "transform.trm")
        _prune_failed_with_siblings(tmp_path)
        assert not target.exists()


# ---------------------------------------------------------------------------
# REQ-PATTERNS-BDRABCZUK-D37175240D7D: delimited filename token equal to the ID
# ---------------------------------------------------------------------------


class TestFilenameTokenMatch:
    def test_failed_id_as_leading_token_is_deleted(self, tmp_path):
        rcrops = _derivatives(tmp_path) / "crops" / "2mm" / "region_a" / "mask" / "Rcrops"
        target = _touch(rcrops / f"{FAILED}_cropped_skeleton.nii.gz")
        _prune_failed_with_siblings(tmp_path)
        assert not target.exists()

    def test_failed_id_as_trailing_token_is_deleted(self, tmp_path):
        skel = _derivatives(tmp_path) / "skeletons" / "2mm" / "R"
        target = _touch(skel / f"Rresampled_skeleton_{FAILED}.nii.gz")
        _prune_failed_with_siblings(tmp_path)
        assert not target.exists()


# ---------------------------------------------------------------------------
# REQ-PATTERNS-BDRABCZUK-742BCBF9CDAF: siblings whose ID contains the pruned ID
# ---------------------------------------------------------------------------


class TestSiblingIdsKept:
    def test_sibling_crop_files_are_kept(self, tmp_path):
        _prune_failed_with_siblings(tmp_path)
        region = _derivatives(tmp_path) / "crops" / "2mm" / "region_a"
        assert sorted(f.name for f in region.iterdir() if f.is_file()) == [f"{s}.nii.gz" for s in sorted(SIBLINGS)]

    def test_sibling_token_files_are_kept(self, tmp_path):
        deriv = _derivatives(tmp_path)
        kept = []
        for sibling in SIBLINGS:
            kept.append(
                _touch(deriv / "crops" / "2mm" / "region_a" / "mask" / "Rcrops" / f"{sibling}_cropped_skeleton.nii.gz")
            )
            kept.append(_touch(deriv / "skeletons" / "2mm" / "R" / f"Rresampled_skeleton_{sibling}.nii.gz"))
            kept.append(_touch(deriv / "transforms" / "R" / f"R_transform_to_ICBM2009c_{sibling}.trm"))
        _prune_failed_with_siblings(tmp_path)
        assert [p for p in kept if not p.exists()] == []

    def test_sibling_directory_is_kept(self, tmp_path):
        kept = [
            _touch(_derivatives(tmp_path) / "transforms" / "R" / sibling / f"R_transform_{sibling}.trm")
            for sibling in SIBLINGS
        ]
        _prune_failed_with_siblings(tmp_path)
        assert [p for p in kept if not p.exists()] == []

    def test_numeric_sibling_id_is_kept(self, tmp_path):
        failed, siblings = "100002", ("1000021", "2100002")
        _make_crops(tmp_path, (failed, *siblings))
        skel = _derivatives(tmp_path) / "skeletons" / "2mm" / "R"
        kept = [_touch(skel / f"Rresampled_skeleton_{s}.nii.gz") for s in siblings]
        _run_prune(tmp_path, [(failed, 0)] + [(s, 1) for s in siblings])

        region = _derivatives(tmp_path) / "crops" / "2mm" / "region_a"
        assert sorted(f.name for f in region.iterdir()) == [f"{s}.nii.gz" for s in sorted(siblings)]
        assert [p for p in kept if not p.exists()] == []
