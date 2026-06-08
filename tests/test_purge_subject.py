#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for purge_subject.py
"""

import numpy as np
import pandas as pd
import pytest

from purge_subject import PurgeSubject


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_script(derivatives: str, subject: str, dry_run: bool = False) -> PurgeSubject:
    script = PurgeSubject()
    args = script.parse_args(
        [derivatives, "--subject", subject] + (["--dry-run"] if dry_run else [])
    )
    script.args = args
    return script


def _make_crop_file(crops_2mm, region: str, side_prefix: str, subject: str, ext: str = ".nii.gz"):
    """Create a fake per-subject crop file under crops/2mm/{region}/mask/{side_prefix}/."""
    mask_dir = crops_2mm / region / "mask" / side_prefix
    mask_dir.mkdir(parents=True, exist_ok=True)
    f = mask_dir / f"{subject}{ext}"
    f.touch()
    return f


def _make_aggregated_array(crops_2mm, region: str, prefix: str, subjects: list[str]):
    """Create a .npy + _subject.csv pair for the given subjects."""
    mask_dir = crops_2mm / region / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((len(subjects), 1, 4, 4, 4), dtype=np.float32)
    np.save(str(mask_dir / f"{prefix}.npy"), arr)
    df = pd.DataFrame({"participant_id": subjects})
    df.to_csv(str(mask_dir / f"{prefix}_subject.csv"), index=False)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

class TestArgumentParsing:
    def test_required_args(self, tmp_path):
        script = PurgeSubject()
        args = script.parse_args([str(tmp_path), "--subject", "sub-01"])
        assert args.derivatives == str(tmp_path)
        assert args.subject == "sub-01"

    def test_dry_run_defaults_false(self, tmp_path):
        script = PurgeSubject()
        args = script.parse_args([str(tmp_path), "--subject", "sub-01"])
        assert args.dry_run is False

    def test_dry_run_flag(self, tmp_path):
        script = PurgeSubject()
        args = script.parse_args([str(tmp_path), "--subject", "sub-01", "--dry-run"])
        assert args.dry_run is True

    def test_missing_subject_raises(self, tmp_path):
        script = PurgeSubject()
        with pytest.raises(SystemExit):
            script.parse_args([str(tmp_path)])


# ---------------------------------------------------------------------------
# run() — missing derivatives directory
# ---------------------------------------------------------------------------

class TestRunValidation:
    def test_raises_when_derivatives_missing(self, tmp_path):
        script = _make_script(str(tmp_path / "nonexistent"), "sub-01")
        with pytest.raises(ValueError, match="not found"):
            script.run()

    def test_skips_crop_purge_when_crops_2mm_missing(self, tmp_path, capsys):
        tmp_path.mkdir(exist_ok=True)
        script = _make_script(str(tmp_path), "sub-01")
        result = script.run()
        assert result == 0
        assert "skipping crop purge" in capsys.readouterr().out.lower()

    def test_returns_zero_on_success(self, tmp_path):
        crops_2mm = tmp_path / "crops" / "2mm"
        crops_2mm.mkdir(parents=True)
        script = _make_script(str(tmp_path), "sub-01")
        assert script.run() == 0


# ---------------------------------------------------------------------------
# _purge_per_subject_dirs
# ---------------------------------------------------------------------------

class TestPurgePerSubjectDirs:
    def test_removes_subject_subdir(self, tmp_path):
        subject = "sub-42"
        skel_dir = tmp_path / "skeletons" / subject
        skel_dir.mkdir(parents=True)
        (skel_dir / "file.nii.gz").touch()

        script = _make_script(str(tmp_path), subject)
        script._purge_per_subject_dirs(str(tmp_path))

        assert not skel_dir.exists()

    def test_removes_all_known_top_level_dirs(self, tmp_path):
        subject = "sub-99"
        dirs = ["skeletons", "foldlabels", "extremities", "transforms", "distmaps"]
        for d in dirs:
            subj_dir = tmp_path / d / subject
            subj_dir.mkdir(parents=True)
            (subj_dir / "x.nii.gz").touch()

        script = _make_script(str(tmp_path), subject)
        script._purge_per_subject_dirs(str(tmp_path))

        for d in dirs:
            assert not (tmp_path / d / subject).exists()

    def test_does_not_remove_other_subjects(self, tmp_path):
        skel_a = tmp_path / "skeletons" / "sub-A"
        skel_b = tmp_path / "skeletons" / "sub-B"
        skel_a.mkdir(parents=True)
        skel_b.mkdir(parents=True)

        script = _make_script(str(tmp_path), "sub-A")
        script._purge_per_subject_dirs(str(tmp_path))

        assert not skel_a.exists()
        assert skel_b.exists()

    def test_missing_top_level_dir_is_skipped(self, tmp_path):
        script = _make_script(str(tmp_path), "sub-01")
        script._purge_per_subject_dirs(str(tmp_path))

    def test_dry_run_does_not_delete_dir(self, tmp_path):
        subject = "sub-01"
        skel_dir = tmp_path / "skeletons" / subject
        skel_dir.mkdir(parents=True)

        script = _make_script(str(tmp_path), subject, dry_run=True)
        script._purge_per_subject_dirs(str(tmp_path))

        assert skel_dir.exists()

    def test_dry_run_prints_delete_dir(self, tmp_path, capsys):
        subject = "sub-01"
        (tmp_path / "skeletons" / subject).mkdir(parents=True)

        script = _make_script(str(tmp_path), subject, dry_run=True)
        script._purge_per_subject_dirs(str(tmp_path))

        out = capsys.readouterr().out
        assert "dry-run" in out.lower()
        assert "skeletons" in out


# ---------------------------------------------------------------------------
# _purge_per_subject_crop_files
# ---------------------------------------------------------------------------

class TestPurgePerSubjectCropFiles:
    def test_removes_matching_nii_gz(self, tmp_path):
        crops_2mm = tmp_path / "crops" / "2mm"
        f = _make_crop_file(crops_2mm, "S.C.-sylv.", "Lcrops", "sub-01")

        script = _make_script(str(tmp_path), "sub-01")
        script._purge_per_subject_crop_files(str(crops_2mm))

        assert not f.exists()

    def test_does_not_remove_other_subject(self, tmp_path):
        crops_2mm = tmp_path / "crops" / "2mm"
        f_keep = _make_crop_file(crops_2mm, "S.C.-sylv.", "Lcrops", "sub-02")
        _make_crop_file(crops_2mm, "S.C.-sylv.", "Lcrops", "sub-01")

        script = _make_script(str(tmp_path), "sub-01")
        script._purge_per_subject_crop_files(str(crops_2mm))

        assert f_keep.exists()

    def test_strips_various_extensions(self, tmp_path):
        crops_2mm = tmp_path / "crops" / "2mm"
        f_nii = _make_crop_file(crops_2mm, "region_x", "Rcrops", "sub-03", ".nii")
        f_minf = _make_crop_file(crops_2mm, "region_x", "Rcrops", "sub-03", ".minf")

        script = _make_script(str(tmp_path), "sub-03")
        script._purge_per_subject_crop_files(str(crops_2mm))

        assert not f_nii.exists()
        assert not f_minf.exists()

    def test_skips_region_without_mask_subdir(self, tmp_path):
        crops_2mm = tmp_path / "crops" / "2mm"
        region_dir = crops_2mm / "region_no_mask"
        region_dir.mkdir(parents=True)
        stray = region_dir / "sub-01.nii.gz"
        stray.touch()

        script = _make_script(str(tmp_path), "sub-01")
        script._purge_per_subject_crop_files(str(crops_2mm))

        assert stray.exists()

    def test_dry_run_does_not_delete_file(self, tmp_path):
        crops_2mm = tmp_path / "crops" / "2mm"
        f = _make_crop_file(crops_2mm, "region_a", "Lcrops", "sub-01")

        script = _make_script(str(tmp_path), "sub-01", dry_run=True)
        script._purge_per_subject_crop_files(str(crops_2mm))

        assert f.exists()


# ---------------------------------------------------------------------------
# _purge_aggregated_arrays
# ---------------------------------------------------------------------------

class TestPurgeAggregatedArrays:
    def test_removes_subject_row_from_npy(self, tmp_path):
        crops_2mm = tmp_path / "crops" / "2mm"
        _make_aggregated_array(crops_2mm, "region_a", "Lskeleton", ["sub-01", "sub-02", "sub-03"])

        script = _make_script(str(tmp_path), "sub-02")
        script._purge_aggregated_arrays(str(crops_2mm))

        arr = np.load(str(crops_2mm / "region_a" / "mask" / "Lskeleton.npy"))
        assert arr.shape[0] == 2

        df = pd.read_csv(str(crops_2mm / "region_a" / "mask" / "Lskeleton_subject.csv"))
        assert "sub-02" not in df.iloc[:, 0].astype(str).tolist()

    def test_preserves_other_subjects_in_npy(self, tmp_path):
        crops_2mm = tmp_path / "crops" / "2mm"
        subjects = ["sub-01", "sub-02", "sub-03"]
        _make_aggregated_array(crops_2mm, "region_a", "Lskeleton", subjects)

        script = _make_script(str(tmp_path), "sub-02")
        script._purge_aggregated_arrays(str(crops_2mm))

        df = pd.read_csv(str(crops_2mm / "region_a" / "mask" / "Lskeleton_subject.csv"))
        remaining = df.iloc[:, 0].astype(str).tolist()
        assert "sub-01" in remaining
        assert "sub-03" in remaining

    def test_handles_subject_not_in_any_array(self, tmp_path, capsys):
        crops_2mm = tmp_path / "crops" / "2mm"
        _make_aggregated_array(crops_2mm, "region_a", "Lskeleton", ["sub-01"])

        script = _make_script(str(tmp_path), "sub-99")
        script._purge_aggregated_arrays(str(crops_2mm))

        out = capsys.readouterr().out
        assert "not found" in out.lower()

    def test_processes_multiple_regions(self, tmp_path):
        crops_2mm = tmp_path / "crops" / "2mm"
        for region in ("region_a", "region_b"):
            _make_aggregated_array(crops_2mm, region, "Lskeleton", ["sub-01", "sub-02"])

        script = _make_script(str(tmp_path), "sub-01")
        script._purge_aggregated_arrays(str(crops_2mm))

        for region in ("region_a", "region_b"):
            df = pd.read_csv(str(crops_2mm / region / "mask" / "Lskeleton_subject.csv"))
            assert "sub-01" not in df.iloc[:, 0].astype(str).tolist()

    def test_dry_run_does_not_modify_npy(self, tmp_path):
        crops_2mm = tmp_path / "crops" / "2mm"
        _make_aggregated_array(crops_2mm, "region_a", "Lskeleton", ["sub-01", "sub-02"])

        script = _make_script(str(tmp_path), "sub-01", dry_run=True)
        script._purge_aggregated_arrays(str(crops_2mm))

        arr = np.load(str(crops_2mm / "region_a" / "mask" / "Lskeleton.npy"))
        assert arr.shape[0] == 2

    def test_numeric_subject_ids_are_matched(self, tmp_path):
        crops_2mm = tmp_path / "crops" / "2mm"
        _make_aggregated_array(crops_2mm, "region_a", "Lskeleton", ["123456", "789012"])

        script = _make_script(str(tmp_path), "123456")
        script._purge_aggregated_arrays(str(crops_2mm))

        df = pd.read_csv(str(crops_2mm / "region_a" / "mask" / "Lskeleton_subject.csv"))
        assert "123456" not in df.iloc[:, 0].astype(str).tolist()
        assert "789012" in df.iloc[:, 0].astype(str).tolist()


# ---------------------------------------------------------------------------
# run() — end-to-end
# ---------------------------------------------------------------------------

class TestRunEndToEnd:
    def test_full_purge_removes_all_traces(self, tmp_path):
        subject = "sub-01"
        crops_2mm = tmp_path / "crops" / "2mm"

        skel_dir = tmp_path / "skeletons" / subject
        skel_dir.mkdir(parents=True)
        (skel_dir / "x.nii.gz").touch()

        crop_file = _make_crop_file(crops_2mm, "region_a", "Lcrops", subject)
        _make_aggregated_array(crops_2mm, "region_a", "Lskeleton", [subject, "sub-02"])

        script = _make_script(str(tmp_path), subject)
        result = script.run()

        assert result == 0
        assert not skel_dir.exists()
        assert not crop_file.exists()

        df = pd.read_csv(str(crops_2mm / "region_a" / "mask" / "Lskeleton_subject.csv"))
        assert subject not in df.iloc[:, 0].astype(str).tolist()

    def test_dry_run_leaves_everything_intact(self, tmp_path):
        subject = "sub-01"
        crops_2mm = tmp_path / "crops" / "2mm"

        skel_dir = tmp_path / "skeletons" / subject
        skel_dir.mkdir(parents=True)
        (skel_dir / "x.nii.gz").touch()

        crop_file = _make_crop_file(crops_2mm, "region_a", "Lcrops", subject)
        _make_aggregated_array(crops_2mm, "region_a", "Lskeleton", [subject, "sub-02"])

        script = _make_script(str(tmp_path), subject, dry_run=True)
        script.run()

        assert skel_dir.exists()
        assert crop_file.exists()
        arr = np.load(str(crops_2mm / "region_a" / "mask" / "Lskeleton.npy"))
        assert arr.shape[0] == 2
