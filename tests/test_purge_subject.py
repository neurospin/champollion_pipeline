#!/usr/bin/env python3
"""
Unit tests for purge_subject.py
"""

import logging

import numpy as np
import pandas as pd
import pytest

from champollion_pipeline.purge_subject import PurgeSubject
from tests.log_assertions import OUTPUT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_script(derivatives: str, subject: str, dry_run: bool = False, masks: str | None = None) -> PurgeSubject:
    script = PurgeSubject()
    argv = [derivatives, "--subject", subject] + (["--dry-run"] if dry_run else [])
    if masks is not None:
        argv += ["--masks", masks]
    args = script.parse_args(argv)
    script.args = args
    return script


def _snapshot(root) -> dict:
    """Map every file under root (relative path) to its bytes."""
    return {str(f.relative_to(root)): f.read_bytes() for f in sorted(root.rglob("*")) if f.is_file()}


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

    @pytest.mark.parametrize(
        ("present", "masks"),
        [
            pytest.param(None, None, id="no-crops"),
            pytest.param(("crops", "2mm"), None, id="legacy-crops-2mm-only"),
            pytest.param(("crops", "canonical_25", "2mm"), "v_test", id="other-mask-version-only"),
        ],
    )
    def test_skips_crop_purge_when_crops_2mm_missing(self, tmp_path, pipeline_caplog, present, masks):
        """REQ-LOGGING-BDRABCZUK-1C45DB9E9400: no crops/<M>/2mm for --masks <M> gives a skipped-crop-purge WARNING."""
        (tmp_path / "skeletons" / "sub-01").mkdir(parents=True)
        if present is not None:
            tmp_path.joinpath(*present).mkdir(parents=True)
        script = _make_script(str(tmp_path), "sub-01", masks=masks)
        result = script.run()
        assert result == 0
        record = pipeline_caplog.assert_logged(logging.WARNING, "skipping crop purge")
        assert record.name == "champollion_pipeline.purge_subject"

    def test_returns_zero_on_success(self, tmp_path):
        crops_2mm = tmp_path / "crops" / "canonical_25" / "2mm"
        crops_2mm.mkdir(parents=True)
        (tmp_path / "skeletons" / "sub-01").mkdir(parents=True)
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

    def test_dry_run_prints_delete_dir(self, tmp_path, pipeline_caplog):
        """REQ-LOGGING-BDRABCZUK-B32699DFEAA6 and -8C8D59E9888B: the dry-run listing is an OUTPUT record."""
        subject = "sub-01"
        (tmp_path / "skeletons" / subject).mkdir(parents=True)

        script = _make_script(str(tmp_path), subject, dry_run=True)
        script._purge_per_subject_dirs(str(tmp_path))

        record = pipeline_caplog.assert_logged(OUTPUT, "Delete dir:")
        assert record.getMessage().startswith("[dry-run]")
        assert str(tmp_path / "skeletons" / subject) in record.getMessage()
        assert record.name == "champollion_pipeline.purge_subject"


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

    def test_handles_subject_not_in_any_array(self, tmp_path, pipeline_caplog):
        """REQ-LOGGING-BDRABCZUK-C1CBF454947E: a subject in no aggregated array is a WARNING record."""
        crops_2mm = tmp_path / "crops" / "2mm"
        _make_aggregated_array(crops_2mm, "region_a", "Lskeleton", ["sub-01"])

        script = _make_script(str(tmp_path), "sub-99")
        script._purge_aggregated_arrays(str(crops_2mm))

        record = pipeline_caplog.assert_logged(logging.WARNING, "not found")
        assert "sub-99" in record.getMessage().split("not found", 1)[0]
        assert record.name == "champollion_pipeline.purge_subject"

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
    @staticmethod
    def _populate(tmp_path, subject):
        """Skeleton dir, crop file and aggregated array of subject under crops/canonical_25/2mm."""
        crops_2mm = tmp_path / "crops" / "canonical_25" / "2mm"
        skel_dir = tmp_path / "skeletons" / subject
        skel_dir.mkdir(parents=True)
        (skel_dir / "x.nii.gz").touch()
        crop_file = _make_crop_file(crops_2mm, "region_a", "Lcrops", subject)
        _make_aggregated_array(crops_2mm, "region_a", "Lskeleton", [subject, "sub-02"])
        return crops_2mm, skel_dir, crop_file

    def test_full_purge_removes_all_traces(self, tmp_path):
        subject = "sub-01"
        crops_2mm, skel_dir, crop_file = self._populate(tmp_path, subject)

        script = _make_script(str(tmp_path), subject)
        result = script.run()

        assert result == 0
        assert not skel_dir.exists()
        assert not crop_file.exists()

        df = pd.read_csv(str(crops_2mm / "region_a" / "mask" / "Lskeleton_subject.csv"))
        assert subject not in df.iloc[:, 0].astype(str).tolist()

    def test_dry_run_leaves_everything_intact(self, tmp_path):
        subject = "sub-01"
        crops_2mm, skel_dir, crop_file = self._populate(tmp_path, subject)

        script = _make_script(str(tmp_path), subject, dry_run=True)
        script.run()

        assert skel_dir.exists()
        assert crop_file.exists()
        arr = np.load(str(crops_2mm / "region_a" / "mask" / "Lskeleton.npy"))
        assert arr.shape[0] == 2


# ---------------------------------------------------------------------------
# --masks: versioned crops root crops/<masks>/2mm
# ---------------------------------------------------------------------------


class TestMasksArgument:
    def test_masks_defaults_to_canonical_25(self, tmp_path):
        """REQ-PATTERNS-BDRABCZUK-7A4C1CB45061: no --masks yields masks canonical_25."""
        args = PurgeSubject().parse_args([str(tmp_path), "--subject", "sub-01"])
        assert getattr(args, "masks", None) == "canonical_25"


class TestMasksNameGuard:
    """A --masks value that cannot name a crops/<masks> directory is refused before any deletion."""

    @staticmethod
    def _populate(tmp_path):
        skel_dir = tmp_path / "skeletons" / "sub-01"
        skel_dir.mkdir(parents=True)
        (skel_dir / "x.nii.gz").touch()
        # Crop files where an unguarded crops/<masks>/2mm would land: crops/canonical_25/2mm
        # ("canonical_25/../canonical_25"), crops/2mm ("" and "."), <derivatives>/2mm ("..").
        for crops_dir in (tmp_path / "crops" / "canonical_25" / "2mm", tmp_path / "crops" / "2mm", tmp_path / "2mm"):
            _make_crop_file(crops_dir, "region_a", "Lcrops", "sub-01")

    @pytest.mark.parametrize("masks", ["a/b", "canonical_25/../canonical_25"])
    def test_masks_with_path_separator_raises_before_deleting(self, tmp_path, masks):
        """REQ-PATTERNS-BDRABCZUK-C023189CF7DA: a --masks value containing a path separator."""
        self._populate(tmp_path)
        before = _snapshot(tmp_path)
        script = _make_script(str(tmp_path), "sub-01", masks=masks)
        with pytest.raises(ValueError):
            script.run()
        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("masks", ["", ".", ".."])
    def test_degenerate_masks_raises_before_deleting(self, tmp_path, masks):
        """REQ-PATTERNS-BDRABCZUK-97378461582B: an empty, '.' or '..' --masks value."""
        self._populate(tmp_path)
        before = _snapshot(tmp_path)
        script = _make_script(str(tmp_path), "sub-01", masks=masks)
        with pytest.raises(ValueError):
            script.run()
        assert _snapshot(tmp_path) == before


class TestVersionedCropsRoot:
    """REQ-PATTERNS-BDRABCZUK-75B777B6C467 (purge half): crops are read from crops/<masks>/2mm only."""

    def test_default_masks_purges_crop_file_under_canonical_25(self, tmp_path):
        """REQ-PATTERNS-BDRABCZUK-75B777B6C467, -7A4C1CB45061, -BF1732DC44EE."""
        (tmp_path / "skeletons" / "sub-01").mkdir(parents=True)
        target = _make_crop_file(tmp_path / "crops" / "canonical_25" / "2mm", "region_a", "Lcrops", "sub-01")
        _make_crop_file(tmp_path / "crops" / "other_v" / "2mm", "region_a", "Lcrops", "sub-01")
        other_before = _snapshot(tmp_path / "crops" / "other_v")

        assert _make_script(str(tmp_path), "sub-01").run() == 0

        assert not target.exists()
        assert _snapshot(tmp_path / "crops" / "other_v") == other_before

    def test_explicit_masks_purges_crop_file_under_that_version(self, tmp_path):
        """REQ-PATTERNS-BDRABCZUK-75B777B6C467, -BF1732DC44EE."""
        (tmp_path / "skeletons" / "sub-01").mkdir(parents=True)
        target = _make_crop_file(tmp_path / "crops" / "v_test" / "2mm", "region_a", "Lcrops", "sub-01")
        _make_crop_file(tmp_path / "crops" / "canonical_25" / "2mm", "region_a", "Lcrops", "sub-01")
        other_before = _snapshot(tmp_path / "crops" / "canonical_25")

        assert _make_script(str(tmp_path), "sub-01", masks="v_test").run() == 0

        assert not target.exists()
        assert _snapshot(tmp_path / "crops" / "canonical_25") == other_before

    def test_explicit_masks_filters_aggregate_rows_under_that_version(self, tmp_path):
        """REQ-PATTERNS-BDRABCZUK-75B777B6C467, -BF1732DC44EE: aggregate rows of the chosen version only."""
        versioned = tmp_path / "crops" / "v_test" / "2mm"
        _make_aggregated_array(versioned, "region_a", "Lskeleton", ["sub-01", "sub-02"])
        other = tmp_path / "crops" / "canonical_25" / "2mm"
        _make_aggregated_array(other, "region_a", "Lskeleton", ["sub-01", "sub-02"])
        other_before = _snapshot(tmp_path / "crops" / "canonical_25")

        assert _make_script(str(tmp_path), "sub-01", masks="v_test").run() == 0

        df = pd.read_csv(str(versioned / "region_a" / "mask" / "Lskeleton_subject.csv"))
        assert df.iloc[:, 0].astype(str).tolist() == ["sub-02"]
        assert np.load(str(versioned / "region_a" / "mask" / "Lskeleton.npy")).shape[0] == 1
        assert _snapshot(tmp_path / "crops" / "canonical_25") == other_before

    def test_missing_versioned_crops_warning_names_path(self, tmp_path, pipeline_caplog):
        """REQ-PATTERNS-BDRABCZUK-4F1E0B3D0980: the warning names crops/<masks>/2mm."""
        (tmp_path / "skeletons" / "sub-01").mkdir(parents=True)

        _make_script(str(tmp_path), "sub-01", masks="v_test").run()

        record = pipeline_caplog.assert_logged(logging.WARNING, str(tmp_path / "crops" / "v_test" / "2mm"))
        assert record.name == "champollion_pipeline.purge_subject"

    def test_legacy_crops_only_warning_names_versioned_path(self, tmp_path, pipeline_caplog):
        """REQ-PATTERNS-BDRABCZUK-4F1E0B3D0980, -BF1732DC44EE: a legacy crops/2mm is neither read nor changed."""
        (tmp_path / "skeletons" / "sub-01").mkdir(parents=True)
        _make_crop_file(tmp_path / "crops" / "2mm", "region_a", "Lcrops", "sub-01")
        legacy_before = _snapshot(tmp_path / "crops" / "2mm")

        _make_script(str(tmp_path), "sub-01").run()

        record = pipeline_caplog.assert_logged(logging.WARNING, str(tmp_path / "crops" / "canonical_25" / "2mm"))
        assert record.name == "champollion_pipeline.purge_subject"
        assert _snapshot(tmp_path / "crops" / "2mm") == legacy_before
