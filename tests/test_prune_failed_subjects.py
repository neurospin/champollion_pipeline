#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for prune_failed_subjects.py
"""

import pytest
from pathlib import Path

from prune_failed_subjects import PruneFailedSubjects
from utils.lib import DERIVATIVES_FOLDER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_crops(base: Path, regions: list[str], subjects: list[str]) -> None:
    """Create fake crop files under crops/2mm/{region}/{subject}.nii.gz."""
    for region in regions:
        region_dir = base / DERIVATIVES_FOLDER / "crops" / "2mm" / region
        region_dir.mkdir(parents=True, exist_ok=True)
        for subject in subjects:
            (region_dir / f"{subject}.nii.gz").touch()


def _make_qc_tsv(path: Path, rows: list[tuple[str, int]]) -> Path:
    """Write a QC TSV file; rows is a list of (participant_id, qc) tuples."""
    path.write_text("participant_id\tqc\n" + "\n".join(f"{s}\t{q}" for s, q in rows))
    return path


def _make_qc_csv(path: Path, rows: list[tuple[str, int]]) -> Path:
    """Write a QC CSV file."""
    path.write_text("participant_id,qc\n" + "\n".join(f"{s},{q}" for s, q in rows))
    return path


def _make_script_with_args(tmp_path: Path, qc_path: Path, dry_run: bool = False):
    """Build a PruneFailedSubjects instance with pre-parsed args."""
    script = PruneFailedSubjects()
    args = script.parse_args([str(tmp_path), "--qc", str(qc_path)] + (["--dry-run"] if dry_run else []))
    script.args = args
    return script


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

class TestArgumentParsing:
    def test_required_output_and_qc(self, tmp_path):
        script = PruneFailedSubjects()
        args = script.parse_args([str(tmp_path), "--qc", "/some/qc.tsv"])
        assert args.output == str(tmp_path)
        assert args.qc == "/some/qc.tsv"

    def test_dry_run_defaults_to_false(self, tmp_path):
        script = PruneFailedSubjects()
        args = script.parse_args([str(tmp_path), "--qc", "/some/qc.tsv"])
        assert args.dry_run is False

    def test_dry_run_flag(self, tmp_path):
        script = PruneFailedSubjects()
        args = script.parse_args([str(tmp_path), "--qc", "/some/qc.tsv", "--dry-run"])
        assert args.dry_run is True

    def test_missing_qc_raises(self, tmp_path):
        script = PruneFailedSubjects()
        with pytest.raises(SystemExit):
            script.parse_args([str(tmp_path)])


# ---------------------------------------------------------------------------
# _discover_subjects
# ---------------------------------------------------------------------------

class TestDiscoverSubjects:
    def test_discovers_subjects_from_nii_gz(self, tmp_path):
        crops = tmp_path / "crops" / "2mm"
        region = crops / "S.C.-sylv."
        region.mkdir(parents=True)
        (region / "sub-01.nii.gz").touch()
        (region / "sub-02.nii.gz").touch()

        script = PruneFailedSubjects()
        result = script._discover_subjects(str(crops))
        assert result == {"sub-01", "sub-02"}

    def test_discovers_subjects_across_regions(self, tmp_path):
        crops = tmp_path / "crops" / "2mm"
        for region in ("region_a", "region_b"):
            d = crops / region
            d.mkdir(parents=True)
            (d / "sub-01.nii.gz").touch()
        (crops / "region_b" / "sub-03.nii.gz").touch()

        script = PruneFailedSubjects()
        result = script._discover_subjects(str(crops))
        assert result == {"sub-01", "sub-03"}

    def test_strips_nii_extension(self, tmp_path):
        crops = tmp_path / "crops" / "2mm"
        region = crops / "region_x"
        region.mkdir(parents=True)
        (region / "sub-04.nii").touch()

        script = PruneFailedSubjects()
        result = script._discover_subjects(str(crops))
        assert "sub-04" in result

    def test_ignores_non_directory_entries_in_crops(self, tmp_path):
        crops = tmp_path / "crops" / "2mm"
        crops.mkdir(parents=True)
        (crops / "stray_file.txt").touch()
        region = crops / "region_y"
        region.mkdir()
        (region / "sub-05.nii.gz").touch()

        script = PruneFailedSubjects()
        result = script._discover_subjects(str(crops))
        assert result == {"sub-05"}

    def test_empty_crops_dir_returns_empty_set(self, tmp_path):
        crops = tmp_path / "crops" / "2mm"
        crops.mkdir(parents=True)

        script = PruneFailedSubjects()
        result = script._discover_subjects(str(crops))
        assert result == set()


# ---------------------------------------------------------------------------
# _read_passing_subjects
# ---------------------------------------------------------------------------

class TestReadPassingSubjects:
    def test_reads_tsv_and_returns_passing(self, tmp_path):
        qc = _make_qc_tsv(tmp_path / "qc.tsv", [("sub-01", 1), ("sub-02", 0), ("sub-03", 1)])
        script = PruneFailedSubjects()
        result = script._read_passing_subjects(str(qc))
        assert result == {"sub-01", "sub-03"}

    def test_reads_csv_and_returns_passing(self, tmp_path):
        qc = _make_qc_csv(tmp_path / "qc.csv", [("sub-A", 1), ("sub-B", 0)])
        script = PruneFailedSubjects()
        result = script._read_passing_subjects(str(qc))
        assert result == {"sub-A"}

    def test_excludes_qc_zero(self, tmp_path):
        qc = _make_qc_tsv(tmp_path / "qc.tsv", [("sub-01", 0)])
        script = PruneFailedSubjects()
        result = script._read_passing_subjects(str(qc))
        assert "sub-01" not in result

    def test_all_failing_returns_empty(self, tmp_path):
        qc = _make_qc_tsv(tmp_path / "qc.tsv", [("sub-01", 0), ("sub-02", 0)])
        script = PruneFailedSubjects()
        result = script._read_passing_subjects(str(qc))
        assert result == set()

    def test_participant_id_cast_to_str(self, tmp_path):
        """Numeric participant IDs in the file should be coerced to strings."""
        qc = tmp_path / "qc.tsv"
        qc.write_text("participant_id\tqc\n12345\t1\n")
        script = PruneFailedSubjects()
        result = script._read_passing_subjects(str(qc))
        assert "12345" in result


# ---------------------------------------------------------------------------
# run() — integration-level (real filesystem, no subprocess)
# ---------------------------------------------------------------------------

class TestRun:
    def test_raises_when_derivatives_dir_missing(self, tmp_path):
        qc = _make_qc_tsv(tmp_path / "qc.tsv", [("sub-01", 1)])
        script = _make_script_with_args(tmp_path, qc)
        with pytest.raises(ValueError, match=DERIVATIVES_FOLDER):
            script.run()

    def test_raises_when_crops_dir_missing(self, tmp_path):
        (tmp_path / DERIVATIVES_FOLDER).mkdir()
        qc = _make_qc_tsv(tmp_path / "qc.tsv", [("sub-01", 1)])
        script = _make_script_with_args(tmp_path, qc)
        with pytest.raises(ValueError, match="crops"):
            script.run()

    def test_nothing_pruned_when_all_pass(self, tmp_path):
        _make_crops(tmp_path, ["region_a"], ["sub-01", "sub-02"])
        qc = _make_qc_tsv(tmp_path / "qc.tsv", [("sub-01", 1), ("sub-02", 1)])
        script = _make_script_with_args(tmp_path, qc)
        script.run()

        remaining = list((tmp_path / DERIVATIVES_FOLDER / "crops" / "2mm" / "region_a").iterdir())
        assert len(remaining) == 2

    def test_failed_subject_files_are_deleted(self, tmp_path):
        _make_crops(tmp_path, ["region_a", "region_b"], ["sub-01", "sub-02"])
        qc = _make_qc_tsv(tmp_path / "qc.tsv", [("sub-01", 1), ("sub-02", 0)])
        script = _make_script_with_args(tmp_path, qc)
        script.run()

        for region in ("region_a", "region_b"):
            region_dir = tmp_path / DERIVATIVES_FOLDER / "crops" / "2mm" / region
            names = {f.name for f in region_dir.iterdir()}
            assert "sub-02.nii.gz" not in names
            assert "sub-01.nii.gz" in names

    def test_subject_absent_from_qc_is_deleted(self, tmp_path):
        """A subject not listed in the QC file should be treated as failed."""
        _make_crops(tmp_path, ["region_a"], ["sub-01", "sub-99"])
        qc = _make_qc_tsv(tmp_path / "qc.tsv", [("sub-01", 1)])
        script = _make_script_with_args(tmp_path, qc)
        script.run()

        region_dir = tmp_path / DERIVATIVES_FOLDER / "crops" / "2mm" / "region_a"
        names = {f.name for f in region_dir.iterdir()}
        assert "sub-99.nii.gz" not in names
        assert "sub-01.nii.gz" in names

    def test_dry_run_does_not_delete_files(self, tmp_path):
        _make_crops(tmp_path, ["region_a"], ["sub-01", "sub-02"])
        qc = _make_qc_tsv(tmp_path / "qc.tsv", [("sub-01", 1), ("sub-02", 0)])
        script = _make_script_with_args(tmp_path, qc, dry_run=True)
        script.run()

        region_dir = tmp_path / DERIVATIVES_FOLDER / "crops" / "2mm" / "region_a"
        names = {f.name for f in region_dir.iterdir()}
        assert "sub-02.nii.gz" in names

    def test_dry_run_output_mentions_would_delete(self, tmp_path, capsys):
        _make_crops(tmp_path, ["region_a"], ["sub-02"])
        qc = _make_qc_tsv(tmp_path / "qc.tsv", [("sub-01", 1)])
        script = _make_script_with_args(tmp_path, qc, dry_run=True)
        script.run()

        out = capsys.readouterr().out
        assert "dry-run" in out.lower()
        assert "sub-02" in out

    def test_deletes_files_across_subdirectories(self, tmp_path):
        """Non-crop subdirs (e.g. skeletons) should also have failing-subject files removed."""
        _make_crops(tmp_path, ["region_a"], ["sub-01", "sub-02"])
        # Add a skeleton file for sub-02 in a different subdir
        skel_dir = tmp_path / DERIVATIVES_FOLDER / "skeletons" / "2mm" / "region_a"
        skel_dir.mkdir(parents=True)
        (skel_dir / "sub-02.nii.gz").touch()
        (skel_dir / "sub-01.nii.gz").touch()

        qc = _make_qc_tsv(tmp_path / "qc.tsv", [("sub-01", 1), ("sub-02", 0)])
        script = _make_script_with_args(tmp_path, qc)
        script.run()

        assert not (skel_dir / "sub-02.nii.gz").exists()
        assert (skel_dir / "sub-01.nii.gz").exists()

    def test_returns_zero_on_success(self, tmp_path):
        _make_crops(tmp_path, ["region_a"], ["sub-01"])
        qc = _make_qc_tsv(tmp_path / "qc.tsv", [("sub-01", 1)])
        script = _make_script_with_args(tmp_path, qc)
        assert script.run() == 0

    def test_no_subjects_in_crops_returns_early(self, tmp_path, capsys):
        (tmp_path / DERIVATIVES_FOLDER / "crops" / "2mm").mkdir(parents=True)
        qc = _make_qc_tsv(tmp_path / "qc.tsv", [("sub-01", 1)])
        script = _make_script_with_args(tmp_path, qc)
        result = script.run()
        assert result == 0
        assert "nothing" in capsys.readouterr().out.lower()
