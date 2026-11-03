#!/usr/bin/env python3
"""
Unit tests for prune_failed_subjects.py
"""

from pathlib import Path

import pytest

from champollion_pipeline.prune_failed_subjects import PruneFailedSubjects
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from tests.log_assertions import OUTPUT
from tests.prune_layout import CROP_SUFFIX, crop_file, crops_2mm, make_crop_files, skeleton_file, touch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_crops(base: Path, regions: list[str], subjects: list[str]) -> None:
    """Create crop files crops/2mm/{region}/mask/Rcrops/{subject}_cropped_skeleton.nii.gz (real layout)."""
    for region in regions:
        make_crop_files(base, region, "R", subjects)


def _crop_names(base: Path, region: str) -> set[str]:
    """Names of the files left in crops/2mm/{region}/mask/Rcrops."""
    return {f.name for f in (crops_2mm(base) / region / "mask" / "Rcrops").iterdir()}


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
    def test_discovers_subjects_from_cropped_skeleton_files(self, tmp_path):
        make_crop_files(tmp_path, "S.C.-sylv.", "R", ["sub-01", "sub-02"])

        script = PruneFailedSubjects()
        result = script._discover_subjects(str(crops_2mm(tmp_path)))
        assert result == {"sub-01", "sub-02"}

    def test_discovers_subjects_across_regions(self, tmp_path):
        for region in ("region_a", "region_b"):
            make_crop_files(tmp_path, region, "R", ["sub-01"])
        make_crop_files(tmp_path, "region_b", "R", ["sub-03"])

        script = PruneFailedSubjects()
        result = script._discover_subjects(str(crops_2mm(tmp_path)))
        assert result == {"sub-01", "sub-03"}

    def test_ignores_non_directory_entries_in_crops(self, tmp_path):
        crops = crops_2mm(tmp_path)
        touch(crops / "stray_file.txt")
        make_crop_files(tmp_path, "region_y", "R", ["sub-05"])

        script = PruneFailedSubjects()
        result = script._discover_subjects(str(crops))
        assert result == {"sub-05"}

    def test_empty_crops_dir_returns_empty_set(self, tmp_path):
        crops = crops_2mm(tmp_path)
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

        assert len(_crop_names(tmp_path, "region_a")) == 2

    def test_failed_subject_files_are_deleted(self, tmp_path):
        _make_crops(tmp_path, ["region_a", "region_b"], ["sub-01", "sub-02"])
        qc = _make_qc_tsv(tmp_path / "qc.tsv", [("sub-01", 1), ("sub-02", 0)])
        script = _make_script_with_args(tmp_path, qc)
        script.run()

        for region in ("region_a", "region_b"):
            names = _crop_names(tmp_path, region)
            assert f"sub-02{CROP_SUFFIX}" not in names
            assert f"sub-01{CROP_SUFFIX}" in names

    def test_subject_absent_from_qc_is_deleted(self, tmp_path):
        """A subject not listed in the QC file should be treated as failed."""
        _make_crops(tmp_path, ["region_a"], ["sub-01", "sub-99"])
        qc = _make_qc_tsv(tmp_path / "qc.tsv", [("sub-01", 1)])
        script = _make_script_with_args(tmp_path, qc)
        script.run()

        names = _crop_names(tmp_path, "region_a")
        assert f"sub-99{CROP_SUFFIX}" not in names
        assert f"sub-01{CROP_SUFFIX}" in names

    def test_dry_run_does_not_delete_files(self, tmp_path):
        _make_crops(tmp_path, ["region_a"], ["sub-01", "sub-02"])
        qc = _make_qc_tsv(tmp_path / "qc.tsv", [("sub-01", 1), ("sub-02", 0)])
        script = _make_script_with_args(tmp_path, qc, dry_run=True)
        script.run()

        assert f"sub-02{CROP_SUFFIX}" in _crop_names(tmp_path, "region_a")

    def test_dry_run_output_mentions_would_delete(self, tmp_path, pipeline_caplog):
        """REQ-LOGGING-BDRABCZUK-728A1E8B1120: the dry-run listing is an OUTPUT record."""
        _make_crops(tmp_path, ["region_a"], ["sub-02"])
        qc = _make_qc_tsv(tmp_path / "qc.tsv", [("sub-01", 1)])
        script = _make_script_with_args(tmp_path, qc, dry_run=True)
        script.run()

        listed = crop_file(tmp_path, "region_a", "R", "sub-02")
        record = pipeline_caplog.assert_logged(OUTPUT, f"[dry-run] Would delete: {listed}")
        assert record.name == "champollion_pipeline.prune_failed_subjects"

    def test_deletes_files_across_subdirectories(self, tmp_path):
        """Non-crop subdirs (e.g. skeletons) should also have failing-subject files removed."""
        _make_crops(tmp_path, ["region_a"], ["sub-01", "sub-02"])
        # Resampled skeletons of both subjects, outside crops/
        failed_skel = touch(skeleton_file(tmp_path, "R", "sub-02"))
        passing_skel = touch(skeleton_file(tmp_path, "R", "sub-01"))

        qc = _make_qc_tsv(tmp_path / "qc.tsv", [("sub-01", 1), ("sub-02", 0)])
        script = _make_script_with_args(tmp_path, qc)
        script.run()

        assert not failed_skel.exists()
        assert passing_skel.exists()

    def test_returns_zero_on_success(self, tmp_path):
        _make_crops(tmp_path, ["region_a"], ["sub-01"])
        qc = _make_qc_tsv(tmp_path / "qc.tsv", [("sub-01", 1)])
        script = _make_script_with_args(tmp_path, qc)
        assert script.run() == 0

    def test_no_subjects_in_crops_returns_early(self, tmp_path, pipeline_caplog):
        """REQ-LOGGING-BDRABCZUK-09E839CCB79E: nothing to prune is an OUTPUT record."""
        crops_2mm(tmp_path).mkdir(parents=True)
        qc = _make_qc_tsv(tmp_path / "qc.tsv", [("sub-01", 1)])
        script = _make_script_with_args(tmp_path, qc)
        result = script.run()
        assert result == 0
        record = pipeline_caplog.assert_logged(OUTPUT, "Nothing to prune")
        assert record.name == "champollion_pipeline.prune_failed_subjects"
