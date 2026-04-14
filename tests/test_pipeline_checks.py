"""Tests for src/file_indexer/pipeline_checks.py"""

import json
import os
import sys
from pathlib import Path

import pytest

# Make sure src/ is on the path (same pattern as other test files in this project)
SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from file_indexer.pipeline_checks import (
    SubjectEligibilityChecker,
    SubjectEligibilityReport,
    OutputIndexReport,
    build_output_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _touch(path: Path, content: bytes = b"") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


# ---------------------------------------------------------------------------
# SubjectEligibilityChecker tests
# ---------------------------------------------------------------------------

class TestSubjectEligibilityChecker:

    def _make_subject(self, root: Path, subject: str, files: list) -> None:
        """Create a fake subject directory with specified relative file paths."""
        (root / subject).mkdir(parents=True, exist_ok=True)
        for f in files:
            _touch(root / subject / f)

    def test_all_eligible(self, tmp_path):
        patterns = ["graphs/Rsub.arg", "skel/Rsub.nii.gz"]
        for sub in ["sub_001", "sub_002", "sub_003"]:
            self._make_subject(tmp_path, sub, ["graphs/Rsub.arg", "skel/Rsub.nii.gz"])
        checker = SubjectEligibilityChecker(str(tmp_path), patterns)
        report = checker.check()
        assert sorted(report.eligible) == ["sub_001", "sub_002", "sub_003"]
        assert report.ineligible == {}

    def test_missing_graph_file(self, tmp_path):
        patterns = ["graphs/R*.arg", "skel/R*.nii.gz"]
        self._make_subject(tmp_path, "sub_ok", ["graphs/Rsub.arg", "skel/Rsub.nii.gz"])
        self._make_subject(tmp_path, "sub_bad", ["skel/Rsub.nii.gz"])  # no .arg
        checker = SubjectEligibilityChecker(str(tmp_path), patterns)
        report = checker.check()
        assert "sub_ok" in report.eligible
        assert "sub_bad" in report.ineligible
        assert any("*.arg" in p for p in report.ineligible["sub_bad"])

    def test_missing_skeleton_file(self, tmp_path):
        patterns = ["graphs/R*.arg", "skel/R*.nii.gz"]
        self._make_subject(tmp_path, "sub_ok", ["graphs/Rsub.arg", "skel/Rsub.nii.gz"])
        self._make_subject(tmp_path, "sub_bad", ["graphs/Rsub.arg"])  # no .nii.gz
        checker = SubjectEligibilityChecker(str(tmp_path), patterns)
        report = checker.check()
        assert "sub_ok" in report.eligible
        assert "sub_bad" in report.ineligible
        assert any("*.nii.gz" in p for p in report.ineligible["sub_bad"])

    def test_empty_patterns_all_eligible(self, tmp_path):
        (tmp_path / "sub_001").mkdir()
        (tmp_path / "sub_002").mkdir()
        checker = SubjectEligibilityChecker(str(tmp_path), [])
        report = checker.check()
        assert sorted(report.eligible) == ["sub_001", "sub_002"]
        assert report.ineligible == {}

    def test_empty_subjects_dir(self, tmp_path):
        checker = SubjectEligibilityChecker(str(tmp_path), ["graphs/R*.arg"])
        report = checker.check()
        assert report.eligible == []
        assert report.ineligible == {}

    def test_save_index_to_json(self, tmp_path):
        self._make_subject(tmp_path, "sub_001", ["graphs/Rsub.arg"])
        index_path = str(tmp_path / "index.json")
        checker = SubjectEligibilityChecker(str(tmp_path), ["graphs/R*.arg"])
        checker.check(save_index_to=index_path)
        assert os.path.exists(index_path)
        data = json.loads(Path(index_path).read_text())
        assert "t" in data and "root" in data

    def test_report_print(self, tmp_path, capsys):
        self._make_subject(tmp_path, "sub_ok", ["graphs/Rsub.arg"])
        self._make_subject(tmp_path, "sub_bad", [])
        checker = SubjectEligibilityChecker(
            str(tmp_path), ["graphs/R*.arg"], stage_name="my_stage"
        )
        report = checker.check()
        report.print()
        out = capsys.readouterr().out
        assert "my_stage" in out
        assert "Eligible" in out
        assert "sub_bad" in out

    def test_wildcard_pattern_matches(self, tmp_path):
        self._make_subject(tmp_path, "sub_001", ["graphs/Rsub_001.arg"])
        checker = SubjectEligibilityChecker(str(tmp_path), ["graphs/R*.arg"])
        report = checker.check()
        assert "sub_001" in report.eligible

    def test_no_wildcard_exact_match(self, tmp_path):
        self._make_subject(tmp_path, "sub_001", ["graphs/Rsub_001.arg"])
        checker = SubjectEligibilityChecker(str(tmp_path), ["graphs/Rsub_001.arg"])
        report = checker.check()
        assert "sub_001" in report.eligible
        # Wrong exact path → ineligible
        checker2 = SubjectEligibilityChecker(str(tmp_path), ["graphs/Rsub_999.arg"])
        report2 = checker2.check()
        assert "sub_001" in report2.ineligible


# ---------------------------------------------------------------------------
# build_output_report tests
# ---------------------------------------------------------------------------

class TestBuildOutputReport:

    def test_counts_files_by_extension(self, tmp_path):
        for i in range(5):
            _touch(tmp_path / f"file{i}.nii.gz")
        for i in range(2):
            _touch(tmp_path / f"meta{i}.json")
        report = build_output_report(str(tmp_path), "my_stage")
        assert report.total_files == 7
        assert report.by_ext.get(".gz", 0) == 5
        assert report.by_ext.get(".json", 0) == 2

    def test_empty_dir_returns_zero(self, tmp_path):
        report = build_output_report(str(tmp_path), "empty_stage")
        assert report.total_files == 0
        assert report.by_ext == {}

    def test_report_print(self, tmp_path, capsys):
        _touch(tmp_path / "out.nii.gz")
        report = build_output_report(str(tmp_path), "cortical_tiles")
        report.print()
        out = capsys.readouterr().out
        assert "cortical_tiles" in out
        assert "Total files" in out
        assert "1" in out

    def test_save_index_to_json(self, tmp_path):
        _touch(tmp_path / "out.nii.gz")
        index_path = str(tmp_path / "index.json")
        build_output_report(str(tmp_path), "stage", save_index_to=index_path)
        assert os.path.exists(index_path)
        data = json.loads(Path(index_path).read_text())
        assert "t" in data and "root" in data
