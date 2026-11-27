#!/usr/bin/env python3
"""Pre-check of main.py on a pure morphologist-cli tree (TASK-ARGDISC-BDRABCZUK-04D084972C8C, found at green).

A morphologist-cli tree (IDEASII) holds sub-<label>/ses-1/run-0/ana-0/... and no t1mri folder. The pre-check judges
scans <subject>/t1mri/<acquisition> only (SubjectEligibilityChecker.list_scans), so it lists no scan for such a tree,
though graph discovery finds its subjects. Known gap, left to a follow-up Task: this test pins it.
"""

import logging
from pathlib import Path

import pytest

from champollion_pipeline.config import PipelineConfig
from champollion_pipeline.file_indexer.pipeline_checks import SubjectEligibilityChecker
from champollion_pipeline.file_indexer.scan_id import ScanId
from champollion_pipeline.pipeline_stages import RunCorticalTilesStage
from tests.morphologist_layout_fixtures import CLI_SUBJECTS, build_morphologist_cli_tree

pytestmark = pytest.mark.unit  # noqa: V107


@pytest.mark.xfail(strict=True, reason="pure cli tree listing: follow-up Task")
def test_pre_check_lists_the_subjects_of_a_pure_cli_tree(tmp_path):
    """Every subject graph discovery finds in a pure cli tree is judged by the pre-check, here as eligible."""
    root = build_morphologist_cli_tree(Path(tmp_path))
    config = PipelineConfig(root_path="/r", data_path="/r/data", models_path="/r/models", outputs_path="/r/outputs")
    config.dataset.path_to_graph = None  # noqa: V101 - graphs are found by stage 2's input QC
    stage = RunCorticalTilesStage("run_cortical_tiles", config, logging.getLogger("pre_check_pure_cli"))

    report = SubjectEligibilityChecker(str(root), stage.required_file_patterns()).check_subject_patterns()

    assert sorted(scan.subject for scan in report.eligible) == sorted(CLI_SUBJECTS)
    assert report.ineligible == {}


def _stage_patterns() -> list[str]:
    config = PipelineConfig(root_path="/r", data_path="/r/data", models_path="/r/models", outputs_path="/r/outputs")
    config.dataset.path_to_graph = None  # noqa: V101 - graphs are found by stage 2's input QC
    return RunCorticalTilesStage(
        "run_cortical_tiles", config, logging.getLogger("pre_check_text")
    ).required_file_patterns()


def test_report_names_the_cli_pattern_of_a_subject_without_t1mri(tmp_path, monkeypatch):
    """A missing skeleton of a subject with no t1mri folder is named by its cli pattern, never the raw "|" string."""
    root = build_morphologist_cli_tree(Path(tmp_path))
    subject = sorted(CLI_SUBJECTS)[0]
    for skeleton in (root / subject).rglob("*_skeleton.nii.gz"):
        skeleton.unlink()
    checker = SubjectEligibilityChecker(str(root), _stage_patterns())
    scan = ScanId(subject, "ses-1")
    monkeypatch.setattr(checker, "list_scans", lambda: [scan])

    report = checker.check_subject_patterns()

    skeleton_texts = [text for text in report.ineligible[scan] if "skeleton" in text]
    assert skeleton_texts == ["**/*_hemi-L_skeleton.nii.gz", "**/*_hemi-R_skeleton.nii.gz"]
    assert not any("|" in text for text in report.ineligible[scan])


def test_report_names_the_classic_pattern_of_a_subject_with_t1mri(tmp_path):
    """A subject with a t1mri folder and no skeleton keeps the classic text of before the cli naming."""
    (tmp_path / "sub1" / "t1mri" / "acq1").mkdir(parents=True)
    checker = SubjectEligibilityChecker(str(tmp_path), _stage_patterns())

    report = checker.check_subject_patterns()

    (missing,) = report.ineligible.values()
    skeleton_texts = [text for text in missing if "skeleton" in text]
    assert [text.rsplit("/", 1)[1] for text in skeleton_texts] == ["Lskeleton*.nii.gz", "Rskeleton*.nii.gz"]
    assert not any("|" in text or "**" in text for text in skeleton_texts)
