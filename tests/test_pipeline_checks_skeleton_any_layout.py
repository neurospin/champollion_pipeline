#!/usr/bin/env python3
"""Pre-check of main.py on a pure morphologist-cli tree (TASK-ARGDISC-BDRABCZUK-04D084972C8C, found at green).

A morphologist-cli tree (IDEASII) holds sub-<label>/ses-1/run-0/ana-0/... and no t1mri folder. The pre-check judges
scans <subject>/t1mri/<acquisition> only (SubjectEligibilityChecker.list_scans), so it lists no scan for such a tree,
though graph discovery finds its subjects. TASK-ARGDISC-BDRABCZUK-A12B4526C531 closes the gap: a subject with no
t1mri folder and a graph below it is listed as one scan (first session only, for now).
"""

import logging
from pathlib import Path

import pytest

from champollion_pipeline.config import PipelineConfig
from champollion_pipeline.file_indexer.pipeline_checks import SubjectEligibilityChecker
from champollion_pipeline.file_indexer.scan_id import ScanId
from champollion_pipeline.pipeline_stages import RunCorticalTilesStage
from tests.morphologist_layout_fixtures import (
    CLASSIC_SUBJECTS,
    CLI_SUBJECTS,
    build_classic_tree,
    build_morphologist_cli_tree,
)

pytestmark = pytest.mark.unit  # noqa: V107


def test_pre_check_lists_the_subjects_of_a_pure_cli_tree(tmp_path):
    """Every subject graph discovery finds in a pure cli tree is judged by the pre-check, here as eligible."""
    root = build_morphologist_cli_tree(Path(tmp_path))
    config = PipelineConfig(root_path="/r", data_path="/r/data", models_path="/r/models", outputs_path="/r/outputs")
    config.dataset.path_to_graph = None  # noqa: V101 - graphs are found by stage 2's input QC
    stage = RunCorticalTilesStage("run_cortical_tiles", config, logging.getLogger("pre_check_pure_cli"))

    report = SubjectEligibilityChecker(str(root), stage.required_file_patterns()).check_subject_patterns()

    assert sorted(scan.subject for scan in report.eligible) == sorted(CLI_SUBJECTS)
    assert report.ineligible == {}


def test_pre_check_lists_a_pure_cli_subject_once_with_its_subject_folder(tmp_path):
    """A pure cli subject is one scan, named by its subject folder, whatever sessions it holds."""
    root = build_morphologist_cli_tree(Path(tmp_path))
    (root / "sub-01" / "ses-2" / "run-0" / "ana-0" / "folds" / "3.1").mkdir(parents=True)
    (root / "sub-01" / "ses-2" / "run-0" / "ana-0" / "folds" / "3.1" / "sub-01_ses-2_run-0_hemi-L.arg").touch()

    scans = SubjectEligibilityChecker(str(root), []).list_scans()

    assert [scan.subject for scan in scans] == sorted(CLI_SUBJECTS)
    assert {scan.acquisition for scan in scans} == {""}


def test_pre_check_lists_classic_and_pure_cli_subjects_of_a_mixed_tree(tmp_path):
    """Classic subjects keep one scan per t1mri acquisition; pure cli subjects join them as one scan each."""
    root = build_classic_tree(Path(tmp_path))
    build_morphologist_cli_tree(root)

    scans = SubjectEligibilityChecker(str(root), []).list_scans()

    expected = [(subject, "default_acquisition") for subject in CLASSIC_SUBJECTS] + [
        (subject, "") for subject in CLI_SUBJECTS
    ]
    assert sorted((scan.subject, scan.acquisition) for scan in scans) == sorted(expected)


def test_pre_check_leaves_out_a_folder_without_graph_or_t1mri(tmp_path):
    """A subject folder holding neither a t1mri acquisition nor a graph gets no scan."""
    root = build_morphologist_cli_tree(Path(tmp_path))
    (root / "sub-99").mkdir()
    (root / "sub-99" / "notes.txt").touch()

    scans = SubjectEligibilityChecker(str(root), []).list_scans()

    assert [scan.subject for scan in scans] == sorted(CLI_SUBJECTS)


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
