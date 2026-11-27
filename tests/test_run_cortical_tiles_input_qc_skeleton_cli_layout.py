#!/usr/bin/env python3
"""Stage 2 input QC finds the skeletons of a morphologist-cli tree (TASK-ARGDISC-BDRABCZUK-04D084972C8C).

A morphologist-cli hemisphere skeleton is <subject>_ses-1_run-0_hemi-<S>_skeleton.nii.gz in the subject folder's own
tree (ses-1/run-0/ana-0/segmentation here), with no t1mri folder, so path_sk_with_hull does not reach it. The skeleton
belongs to the subject folder graph discovery lists the graph under: no second subject-folder rule. Layouts are the
names-only trees of tests/morphologist_layout_fixtures.py; the input QC runs through fetch_input_qc_rows.

The Requirement ids are listed by the red report of the Task.
"""

import logging
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline import run_cortical_tiles_inputs as inputs
from champollion_pipeline.config import PipelineConfig
from champollion_pipeline.file_indexer.pipeline_checks import FileIndexStageIndexer, SubjectEligibilityChecker
from champollion_pipeline.pipeline_stages import RunCorticalTilesStage
from champollion_pipeline.run_cortical_tiles_graphs import list_subject_graphs
from tests.morphologist_layout_fixtures import CLASSIC_SUBJECTS, build_classic_tree, build_morphologist_cli_tree
from tests.stage2_skeletons import (  # noqa: F401 - the fixture overrides the conftest stub in this file
    _stub_stage2_skeletons,
    skeleton_volume,
)

CLASSIC_SK_FOLDER = "t1mri/default_acquisition/default_analysis/segmentation"
CLI_SK_FOLDER = "ses-1/run-0/ana-0/segmentation"
CLI_SK_STEM = "{subject}_ses-1_run-0_hemi-{side}_skeleton.nii.gz"
SUBJECT = "sub-01"
OTHER_SUBJECT = "sub-02"


def _run_qc(root: Path, path_sk_with_hull: str = CLASSIC_SK_FOLDER, empty: tuple[str, ...] = ()) -> dict[str, tuple]:
    """Input QC rows of ``root`` as {subject: (qc, reason)}; the files named in ``empty`` hold no voxel."""

    def read(path: str, *_args, **_kwargs):
        return skeleton_volume(0) if Path(path).name in empty else skeleton_volume()

    listing = list_subject_graphs(str(root), None, FileIndexStageIndexer("").discover_graphs)
    with patch.object(inputs.aims, "read", side_effect=read):
        rows = inputs.fetch_input_qc_rows(str(root), None, None, False, False, path_sk_with_hull, listing)
    return {row["participant_id"]: (row["qc"], row["reason"]) for row in rows}


def _cli_skeleton(subject: str, side: str) -> Path:
    return Path(subject) / CLI_SK_FOLDER / CLI_SK_STEM.format(subject=subject, side=side)


@pytest.fixture
def cli_tree(tmp_path) -> Path:
    return build_morphologist_cli_tree(tmp_path)


def test_cli_skeletons_count_as_found(cli_tree):
    """Both hemispheres of both subjects have a skeleton named hemi-<S>_skeleton: no missing_skeleton reason."""
    rows = _run_qc(cli_tree)

    assert rows[SUBJECT] == (1, "")
    assert rows[OTHER_SUBJECT] == (1, "")


@pytest.mark.parametrize("side", "LR")
def test_absent_cli_skeleton_gets_its_reason(cli_tree, side):
    """Other hemisphere and other hemi-<S> volumes (cortex, roots) do not stand in for the skeleton."""
    (cli_tree / _cli_skeleton(OTHER_SUBJECT, side)).unlink()

    rows = _run_qc(cli_tree)

    assert rows[OTHER_SUBJECT] == (0, f"missing_skeleton_{side}")
    assert rows[SUBJECT] == (1, "")


def test_empty_cli_skeleton_is_unusable(cli_tree):
    name = CLI_SK_STEM.format(subject=SUBJECT, side="L")

    rows = _run_qc(cli_tree, empty=(name,))

    assert rows[SUBJECT] == (0, "unusable_skeleton_L")


def test_skeleton_of_another_subject_folder_does_not_count(cli_tree):
    """The skeleton must sit below the subject folder of the graph: a stray one in a sibling folder is not it."""
    (cli_tree / _cli_skeleton(OTHER_SUBJECT, "R")).unlink()
    stray = cli_tree / _cli_skeleton(SUBJECT, "R")
    moved = cli_tree / SUBJECT / CLI_SK_FOLDER / CLI_SK_STEM.format(subject=OTHER_SUBJECT, side="R")
    moved.write_bytes(stray.read_bytes())

    rows = _run_qc(cli_tree)

    assert rows[OTHER_SUBJECT] == (0, "missing_skeleton_R")


def test_classic_skeletons_still_count(tmp_path):
    """The classic <S>skeleton_<subject>.nii.gz under path_sk_with_hull counts as before."""
    root = build_classic_tree(tmp_path)

    rows = _run_qc(root)

    assert set(rows.values()) == {(1, "")}


def test_classic_skeleton_outside_path_sk_with_hull_still_missing(tmp_path):
    """A classic name is found through path_sk_with_hull only: a wrong folder keeps both skeletons missing."""
    root = build_classic_tree(tmp_path)

    rows = _run_qc(root, path_sk_with_hull="t1mri/default_acquisition/default_analysis/nowhere")

    assert set(rows.values()) == {(0, "missing_skeleton_L;missing_skeleton_R")}


# ====== the pre-check of main.py takes both namings by the same rule as the input QC (user decision 2026-11-26) ======
# The pre-check judges the scans <subject>/t1mri/<acquisition> (SubjectEligibilityChecker): a morphologist-cli tree
# with no t1mri folder lists none, so these trees are classic subjects whose skeletons carry the cli names.

CLASSIC_SUBJECT, OTHER_CLASSIC_SUBJECT = CLASSIC_SUBJECTS


def _classic_with_cli_skeletons(root: Path, subject: str, cli_sides: str = "LR", classic_sides: str = "") -> None:
    """In the classic tree ``root``, ``subject`` keeps the classic skeleton of ``classic_sides`` and gains the cli-named
    one of ``cli_sides`` below its own ses-1/run-0/ana-0/segmentation folder."""
    segmentation = root / subject / CLASSIC_SK_FOLDER
    for side in "LR":
        if side not in classic_sides:
            (segmentation / f"{side}skeleton_{subject}.nii.gz").unlink()
    cli_folder = root / subject / CLI_SK_FOLDER
    cli_folder.mkdir(parents=True)
    for side in cli_sides:
        (cli_folder / CLI_SK_STEM.format(subject=subject, side=side)).touch()


def _pre_check(root: Path, path_sk_with_hull: str = CLASSIC_SK_FOLDER) -> dict[str, list[str]]:
    """Ineligible subjects of main.py's stage 2 pre-check on ``root``, {subject: missing patterns}."""
    config = PipelineConfig(root_path="/r", data_path="/r/data", models_path="/r/models", outputs_path="/r/outputs")
    config.dataset.path_to_graph = None  # noqa: V101 - graphs are found by stage 2's input QC
    config.dataset.path_sk_with_hull = path_sk_with_hull  # noqa: V101
    stage = RunCorticalTilesStage("run_cortical_tiles", config, logging.getLogger("pre_check_cli_layout"))
    report = SubjectEligibilityChecker(str(root), stage.required_file_patterns()).check_subject_patterns()
    assert {scan.subject for scan in [*report.eligible, *report.ineligible]} == set(CLASSIC_SUBJECTS)
    return {scan.subject: patterns for scan, patterns in report.ineligible.items()}


@pytest.fixture
def classic_tree(tmp_path) -> Path:
    return build_classic_tree(tmp_path)


def test_pre_check_accepts_cli_skeletons(classic_tree):
    """REQ-ARGDISC-BDRABCZUK-172988E4F038: a cli-named skeleton below the subject folder counts."""
    _classic_with_cli_skeletons(classic_tree, CLASSIC_SUBJECT)

    assert set(_pre_check(classic_tree)) == set()


def test_pre_check_accepts_classic_skeletons(classic_tree):
    """The classic naming under path_sk_with_hull passes the pre-check as before."""
    assert _pre_check(classic_tree) == {}


def test_pre_check_accepts_one_hemisphere_of_each_naming(classic_tree):
    """The two namings are alternatives per hemisphere: classic L with cli R is complete."""
    _classic_with_cli_skeletons(classic_tree, CLASSIC_SUBJECT, cli_sides="R", classic_sides="L")

    assert _pre_check(classic_tree) == {}


@pytest.mark.parametrize("side", "LR")
def test_pre_check_names_an_absent_skeleton(classic_tree, side):
    """Only the subject without its hemisphere skeleton, in either naming, is ineligible; one pattern names it."""
    other = "R" if side == "L" else "L"
    _classic_with_cli_skeletons(classic_tree, CLASSIC_SUBJECT, cli_sides=other)

    missing = _pre_check(classic_tree)

    assert set(missing) == {CLASSIC_SUBJECT}
    assert len(missing[CLASSIC_SUBJECT]) == 1
    assert f"{side}skeleton" in missing[CLASSIC_SUBJECT][0]


def test_pre_check_ignores_a_skeleton_of_another_subject_folder(classic_tree):
    """Same rule as the input QC: a skeleton file below a sibling subject folder does not count."""
    _classic_with_cli_skeletons(classic_tree, CLASSIC_SUBJECT, cli_sides="L")
    stray_folder = classic_tree / OTHER_CLASSIC_SUBJECT / CLI_SK_FOLDER
    stray_folder.mkdir(parents=True)
    (stray_folder / CLI_SK_STEM.format(subject=CLASSIC_SUBJECT, side="R")).touch()

    assert set(_pre_check(classic_tree)) == {CLASSIC_SUBJECT}


def _missing_sides(patterns: list[str]) -> set[str]:
    """Hemispheres named by the missing patterns of the pre-check, whichever naming a pattern uses."""
    return {
        match[0] or match[1]
        for pattern in patterns
        for match in re.findall(r"/([LR])skeleton|hemi-([LR])_skeleton", pattern)
    }


# REQ-ARGDISC-BDRABCZUK-EC97DFB15852: the pre-check and the input QC name the same subjects and hemispheres as lacking.
@pytest.mark.parametrize("side", "LR")
def test_pre_check_and_input_qc_agree_on_cli_names(classic_tree, side):
    present = "R" if side == "L" else "L"
    _classic_with_cli_skeletons(classic_tree, CLASSIC_SUBJECT, cli_sides=present)

    qc = _run_qc(classic_tree)
    missing = _pre_check(classic_tree)

    assert set(missing) == {name for name, (qc_value, _) in qc.items() if qc_value == 0} == {CLASSIC_SUBJECT}
    assert _missing_sides(missing[CLASSIC_SUBJECT]) == {side}
    assert qc[CLASSIC_SUBJECT][1] == f"missing_skeleton_{side}"


def test_pre_check_and_input_qc_agree_on_a_wrong_classic_folder(classic_tree):
    """A classic name outside path_sk_with_hull is absent for both."""
    wrong = "t1mri/default_acquisition/default_analysis/nowhere"

    qc_missing = {name for name, (qc, _) in _run_qc(classic_tree, path_sk_with_hull=wrong).items() if qc == 0}

    assert set(_pre_check(classic_tree, path_sk_with_hull=wrong)) == qc_missing == set(CLASSIC_SUBJECTS)
