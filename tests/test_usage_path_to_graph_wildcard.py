#!/usr/bin/env python3
"""Documentation guards: docs/usage.md Stage 2 states how --path_to_graph must resolve per subject.

TASK-DOCS-BDRABCZUK-556231937915 (user decision 2026-11-23, docs epic follow-up item 2): without --bids,
--path_to_graph, read with '*' as a wildcard, must match exactly one folder per subject; a subject with none
or two or more gets qc 0 in labelling_qc.tsv and is left out of the whole-brain output with a warning, which
does not change the stage's exit status; with --bids the folder is the one whose session labels give the
volume id.

Each Requirement has a docs-text test (the Stage 2 section says it) and a behaviour test that keeps the
sentence true (run_cortical_tiles.py does it, on a tmp_path tree: no real data, no network):

* REQ-DOCS-BDRABCZUK-D8169BD03FF1 - exactly one folder without --bids (``test_states_exactly_one_folder_without_bids``,
  ``test_resolve_graph_paths_counts_matching_folders``).
* REQ-DOCS-BDRABCZUK-4AB69E522F11 - qc 0 in labelling_qc.tsv for a subject with a fused volume
  (``test_states_qc_zero_in_labelling_qc``, ``test_labelling_qc_row_needs_one_folder``,
  ``test_labelling_qc_rows_only_for_fused_subjects``).
* REQ-DOCS-BDRABCZUK-BB7B4A957218 - left out of the whole-brain output (``test_states_left_out_of_whole_brain``,
  ``test_whole_brain_leaves_out_unresolved_subjects``).
* REQ-DOCS-BDRABCZUK-DEADC0F0F102 (``[no_folder]``), REQ-DOCS-BDRABCZUK-57EC3D408814 (``[several_folders]``) -
  the quoted warning texts (``test_quotes_warning_text``, ``test_run_warning_holds_quoted_text``).
* REQ-DOCS-BDRABCZUK-E42C1DA9D0A9 - the exit status is unchanged (``test_states_exit_status_unchanged``,
  ``test_whole_brain_exclusion_keeps_stage_status``).
* REQ-DOCS-BDRABCZUK-578EBCBDA59A - the --bids match (``test_states_bids_session_match``,
  ``test_resolve_bids_graph_paths_matches_volume_id``).
* REQ-DOCS-BDRABCZUK-63EE4E80BBCD - the "Whole-brain volume" subsection: leaving out a subject with no labelled
  graph does not change the exit status (``test_whole_brain_volume_states_exit_status_unchanged``,
  ``test_no_labelled_graph_exclusion_keeps_stage_status``).

Named outside ``tests/test_docs_*.py`` so ``verify_ignore`` keeps it in every full run.
"""

from __future__ import annotations

import csv
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.run_cortical_tiles import (
    CROP_STATUS_PARTIAL,
    RunCorticalTiles,
    fetch_labelling_qc_row,
    resolve_bids_graph_paths,
    resolve_graph_paths,
)
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER, LABELLING_SESSION_CANDIDATES
from tests.docs_test_helpers import md_prose_lines, md_section, read_text

ROOT = Path(__file__).resolve().parents[1]
USAGE = ROOT / "docs" / "usage.md"
STAGE_2_HEADING = r"Stage 2 — Extract Sulcal Region Crops"
WHOLE_BRAIN_HEADING = "Whole-brain volume"

MODULE = "champollion_pipeline.run_cortical_tiles"
PATTERN = "graph_*"  # --path_to_graph of the tmp tree
MATCHING = {"s0": (), "s1": ("graph_a",), "s2": ("graph_a", "graph_b")}  # subject -> folders PATTERN matches
SESSION = LABELLING_SESSION_CANDIDATES[0]
NO_FOLDER_TEXT = "matches no folder"
SEVERAL_TEXT = "ambiguous, not guessed"
WITHOUT_BIDS = re.compile(r"\bwithout --bids\b", re.IGNORECASE)
WITH_BIDS = re.compile(r"(?<![\w-])with --bids\b", re.IGNORECASE)
NOT_ONE_FOLDER = re.compile(r"exactly one folder|no folder|several folders|two or more folders", re.IGNORECASE)
QC_ZERO = re.compile(r"\bqc\s*(?:=\s*)?0\b")
LEFT_OUT = re.compile(r"left out|excluded|leaves out", re.IGNORECASE)
FUSED = re.compile(r"\bfused (?:whole-brain )?volume", re.IGNORECASE)
UNCHANGED = re.compile(r"(?:does not|doesn't|do not|never) change|unchanged|stays the same", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Stage 2 section text
# ---------------------------------------------------------------------------


def _paragraphs(subsection: str | None = None) -> list[str]:
    """Prose paragraphs of the Stage 2 section (subsections included), or of its ``### <subsection>`` only,
    backticks dropped, whitespace collapsed."""
    section = md_section(read_text(USAGE), STAGE_2_HEADING)
    if subsection is not None:
        found = re.search(rf"^###\s+{re.escape(subsection)}$(.*?)(?=^###?\s|\Z)", section, re.MULTILINE | re.DOTALL)
        assert found, f"docs/usage.md Stage 2: no '### {subsection}' subsection"
        section = found.group(1)
    text = "\n".join(md_prose_lines(section)).replace("`", "")
    return [" ".join(block.split()) for block in re.split(r"\n\s*\n", text) if block.strip()]


def _assert_paragraph(what: str, *checks, subsection: str | None = None) -> None:
    """Assert one Stage 2 paragraph (of ``subsection`` when given) passes every check (a str to contain, or a
    compiled regex to search)."""

    def passes(paragraph: str) -> bool:
        return all(c.search(paragraph) if isinstance(c, re.Pattern) else c in paragraph for c in checks)

    where = f"Stage 2 '{subsection}'" if subsection else "Stage 2"
    assert any(passes(p) for p in _paragraphs(subsection)), f"docs/usage.md {where}: no paragraph {what}"


def test_states_exactly_one_folder_without_bids():
    _assert_paragraph(
        "states that, without --bids, --path_to_graph must match exactly one folder per subject",
        WITHOUT_BIDS,
        "--path_to_graph",
        "exactly one folder",
    )


def test_states_qc_zero_in_labelling_qc():
    _assert_paragraph(
        "states that a subject with a fused whole-brain volume whose --path_to_graph does not match exactly one "
        "folder gets qc 0 in labelling_qc.tsv",
        "--path_to_graph",
        NOT_ONE_FOLDER,
        "labelling_qc.tsv",
        FUSED,
        QC_ZERO,
    )


def test_states_left_out_of_whole_brain():
    _assert_paragraph(
        "states that the whole-brain output leaves out a subject whose --path_to_graph matches not exactly one folder",
        "--path_to_graph",
        NOT_ONE_FOLDER,
        "whole-brain",
        LEFT_OUT,
    )


@pytest.mark.parametrize("text", [NO_FOLDER_TEXT, SEVERAL_TEXT], ids=["no_folder", "several_folders"])
def test_quotes_warning_text(text):
    _assert_paragraph(f"quotes the warning text {text!r}", "--path_to_graph", text)


def test_states_exit_status_unchanged():
    _assert_paragraph(
        "states that leaving a subject out of the whole-brain output for a --path_to_graph match keeps the exit status",
        "--path_to_graph",
        "whole-brain",
        "exit status",
        UNCHANGED,
    )


def test_whole_brain_volume_states_exit_status_unchanged():
    _assert_paragraph(
        "states that leaving out a subject with no labelled graph does not change the exit status of Stage 2",
        re.compile(r"no labelled graph", re.IGNORECASE),
        LEFT_OUT,
        "exit status",
        UNCHANGED,
        subsection=WHOLE_BRAIN_HEADING,
    )


def test_states_bids_session_match():
    _assert_paragraph(
        "states which folder --path_to_graph must match with --bids (ses-, acq- and run- labels give the volume id)",
        WITH_BIDS,
        "--path_to_graph",
        "exactly one folder",
        re.compile(r"volume id", re.IGNORECASE),
        "ses-",
    )


# ---------------------------------------------------------------------------
# Behaviour the text describes (tmp_path trees only)
# ---------------------------------------------------------------------------


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


@pytest.fixture
def graphs(tmp_path: Path) -> Path:
    """Morphologist-like input: s0 matches no folder, s1 one (labelled for SESSION), s2 two."""
    input_dir = tmp_path / "input"
    for subject, folders in MATCHING.items():
        (input_dir / subject / "other").mkdir(parents=True)
        for folder in folders:
            session_dir = input_dir / subject / folder / SESSION
            session_dir.mkdir(parents=True)
            for side in ("L", "R"):
                _touch(session_dir / f"{side}{subject}_{SESSION}.arg")
    return input_dir


@pytest.mark.parametrize("subject", sorted(MATCHING))
def test_resolve_graph_paths_counts_matching_folders(graphs, subject):
    assert resolve_graph_paths(str(graphs), PATTERN, subject) == list(MATCHING[subject])


@pytest.mark.parametrize(("subject", "qc"), [("s0", "0"), ("s1", "1"), ("s2", "0")])
def test_labelling_qc_row_needs_one_folder(graphs, subject, qc):
    row = fetch_labelling_qc_row(str(graphs), PATTERN, subject, LABELLING_SESSION_CANDIDATES, False)
    assert row["qc"] == qc, row


@dataclass
class WholeBrainRun:
    """What _run_whole_brain did with fusion and remove_ventricle mocked."""

    status: int
    staged: list[list[str]]  # subjects of the F/ folder each remove_ventricle call read
    warnings: list[str]


def _run_whole_brain(
    input_dir: Path, output_dir: Path, caplog, fused: list[str], crop_status: int = 0
) -> WholeBrainRun:
    """Run the whole-brain step after a crop of ``crop_status``, with a fused volume for each of ``fused``."""
    fused_dir = output_dir / DERIVATIVES_FOLDER / "skeletons" / "2mm" / "F"
    for subject in fused:
        _touch(fused_dir / f"Fresampled_skeleton_{subject}.nii.gz")
    script = RunCorticalTiles()
    script.parse_args([str(input_dir), str(output_dir), "--path_to_graph", PATTERN, "--path_sk_with_hull", "skeleton"])
    staged: list[list[str]] = []

    def remove_ventricle(**kwargs):
        names = sorted(os.listdir(Path(kwargs["src_dir"], "F")))
        staged.append([name.removeprefix("Fresampled_skeleton_").removesuffix(".nii.gz") for name in names])
        for subject in staged[-1]:
            _touch(Path(kwargs["output_dir"], "F", f"Fskeleton_generated_without_ventricle_{subject}.nii.gz"))

    with (
        patch.object(script, "_fuse_whole_brain", return_value=True),
        patch(f"{MODULE}.remove_ventricle.remove_ventricle", side_effect=remove_ventricle),
    ):
        status = script._run_whole_brain(str(input_dir), str(output_dir), crop_status)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    return WholeBrainRun(status=status, staged=staged, warnings=warnings)


@pytest.fixture
def whole_brain_run(graphs, tmp_path, pipeline_caplog) -> WholeBrainRun:
    """Run the whole-brain step of a complete crop (status 0) on the three subjects of ``graphs``."""
    return _run_whole_brain(graphs, tmp_path / "output", pipeline_caplog, list(MATCHING))


def test_labelling_qc_rows_only_for_fused_subjects(graphs, tmp_path, pipeline_caplog):
    output_dir = tmp_path / "output"
    _run_whole_brain(graphs, output_dir, pipeline_caplog, ["s1", "s2"])
    with open(output_dir / DERIVATIVES_FOLDER / "labelling_qc.tsv", newline="") as qc_file:
        rows = {row["participant_id"]: row["qc"] for row in csv.DictReader(qc_file, delimiter="\t")}
    assert rows == {"s1": "1", "s2": "0"}


def test_no_labelled_graph_exclusion_keeps_stage_status(graphs, tmp_path, pipeline_caplog):
    """s3 matches one folder that holds no labelling session: left out, the partial crop status (3) stands."""
    (graphs / "s3" / "graph_a").mkdir(parents=True)
    run = _run_whole_brain(graphs, tmp_path / "output", pipeline_caplog, ["s1", "s3"], CROP_STATUS_PARTIAL)
    assert run.staged == [["s1"]]
    assert any("no labelled graph" in w and "s3" in w for w in run.warnings), run.warnings
    assert run.status == CROP_STATUS_PARTIAL


def test_whole_brain_leaves_out_unresolved_subjects(whole_brain_run):
    assert whole_brain_run.staged == [["s1"]]


@pytest.mark.parametrize(
    ("text", "subject"), [(NO_FOLDER_TEXT, "s0"), (SEVERAL_TEXT, "s2")], ids=["no_folder", "several_folders"]
)
def test_run_warning_holds_quoted_text(whole_brain_run, text, subject):
    assert any(text in w and subject in w for w in whole_brain_run.warnings), whole_brain_run.warnings


def test_whole_brain_exclusion_keeps_stage_status(whole_brain_run):
    assert whole_brain_run.status == 0


def test_resolve_bids_graph_paths_matches_volume_id(tmp_path):
    for session in ("ses-1", "ses-2"):
        (tmp_path / "sub-01" / session / "anat" / "graph").mkdir(parents=True)
    assert resolve_bids_graph_paths(str(tmp_path), "*/anat/graph", "sub-01_ses-1") == ["ses-1/anat/graph"]
    assert resolve_bids_graph_paths(str(tmp_path), "*/anat/graph", "sub-01_ses-3") == []
