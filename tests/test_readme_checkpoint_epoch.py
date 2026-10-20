#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Documentation guards for the configurable checkpoint epoch (epic WIPMERGE, task EPOCH80 docs).

Behaviour documented (read from the code, branch at 5339fbb):

* ``champollion-embeddings --checkpoint-epoch N`` (default 80, 0-based; pipeline config key
  ``dataset.checkpoint_epoch``) evaluates each region with the Lightning checkpoint
  ``logs/lightning_logs/version_*/checkpoints/epoch=N-step=*.ckpt`` (any ``version_*``, highest step first).
* Interim: a region without that checkpoint is evaluated with ``logs/best_model_weights.pt`` and the stage
  prints one WARNING line for that region.
* A region with neither file stops the stage before any region is evaluated.
* ``--use_last_checkpoint`` is removed (argparse exit status 2); the YAML keys
  ``dataset.use_last_checkpoint`` and ``dataset.use_best_model`` are ignored with a warning.

Requirements pinned here (one test per requirement):

* REQ-EPOCH80-BDRABCZUK-BA917603BA48 - section 5 option table has a ``--checkpoint-epoch`` row stating
  the default 80.
* REQ-EPOCH80-BDRABCZUK-853D6322D1E1 - that row names the config key ``dataset.checkpoint_epoch``.
* REQ-EPOCH80-BDRABCZUK-DCD70561ED8C - that row states the epoch number is 0-based.
* REQ-EPOCH80-BDRABCZUK-22931E2D3708 - the "Checkpoint selection" subsection states the checkpoint path
  pattern.
* REQ-EPOCH80-BDRABCZUK-3B430A47EEA9 - that subsection states the interim best-weights stand-in.
* REQ-EPOCH80-BDRABCZUK-AA625C7E72E8 - that subsection states the per-region stand-in WARNING.
* REQ-EPOCH80-BDRABCZUK-86A36F3BE8AB - that subsection states the fail-fast on a region without weights.
* REQ-EPOCH80-BDRABCZUK-89D3B534DF7D - that subsection states ``--use_last_checkpoint`` exits with status 2.
* REQ-EPOCH80-BDRABCZUK-A91067563B0C - that subsection states ``dataset.use_last_checkpoint`` is ignored
  with a warning.
* REQ-EPOCH80-BDRABCZUK-2FEF3081C059 - that subsection states ``dataset.use_best_model`` is ignored with a
  warning.
* REQ-EPOCH80-BDRABCZUK-FE248165AEE8 - docs/workflow.md Inference subsection names ``epoch=N-step=*.ckpt``.
* REQ-EPOCH80-BDRABCZUK-B5C16C56B1CF - that subsection names the ``logs/best_model_weights.pt`` stand-in.
* REQ-EPOCH80-BDRABCZUK-8F116839394A - that subsection names no ``best_model.pt``.

README.md and docs/workflow.md are read as plain text: no package import, no Sphinx build, no network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

README = PROJECT_ROOT / "README.md"
WORKFLOW = PROJECT_ROOT / "docs" / "workflow.md"

EMBEDDINGS_SECTION_HEADING = r"\d+\.\s+Generate Embeddings"
OPTION_NAME = "--checkpoint-epoch"
CKPT_PATH_PATTERN = "logs/lightning_logs/version_*/checkpoints/epoch=N-step=*.ckpt"
CKPT_FILE_PATTERN = "epoch=N-step=*.ckpt"
BEST_WEIGHTS = "logs/best_model_weights.pt"


def _read(path: Path) -> str:
    assert path.is_file(), f"expected file to exist: {path}"
    return path.read_text(encoding="utf-8")


def _section(text: str, heading_regex: str) -> str:
    """Body of the ``## <heading>`` section, up to the next level-2 heading."""
    m = re.search(rf"^##\s+{heading_regex}\s*$(.*?)(?=^##\s|\Z)", text, re.MULTILINE | re.DOTALL)
    assert m, f"no '## {heading_regex}' section found"
    return m.group(1)


def _subsection(text: str, heading_regex: str, where: str) -> str:
    """Body of the ``### <heading>`` subsection, up to the next level-2/3 heading or ``---`` rule."""
    m = re.search(
        rf"^###\s+{heading_regex}[^\n]*$(.*?)(?=^#{{2,3}}\s|^---\s*$|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert m, f"{where} has no '### {heading_regex}' subsection"
    return m.group(1)


def _paragraphs(text: str) -> list[str]:
    """Blocks of text separated by blank lines, whitespace-collapsed."""
    return [" ".join(block.split()) for block in re.split(r"\n\s*\n", text) if block.strip()]


def _embeddings_section() -> str:
    return _section(_read(README), EMBEDDINGS_SECTION_HEADING)


def _checkpoint_selection() -> str:
    return _subsection(_embeddings_section(), r"Checkpoint selection", "README.md section '5. Generate Embeddings'")


def _inference_subsection() -> str:
    return _subsection(_read(WORKFLOW), r"Inference\b", "docs/workflow.md")


def _cells(line: str) -> list[str]:
    """Cells of one Markdown table line, stripped of surrounding whitespace."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _option_table_rows(section: str) -> list[list[str]]:
    """Body rows of the table whose header row starts with ``Option``."""
    rows: list[list[str]] = []
    in_table = False
    for line in section.splitlines():
        if not line.lstrip().startswith("|"):
            in_table = False
            continue
        cells = _cells(line)
        if not in_table:
            in_table = cells[0] == "Option"
            continue
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
            continue  # separator row
        rows.append(cells)
    return rows


def _checkpoint_epoch_row_description() -> str:
    """Description cell of the ``--checkpoint-epoch`` row of the section 5 option table."""
    rows = _option_table_rows(_embeddings_section())
    assert rows, "README.md section '5. Generate Embeddings' has no table whose header row starts with 'Option'"
    matches = [r for r in rows if r[0].strip("`").split()[0:1] == [OPTION_NAME]]
    assert len(matches) == 1, (
        f"README.md section 5 option table must contain exactly one '{OPTION_NAME}' row; "
        f"option cells found: {[r[0] for r in rows]}"
    )
    return " | ".join(matches[0][1:])


def _paragraph_with(text: str, where: str, *patterns: str) -> str:
    """The first paragraph matching every regex in ``patterns`` (case-sensitive), else fail."""
    for para in _paragraphs(text):
        if all(re.search(p, para) for p in patterns):
            return para
    raise AssertionError(f"{where}: no paragraph matches all of {list(patterns)}")


SELECTION = "README.md 'Checkpoint selection' subsection"


@pytest.mark.smoke
class TestReadmeCheckpointEpochOptionRow:
    """REQ-EPOCH80-BDRABCZUK-BA917603BA48."""

    def test_option_row_states_default_80(self):
        description = _checkpoint_epoch_row_description()
        assert re.search(r"(?i)\bdefault\b", description) and re.search(r"(?<![\w.])80(?![\w.])", description), (
            f"README.md section 5 '{OPTION_NAME}' row must state the default value 80; description: {description!r}"
        )


@pytest.mark.smoke
class TestReadmeCheckpointEpochConfigKey:
    """REQ-EPOCH80-BDRABCZUK-853D6322D1E1."""

    def test_option_row_names_config_key(self):
        description = _checkpoint_epoch_row_description()
        assert re.search(r"(?<![\w.])dataset\.checkpoint_epoch(?!\w)", description), (
            f"README.md section 5 '{OPTION_NAME}' row must name the pipeline config key dataset.checkpoint_epoch; "
            f"description: {description!r}"
        )


@pytest.mark.smoke
class TestReadmeCheckpointEpochZeroBased:
    """REQ-EPOCH80-BDRABCZUK-DCD70561ED8C."""

    def test_option_row_states_zero_based(self):
        description = _checkpoint_epoch_row_description()
        assert re.search(r"(?i)\b(0|zero)-based\b", description), (
            f"README.md section 5 '{OPTION_NAME}' row must state that the epoch number is 0-based; "
            f"description: {description!r}"
        )


@pytest.mark.smoke
class TestReadmeCheckpointPathPattern:
    """REQ-EPOCH80-BDRABCZUK-22931E2D3708."""

    def test_subsection_states_checkpoint_path_pattern(self):
        assert CKPT_PATH_PATTERN in _checkpoint_selection(), (
            f"{SELECTION} must state the checkpoint path pattern {CKPT_PATH_PATTERN}"
        )


@pytest.mark.smoke
class TestReadmeBestWeightsStandIn:
    """REQ-EPOCH80-BDRABCZUK-3B430A47EEA9."""

    def test_subsection_states_interim_stand_in(self):
        _paragraph_with(
            _checkpoint_selection(),
            f"{SELECTION} must state that, as an interim measure, {BEST_WEIGHTS} stands in for a region "
            "lacking the epoch-N checkpoint",
            re.escape(BEST_WEIGHTS),
            r"(?i)\binterim\b",
            r"(?i)\bstands? in\b",
        )


@pytest.mark.smoke
class TestReadmeStandInWarning:
    """REQ-EPOCH80-BDRABCZUK-AA625C7E72E8."""

    def test_subsection_states_stand_in_warning(self):
        _paragraph_with(
            _checkpoint_selection(),
            f"{SELECTION} must state that a WARNING line is printed for each region evaluated with {BEST_WEIGHTS}",
            r"\bWARNING\b",
            re.escape("best_model_weights.pt"),
            r"(?i)\b(each|per) region\b",
        )


@pytest.mark.smoke
class TestReadmeFailFast:
    """REQ-EPOCH80-BDRABCZUK-86A36F3BE8AB."""

    def test_subsection_states_fail_fast(self):
        _paragraph_with(
            _checkpoint_selection(),
            f"{SELECTION} must state that a region holding neither {BEST_WEIGHTS} nor the epoch-N checkpoint "
            "stops the embeddings stage before any region is evaluated",
            r"(?i)\bneither\b",
            re.escape("best_model_weights.pt"),
            r"(?i)\bstops?\b",
            r"(?i)\bbefore any region is evaluated\b",
        )


@pytest.mark.smoke
class TestReadmeRemovedOption:
    """REQ-EPOCH80-BDRABCZUK-89D3B534DF7D."""

    def test_subsection_states_removed_option_exit_2(self):
        _paragraph_with(
            _checkpoint_selection(),
            f"{SELECTION} must state that champollion-embeddings exits with status 2 when given the removed "
            "--use_last_checkpoint option",
            r"(?<![\w-])--use_last_checkpoint(?!\w)",
            r"(?i)\bremoved\b",
            r"(?i)\bstatus 2\b",
        )


@pytest.mark.smoke
class TestReadmeRemovedConfigKeys:
    """REQ-EPOCH80-BDRABCZUK-A91067563B0C, REQ-EPOCH80-BDRABCZUK-2FEF3081C059 (one test each)."""

    def test_subsection_states_use_last_checkpoint_key_ignored(self):
        _paragraph_with(
            _checkpoint_selection(),
            f"{SELECTION} must state that the config key dataset.use_last_checkpoint is ignored with a warning",
            r"(?<![\w.])dataset\.use_last_checkpoint(?!\w)",
            r"(?i)\bignored\b",
            r"(?i)\bwarning\b",
        )

    def test_subsection_states_use_best_model_key_ignored(self):
        _paragraph_with(
            _checkpoint_selection(),
            f"{SELECTION} must state that the config key dataset.use_best_model is ignored with a warning",
            r"(?<![\w.])dataset\.use_best_model(?!\w)",
            r"(?i)\bignored\b",
            r"(?i)\bwarning\b",
        )


@pytest.mark.smoke
class TestWorkflowInferenceCheckpoint:
    """REQ-EPOCH80-BDRABCZUK-FE248165AEE8, -B5C16C56B1CF, -8F116839394A (one test each)."""

    def test_inference_names_epoch_checkpoint_pattern(self):
        assert CKPT_FILE_PATTERN in _inference_subsection(), (
            f"docs/workflow.md Inference subsection must name the Lightning checkpoint pattern {CKPT_FILE_PATTERN}"
        )

    def test_inference_names_best_weights_stand_in(self):
        _paragraph_with(
            _inference_subsection(),
            f"docs/workflow.md Inference subsection must name {BEST_WEIGHTS} as the stand-in weights",
            re.escape(BEST_WEIGHTS),
            r"(?i)\bstand(s? in|-in)\b",
        )

    def test_inference_names_no_best_model_pt(self):
        subsection = _inference_subsection()
        assert not re.search(r"(?<![\w.])best_model\.pt\b", subsection), (
            "docs/workflow.md Inference subsection must name no file best_model.pt (inference evaluates the "
            f"epoch-N checkpoint); subsection: {subsection!r}"
        )
