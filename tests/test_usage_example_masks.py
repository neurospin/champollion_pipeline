#!/usr/bin/env python3
"""Documentation guard: the Stage 2 to Stage 6 examples of docs/usage.md use one mask version.

TASK-DOCS-BDRABCZUK-F0DF7ECE3CB3. Stage 2 writes the crops under ``crops/<mask version>/2mm``; each later stage
reads them, or what was built from them, under the same mask version. An example command that names a mask version
(a ``--masks`` value, or a path segment that is a known mask version) must therefore use the mask version the first
Stage 2 example passes to ``--masks``. A stage whose parser declares ``--masks`` uses its ``--masks`` value, else
the parser's default: the Stage 3 example naming ``crops/canonical_25/2mm`` without ``--masks`` writes YAMLs that
point at the default mask version's crops.

Requirement pinned here: REQ-DOCS-BDRABCZUK-9BC683F45F19 (``test_examples_use_stage2_mask_version``, one case
per stage command of Stages 2 to 6).

Named outside ``tests/test_docs_*.py`` so ``verify_ignore`` keeps it in every full run. docs/usage.md is read as
text; the parsers are only built and inspected, never run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from champollion_pipeline.derivatives_layout import MASK_VERSIONS
from tests.docs_test_helpers import STAGE_COMMANDS, md_section, pixi_run_examples, read_text
from tests.test_usage_stage_sections import STAGE_BUILDERS

USAGE = Path(__file__).resolve().parents[1] / "docs" / "usage.md"
MASKS = "--masks"
CHECKED_STAGES = range(2, 7)


def _command(stage: int) -> str:
    return STAGE_COMMANDS[stage - 1]


def _examples(stage: int) -> list[list[str]]:
    """Argument tokens of every ``pixi run <stage command>`` example in the ``## Stage <stage>`` section."""
    section = md_section(read_text(USAGE), rf"Stage {stage}\b[^\n]*")
    return pixi_run_examples(section, _command(stage))


def _masks_value(tokens: list[str]) -> str | None:
    """The value given to ``--masks`` in ``tokens`` (``--masks V`` or ``--masks=V``), else None."""
    for i, tok in enumerate(tokens):
        if tok == MASKS and i + 1 < len(tokens):
            return tokens[i + 1]
        if tok.startswith(MASKS + "="):
            return tok.split("=", 1)[1]
    return None


def _path_versions(tokens: list[str]) -> set[str]:
    """Known mask versions found as path segments of the tokens."""
    return {seg for tok in tokens for seg in tok.split("/") if seg in MASK_VERSIONS}


def _masks_default(stage: int) -> tuple[bool, str | None]:
    """Whether the stage parser declares ``--masks``, and its default."""
    parser = STAGE_BUILDERS[_command(stage)]().parser
    for action in parser._actions:
        if MASKS in action.option_strings:
            return True, action.default
    return False, None


def _stage2_version() -> str:
    examples = _examples(2)
    assert examples, "docs/usage.md Stage 2 shows no 'pixi run champollion-cortical-tiles' example"
    version = _masks_value(examples[0])
    assert version, "the first Stage 2 example of docs/usage.md passes no --masks value"
    return version


@pytest.mark.smoke
@pytest.mark.parametrize("stage", CHECKED_STAGES, ids=_command)
def test_examples_use_stage2_mask_version(stage):
    expected = _stage2_version()
    examples = _examples(stage)
    assert examples, f"docs/usage.md Stage {stage} shows no 'pixi run {_command(stage)}' example"
    declares_masks, default = _masks_default(stage)
    wrong = []
    for tokens in examples:
        given = _masks_value(tokens)
        named = _path_versions(tokens) | ({given} if given else set())
        if not named:
            continue
        used = set(_path_versions(tokens))
        if declares_masks:
            used.add(given if given is not None else default)
        if used != {expected}:
            wrong.append(f"{' '.join(tokens)} -> uses {sorted(map(str, used))}")
    assert not wrong, (
        f"Stage {stage} examples of docs/usage.md must use mask version {expected!r} (the first Stage 2 example's "
        "--masks); add --masks " + expected + " where the parser default differs:\n  " + "\n  ".join(wrong)
    )
