#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for REQ-CONTRIB-01 — CONTRIBUTING.md at the repository root.

Run with: pixi run test-specific tests/test_contributing_doc.py

REQ-CONTRIB-01 states that ``CONTRIBUTING.md`` at the repository root shall
document the ``pixi run`` tasks ``test``, ``lint``, ``lint-fix``, and
``format``; the commit conventions (no AI attribution, one logical change per
commit, imperative mood); and that the ``external/champollion_V1`` and
``external/cortical_tiles`` submodules are never edited directly, with fixes
going upstream.

REQ-CONTRIB-02 supersedes the "no AI attribution" clause of REQ-CONTRIB-01:
``CONTRIBUTING.md`` shall not prescribe any AI-attribution practice in commit
messages, since that rule is an internal team norm (workspace ``CLAUDE.md``),
not an obligation on external contributors.

Each test pins down one distinct acquired behaviour, so a partial document
reads as a partial failure rather than an all-or-nothing one. Keyword checks
are case-insensitive and accept a few phrasings, so the requirement constrains
the content, not the exact prose.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRIBUTING_MD = PROJECT_ROOT / "CONTRIBUTING.md"

PIXI_COMMANDS = ("pixi run test", "pixi run lint", "pixi run lint-fix", "pixi run format")
SUBMODULE_PATHS = ("external/champollion_V1", "external/cortical_tiles")


def _read_contributing_md() -> str:
    """Contents of ``CONTRIBUTING.md``, or ``""`` when it is missing.

    Returning empty rather than raising keeps each test's own assertion as the
    reported failure, instead of an unrelated ``FileNotFoundError``.
    """
    return CONTRIBUTING_MD.read_text(encoding="utf-8") if CONTRIBUTING_MD.is_file() else ""


def _mentions_any(patterns: tuple[str, ...]) -> bool:
    text = _read_contributing_md()
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


@pytest.mark.smoke
class TestContributingDoc:
    """REQ-CONTRIB-01: CONTRIBUTING.md documents workflow, conventions, submodule boundary."""

    def test_contributing_md_exists_at_repo_root(self):
        assert CONTRIBUTING_MD.is_file(), f"{CONTRIBUTING_MD} does not exist."

    @pytest.mark.parametrize("command", PIXI_COMMANDS)
    def test_documents_pixi_dev_command(self, command):
        # Word boundary after the command so "pixi run lint" is not satisfied
        # solely by "pixi run lint-fix".
        pattern = re.escape(command) + r"(?![\w-])"
        assert re.search(pattern, _read_contributing_md()), f"{CONTRIBUTING_MD} does not mention '{command}'."

    def test_does_not_prescribe_ai_attribution_practice(self):
        # REQ-CONTRIB-02 (supersedes REQ-CONTRIB-01's "no AI attribution"
        # clause): the no-AI-attribution rule is an internal team norm kept in
        # the workspace CLAUDE.md, not an obligation on external contributors,
        # who choose their own AI-attribution practice. A plain human
        # "Co-Authored-By" mention is not an AI-attribution statement, so only
        # AI-related trailers/footers are matched.
        assert not _mentions_any(
            (
                r"\bAI\b[^\n]*attribution",
                r"attribution[^\n]*\bAI\b",
                r"Co-Authored-By[^\n]*(\bAI\b|Claude|Anthropic|Copilot|GPT)",
                r"Generated with[^\n]*(\bAI\b|Claude|Copilot|GPT)",
            )
        ), (
            f"{CONTRIBUTING_MD} prescribes an AI-attribution commit practice; per REQ-CONTRIB-02 "
            "that convention is internal-only and must not be imposed on external contributors."
        )

    def test_documents_one_logical_change_per_commit_convention(self):
        assert _mentions_any((r"one logical change",)), (
            f"{CONTRIBUTING_MD} does not state the one-logical-change-per-commit convention."
        )

    def test_documents_imperative_mood_convention(self):
        assert _mentions_any((r"imperative",)), (
            f"{CONTRIBUTING_MD} does not state the imperative-mood commit-message convention."
        )

    @pytest.mark.parametrize("submodule", SUBMODULE_PATHS)
    def test_mentions_submodule_path(self, submodule):
        assert submodule in _read_contributing_md(), f"{CONTRIBUTING_MD} does not mention '{submodule}'."

    def test_states_submodules_must_not_be_edited_directly(self):
        assert _mentions_any(
            (
                r"never\s+(be\s+)?edit(ed)?[^\n]*directly",
                r"(do\s+not|don't|must\s+not|should\s+not)\s+(be\s+)?edit(ed)?[^\n]*directly",
            )
        ), f"{CONTRIBUTING_MD} does not state that the submodules must never be edited directly."

    def test_states_fixes_go_upstream(self):
        assert _mentions_any((r"upstream",)), f"{CONTRIBUTING_MD} does not state that submodule fixes belong upstream."
