#!/usr/bin/env python3
"""Documentation guards: README.md keeps one short stage table, the per-stage detail lives in docs/usage.md.

TASK-DOCS-BDRABCZUK-84BE8AB6FC73 (user decision 2026-11-23, 'Move to usage.md, short table'): README sections
2 to 7 (Morphologist graphs, sulcal region crops, Champollion configuration, embeddings, combine, snapshots)
move into the Stage sections of docs/usage.md; README.md keeps one table whose rows link to them.

Requirements pinned here (one test each):

* REQ-DOCS-BDRABCZUK-25B28869C99D - one table, six rows naming the stage console scripts in order, each with a
  link to docs/usage.md.
* REQ-DOCS-BDRABCZUK-91EDE22AA9A9 - each row links the usage.md level-2 section showing its console script.
* REQ-DOCS-BDRABCZUK-696EC8E1E982 - no level-2 heading of README.md names a stage.
* REQ-DOCS-BDRABCZUK-EB6EFE9020B9 - at most one pixi run command line per stage console script.
* REQ-DOCS-BDRABCZUK-98E8342B8297 - no README.md level-3 heading is also a heading of a usage.md Stage section.

Named outside ``tests/test_docs_*.py`` so ``verify_ignore`` keeps it in every full run. Plain text only:
no package import, no network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.docs_test_helpers import STAGE_COMMANDS, pixi_run_examples, read_text, strip_comment

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
USAGE = ROOT / "docs" / "usage.md"

STAGE_TITLES = (
    "Generate Morphologist Graphs",
    "Generate Sulcal Region Crops",
    "Generate Champollion Configuration",
    "Generate Embeddings",
    "Combine Embeddings",
    "Generate Visualization Snapshots",
)
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
SEPARATOR_CELL_RE = re.compile(r":?-{3,}:?")


def _split_fences(text: str) -> tuple[list[str], list[str]]:
    """(lines outside fenced code blocks, lines inside them); the fence lines themselves are dropped."""
    outside: list[str] = []
    inside: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        m = FENCE_RE.match(line)
        if m and fence is None:
            fence = m.group(1)[0]
            continue
        if m and fence is not None and m.group(1)[0] == fence:
            fence = None
            continue
        (outside if fence is None else inside).append(line)
    return outside, inside


def _outside_fences(text: str) -> list[str]:
    """Lines of ``text`` that are not inside a fenced code block."""
    return _split_fences(text)[0]


def _code_lines(text: str) -> list[str]:
    """Lines of ``text`` inside fenced code blocks, shell comments dropped."""
    return [strip_comment(line) for line in _split_fences(text)[1]]


def _headings(text: str, level: int) -> list[str]:
    """Titles of the headings of exactly ``level`` outside code fences, backticks removed."""
    found = []
    for line in _outside_fences(text):
        m = re.match(rf"^#{{{level}}}\s+(.*?)\s*#*\s*$", line)
        if m and not line.startswith("#" * (level + 1)):
            found.append(m.group(1).replace("`", "").strip())
    return found


def slug(title: str) -> str:
    """Anchor of a heading as GitHub and MyST compute it: lower case, punctuation dropped, spaces to hyphens."""
    text = title.strip().lower().replace("`", "")
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _tables(text: str) -> list[list[list[str]]]:
    """Body rows (cells) of every Markdown table outside code fences; header and separator rows dropped."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] | None = None
    for line in [*_outside_fences(text), ""]:
        if line.lstrip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if current is None:
                current = []  # header row
                continue
            if all(SEPARATOR_CELL_RE.fullmatch(c) for c in cells):
                continue
            current.append(cells)
            continue
        if current is not None:
            tables.append(current)
            current = None
    return tables


def _usage_links(row: list[str]) -> list[str]:
    return [href for href in LINK_RE.findall(" | ".join(row)) if href.split("#", 1)[0] == "docs/usage.md"]


def _commands_in(row: list[str]) -> list[str]:
    joined = " | ".join(row)
    return [c for c in STAGE_COMMANDS if re.search(rf"(?<![\w-]){re.escape(c)}(?![\w-])", joined)]


def _stage_table() -> list[list[str]]:
    """The one README.md table whose rows link to docs/usage.md."""
    found = [t for t in _tables(read_text(README)) if any(_usage_links(row) for row in t)]
    assert len(found) == 1, (
        f"README.md must hold exactly one table whose rows link to docs/usage.md; found {len(found)}"
    )
    return found[0]


def _usage_sections() -> dict[str, str]:
    """Level-2 sections of docs/usage.md, by anchor."""
    sections: dict[str, str] = {}
    for m in re.finditer(r"^##\s+([^\n]*?)\s*$(.*?)(?=^##\s|\Z)", read_text(USAGE), re.MULTILINE | re.DOTALL):
        sections[slug(m.group(1))] = m.group(2)
    return sections


@pytest.mark.smoke
class TestStageTable:
    """REQ-DOCS-BDRABCZUK-25B28869C99D."""

    def test_one_table_names_the_six_stage_scripts_in_order(self):
        rows = _stage_table()
        assert len(rows) == len(STAGE_COMMANDS), f"the README stage table must have six body rows; found {len(rows)}"
        named = [_commands_in(row) for row in rows]
        assert all(len(n) == 1 for n in named), f"each stage table row must name exactly one stage script: {named}"
        assert [n[0] for n in named] == list(STAGE_COMMANDS), f"stage table rows out of pipeline order: {named}"
        unlinked = [row for row in rows if not _usage_links(row)]
        assert not unlinked, f"stage table rows without a docs/usage.md link: {unlinked}"


@pytest.mark.smoke
class TestStageTableLinks:
    """REQ-DOCS-BDRABCZUK-91EDE22AA9A9."""

    def test_row_links_the_usage_section_of_its_stage(self):
        sections = _usage_sections()
        wrong = []
        for row in _stage_table():
            command = _commands_in(row)[0]
            anchors = [href.split("#", 1)[1] if "#" in href else "" for href in _usage_links(row)]
            if not any(a in sections and pixi_run_examples(sections[a], command) for a in anchors):
                wrong.append(f"{command}: links {anchors}")
        assert not wrong, (
            "each stage table row must link the docs/usage.md level-2 section showing a "
            "'pixi run <its console script>' line:\n" + "\n".join(wrong)
        )


@pytest.mark.smoke
class TestNoStageSections:
    """REQ-DOCS-BDRABCZUK-696EC8E1E982."""

    def test_no_level_2_heading_names_a_stage(self):
        offending = [
            h for h in _headings(read_text(README), 2) if any(title.lower() in h.lower() for title in STAGE_TITLES)
        ]
        assert not offending, f"README.md still has level-2 stage sections (moved to docs/usage.md): {offending}"


@pytest.mark.smoke
class TestStageExamplesMoved:
    """REQ-DOCS-BDRABCZUK-EB6EFE9020B9 and REQ-DOCS-BDRABCZUK-98E8342B8297 (one test each)."""

    def test_at_most_one_command_line_per_stage_script(self):
        lines = _code_lines(read_text(README))
        counts = {
            c: sum(1 for ln in lines if re.search(rf"\bpixi run {re.escape(c)}(?![\w-])", ln)) for c in STAGE_COMMANDS
        }
        over = {c: n for c, n in counts.items() if n > 1}
        assert not over, f"README.md shows more than one pixi run command line for these stage scripts: {over}"

    def test_no_level_3_heading_shared_with_a_usage_stage_section(self):
        usage = read_text(USAGE)
        stage_sections = re.findall(r"^##\s+Stage \d\b[^\n]*$(.*?)(?=^##\s|\Z)", usage, re.MULTILINE | re.DOTALL)
        assert len(stage_sections) == len(STAGE_COMMANDS), "docs/usage.md must keep its six '## Stage N' sections"
        usage_h3 = {h for body in stage_sections for h in _headings(body, 3)}
        shared = sorted(usage_h3 & set(_headings(read_text(README), 3)))
        assert not shared, f"README.md repeats level-3 headings of the docs/usage.md Stage sections: {shared}"
