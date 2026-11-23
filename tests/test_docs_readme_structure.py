#!/usr/bin/env python3
"""Documentation guards: README's stage-by-stage commands sit in a collapsed Advanced block (TASK-DOCS A).

Run by hand (``verify_ignore`` skips ``tests/test_docs_*.py``):
``pixi run pytest tests/test_docs_readme_structure.py``.

Requirements pinned here (one test each):

* REQ-DOCS-BDRABCZUK-EC0C61CDDFAC - in README's first level-2 section, the per-stage ``champollion-*``
  command lines sit inside a ``<details>`` element without an ``open`` attribute.
* REQ-DOCS-BDRABCZUK-14C96FADF5E0 - the summary of that element contains the word Advanced.
* REQ-DOCS-BDRABCZUK-EEEF530BFBC1 - its fenced bash block holds no whole-line shell comment.
* REQ-DOCS-BDRABCZUK-F2FEAF43A2BF - it links to ``docs/usage.md``.

README.md is read as plain text: no import of the package, no network.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.docs_test_helpers import DETAILS_RE, STAGE_LINE_RE, bash_blocks, first_section, read_text, stage_details

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def _section() -> str:
    return first_section(read_text(README))


def test_quick_start_stage_commands_sit_in_closed_details():
    section = _section()
    lines = list(STAGE_LINE_RE.finditer(section))
    assert lines, "no 'pixi run champollion-<stage>' line in README's first level-2 section"
    spans = [(d.start(), d.end(), d.group("attrs")) for d in DETAILS_RE.finditer(section)]
    outside = [m.group(1) for m in lines if not any(s <= m.start() < e for s, e, _ in spans)]
    assert not outside, f"per-stage command lines outside any <details> element: {outside}"
    opened = [
        m.group(1)
        for m in lines
        for s, e, attrs in spans
        if s <= m.start() < e and re.search(r"(?<![\w-])open(?![\w-])", attrs)
    ]
    assert not opened, f"per-stage command lines inside a <details open> element (expanded by default): {opened}"


def test_quick_start_details_summary_names_advanced():
    body = stage_details(_section()).group("body")
    summary = re.search(r"<summary[^>]*>(.*?)</summary>", body, re.DOTALL)
    assert summary, "the <details> element holding the stage commands has no <summary>"
    assert re.search(r"\bAdvanced\b", summary.group(1)), f"summary lacks the word Advanced: {summary.group(1)!r}"


def test_quick_start_details_bash_block_has_no_comment_line():
    blocks = bash_blocks(stage_details(_section()).group("body"))
    assert blocks, "the <details> element holding the stage commands has no fenced bash block"
    comments = [ln.strip() for b in blocks for ln in b.splitlines() if ln.lstrip().startswith("#")]
    assert not comments, f"whole-line shell comments remain in the Advanced example: {comments}"


def test_quick_start_details_links_usage_md():
    body = stage_details(_section()).group("body")
    assert re.search(r"\]\(docs/usage\.md(?:#[^)\s]*)?\)", body), (
        "the <details> element holding the stage commands has no Markdown link to docs/usage.md"
    )
