#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Documentation guard: shell examples stay copy-pasteable.

Requirement pinned here: DEF-DOCS-BDRABCZUK-8F5506B8A366 - in a fenced shell block of README.md or
docs/**/*.md, no line ends with a line-continuation backslash followed by whitespace or by a
comment. Pasted into bash, such a line ends the command there and the next line runs on its own
(``--masks: command not found``).
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SHELL_TAGS = {"", "bash", "sh", "shell", "console"}
FENCE = re.compile(r"^\s*(`{3,}|~{3,})\s*([A-Za-z0-9_+-]*)")
BAD_CONTINUATION = re.compile(r"\\[ \t]+(#.*)?$")


def _markdown_files():
    files = [ROOT / "README.md"]
    files += sorted((ROOT / "docs").rglob("*.md"))
    return [f for f in files if f.is_file()]


def _offending_lines(path):
    """Return ``(line number, text)`` of each bad continuation in a shell fenced block."""
    found = []
    fence = None
    in_shell = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = FENCE.match(line)
        if match and fence is None:
            fence = match.group(1)
            in_shell = match.group(2).lower() in SHELL_TAGS
        elif match and fence is not None and match.group(1)[0] == fence[0] and not match.group(2):
            fence = None
            in_shell = False
        elif fence is not None and in_shell and BAD_CONTINUATION.search(line):
            found.append((number, line.strip()))
    return found


@pytest.mark.parametrize("path", _markdown_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_shell_blocks_have_no_backslash_followed_by_space_or_comment(path):
    offenders = _offending_lines(path)
    assert not offenders, "backslash continuation followed by whitespace or comment: " + "; ".join(
        f"line {n}: {text}" for n, text in offenders
    )
