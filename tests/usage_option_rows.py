"""Test helper: the rows of the ``| Option |`` tables of a Markdown section, keyed by their ``--flag``.

Shared by tests/test_docs_modelver.py and tests/test_profiling_deprecation.py, which both read the
options tables of the Stage 4 section of ``docs/usage.md``. Reads text only: no filesystem writes,
no network.
"""

from __future__ import annotations

import re

OPTION_TABLE_HEADER_RE = re.compile(r"^\|\s*Option\s*\|")
FLAG_RE = re.compile(r"--[A-Za-z][A-Za-z0-9_-]*")


def option_table_rows(section: str) -> dict[str, str]:
    """Map the first-cell ``--flag`` of each row of the section's 'Option' tables to the whole row."""
    rows: dict[str, str] = {}
    in_table = False
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            in_table = False
            continue
        if OPTION_TABLE_HEADER_RE.match(stripped):
            in_table = True
            continue
        if in_table:
            first = stripped.strip("|").split("|", 1)[0]
            flag = FLAG_RE.search(first)
            if flag:
                rows[flag.group(0)] = stripped
    assert rows, "section has no '| Option |' table rows (scope helper broken?)"
    return rows
