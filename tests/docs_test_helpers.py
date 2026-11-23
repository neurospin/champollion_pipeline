"""Test helper: read the shell examples of the Markdown docs (``README.md``, ``docs/usage.md``).

Shared by the ``tests/test_docs_*.py`` guards that check a documented ``pixi run <command>``
example against the real ``argparse`` parser. Everything here reads text or inspects a parser
that is built, never parsed: no filesystem writes, no network.
"""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path


def read_text(path: Path) -> str:
    """Text of ``path``, asserting the file exists."""
    assert path.is_file(), f"expected file to exist: {path}"
    return path.read_text(encoding="utf-8")


def norm(path: str) -> str:
    """Compare directories regardless of a trailing slash."""
    return path.rstrip("/")


def md_section(text: str, heading_regex: str) -> str:
    """Body of the ``## <heading>`` section whose heading line matches ``heading_regex`` to its end."""
    m = re.search(rf"^##\s+{heading_regex}$(.*?)(?=^##\s|\Z)", text, re.MULTILINE | re.DOTALL)
    assert m, f"no '## {heading_regex}' section found"
    return m.group(1)


def bash_blocks(text: str) -> list[str]:
    """Bodies of every fenced ``bash`` code block."""
    return re.findall(r"```bash\n(.*?)```", text, re.DOTALL)


def strip_comment(line: str) -> str:
    """Drop a shell comment: a whole comment line, or a `` # ...`` tail after whitespace."""
    if line.lstrip().startswith("#"):
        return ""
    return re.sub(r"\s+#.*$", "", line)


def pixi_run_examples(text: str, command: str) -> list[list[str]]:
    """Argument tokens (after the command name) of every ``pixi run <command>`` example in ``text``.

    Backslash-continued lines are joined; shell comments are dropped. Returns ``[]`` when there
    is no example: the caller asserts what it needs.
    """
    found: list[list[str]] = []
    for block in bash_blocks(text):
        lines = block.splitlines()
        i = 0
        while i < len(lines):
            line = strip_comment(lines[i]).rstrip()
            if re.search(rf"\bpixi run {re.escape(command)}(?:\s|\\|$)", line):
                parts = [line]
                while parts[-1].endswith("\\") and i + 1 < len(lines):
                    i += 1
                    parts.append(strip_comment(lines[i]).rstrip())
                joined = " ".join(p[:-1] if p.endswith("\\") else p for p in parts)
                tokens = shlex.split(joined)
                found.append(tokens[tokens.index(command) + 1 :])
            i += 1
    return found


def required_examples(text: str, command: str, where: str) -> list[list[str]]:
    """``pixi_run_examples``, asserting ``where`` shows at least one (a sanity check of the parser)."""
    found = pixi_run_examples(text, command)
    assert found, f"{where}: no 'pixi run {command}' example found (test parser sanity check)"
    return found


def option_strings(parser: argparse.ArgumentParser) -> set[str]:
    """Every option string declared on ``parser``."""
    return {opt for action in parser._actions for opt in action.option_strings}


def value_taking_options(parser: argparse.ArgumentParser) -> set[str]:
    """Option strings of ``parser`` that consume a following value."""
    return {
        opt
        for action in parser._actions
        if action.option_strings and action.nargs != 0
        for opt in action.option_strings
    }


def positional_args(tokens: list[str], parser: argparse.ArgumentParser) -> list[str]:
    """Tokens that are neither a ``--flag`` nor the value of a value-taking flag of ``parser``."""
    takes_value = value_taking_options(parser)
    declared = option_strings(parser)
    found: list[str] = []
    skip_next = False
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue
        if tok.startswith("-"):
            name = tok.split("=", 1)[0]
            # An unknown flag is assumed to take a value, as the docs examples write ``--flag value``.
            skip_next = "=" not in tok and (name in takes_value or name not in declared)
            continue
        found.append(tok)
    return found


STAGE_COMMANDS = (
    "champollion-morphologist",
    "champollion-cortical-tiles",
    "champollion-config",
    "champollion-embeddings",
    "champollion-combine",
    "champollion-snapshots",
)
STAGE_LINE_RE = re.compile(
    r"^[ \t]*pixi run (" + "|".join(re.escape(c) for c in STAGE_COMMANDS) + r")(?![\w-])", re.MULTILINE
)
DETAILS_RE = re.compile(r"<details(?P<attrs>[^>]*)>(?P<body>.*?)</details>", re.DOTALL)


def first_section(text: str) -> str:
    """Body of the first ``## `` section of a Markdown text (the README's Quick start)."""
    m = re.search(r"^##\s+.*?$(.*?)(?=^##\s|\Z)", text, re.MULTILINE | re.DOTALL)
    assert m, "no level-2 section found"
    return m.group(1)


def stage_details(section: str) -> re.Match:
    """The one ``<details>`` element of ``section`` holding a ``pixi run <stage command>`` line."""
    found = [d for d in DETAILS_RE.finditer(section) if STAGE_LINE_RE.search(d.group("body"))]
    assert len(found) == 1, (
        f"expected exactly one <details> element holding the per-stage champollion-* commands, found {len(found)}"
    )
    return found[0]


# Module READMEs (``src/champollion_pipeline/**/README.md``): the format shared by the
# ``tests/test_docs_module_readmes*.py`` guards (a level-1 title, then the level-2 sections below).
MODULE_README_SECTIONS = ["Purpose", "Files", "Entry points", "See also"]
MD_FENCE_RE = re.compile(r"^\s*(```|~~~)")
MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
MD_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")


def md_prose_lines(text: str) -> list[str]:
    """Lines of ``text`` outside fenced code blocks."""
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if MD_FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return lines


def md_headings(text: str) -> list[tuple[int, str]]:
    """``(level, title)`` of each Markdown heading outside fenced code blocks."""
    found = []
    for line in md_prose_lines(text):
        m = MD_HEADING_RE.match(line)
        if m:
            found.append((len(m.group(1)), m.group(2)))
    return found


def md_section_lines(text: str, title: str, *, prose_only: bool = True) -> list[str]:
    """Lines of the level-2 section ``title``, up to the next level-1 or level-2 heading.

    With ``prose_only`` (the default) fenced code blocks are left out; otherwise their lines are kept,
    so a command shown in a code block still counts as named.
    """
    body: list[str] = []
    inside = found = in_fence = False
    for line in text.splitlines():
        if MD_FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        m = None if in_fence else MD_HEADING_RE.match(line)
        if m and len(m.group(1)) <= 2:
            inside = len(m.group(1)) == 2 and m.group(2) == title
            found = found or inside
            continue
        if inside and not (in_fence and prose_only):
            body.append(line)
    assert found, f"no '## {title}' section found"
    return body


def md_cell_text(cell: str) -> str:
    """Plain text of a table cell: Markdown links reduced to their text, backticks dropped."""
    cell = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", cell)
    return cell.replace("`", "").strip()


def module_readme_files_rows(text: str) -> list[list[str]]:
    """Body rows (cells as plain text) of the table in the Files section; header and separator dropped."""
    rows = []
    for line in md_section_lines(text, "Files"):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [md_cell_text(c) for c in stripped.strip("|").split("|")]
        if all(MD_SEPARATOR_CELL_RE.match(c) for c in cells if c):
            continue
        rows.append(cells)
    assert rows, "the Files section holds no Markdown table"
    return rows[1:]


def assert_module_readme_layout(text: str, name: str) -> None:
    """Assert ``text`` holds one level-1 title, first, then exactly the ``MODULE_README_SECTIONS`` in order."""
    headings = md_headings(text)
    assert headings, f"{name}/README.md has no heading"
    assert headings[0][0] == 1, f"{name}/README.md does not open with a level-1 title: {headings[0]}"
    level_one = [t for lvl, t in headings if lvl == 1]
    assert len(level_one) == 1, f"{name}/README.md has {len(level_one)} level-1 headings: {level_one}"
    level_two = [t for lvl, t in headings if lvl == 2]
    assert level_two == MODULE_README_SECTIONS, (
        f"{name}/README.md level-2 sections are {level_two}, expected {MODULE_README_SECTIONS}"
    )
