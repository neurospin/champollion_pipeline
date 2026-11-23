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
