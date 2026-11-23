"""README.md lists every option of champollion-pipeline --help (TASK-DOCS-BDRABCZUK-84C942838E50).

The first level-2 section of README.md (Quick start) holds a collapsed details element whose summary contains
"champollion-pipeline options"; inside it, a Markdown table with the columns Option, Argument, Default, Meaning has one
row per option of the champollion-pipeline argument parser.

The option list comes from the parser itself, never from a copy: a new interpreter loads the
``[project.scripts] champollion-pipeline`` target of pyproject.toml, runs it with ``--help`` and captures the
ArgumentParser at its first parse call, before any stage, update check or help output. Going through the console-script
target keeps the test valid whatever module builds the parser (the cli.py split TASK-CODE-BDRABCZUK-D457E2861B14).

This file is deliberately not a tests/test_docs_*.py file: ``verify_ignore`` skips those in full runs, and this test
must run in every full suite so that a new or removed flag fails it.

Requirements (parent REQ-DOCS-BDRABCZUK-CA7CB217BDB1):
- the first level-2 section of README.md holds the champollion-pipeline options details element;
- the element carries no open attribute (collapsed);
- its Markdown table has the header cells Option, Argument, Default, Meaning;
- each option string of the parser is in the Option cell of a row;
- no Option cell names an option string the parser lacks;
- a value-taking option's Argument cell holds a placeholder other than a single dash;
- an option taking no value has a single dash in its Argument cell;
- each row has a non-empty Default cell;
- each row has a Meaning cell with at least one letter.
"""

from __future__ import annotations

import functools
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SUMMARY_TEXT = "champollion-pipeline options"
HEADER = ["Option", "Argument", "Default", "Meaning"]
OPTION_RE = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")

# Run in a new interpreter: the console-script target sets up logging for the process, which must not leak into the
# test worker. The parser is captured at its first parse call and dumped as JSON; nothing else of the run happens.
_CAPTURE_SCRIPT = r"""
import argparse, importlib, json, sys, tomllib

root, out = sys.argv[1], sys.argv[2]
with open(root + "/pyproject.toml", "rb") as handle:
    target = tomllib.load(handle)["project"]["scripts"]["champollion-pipeline"]
module_name, _, attr = target.partition(":")


class _Captured(Exception):
    pass


def _capture(self, args=None, namespace=None):
    rows = [
        {"option_strings": list(action.option_strings), "nargs": action.nargs}
        for action in self._actions
        if action.option_strings
    ]
    with open(out, "w") as handle:
        json.dump(rows, handle)
    raise _Captured


argparse.ArgumentParser.parse_known_args = _capture
sys.argv = ["champollion-pipeline", "--help"]
entry = importlib.import_module(module_name)
for part in attr.split("."):
    entry = getattr(entry, part)
try:
    entry()
except _Captured:
    sys.exit(0)
sys.exit("the champollion-pipeline entry point returned without parsing its arguments")
"""


@functools.lru_cache(maxsize=1)
def _parser_options() -> tuple[tuple[tuple[str, ...], bool], ...]:
    """Each option of the champollion-pipeline parser: (its option strings, whether it takes a value)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), env.get("PYTHONPATH", "")]))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "options.json"
        result = subprocess.run(
            [sys.executable, "-c", _CAPTURE_SCRIPT, str(ROOT), str(out)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert result.returncode == 0 and out.exists(), (
            f"could not capture the champollion-pipeline parser (exit {result.returncode}):\n{result.stderr[-2000:]}"
        )
        rows = json.loads(out.read_text(encoding="utf-8"))
    return tuple((tuple(row["option_strings"]), row["nargs"] != 0) for row in rows)


def _parser_option_strings() -> set[str]:
    return {string for strings, _ in _parser_options() for string in strings}


def _first_section() -> str:
    """README.md from its first level-2 heading to the next one."""
    text = README.read_text(encoding="utf-8")
    start = re.search(r"^## .*$", text, re.MULTILINE)
    assert start, "README.md has no level-2 heading"
    end = re.search(r"^## ", text[start.end() :], re.MULTILINE)
    return text[start.start() : start.end() + end.start()] if end else text[start.start() :]


def _options_details() -> re.Match:
    """The details element of the first section whose summary contains 'champollion-pipeline options'."""
    for match in re.finditer(r"<details(?P<attrs>[^>]*)>(?P<body>.*?)</details>", _first_section(), re.DOTALL):
        summary = re.search(r"<summary>(.*?)</summary>", match.group("body"), re.DOTALL)
        if summary and SUMMARY_TEXT in summary.group(1):
            return match
    pytest.fail(f"the first section of README.md has no details element whose summary contains {SUMMARY_TEXT!r}")


def _cells(line: str) -> list[str]:
    """Cells of one table line; an escaped pipe (\\|) stays inside its cell."""
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def _table() -> tuple[list[str], list[list[str]]]:
    """Header cells and body rows of the first Markdown table in the options details element."""
    lines = [line for line in _options_details().group("body").splitlines() if line.strip().startswith("|")]
    assert len(lines) >= 2, "the champollion-pipeline options details element holds no Markdown table"
    assert re.fullmatch(r"\|?(\s*:?-+:?\s*\|)+\s*:?-*:?\s*\|?", lines[1].strip()), (
        "the second table line is not a Markdown delimiter row"
    )
    return _cells(lines[0]), [_cells(line) for line in lines[2:]]


def _rows() -> list[dict[str, str]]:
    header, rows = _table()
    assert header == HEADER, f"options table header is {header}, expected {HEADER}"
    return [dict(zip(HEADER, (row + [""] * len(HEADER))[: len(HEADER)], strict=True)) for row in rows]


def _row_options(row: dict[str, str]) -> set[str]:
    return set(OPTION_RE.findall(row["Option"]))


def _row_of(strings: tuple[str, ...]) -> dict[str, str]:
    for row in _rows():
        if _row_options(row) & set(strings):
            return row
    pytest.fail(f"no row of the README.md options table names {strings[0]}")


def _argument(row: dict[str, str]) -> str:
    return row["Argument"].strip().strip("`").strip()


def test_quick_start_has_options_details():
    """The first level-2 section of README.md holds the champollion-pipeline options details element."""
    assert _options_details()


def test_options_details_is_collapsed():
    """The champollion-pipeline options details element carries no open attribute."""
    attrs = _options_details().group("attrs")
    assert not re.search(r"\bopen\b", attrs), f"the details element is open: <details{attrs}>"


def test_options_table_header():
    """The options table has the header cells Option, Argument, Default, Meaning, in this order."""
    header, _ = _table()
    assert header == HEADER


def test_each_parser_option_has_a_row():
    """Each option string of the champollion-pipeline parser is in the Option cell of a row."""
    listed = {string for row in _rows() for string in _row_options(row)}
    missing = sorted(_parser_option_strings() - listed)
    assert not missing, f"options of champollion-pipeline --help missing from the README.md table: {missing}"


def test_no_row_names_an_unknown_option():
    """No Option cell names an option string the champollion-pipeline parser lacks."""
    listed = {string for row in _rows() for string in _row_options(row)}
    unknown = sorted(listed - _parser_option_strings())
    assert not unknown, f"README.md options table names options champollion-pipeline does not have: {unknown}"


def test_value_option_rows_name_an_argument():
    """A value-taking option's Argument cell holds a placeholder other than a single dash."""
    bad = [
        strings[0]
        for strings, takes_value in _parser_options()
        if takes_value and _argument(_row_of(strings)) in ("", "-")
    ]
    assert not bad, f"value-taking options without an argument placeholder: {bad}"


def test_flag_rows_show_a_dash_argument():
    """An option taking no value has a single dash in its Argument cell."""
    bad = [
        strings[0]
        for strings, takes_value in _parser_options()
        if not takes_value and _argument(_row_of(strings)) != "-"
    ]
    assert not bad, f"options taking no value whose Argument cell is not a single dash: {bad}"


def test_rows_have_a_default():
    """Each row of the options table has a non-empty Default cell."""
    rows = _rows()
    assert rows, "the README.md options table has no row"
    empty = [row["Option"] for row in rows if not row["Default"].strip("` ")]
    assert not empty, f"rows with an empty Default cell: {empty}"


def test_rows_have_a_meaning():
    """Each row of the options table has a Meaning cell with at least one letter."""
    rows = _rows()
    assert rows, "the README.md options table has no row"
    blank = [row["Option"] for row in rows if not re.search(r"[A-Za-z]", row["Meaning"])]
    assert not blank, f"rows whose Meaning cell has no letter: {blank}"
