"""The options table of each Stage N section of docs/usage.md against the parser of its stage command.

TASK-DOCS-BDRABCZUK-B56C14E1874F. Stage N runs the N-th console script of ``STAGE_COMMANDS``
(champollion-morphologist, champollion-cortical-tiles, champollion-config, champollion-embeddings,
champollion-combine, champollion-snapshots). Its section runs from the second-level heading starting
``Stage N`` to the next first- or second-level heading; its options table is the one Markdown table of
that section whose header row reads ``Option | Argument | Default | Effect``.

The parser is built, never run: the module comes from the command's entry point in pyproject.toml's
[project.scripts], and the parser is the ``.parser`` of the one ScriptBuilder subclass that module
defines. No module path is written here, so the test survives a move of the parser between modules.
``-h`` and ``--help`` are left out: argparse adds them to every parser.

The Markdown readers are the shared ones of tests/docs_test_helpers.py (``md_prose_lines``,
``MD_HEADING_RE``, ``md_cell_text``, ``MD_SEPARATOR_CELL_RE``, ``option_strings``, ``STAGE_COMMANDS``);
the table and parser lookups below are private to this file (no new shared helper).

This file is deliberately not named tests/test_docs_*.py: verify_ignore leaves those out of the full
suite, and this check must run in every full suite so that a parser change cannot leave the page stale.

Requirements (system level, deriving from REQ-DOCS-BDRABCZUK-CA7CB217BDB1), one per stage:
REQ-DOCS-BDRABCZUK-BEA4A04F2229 (Stage 1, champollion-morphologist),
REQ-DOCS-BDRABCZUK-2B447F8178D4 (Stage 2, champollion-cortical-tiles),
REQ-DOCS-BDRABCZUK-103D513C686E (Stage 3, champollion-config),
REQ-DOCS-BDRABCZUK-2867E2599A40 (Stage 4, champollion-embeddings),
REQ-DOCS-BDRABCZUK-1EDFBD123A49 (Stage 5, champollion-combine),
REQ-DOCS-BDRABCZUK-C20A893BC6DE (Stage 6, champollion-snapshots).
"""

from __future__ import annotations

import argparse
import functools
import importlib
import inspect
import re
from pathlib import Path

import pytest
import tomllib
from champollion_utils.script_builder import ScriptBuilder

from tests.docs_test_helpers import (
    MD_HEADING_RE,
    MD_SEPARATOR_CELL_RE,
    STAGE_COMMANDS,
    md_cell_text,
    md_prose_lines,
    option_strings,
    read_text,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
USAGE = REPO_ROOT / "docs" / "usage.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"

TABLE_HEADER = ["Option", "Argument", "Default", "Effect"]
HELP_OPTIONS = frozenset({"-h", "--help"})

_PIPE = re.compile(r"(?<!\\)\|")
_OPTION_STRING = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")


def _stage_number(command: str) -> int:
    return STAGE_COMMANDS.index(command) + 1


# --- the stage parsers --------------------------------------------------------------------------------------------


@functools.cache
def _parser(command: str) -> argparse.ArgumentParser:
    """The parser of ``command``: that of the one ScriptBuilder subclass its entry-point module defines."""
    scripts = tomllib.loads(read_text(PYPROJECT))["project"]["scripts"]
    assert command in scripts, f"pyproject.toml [project.scripts] declares no {command}"
    module = importlib.import_module(scripts[command].partition(":")[0])
    builders = [
        cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if issubclass(cls, ScriptBuilder) and cls.__module__ == module.__name__
    ]
    assert len(builders) == 1, f"{module.__name__} defines {len(builders)} ScriptBuilder subclasses, expected 1"
    return builders[0]().parser


def _parser_options(command: str) -> set[str]:
    return option_strings(_parser(command)) - HELP_OPTIONS


# --- docs/usage.md ------------------------------------------------------------------------------------------------


def _stage_section(stage: int) -> list[str]:
    """Prose lines of the ``## Stage <stage> ...`` section, up to the next first- or second-level heading."""
    title = re.compile(rf"Stage {stage}(?!\d)")
    body: list[str] = []
    inside = found = False
    for line in md_prose_lines(read_text(USAGE)):
        heading = MD_HEADING_RE.match(line)
        if heading and len(heading.group(1)) <= 2:
            inside = len(heading.group(1)) == 2 and bool(title.match(heading.group(2)))
            found = found or inside
        elif inside:
            body.append(line)
    assert found, f"docs/usage.md has no '## Stage {stage}' section"
    return body


def _table_rows(lines: list[str]) -> list[list[list[str]]]:
    """Each Markdown table of ``lines`` as its rows of plain-text cells, separator rows dropped."""
    tables: list[list[list[str]]] = []
    rows: list[list[str]] = []
    for line in [*lines, ""]:
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [md_cell_text(cell) for cell in _PIPE.split(stripped.strip("|"))]
            if not all(MD_SEPARATOR_CELL_RE.match(cell) for cell in cells):
                rows.append(cells)
        elif rows:
            tables.append(rows)
            rows = []
    return tables


@functools.cache
def _options_table(command: str) -> tuple[tuple[str, ...], ...]:
    """Body rows of the Stage section's table headed ``Option | Argument | Default | Effect``."""
    stage = _stage_number(command)
    found = [rows[1:] for rows in _table_rows(_stage_section(stage)) if rows[0] == TABLE_HEADER]
    assert len(found) == 1, (
        f"the Stage {stage} section of docs/usage.md holds {len(found)} tables headed "
        f"'| {' | '.join(TABLE_HEADER)} |', expected 1 ({command} options)"
    )
    return tuple(tuple(row) for row in found[0])


def _documented_options(command: str) -> set[str]:
    return {opt for row in _options_table(command) for opt in _OPTION_STRING.findall(row[0])}


# --- tests --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("command", STAGE_COMMANDS)
def test_table_names_each_parser_option(command: str) -> None:
    missing = sorted(_parser_options(command) - _documented_options(command))
    assert not missing, (
        f"the Stage {_stage_number(command)} options table of docs/usage.md has no row for these "
        f"{command} options: {missing}"
    )


@pytest.mark.parametrize("command", STAGE_COMMANDS)
def test_table_names_no_option_the_parser_lacks(command: str) -> None:
    unknown = sorted(_documented_options(command) - _parser_options(command))
    assert not unknown, (
        f"the Stage {_stage_number(command)} options table of docs/usage.md names options that the "
        f"{command} parser lacks (or -h/--help): {unknown}"
    )


@pytest.mark.parametrize("command", STAGE_COMMANDS)
def test_table_rows_have_no_empty_cell(command: str) -> None:
    rows = _options_table(command)
    assert rows, f"the Stage {_stage_number(command)} options table of docs/usage.md has no row"
    incomplete = [
        row[0] or "(empty Option cell)"
        for row in rows
        if len(row) != len(TABLE_HEADER) or not all(cell.strip() for cell in row)
    ]
    assert not incomplete, (
        f"the Stage {_stage_number(command)} options table of docs/usage.md leaves an Option, Argument, "
        f"Default or Effect cell empty (or has the wrong number of cells) in rows: {incomplete}"
    )
