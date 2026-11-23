"""The eleven champollion-pipeline options no stage reads are documented as unused (TASK-DOCS-BDRABCZUK-E219A2CD1366).

User decision 2026-11-23: --datasets, --labels, --classifier-name, --subsets, --epochs, --split, --cv,
--splits-basedir, --idx-region-evaluation, --short-name and --champollion-v1-path stay in the parsers. Each is
described, in the README.md options table, in the docs/usage.md champollion-pipeline reference options table and in
its --help text, by the same statement; none of those texts says that training reads it (champollion-train builds
its own parser and never loads the Config file).

This file is deliberately not a tests/test_docs_*.py file: ``verify_ignore`` skips those in full runs, and this
check must run in every full suite.

Requirements (system level, deriving from REQ-DOCS-BDRABCZUK-CA7CB217BDB1): the README.md Meaning cell, the
docs/usage.md Effect cell and the --help text of each option contain the statement and not the word 'training'.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from champollion_pipeline.cli import ChampollionPipeline
from tests.docs_test_helpers import DETAILS_RE, first_section, md_cell_text, md_section_lines, read_text

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
USAGE = ROOT / "docs" / "usage.md"
STATEMENT = "Config file field, read by no stage and not by champollion-train"
TRAINING_RE = re.compile(r"\btraining\b", re.IGNORECASE)
UNREAD_OPTIONS = (
    "--datasets",
    "--labels",
    "--classifier-name",
    "--subsets",
    "--epochs",
    "--split",
    "--cv",
    "--splits-basedir",
    "--idx-region-evaluation",
    "--short-name",
    "--champollion-v1-path",
)
README_SUMMARY = "champollion-pipeline options"
USAGE_SECTION = "champollion-pipeline reference"
_CELL_SPLIT = re.compile(r"(?<!\\)\|")


def _table_rows(lines: list[str]) -> list[dict[str, str]]:
    """Rows of the first Markdown table in ``lines``, keyed by the header cells (plain text)."""
    table = [line.strip() for line in lines if line.strip().startswith("|")]
    assert len(table) >= 2, "no Markdown table found"
    header = [md_cell_text(cell) for cell in _CELL_SPLIT.split(table[0].strip("|"))]
    rows = []
    for line in table[2:]:
        cells = [md_cell_text(cell) for cell in _CELL_SPLIT.split(line.strip("|"))]
        rows.append(dict(zip(header, cells, strict=False)))
    return rows


def _row_of(rows: list[dict[str, str]], option: str, where: str) -> dict[str, str]:
    """The row whose Option cell names ``option`` exactly (``--split`` is not ``--splits-basedir``)."""
    for row in rows:
        if option in re.split(r"[\s,]+", row.get("Option", "")):
            return row
    pytest.fail(f"{where}: no row names {option}")


def _readme_rows() -> list[dict[str, str]]:
    for match in DETAILS_RE.finditer(first_section(read_text(README))):
        if README_SUMMARY in match.group("body").split("</summary>", 1)[0]:
            return _table_rows(match.group("body").splitlines())
    pytest.fail(f"README.md: no details element whose summary contains {README_SUMMARY!r}")


def _usage_rows() -> list[dict[str, str]]:
    lines = md_section_lines(read_text(USAGE), USAGE_SECTION)
    option_tables = [i for i, line in enumerate(lines) if re.match(r"^\s*\|\s*Option\s*\|", line)]
    assert option_tables, f"docs/usage.md {USAGE_SECTION}: no table with an Option column"
    return _table_rows(lines[option_tables[0] :])


def _help_texts() -> dict[str, str]:
    """Help text of each option string of the champollion-pipeline parser."""
    return {
        string: action.help or ""
        for action in ChampollionPipeline().parser._actions
        for string in action.option_strings
    }


def _assert_marks_unread(text: str, where: str) -> None:
    assert STATEMENT in text, f"{where} lacks {STATEMENT!r}: {text!r}"
    assert not TRAINING_RE.search(text), f"{where} names training: {text!r}"


@pytest.mark.parametrize("option", UNREAD_OPTIONS)
def test_readme_meaning_cell_marks_option_unread(option):
    """README.md options table: the Meaning cell of the option's row states it is unread, without 'training'."""
    row = _row_of(_readme_rows(), option, "README.md options table")
    _assert_marks_unread(row.get("Meaning", ""), f"README.md Meaning cell of {option}")


@pytest.mark.parametrize("option", UNREAD_OPTIONS)
def test_usage_effect_cell_marks_option_unread(option):
    """docs/usage.md reference options table: the Effect cell states the option is unread, without 'training'."""
    row = _row_of(_usage_rows(), option, f"docs/usage.md {USAGE_SECTION} options table")
    _assert_marks_unread(row.get("Effect", ""), f"docs/usage.md Effect cell of {option}")


@pytest.mark.parametrize("option", UNREAD_OPTIONS)
def test_help_text_marks_option_unread(option):
    """champollion-pipeline --help: the option's help text states it is unread, without 'training'."""
    helps = _help_texts()
    assert option in helps, f"champollion-pipeline has no {option} option"
    _assert_marks_unread(helps[option], f"--help text of {option}")
