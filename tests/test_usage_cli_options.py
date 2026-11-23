"""The champollion-pipeline reference section of docs/usage.md against champollion-pipeline --help.

TASK-DOCS-BDRABCZUK-7BE09099D949. The reference section is the text from docs/usage.md's first second-level heading,
``## champollion-pipeline reference``, to the next second-level heading. Its options table is the Markdown table of
that section whose header row names an ``Option`` column (with ``Argument``, ``Default``, ``Config key`` and
``Effect`` columns); its exit-status table is the one whose header row names an ``Exit status`` column.

The --help output is read by running the console-script entry point that pyproject.toml's [project.scripts] declares
for champollion-pipeline, the way the generated wrapper runs it, in a new interpreter with src/ first on the path. No
module path of the parser is written here, so the test survives a move of the parser between modules. The parser
introspection helper is private to this file (no shared helper).

The config keys and their defaults come from the dataclass fields of champollion_pipeline.config: a PipelineConfig
field is a top-level Config file key, a DatasetConfig field is ``dataset.<field>``.

This file is deliberately not named tests/test_docs_*.py: verify_ignore leaves those out of the full suite, and this
check must run in every full suite so that a parser change cannot leave the page stale.

Requirements (system level, deriving from REQ-DOCS-BDRABCZUK-CA7CB217BDB1):
REQ-DOCS-BDRABCZUK-839EF26839A7 (first second-level heading reads champollion-pipeline reference),
REQ-DOCS-BDRABCZUK-425F420A4596 (Option column lists each --help option and positional argument),
REQ-DOCS-BDRABCZUK-AACA34F3F7FC (section names no option absent from --help),
REQ-DOCS-BDRABCZUK-B7184099E694 (Argument cell: value placeholder, or a dash for no value),
REQ-DOCS-BDRABCZUK-8A622B1E3194 (Default cell states the value in force without the option),
REQ-DOCS-BDRABCZUK-AC0C1B97F6BE (Default cell contains the Config file default),
REQ-DOCS-BDRABCZUK-BD01D1040325 (Config key cell names the key the option sets, or a dash),
REQ-DOCS-BDRABCZUK-B24FCC58FF1A (Effect cell describes the option's effect),
REQ-DOCS-BDRABCZUK-923DA5B0E341 (example passing --input-kind t1),
REQ-DOCS-BDRABCZUK-5EE459A074D0 (example passing --input-kind morphologist),
REQ-DOCS-BDRABCZUK-39409D77698C (exit-status table rows 0, 1, 2, 3; superseded from rows 0, 1, 2).
"""

import dataclasses
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

from champollion_pipeline.config import DatasetConfig, PipelineConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
USAGE = REPO_ROOT / "docs" / "usage.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"
COMMAND = "champollion-pipeline"
REFERENCE_HEADING = "champollion-pipeline reference"
HELP_COLUMNS = "1000"
DASH = "-"

OPTION_COLUMN = "Option"
ARGUMENT_COLUMN = "Argument"
DEFAULT_COLUMN = "Default"
CONFIG_KEY_COLUMN = "Config key"
EFFECT_COLUMN = "Effect"
EXIT_STATUS_COLUMN = "Exit status"
EXIT_STATUSES = {"0", "1", "2", "3"}
NO_SECTION = f"docs/usage.md has no '## {REFERENCE_HEADING}' section"
NO_OPTIONS_TABLE = f"the {REFERENCE_HEADING} section has no table with an '{OPTION_COLUMN}' column"

# Arguments whose Config file key is not their flat name.
NON_FLAT_KEYS = {
    "--dataset-name": "dataset.name",
    "--profile": "profile_dir",
    "--output-path": "dataset.datasets_root",
    "--enable-all-stages": "stages",
    "INPUT": "dataset.input_path",
    "OUTPUT": "dataset.datasets_root",
}

# What the generated console-script wrapper does: import the entry point, call it with the command's argv.
_RUN_ENTRY_POINT = (
    "import importlib, sys\n"
    "module, _, function = sys.argv[1].partition(':')\n"
    "sys.argv = sys.argv[2:]\n"
    "sys.exit(getattr(importlib.import_module(module), function)())\n"
)

_SECTION_HEADER = re.compile(r"^(\S[^:]*):$")
_HELP_ENTRY = re.compile(r"^  (\S.*?)(?: {2,}.*)?$")
_OPTION_STRING = re.compile(r"(?<![\w.-])(--?[A-Za-z][\w-]*)")
_CODE_SPAN = re.compile(r"`([^`]+)`")
_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")
_CELL_SPLIT = re.compile(r"(?<!\\)\|")


@dataclasses.dataclass(frozen=True)
class _HelpArgument:
    """One argument of the --help output: its option strings (or positional name) and its value placeholder."""

    names: tuple[str, ...]
    placeholder: str


# --- champollion-pipeline --help ---------------------------------------------------------------------------------


def _entry_point() -> str:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["scripts"][COMMAND]


def _run_help() -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT / "src"), env.get("PYTHONPATH")) if p)
    env["COLUMNS"] = HELP_COLUMNS
    result = subprocess.run(
        [sys.executable, "-c", _RUN_ENTRY_POINT, _entry_point(), COMMAND, "--help"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"{COMMAND} --help exited {result.returncode}: {result.stderr[-2000:]}"
    return result.stdout


def _help_arguments(help_text: str) -> list[_HelpArgument]:
    """The arguments of the help's argument sections; the indented help lines and the epilog's commands are skipped."""
    arguments = []
    section = None
    for line in help_text.splitlines():
        header = _SECTION_HEADER.match(line)
        if header:
            section = header.group(1)
            continue
        if section is None or not line.startswith("  ") or line.startswith("   "):
            continue
        invocation = _HELP_ENTRY.match(line).group(1)
        if section == "positional arguments":
            name = invocation.split()[0]
            arguments.append(_HelpArgument((name,), name))
        elif invocation.startswith(DASH):
            parts = invocation.split(", ")
            arguments.append(
                _HelpArgument(tuple(part.split(" ", 1)[0] for part in parts), parts[0].partition(" ")[2].strip())
            )
    return arguments


@pytest.fixture(scope="module")
def help_arguments() -> list[_HelpArgument]:
    arguments = _help_arguments(_run_help())
    assert arguments, f"no argument read from {COMMAND} --help"
    return arguments


# --- docs/usage.md --------------------------------------------------------------------------------------------------


def _headings(lines: list[str]) -> list[tuple[int, str]]:
    """(line index, text) of each second-level heading outside fenced code blocks."""
    headings = []
    fenced = False
    for index, line in enumerate(lines):
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and line.startswith("## "):
            headings.append((index, line[3:].replace("`", "").strip()))
    return headings


def _usage_lines() -> list[str]:
    return USAGE.read_text(encoding="utf-8").splitlines()


def _reference_section() -> str:
    """The text from the reference heading to the next second-level heading; empty when the heading is missing."""
    lines = _usage_lines()
    headings = _headings(lines)
    starts = [position for position, (_, text) in enumerate(headings) if text == REFERENCE_HEADING]
    if not starts:
        return ""
    start = headings[starts[0]][0]
    following = headings[starts[0] + 1 :]
    end = following[0][0] if following else len(lines)
    return "\n".join(lines[start:end])


@pytest.fixture(scope="module")
def section() -> str:
    return _reference_section()


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in _CELL_SPLIT.split(line.strip().strip("|"))]


def _tables(text: str) -> list[list[list[str]]]:
    tables, current = [], []
    for line in text.splitlines():
        if line.lstrip().startswith("|"):
            current.append(_cells(line))
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def _table_rows(text: str, column: str) -> list[dict[str, str]]:
    """The rows of the first table whose header row names ``column``, as header-to-cell dicts."""
    for table in _tables(text):
        header = [cell.replace("`", "") for cell in table[0]]
        if column in header:
            body = [row for row in table[1:] if not all(_SEPARATOR_CELL.match(cell) for cell in row)]
            return [dict(zip(header, row, strict=False)) for row in body]
    return []


@pytest.fixture(scope="module")
def option_rows(section: str) -> list[dict[str, str]]:
    return _table_rows(section, OPTION_COLUMN)


def _row_names(row: dict[str, str]) -> set[str]:
    """The option strings and positional names in a row's Option cell: the first word of each code span."""
    return {span.split()[0] for span in _CODE_SPAN.findall(row.get(OPTION_COLUMN, "")) if span.split()}


def _plain(cell: str) -> str:
    return cell.replace("`", "").strip()


def _rows_with_arguments(
    rows: list[dict[str, str]], arguments: list[_HelpArgument]
) -> list[tuple[dict[str, str], _HelpArgument]]:
    """Each row paired with the --help argument it documents; rows naming no --help argument are left out."""
    pairs = []
    for row in rows:
        names = _row_names(row)
        matches = [argument for argument in arguments if names & set(argument.names)]
        if matches:
            pairs.append((row, matches[0]))
    return pairs


def _config_fields() -> dict[str, dataclasses.Field]:
    fields = {field.name: field for field in dataclasses.fields(PipelineConfig) if field.name != "dataset"}
    fields.update({f"dataset.{field.name}": field for field in dataclasses.fields(DatasetConfig)})
    return fields


def _expected_key(argument: _HelpArgument) -> str:
    for name in argument.names:
        if name in NON_FLAT_KEYS:
            return NON_FLAT_KEYS[name]
    long_names = [name for name in argument.names if name.startswith("--")]
    if not long_names:
        return DASH
    flat = long_names[0][2:].replace("-", "_")
    fields = _config_fields()
    for key in (flat, f"dataset.{flat}"):
        if key in fields:
            return key
    return DASH


def _config_file_spelling(value: object) -> str | None:
    """The default as a Config file writes it, for a boolean, integer or non-empty text default; else None."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value:
        return value
    return None


def _code_blocks(text: str) -> list[str]:
    blocks, current, fenced = [], [], False
    for line in text.splitlines():
        if line.startswith("```"):
            if fenced:
                blocks.append("\n".join(current))
                current = []
            fenced = not fenced
        elif fenced:
            current.append(line)
    return blocks


def _example_with_input_kind(text: str, kind: str) -> bool:
    flag = re.compile(rf"--input-kind[ =]{re.escape(kind)}\b")
    return any(COMMAND in block and flag.search(block) for block in _code_blocks(text))


# --- Tests ----------------------------------------------------------------------------------------------------------


def test_first_section_is_the_champollion_pipeline_reference():
    headings = _headings(_usage_lines())
    assert headings, "docs/usage.md has no second-level heading"
    assert headings[0][1] == REFERENCE_HEADING, (
        f"the first second-level heading of docs/usage.md is '{headings[0][1]}', not '{REFERENCE_HEADING}'"
    )


def test_option_column_lists_each_help_option(option_rows, help_arguments):
    assert option_rows, NO_OPTIONS_TABLE
    listed = set().union(*(_row_names(row) for row in option_rows))
    missing = [name for argument in help_arguments for name in argument.names if name not in listed]
    assert not missing, f"{COMMAND} --help arguments missing from the {OPTION_COLUMN} column: {missing}"


def test_section_names_no_option_absent_from_help(section, help_arguments):
    assert section, NO_SECTION
    known = {name for argument in help_arguments for name in argument.names}
    stale = sorted({name for name in _OPTION_STRING.findall(section) if name not in known})
    assert not stale, f"the {REFERENCE_HEADING} section names options {COMMAND} --help lacks: {stale}"


def test_argument_cell_shows_placeholder_or_dash(option_rows, help_arguments):
    assert option_rows, NO_OPTIONS_TABLE
    wrong = []
    for row, argument in _rows_with_arguments(option_rows, help_arguments):
        cell = _plain(row.get(ARGUMENT_COLUMN, ""))
        if not argument.placeholder:
            ok = cell == DASH
        elif argument.placeholder.startswith("{"):
            choices = argument.placeholder[1:].partition("}")[0].split(",")
            ok = any(choice in cell for choice in choices)
        else:
            ok = argument.placeholder.split()[0].strip("[]") in cell
        if not ok:
            wrong.append((argument.names[0], argument.placeholder, cell))
    assert not wrong, f"{ARGUMENT_COLUMN} cells not matching the --help placeholder: {wrong}"


def test_default_cell_is_filled(option_rows):
    assert option_rows, NO_OPTIONS_TABLE
    empty = [sorted(_row_names(row)) for row in option_rows if not _plain(row.get(DEFAULT_COLUMN, ""))]
    assert not empty, f"rows with an empty {DEFAULT_COLUMN} cell: {empty}"


def test_default_cell_shows_the_config_default(option_rows, help_arguments):
    assert option_rows, NO_OPTIONS_TABLE
    fields = _config_fields()
    wrong = []
    for row, argument in _rows_with_arguments(option_rows, help_arguments):
        field = fields.get(_expected_key(argument))
        if field is None or field.default is dataclasses.MISSING:
            continue
        spelling = _config_file_spelling(field.default)
        if spelling is not None and spelling not in row.get(DEFAULT_COLUMN, ""):
            wrong.append((argument.names[0], spelling, row.get(DEFAULT_COLUMN, "")))
    assert not wrong, f"{DEFAULT_COLUMN} cells without the Config file default: {wrong}"


def test_config_key_cell_names_the_key_the_option_sets(option_rows, help_arguments):
    assert option_rows, NO_OPTIONS_TABLE
    wrong = []
    for row, argument in _rows_with_arguments(option_rows, help_arguments):
        expected = _expected_key(argument)
        cell = _plain(row.get(CONFIG_KEY_COLUMN, ""))
        if cell != expected:
            wrong.append((argument.names[0], expected, cell))
    assert not wrong, f"{CONFIG_KEY_COLUMN} cells (option, expected, found): {wrong}"


def test_effect_cell_is_filled(option_rows):
    assert option_rows, NO_OPTIONS_TABLE
    empty = [sorted(_row_names(row)) for row in option_rows if _plain(row.get(EFFECT_COLUMN, "")) in ("", DASH)]
    assert not empty, f"rows with an empty {EFFECT_COLUMN} cell: {empty}"


def test_t1_example_passes_input_kind_t1(section):
    assert section, NO_SECTION
    assert _example_with_input_kind(section, "t1"), (
        f"no {COMMAND} example command passing --input-kind t1 in the {REFERENCE_HEADING} section"
    )


def test_morphologist_example_passes_input_kind_morphologist(section):
    assert section, NO_SECTION
    assert _example_with_input_kind(section, "morphologist"), (
        f"no {COMMAND} example command passing --input-kind morphologist in the {REFERENCE_HEADING} section"
    )


def test_exit_status_table_lists_0_1_2_3(section):
    assert section, NO_SECTION
    rows = _table_rows(section, EXIT_STATUS_COLUMN)
    statuses = [_plain(row.get(EXIT_STATUS_COLUMN, "")) for row in rows]
    assert sorted(statuses) == sorted(EXIT_STATUSES), (
        f"the {EXIT_STATUS_COLUMN} table lists {statuses}, not one row per status {sorted(EXIT_STATUSES)}"
    )
