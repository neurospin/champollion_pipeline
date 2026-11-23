#!/usr/bin/env python3
"""Documentation guards: module READMEs of snapshot_types, utils and compare_sulcal (TASK-DOCS M3).

Run by hand (``verify_ignore`` skips ``tests/test_docs_*.py``):
``pixi run pytest tests/test_docs_module_readmes_utils.py``.

The module README format (title, Purpose, Files, Entry points, See also) is owned by
TASK-DOCS-BDRABCZUK-8B4AD3315AFA (M1, ``tests/test_docs_module_readmes.py``); this Task restates it
in its own Requirements and test (user decision, REPLAN 2026-11-23: parallelise the docs).

Requirements pinned here (one test each):

* REQ-DOCS-BDRABCZUK-C72349F4FA44 - each README starts with a level-1 heading naming its package.
* REQ-DOCS-BDRABCZUK-D9D2BE7E723F - each README has a non-empty ``## Purpose`` section.
* REQ-DOCS-BDRABCZUK-070EA00F89FF - each README's ``## Files`` table has exactly one row per file of
  the directory other than README.md, with the file name and a one-line description.
* REQ-DOCS-BDRABCZUK-57C44C198979 - each README's ``## See also`` section links a page under docs/.
* REQ-DOCS-BDRABCZUK-3F37EA21F479 - each relative link of each README names an existing file.
* REQ-DOCS-BDRABCZUK-D59B22AA058B - snapshot_types' Entry points name ``champollion-snapshots``.
* REQ-DOCS-BDRABCZUK-595F3486FBD3 - utils' Entry points state the package has no console script.
* REQ-DOCS-BDRABCZUK-9C564B93B94E - compare_sulcal's Entry points name ``src/compare.py`` and each
  subcommand of the ``Compare`` parser.

The READMEs are read as plain text; the code facts they state are read from pyproject.toml, the
sources and the built (never parsed) ``Compare`` parser. No filesystem writes, no network.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest
import tomllib

from compare_sulcal.cli import Compare
from tests.docs_test_helpers import read_text

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DOCS = ROOT / "docs"

# directory key -> (directory, package name the title must carry)
MODULE_DIRS: dict[str, tuple[Path, str]] = {
    "snapshot_types": (SRC / "champollion_pipeline" / "snapshot_types", "snapshot_types"),
    "utils": (SRC / "champollion_pipeline" / "utils", "utils"),
    "compare_sulcal": (SRC / "compare_sulcal", "compare_sulcal"),
}

# [text](target) with an optional "title"; images and autolinks are irrelevant here.
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
MAIN_GUARD_RE = re.compile(r"""^if\s+__name__\s*==\s*["']__main__["']\s*:""", re.MULTILINE)


def _readme(key: str) -> str:
    directory, _ = MODULE_DIRS[key]
    return read_text(directory / "README.md")


def _section(text: str, heading: str) -> str:
    """Body of the ``## <heading>`` section (to the next level-2 heading or the end)."""
    m = re.search(rf"^##\s+{re.escape(heading)}\s*$(.*?)(?=^##\s|\Z)", text, re.MULTILINE | re.DOTALL)
    assert m, f"no '## {heading}' section found"
    return m.group(1)


def _strip_code_blocks(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def _directory_files(directory: Path) -> set[str]:
    """Regular files of ``directory`` a README must list: everything but README.md, caches, hidden files."""
    return {
        p.name
        for p in directory.iterdir()
        if p.is_file() and p.name != "README.md" and not p.name.startswith(".") and p.suffix not in {".pyc", ".pyo"}
    }


def _table_rows(section: str) -> list[list[str]]:
    """Cells of each body row of the first Markdown table in ``section`` (header and separator dropped)."""
    rows = [line.strip() for line in section.splitlines() if line.strip().startswith("|")]
    assert len(rows) >= 2, "expected a Markdown table (header and separator rows)"
    assert re.fullmatch(r"\|(\s*:?-{3,}:?\s*\|)+", rows[1]), f"second table line is not a separator: {rows[1]}"
    return [[cell.strip() for cell in row.strip("|").split("|")] for row in rows[2:]]


def _relative_links(text: str) -> list[str]:
    """Link targets that are paths in the repository (no URL scheme, no pure in-page anchor)."""
    targets = []
    for target in LINK_RE.findall(_strip_code_blocks(text)):
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
            continue
        targets.append(target.split("#", 1)[0])
    return targets


def _console_scripts() -> dict[str, str]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["scripts"]


def _modules_with_main_guard(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.glob("*.py") if MAIN_GUARD_RE.search(p.read_text(encoding="utf-8")))


@pytest.mark.parametrize("key", sorted(MODULE_DIRS))
def test_module_readme_title(key: str) -> None:
    """REQ-DOCS-BDRABCZUK-C72349F4FA44: first non-empty line is a level-1 heading naming the package."""
    _, package = MODULE_DIRS[key]
    first = next(line for line in _readme(key).splitlines() if line.strip())
    assert re.match(r"^#\s+\S", first), f"first line is not a level-1 heading: {first!r}"
    assert package in first, f"title {first!r} does not name the package {package!r}"


@pytest.mark.parametrize("key", sorted(MODULE_DIRS))
def test_module_readme_purpose(key: str) -> None:
    """REQ-DOCS-BDRABCZUK-D9D2BE7E723F: a non-empty ``## Purpose`` section."""
    body = _section(_readme(key), "Purpose").strip()
    assert body, "the Purpose section is empty"


@pytest.mark.parametrize("key", sorted(MODULE_DIRS))
def test_module_readme_files_table(key: str) -> None:
    """REQ-DOCS-BDRABCZUK-070EA00F89FF: one ``## Files`` row per file, with a name and a description."""
    directory, _ = MODULE_DIRS[key]
    rows = _table_rows(_section(_readme(key), "Files"))
    named = []
    for cells in rows:
        assert len(cells) >= 2, f"row {cells} has no description column"
        m = re.fullmatch(r"`([^`]+)`", cells[0])
        assert m, f"first cell {cells[0]!r} is not a file name in backticks"
        assert cells[1], f"row for {m.group(1)} has an empty description"
        assert "\n" not in cells[1]
        named.append(m.group(1))
    duplicates = sorted({name for name in named if named.count(name) > 1})
    assert not duplicates, f"files named in more than one row: {duplicates}"
    expected = _directory_files(directory)
    assert set(named) == expected, (
        f"missing rows: {sorted(expected - set(named))}; rows for absent files: {sorted(set(named) - expected)}"
    )


@pytest.mark.parametrize("key", sorted(MODULE_DIRS))
def test_module_readme_see_also_links_docs(key: str) -> None:
    """REQ-DOCS-BDRABCZUK-57C44C198979: ``## See also`` links a Markdown page under docs/ by a relative link."""
    directory, _ = MODULE_DIRS[key]
    section = _section(_readme(key), "See also")
    pages = [
        (directory / target).resolve()
        for target in _relative_links(section)
        if target.endswith(".md") and (directory / target).resolve().parent == DOCS.resolve()
    ]
    assert pages, "the See also section has no relative link to a docs/*.md page"


@pytest.mark.parametrize("key", sorted(MODULE_DIRS))
def test_module_readme_relative_links_resolve(key: str) -> None:
    """REQ-DOCS-BDRABCZUK-3F37EA21F479: each relative link target names an existing repository file."""
    directory, _ = MODULE_DIRS[key]
    targets = _relative_links(_readme(key))
    broken = [t for t in targets if not (directory / t).resolve().is_file()]
    assert not broken, f"relative links naming no existing file: {broken}"
    outside = [t for t in targets if not (directory / t).resolve().is_relative_to(ROOT)]
    assert not outside, f"relative links leaving the repository: {outside}"


def test_snapshot_types_readme_entry_points() -> None:
    """REQ-DOCS-BDRABCZUK-D59B22AA058B: Entry points name champollion-snapshots, which runs the package."""
    directory, _ = MODULE_DIRS["snapshot_types"]
    section = _section(_readme("snapshot_types"), "Entry points")
    assert "champollion-snapshots" in section, "Entry points do not name champollion-snapshots"
    # The README's claim, checked against the code.
    assert _console_scripts()["champollion-snapshots"] == "champollion_pipeline.generate_snapshots:main"
    generator = read_text(SRC / "champollion_pipeline" / "generate_snapshots.py")
    assert "snapshot_types" in generator, "generate_snapshots no longer imports snapshot_types"
    assert _modules_with_main_guard(directory) == []


def test_utils_readme_entry_points() -> None:
    """REQ-DOCS-BDRABCZUK-595F3486FBD3: Entry points state that utils defines no console script."""
    directory, _ = MODULE_DIRS["utils"]
    section = _section(_readme("utils"), "Entry points")
    assert re.search(r"no console script", section, re.IGNORECASE), "Entry points do not say 'no console script'"
    # The README's claim, checked against the code.
    into_utils = [name for name, ref in _console_scripts().items() if ref.startswith("champollion_pipeline.utils")]
    assert into_utils == [], f"console scripts pointing into utils: {into_utils}"
    assert _modules_with_main_guard(directory) == []


def test_compare_sulcal_readme_entry_points() -> None:
    """REQ-DOCS-BDRABCZUK-9C564B93B94E: Entry points name src/compare.py and each Compare subcommand."""
    section = _section(_readme("compare_sulcal"), "Entry points")
    assert "src/compare.py" in section, "Entry points do not name src/compare.py"
    actions = [a for a in Compare().parser._actions if isinstance(a, argparse._SubParsersAction)]
    assert len(actions) == 1, "expected one subcommand group on the Compare parser"
    subcommands = sorted(actions[0].choices)
    assert subcommands, "the Compare parser declares no subcommand"
    missing = [name for name in subcommands if not re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", section)]
    assert not missing, f"subcommands not named in Entry points: {missing}"
