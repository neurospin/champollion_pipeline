#!/usr/bin/env python3
"""Documentation guards: README.md of the top-level package and of resources/ (TASK-DOCS M1).

Run by hand (``verify_ignore`` skips ``tests/test_docs_*.py``):
``pixi run pytest tests/test_docs_module_readmes.py``.

Covered directories: ``src/champollion_pipeline/`` and ``src/champollion_pipeline/resources/``.
The tests read the directory listings at test time, so a file added later is checked once it exists.

This Task owns the module-README format (restated by M2 ``tests/test_docs_module_readmes_engine.py``
and M3 ``tests/test_docs_module_readmes_utils.py``, user decision REPLAN 2026-11-23): a level-1
title, then the level-2 sections ``Purpose``, ``Files`` (a table, one row per file), ``Entry points``
and ``See also`` (relative links to ``docs/*.md``). The format helpers live in
``tests/docs_test_helpers.py`` (user decision 2026-11-23).

Requirements pinned here (one test each):

* REQ-DOCS-BDRABCZUK-C6B2C25A9191 - level-1 title, then Purpose, Files, Entry points, See also in order.
* REQ-DOCS-BDRABCZUK-87666EAC8E4E - the Purpose section holds prose.
* REQ-DOCS-BDRABCZUK-F63AB845F830 - the Files table has one described row per file of the directory.
* REQ-DOCS-BDRABCZUK-CBC272250122 - the Files table names no file the directory lacks.
* REQ-DOCS-BDRABCZUK-D3EBBBA53C7C - package Entry points name each ``[project.scripts]`` console script.
* REQ-DOCS-BDRABCZUK-6E27DE7139E0 - package Entry points name ``python -m`` for each ``__main__`` module.
* REQ-DOCS-BDRABCZUK-0CCBF6E67D56 - package Entry points name no undeclared console script.
* REQ-DOCS-BDRABCZUK-34738BC824FA - package Entry points name no ``python -m`` module without a guard.
* REQ-DOCS-BDRABCZUK-976CCFF7462E - resources Entry points state there is no console script.
* REQ-DOCS-BDRABCZUK-55BED3BD37E9 - resources Entry points name a reader module per data file.
* REQ-DOCS-BDRABCZUK-11AF08B8CA60 - See also links a Markdown page under docs/ by a relative path.
* REQ-DOCS-BDRABCZUK-901AF795D8F3 - each relative link of the README resolves to a repository file.

The READMEs are read as plain text; the code facts come from pyproject.toml and the source text.
No import of the package, no filesystem writes, no network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tomllib

from tests.docs_test_helpers import (
    MD_HEADING_RE,
    assert_module_readme_layout,
    md_prose_lines,
    md_section_lines,
    module_readme_files_rows,
    read_text,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "champollion_pipeline"
RESOURCES = PACKAGE / "resources"
DOCS = ROOT / "docs"
DIRECTORIES = {"champollion_pipeline": PACKAGE, "resources": RESOURCES}

LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
BACKTICK_RE = re.compile(r"`([^`]+)`")
MAIN_GUARD_RE = re.compile(r"""^if\s+__name__\s*==\s*["']__main__["']\s*:""", re.MULTILINE)
SCRIPT_NAME_RE = re.compile(r"(?<![\w-])(champollion-[a-z0-9-]+)(?![\w-])")
PYTHON_M_RE = re.compile(r"python3?\s+-m\s+champollion_pipeline\.([A-Za-z_][\w.]*)")
URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")

parametrise_directories = pytest.mark.parametrize("key", sorted(DIRECTORIES))


def _readme(key: str) -> str:
    return read_text(DIRECTORIES[key] / "README.md")


def _listed_files(directory: Path) -> set[str]:
    """Names of the regular files of ``directory``, README.md, hidden files and bytecode excepted."""
    return {
        p.name
        for p in directory.iterdir()
        if p.is_file() and p.name != "README.md" and not p.name.startswith(".") and p.suffix not in {".pyc", ".pyo"}
    }


def _console_scripts() -> dict[str, str]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["scripts"]


def _main_modules() -> set[str]:
    """Stems of the top-level package's Python files that hold a ``__main__`` guard."""
    return {p.stem for p in PACKAGE.glob("*.py") if MAIN_GUARD_RE.search(p.read_text(encoding="utf-8"))}


def _relative_links(lines: list[str]) -> list[str]:
    """Link targets that are paths (no URL scheme, no pure in-page anchor), anchors stripped."""
    targets = [m.group(1) for line in lines for m in LINK_RE.finditer(line)]
    return [t.split("#", 1)[0] for t in targets if not URL_RE.match(t) and not t.startswith("#")]


@parametrise_directories
def test_module_readme_has_title_and_sections(key):
    """REQ-DOCS-BDRABCZUK-C6B2C25A9191: a level-1 title, then the four level-2 sections in order."""
    assert_module_readme_layout(_readme(key), key)


@parametrise_directories
def test_module_readme_purpose_not_empty(key):
    """REQ-DOCS-BDRABCZUK-87666EAC8E4E: the Purpose section holds one or more lines of prose."""
    prose = [
        line for line in md_section_lines(_readme(key), "Purpose") if line.strip() and not MD_HEADING_RE.match(line)
    ]
    assert prose, f"{key}/README.md Purpose section is empty"


@parametrise_directories
def test_module_readme_files_table_names_each_file(key):
    """REQ-DOCS-BDRABCZUK-F63AB845F830: one described Files row per file of the directory."""
    rows = module_readme_files_rows(_readme(key))
    named = [r[0] for r in rows]
    missing = sorted(_listed_files(DIRECTORIES[key]) - set(named))
    assert not missing, f"{key}/README.md Files table has no row for: {missing}"
    duplicated = sorted({n for n in named if named.count(n) > 1})
    assert not duplicated, f"{key}/README.md Files table names more than once: {duplicated}"
    undescribed = sorted(r[0] for r in rows if len(r) < 2 or not r[1])
    assert not undescribed, f"{key}/README.md Files table rows without a description: {undescribed}"


@parametrise_directories
def test_module_readme_files_table_has_no_stale_row(key):
    """REQ-DOCS-BDRABCZUK-CBC272250122: no Files row for a file the directory lacks."""
    named = {r[0] for r in module_readme_files_rows(_readme(key))}
    stale = sorted(named - _listed_files(DIRECTORIES[key]))
    assert not stale, f"{key}/README.md Files table names files absent from the directory: {stale}"


def _package_entry_points() -> str:
    return "\n".join(md_section_lines(_readme("champollion_pipeline"), "Entry points", prose_only=False))


def test_package_readme_names_each_console_script():
    """REQ-DOCS-BDRABCZUK-D3EBBBA53C7C: Entry points name each console script of pyproject.toml."""
    named = set(SCRIPT_NAME_RE.findall(_package_entry_points()))
    missing = sorted(set(_console_scripts()) - named)
    assert not missing, f"champollion_pipeline/README.md Entry points do not name: {missing}"


def test_package_readme_names_each_main_module():
    """REQ-DOCS-BDRABCZUK-6E27DE7139E0: Entry points name python -m for each module with a __main__ guard."""
    named = set(PYTHON_M_RE.findall(_package_entry_points()))
    missing = sorted(_main_modules() - named)
    assert not missing, (
        f"champollion_pipeline/README.md Entry points name no 'python -m champollion_pipeline.<module>' for: {missing}"
    )


def test_package_readme_names_no_unknown_console_script():
    """REQ-DOCS-BDRABCZUK-0CCBF6E67D56: Entry points name no console script pyproject.toml lacks."""
    section = _package_entry_points()
    assert SCRIPT_NAME_RE.search(section), "champollion_pipeline/README.md Entry points name no console script"
    unknown = sorted(set(SCRIPT_NAME_RE.findall(section)) - set(_console_scripts()))
    assert not unknown, f"champollion_pipeline/README.md Entry points name undeclared console scripts: {unknown}"


def test_package_readme_names_no_unguarded_main_module():
    """REQ-DOCS-BDRABCZUK-34738BC824FA: Entry points name no python -m module lacking a __main__ guard."""
    named = set(PYTHON_M_RE.findall(_package_entry_points()))
    assert named, "champollion_pipeline/README.md Entry points name no python -m module"
    unguarded = sorted(named - _main_modules())
    assert not unguarded, (
        f"champollion_pipeline/README.md Entry points name modules without a __main__ guard: {unguarded}"
    )


def _resources_entry_points() -> str:
    return "\n".join(md_section_lines(_readme("resources"), "Entry points"))


def test_resources_readme_states_no_console_script():
    """REQ-DOCS-BDRABCZUK-976CCFF7462E: resources Entry points state the directory defines no console script."""
    section = _resources_entry_points()
    assert re.search(r"no console script", section, re.IGNORECASE), (
        "resources/README.md Entry points do not say 'no console script'"
    )
    # The README's claim, checked against the code.
    assert not list(RESOURCES.glob("*.py")), "resources/ holds Python files"
    into = [name for name, ref in _console_scripts().items() if ref.startswith("champollion_pipeline.resources")]
    assert into == [], f"console scripts pointing into resources: {into}"


def test_resources_readme_names_reader_of_each_file():
    """REQ-DOCS-BDRABCZUK-55BED3BD37E9: per data file, Entry points name a package module whose source names it."""
    names = {m.group(1) for line in _resources_entry_points().splitlines() for m in BACKTICK_RE.finditer(line)}
    named_stems = {re.sub(r"\.py$", "", n.strip()).split(".")[-1] for n in names}
    sources = {p.stem: p.read_text(encoding="utf-8") for p in sorted(PACKAGE.glob("*.py"))}
    unread = []
    for data_file in sorted(_listed_files(RESOURCES)):
        readers = {stem for stem, text in sources.items() if data_file in text}
        if not readers & named_stems:
            unread.append(f"{data_file} (readers: {sorted(readers)})")
    assert not unread, f"resources/README.md Entry points name no reader module for: {unread}"


@parametrise_directories
def test_module_readme_see_also_links_docs_page(key):
    """REQ-DOCS-BDRABCZUK-11AF08B8CA60: See also links a Markdown page under docs/ by a relative path."""
    directory = DIRECTORIES[key]
    targets = _relative_links(md_section_lines(_readme(key), "See also"))
    pages = [
        t
        for t in targets
        if t.endswith(".md")
        and (directory / t).resolve().is_relative_to(DOCS.resolve())
        and (directory / t).resolve().is_file()
    ]
    assert pages, f"{key}/README.md See also holds no relative link to a docs/*.md page (links: {targets})"


@parametrise_directories
def test_module_readme_relative_links_resolve(key):
    """REQ-DOCS-BDRABCZUK-901AF795D8F3: each relative link of the README names an existing repository file."""
    directory = DIRECTORIES[key]
    targets = [t for t in _relative_links(md_prose_lines(_readme(key))) if t]
    broken = sorted(t for t in targets if not (directory / t).resolve().is_file())
    assert not broken, f"{key}/README.md relative links naming no existing file: {broken}"
    outside = sorted(t for t in targets if not (directory / t).resolve().is_relative_to(ROOT))
    assert not outside, f"{key}/README.md relative links leaving the repository: {outside}"
