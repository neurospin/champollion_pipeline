#!/usr/bin/env python3
"""Documentation guards: README.md of the inference-side packages (TASK-DOCS M2).

Run by hand (``verify_ignore`` skips ``tests/test_docs_*.py``):
``pixi run pytest tests/test_docs_module_readmes_engine.py``.

Covered directories, under ``src/champollion_pipeline/``: ``embedding_engine/``, ``file_indexer/``,
``models/`` and ``streaming/``. Each test is parametrised over the four of them and reads the directory
listing at test time, so a file added later (``models/region_crops.py``) is checked once it exists.

The module-README format (shared with the top-level package README Task M1 and the utilities Task M3,
restated here by user decision): a level-1 title, then the level-2 sections ``Purpose``, ``Files``
(a table, one row per file), ``Entry points`` and ``See also`` (relative links to ``docs/*.md``).
The format helpers live in ``tests/docs_test_helpers.py`` (user decision 2026-11-23).

Requirements pinned here (one test each):

* REQ-DOCS-BDRABCZUK-B3DBAD3700E2 - level-1 title, then Purpose, Files, Entry points, See also in order.
* REQ-DOCS-BDRABCZUK-AB06E2C04501 - the Files table has one described row per file of the directory.
* REQ-DOCS-BDRABCZUK-496D0062E7BB - the Files table names no file the directory lacks.
* REQ-DOCS-BDRABCZUK-ADF294798416 - Entry points names a top-level function or class of the directory.
* REQ-DOCS-BDRABCZUK-71191FE398D7 - each backticked name in Entry points ends with a name defined there.
* REQ-DOCS-BDRABCZUK-9BE4A2EE7605 - See also links to existing pages under docs/ by relative paths.

The READMEs are read as plain text and the package sources parsed with ``ast``: no import of the
package, no filesystem writes, no network.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests.docs_test_helpers import (
    assert_module_readme_layout,
    md_section_lines,
    module_readme_files_rows,
    read_text,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "champollion_pipeline"
DOCS = ROOT / "docs"
DIRECTORIES = ["embedding_engine", "file_indexer", "models", "streaming"]

LINK_RE = re.compile(r"\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
BACKTICK_RE = re.compile(r"`([^`]+)`")

parametrise_directories = pytest.mark.parametrize("directory", DIRECTORIES)


def _readme(directory: str) -> str:
    return read_text(PACKAGE / directory / "README.md")


def _listed_files(directory: str) -> set[str]:
    """Names of the files of ``directory``, README.md and hidden files excepted."""
    return {
        p.name
        for p in (PACKAGE / directory).iterdir()
        if p.is_file() and p.name != "README.md" and not p.name.startswith(".")
    }


def _top_level_defs(directory: str) -> set[str]:
    """Names of the top-level functions and classes of the directory's Python files."""
    names: set[str] = set()
    for path in sorted((PACKAGE / directory).glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names.update(
            node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
    return names


def _modules(directory: str) -> set[str]:
    return {p.stem for p in (PACKAGE / directory).glob("*.py")}


def _last_segment(name: str) -> str:
    """Last dotted segment of a backticked name, a trailing call ``(...)`` dropped."""
    return re.sub(r"\(.*\)$", "", name.strip()).split(".")[-1]


def _entry_point_names(text: str) -> list[str]:
    return [m.group(1) for line in md_section_lines(text, "Entry points") for m in BACKTICK_RE.finditer(line)]


@parametrise_directories
def test_module_readme_has_title_and_sections(directory):
    assert_module_readme_layout(_readme(directory), directory)


@parametrise_directories
def test_module_readme_files_table_names_each_file(directory):
    rows = module_readme_files_rows(_readme(directory))
    named = [r[0] for r in rows]
    missing = sorted(_listed_files(directory) - set(named))
    assert not missing, f"{directory}/README.md Files table has no row for: {missing}"
    duplicated = sorted({n for n in named if named.count(n) > 1})
    assert not duplicated, f"{directory}/README.md Files table names more than once: {duplicated}"
    undescribed = sorted(r[0] for r in rows if len(r) < 2 or not r[1])
    assert not undescribed, f"{directory}/README.md Files table rows without a description: {undescribed}"


@parametrise_directories
def test_module_readme_files_table_has_no_stale_row(directory):
    named = {r[0] for r in module_readme_files_rows(_readme(directory))}
    stale = sorted(named - _listed_files(directory))
    assert not stale, f"{directory}/README.md Files table names files absent from the directory: {stale}"


@parametrise_directories
def test_module_readme_entry_points_named(directory):
    names = _entry_point_names(_readme(directory))
    defined = _top_level_defs(directory)
    hits = [n for n in names if _last_segment(n) in defined]
    assert hits, (
        f"{directory}/README.md Entry points names no top-level function or class of the directory "
        f"(backticked: {names})"
    )


@parametrise_directories
def test_module_readme_entry_points_exist(directory):
    names = _entry_point_names(_readme(directory))
    known = _top_level_defs(directory) | _modules(directory)
    unknown = sorted(n for n in names if _last_segment(n) not in known)
    assert not unknown, f"{directory}/README.md Entry points names what the directory does not define: {unknown}"


@parametrise_directories
def test_module_readme_see_also_links_resolve(directory):
    readme_dir = PACKAGE / directory
    targets = [m.group(1) for line in md_section_lines(_readme(directory), "See also") for m in LINK_RE.finditer(line)]
    relative = [t for t in targets if not re.match(r"^[a-z][a-z0-9+.-]*:", t) and not t.startswith("#")]
    assert relative, f"{directory}/README.md See also holds no relative link (links: {targets})"
    resolved = {t: (readme_dir / t.split("#", 1)[0]).resolve() for t in relative}
    broken = sorted(t for t, p in resolved.items() if not p.is_file())
    assert not broken, f"{directory}/README.md See also links resolve to no file: {broken}"
    docs_pages = [t for t, p in resolved.items() if p.suffix == ".md" and p.is_relative_to(DOCS.resolve())]
    assert docs_pages, f"{directory}/README.md See also links to no Markdown page under docs/: {relative}"
