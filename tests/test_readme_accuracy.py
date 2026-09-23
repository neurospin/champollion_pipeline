#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""README accuracy guards for REQ-DOCS-05, REQ-DOCS-06, and REQ-DOCS-09.

Joël asked (FEEDBACK-JOEL-G1) for ``README.md`` to reflect the pipeline's
current architecture and naming. Two independent, mechanically checkable
claims come out of that:

* **REQ-DOCS-05** — the ``Project Structure`` tree is the README's only
  architectural map of the installable package, so it must list every module
  that actually ships in ``src/champollion_pipeline/``.
* **REQ-DOCS-06** — the ``champollion-*`` console scripts declared in
  ``pyproject.toml``'s ``[project.scripts]`` are the pipeline's public command
  names, so every one of them must be named in the README.

Julien separately flagged (FEEDBACK-JULIEN-G1) that the README's "Project
website" link is dead (404) while the paper is under journal editorial
review, so an editor may click through from GitHub right now:

* **REQ-DOCS-09** — README.md must not contain the dead project-website URL.

Both artifacts are read as plain text: no import of the package, no Sphinx
build, no network, and no checked-out ``external/`` submodule is required.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

README = PROJECT_ROOT / "README.md"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
PKG_DIR = PROJECT_ROOT / "src" / "champollion_pipeline"


def _read(path: Path) -> str:
    assert path.is_file(), f"expected file to exist: {path}"
    return path.read_text(encoding="utf-8")


def _shipped_modules() -> list[str]:
    """Every ``*.py`` module in the installable package except ``__init__.py``.

    Sub-packages (e.g. ``utils/``) are deliberately excluded: REQ-DOCS-05
    constrains the top-level module files only.
    """
    return sorted(p.name for p in PKG_DIR.glob("*.py") if p.name != "__init__.py")


def _declared_console_scripts() -> list[str]:
    """Keys of ``[project.scripts]`` in ``pyproject.toml``.

    Parsed as text rather than via ``tomllib`` so the test behaves identically
    on every interpreter the project supports.
    """
    text = _read(PYPROJECT)
    section = re.search(r"^\[project\.scripts\]\s*$(.*?)(?=^\[|\Z)", text, re.MULTILINE | re.DOTALL)
    assert section, "no [project.scripts] table found in pyproject.toml"
    names = re.findall(r"^([A-Za-z0-9_.-]+)\s*=", section.group(1), re.MULTILINE)
    assert names, "[project.scripts] declares no entry points"
    return sorted(names)


def _project_structure_block() -> str:
    """The fenced code block that follows the ``## Project Structure`` heading."""
    text = _read(README)
    after = re.search(r"^##\s+Project Structure\s*$(.*?)(?=^##\s)", text, re.MULTILINE | re.DOTALL)
    assert after, "README.md has no '## Project Structure' section"
    block = re.search(r"```[^\n]*\n(.*?)```", after.group(1), re.DOTALL)
    assert block, "the 'Project Structure' section contains no fenced tree block"
    return block.group(1)


@pytest.mark.smoke
class TestProjectStructureTree:
    """REQ-DOCS-05: the tree lists every shipped pipeline module."""

    def test_tree_block_is_present(self):
        """The README still has a fenced tree under '## Project Structure'."""
        assert _project_structure_block().strip(), "the Project Structure tree block is empty"

    @pytest.mark.parametrize("module", _shipped_modules())
    def test_module_is_listed_in_tree(self, module):
        """Each ``src/champollion_pipeline/<module>`` appears as a tree entry."""
        lines = [ln for ln in _project_structure_block().splitlines() if module in ln]
        assert lines, (
            f"{module} ships in src/champollion_pipeline/ but is not listed in the README's Project Structure tree"
        )


@pytest.mark.smoke
class TestConsoleScriptNames:
    """REQ-DOCS-06: the README names every declared console script."""

    @pytest.mark.parametrize("script", _declared_console_scripts())
    def test_console_script_is_named_in_readme(self, script):
        """Each ``[project.scripts]`` key occurs verbatim somewhere in the README."""
        assert script in _read(README), (
            f"{script} is declared in pyproject.toml's [project.scripts] but is never named in README.md"
        )


@pytest.mark.smoke
class TestProjectWebsiteLink:
    """REQ-DOCS-09: the README does not link to the dead project website."""

    def test_readme_does_not_contain_dead_project_website_url(self):
        """``neurospin.fr/champollion_pipeline`` (404) must not appear in the README."""
        assert "neurospin.fr/champollion_pipeline" not in _read(README), (
            "README.md still contains the dead project-website URL "
            "'https://www.neurospin.fr/champollion_pipeline' (404)"
        )
