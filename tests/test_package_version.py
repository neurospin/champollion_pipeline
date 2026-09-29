#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package metadata in `champollion_pipeline/__init__.py` agrees with the manifests.

Run with: pixi run test-specific tests/test_package_version.py

REQ-INSTALL-07: every package fact stated in
`src/champollion_pipeline/__init__.py` (its `__version__` value and each
dependency its comments name) shall agree with the project manifests:
`__version__` equals `pyproject.toml`'s `[project] version`, and each named
dependency appears in `pyproject.toml` or `pixi.toml`.

The version check reads `__version__` from the imported module at runtime, so
both a literal string and a value derived via `importlib.metadata` satisfy it.
"""

import io
import re
import tokenize
from pathlib import Path

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
PIXI_TOML = REPO_ROOT / "pixi.toml"
INIT_PY = REPO_ROOT / "src" / "champollion_pipeline" / "__init__.py"

# A parenthesised, comma-separated list following the word "dependencies" in a
# comment, e.g. "optional dependencies (brainvisa, torch…)".
_DEPENDENCY_LIST = re.compile(r"dependenc\w*\s*\(([^)]*)\)", re.IGNORECASE)


def _pyproject_version() -> str:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def _comment_text(path: Path) -> str:
    """Return every `#` comment in a Python file, joined by newlines."""
    source = path.read_text(encoding="utf-8")
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return "\n".join(tok.string for tok in tokens if tok.type == tokenize.COMMENT)


def _dependencies_named_in_comments(path: Path) -> list[str]:
    """Return every dependency name listed in a "dependencies (...)" comment."""
    # Join comment lines so a list wrapped across two lines is still matched.
    text = " ".join(line.lstrip("#").strip() for line in _comment_text(path).splitlines())
    names = []
    for match in _DEPENDENCY_LIST.finditer(text):
        for raw in match.group(1).split(","):
            name = raw.strip().strip("…").rstrip(".").strip()
            if name:
                names.append(name)
    return names


@pytest.mark.smoke
class TestPackageMetadataMatchesManifests:
    """REQ-INSTALL-07: `__init__.py` states no package fact the manifests contradict."""

    def test_version_matches_pyproject(self):
        import champollion_pipeline

        expected = _pyproject_version()
        assert champollion_pipeline.__version__ == expected, (
            f"champollion_pipeline.__version__ is {champollion_pipeline.__version__!r}, "
            f"but pyproject.toml declares version = {expected!r}"
        )

    def test_dependencies_named_in_comments_are_declared_in_manifests(self):
        manifests = (PYPROJECT.read_text(encoding="utf-8") + "\n" + PIXI_TOML.read_text(encoding="utf-8")).lower()
        named = _dependencies_named_in_comments(INIT_PY)
        undeclared = [name for name in named if not re.search(rf"\b{re.escape(name.lower())}\b", manifests)]
        assert not undeclared, (
            f"{INIT_PY.relative_to(REPO_ROOT)} comments name dependencies {undeclared} "
            "that appear in neither pyproject.toml nor pixi.toml"
        )
