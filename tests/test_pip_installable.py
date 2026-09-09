#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pip-installability of the pipeline's declared dependencies.

Run with: pixi run test-specific tests/test_pip_installable.py

REQ-INSTALL-01: `champollion-utils` is not published on PyPI. Declaring it
as a bare name in `[project].dependencies` makes `pip install -e .` fail with
"Could not find a version that satisfies the requirement champollion-utils".
It must instead be declared as a PEP 508 direct reference pointing at the
neurospin/champollion_utils GitHub repository.
"""

from pathlib import Path

import pytest
import tomllib

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _project_dependencies() -> list[str]:
    """Return the raw `[project].dependencies` list from pyproject.toml."""
    with PYPROJECT.open("rb") as handle:
        data = tomllib.load(handle)
    return data["project"]["dependencies"]


def _champollion_utils_entry() -> str:
    """Return the single dependency entry that declares champollion-utils."""
    matches = [dep for dep in _project_dependencies() if "champollion" in dep and "utils" in dep]
    assert len(matches) == 1, f"expected exactly one champollion-utils entry in [project].dependencies, got {matches!r}"
    return matches[0]


class TestChampollionUtilsDependency:
    """`champollion-utils` must resolve without a PyPI lookup."""

    @pytest.mark.smoke
    def test_entry_is_not_a_bare_name(self):
        """The entry must not be the bare, PyPI-resolved name.

        A bare `champollion-utils` sends pip to PyPI, where the package does
        not exist, and `pip install -e .` aborts.
        """
        entry = _champollion_utils_entry()

        assert entry.strip() != "champollion-utils", (
            "[project].dependencies declares a bare 'champollion-utils'; "
            "champollion-utils is not on PyPI, so `pip install -e .` cannot resolve it. "
            "Use the PEP 508 direct reference form instead."
        )

    @pytest.mark.smoke
    def test_entry_uses_a_git_url(self):
        """The entry must resolve through a `git+` VCS URL, not an index lookup."""
        entry = _champollion_utils_entry()

        assert "git+" in entry, (
            f"champollion-utils entry {entry!r} does not use a 'git+' URL; "
            "it would still be resolved from a package index."
        )

    @pytest.mark.smoke
    def test_entry_uses_pep508_direct_reference_form(self):
        """The entry must keep the package name explicit: `name @ URL`."""
        entry = _champollion_utils_entry()

        name, separator, url = entry.partition("@")
        assert separator == "@", (
            f"champollion-utils entry {entry!r} is not in PEP 508 direct reference form "
            "('champollion-utils @ git+https://...'); the package name must stay explicit."
        )
        assert name.strip() == "champollion-utils", (
            f"expected the direct reference to be named 'champollion-utils', got {name.strip()!r}"
        )
        assert url.strip().startswith("git+"), f"expected a 'git+' URL after '@', got {url.strip()!r}"

    @pytest.mark.smoke
    def test_url_points_at_the_neurospin_champollion_utils_repo(self):
        """The `git+` URL must target github.com/neurospin/champollion_utils."""
        entry = _champollion_utils_entry()

        assert "github.com/neurospin/champollion_utils" in entry, (
            f"champollion-utils entry {entry!r} does not point at "
            "https://github.com/neurospin/champollion_utils; anyone cloning the "
            "pipeline from GitHub must be able to resolve it from there."
        )
