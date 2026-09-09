#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for REQ-INSTALL-02 — virtual-environment installation documentation.

Run with: pixi run test-specific tests/test_venv_install_docs.py

``pixi run install-all`` is the supported install path, but users who already
manage their own virtual environment (venv, virtualenv, uv) need a documented
pip-only route. REQ-INSTALL-02 states that ``docs/installation.md`` shall carry
a "Virtual-environment installation" section documenting, in order,
``pip install -e .``, ``pip install -e external/cortical_tiles`` after
submodule initialization, and a note that BrainVISA/Morphologist requires conda
and cannot be pip-installed.

The artifact under test is the documentation page itself. Each test pins down
one distinct acquired behaviour of that section so a partial edit reads as a
partial pass rather than an all-or-nothing failure.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTALLATION_MD = PROJECT_ROOT / "docs" / "installation.md"

# A heading introducing the pip-into-a-virtual-environment route. Matched
# loosely on wording so the requirement constrains the content, not the exact
# title casing.
VENV_HEADING_RE = re.compile(
    r"^#{1,6}\s+.*(virtual[\s-]?env(ironment)?|venv|pip install).*$",
    re.IGNORECASE | re.MULTILINE,
)


def _read_installation_md() -> str:
    """Contents of ``docs/installation.md``, or ``""`` when it is missing.

    Returning empty rather than raising keeps each test's own assertion as the
    reported failure, so a missing page reads as a plain red test instead of an
    unrelated ``FileNotFoundError``.
    """
    return INSTALLATION_MD.read_text(encoding="utf-8") if INSTALLATION_MD.is_file() else ""


def _venv_section() -> str:
    """The virtual-environment section's body, or ``""`` when absent.

    Slices from the venv heading up to the next heading of the same or a
    higher level, so assertions below are scoped to that section and cannot be
    satisfied accidentally by unrelated prose elsewhere on the page.
    """
    text = _read_installation_md()
    match = VENV_HEADING_RE.search(text)
    if match is None:
        return ""

    heading = match.group(0)
    level = len(heading) - len(heading.lstrip("#"))
    rest = text[match.end() :]

    next_heading = re.search(rf"^#{{1,{level}}}\s+", rest, re.MULTILINE)
    return rest[: next_heading.start()] if next_heading else rest


@pytest.mark.smoke
class TestVirtualEnvironmentInstallSection:
    """REQ-INSTALL-02: ``docs/installation.md`` documents the pip-only route."""

    def test_page_has_a_virtual_environment_section(self):
        """A heading introduces the virtual-environment / pip install route."""
        text = _read_installation_md()

        assert VENV_HEADING_RE.search(text) is not None, (
            f"{INSTALLATION_MD} has no heading covering virtual-environment / venv / "
            "pip installation; users managing their own environment have no documented "
            "install path."
        )

    def test_section_documents_editable_install_of_the_pipeline(self):
        """The section shows ``pip install -e .`` for the pipeline itself."""
        section = _venv_section()

        assert "pip install -e ." in section, (
            "the virtual-environment section of "
            f"{INSTALLATION_MD} does not show `pip install -e .`; the first step of the "
            "pip-only route is installing the pipeline in editable mode."
        )

    def test_section_documents_editable_install_of_cortical_tiles(self):
        """The section shows the editable install of ``external/cortical_tiles``."""
        section = _venv_section()

        assert "external/cortical_tiles" in section, (
            "the virtual-environment section of "
            f"{INSTALLATION_MD} does not mention `external/cortical_tiles`; the "
            "cortical-tiles submodule must be installed in editable mode too, after "
            "submodule initialization."
        )

    def test_section_warns_that_brainvisa_cannot_be_pip_installed(self):
        """The section states BrainVISA/Morphologist needs conda, not pip."""
        section = _venv_section().lower()

        mentions_brainvisa = "brainvisa" in section or "morphologist" in section
        assert mentions_brainvisa, (
            "the virtual-environment section of "
            f"{INSTALLATION_MD} does not mention BrainVISA/Morphologist; readers "
            "following the pip-only route must be told it does not cover step 1."
        )

        explains_the_limit = "conda" in section or "pip" in section
        assert explains_the_limit, (
            "the virtual-environment section of "
            f"{INSTALLATION_MD} mentions BrainVISA/Morphologist but does not say it "
            "requires conda and cannot be pip-installed."
        )
