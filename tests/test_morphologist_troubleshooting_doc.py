#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for REQ-DOCS-04 — Morphologist BIDS-filename crash troubleshooting entry.

Run with: pixi run test-specific tests/test_morphologist_troubleshooting_doc.py

``morphologist-cli`` auto-detects the BIDS input format from a filename that
carries BIDS entities (``_acq-``, ``_run-``, ``_ses-``), silently overrides the
``--if morphologist-auto-nonoverlap-1.0`` flag, and then fails with
``RuntimeError: the parameter input is not readable or does not exist``.
REQ-DOCS-04 states that ``docs/troubleshooting.md`` shall document that crash,
naming ``champollion-morphologist`` as the fix and the explicit
``--if morphologist-auto-nonoverlap-1.0`` flag as the workaround for callers
who must invoke ``morphologist-cli`` directly.

The artifact under test is the documentation page itself; these are regression
guards over content already committed in f12b1ee, so they are expected to pass
and to turn red only if that section is later removed or reworded past
recognition. Each test pins down one distinct acquired behaviour, so a partial
edit reads as a partial failure rather than an all-or-nothing one.

Assertions are scoped to the crash section's body — sliced from its heading up
to the next heading of equal or higher level — so unrelated prose elsewhere on
the page cannot satisfy them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TROUBLESHOOTING_MD = PROJECT_ROOT / "docs" / "troubleshooting.md"

# A heading introducing the Morphologist crash entry. Matched loosely on
# wording so the requirement constrains the content of the section, not the
# exact title casing or phrasing.
CRASH_HEADING_RE = re.compile(
    r"^#{1,6}\s+.*morphologist.*$",
    re.IGNORECASE | re.MULTILINE,
)

# Any of these identifies the failure mode being documented: a BIDS entity
# suffix, the word BIDS itself, or the verbatim error text users will paste
# into a search box.
CRASH_MARKERS = ("_acq-", "BIDS", "parameter input is not readable")


def _read_troubleshooting_md() -> str:
    """Contents of ``docs/troubleshooting.md``, or ``""`` when it is missing.

    Returning empty rather than raising keeps each test's own assertion as the
    reported failure, so a deleted page reads as a plain red test instead of an
    unrelated ``FileNotFoundError``.
    """
    return TROUBLESHOOTING_MD.read_text(encoding="utf-8") if TROUBLESHOOTING_MD.is_file() else ""


def _crash_section() -> str:
    """The Morphologist crash section's body, or ``""`` when absent.

    Includes the heading line itself, since the error string that identifies
    the entry is part of the heading in the committed page.
    """
    text = _read_troubleshooting_md()
    match = CRASH_HEADING_RE.search(text)
    if match is None:
        return ""

    heading = match.group(0)
    level = len(heading) - len(heading.lstrip("#"))
    rest = text[match.end() :]

    next_heading = re.search(rf"^#{{1,{level}}}\s+", rest, re.MULTILINE)
    body = rest[: next_heading.start()] if next_heading else rest
    return heading + body


@pytest.mark.smoke
class TestMorphologistBidsCrashSection:
    """REQ-DOCS-04: troubleshooting.md documents the BIDS-filename crash."""

    def test_page_has_a_morphologist_crash_section(self):
        """A heading introduces the Morphologist crash entry."""
        text = _read_troubleshooting_md()

        assert CRASH_HEADING_RE.search(text) is not None, (
            f"{TROUBLESHOOTING_MD} has no heading mentioning Morphologist; the "
            "BIDS-filename crash entry is missing."
        )

    def test_section_identifies_the_bids_filename_failure_mode(self):
        """The section names the trigger: a BIDS entity, or the error text."""
        section = _crash_section()

        assert any(marker in section for marker in CRASH_MARKERS), (
            f"{TROUBLESHOOTING_MD}'s Morphologist section mentions none of "
            f"{CRASH_MARKERS!r}; a user hitting the crash cannot recognise it as "
            "the BIDS-filename failure mode."
        )

    def test_section_names_the_champollion_wrapper_as_the_fix(self):
        """``champollion-morphologist`` is offered as the fix."""
        section = _crash_section()

        assert "champollion-morphologist" in section, (
            f"{TROUBLESHOOTING_MD}'s Morphologist section does not name "
            "'champollion-morphologist'; the recommended fix (use the wrapper "
            "instead of raw morphologist-cli) is undocumented."
        )

    def test_section_gives_the_explicit_input_format_flag_workaround(self):
        """``--if morphologist-auto-nonoverlap-1.0`` is given as the workaround."""
        section = _crash_section()

        assert "--if morphologist-auto-nonoverlap-1.0" in section, (
            f"{TROUBLESHOOTING_MD}'s Morphologist section does not show "
            "'--if morphologist-auto-nonoverlap-1.0'; callers who must invoke "
            "morphologist-cli directly have no documented workaround."
        )

    def test_section_quotes_the_runtimeerror_as_context(self):
        """``RuntimeError`` appears, so the entry is searchable by the traceback."""
        section = _crash_section()

        assert "RuntimeError" in section, (
            f"{TROUBLESHOOTING_MD}'s Morphologist section does not mention "
            "'RuntimeError'; a user searching the page for their traceback will "
            "not land on this entry."
        )
