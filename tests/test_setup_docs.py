"""Tests for REQ-WIZARD-04 — the setup decision-tree documentation page.

# TASK-010

The artifact under test is ``docs/setup.md`` itself: the requirement states
that the page shall contain a Mermaid flowchart mapping location x use-case to
the correct pixi command, plus a summary table.

Each acquired behaviour gets its own test so a partially written page reports
precisely which element is still missing — a page with a Mermaid fence but no
``flowchart`` declaration, or a flowchart with no accompanying summary table,
is a different defect from a page that does not exist at all.

The two coverage tests (``training``, Jean-Zay) pin down the *content* of the
mapping rather than its shape: the page exists to route a reader to the right
environment, so it is only correct if it mentions the ``training``
environment introduced by REQ-WIZARD-01 and the Jean-Zay cluster, which is the
"location" axis the flowchart branches on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
SETUP_MD = DOCS_DIR / "setup.md"


def _read(path: Path) -> str:
    """Contents of ``path``, or an empty string when it does not exist yet.

    Returning empty (rather than raising) keeps each test's own assertion as
    the reported failure, so a missing page reads as a plain red test rather
    than an unrelated ``FileNotFoundError``.
    """
    return path.read_text(encoding="utf-8") if path.is_file() else ""


@pytest.mark.smoke
class TestSetupDocPage:
    """REQ-WIZARD-04: docs/setup.md carries the setup decision tree."""

    def test_setup_md_exists(self):
        """``docs/setup.md`` is present at the documented location."""
        assert SETUP_MD.is_file(), f"{SETUP_MD} does not exist"

    def test_setup_md_has_mermaid_block(self):
        """The page opens a fenced ```mermaid block."""
        assert "```mermaid" in _read(SETUP_MD), f"{SETUP_MD} contains no ```mermaid fenced block"

    def test_setup_md_has_flowchart(self):
        """The Mermaid block declares a ``flowchart``, not some other diagram type."""
        assert "flowchart" in _read(SETUP_MD), f"{SETUP_MD} declares no 'flowchart' diagram"

    def test_setup_md_has_pixi_commands(self):
        """The decision tree terminates in concrete ``pixi run`` commands."""
        assert "pixi run" in _read(SETUP_MD), f"{SETUP_MD} names no 'pixi run' command"

    def test_setup_md_has_summary_table(self):
        """The page carries a markdown table summarising the same mapping."""
        content = _read(SETUP_MD)
        has_table_row = any(line.strip().startswith("|") for line in content.splitlines())
        assert has_table_row, f"{SETUP_MD} contains no markdown table row (no line starting with '|')"

    def test_setup_md_covers_training_env(self):
        """The mapping routes readers to the ``training`` environment."""
        assert "training" in _read(SETUP_MD), f"{SETUP_MD} does not mention the 'training' environment"

    def test_setup_md_covers_jean_zay(self):
        """The mapping covers Jean-Zay as one of the locations it branches on."""
        content = _read(SETUP_MD)
        assert "Jean-Zay" in content or "jean-zay" in content, f"{SETUP_MD} does not mention the Jean-Zay cluster"
