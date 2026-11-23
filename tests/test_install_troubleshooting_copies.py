"""The install and troubleshooting text that README.md repeats from docs/ stays in step with it.

TASK-DOCS-BDRABCZUK-17DCE2E33CD6. README.md carries a copy of the install guide
(``docs/installation.md``) and of three troubleshooting entries (``docs/troubleshooting.md``).
User decision 2026-11-23: pin the shared parts only, so the surrounding prose may differ.

- REQ-DOCS-BDRABCZUK-1FF345C91B0D: the fenced command blocks of the Setup, Uninstallation and
  Updating an existing install sections are equal, in order, in both copies.
- REQ-DOCS-BDRABCZUK-E3038C7B9997: the Prerequisites section of each copy states the minimum pixi
  version of ``pixi.toml``'s ``requires-pixi`` (README side: REQ-PIXIVER-BDRABCZUK-73BC45B95B68).
- REQ-DOCS-BDRABCZUK-747BEC077877: each README troubleshooting heading is a heading of the docs page.
- REQ-DOCS-BDRABCZUK-56592C679456: the fenced fix commands of each shared troubleshooting entry
  are equal, in order, in both copies.

Text reads only; no filesystem writes, no network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tomllib

from tests.docs_test_helpers import MD_FENCE_RE, MD_HEADING_RE, bash_blocks, md_section, read_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = PROJECT_ROOT / "README.md"
INSTALLATION = PROJECT_ROOT / "docs" / "installation.md"
TROUBLESHOOTING = PROJECT_ROOT / "docs" / "troubleshooting.md"
PIXI_TOML = PROJECT_ROOT / "pixi.toml"

SHARED_INSTALL_SECTIONS = ("Setup", "Uninstallation", "Updating an existing install")


def _sections(text: str, level: int) -> dict[str, str]:
    """``{title: body}`` of each heading of ``level`` outside fenced code.

    A body ends at the next heading of that level or above.
    """
    found: dict[str, str] = {}
    title: str | None = None
    in_fence = False
    for line in text.splitlines(keepends=True):
        if MD_FENCE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            m = MD_HEADING_RE.match(line.rstrip("\n"))
            if m and len(m.group(1)) <= level:
                title = m.group(2) if len(m.group(1)) == level else None
                if title is not None:
                    found.setdefault(title, "")
                continue
        if title is not None:
            found[title] += line
    return found


def _command_blocks(body: str) -> list[str]:
    """Fenced command blocks of ``body``, trailing whitespace dropped per line."""
    return ["\n".join(line.rstrip() for line in block.strip("\n").splitlines()) for block in bash_blocks(body)]


def _readme_install_sections() -> dict[str, str]:
    return _sections(md_section(read_text(README), r"1\. Installation"), 3)


def _readme_troubleshooting_entries() -> dict[str, str]:
    entries = _sections(md_section(read_text(README), "Troubleshooting"), 3)
    assert entries, "README.md '## Troubleshooting' holds no '### ' entry (test parser sanity check)"
    return entries


def _requires_pixi_minimum() -> str:
    with PIXI_TOML.open("rb") as handle:
        spec = tomllib.load(handle).get("workspace", {}).get("requires-pixi")
    assert spec, "pixi.toml [workspace] declares no requires-pixi"
    m = re.search(r">=\s*v?(\d+(?:\.\d+)*)", spec)
    assert m, f"requires-pixi {spec!r} has no '>=' lower bound"
    return m.group(1)


@pytest.mark.parametrize("title", SHARED_INSTALL_SECTIONS)
def test_install_section_command_blocks_agree(title: str) -> None:
    """REQ-DOCS-BDRABCZUK-1FF345C91B0D."""
    readme = _readme_install_sections()
    docs = _sections(read_text(INSTALLATION), 2)
    assert title in readme, f"README.md '## 1. Installation' has no '### {title}' section"
    assert title in docs, f"docs/installation.md has no '## {title}' section"
    readme_blocks = _command_blocks(readme[title])
    assert readme_blocks, f"README.md '### {title}' holds no fenced command block (test parser sanity check)"
    assert readme_blocks == _command_blocks(docs[title]), (
        f"'{title}': the fenced commands of README.md and docs/installation.md differ"
    )


@pytest.mark.parametrize(
    ("copy", "sections"),
    [
        ("README.md", _readme_install_sections),
        ("docs/installation.md", lambda: _sections(read_text(INSTALLATION), 2)),
    ],
)
def test_prerequisites_state_requires_pixi_minimum(copy: str, sections) -> None:
    """REQ-DOCS-BDRABCZUK-E3038C7B9997 (README side also REQ-PIXIVER-BDRABCZUK-73BC45B95B68)."""
    found = sections()
    assert "Prerequisites" in found, f"{copy} has no Prerequisites section"
    minimum = _requires_pixi_minimum()
    assert re.search(rf"(?<![\d.]){re.escape(minimum)}(?![\d.])", found["Prerequisites"]), (
        f"{copy} Prerequisites does not state the minimum pixi version {minimum} of pixi.toml's requires-pixi"
    )


def test_readme_troubleshooting_headings_in_docs_page() -> None:
    """REQ-DOCS-BDRABCZUK-747BEC077877."""
    docs = _sections(read_text(TROUBLESHOOTING), 2)
    missing = [title for title in _readme_troubleshooting_entries() if title not in docs]
    assert not missing, f"docs/troubleshooting.md has no '## ' heading for README.md entries: {missing}"


def test_shared_troubleshooting_fix_commands_agree() -> None:
    """REQ-DOCS-BDRABCZUK-56592C679456."""
    readme = _readme_troubleshooting_entries()
    docs = _sections(read_text(TROUBLESHOOTING), 2)
    shared = [title for title in readme if title in docs]
    assert shared, "README.md and docs/troubleshooting.md share no troubleshooting entry"
    differing = [title for title in shared if _command_blocks(readme[title]) != _command_blocks(docs[title])]
    assert not differing, f"the fenced fix commands of README.md and docs/troubleshooting.md differ for: {differing}"
