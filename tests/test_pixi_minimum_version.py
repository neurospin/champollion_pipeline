"""Tests for REQ-PIXIVER-01..03 — the minimum pixi version is declared and documented.

TASK-142. ``pixi.toml``'s ``[workspace].platforms`` uses the rich-platform
table form (``{ name = "linux-64-cuda", platform = "linux-64", cuda = "12.0" }``).
Verified 2026-10-01 against the released linux-x86_64 binaries: pixi 0.70.2 and
older reject it with ``expected a string, found table``; 0.71.0 is the first
release that parses it (and reads the v7 ``pixi.lock``). ``requires-pixi`` is
only evaluated before platform parsing from pixi 0.68.0 on, so older pixi
(e.g. 0.63.2) still shows the parse error — the README and the setup wizard
are the mitigation for those users.

- REQ-PIXIVER-01: ``requires-pixi`` rejects every pixi below 0.71.0 and
  accepts 0.80.0.
- REQ-PIXIVER-02: README's Installation section states the minimum version
  and ``pixi self-update``.
- REQ-PIXIVER-03: the setup wizard prints both before its first question.
"""

from __future__ import annotations

import importlib.util
import io
import re
from pathlib import Path

import pytest
import tomllib
from packaging.specifiers import SpecifierSet
from packaging.version import Version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIXI_TOML = PROJECT_ROOT / "pixi.toml"
README = PROJECT_ROOT / "README.md"
WIZARD = PROJECT_ROOT / "scripts" / "setup_wizard.py"

SELF_UPDATE = "pixi self-update"
TOO_OLD = ["0.63.2", "0.67.0", "0.68.0", "0.70.2", "0.70.99"]
KNOWN_GOOD = "0.80.0"


def _requires_pixi() -> str | None:
    with PIXI_TOML.open("rb") as handle:
        return tomllib.load(handle).get("workspace", {}).get("requires-pixi")


def _minimum_version() -> str:
    spec = _requires_pixi()
    assert spec, "pixi.toml [workspace] declares no requires-pixi (REQ-PIXIVER-01)"
    match = re.search(r">=\s*v?(\d+(?:\.\d+)*)", spec)
    assert match, f"requires-pixi {spec!r} has no '>=' lower bound to state as the minimum version"
    return match.group(1)


def _installation_section() -> str:
    text = README.read_text(encoding="utf-8")
    match = re.search(r"^## 1\. Installation\s*$(.*?)(?=^## )", text, re.MULTILINE | re.DOTALL)
    assert match, "README.md has no '## 1. Installation' section"
    return match.group(1)


@pytest.mark.smoke
class TestRequiresPixiDeclared:
    """REQ-PIXIVER-01: requires-pixi excludes pixi that cannot parse the manifest."""

    def test_requires_pixi_rejects_versions_without_rich_platforms(self):
        spec = _requires_pixi()
        assert spec, "REQ-PIXIVER-01: pixi.toml [workspace] declares no requires-pixi"
        specifier = SpecifierSet(spec)
        accepted_too_old = [v for v in TOO_OLD if Version(v) in specifier]
        assert not accepted_too_old, (
            f"REQ-PIXIVER-01: requires-pixi {spec!r} accepts {accepted_too_old}, which cannot parse "
            "the rich-platform table in [workspace].platforms"
        )

    def test_requires_pixi_accepts_known_good_version(self):
        spec = _requires_pixi()
        assert spec, "REQ-PIXIVER-01: pixi.toml [workspace] declares no requires-pixi"
        assert Version(KNOWN_GOOD) in SpecifierSet(spec), (
            f"REQ-PIXIVER-01: requires-pixi {spec!r} rejects pixi {KNOWN_GOOD}"
        )


@pytest.mark.smoke
class TestReadmeStatesMinimumPixi:
    """REQ-PIXIVER-02: README's Installation section states the minimum and how to upgrade."""

    def test_installation_section_mentions_self_update(self):
        assert SELF_UPDATE in _installation_section(), (
            f"REQ-PIXIVER-02: README.md's Installation section does not contain {SELF_UPDATE!r}"
        )

    def test_installation_section_states_minimum_version(self):
        minimum = _minimum_version()
        assert minimum in _installation_section(), (
            f"REQ-PIXIVER-02: README.md's Installation section does not state the minimum pixi version {minimum}"
        )


class _StopAtFirstQuestion(Exception):
    """Raised in place of the wizard's first prompt."""


@pytest.fixture
def wizard_startup_output(monkeypatch) -> str:
    """Everything the wizard prints, with pixi present, before its first question."""
    from rich.console import Console

    spec = importlib.util.spec_from_file_location("setup_wizard_pixiver", WIZARD)
    wizard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wizard)

    buffer = io.StringIO()

    def stop(*_args, **_kwargs):
        raise _StopAtFirstQuestion

    monkeypatch.setattr(wizard, "console", Console(file=buffer, width=400, force_terminal=False, color_system=None))
    monkeypatch.setattr(wizard.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(wizard, "ask_choice", stop)
    monkeypatch.setattr(wizard.Prompt, "ask", stop)
    monkeypatch.setattr(wizard.Confirm, "ask", stop)
    with pytest.raises(_StopAtFirstQuestion):
        wizard.main(dry_run=True)
    return buffer.getvalue()


@pytest.mark.smoke
class TestWizardStatesMinimumPixi:
    """REQ-PIXIVER-03: the setup wizard states the minimum and how to upgrade before asking anything."""

    def test_wizard_mentions_self_update_before_first_question(self, wizard_startup_output):
        assert SELF_UPDATE in wizard_startup_output, (
            f"REQ-PIXIVER-03: setup wizard does not print {SELF_UPDATE!r} before its first question.\n"
            f"Output:\n{wizard_startup_output}"
        )

    def test_wizard_states_minimum_version_before_first_question(self, wizard_startup_output):
        minimum = _minimum_version()
        assert minimum in wizard_startup_output, (
            f"REQ-PIXIVER-03: setup wizard does not print the minimum pixi version {minimum} before its first "
            f"question.\nOutput:\n{wizard_startup_output}"
        )
