"""Tests for REQ-WIZARD-03 — the install.sh one-command bootstrap wrapper.

# TASK-013

The artifact under test is ``install.sh`` at the project root: the requirement
states that it shall verify pixi is installed, then run ``pixi run setup``,
exiting 1 with a message if pixi is absent.

Each acquired behaviour gets its own test so a half-written script reports
precisely which element is still missing — a script that runs ``pixi run
setup`` with no guard is a different defect from one that guards but never
runs the wizard, and both differ from a script that is not executable and so
cannot be invoked as ``./install.sh`` at all.

The assertions read the script's *text* rather than executing it: running the
real script would install into the developer's own environment, and the
requirement constrains the script's content (a pixi presence check, the setup
invocation, a failing exit status), not the outcome of a live install. The
wording of the diagnostic message is deliberately left unconstrained by the
requirement, so no test pins it down.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = PROJECT_ROOT / "install.sh"


def _read(path: Path) -> str:
    """Contents of ``path``, or an empty string when it does not exist yet.

    Returning empty (rather than raising) keeps each test's own assertion as
    the reported failure, so a missing script reads as a plain red test rather
    than an unrelated ``FileNotFoundError``.
    """
    return path.read_text(encoding="utf-8") if path.is_file() else ""


@pytest.mark.smoke
class TestInstallSh:
    """REQ-WIZARD-03: install.sh guards on pixi, then runs the setup wizard."""

    def test_install_sh_exists(self):
        """``install.sh`` is present at the project root."""
        assert INSTALL_SH.is_file(), f"{INSTALL_SH} does not exist"

    def test_install_sh_is_bash(self):
        """The script declares a bash shebang on its first line."""
        first_line = _read(INSTALL_SH).splitlines()[0] if _read(INSTALL_SH) else ""
        assert first_line in (
            "#!/usr/bin/env bash",
            "#!/bin/bash",
        ), f"{INSTALL_SH} first line is {first_line!r}, expected a bash shebang"

    def test_install_sh_checks_pixi(self):
        """The script verifies pixi is installed before using it."""
        content = _read(INSTALL_SH)
        assert "pixi" in content, f"{INSTALL_SH} never mentions pixi"
        assert "command -v" in content or "which" in content, (
            f"{INSTALL_SH} contains no pixi presence check (no 'command -v' or 'which')"
        )

    def test_install_sh_runs_setup(self):
        """The script invokes the setup wizard via ``pixi run setup``."""
        assert "pixi run setup" in _read(INSTALL_SH), f"{INSTALL_SH} does not run 'pixi run setup'"

    def test_install_sh_exits_on_missing_pixi(self):
        """The script exits with status 1 when pixi is absent."""
        assert "exit 1" in _read(INSTALL_SH), f"{INSTALL_SH} contains no 'exit 1' failure path"

    def test_install_sh_is_executable(self):
        """The script carries an owner execute bit, so ``./install.sh`` works."""
        assert INSTALL_SH.is_file(), f"{INSTALL_SH} does not exist"
        mode = INSTALL_SH.stat().st_mode
        assert mode & 0o100, (
            f"{INSTALL_SH} mode is {oct(mode)}; owner execute bit is not set "
            f"(os.access X_OK: {os.access(INSTALL_SH, os.X_OK)})"
        )
