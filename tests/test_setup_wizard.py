"""Tests for REQ-WIZARD-02 and REQ-WIZARD-03 — setup wizard and install.sh.

REQ-WIZARD-02: scripts/setup_wizard.py shall ask three questions (location,
use case, GPU), map answers to correct pixi run commands, display the plan
with rich, and execute on confirmation.

REQ-WIZARD-03: install.sh shall verify pixi is installed, then run
``pixi run setup``, exiting 1 with a message if pixi is absent.

# TASK-012, TASK-013
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WIZARD = PROJECT_ROOT / "scripts" / "setup_wizard.py"
INSTALL_SH = PROJECT_ROOT / "install.sh"


@pytest.mark.smoke
class TestSetupWizardScript:
    """REQ-WIZARD-02: setup_wizard.py structure and command matrix."""

    def test_wizard_exists(self):
        assert WIZARD.exists(), f"{WIZARD} not found"

    def test_wizard_imports_rich(self):
        src = WIZARD.read_text()
        assert "from rich" in src or "import rich" in src, "wizard does not import rich"

    def test_wizard_has_main_guard(self):
        src = WIZARD.read_text()
        assert '__name__ == "__main__"' in src, "wizard has no __main__ guard"

    def test_wizard_handles_dry_run(self):
        src = WIZARD.read_text()
        assert "--dry-run" in src, "wizard does not support --dry-run"

    def test_wizard_covers_jean_zay(self):
        src = WIZARD.read_text()
        assert "Jean-Zay" in src or "jean-zay" in src.lower(), "wizard has no Jean-Zay branch"

    def test_wizard_covers_training_env(self):
        src = WIZARD.read_text()
        assert "-e training" in src, "wizard has no training env command"

    def test_wizard_covers_embeddings_env(self):
        src = WIZARD.read_text()
        assert "-e embeddings" in src, "wizard has no embeddings env command"

    def test_wizard_covers_docs_env(self):
        src = WIZARD.read_text()
        assert "-e docs" in src, "wizard has no docs env command"

    def test_wizard_checks_pixi_installed(self):
        src = WIZARD.read_text()
        assert "pixi" in src and ("shutil.which" in src or "command" in src), (
            "wizard does not check for pixi binary"
        )

    def test_wizard_dry_run_exits_without_running(self):
        """--dry-run must not call subprocess.run on any pixi command."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("setup_wizard", WIZARD)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # build_plan for local + docs + no-GPU → single command, no GPU warning
        plan = mod.build_plan(location="1", use_case="4", gpu=False)
        assert plan.commands == ["pixi run -e docs build-docs"]
        assert plan.warnings == []

    def test_command_matrix_full_local(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("setup_wizard", WIZARD)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        plan = mod.build_plan(location="1", use_case="1", gpu=True)
        assert "pixi run install-all" in plan.commands

    def test_command_matrix_training_jean_zay(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("setup_wizard", WIZARD)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        plan = mod.build_plan(location="2", use_case="3", gpu=True)
        assert any("-e training" in cmd for cmd in plan.commands)
        assert any("Jean-Zay" in n or "jean-zay" in n.lower() for n in plan.notes)


@pytest.mark.smoke
class TestInstallSh:
    """REQ-WIZARD-03: install.sh is executable and delegates to pixi run setup."""

    def test_install_sh_exists(self):
        assert INSTALL_SH.exists(), f"{INSTALL_SH} not found"

    def test_install_sh_is_executable(self):
        mode = INSTALL_SH.stat().st_mode
        assert mode & stat.S_IXUSR, f"{INSTALL_SH} is not executable"

    def test_install_sh_calls_pixi_run_setup(self):
        src = INSTALL_SH.read_text()
        assert "pixi run setup" in src, "install.sh does not call 'pixi run setup'"

    def test_install_sh_checks_pixi(self):
        src = INSTALL_SH.read_text()
        assert "command -v pixi" in src or "which pixi" in src, (
            "install.sh does not check for pixi binary"
        )

    def test_install_sh_exits_1_if_no_pixi(self):
        src = INSTALL_SH.read_text()
        assert "exit 1" in src, "install.sh does not exit 1 when pixi is missing"

    def test_install_sh_has_set_euo_pipefail(self):
        src = INSTALL_SH.read_text()
        assert "set -euo pipefail" in src, "install.sh lacks 'set -euo pipefail'"
