"""Source-text and structural tests for REQ-HEALTH-01.

Verifies:
1. scripts/install_health.py exists and is stdlib-only (no third-party imports at module level)
2. The script exposes --fix, --pre-update, and --report CLI flags
3. The script is callable via bare python3 (no pixi env required)
4. pixi.toml wires install-pipeline into install-all and declares check-install/pre-update tasks
5. install.sh traps errors and calls install_health.py --report

All assertions read source text — no runtime execution (avoids triggering real git/pip ops).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTALL_HEALTH = PROJECT_ROOT / "scripts" / "install_health.py"
PIXI_TOML = PROJECT_ROOT / "pixi.toml"
INSTALL_SH = PROJECT_ROOT / "install.sh"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


@pytest.fixture(scope="module")
def pixi_config() -> dict:
    with PIXI_TOML.open("rb") as fh:
        return tomllib.load(fh)


@pytest.fixture(scope="module")
def health_source() -> str:
    return _read(INSTALL_HEALTH)


@pytest.fixture(scope="module")
def install_sh_source() -> str:
    return _read(INSTALL_SH)


# ── Existence ─────────────────────────────────────────────────────────────────


@pytest.mark.smoke
class TestInstallHealthExists:
    def test_script_exists(self):
        assert INSTALL_HEALTH.is_file(), f"{INSTALL_HEALTH} not found"

    def test_script_has_python_shebang(self, health_source):
        first = health_source.splitlines()[0] if health_source else ""
        assert first.startswith("#!/usr/bin/env python"), (
            f"shebang must be #!/usr/bin/env python3 or similar, got: {first!r}"
        )


# ── stdlib-only (callable without pixi env) ───────────────────────────────────


@pytest.mark.smoke
class TestInstallHealthStdlibOnly:
    ALLOWED_THIRD_PARTY = {"rich"}  # optional — must be guarded by try/except

    def _module_imports(self, source: str) -> list[str]:
        """Top-level import names (not from-imports of submodules)."""
        names: list[str] = []
        for line in source.splitlines():
            m = re.match(r"^import\s+(\S+)", line)
            if m:
                names.append(m.group(1).split(".")[0])
            m2 = re.match(r"^from\s+(\S+)\s+import", line)
            if m2:
                names.append(m2.group(1).split(".")[0])
        return names

    def test_rich_import_is_guarded(self, health_source):
        """rich must be imported inside try/except ImportError to allow system-python invocation."""
        if "import rich" not in health_source and "from rich" not in health_source:
            return  # rich not used at all — fine
        assert "except ImportError" in health_source, (
            "rich is imported unconditionally; must be wrapped in try/except ImportError "
            "so the script runs on system Python without the pixi env"
        )

    def test_no_unconditional_third_party(self, health_source):
        """No third-party package other than optionally rich is imported at module level."""
        stdlib = {
            "os", "sys", "re", "json", "subprocess", "pathlib", "argparse",
            "datetime", "shutil", "platform", "importlib", "importlib.util",
            "importlib.metadata", "textwrap", "collections", "dataclasses",
            "contextlib", "io", "traceback", "typing", "__future__", "socket",
            "time", "uuid", "hashlib", "stat", "tomllib", "enum",
        }
        # Everything between module start and first try/except block
        before_try: list[str] = []
        for line in health_source.splitlines():
            if line.startswith("try:"):
                break
            before_try.append(line)
        top_imports = self._module_imports("\n".join(before_try))
        bad = [n for n in top_imports if n not in stdlib and n not in self.ALLOWED_THIRD_PARTY]
        assert not bad, (
            f"Unconditional third-party imports at module level: {bad}. "
            "Use stdlib only so the script runs without the pixi env."
        )


# ── CLI flags ─────────────────────────────────────────────────────────────────


@pytest.mark.smoke
class TestInstallHealthCLI:
    def test_fix_flag_present(self, health_source):
        assert "--fix" in health_source, "script must accept --fix flag"

    def test_pre_update_flag_present(self, health_source):
        assert "--pre-update" in health_source, "script must accept --pre-update flag"

    def test_report_flag_present(self, health_source):
        assert "--report" in health_source, "script must accept --report flag"

    def test_output_flag_present(self, health_source):
        assert "--output" in health_source, "script must accept --output <file> flag"


# ── Verbose output markers ─────────────────────────────────────────────────────


@pytest.mark.smoke
class TestInstallHealthVerbosity:
    def test_check_prefix_used(self, health_source):
        assert "[CHECK]" in health_source or "CHECK" in health_source, (
            "script must label each check with a [CHECK] prefix"
        )

    def test_fix_prefix_used(self, health_source):
        assert "[FIX]" in health_source or "FIX" in health_source, (
            "script must label auto-fix actions with a [FIX] prefix"
        )

    def test_warn_prefix_used(self, health_source):
        assert "[WARN]" in health_source or "WARN" in health_source, (
            "script must label warnings with a [WARN] prefix"
        )

    def test_report_header_present(self, health_source):
        assert "CHAMPOLLION INSTALL REPORT" in health_source, (
            "script must emit a CHAMPOLLION INSTALL REPORT header in --report mode"
        )

    def test_report_includes_system_section(self, health_source):
        assert "SYSTEM" in health_source, "report must include a SYSTEM section"

    def test_report_includes_package_status(self, health_source):
        assert "PACKAGE STATUS" in health_source or "champollion_pipeline" in health_source, (
            "report must show package installation status"
        )

    def test_report_includes_what_cannot_fix(self, health_source):
        assert "cannot" in health_source.lower() or "CANNOT" in health_source, (
            "report must explicitly state what it cannot fix"
        )

    def test_report_includes_manual_commands(self, health_source):
        assert "COMMANDS TO RUN MANUALLY" in health_source or "manually" in health_source.lower(), (
            "report must list commands for the user to run manually"
        )


# ── pixi.toml: install-all must include install-pipeline ─────────────────────


@pytest.mark.smoke
class TestPixiInstallAllDeps:
    def test_install_pipeline_task_defined_in_shared_tasks(self, pixi_config):
        tasks = pixi_config.get("tasks", {})
        assert "install-pipeline" in tasks, (
            "pixi.toml must define install-pipeline in [tasks] (not only in feature.embeddings.tasks) "
            "so install-all can depend on it"
        )

    def test_install_all_depends_on_install_pipeline(self, pixi_config):
        tasks = pixi_config.get("tasks", {})
        install_all = tasks.get("install-all", {})
        depends = install_all.get("depends-on", []) if isinstance(install_all, dict) else []
        assert "install-pipeline" in depends, (
            f"install-all.depends-on does not include install-pipeline: {depends}. "
            "This causes ModuleNotFoundError for champollion_pipeline after install."
        )

    def test_check_install_task_defined(self, pixi_config):
        tasks = pixi_config.get("tasks", {})
        assert "check-install" in tasks, "pixi.toml must define a check-install task"

    def test_pre_update_task_defined(self, pixi_config):
        tasks = pixi_config.get("tasks", {})
        assert "pre-update" in tasks, "pixi.toml must define a pre-update task"

    def test_install_all_ends_with_check_install(self, pixi_config):
        tasks = pixi_config.get("tasks", {})
        install_all = tasks.get("install-all", {})
        depends = install_all.get("depends-on", []) if isinstance(install_all, dict) else []
        assert "check-install" in depends, (
            "install-all must call check-install at the end so users see health status post-install"
        )


# ── install.sh: error trap + report ──────────────────────────────────────────


@pytest.mark.smoke
class TestInstallShErrorHandling:
    def test_install_sh_has_error_trap(self, install_sh_source):
        assert "trap" in install_sh_source, (
            "install.sh must set an ERR trap to call install_health.py --report on failure"
        )

    def test_install_sh_calls_install_health_on_error(self, install_sh_source):
        assert "install_health.py" in install_sh_source, (
            "install.sh ERR trap must call scripts/install_health.py to generate a report"
        )

    def test_install_sh_saves_report_file(self, install_sh_source):
        assert "install_report" in install_sh_source or "REPORT" in install_sh_source, (
            "install.sh must save report to a timestamped file users can send"
        )

    def test_install_sh_runs_pixi_install(self, install_sh_source):
        assert "pixi install" in install_sh_source, (
            "install.sh must run 'pixi install' to resolve the conda env "
            "(not only 'pixi run setup' which assumes env already exists)"
        )

    def test_install_sh_runs_install_all(self, install_sh_source):
        assert "pixi run install-all" in install_sh_source, (
            "install.sh must run 'pixi run install-all' to wire all pip editable installs"
        )
