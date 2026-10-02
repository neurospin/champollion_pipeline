"""Tests for REQ-PIXIVER-04/05 — install.sh refuses a pixi older than ``requires-pixi``.

TASK-157 (follow-up of TASK-142). ``install.sh`` runs ``pixi install -e default``
before the setup wizard, so a pixi older than 0.68 fails on ``pixi.toml``'s
rich-platform table (``expected a string, found table``) before the wizard can
print any version guidance (REQ-PIXIVER-03's recorded caveat).

- REQ-PIXIVER-04: below the ``>=`` bound of ``requires-pixi``, install.sh exits
  non-zero without running ``pixi install`` or ``pixi run``.
- REQ-PIXIVER-05: below that bound, install.sh prints the bound and
  ``pixi self-update``.

Harness: install.sh is executed for real, but from a copy in ``tmp_path``
(with ``pixi.toml`` and the top-level files of ``scripts/``) so that its ERR
trap can never drop an ``install_report_*.txt`` into the repository. A fake
``pixi`` placed first on ``PATH`` answers ``--version``/``-V`` with a chosen
version, logs every invocation's arguments, and succeeds without installing
anything. stdin is not a TTY, so the post-check path is ``pixi run install-all``
(also faked).

The proceed-case and missing-pixi tests are anchors: they pass today and guard
against a version check that blocks valid pixi versions or breaks the existing
``pixi not found`` path (REQ-WIZARD-03).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = PROJECT_ROOT / "install.sh"
PIXI_TOML = PROJECT_ROOT / "pixi.toml"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

SELF_UPDATE = "pixi self-update"
TOO_OLD = ["0.63.2", "0.67.0", "0.79.9"]
# 0.100.0 catches a lexicographic (string) comparison against "0.80.0".
NEW_ENOUGH = ["0.80.0", "0.81.2", "0.100.0", "1.0.0"]

FAKE_PIXI = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_PIXI_LOG"
case "${1:-}" in
    --version|-V) echo "pixi $FAKE_PIXI_VERSION" ;;
esac
exit 0
"""


def _minimum_version() -> str:
    """The ``>=`` bound of ``pixi.toml``'s ``requires-pixi``."""
    with PIXI_TOML.open("rb") as handle:
        spec = tomllib.load(handle).get("workspace", {}).get("requires-pixi", "")
    match = re.search(r">=\s*v?(\d+(?:\.\d+)*)", spec)
    assert match, f"pixi.toml requires-pixi {spec!r} has no '>=' lower bound"
    return match.group(1)


def _sandbox(tmp_path: Path) -> Path:
    """Copy install.sh, pixi.toml and top-level scripts/ files into ``tmp_path/project``."""
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    shutil.copy2(INSTALL_SH, project / "install.sh")
    shutil.copy2(PIXI_TOML, project / "pixi.toml")
    for item in SCRIPTS_DIR.iterdir():
        if item.is_file():
            shutil.copy2(item, project / "scripts" / item.name)
    return project


def _run_install(tmp_path: Path, version: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run the sandboxed install.sh with a fake pixi reporting ``version``.

    Returns the completed process (stdout and stderr merged) and the list of
    argument strings the fake pixi was invoked with.
    """
    project = _sandbox(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "pixi"
    fake.write_text(FAKE_PIXI, encoding="utf-8")
    fake.chmod(0o755)
    log = tmp_path / "pixi_calls.log"
    log.touch()
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "FAKE_PIXI_LOG": str(log),
        "FAKE_PIXI_VERSION": version,
    }
    result = subprocess.run(
        ["bash", str(project / "install.sh")],
        cwd=project,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
        check=False,
    )
    calls = log.read_text(encoding="utf-8").splitlines()
    return result, calls


def _installing_calls(calls: list[str]) -> list[str]:
    """Fake-pixi invocations that would install or run anything."""
    return [c for c in calls if c.split()[:1] in (["install"], ["run"])]


@pytest.mark.smoke
class TestInstallShRejectsOldPixi:
    """REQ-PIXIVER-04: below the requires-pixi floor, exit non-zero before any install."""

    @pytest.mark.parametrize("version", TOO_OLD)
    def test_old_pixi_exits_non_zero(self, tmp_path, version):
        """install.sh exits with a non-zero status when pixi is older than the floor."""
        result, calls = _run_install(tmp_path, version)
        assert result.returncode != 0, (
            f"REQ-PIXIVER-04: install.sh exited 0 with pixi {version} (floor {_minimum_version()}); "
            f"pixi calls: {calls}\n{result.stdout}"
        )

    @pytest.mark.parametrize("version", TOO_OLD)
    def test_old_pixi_runs_no_install(self, tmp_path, version):
        """install.sh never runs ``pixi install``/``pixi run`` when pixi is older than the floor."""
        _, calls = _run_install(tmp_path, version)
        assert not _installing_calls(calls), (
            f"REQ-PIXIVER-04: install.sh invoked {_installing_calls(calls)} with pixi {version} "
            f"(floor {_minimum_version()})"
        )


@pytest.mark.smoke
class TestInstallShExplainsOldPixi:
    """REQ-PIXIVER-05: below the floor, print the floor and ``pixi self-update``."""

    @pytest.mark.parametrize("version", TOO_OLD)
    def test_old_pixi_message_states_floor(self, tmp_path, version):
        """The output names the ``>=`` bound of ``requires-pixi``."""
        floor = _minimum_version()
        result, _ = _run_install(tmp_path, version)
        assert floor in result.stdout, (
            f"REQ-PIXIVER-05: install.sh output with pixi {version} does not contain the floor {floor!r}:\n"
            f"{result.stdout}"
        )

    @pytest.mark.parametrize("version", TOO_OLD)
    def test_old_pixi_message_states_self_update(self, tmp_path, version):
        """The output names the upgrade command ``pixi self-update``."""
        result, _ = _run_install(tmp_path, version)
        assert SELF_UPDATE in result.stdout, (
            f"REQ-PIXIVER-05: install.sh output with pixi {version} does not contain {SELF_UPDATE!r}:\n{result.stdout}"
        )


@pytest.mark.smoke
class TestInstallShAcceptsCurrentPixi:
    """Anchors (pass today): a new-enough pixi, or a missing one, keeps its existing behaviour."""

    @pytest.mark.parametrize("version", NEW_ENOUGH)
    def test_new_enough_pixi_proceeds_to_install(self, tmp_path, version):
        """At or above the floor, install.sh runs ``pixi install -e default`` and exits 0."""
        result, calls = _run_install(tmp_path, version)
        assert "install -e default" in calls, (
            f"install.sh did not run 'pixi install -e default' with pixi {version}; calls: {calls}\n{result.stdout}"
        )
        assert result.returncode == 0, f"install.sh exited {result.returncode} with pixi {version}:\n{result.stdout}"

    def test_missing_pixi_exits_with_message(self, tmp_path):
        """With no pixi on PATH, install.sh exits 1 and says pixi was not found (REQ-WIZARD-03)."""
        project = _sandbox(tmp_path)
        path = os.pathsep.join(
            d for d in os.environ.get("PATH", "").split(os.pathsep) if d and not (Path(d) / "pixi").exists()
        )
        result = subprocess.run(
            ["bash", str(project / "install.sh")],
            cwd=project,
            env={**os.environ, "PATH": path},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 1, f"install.sh exited {result.returncode} without pixi:\n{result.stdout}"
        assert "pixi" in result.stdout.lower(), f"install.sh gave no pixi message:\n{result.stdout}"
