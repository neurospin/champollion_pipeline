"""Tests for REQ-CLEANUP-FORMAT-02 — ``main.py`` and ``scripts/`` are ruff-formatted and in the format task's scope.

The requirement: every Git-tracked Python file at ``main.py`` or under
``scripts/`` shall be formatted per the repository's ruff configuration, with
both paths among the pixi ``format`` task's targets, such that
``ruff format --check`` over those files exits 0.

The manifest check parses ``pixi.toml`` only. The format check runs ruff from
the repo root (so ``pyproject.toml``'s ``[tool.ruff]`` table applies) over
``git ls-files`` output, so untracked files never affect the verdict.
"""

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

pytestmark = pytest.mark.smoke  # noqa: V107

REPO_ROOT = Path(__file__).resolve().parents[1]
PIXI_TOML = REPO_ROOT / "pixi.toml"
TASK_NAME = "format"
REQUIRED_TARGETS = ("src/", "tests/", "main.py", "scripts/")
TRACKED_PATTERNS = ("main.py", "scripts/*.py")


def _ruff_command() -> list[str] | None:
    """Return the argv prefix that invokes ruff, or ``None`` if unavailable."""
    binary = shutil.which("ruff")
    if binary:
        return [binary]
    probe = subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True)
    return [sys.executable, "-m", "ruff"] if probe.returncode == 0 else None


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", *TRACKED_PATTERNS],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _task_command(task) -> str:
    """Return a pixi task's shell command, whether it is a string or a table."""
    if isinstance(task, dict):
        return task.get("cmd", "")
    return task


def _normalise_target(token: str) -> str:
    """Treat ``scripts``, ``scripts/`` and ``./scripts`` as the same target; leave files as-is."""
    token = token.removeprefix("./")
    if token.endswith(".py") or token.endswith("/"):
        return token
    return f"{token}/"


class TestFormatTaskScope:
    """REQ-CLEANUP-FORMAT-02."""

    def test_format_task_targets_main_py_and_scripts(self):
        with PIXI_TOML.open("rb") as handle:
            tasks = tomllib.load(handle).get("tasks", {})
        assert TASK_NAME in tasks, f"REQ-CLEANUP-FORMAT-02: pixi.toml defines no '{TASK_NAME}' task"
        tokens = shlex.split(_task_command(tasks[TASK_NAME]))
        assert tokens[:2] == ["ruff", "format"], (
            f"REQ-CLEANUP-FORMAT-02: '{TASK_NAME}' task does not run 'ruff format': {tokens!r}"
        )
        targets = {_normalise_target(t) for t in tokens[2:] if not t.startswith("-")}
        missing = [t for t in REQUIRED_TARGETS if t not in targets]
        assert not missing, (
            f"REQ-CLEANUP-FORMAT-02: '{TASK_NAME}' task's ruff format targets {sorted(targets)} are missing {missing}"
        )

    @pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
    def test_ruff_format_check_passes_on_tracked_main_py_and_scripts(self):
        ruff = _ruff_command()
        if ruff is None:
            pytest.skip("ruff not available")
        files = _tracked_files()
        assert files, "REQ-CLEANUP-FORMAT-02: git ls-files found no tracked main.py or scripts/ Python files"
        result = subprocess.run(
            [*ruff, "format", "--check", *files],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        output = (result.stdout + result.stderr).strip()
        assert result.returncode == 0, (
            "REQ-CLEANUP-FORMAT-02: ruff format --check reports unformatted tracked main.py / scripts/ "
            f"files (exit {result.returncode}):\n{output}"
        )
