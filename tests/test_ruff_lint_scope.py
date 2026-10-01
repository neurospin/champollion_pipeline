"""Tests for REQ-CLEANUP-LINT-01 — the pixi ``lint`` task covers and passes on ``main.py`` and ``scripts/``.

The requirement: the pixi ``lint`` task shall include ``main.py`` and
``scripts/`` among its ``ruff check`` targets, such that ``ruff check`` over
the Git-tracked Python files at those paths exits 0.

The manifest check parses ``pixi.toml`` only. The lint check runs ruff from the
repo root (so ``pyproject.toml``'s ``[tool.ruff]`` table applies) over
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
TASK_NAME = "lint"
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


class TestLintTaskScope:
    """REQ-CLEANUP-LINT-01."""

    def test_lint_task_targets_main_py_and_scripts(self):
        with PIXI_TOML.open("rb") as handle:
            tasks = tomllib.load(handle).get("tasks", {})
        assert TASK_NAME in tasks, f"REQ-CLEANUP-LINT-01: pixi.toml defines no '{TASK_NAME}' task"
        tokens = shlex.split(_task_command(tasks[TASK_NAME]))
        assert tokens[:2] == ["ruff", "check"], (
            f"REQ-CLEANUP-LINT-01: '{TASK_NAME}' task does not run 'ruff check': {tokens!r}"
        )
        targets = {_normalise_target(t) for t in tokens[2:] if not t.startswith("-")}
        missing = [t for t in REQUIRED_TARGETS if t not in targets]
        assert not missing, (
            f"REQ-CLEANUP-LINT-01: '{TASK_NAME}' task's ruff check targets {sorted(targets)} are missing {missing}"
        )

    @pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
    def test_ruff_check_passes_on_tracked_main_py_and_scripts(self):
        ruff = _ruff_command()
        if ruff is None:
            pytest.skip("ruff not available")
        files = _tracked_files()
        assert files, "REQ-CLEANUP-LINT-01: git ls-files found no tracked main.py or scripts/ Python files"
        result = subprocess.run(
            [*ruff, "check", "--output-format", "concise", *files],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        output = (result.stdout + result.stderr).strip()
        assert result.returncode == 0, (
            "REQ-CLEANUP-LINT-01: ruff check reports findings in tracked main.py / scripts/ "
            f"files (exit {result.returncode}):\n{output}"
        )
