"""Tests for REQ-CLEANUP-FORMAT-01 — tracked ``src/``/``tests/`` files are ruff-formatted.

The requirement: every Git-tracked Python file under ``src/`` and ``tests/``
shall be formatted per the repository's ruff configuration, such that
``ruff format --check`` over those files exits 0.

Only Git-tracked files are checked (``git ls-files``), so stray untracked files
in a working tree do not affect the verdict. Ruff runs from the repo root so
``pyproject.toml``'s ``[tool.ruff]`` table (line length, excludes) applies.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("src", "tests")


def _ruff_command() -> list[str] | None:
    """Return the argv prefix that invokes ruff, or ``None`` if unavailable."""
    binary = shutil.which("ruff")
    if binary:
        return [binary]
    probe = subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True)
    return [sys.executable, "-m", "ruff"] if probe.returncode == 0 else None


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", *(f"{d}/*.py" for d in SCAN_DIRS)],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


@pytest.mark.smoke
@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
class TestRuffFormatClean:
    """REQ-CLEANUP-FORMAT-01."""

    def test_ruff_format_check_passes_on_tracked_files(self):
        ruff = _ruff_command()
        if ruff is None:
            pytest.skip("ruff not available")
        files = _tracked_files()
        assert files, "REQ-CLEANUP-FORMAT-01: git ls-files found no tracked src/ or tests/ Python files"
        result = subprocess.run(
            [*ruff, "format", "--check", *files],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        output = (result.stdout + result.stderr).strip()
        assert result.returncode == 0, (
            "REQ-CLEANUP-FORMAT-01: ruff format --check reports unformatted tracked src/ and tests/ "
            f"files (exit {result.returncode}):\n{output}"
        )
