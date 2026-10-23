"""Tests for the ruff UP (pyupgrade), B (flake8-bugbear) and SIM (flake8-simplify) sweep.

- REQ-LINTRULES-BDRABCZUK-73E61E51C2F1: ruff, run with the repository ``ruff.toml``,
  reports UP, B and SIM findings in ``main.py`` and in each Python file under ``src/``
  or ``tests/``.
- REQ-LINTRULES-BDRABCZUK-509F90FC030A: the UP, B and SIM families, applied for Python 3.10
  without the ``ruff.toml`` per-file ignores, report zero findings in ``main.py`` and
  the Git-tracked Python files under ``src/`` or ``tests/``.

REQ-LINTRULES-BDRABCZUK-9F7DB87532EE (each test keeps its outcome from WIP 5a8ed07) is
verified by comparing full-suite outcomes, not by a test in this file.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke  # noqa: V107

REPO_ROOT = Path(__file__).resolve().parents[1]
RUFF_TOML = REPO_ROOT / "ruff.toml"
FAMILIES = ("UP", "B", "SIM")
TARGET_VERSION = "py310"  # pyproject.toml requires-python >= 3.10
FAMILY_PATTERN = re.compile(r"^(UP|B|SIM)\d+$")
# One violation per family: UP035 (typing.List), B006 (mutable default), SIM102 (nested if).
PROBE_SOURCE = (
    "from typing import List\n"
    "\n"
    "\n"
    "def _probe(values=[]):\n"
    "    flag = len(values)\n"
    "    if values:\n"
    "        if flag:\n"
    "            return List\n"
    "    return None\n"
)


def _ruff_command() -> list[str] | None:
    """Return the argv prefix that invokes ruff, or ``None`` if unavailable."""
    binary = shutil.which("ruff")
    if binary:
        return [binary]
    probe = subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True)
    return [sys.executable, "-m", "ruff"] if probe.returncode == 0 else None


def _ruff() -> list[str]:
    ruff = _ruff_command()
    if ruff is None:
        pytest.skip("ruff not available")
    return ruff


def _scoped_files() -> list[str]:
    """Git-tracked ``main.py``, ``src/**/*.py`` and ``tests/**/*.py``."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", "main.py", "src/*.py", "tests/*.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    files = [line for line in result.stdout.splitlines() if line]
    assert "main.py" in files, "git ls-files found no tracked main.py"
    return files


def _family(code: str) -> str | None:
    match = FAMILY_PATTERN.match(code or "")
    return match.group(1) if match else None


class TestRuffUpBSim:
    """REQ-LINTRULES-BDRABCZUK-73E61E51C2F1, -509F90FC030A."""

    def test_ruff_config_flags_up_b_sim(self):
        ruff = _ruff()
        missing = {}
        for rel_path in _scoped_files():
            result = subprocess.run(
                [
                    *ruff,
                    "check",
                    "--no-cache",
                    "--config",
                    str(RUFF_TOML),
                    "--output-format",
                    "json",
                    "--stdin-filename",
                    rel_path,
                    "-",
                ],
                cwd=REPO_ROOT,
                input=PROBE_SOURCE,
                capture_output=True,
                text=True,
            )
            seen = {_family(finding["code"]) for finding in json.loads(result.stdout or "[]")}
            absent = [family for family in FAMILIES if family not in seen]
            if absent:
                missing[rel_path] = absent
        assert not missing, (
            "REQ-LINTRULES-BDRABCZUK-73E61E51C2F1: ruff.toml does not report "
            f"{'/'.join(FAMILIES)} findings in {len(missing)} file(s) of main.py / src/ / tests/ "
            f"(file: families not reported): {missing}"
        )

    def test_no_up_b_sim_findings(self):
        ruff = _ruff()
        files = _scoped_files()
        result = subprocess.run(
            [
                *ruff,
                "check",
                "--isolated",
                "--no-cache",
                "--target-version",
                TARGET_VERSION,
                "--select",
                ",".join(FAMILIES),
                "--output-format",
                "concise",
                *files,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        findings = [line for line in result.stdout.splitlines() if re.search(r": (UP|B|SIM)\d+ ", line)]
        assert not findings, (
            f"REQ-LINTRULES-BDRABCZUK-509F90FC030A: {len(findings)} UP/B/SIM finding(s) in main.py / src/ / tests/:\n"
            + "\n".join(findings)
        )
        assert result.returncode == 0, f"ruff exited {result.returncode}:\n{result.stdout}{result.stderr}"
