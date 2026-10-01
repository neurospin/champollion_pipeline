"""Tests for REQ-TEST-DEADCODE-01 — ``pixi run dead-code`` is clean on tracked files.

The requirement: ``pixi run dead-code`` shall report zero vulture findings over
the Git-tracked Python files under ``src/`` and ``tests/``, without excluding
any tracked file from the scan and without raising vulture's minimum
confidence above its default of 0.

The scan reuses the ``dead-code`` task's own arguments from ``pixi.toml`` (so a
whitelist file named there is honoured) and runs from the repo root (so a
``[tool.vulture]`` table in ``pyproject.toml`` is honoured). Only the
``src/``/``tests/`` directory arguments are swapped for the Git-tracked ``.py``
files beneath them, so stray untracked files in a working tree do not affect
the verdict. This file itself and any untracked ``*vulture*`` whitelist under
``tests/`` are added too, so the check is meaningful before they are committed.
"""

import fnmatch
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
PIXI_TOML = REPO_ROOT / "pixi.toml"
PYPROJECT = REPO_ROOT / "pyproject.toml"
TASK_NAME = "dead-code"
SCAN_DIRS = ("src", "tests")


def _task_argv() -> list[str]:
    """Return the ``dead-code`` task's argv after the ``vulture`` executable."""
    with PIXI_TOML.open("rb") as handle:
        task = tomllib.load(handle)["tasks"][TASK_NAME]
    command = task if isinstance(task, str) else task["cmd"]
    argv = shlex.split(command)
    if argv[:3] == ["python", "-m", "vulture"]:
        return argv[3:]
    assert argv[0] == "vulture", f"unexpected dead-code task command: {command!r}"
    return argv[1:]


def _vulture_config() -> dict:
    """Return ``pyproject.toml``'s ``[tool.vulture]`` table (empty if absent)."""
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle).get("tool", {}).get("vulture", {})


def _git_ls(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", *args, "--", *(f"{d}/*.py" for d in SCAN_DIRS)],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _tracked_files() -> list[str]:
    return _git_ls()


def _scan_files() -> list[str]:
    """Tracked files, plus this file and any untracked vulture whitelist."""
    files = set(_tracked_files())
    files.add(str(Path(__file__).resolve().relative_to(REPO_ROOT)))
    for path in _git_ls("--others", "--exclude-standard"):
        if path.startswith("tests/") and "vulture" in Path(path).name:
            files.add(path)
    return sorted(files)


def _is_scan_dir(arg: str) -> bool:
    return arg.strip("./").rstrip("/") in SCAN_DIRS


def _option_values(argv: list[str], option: str) -> list[str]:
    """Return every value given to ``option`` (``--opt v`` or ``--opt=v``)."""
    values = []
    for i, arg in enumerate(argv):
        if arg == option and i + 1 < len(argv):
            values.append(argv[i + 1])
        elif arg.startswith(option + "="):
            values.append(arg.split("=", 1)[1])
    return values


def _exclude_patterns() -> list[str]:
    patterns = []
    for value in _option_values(_task_argv(), "--exclude"):
        patterns.extend(p for p in value.split(",") if p)
    config_exclude = _vulture_config().get("exclude", [])
    if isinstance(config_exclude, str):
        config_exclude = config_exclude.split(",")
    patterns.extend(p for p in config_exclude if p)
    return patterns


def _matches_vulture_exclude(path: str, pattern: str) -> bool:
    """Mirror vulture's rule: a pattern without wildcards is wrapped in ``*``."""
    if not any(char in pattern for char in "*?["):
        pattern = f"*{pattern}*"
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(str(REPO_ROOT / path), pattern)


@pytest.mark.smoke
@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
class TestDeadCodeClean:
    """REQ-TEST-DEADCODE-01."""

    def test_vulture_reports_no_findings_on_tracked_files(self):
        argv = [arg for arg in _task_argv() if not _is_scan_dir(arg)]
        result = subprocess.run(
            [sys.executable, "-m", "vulture", *argv, *_scan_files()],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        findings = (result.stdout + result.stderr).strip()
        assert result.returncode == 0 and not findings, (
            "REQ-TEST-DEADCODE-01: vulture still reports findings on tracked src/ and tests/ "
            f"files (exit {result.returncode}):\n{findings}"
        )

    def test_min_confidence_is_not_raised(self):
        values = _option_values(_task_argv(), "--min-confidence")
        config_value = _vulture_config().get("min_confidence")
        if config_value is not None:
            values.append(str(config_value))
        raised = [v for v in values if float(v) > 0]
        assert not raised, f"REQ-TEST-DEADCODE-01: vulture min-confidence raised above 0: {raised}"

    def test_no_tracked_file_is_excluded(self):
        excluded = sorted(
            path
            for path in _tracked_files()
            for pattern in _exclude_patterns()
            if _matches_vulture_exclude(path, pattern)
        )
        assert not excluded, f"REQ-TEST-DEADCODE-01: tracked files excluded from the vulture scan: {excluded}"
