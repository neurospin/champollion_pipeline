"""Tests for the repo rule "imports at the top of the file", enforced by ruff PLC0415.

- REQ-IMPORTS-BDRABCZUK-F1E73426B9AA: ruff, run with the repository ``ruff.toml``,
  reports PLC0415 for a function-level import placed in a Python file under ``src/``
  other than ``src/champollion_pipeline/generate_snapshots.py``.
- REQ-IMPORTS-BDRABCZUK-E790A93354D3: PLC0415, applied without the ``ruff.toml``
  per-file ignores, reports zero findings in the Git-tracked Python files under
  ``src/`` other than ``generate_snapshots.py``.
- REQ-IMPORTS-BDRABCZUK-7F4E1210D32B: those files and ``main.py`` carry zero ``noqa``
  comments naming PLC0415.
- REQ-LAZYIMPORT-BDRABCZUK-B236C18DBD18: among the ``ruff.toml`` per-file-ignores keys,
  only ``main.py`` lists PLC0415, for the lazy stage imports of main.py.

``generate_snapshots.py`` is exempt until its guarded top-level imports land
(TASK-REFACTOR-BDRABCZUK-33C4A004C981).
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

pytestmark = pytest.mark.smoke  # noqa: V107

REPO_ROOT = Path(__file__).resolve().parents[1]
RUFF_TOML = REPO_ROOT / "ruff.toml"
RULE = "PLC0415"
EXEMPT = "src/champollion_pipeline/generate_snapshots.py"
LAZY_IMPORTS_FILE = "main.py"  # REQ-LAZYIMPORT-BDRABCZUK-B236C18DBD18: the one file exempt from PLC0415
MAX_SUPPRESSIONS = 0  # REQ-IMPORTS-BDRABCZUK-7F4E1210D32B: no PLC0415 noqa allowed
NOQA_PATTERN = re.compile(r"#\s*noqa:[^\n]*\bPLC0415\b")
PROBE_SOURCE = 'def _probe():\n    import json\n\n    return json.dumps("")\n'


def _ruff_command() -> list[str] | None:
    """Return the argv prefix that invokes ruff, or ``None`` if unavailable."""
    binary = shutil.which("ruff")
    if binary:
        return [binary]
    probe = subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True)
    return [sys.executable, "-m", "ruff"] if probe.returncode == 0 else None


def _scoped_files() -> list[str]:
    """Git-tracked ``src/**/*.py``, minus the exempt snapshots module."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", "src/*.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    files = [line for line in result.stdout.splitlines() if line and line != EXEMPT]
    assert files, "git ls-files found no tracked Python file under src/"
    return files


def _ruff() -> list[str]:
    ruff = _ruff_command()
    if ruff is None:
        pytest.skip("ruff not available")
    return ruff


def _probe_codes(ruff: list[str], rel_path: str) -> set[str]:
    """Rule codes ruff, run with ``ruff.toml``, reports for a function-level import placed in ``rel_path``."""
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
    return {finding["code"] for finding in json.loads(result.stdout or "[]")}


class TestImportsTopLevel:
    """REQ-IMPORTS-BDRABCZUK-F1E73426B9AA, -E790A93354D3, -7F4E1210D32B, REQ-LAZYIMPORT-BDRABCZUK-B236C18DBD18."""

    def test_ruff_config_flags_function_level_import(self):
        ruff = _ruff()
        unflagged = [rel_path for rel_path in _scoped_files() if RULE not in _probe_codes(ruff, rel_path)]
        assert not unflagged, (
            f"REQ-IMPORTS-BDRABCZUK-F1E73426B9AA: ruff.toml does not report {RULE} for a "
            f"function-level import in {len(unflagged)} file(s): {unflagged}"
        )

    def test_no_import_outside_top_level(self):
        ruff = _ruff()
        files = _scoped_files()
        result = subprocess.run(
            [*ruff, "check", "--isolated", "--no-cache", "--select", RULE, "--output-format", "concise", *files],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        findings = [line for line in result.stdout.splitlines() if f" {RULE} " in line]
        assert not findings, (
            f"REQ-IMPORTS-BDRABCZUK-E790A93354D3: {len(findings)} {RULE} finding(s) in src/:\n" + "\n".join(findings)
        )
        assert result.returncode == 0, f"ruff exited {result.returncode}:\n{result.stdout}{result.stderr}"

    def test_plc0415_suppressions_do_not_grow(self):
        per_file = {}
        for rel_path in [*_scoped_files(), LAZY_IMPORTS_FILE, EXEMPT]:
            path = REPO_ROOT / rel_path
            if path.is_file():
                hits = len(NOQA_PATTERN.findall(path.read_text(encoding="utf-8")))
                if hits:
                    per_file[rel_path] = hits
        total = sum(per_file.values())
        assert total <= MAX_SUPPRESSIONS, (
            f"REQ-IMPORTS-BDRABCZUK-7F4E1210D32B: {total} noqa comments name {RULE} "
            f"(limit {MAX_SUPPRESSIONS}): {per_file}"
        )

    def test_plc0415_per_file_ignore_names_main_py_alone(self):
        table = tomllib.loads(RUFF_TOML.read_text(encoding="utf-8"))["lint"]["per-file-ignores"]
        keys = sorted(key for key, rules in table.items() if RULE in rules)
        assert keys == [LAZY_IMPORTS_FILE], (
            f"REQ-LAZYIMPORT-BDRABCZUK-B236C18DBD18: ruff.toml per-file-ignores list {RULE} under {keys}"
        )

    def test_ruff_config_leaves_function_level_import_in_main_py_unflagged(self):
        codes = _probe_codes(_ruff(), LAZY_IMPORTS_FILE)
        assert RULE not in codes, (
            f"REQ-LAZYIMPORT-BDRABCZUK-B236C18DBD18: ruff.toml reports {RULE} for a function-level import in "
            f"{LAZY_IMPORTS_FILE}"
        )
