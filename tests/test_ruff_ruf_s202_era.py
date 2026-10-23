"""Tests for the ruff rules RUF (Ruff-specific), S202 (tarfile-unsafe-members) and ERA001 (commented-out code).

- REQ-LINTRULES-BDRABCZUK-92710C241D0F: ruff, run with the repository ``ruff.toml``, reports RUF findings
  other than RUF001 and RUF002, S202 findings and ERA001 findings in ``main.py`` and in each Python file
  under ``src/``, ``tests/`` or ``scripts/``.
- REQ-LINTRULES-BDRABCZUK-77267008F10F: ruff, run with the repository ``ruff.toml`` minus its per-file
  ignores, reports zero RUF findings other than RUF001 and RUF002, zero S202 findings and zero ERA001
  findings in ``main.py`` and in the Git-tracked Python files under ``src/``, ``tests/`` or ``scripts/``.
- REQ-LINTRULES-BDRABCZUK-8D79A928EBAE: ruff, run with the repository ``ruff.toml``, reports no
  invalid-rule-code (RUF102) finding for a noqa comment naming the vulture codes V101, V103, V105 or V107.

REQ-LINTRULES-BDRABCZUK-7719C6F5C3FB (each test keeps its outcome from commit 9ff330f) is verified by
comparing full-suite outcomes, not by a test in this file.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke  # noqa: V107

REPO_ROOT = Path(__file__).resolve().parents[1]
RUFF_TOML = REPO_ROOT / "ruff.toml"
EXEMPT_RUF_RULES = ("RUF001", "RUF002")
VULTURE_CODES = ("V101", "V103", "V105", "V107")

# One finding per probed rule: RUF005, RUF010, S202, ERA001.
PROBE_SOURCE = "\n".join(
    [
        "import tarfile",
        "",
        "a = [1]",
        "b = [0] + a",
        'c = f"{str(a)}"',
        "# b = a + 1",
        'with tarfile.open("x.tar") as archive:',
        '    archive.extractall("out")',
        "",
    ]
)
PROBE_RULES = ("RUF005", "RUF010", "S202", "ERA001")

# Ambiguous unicode in a string (RUF001) and a docstring (RUF002): must stay unreported.
EXEMPT_PROBE_SOURCE = '"""Crop of 2 × 2 voxels."""\n\nLABEL = "2 × 2"\n'


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
    """Git-tracked ``main.py``, ``src/**/*.py``, ``tests/**/*.py`` and ``scripts/**/*.py``."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", "main.py", "src/*.py", "tests/*.py", "scripts/*.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    files = [line for line in result.stdout.splitlines() if line]
    assert "main.py" in files, "git ls-files found no tracked main.py"
    return files


def _stdin_codes(source: str, rel_path: str, *extra: str) -> set[str]:
    """Rule codes ruff reports for *source* checked as *rel_path* with the repository ruff.toml."""
    result = subprocess.run(
        [
            *_ruff(),
            "check",
            "--no-cache",
            "--config",
            str(RUFF_TOML),
            *extra,
            "--output-format",
            "json",
            "--stdin-filename",
            rel_path,
            "-",
        ],
        cwd=REPO_ROOT,
        input=source,
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 1), f"ruff exited {result.returncode}:\n{result.stderr}"
    return {finding["code"] for finding in json.loads(result.stdout or "[]")}


class TestRuffRufS202Era:
    """REQ-LINTRULES-BDRABCZUK-92710C241D0F, -77267008F10F, -8D79A928EBAE."""

    def test_ruff_config_flags_ruf_s202_era(self):
        missing = {}
        for rel_path in _scoped_files():
            seen = _stdin_codes(PROBE_SOURCE, rel_path)
            absent = [rule for rule in PROBE_RULES if rule not in seen]
            if absent:
                missing[rel_path] = absent
        assert not missing, (
            "REQ-LINTRULES-BDRABCZUK-92710C241D0F: ruff.toml does not report "
            f"{'/'.join(PROBE_RULES)} findings in {len(missing)} file(s) of main.py / src/ / tests/ / scripts/ "
            f"(file: rules not reported): {dict(list(missing.items())[:5])}"
        )

    def test_ruff_config_skips_ruf001_ruf002(self):
        reported = {}
        for rel_path in ("main.py", "src/champollion_pipeline/generate_masks.py", "tests/test_compare.py"):
            seen = _stdin_codes(EXEMPT_PROBE_SOURCE, rel_path) & set(EXEMPT_RUF_RULES)
            if seen:
                reported[rel_path] = sorted(seen)
        assert not reported, (
            f"REQ-LINTRULES-BDRABCZUK-92710C241D0F: ruff.toml reports {'/'.join(EXEMPT_RUF_RULES)}, "
            f"which stay exempt (file: rules reported): {reported}"
        )

    def test_no_ruf_s202_era_findings(self):
        files = _scoped_files()
        result = subprocess.run(
            [
                *_ruff(),
                "check",
                "--no-cache",
                "--config",
                str(RUFF_TOML),
                "--config",
                "lint.per-file-ignores = {}",
                "--extend-select",
                "RUF,S202,ERA001",
                "--ignore",
                ",".join(EXEMPT_RUF_RULES),
                "--output-format",
                "json",
                *files,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode in (0, 1), f"ruff exited {result.returncode}:\n{result.stderr}"
        findings = [
            f
            for f in json.loads(result.stdout or "[]")
            if f["code"] and (f["code"].startswith("RUF") or f["code"] in ("S202", "ERA001"))
        ]
        lines = [
            f"{Path(f['filename']).relative_to(REPO_ROOT)}:{f['location']['row']}: {f['code']} {f['message']}"
            for f in findings
        ]
        assert not lines, (
            f"REQ-LINTRULES-BDRABCZUK-77267008F10F: {len(lines)} RUF/S202/ERA001 finding(s) "
            "in main.py / src/ / tests/ / scripts/:\n" + "\n".join(lines[:60])
        )

    def test_ruff_config_accepts_vulture_noqa_codes(self):
        source = "\n".join(f"value_{code.lower()} = 1  # noqa: {code}" for code in VULTURE_CODES) + "\n"
        reported = {}
        for rel_path in ("main.py", "src/champollion_pipeline/generate_masks.py", "tests/test_compare.py"):
            seen = _stdin_codes(source, rel_path, "--extend-select", "RUF102")
            if "RUF102" in seen:
                reported[rel_path] = "RUF102"
        assert not reported, (
            "REQ-LINTRULES-BDRABCZUK-8D79A928EBAE: ruff.toml reports invalid-rule-code (RUF102) for noqa "
            f"comments naming {'/'.join(VULTURE_CODES)}: {reported}"
        )
