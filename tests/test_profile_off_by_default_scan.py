#!/usr/bin/env python3
"""
Profiling is never on by default in what the repository ships (profiling design Contract 0, test 0c).

Scanned: pixi.toml, reference.yaml, pipeline_loop_2mm.json, install.sh, the YAML files under src/, the
files under slurm/ and the files under scripts/ outside scripts/e2e/ (the e2e runner profiles every run
on purpose: user decision 2026-10-27, Q17).

Requirements:
- REQ-PROFILING-BDRABCZUK-2C976935B914: no profile_dir key with a non-null value.
- REQ-PROFILING-BDRABCZUK-22BEFAA2777A: no --profile command-line option.
- REQ-PROFILING-BDRABCZUK-A32131FA33E0: no assignment to the CHAMPOLLION_PROFILE environment variable.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOP_LEVEL_FILES = ("pixi.toml", "reference.yaml", "pipeline_loop_2mm.json", "install.sh")
EXEMPT_DIR = REPO_ROOT / "scripts" / "e2e"
MAX_BYTES = 2_000_000
# profile_dir is a config key: Python sources may name a variable or annotate a parameter profile_dir.
CONFIG_SUFFIXES = {".yaml", ".yml", ".json", ".toml", ".sh", ".env", ".cfg", ".ini", ""}

NON_NULL_PROFILE_DIR = re.compile(
    r"""["']?\bprofile_dir["']?[ \t]*[:=](?!=)(?![ \t]*(?:null\b|None\b|~|""|''|$|[,}#]))""", re.MULTILINE
)
PROFILE_OPTION = re.compile(r"(?<![\w-])--profile(?![\w-])")
PROFILE_ASSIGNMENT = re.compile(
    r"""\bCHAMPOLLION_PROFILE\b["']?\]?\s*=(?!=)"""
    r"""|\bexport\s+CHAMPOLLION_PROFILE\b"""
    r"""|\bsetdefault\(\s*["']CHAMPOLLION_PROFILE["']"""
)


def _scanned_files() -> list[Path]:
    """The shipped files of Contract 0 test 0c that exist in this checkout."""
    files = [REPO_ROOT / name for name in TOP_LEVEL_FILES]
    files += [*(REPO_ROOT / "src").rglob("*.yaml"), *(REPO_ROOT / "src").rglob("*.yml")]
    for directory in (REPO_ROOT / "slurm", REPO_ROOT / "scripts"):
        files += [path for path in directory.rglob("*") if path.is_file()]
    return sorted(
        path for path in files if path.is_file() and EXEMPT_DIR not in path.parents and path.stat().st_size <= MAX_BYTES
    )


def _findings(pattern: re.Pattern, suffixes: set[str] | None = None) -> list[str]:
    found = []
    for path in _scanned_files():
        if suffixes is not None and path.suffix not in suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            found.append(f"{path.relative_to(REPO_ROOT)}:{line}: {match.group(0).strip()}")
    return found


@pytest.mark.unit
def test_scan_covers_top_level_files():
    """The scan reads the top-level shipped files (guards against a scan that silently reads nothing)."""
    scanned = {path.relative_to(REPO_ROOT).as_posix() for path in _scanned_files()}
    assert "pixi.toml" in scanned
    assert "reference.yaml" in scanned


@pytest.mark.unit
def test_no_shipped_non_null_profile_dir():
    """REQ-PROFILING-BDRABCZUK-2C976935B914: no shipped file sets profile_dir to a non-null value."""
    assert _findings(NON_NULL_PROFILE_DIR, CONFIG_SUFFIXES) == []


@pytest.mark.unit
def test_no_shipped_profile_option():
    """REQ-PROFILING-BDRABCZUK-22BEFAA2777A: no shipped file passes --profile."""
    assert _findings(PROFILE_OPTION) == []


@pytest.mark.unit
def test_no_shipped_champollion_profile_assignment():
    """REQ-PROFILING-BDRABCZUK-A32131FA33E0: no shipped file assigns CHAMPOLLION_PROFILE."""
    assert _findings(PROFILE_ASSIGNMENT) == []
