"""Cyclomatic complexity of the compare sources (TASK-REFACTOR-BDRABCZUK-79129AACE6AB).

Requirement REQ-COMPARE-BDRABCZUK-72B72A943D1E: Ruff rule C901 with
max-complexity 15 shall report zero findings in each Python file whose path
under src/ begins with compare.

Ruff runs from the repo root, so ``ruff.toml`` (excludes, line length) applies;
only the rule selection and the McCabe threshold are overridden here.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
MAX_COMPLEXITY = 15


def _ruff_command() -> list[str] | None:
    """Return the argv prefix that invokes ruff, or ``None`` if unavailable."""
    binary = shutil.which("ruff")
    if binary:
        return [binary]
    probe = subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True)
    return [sys.executable, "-m", "ruff"] if probe.returncode == 0 else None


def _compare_files() -> list[Path]:
    """Python files whose path relative to src/ begins with 'compare'."""
    return sorted(
        path
        for path in SRC_DIR.rglob("*.py")
        if "__pycache__" not in path.parts and path.relative_to(SRC_DIR).as_posix().startswith("compare")
    )


COMPARE_FILES = _compare_files()


def test_compare_files_found():
    assert COMPARE_FILES, "no Python file under src/ begins with 'compare'"


@pytest.mark.parametrize("path", COMPARE_FILES, ids=lambda p: p.relative_to(SRC_DIR).as_posix())
def test_compare_file_has_no_c901_finding(path: Path):
    ruff = _ruff_command()
    if ruff is None:
        pytest.skip("ruff is not installed")
    result = subprocess.run(
        [
            *ruff,
            "check",
            "--select",
            "C901",
            "--config",
            f"lint.mccabe.max-complexity={MAX_COMPLEXITY}",
            "--output-format",
            "concise",
            "--no-cache",
            str(path.relative_to(REPO_ROOT)),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"REQ-COMPARE-BDRABCZUK-72B72A943D1E: C901 findings above {MAX_COMPLEXITY} in "
        f"{path.relative_to(REPO_ROOT)}:\n{result.stdout}{result.stderr}"
    )
