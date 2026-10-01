"""Tests for REQ-SLURM-01..09 — the tracked Jean-Zay training SLURM template.

# TASK-095

Jean-Zay job 372284 reported COMPLETED although training failed
(``pixi: command not found``): the cluster-local script had no strict mode.
``slurm/`` and ``scripts/slurm/`` stay gitignored (user decision 2026-10-01),
so a hardened template ships at ``scripts/templates/`` instead.

These tests parse the file only; no SLURM, GPU or pixi is needed.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit  # noqa: V107

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_REL = "scripts/templates/train_champollion.slurm.example"
TEMPLATE = PROJECT_ROOT / TEMPLATE_REL

_PIXI_RUN = re.compile(r"\bpixi\s+run\b")
_ENV_OPTS = ("-e", "--environment", "--manifest-path")


def _read_template() -> str:
    assert TEMPLATE.is_file(), f"training SLURM template missing: {TEMPLATE_REL}"
    return TEMPLATE.read_text(encoding="utf-8")


def _logical_lines(text: str) -> list[str]:
    """Join backslash continuations; drop blank and full-line comment lines."""
    lines: list[str] = []
    buf = ""
    for raw in text.splitlines():
        if raw.rstrip().endswith("\\"):
            buf += raw.rstrip()[:-1] + " "
            continue
        joined = (buf + raw).strip()
        buf = ""
        if joined and not joined.startswith("#"):
            lines.append(joined)
    if buf.strip():
        lines.append(buf.strip())
    return lines


def _pixi_run_lines(lines: list[str]) -> list[tuple[int, str]]:
    return [(i, line) for i, line in enumerate(lines) if _PIXI_RUN.search(line)]


def _task_of(line: str) -> str | None:
    """First non-option argument after ``pixi run`` (the task/command name)."""
    tokens = shlex.split(line, posix=True)
    try:
        start = tokens.index("run", tokens.index("pixi")) + 1
    except ValueError:
        return None
    rest = tokens[start:]
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok in _ENV_OPTS:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        return tok
    return None


def _train_line_index(lines: list[str]) -> int:
    hits = [i for i, line in _pixi_run_lines(lines) if _task_of(line) == "train"]
    assert hits, "no `pixi run ... train` invocation found in the template"
    return hits[0]


class TestTemplateTracked:
    """REQ-SLURM-01: the template exists at a non-gitignored path."""

    def test_template_file_exists(self) -> None:
        assert TEMPLATE.is_file(), f"training SLURM template missing: {TEMPLATE_REL}"

    @pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
    def test_template_path_not_gitignored(self) -> None:
        result = subprocess.run(
            ["git", "check-ignore", "-q", TEMPLATE_REL],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 1, (
            f"{TEMPLATE_REL} is matched by a .gitignore rule (git check-ignore exit {result.returncode})"
        )


class TestStrictMode:
    """REQ-SLURM-02: ``set -euo pipefail`` precedes the first pixi call."""

    def test_set_euo_pipefail_precedes_first_pixi_call(self) -> None:
        lines = _logical_lines(_read_template())
        strict = [i for i, line in enumerate(lines) if re.match(r"set\s+-euo\s+pipefail\b", line)]
        pixi = [i for i, line in enumerate(lines) if re.search(r"\bpixi\b", line)]
        assert pixi, "template invokes pixi nowhere"
        assert strict, "template never runs `set -euo pipefail`"
        assert strict[0] < pixi[0], (
            f"`set -euo pipefail` (line {strict[0]}) must precede first pixi call: {lines[pixi[0]]!r}"
        )


class TestPixiInvocations:
    """REQ-SLURM-03 / REQ-SLURM-04: every ``pixi run`` is explicit and frozen."""

    def test_every_pixi_run_selects_training_env(self) -> None:
        runs = _pixi_run_lines(_logical_lines(_read_template()))
        assert runs, "template contains no `pixi run` invocation"
        env = re.compile(r"(?:^|\s)(?:-e|--environment)(?:\s+|=)[\"']?training[\"']?(?:\s|$)")
        offenders = [line for _, line in runs if not env.search(line)]
        assert not offenders, f"`pixi run` without literal `-e training`: {offenders}"

    def test_every_pixi_run_is_frozen(self) -> None:
        runs = _pixi_run_lines(_logical_lines(_read_template()))
        assert runs, "template contains no `pixi run` invocation"
        offenders = [line for _, line in runs if not re.search(r"(?:^|\s)--frozen(?:\s|$)", line)]
        assert not offenders, f"`pixi run` without `--frozen`: {offenders}"


class TestGpuPreflight:
    """REQ-SLURM-05: a CUDA availability check runs before the train task."""

    def test_cuda_check_precedes_train_task(self) -> None:
        lines = _logical_lines(_read_template())
        train_idx = _train_line_index(lines)
        checks = [i for i, line in _pixi_run_lines(lines) if "torch.cuda.is_available()" in line and i < train_idx]
        assert checks, "no `pixi run ... python ... torch.cuda.is_available()` check before the train task"


class TestUnbufferedOutput:
    """REQ-SLURM-06: PYTHONUNBUFFERED=1 is exported before the train task."""

    def test_pythonunbuffered_exported_before_train(self) -> None:
        lines = _logical_lines(_read_template())
        train_idx = _train_line_index(lines)
        exports = [
            i for i, line in enumerate(lines) if re.match(r"export\s+PYTHONUNBUFFERED=[\"']?1[\"']?(?:\s|;|$)", line)
        ]
        assert exports, "template never runs `export PYTHONUNBUFFERED=1`"
        assert exports[0] < train_idx, "`export PYTHONUNBUFFERED=1` must precede the train task"


class TestBashSyntax:
    """REQ-SLURM-07: ``bash -n`` accepts the template."""

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
    def test_bash_n_accepts_template(self) -> None:
        _read_template()
        result = subprocess.run(["bash", "-n", str(TEMPLATE)], capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"bash -n failed: {result.stderr}"


class TestNoPersonalIdentifiers:
    """REQ-SLURM-08: no /lustre/ paths, login or project account."""

    def test_no_lustre_paths(self) -> None:
        assert "/lustre/" not in _read_template(), "template hardcodes a /lustre/ path"

    @pytest.mark.parametrize("word", ["ugb35mu", "miu"])
    def test_no_login_or_project_account(self, word: str) -> None:
        text = _read_template()
        assert not re.search(rf"\b{word}\b", text), f"template contains `{word}`"


class TestAccountPlaceholder:
    """REQ-SLURM-09: the SLURM account is a placeholder."""

    def test_sbatch_account_is_placeholder(self) -> None:
        lines = [line.strip() for line in _read_template().splitlines()]
        assert "#SBATCH --account=<ACCOUNT>" in lines, "template lacks the line `#SBATCH --account=<ACCOUNT>`"
