"""Tests for the ruff complexity rules C901 (mccabe), PLR0913 and PLR0915.

- REQ-LINTRULES-BDRABCZUK-31EAF3FE7BDA: ruff, run with the repository ``ruff.toml``,
  reports C901 findings above complexity 15, PLR0913 and PLR0915 findings in ``main.py``
  and in each Python file under ``src/``, ``tests/`` or ``scripts/``.
- REQ-LINTRULES-BDRABCZUK-632CC2004858: C901 at max-complexity 15, with PLR0913 and
  PLR0915 at the ruff default thresholds, reports zero findings in ``main.py`` and the
  Git-tracked Python files under ``src/``, ``tests/`` or ``scripts/``.
- REQ-LINTRULES-BDRABCZUK-29014F43463F: each noqa comment suppressing C901, PLR0913 or
  PLR0915 in those files states a reason after the rule codes.
- REQ-LINTRULES-BDRABCZUK-C24EEB2D9BBF: ignoring noqa comments, C901 at max-complexity 10
  reports no finding for ``PipelineOrchestrator.run`` in ``main.py``.
- REQ-LINTRULES-BDRABCZUK-9C21AB5C50AB: ignoring noqa comments, PLR0915 reports no finding
  for ``PipelineOrchestrator.run`` in ``main.py``.
- REQ-LINTRULES-BDRABCZUK-F5B485D4C928: ignoring noqa comments, C901 at max-complexity 10
  reports no finding for the ``_apply_cli_overrides`` function of ``main.py``.

REQ-LINTRULES-BDRABCZUK-3A48E7C78B13 (each test keeps its outcome from commit 5839e8c) is
verified by comparing full-suite outcomes, not by a test in this file.
"""

import ast
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
MAIN_PY = REPO_ROOT / "main.py"
RULES = ("C901", "PLR0913", "PLR0915")
CONFIG_MAX_COMPLEXITY = 15
REFACTORED_MAX_COMPLEXITY = 10
NOQA_PATTERN = re.compile(r"#\s*noqa:\s*(?P<codes>[A-Z]+[0-9]+(?:\s*,\s*[A-Z]+[0-9]+)*)(?P<rest>.*)$")


def _probe_source() -> str:
    """One function with 6 arguments, complexity 17 and more than 50 statements."""
    lines = ["def _probe(a, b, c, d, e, f):"]
    lines += [f"    if a == {i}:\n        b += {i}" for i in range(16)]
    lines += ["    c = c + 1"] * 20
    lines.append("    return a, b, c, d, e, f")
    return "\n".join(lines) + "\n"


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


def _isolated_findings(rule: str, paths: list[str], *, max_complexity: int, ignore_noqa: bool) -> list[dict]:
    """Run ruff with only *rule* selected, outside the repository config; JSON findings."""
    argv = [
        *_ruff(),
        "check",
        "--isolated",
        "--no-cache",
        "--select",
        rule,
        "--config",
        f"lint.mccabe.max-complexity={max_complexity}",
        "--output-format",
        "json",
    ]
    if ignore_noqa:
        argv.append("--ignore-noqa")
    result = subprocess.run([*argv, *paths], cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode in (0, 1), f"ruff exited {result.returncode}:\n{result.stderr}"
    return json.loads(result.stdout or "[]")


def _def_row(qualname: str) -> int:
    """Line of the ``def`` of *qualname* (``func`` or ``Class.method``) in main.py."""
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))
    parts = qualname.split(".")
    body = tree.body
    node = None
    for part in parts:
        node = next(
            (
                n
                for n in body
                if isinstance(n, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and n.name == part
            ),
            None,
        )
        assert node is not None, f"{qualname} not found in main.py"
        body = node.body
    return node.lineno


def _function_findings(rule: str, qualname: str, *, max_complexity: int = CONFIG_MAX_COMPLEXITY) -> list[dict]:
    row = _def_row(qualname)
    findings = _isolated_findings(rule, ["main.py"], max_complexity=max_complexity, ignore_noqa=True)
    return [f for f in findings if f["code"] == rule and f["location"]["row"] == row]


class TestRuffComplexity:
    """REQ-LINTRULES-BDRABCZUK-31EAF3FE7BDA, -632CC2004858, -29014F43463F."""

    def test_ruff_config_flags_complexity_rules(self):
        ruff = _ruff()
        source = _probe_source()
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
                input=source,
                capture_output=True,
                text=True,
            )
            seen = {finding["code"] for finding in json.loads(result.stdout or "[]")}
            absent = [rule for rule in RULES if rule not in seen]
            if absent:
                missing[rel_path] = absent
        assert not missing, (
            "REQ-LINTRULES-BDRABCZUK-31EAF3FE7BDA: ruff.toml does not report "
            f"{'/'.join(RULES)} findings in {len(missing)} file(s) of main.py / src/ / tests/ / scripts/ "
            f"(file: rules not reported): {dict(list(missing.items())[:5])}"
        )

    def test_ruff_config_max_complexity_is_15(self):
        ruff = _ruff()
        # complexity 16 must be reported, complexity 15 must not
        reported = {}
        for ifs in (14, 15):
            lines = ["def _probe(a):"] + [f"    if a == {i}:\n        a += {i}" for i in range(ifs)] + ["    return a"]
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
                    "main.py",
                    "-",
                ],
                cwd=REPO_ROOT,
                input="\n".join(lines) + "\n",
                capture_output=True,
                text=True,
            )
            reported[ifs + 1] = any(f["code"] == "C901" for f in json.loads(result.stdout or "[]"))
        assert reported == {15: False, 16: True}, (
            "REQ-LINTRULES-BDRABCZUK-31EAF3FE7BDA: ruff.toml must report C901 above complexity 15 only "
            f"(complexity: reported) {reported}"
        )

    def test_no_complexity_findings(self):
        files = _scoped_files()
        findings = []
        for rule in RULES:
            findings += _isolated_findings(rule, files, max_complexity=CONFIG_MAX_COMPLEXITY, ignore_noqa=False)
        lines = [f"{f['filename']}:{f['location']['row']}: {f['code']} {f['message']}" for f in findings]
        assert not lines, (
            f"REQ-LINTRULES-BDRABCZUK-632CC2004858: {len(lines)} C901/PLR0913/PLR0915 finding(s) "
            "in main.py / src/ / tests/ / scripts/:\n" + "\n".join(lines)
        )

    def test_complexity_noqa_states_reason(self):
        bare = []
        for rel_path in _scoped_files():
            text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
            for number, line in enumerate(text.splitlines(), start=1):
                match = NOQA_PATTERN.search(line)
                if not match:
                    continue
                codes = {code.strip() for code in match.group("codes").split(",")}
                if codes & set(RULES) and not re.search(r"\w", match.group("rest")):
                    bare.append(f"{rel_path}:{number}: {line.strip()}")
        assert not bare, (
            "REQ-LINTRULES-BDRABCZUK-29014F43463F: noqa comment(s) suppressing C901/PLR0913/PLR0915 "
            "without a reason:\n" + "\n".join(bare)
        )


class TestRefactoredFunctions:
    """REQ-LINTRULES-BDRABCZUK-C24EEB2D9BBF, -9C21AB5C50AB, -F5B485D4C928."""

    def test_orchestrator_run_complexity(self):
        findings = _function_findings("C901", "PipelineOrchestrator.run", max_complexity=REFACTORED_MAX_COMPLEXITY)
        assert not findings, (
            "REQ-LINTRULES-BDRABCZUK-C24EEB2D9BBF: PipelineOrchestrator.run exceeds complexity "
            f"{REFACTORED_MAX_COMPLEXITY}: {[f['message'] for f in findings]}"
        )

    def test_orchestrator_run_statements(self):
        findings = _function_findings("PLR0915", "PipelineOrchestrator.run")
        assert not findings, (
            "REQ-LINTRULES-BDRABCZUK-9C21AB5C50AB: PipelineOrchestrator.run has too many statements: "
            f"{[f['message'] for f in findings]}"
        )

    def test_apply_cli_overrides_complexity(self):
        findings = _function_findings("C901", "_apply_cli_overrides", max_complexity=REFACTORED_MAX_COMPLEXITY)
        assert not findings, (
            "REQ-LINTRULES-BDRABCZUK-F5B485D4C928: _apply_cli_overrides exceeds complexity "
            f"{REFACTORED_MAX_COMPLEXITY}: {[f['message'] for f in findings]}"
        )
