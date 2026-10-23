"""Tests for the repo rule "imports at the top of the file" applied to ``tests/`` and ``scripts/``.

- REQ-IMPORTS-BDRABCZUK-E0B759C5CEA0: PLC0415, applied without the ``ruff.toml``
  per-file ignores, reports zero findings in the Git-tracked Python files under
  ``tests/`` and ``scripts/``.
- REQ-IMPORTS-BDRABCZUK-A78F2F345A12: ruff, run with the repository ``ruff.toml``,
  reports PLC0415 for a function-level import placed in a Python file under
  ``tests/`` or ``scripts/``.
- REQ-IMPORTS-BDRABCZUK-F3EF5550617E: those files carry zero ``noqa`` comments
  naming PLC0415.
- REQ-IMPORTS-BDRABCZUK-7F5CEC0AD821: pytest collects
  ``tests/test_installation_sanity.py`` while a module one of its tests checks
  cannot be imported.
- REQ-IMPORTS-BDRABCZUK-EDA99EC97FFA: those checks import the module through
  ``importlib.import_module`` inside the test, which ruff accepts, so each check
  still fails on its own.
"""

import ast
import io
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import tokenize
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUFF_TOML = REPO_ROOT / "ruff.toml"
RULE = "PLC0415"
SCOPE = ("tests/*.py", "scripts/*.py")
NOQA_PATTERN = re.compile(r"noqa:[^#]*\bPLC0415\b")
PROBE_SOURCE = 'def _probe():\n    import json\n\n    return json.dumps("")\n'
SANITY_FILE = REPO_ROOT / "tests" / "test_installation_sanity.py"
# Top-level packages whose importability tests/test_installation_sanity.py checks.
CHECKED_PACKAGES = (
    "gmpy2",
    "sympy",
    "torch",
    "champollion_utils",
    "cortical_tiles",
    "pandas",
    "huggingface_hub",
    "champollion_pipeline",
)
COLLECT_CHILD = textwrap.dedent(
    """
    import sys

    blocked = sys.argv[1]
    sys.modules[blocked] = None
    import pytest

    sys.exit(pytest.main(["--collect-only", "-q", "-p", "no:cacheprovider", "-o", "addopts=", sys.argv[2]]))
    """
)


def _ruff() -> list[str]:
    """Return the argv prefix that invokes ruff, or skip when ruff is unavailable."""
    binary = shutil.which("ruff")
    if binary:
        return [binary]
    probe = subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("ruff not available")
    return [sys.executable, "-m", "ruff"]


def _scoped_files() -> list[str]:
    """Git-tracked Python files directly under ``tests/`` and ``scripts/``."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", *SCOPE],
        check=True,
        capture_output=True,
        text=True,
    )
    files = [line for line in result.stdout.splitlines() if line]
    assert any(f.startswith("tests/") for f in files), "git ls-files found no tracked tests/ Python file"
    assert any(f.startswith("scripts/") for f in files), "git ls-files found no tracked scripts/ Python file"
    return files


@pytest.mark.smoke
class TestImportsTopLevelTestsScripts:
    """REQ-IMPORTS-BDRABCZUK-E0B759C5CEA0, -A78F2F345A12, -F3EF5550617E."""

    def test_no_import_outside_top_level(self):
        result = subprocess.run(
            [
                *_ruff(),
                "check",
                "--isolated",
                "--no-cache",
                "--select",
                RULE,
                "--output-format",
                "concise",
                *_scoped_files(),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        findings = [line for line in result.stdout.splitlines() if f" {RULE} " in line]
        assert not findings, (
            f"REQ-IMPORTS-BDRABCZUK-E0B759C5CEA0: {len(findings)} {RULE} finding(s) in tests/ and scripts/:\n"
            + "\n".join(findings)
        )
        assert result.returncode == 0, f"ruff exited {result.returncode}:\n{result.stdout}{result.stderr}"

    def test_ruff_config_flags_function_level_import(self):
        ruff = _ruff()
        unflagged = []
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
            codes = {finding["code"] for finding in json.loads(result.stdout or "[]")}
            if RULE not in codes:
                unflagged.append(rel_path)
        assert not unflagged, (
            f"REQ-IMPORTS-BDRABCZUK-A78F2F345A12: ruff.toml does not report {RULE} for a "
            f"function-level import in {len(unflagged)} file(s), e.g. {unflagged[:5]}"
        )

    def test_no_plc0415_suppressions(self):
        per_file = {}
        for rel_path in _scoped_files():
            source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
            tokens = tokenize.generate_tokens(io.StringIO(source).readline)
            hits = sum(1 for tok in tokens if tok.type == tokenize.COMMENT and NOQA_PATTERN.search(tok.string))
            if hits:
                per_file[rel_path] = hits
        assert not per_file, f"REQ-IMPORTS-BDRABCZUK-F3EF5550617E: noqa comments name {RULE}: {per_file}"


class TestInstallationSanityIsolation:
    """REQ-IMPORTS-BDRABCZUK-7F5CEC0AD821, -EDA99EC97FFA."""

    @pytest.mark.parametrize("package", CHECKED_PACKAGES)
    def test_collects_with_checked_module_missing(self, package):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT / "src"), env.get("PYTHONPATH")]))
        result = subprocess.run(
            [sys.executable, "-c", COLLECT_CHILD, package, str(SANITY_FILE)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"REQ-IMPORTS-BDRABCZUK-7F5CEC0AD821: collecting {SANITY_FILE.name} with {package!r} "
            f"unimportable exited {result.returncode}:\n{result.stdout[-2000:]}{result.stderr[-2000:]}"
        )

    @pytest.mark.smoke
    def test_checked_modules_imported_through_importlib(self):
        tree = ast.parse(SANITY_FILE.read_text(encoding="utf-8"))
        through_importlib = set()
        statement_imports = []
        for func in (node for node in tree.body if isinstance(node, ast.FunctionDef)):
            for node in ast.walk(func):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    statement_imports.append(f"{func.name}:{node.lineno}")
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "import_module"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "importlib"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    through_importlib.add(node.args[0].value.split(".")[0])
        missing = sorted(set(CHECKED_PACKAGES) - through_importlib)
        assert not statement_imports and not missing, (
            "REQ-IMPORTS-BDRABCZUK-EDA99EC97FFA: test_installation_sanity.py checks must call "
            f"importlib.import_module inside the test; import statements inside tests at {statement_imports}, "
            f"packages not imported through importlib.import_module: {missing}"
        )
