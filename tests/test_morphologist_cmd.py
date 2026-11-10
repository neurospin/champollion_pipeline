#!/usr/bin/env python3
"""
Regression tests for the morphologist-cli command built by
generate_morphologist_graphs.py (REQ-MORPHO-01).

Guards the explicit `--if morphologist-auto-nonoverlap-1.0` input-format
flag: without it, morphologist-cli falls back to its own default input
format, morphologist-bids-2.0, whatever the input filenames.

Uses AST source-parsing instead of runtime import to avoid the
champollion_utils dependency in the test environment.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = PROJECT_ROOT / "src" / "champollion_pipeline" / "generate_morphologist_graphs.py"

FORMAT = "morphologist-auto-nonoverlap-1.0"
BIDS_FORMAT = "morphologist-bids-2.0"


def _run_methods() -> list[ast.FunctionDef]:
    """Return each `run` method of a class of generate_morphologist_graphs.py."""
    tree = ast.parse(SOURCE_FILE.read_text(encoding="utf-8"))
    return [
        item
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == "run"
    ]


def _default_text(value: ast.expr) -> str | None:
    """Return the text of a string constant, or of the `orelse` branch of `A if cond else "text"`, else None."""
    branch = value.orelse if isinstance(value, ast.IfExp) else value
    return branch.value if isinstance(branch, ast.Constant) and isinstance(branch.value, str) else None


def _parse_cmd_literal() -> list[str]:
    """Return the default-case string elements of the `cmd = [...]` list in `run()`.

    Walks the AST of generate_morphologist_graphs.py, finds the assignment
    ``cmd = [...]`` inside the ``run`` method body, and extracts the constant
    string nodes. A `Name` element referencing a same-scope string constant,
    or the `orelse` branch of a same-scope `x = A if cond else B`, is resolved
    to that text. Other non-string nodes (e.g. starred unpacks, f-strings) are
    skipped.
    """
    for run in _run_methods():
        assignments = [stmt for stmt in run.body if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1]
        named = [stmt for stmt in assignments if isinstance(stmt.targets[0], ast.Name)]
        texts = {stmt.targets[0].id: _default_text(stmt.value) for stmt in named}
        for stmt in named:
            if stmt.targets[0].id != "cmd" or not isinstance(stmt.value, ast.List):
                continue
            result = []
            for elt in stmt.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    result.append(elt.value)
                elif isinstance(elt, ast.Name) and texts.get(elt.id):
                    result.append(texts[elt.id])
            return result
    return []


@pytest.mark.smoke
class TestMorphologistInputFormatFlag:
    """REQ-MORPHO-01 — explicit --if before --of, after the -- separator."""

    def test_cmd_contains_input_format_flag(self):
        """The command passes --if explicitly, not relying on auto-detection."""
        assert "--if" in _parse_cmd_literal()

    def test_input_format_flag_precedes_output_format_flag(self):
        """--if is positioned before --of in the command."""
        cmd = _parse_cmd_literal()
        assert "--if" in cmd and "--of" in cmd
        assert cmd.index("--if") < cmd.index("--of")

    def test_both_format_flags_use_auto_nonoverlap(self):
        """REQ-BIDS-BDRABCZUK-579916807967: --if and --of are this format, and run() names no other, --bids included."""
        cmd = _parse_cmd_literal()
        assert "--if" in cmd and "--of" in cmd
        assert cmd[cmd.index("--if") + 1] == FORMAT
        assert cmd[cmd.index("--of") + 1] == FORMAT
        constants = [node.value for run in _run_methods() for node in ast.walk(run) if isinstance(node, ast.Constant)]
        assert BIDS_FORMAT not in constants

    def test_input_format_flag_follows_double_dash_separator(self):
        """--if sits after --, marking it a morphologist param, not a CLI flag."""
        cmd = _parse_cmd_literal()
        assert "--" in cmd and "--if" in cmd
        assert cmd.index("--") < cmd.index("--if")
