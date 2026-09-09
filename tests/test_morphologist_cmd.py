#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression tests for the morphologist-cli command built by
generate_morphologist_graphs.py (REQ-MORPHO-01).

Guards the explicit `--if morphologist-auto-nonoverlap-1.0` input-format
flag: without it, morphologist-cli auto-detects BIDS when input filenames
carry BIDS entities (_acq-, _run-, ...) and silently overrides the input
format to morphologist-bids-2.0.

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


def _parse_cmd_literal() -> list[str]:
    """Return the string elements of the `cmd = [...]` list in `run()`.

    Walks the AST of generate_morphologist_graphs.py, finds the assignment
    ``cmd = [...]`` inside the ``run`` method body, and extracts the constant
    string nodes.  Non-string nodes (e.g. starred unpacks, f-strings) are
    skipped — the test only needs the static flag tokens.
    """
    source = SOURCE_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not (isinstance(item, ast.FunctionDef) and item.name == "run"):
                continue
            for stmt in item.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                targets = stmt.targets
                if not (len(targets) == 1 and isinstance(targets[0], ast.Name) and targets[0].id == "cmd"):
                    continue
                if not isinstance(stmt.value, ast.List):
                    continue
                return [
                    elt.value
                    for elt in stmt.value.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
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
        """--if and --of are both set to morphologist-auto-nonoverlap-1.0."""
        cmd = _parse_cmd_literal()
        assert "--if" in cmd and "--of" in cmd
        assert cmd[cmd.index("--if") + 1] == FORMAT
        assert cmd[cmd.index("--of") + 1] == FORMAT

    def test_input_format_flag_follows_double_dash_separator(self):
        """--if sits after --, marking it a morphologist param, not a CLI flag."""
        cmd = _parse_cmd_literal()
        assert "--" in cmd and "--if" in cmd
        assert cmd.index("--") < cmd.index("--if")
