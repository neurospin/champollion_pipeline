#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guards pytest collection integrity for the test suite itself.

A test module that injects a stale directory into ``sys.path`` does not fail
in isolation: its import raises ``ModuleNotFoundError`` at collection time and
pytest aborts the *entire* run, so no test in any file executes. This module
turns that class of breakage into an ordinary assertion failure naming the
offending file and path.

Covers REQ-COLLECT-01.
"""

import ast
import os
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

TEST_MODULES = sorted(p for p in TESTS_DIR.glob("*.py") if p.name != "__init__.py")


def _syspath_insert_expressions(module_path):
    """Yield (lineno, source_segment) for each sys.path.insert/append argument.

    Only the path argument is returned: ``sys.path.insert(0, <expr>)`` yields
    ``<expr>``, ``sys.path.append(<expr>)`` likewise.
    """
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in ("insert", "append"):
            continue
        # Match the attribute chain sys.path.<insert|append>
        owner = func.value
        if not (
            isinstance(owner, ast.Attribute)
            and owner.attr == "path"
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "sys"
        ):
            continue

        path_arg = node.args[1] if func.attr == "insert" and len(node.args) > 1 else node.args[0]
        segment = ast.get_source_segment(source, path_arg)
        if segment is not None:
            yield node.lineno, segment


def _resolve(expression, module_path):
    """Evaluate a sys.path expression as the test module itself would."""
    namespace = {"os": os, "sys": sys, "Path": Path, "__file__": str(module_path)}
    return eval(expression, namespace)  # noqa: S307 - test-only, inputs are our own sources


@pytest.mark.smoke
@pytest.mark.parametrize("module_path", TEST_MODULES, ids=lambda p: p.name)
def test_syspath_insertions_point_at_existing_directories(module_path):
    """Every path a test module puts on sys.path must exist on disk.

    A stale entry (e.g. a submodule directory renamed upstream) makes the
    subsequent top-level import fail and aborts collection for the whole suite.
    """
    broken = []
    for lineno, expression in _syspath_insert_expressions(module_path):
        resolved = _resolve(expression, module_path)
        if not os.path.isdir(resolved):
            broken.append(f"{module_path.name}:{lineno} -> {resolved}")

    assert not broken, "sys.path entries that do not resolve to an existing directory:\n" + "\n".join(broken)
