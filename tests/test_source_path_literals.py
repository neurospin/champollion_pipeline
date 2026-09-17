#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guards ``__file__``-relative path literals in the pipeline *source* tree.

Sibling of ``test_collection_integrity.py``, which guards the same class of
defect inside ``tests/``. Here the blast radius is worse: a stale literal in
``src/`` does not surface at import time, it surfaces mid-run as a
``FileNotFoundError`` after an expensive earlier stage has already completed.

Covers REQ-PATH-02 (submodule directory literals) and REQ-PATH-03 (bundled
JSON data-file literals).
"""

import ast
from os.path import abspath, dirname, isdir, isfile, join
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

SOURCE_MODULES = sorted(SRC_DIR.rglob("*.py"))


def _join_calls(module_path):
    """Yield (lineno, expression, constant_args) for every ``join(...)`` call.

    ``constant_args`` holds only the literal string arguments, so a caller can
    filter on them before attempting to evaluate the expression.
    """
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "join":
            continue

        constants = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        segment = ast.get_source_segment(source, node)
        if segment is not None:
            yield node.lineno, segment, constants


def _resolve(expression, module_path):
    """Evaluate a path expression as the source module itself would."""
    namespace = {
        "join": join,
        "dirname": dirname,
        "abspath": abspath,
        "__file__": str(module_path),
        "_SCRIPT_DIR": dirname(str(module_path)),
    }
    try:
        return abspath(eval(expression, namespace))  # noqa: S307 - test-only, inputs are our own sources
    except NameError:
        # Depends on a runtime value (an argparse result, a computed output
        # dir); not a statically-resolvable bundled path, so out of scope.
        return None


@pytest.mark.smoke
@pytest.mark.parametrize("module_path", SOURCE_MODULES, ids=lambda p: p.name)
def test_cortical_tiles_path_literals_resolve_to_existing_directories(module_path):
    """Path literals pointing into external/cortical_tiles must exist on disk.

    The submodule renamed its internal package ``deep_folding/`` ->
    ``cortical_tiles/`` at commit fc7dee1; every copy of the old literal in
    ``src/`` is a runtime ``FileNotFoundError`` waiting for stage 2 to reach it.
    """
    broken = []
    for lineno, expression, constants in _join_calls(module_path):
        if "cortical_tiles" not in constants or "external" not in constants:
            continue
        resolved = _resolve(expression, module_path)
        if resolved is None:
            continue
        # A literal naming a file (…/generate_sulcal_regions.py) is checked as
        # a file; every other one names the directory it is chdir'd/inserted into.
        exists = isfile(resolved) if constants[-1].endswith(".py") else isdir(resolved)
        if not exists:
            broken.append(f"{module_path.name}:{lineno} -> {resolved}")

    assert not broken, "external/cortical_tiles path literals that do not exist:\n" + "\n".join(broken)


@pytest.mark.smoke
@pytest.mark.parametrize("module_path", SOURCE_MODULES, ids=lambda p: p.name)
def test_bundled_json_path_literals_resolve_to_existing_files(module_path):
    """``__file__``-relative literals naming a .json file must exist on disk.

    ``pipeline_loop_2mm.json`` and ``sulci_regions_champollion_V1.json`` ship at
    the repository root; a module under ``src/champollion_pipeline/`` has to
    ascend two levels to reach them, not one.
    """
    broken = []
    for lineno, expression, constants in _join_calls(module_path):
        if not constants or not constants[-1].endswith(".json"):
            continue
        resolved = _resolve(expression, module_path)
        if resolved is None:
            continue
        if not isfile(resolved):
            broken.append(f"{module_path.name}:{lineno} -> {resolved}")

    assert not broken, "bundled .json path literals that do not exist:\n" + "\n".join(broken)
