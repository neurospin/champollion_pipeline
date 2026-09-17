#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guards the *package name* the pipeline source imports from the
``external/cortical_tiles`` submodule.

Sibling of ``test_source_path_literals.py``, which guards the same upstream
rename as it appears in ``__file__``-relative *path* literals. Here the defect
hides one layer deeper: the import statement itself still names the old
top-level package ``deep_folding``, which the submodule renamed to
``cortical_tiles`` at commit fc7dee1 (``info.py`` now declares
``NAME = 'cortical_tiles'``).

The check is deliberately static. ``cortical_tiles`` is installed only in the
``brainvisa`` pixi environment, so an import-time probe would be skipped in the
``default`` environment where the suite normally runs — and the tests that mock
these modules out via ``sys.modules`` would keep passing against either name.

Covers REQ-IMPORT-01.
"""

import ast
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

SOURCE_MODULES = sorted(SRC_DIR.rglob("*.py"))

RENAMED_PACKAGE = "deep_folding"


def _imported_top_level_packages(module_path):
    """Yield (lineno, statement_source, top_level_package) for every import."""
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))

    for node in ast.walk(tree):
        segment = ast.get_source_segment(source, node)
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, segment, alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            # A relative import (level > 0) names a package inside this
            # project, never the submodule, so it is out of scope.
            if node.level == 0 and node.module:
                yield node.lineno, segment, node.module.split(".")[0]


@pytest.mark.smoke
@pytest.mark.parametrize("module_path", SOURCE_MODULES, ids=lambda p: p.name)
def test_no_source_module_imports_the_renamed_deep_folding_package(module_path):
    """No module under ``src/`` may import the old ``deep_folding`` package.

    The importable name is ``cortical_tiles`` since the submodule's internal
    package rename; every surviving ``deep_folding`` import raises
    ``ModuleNotFoundError`` the moment that code path is reached at runtime.
    """
    offenders = [
        f"{module_path.name}:{lineno} -> {statement}"
        for lineno, statement, package in _imported_top_level_packages(module_path)
        if package == RENAMED_PACKAGE
    ]

    assert not offenders, f"imports of the renamed '{RENAMED_PACKAGE}' package (now 'cortical_tiles'):\n" + "\n".join(
        offenders
    )
