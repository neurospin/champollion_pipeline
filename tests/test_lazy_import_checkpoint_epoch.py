#!/usr/bin/env python3
"""
Import-time isolation of DEFAULT_CHECKPOINT_EPOCH and of the pipeline config module.

conftest.py stubs ``soma`` in this process, so the toolchain checks run in a fresh Python
subprocess that imports one module and reports which modules of the toolchain packages
(soma, cortical_tiles, champollion) that import added to ``sys.modules``.

The module that assigns DEFAULT_CHECKPOINT_EPOCH is found by parsing the sources under
src/champollion_pipeline, so the test does not fix where the constant lives.

Requirements: REQ-LAZYIMPORT-BDRABCZUK-82B7DF9D66CC (the constant's module imports no toolchain),
REQ-LAZYIMPORT-BDRABCZUK-BB2CC03C17D6 (champollion_pipeline.config imports no toolchain).
"""

import ast
import importlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
_PACKAGE_DIR = _SRC_DIR / "champollion_pipeline"
_TOOLCHAIN_PACKAGES = ("soma", "cortical_tiles", "champollion")
_CONSTANT = "DEFAULT_CHECKPOINT_EPOCH"


def _assigns_constant(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == _CONSTANT for t in node.targets):
                return True
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == _CONSTANT:
            return True
    return False


def _constant_module_name() -> str:
    """Dotted name of the single champollion_pipeline module assigning DEFAULT_CHECKPOINT_EPOCH."""
    found = []
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        if _assigns_constant(ast.parse(path.read_text(encoding="utf-8"))):
            parts = list(path.relative_to(_SRC_DIR).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            found.append(".".join(parts))
    assert len(found) == 1, f"expected one module assigning {_CONSTANT}, found {found}"
    return found[0]


def _toolchain_modules_added_by_import(module_name: str) -> list[str]:
    """Import module_name in a new interpreter; return the toolchain modules the import added."""
    code = textwrap.dedent(
        f"""
        import importlib, json, sys
        before = set(sys.modules)
        importlib.import_module({module_name!r})
        added = set(sys.modules) - before
        print(json.dumps(sorted(m for m in added if m.split(".")[0] in {_TOOLCHAIN_PACKAGES!r})))
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(_SRC_DIR), str(_REPO_ROOT), env.get("PYTHONPATH")) if p)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"importing {module_name} failed:\n{result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_constant_module_imports_no_toolchain():
    """REQ-LAZYIMPORT-BDRABCZUK-82B7DF9D66CC."""
    module_name = _constant_module_name()
    assert _toolchain_modules_added_by_import(module_name) == [], module_name


def test_config_module_imports_no_toolchain():
    """REQ-LAZYIMPORT-BDRABCZUK-BB2CC03C17D6."""
    assert _toolchain_modules_added_by_import("champollion_pipeline.config") == []


def test_checkpoint_epoch_value_shared():
    """Regression guard: the config default and generate_embeddings use the defining module's value, 80."""
    constant = getattr(importlib.import_module(_constant_module_name()), _CONSTANT)
    config = importlib.import_module("champollion_pipeline.config")
    generate_embeddings = importlib.import_module("champollion_pipeline.generate_embeddings")
    assert constant == 80
    assert constant == generate_embeddings.DEFAULT_CHECKPOINT_EPOCH
    assert config.DatasetConfig().checkpoint_epoch == constant
