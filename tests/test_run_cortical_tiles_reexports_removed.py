"""
Guard for the removal of the run_cortical_tiles.py re-exports (TASK-CODE-BDRABCZUK-8636CD45A953, refactor,
behaviour-preserving).

Requirements:
REQ-CODE-BDRABCZUK-2387D87CFE78 (run_cortical_tiles declares no __all__ list),
REQ-CODE-BDRABCZUK-9E2FA83A25E4 (each file imports the moved names from their new modules).
Three ways to reach a removed name: a from-import, an attribute of the module (including an alias bound by
import or by sys.modules), and a patch target string.
"""

import ast
from pathlib import Path

import pytest

import champollion_pipeline.run_cortical_tiles as run_cortical_tiles

pytestmark = pytest.mark.unit  # noqa: V107

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "champollion_pipeline"
OLD_MODULE = "champollion_pipeline.run_cortical_tiles"
NEW_MODULES = ("run_cortical_tiles_inputs.py", "run_cortical_tiles_whole_brain.py")


def _defined_names(path: Path) -> set[str]:
    """Names a module defines at its top level (constants, functions, classes), not those it imports."""
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _moved_names() -> set[str]:
    return set().union(*(_defined_names(PKG / name) for name in NEW_MODULES))


def _python_files() -> list[Path]:
    files = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py"))
    return [f for f in files if f != Path(__file__).resolve()]


def test_run_cortical_tiles_declares_no_all():
    assert not hasattr(run_cortical_tiles, "__all__")


def test_no_file_imports_a_moved_name_from_run_cortical_tiles():
    moved = _moved_names()
    assert moved, "no moved names found: the guard would pass for nothing"
    offenders = []
    for path in _python_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module == OLD_MODULE:
                offenders.extend(f"{path.relative_to(ROOT)}: {a.name}" for a in node.names if a.name in moved)
    assert offenders == []


def _assigns_module(node: ast.Assign) -> bool:
    """True when the assigned value mentions the run_cortical_tiles module name, as in a sys.modules lookup."""
    return OLD_MODULE in {c.value for c in ast.walk(node.value) if isinstance(c, ast.Constant)}


def _node_aliases(node: ast.AST, pkg: str, leaf: str) -> set[str]:
    """Local names one node binds to the run_cortical_tiles module."""
    if isinstance(node, ast.Import):
        return {a.asname or a.name for a in node.names if a.name == OLD_MODULE}
    if isinstance(node, ast.ImportFrom) and node.module == pkg:
        return {a.asname or a.name for a in node.names if a.name == leaf}
    if isinstance(node, ast.Assign) and _assigns_module(node):
        return {t.id for t in node.targets if isinstance(t, ast.Name)}
    return set()


def _module_aliases(tree: ast.AST) -> set[str]:
    """Local names bound to the run_cortical_tiles module: by import, from-import or a sys.modules lookup."""
    pkg, _, leaf = OLD_MODULE.rpartition(".")
    return set().union(*(_node_aliases(node, pkg, leaf) for node in ast.walk(tree)))


def test_no_file_reads_a_moved_name_as_an_attribute_of_run_cortical_tiles():
    moved = _moved_names()
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases = _module_aliases(tree)
        offenders.extend(
            f"{path.relative_to(ROOT)}:{node.lineno}: {node.value.id}.{node.attr}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
            and node.attr in moved
        )
    assert offenders == []


def test_no_patch_target_names_a_moved_name_of_run_cortical_tiles():
    moved = _moved_names()
    prefix = OLD_MODULE + "."
    offenders = []
    for path in _python_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith(prefix)
                and node.value[len(prefix) :].split(".")[0] in moved
            ):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.value}")
    assert offenders == []
