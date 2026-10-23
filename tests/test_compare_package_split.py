"""Split of src/compare.py into files by responsibility (TASK-REFACTOR-BDRABCZUK-EC9C8548AD42).

Requirements (area COMPARE, author BDRABCZUK):
- 7594D2440597  Compare, voxel_diff, _compare_region_side, linear_cka, _mask_stats and
                visualise_mask_diffs each defined once at top level, in six different files
                whose path under src/ begins with compare
- 1593CCA3A176  every Python file whose path under src/ begins with compare has at most 900 lines
- 1CA3406465F8  the src/ files defining the five non-CLI names do not import
                champollion_utils.script_builder

The tests read source files only (ast), so they hold whatever layout design picks:
a package next to a src/compare.py shim, or a package with __main__.
"""

import ast
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

MAX_LINES = 900

# One representative top-level name per responsibility: CLI, mask metrics,
# crops alignment, embeddings, databases, viewer.
RESPONSIBILITY_NAMES = (
    "Compare",
    "voxel_diff",
    "_compare_region_side",
    "linear_cka",
    "_mask_stats",
    "visualise_mask_diffs",
)
DOMAIN_NAMES = tuple(name for name in RESPONSIBILITY_NAMES if name != "Compare")

_DEFINITION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_SCRIPT_BUILDER_MODULE = "champollion_utils.script_builder"


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _compare_files() -> list[Path]:
    """Python files whose path relative to src/ begins with 'compare'."""
    return [path for path in _python_files(SRC_DIR) if path.relative_to(SRC_DIR).as_posix().startswith("compare")]


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_level_names(tree: ast.Module) -> list[str]:
    return [node.name for node in tree.body if isinstance(node, _DEFINITION_NODES)]


def _definitions(files: list[Path], names: tuple[str, ...]) -> dict[str, list[Path]]:
    found = {name: [] for name in names}
    for path in files:
        for name in _top_level_names(_parse(path)):
            if name in found:
                found[name].append(path)
    return found


def _imports_script_builder(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == _SCRIPT_BUILDER_MODULE or alias.name.startswith(_SCRIPT_BUILDER_MODULE + ".")
                for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == _SCRIPT_BUILDER_MODULE or node.module.startswith(_SCRIPT_BUILDER_MODULE + "."):
                return True
            if node.module == "champollion_utils" and any(alias.name == "script_builder" for alias in node.names):
                return True
    return False


def _rel(path: Path) -> str:
    return path.relative_to(SRC_DIR).as_posix()


def test_responsibility_names_defined_once_in_six_files():
    """REQ-COMPARE-BDRABCZUK-7594D2440597."""
    found = _definitions(_compare_files(), RESPONSIBILITY_NAMES)
    counts = {name: [_rel(path) for path in paths] for name, paths in found.items()}
    assert all(len(paths) == 1 for paths in counts.values()), f"expected one definition per name: {counts}"
    files = {paths[0] for paths in counts.values()}
    assert len(files) == len(RESPONSIBILITY_NAMES), f"expected six different files: {counts}"


def test_compare_source_files_at_most_900_lines():
    """REQ-COMPARE-BDRABCZUK-1593CCA3A176."""
    files = _compare_files()
    assert files, "no Python file under src/ begins with compare"
    sizes = {_rel(path): len(path.read_text(encoding="utf-8").splitlines()) for path in files}
    too_long = {name: lines for name, lines in sizes.items() if lines > MAX_LINES}
    assert not too_long, f"files over {MAX_LINES} lines: {too_long}"


def test_domain_files_do_not_import_script_builder():
    """REQ-COMPARE-BDRABCZUK-1CA3406465F8."""
    found = _definitions(_python_files(SRC_DIR), DOMAIN_NAMES)
    missing = [name for name, paths in found.items() if not paths]
    assert not missing, f"no top-level definition under src/ for {missing}"
    defining_files = sorted({path for paths in found.values() for path in paths})
    offenders = [_rel(path) for path in defining_files if _imports_script_builder(_parse(path))]
    assert not offenders, f"files defining {DOMAIN_NAMES} import {_SCRIPT_BUILDER_MODULE}: {offenders}"
