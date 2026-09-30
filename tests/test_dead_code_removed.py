#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guards the removal of dead code confirmed by the 2026-09-30 vulture scan.

Covers REQ-CLEANUP-DEADCODE-01: the pipeline source shall not define the
unused ``ScanId`` import and ``StageResult.outputs`` field in ``main.py``,
``PruneFailedSubjects._read_non_passing_subjects``, or the ``are_paths_valid``
and ``get_nth_parent_dir`` functions in ``utils/lib.py``.

The checks are deliberately static (``ast``), matching
``test_source_import_names.py``: importing ``main.py`` pulls in optional
BrainVISA-only modules behind ``try``/``except ImportError`` fallbacks, and
running vulture itself would couple the result to its confidence heuristics.
Parsing the source pins down exactly the five definitions named by the
requirement and nothing else.

The ``TestLiveSiblingsKept`` guards pass today and must keep passing: they
stop the deletion from overreaching into the live neighbours of the dead code.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_PY = REPO_ROOT / "main.py"
PKG_DIR = REPO_ROOT / "src" / "champollion_pipeline"
PRUNE_PY = PKG_DIR / "prune_failed_subjects.py"
LIB_PY = PKG_DIR / "utils" / "lib.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class_def(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name!r} not found at module level")


def _function_names(body) -> set:
    return {node.name for node in body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _annotated_field_names(class_node: ast.ClassDef) -> set:
    return {
        node.target.id
        for node in class_node.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }


@pytest.mark.smoke
class TestDeadCodeRemoved:
    """REQ-CLEANUP-DEADCODE-01: each confirmed-dead definition is gone."""

    def test_main_does_not_bind_scan_id(self):
        """``main.py`` neither imports ``ScanId`` nor assigns its ``None`` fallback."""
        tree = _tree(MAIN_PY)
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if (alias.asname or alias.name.split(".")[-1]) == "ScanId":
                        offenders.append(f"main.py:{node.lineno} import binds ScanId")
            elif isinstance(node, ast.Name) and node.id == "ScanId":
                offenders.append(f"main.py:{node.lineno} name ScanId")
        assert not offenders, "REQ-CLEANUP-DEADCODE-01: unused ScanId still bound:\n" + "\n".join(offenders)

    def test_stage_result_has_no_outputs_field(self):
        """``StageResult`` no longer declares the never-set, never-read ``outputs`` field."""
        fields = _annotated_field_names(_class_def(_tree(MAIN_PY), "StageResult"))
        assert "outputs" not in fields, "REQ-CLEANUP-DEADCODE-01: StageResult still declares 'outputs'"

    def test_prune_failed_subjects_has_no_read_non_passing_subjects(self):
        """``PruneFailedSubjects`` no longer defines the uncalled ``_read_non_passing_subjects``."""
        methods = _function_names(_class_def(_tree(PRUNE_PY), "PruneFailedSubjects").body)
        assert "_read_non_passing_subjects" not in methods, (
            "REQ-CLEANUP-DEADCODE-01: PruneFailedSubjects still defines '_read_non_passing_subjects'"
        )

    def test_lib_has_no_are_paths_valid(self):
        """``utils/lib.py`` no longer defines ``are_paths_valid`` (no production caller)."""
        assert "are_paths_valid" not in _function_names(_tree(LIB_PY).body), (
            "REQ-CLEANUP-DEADCODE-01: utils/lib.py still defines 'are_paths_valid'"
        )

    def test_lib_has_no_get_nth_parent_dir(self):
        """``utils/lib.py`` no longer defines ``get_nth_parent_dir`` (no production caller)."""
        assert "get_nth_parent_dir" not in _function_names(_tree(LIB_PY).body), (
            "REQ-CLEANUP-DEADCODE-01: utils/lib.py still defines 'get_nth_parent_dir'"
        )


@pytest.mark.smoke
class TestLiveSiblingsKept:
    """Over-deletion guards: live neighbours of the dead code stay in place."""

    def test_stage_result_keeps_its_live_fields(self):
        fields = _annotated_field_names(_class_def(_tree(MAIN_PY), "StageResult"))
        assert {"stage_name", "success", "message", "return_code"} <= fields

    def test_prune_failed_subjects_keeps_read_passing_subjects(self):
        methods = _function_names(_class_def(_tree(PRUNE_PY), "PruneFailedSubjects").body)
        assert "_read_passing_subjects" in methods

    def test_lib_keeps_find_dataset_folder(self):
        assert "find_dataset_folder" in _function_names(_tree(LIB_PY).body)

    def test_main_keeps_pipeline_checks_fallbacks(self):
        """The sibling ``SubjectEligibilityChecker``/``build_output_report`` fallbacks survive."""
        assigned = {
            target.id
            for node in ast.walk(_tree(MAIN_PY))
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        assert {"SubjectEligibilityChecker", "build_output_report"} <= assigned
