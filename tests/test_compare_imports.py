"""Import placement in src/compare.py (TASK-185).

- TestTask185CompareImports  REQ-COMPARE-91  no import inside a function or method body
                             REQ-COMPARE-92  _load_database calls compare.get_all_subjects_as_dictionary

Project rule (CLAUDE.md): all imports go at the top of the file.
REQ-COMPARE-53 already covered visualise_mask_diffs; this file covers the
rest of the module (Compare._load_database, Compare._run_databases and the
joblib workers _get_subject_voxel_counts and _mask_stats).
"""

import ast
from pathlib import Path
from unittest.mock import MagicMock

import compare
from compare import Compare

SRC_COMPARE = Path(__file__).resolve().parent.parent / "src" / "compare.py"

_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
_IMPORT_NODES = (ast.Import, ast.ImportFrom)


def _function_level_imports(tree: ast.AST) -> list:
    """Return (lineno, enclosing function, statement) for every nested import."""
    found = []

    def _walk(node, enclosing):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _FUNCTION_NODES):
                _walk(child, enclosing + [child.name])
                continue
            if isinstance(child, _IMPORT_NODES) and enclosing:
                found.append((child.lineno, ".".join(enclosing), ast.unparse(child)))
            _walk(child, enclosing)

    _walk(tree, [])
    return found


def _make_script(argv: list) -> Compare:
    script = Compare()
    script.args = script.parse_args(argv)
    return script


class TestTask185CompareImports:
    def test_compare_module_has_no_function_level_import(self):
        tree = ast.parse(SRC_COMPARE.read_text())
        assert _function_level_imports(tree) == []

    def test_load_database_calls_module_level_get_all_subjects_as_dictionary(self, tmp_path, monkeypatch):
        lister = MagicMock(return_value=[{"subject": "sub-01", "dir": "d", "graph_file": "g"}])
        # raising=True (the default): the attribute must already exist on the module.
        monkeypatch.setattr(compare, "get_all_subjects_as_dictionary", lister)
        monkeypatch.setattr(compare, "_get_subject_voxel_counts", lambda sub, _bv: (sub["subject"], {"S.C.": 3}))
        script = _make_script(
            [
                "databases",
                "--labeled_subjects_dir",
                str(tmp_path),
                "--path_to_graph_a",
                "folds/3.3/a",
                "--path_to_graph_b",
                "folds/3.3/b",
            ]
        )

        data = script._load_database("folds/3.3/a", ["L", "R"], str(tmp_path), 1, str(tmp_path))

        assert data == {"sub-01": {"S.C.": 3}}
        assert lister.call_count == 2
        assert [c.args[2] for c in lister.call_args_list] == ["L", "R"]
