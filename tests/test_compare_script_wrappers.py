#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The stand-alone compare scripts are thin wrappers over src/compare.py.

src/compare_masks.py, src/compare_cortical_tiles.py and src/compare_databases.py
keep their own command line (pinned by tests/test_compare_cli_surface.py) but
hold no second copy of the comparison code: each one runs the matching
subcommand of ``compare.Compare`` with its parsed option values, and none
redefines a module-level function of src/compare.py.

The delegation tests replace ``compare.Compare.run`` with a spy that records the
parsed arguments, so no mask, graph or PyAIMS reader is ever touched. The import
test runs in a fresh Python subprocess where ``soma`` and ``cortical_tiles`` are
blocked through ``sys.modules[...] = None`` (conftest.py stubs ``soma`` in this
process).

Requirements (area COMPARE, author BDRABCZUK): 6F374B546BB4 no redefined
function, 0DA1ADB31D7A 0AA1B1A259B1 E0BC6E28176B delegation, 66C86B6209DD
import outside BrainVISA. Task: TASK-TASK-BDRABCZUK-8D3C019A018B.
"""

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import champollion_utils.script_builder as script_builder
import pytest

import compare
import compare_cortical_tiles
import compare_databases
import compare_masks

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"

_SCRIPTS = ("compare_masks", "compare_cortical_tiles", "compare_databases")

_BLOCKED_MODULES = (
    "soma",
    "soma.aims",
    "cortical_tiles",
    "cortical_tiles.brainvisa",
    "cortical_tiles.brainvisa.utils",
    "cortical_tiles.brainvisa.utils.subjects",
)

_IMPORT_MARKER = "COMPARE_SCRIPT_IMPORTED_OK"


def _module_level_functions(path: Path) -> set:
    """Return the names of the functions defined at the top level of the file at path."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


@pytest.fixture
def compare_run_calls(monkeypatch):
    """Replace Compare.run with a spy; return the list of parsed-argument dicts it saw."""
    calls = []

    def _spy_run(self):
        calls.append(dict(vars(self.args)))
        return 0

    monkeypatch.setattr(compare.Compare, "run", _spy_run)
    monkeypatch.setattr(script_builder, "check_for_updates", lambda *_args, **_kwargs: None)
    return calls


def _run_main(module, monkeypatch, script: str, argv: list) -> None:
    """Call module.main() as if `python src/<script>.py <argv>` had been typed."""
    monkeypatch.setattr(sys, "argv", [str(_SRC_DIR / f"{script}.py")] + argv)
    try:
        module.main()
    except SystemExit:
        pass


def _set_dirs(tmp_path: Path) -> tuple:
    set_a = tmp_path / "set_a"
    set_b = tmp_path / "set_b"
    set_a.mkdir()
    set_b.mkdir()
    return str(set_a), str(set_b)


# --------------------------------------------------------------------------- #
# One implementation: the scripts redefine no function of src/compare.py
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("script", _SCRIPTS)
def test_script_defines_no_compare_function(script):
    """REQ-COMPARE-BDRABCZUK-6F374B546BB4."""
    shared = _module_level_functions(_SRC_DIR / "compare.py") & _module_level_functions(_SRC_DIR / f"{script}.py")
    assert shared - {"main"} == set()


# --------------------------------------------------------------------------- #
# Delegation to Compare
# --------------------------------------------------------------------------- #


def test_compare_masks_runs_compare_masks_mode(compare_run_calls, monkeypatch, tmp_path):
    """REQ-COMPARE-BDRABCZUK-0DA1ADB31D7A."""
    set_a, set_b = _set_dirs(tmp_path)
    output = str(tmp_path / "out" / "masks.json")
    argv = ["--set_a", set_a, "--set_b", set_b, "--output", output, "--metric", "both", "--bucket_step", "2.5"]

    _run_main(compare_masks, monkeypatch, "compare_masks", argv)

    assert len(compare_run_calls) == 1, "src/compare_masks.py did not run compare.Compare"
    args = compare_run_calls[0]
    assert args["mode"] == "masks"
    assert (args["set_a"], args["set_b"], args["output"]) == (set_a, set_b, output)
    assert args["metric"] == "both"
    assert float(args["bucket_step"]) == 2.5


def test_compare_cortical_tiles_runs_compare_cortical_tiles_mode(compare_run_calls, monkeypatch, tmp_path):
    """REQ-COMPARE-BDRABCZUK-0AA1B1A259B1."""
    set_a, set_b = _set_dirs(tmp_path)
    output = str(tmp_path / "out" / "tiles.json")
    argv = [
        "--set_a",
        set_a,
        "--set_b",
        set_b,
        "--output",
        output,
        "--pattern",
        "*mask_cropped.nii.gz",
        "--metric",
        "wasserstein",
        "--bucket_step",
        "0.5",
    ]

    _run_main(compare_cortical_tiles, monkeypatch, "compare_cortical_tiles", argv)

    assert len(compare_run_calls) == 1, "src/compare_cortical_tiles.py did not run compare.Compare"
    args = compare_run_calls[0]
    assert args["mode"] == "cortical_tiles"
    assert (args["set_a"], args["set_b"], args["output"]) == (set_a, set_b, output)
    assert args["pattern"] == "*mask_cropped.nii.gz"
    assert args["metric"] == "wasserstein"
    assert float(args["bucket_step"]) == 0.5


def test_compare_databases_runs_compare_databases_mode(compare_run_calls, monkeypatch, tmp_path):
    """REQ-COMPARE-BDRABCZUK-E0BC6E28176B."""
    subjects_dir = tmp_path / "subjects"
    (subjects_dir / "sub-01").mkdir(parents=True)
    expected = {
        "labeled_subjects_dir": str(subjects_dir),
        "path_to_graph_a": "graph/a",
        "path_to_graph_b": "graph/b",
        "label_a": "campA",
        "label_b": "campB",
        "side": "L",
        "masks_a": str(tmp_path / "masks_a"),
        "masks_b": str(tmp_path / "masks_b"),
        "output": str(tmp_path / "out" / "db.csv"),
    }
    argv = [token for name, value in expected.items() for token in (f"--{name}", value)] + ["--njobs", "3"]

    _run_main(compare_databases, monkeypatch, "compare_databases", argv)

    assert len(compare_run_calls) == 1, "src/compare_databases.py did not run compare.Compare"
    args = compare_run_calls[0]
    assert args["mode"] == "databases"
    assert {name: args[name] for name in expected} == expected
    assert args["njobs"] == 3


# --------------------------------------------------------------------------- #
# Import outside BrainVISA
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("script", _SCRIPTS)
def test_script_imports_without_brainvisa(script, tmp_path):
    """REQ-COMPARE-BDRABCZUK-66C86B6209DD."""
    code = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(_SRC_DIR)!r})
        for _name in {_BLOCKED_MODULES!r}:
            sys.modules[_name] = None
        import {script}
        print({_IMPORT_MARKER!r})
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(tmp_path), capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert _IMPORT_MARKER in result.stdout
