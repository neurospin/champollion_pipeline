"""Tests for REQ-TRAIN-SUMMARY-01, deliverable (b) — the pin carries the fix.

The behavioural tests live upstream in
``external/champollion_V1/test/test_train_model_summary.py`` and run against
the submodule's *working tree*. This test checks what the pipeline actually
records: the ``external/champollion_V1`` gitlink staged in this repo's index
must point at a champollion_V1 commit whose ``champollion/train.py`` defines
the extracted ``print_model_summary`` helper and no longer carries the bug's
signature: no ``summary(...)`` call passes ``batch_dim=``, and every
``input_data=`` value is built inputs (a call such as
``make_summary_input(config)``, or a list literal), never a bare shape
(a name, tuple or constant such as the old ``input_data=input_size``).

Offline: reads the gitlink via ``git ls-files -s`` and the file via
``git show <sha>:champollion/train.py`` inside the local submodule.
"""

import ast
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMODULE = "external/champollion_V1"


def _git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)


@pytest.fixture(scope="module")
def pinned_train_py() -> ast.Module:
    entry = _git("ls-files", "-s", SUBMODULE, cwd=REPO_ROOT).stdout.split()
    if len(entry) < 2 or entry[0] != "160000":
        pytest.skip(f"{SUBMODULE} is not a gitlink in this checkout")
    sha = entry[1]
    submodule_dir = REPO_ROOT / SUBMODULE
    if not (submodule_dir / ".git").exists():
        pytest.skip(f"{SUBMODULE} is not checked out")
    shown = _git("show", f"{sha}:champollion/train.py", cwd=submodule_dir)
    if shown.returncode != 0:
        pytest.skip(f"pinned commit {sha} not available locally: {shown.stderr.strip()}")
    return ast.parse(shown.stdout)


def test_pinned_train_py_defines_print_model_summary(pinned_train_py):
    names = {node.name for node in pinned_train_py.body if isinstance(node, ast.FunctionDef)}
    assert "print_model_summary" in names, (
        "pinned champollion_V1 train.py lacks print_model_summary — bump the "
        f"{SUBMODULE} gitlink to the fix-torchinfo-summary commit"
    )


def _summary_calls(tree: ast.Module) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "summary"
    ]


def test_pinned_train_py_summary_calls_have_no_shape_input_data_or_batch_dim(pinned_train_py):
    offending = []
    for call in _summary_calls(pinned_train_py):
        for kw in call.keywords:
            if kw.arg == "batch_dim":
                offending.append(f"line {call.lineno}: batch_dim=")
            elif kw.arg == "input_data" and not isinstance(kw.value, (ast.Call, ast.List)):
                offending.append(f"line {call.lineno}: input_data={ast.unparse(kw.value)}")
    assert not offending, (
        "pinned train.py still passes a shape as summary() inputs (torchinfo treats input_data "
        f"as forward() arguments): {offending}"
    )
