#!/usr/bin/env python3
"""
generate_embeddings.py is split: the model fetch strategies, the model set
marker, the models cache directory helper and the checkpoint lookup move to the
champollion_pipeline.models package (TASK-REFACTOR-BDRABCZUK-2A5EA38957D3).
Behaviour is unchanged: champollion_pipeline.generate_embeddings keeps every
public name main.py and the tests import.

* REQ-EMBSPLIT-BDRABCZUK-DB5EF698BE97 - generate_embeddings module public names.
* REQ-EMBSPLIT-BDRABCZUK-2DF0D8A1A225 - fetch strategies defined in models.fetch.
* REQ-EMBSPLIT-BDRABCZUK-D0FE54E262AC - model set and cache names defined in the models package.
* REQ-EMBSPLIT-BDRABCZUK-C2C9B8E2EEB7 - models.checkpoint.find_epoch_checkpoint.
* REQ-EMBSPLIT-BDRABCZUK-78FFACB91000 - embedding source files at most 600 lines.

The 600-line cap is about half of the 1153-line generate_embeddings.py before
the split: it is reached by moving what this Task names (strategies and their
orchestration, model set marker and cache helpers, checkpoint and region weight
lookup) and keeps argument parsing and the per-region run loop in place.
"""

import ast
from pathlib import Path

import pytest

import champollion_pipeline.generate_embeddings as ge
from champollion_pipeline.models.checkpoint import find_epoch_checkpoint

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "champollion_pipeline"
MODELS_DIR = PACKAGE_DIR / "models"
GENERATE_EMBEDDINGS = PACKAGE_DIR / "generate_embeddings.py"
MAX_LINES = 600

PUBLIC_NAMES = (
    "GenerateEmbeddings",
    "main",
    "ModelSet",
    "read_model_set",
    "write_model_set",
    "read_pipeline_commit",
    "ensure_models_cache_dir",
    "MODEL_SET_FILE_NAME",
    "PROVENANCE_FILE_NAME",
    "DEFAULT_CHECKPOINT_EPOCH",
    "ModelFetchStrategy",
    "LocalPathStrategy",
    "HuggingFaceStrategy",
    "RemoteArchiveStrategy",
    "InteractiveFallbackStrategy",
)
STRATEGY_NAMES = (
    "ModelFetchStrategy",
    "LocalPathStrategy",
    "HuggingFaceStrategy",
    "RemoteArchiveStrategy",
    "InteractiveFallbackStrategy",
)
MODEL_SET_AND_CACHE_NAMES = (
    "ModelSet",
    "read_model_set",
    "write_model_set",
    "ensure_models_cache_dir",
    "MODEL_SET_FILE_NAME",
    "DEFAULT_CHECKPOINT_EPOCH",
)


def _top_level_definitions(path: Path) -> set[str]:
    """Names a module defines itself at top level (class, def, assignment), not the ones it imports."""
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _write_ckpt(model_dir: Path, version: int, name: str) -> Path:
    ckpt = model_dir / "logs" / "lightning_logs" / f"version_{version}" / "checkpoints" / name
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"ckpt")
    return ckpt


def test_generate_embeddings_public_names():
    missing = [name for name in PUBLIC_NAMES if not hasattr(ge, name)]
    assert not missing, f"champollion_pipeline.generate_embeddings lost public names: {missing}"


def test_fetch_strategies_defined_in_models_fetch():
    modules = {name: getattr(ge, name).__module__ for name in STRATEGY_NAMES}
    wrong = {name: module for name, module in modules.items() if module != "champollion_pipeline.models.fetch"}
    assert not wrong, f"strategy classes not defined in champollion_pipeline.models.fetch: {wrong}"


def test_model_set_and_cache_names_defined_in_models_package():
    defined_in_models = set().union(*(_top_level_definitions(p) for p in sorted(MODELS_DIR.glob("*.py"))))
    defined_in_ge = _top_level_definitions(GENERATE_EMBEDDINGS)
    not_in_models = [name for name in MODEL_SET_AND_CACHE_NAMES if name not in defined_in_models]
    still_in_ge = [name for name in MODEL_SET_AND_CACHE_NAMES if name in defined_in_ge]
    assert not not_in_models and not still_in_ge, (
        f"not defined in champollion_pipeline.models: {not_in_models}; "
        f"still defined in generate_embeddings.py: {still_in_ge}"
    )


def test_find_epoch_checkpoint_highest_step(tmp_path):
    _write_ckpt(tmp_path, 0, "epoch=80-step=1000.ckpt")
    best = _write_ckpt(tmp_path, 1, "epoch=80-step=2000.ckpt")
    _write_ckpt(tmp_path, 2, "epoch=80-step=1500.ckpt")
    assert find_epoch_checkpoint(str(tmp_path), 80) == best


def test_find_epoch_checkpoint_ignores_other_epochs(tmp_path):
    wanted = _write_ckpt(tmp_path, 0, "epoch=3-step=10.ckpt")
    _write_ckpt(tmp_path, 0, "epoch=80-step=9999.ckpt")
    _write_ckpt(tmp_path, 0, "epoch=30-step=9999.ckpt")
    _write_ckpt(tmp_path, 0, "last.ckpt")
    assert find_epoch_checkpoint(str(tmp_path), 3) == wanted


@pytest.mark.parametrize("layout", ["no_logs", "other_epoch_only"])
def test_find_epoch_checkpoint_none_when_absent(tmp_path, layout):
    if layout == "other_epoch_only":
        _write_ckpt(tmp_path, 0, "epoch=79-step=500.ckpt")
    assert find_epoch_checkpoint(str(tmp_path), 80) is None


def test_embedding_source_files_at_most_600_lines():
    files = [GENERATE_EMBEDDINGS, *sorted(MODELS_DIR.glob("*.py"))]
    too_long = {
        str(p.relative_to(PACKAGE_DIR)): n
        for p in files
        if (n := len(p.read_text(encoding="utf-8").splitlines())) > MAX_LINES
    }
    assert not too_long, f"files over {MAX_LINES} lines: {too_long}"
