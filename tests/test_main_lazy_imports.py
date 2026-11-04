#!/usr/bin/env python3
"""
Lazy imports of main.py: main.py and its config loading need no toolchain package, and import none.

conftest.py stubs ``soma`` in this process, so the toolchain checks run in a fresh Python
subprocess. The "needs none" checks set ``sys.modules[<package>] = None``: any import of soma,
cortical_tiles or champollion then raises ImportError, as in a pixi environment that lacks the
package. The "imports none" checks load main.py with the toolchain available and list what it added.

The stage check patches each script class where it is defined, then runs the stage with
``run_script`` replaced, so no real script runs.

main.py lives at the repo root, not in the installed package, so it is loaded by file path under
a private module name.

Requirements: REQ-LAZYIMPORT-BDRABCZUK-D386EDA5EB60 (main.py loads without the toolchain),
REQ-LAZYIMPORT-BDRABCZUK-6B66567F41E8 (a --config YAML file loads without the toolchain),
REQ-LAZYIMPORT-BDRABCZUK-20C1DF840EFE (stages take the script class from its module during execute),
REQ-LAZYIMPORT-BDRABCZUK-C85B2829CDA2 (loading main.py adds no toolchain module to sys.modules),
REQ-STREAMING-BDRABCZUK-DB02C2CD6A1C (main.py imports the streaming runner only in streaming mode).
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.main_stage_helpers import MAIN_PY, load_main_module, stage_paths_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_MARKER = "CONFIG="
MODULES_MARKER = "MODULES="
TOOLCHAIN_PACKAGES = ("soma", "cortical_tiles", "champollion")
STREAMING_RUNNER = "champollion_pipeline.streaming.runner"
PROBE_MODELS_PATH = "/probe/models"
PROBE_MASKS_VERSION = "probe_masks"
STAGE_SCRIPT_CLASSES = {
    "GenerateMorphologistGraphsStage": "champollion_pipeline.generate_morphologist_graphs.GenerateMorphologistGraphs",
    "RunCorticalTilesStage": "champollion_pipeline.run_cortical_tiles.RunCorticalTiles",
    "GenerateChampollionConfigStage": "champollion_pipeline.generate_champollion_config.GenerateChampollionConfig",
    "GenerateEmbeddingsStage": "champollion_pipeline.generate_embeddings.GenerateEmbeddings",
    "PutTogetherEmbeddingsStage": "champollion_pipeline.put_together_embeddings.PutTogetherEmbeddings",
    "GenerateSnapshotsStage": "champollion_pipeline.generate_snapshots.GenerateSnapshots",
}

# Makes the toolchain unimportable, then loads main.py the way `python main.py` would see it, without running main().
_LOAD_MAIN_WITHOUT_TOOLCHAIN = """
import argparse, importlib.util, json, sys

for package in ("soma", "cortical_tiles", "champollion"):
    sys.modules[package] = None
spec = importlib.util.spec_from_file_location("_champollion_main_lazy_probe", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
"""

_LOAD_CONFIG_WITHOUT_TOOLCHAIN = (
    _LOAD_MAIN_WITHOUT_TOOLCHAIN
    + f"""
config = module._load_config(argparse.Namespace(config=sys.argv[2]))
print({CONFIG_MARKER!r} + json.dumps([config.models_path, config.dataset.masks_version]))
"""
)


# Loads main.py with the toolchain importable and reports the modules that load added.
_LOAD_MAIN_LISTING_MODULES = f"""
import importlib.util, json, sys

before = set(sys.modules)
spec = importlib.util.spec_from_file_location("_champollion_main_lazy_probe", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print({MODULES_MARKER!r} + json.dumps(sorted(set(sys.modules) - before)))
"""


def _run_probe(code: str, *args: str) -> subprocess.CompletedProcess:
    """Run code in a new interpreter from the repo root, with src/ on the path."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT / "src"), env.get("PYTHONPATH")) if p)
    return subprocess.run(
        [sys.executable, "-c", code, str(MAIN_PY), *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _modules_added_by_loading_main() -> list[str]:
    result = _run_probe(_LOAD_MAIN_LISTING_MODULES)
    assert result.returncode == 0, f"loading main.py failed:\n{result.stderr}"
    lines = [line for line in result.stdout.splitlines() if line.startswith(MODULES_MARKER)]
    assert len(lines) == 1, f"probe printed no {MODULES_MARKER} line:\n{result.stdout}"
    return json.loads(lines[0][len(MODULES_MARKER) :])


def test_main_loads_without_toolchain():
    """REQ-LAZYIMPORT-BDRABCZUK-D386EDA5EB60."""
    result = _run_probe(_LOAD_MAIN_WITHOUT_TOOLCHAIN)
    assert result.returncode == 0, f"loading main.py needs the toolchain:\n{result.stderr}"


def test_config_file_loads_without_toolchain(tmp_path):
    """REQ-LAZYIMPORT-BDRABCZUK-6B66567F41E8."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"models_path: {PROBE_MODELS_PATH}\ndataset:\n  masks_version: {PROBE_MASKS_VERSION}\n",
        encoding="utf-8",
    )
    result = _run_probe(_LOAD_CONFIG_WITHOUT_TOOLCHAIN, str(config_file))
    assert result.returncode == 0, f"loading a config through main.py needs the toolchain:\n{result.stderr}"
    lines = [line for line in result.stdout.splitlines() if line.startswith(CONFIG_MARKER)]
    assert len(lines) == 1, f"probe printed no {CONFIG_MARKER} line:\n{result.stdout}"
    assert json.loads(lines[0][len(CONFIG_MARKER) :]) == [PROBE_MODELS_PATH, PROBE_MASKS_VERSION]


def test_main_load_imports_no_toolchain():
    """REQ-LAZYIMPORT-BDRABCZUK-C85B2829CDA2."""
    added = [name for name in _modules_added_by_loading_main() if name.split(".")[0] in TOOLCHAIN_PACKAGES]
    assert added == [], f"loading main.py imported {len(added)} toolchain module(s), first: {added[:5]}"


def test_main_load_leaves_streaming_runner_unimported():
    """REQ-STREAMING-BDRABCZUK-DB02C2CD6A1C."""
    assert STREAMING_RUNNER not in _modules_added_by_loading_main()


@pytest.mark.parametrize(("stage_class_name", "script_class_path"), STAGE_SCRIPT_CLASSES.items())
def test_stage_execute_takes_script_class_from_defining_module(stage_class_name, script_class_path):
    """REQ-LAZYIMPORT-BDRABCZUK-20C1DF840EFE: a class replaced after main.py loaded is the one execute uses."""
    main_module = load_main_module("_champollion_main_lazy_imports_under_test")
    stage_cls = getattr(main_module, stage_class_name)
    stage = stage_cls(stage_class_name, stage_paths_config(main_module), logging.getLogger("test_main_lazy_imports"))
    with patch(script_class_path) as script_cls, patch.object(stage, "run_script", return_value=0):
        result = stage.execute()
    assert result.success, result.message
    script_cls.assert_called_once_with()
