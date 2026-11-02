"""Tests for the e2e run configuration ``scripts/e2e/pipeline.yaml``.

TASK-E2ECI-BDRABCZUK-348A9B6E5C92 (PCFG, epic TASK-E2ECI-BDRABCZUK-F3B0138F58F3).
``scripts/e2e/pipeline.yaml`` is the ``main.py --config`` file of every e2e tier and
layout, read inside the env SIF. It names container mount points only: the run
script binds the host directories there (code at /opt/code, input, Morphologist
graphs and models read from their binds, every output under the writable run bind).
The tier table (stages per tier, layouts) stays in ``scripts/e2e/e2e.env``; the run
script passes ``--stages``, ``--bids`` and ``--profile`` on the command line, so the
YAML holds none of them.

Requirements (one test each):
- REQ-E2ECI-BDRABCZUK-E0893135F050 ConfigLoader loads the file
- REQ-E2ECI-BDRABCZUK-3360682D25EC models_path is the models mount point
- REQ-E2ECI-BDRABCZUK-FFF40B5A4929 Hugging Face disabled
- REQ-E2ECI-BDRABCZUK-834C34BF46FB no dataset.hf_token key
- REQ-E2ECI-BDRABCZUK-9D4B108A137C dataset.input_path is the input mount point
- REQ-E2ECI-BDRABCZUK-2526A9173504 dataset.morphologist_graphs is the Morphologist mount point
- REQ-E2ECI-BDRABCZUK-3C0110C5E187 outputs_path is the run mount point
- REQ-E2ECI-BDRABCZUK-905E49E3E095 dataset.datasets_root is the run mount point
- REQ-E2ECI-BDRABCZUK-E3F9E751598A every other absolute path lies under the run mount point
- REQ-E2ECI-BDRABCZUK-5A5ECEF8E9E6 no profile_dir key
- REQ-E2ECI-BDRABCZUK-008D5F9D0154 no stages key
- REQ-E2ECI-BDRABCZUK-FE798EF34925 no dataset.bids key
- REQ-E2ECI-BDRABCZUK-6CF247336E92 no string value matches a banned dataset pattern
"""

from __future__ import annotations

import importlib.util
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from champollion_pipeline.config import ConfigLoader, PipelineConfig

REPO = Path(__file__).resolve().parents[1]
E2E_DIR = REPO / "scripts" / "e2e"
PIPELINE_YAML = E2E_DIR / "pipeline.yaml"
ENV_FILE = E2E_DIR / "e2e.env"
CHECKER = E2E_DIR / "e2e_env.py"
BASH = shutil.which("bash")

MODELS_MOUNT = "/opt/e2e/models"
INPUT_MOUNT = "/opt/e2e/input"
MORPHOLOGIST_MOUNT = "/opt/e2e/morphologist"
RUN_MOUNT = "/opt/e2e/run"
# Keys whose absolute path may lie outside the run mount point (read-only binds).
OUTSIDE_RUN_KEYS = ("models_path", "dataset.input_path", "dataset.morphologist_graphs")


def _raw() -> dict[str, Any]:
    """The YAML mapping of scripts/e2e/pipeline.yaml, as written (no defaults filled in)."""
    assert PIPELINE_YAML.is_file(), f"{PIPELINE_YAML.relative_to(REPO)} is missing"
    data = yaml.safe_load(PIPELINE_YAML.read_text())
    assert isinstance(data, dict), f"{PIPELINE_YAML.relative_to(REPO)} is not a YAML mapping"
    return data


def _dataset() -> dict[str, Any]:
    """The ``dataset`` mapping of the file, empty when absent."""
    dataset = _raw().get("dataset") or {}
    assert isinstance(dataset, dict), "dataset is not a YAML mapping"
    return dataset


def _string_values(node: Any, prefix: str = "") -> Iterator[tuple[str, str]]:
    """(dotted key, value) for every string value of the mapping, nested lists included."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _string_values(value, f"{prefix}{key}.")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _string_values(value, f"{prefix}{index}.")
    elif isinstance(node, str):
        yield prefix.rstrip("."), node


def _banned_patterns(home: Path) -> list[str]:
    """E2E_BANNED_DATA_PATTERNS of the real scripts/e2e/e2e.env, read with e2e_env.load_settings."""
    spec = importlib.util.spec_from_file_location("_e2e_env_for_pipeline_yaml", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    patterns = module.load_settings(ENV_FILE, home=str(home)).get("E2E_BANNED_DATA_PATTERNS", "").split()
    assert patterns, "E2E_BANNED_DATA_PATTERNS is empty in scripts/e2e/e2e.env"
    return patterns


def test_config_loader_loads_pipeline_yaml() -> None:
    """REQ-E2ECI-BDRABCZUK-E0893135F050: ConfigLoader.load_from_yaml returns a PipelineConfig."""
    assert PIPELINE_YAML.is_file(), f"{PIPELINE_YAML.relative_to(REPO)} is missing"
    config = ConfigLoader.load_from_yaml(str(PIPELINE_YAML))
    assert isinstance(config, PipelineConfig)


def test_models_path_is_models_mount_point() -> None:
    """REQ-E2ECI-BDRABCZUK-3360682D25EC: models_path is /opt/e2e/models."""
    assert _raw().get("models_path") == MODELS_MOUNT


def test_hf_disabled() -> None:
    """REQ-E2ECI-BDRABCZUK-FFF40B5A4929: dataset.hf_enabled is the YAML boolean false."""
    dataset = _dataset()
    assert "hf_enabled" in dataset, "dataset.hf_enabled is not set"
    assert dataset["hf_enabled"] is False


def test_no_hf_token_key() -> None:
    """REQ-E2ECI-BDRABCZUK-834C34BF46FB: the file holds no dataset.hf_token key."""
    assert "hf_token" not in _dataset()


def test_input_path_is_input_mount_point() -> None:
    """REQ-E2ECI-BDRABCZUK-9D4B108A137C: dataset.input_path is /opt/e2e/input."""
    assert _dataset().get("input_path") == INPUT_MOUNT


def test_morphologist_graphs_is_morphologist_mount_point() -> None:
    """REQ-E2ECI-BDRABCZUK-2526A9173504: dataset.morphologist_graphs is /opt/e2e/morphologist."""
    assert _dataset().get("morphologist_graphs") == MORPHOLOGIST_MOUNT


def test_outputs_path_is_run_mount_point() -> None:
    """REQ-E2ECI-BDRABCZUK-3C0110C5E187: outputs_path is /opt/e2e/run."""
    assert _raw().get("outputs_path") == RUN_MOUNT


def test_datasets_root_is_run_mount_point() -> None:
    """REQ-E2ECI-BDRABCZUK-905E49E3E095: dataset.datasets_root is /opt/e2e/run."""
    assert _dataset().get("datasets_root") == RUN_MOUNT


def test_other_paths_under_run_mount_point() -> None:
    """REQ-E2ECI-BDRABCZUK-E3F9E751598A: any other value starting with / is /opt/e2e/run or below it."""
    outside = [
        f"{key}={value}"
        for key, value in _string_values(_raw())
        if value.startswith("/")
        and key not in OUTSIDE_RUN_KEYS
        and not (value == RUN_MOUNT or value.startswith(RUN_MOUNT + "/"))
    ]
    assert not outside, f"absolute paths outside {RUN_MOUNT}: {outside}"


def test_no_profile_dir_key() -> None:
    """REQ-E2ECI-BDRABCZUK-5A5ECEF8E9E6: profiling comes from --profile on the command line, not the file."""
    assert "profile_dir" not in _raw()


def test_no_stages_key() -> None:
    """REQ-E2ECI-BDRABCZUK-008D5F9D0154: the stages of a tier come from e2e.env through --stages."""
    assert "stages" not in _raw()


def test_no_bids_key() -> None:
    """REQ-E2ECI-BDRABCZUK-FE798EF34925: the layout comes from e2e.env through --bids."""
    assert "bids" not in _dataset()


@pytest.mark.skipif(BASH is None, reason="bash executable not available")
def test_no_banned_dataset_value(tmp_path: Path) -> None:
    """REQ-E2ECI-BDRABCZUK-6CF247336E92: no string value matches an E2E_BANNED_DATA_PATTERNS regex."""
    patterns = _banned_patterns(tmp_path)
    hits = [
        f"{key}={value} ({pattern})"
        for key, value in _string_values(_raw())
        for pattern in patterns
        if re.search(pattern, value, re.IGNORECASE)
    ]
    assert not hits, f"banned dataset references: {hits}"
