#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for how main() applies ``--dataset-name`` to ``dataset.datasets_root``
(REQ-DEFROOT-03) without rewriting an explicit ``--config`` YAML value
(REQ-DEFROOT-04).

The observable is the PipelineConfig main() hands to PipelineOrchestrator:
sys.argv drives parse_arguments(), check_for_updates() is stubbed (no network)
and PipelineOrchestrator is replaced by a recorder whose run() returns 0, so no
stage ever executes.

main.py lives at the repo root (not inside the installed package), so it is
loaded by file path under a private module name, as in test_main_stage_argv.py.
"""

import importlib.util
from pathlib import Path

import pytest

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_dataset_name_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def main_module():
    return _load_main_module()


@pytest.fixture
def run_main(main_module, monkeypatch):
    """Run main() with the given CLI args; return the config handed to PipelineOrchestrator."""
    captured = []

    class _RecordingOrchestrator:
        STAGE_REGISTRY = main_module.PipelineOrchestrator.STAGE_REGISTRY  # parse_arguments() reads it

        def __init__(self, config):
            captured.append(config)

        def run(self):
            return 0

    monkeypatch.setattr(main_module, "check_for_updates", lambda: None)
    monkeypatch.setattr(main_module, "PipelineOrchestrator", _RecordingOrchestrator)

    def _run(*cli_args):
        monkeypatch.setattr("sys.argv", ["main.py", *cli_args])
        assert main_module.main() == 0
        assert len(captured) == 1, "main() did not hand exactly one config to PipelineOrchestrator"
        return captured[0]

    return _run


def _write_yaml_with_datasets_root(main_module, yaml_path: Path, datasets_root: str) -> None:
    config = main_module.create_default_config()
    config.dataset.datasets_root = datasets_root
    main_module.ConfigLoader.save_to_yaml(config, str(yaml_path))


@pytest.mark.unit
class TestDatasetNameRederivesDefaultRoot:
    """REQ-DEFROOT-03: --dataset-name without --config re-derives datasets_root for the new name."""

    def test_dataset_name_without_config_sets_datasets_root_to_data_path_joined_with_name(self, run_main):
        """REQ-DEFROOT-03: datasets_root equals Path(data_path) / the --dataset-name value."""
        config = run_main("--dataset-name", "foo")
        assert Path(config.dataset.datasets_root) == Path(config.data_path) / "foo"


@pytest.mark.unit
class TestExplicitYamlDatasetsRootUntouched:
    """REQ-DEFROOT-04: a datasets_root set in the --config YAML is never rewritten by --dataset-name."""

    def test_explicit_yaml_datasets_root_kept_with_dataset_name(self, main_module, run_main, tmp_path):
        """REQ-DEFROOT-04: a custom YAML datasets_root survives --dataset-name unchanged."""
        explicit_root = str(tmp_path / "my_datasets" / "custom_root")
        yaml_path = tmp_path / "config.yaml"
        _write_yaml_with_datasets_root(main_module, yaml_path, explicit_root)

        config = run_main("--config", str(yaml_path), "--dataset-name", "foo")

        assert config.dataset.datasets_root == explicit_root

    def test_yaml_datasets_root_equal_to_default_kept_with_dataset_name(self, main_module, run_main, tmp_path):
        """REQ-DEFROOT-04: a YAML datasets_root equal to <data_path>/<dataset.name> is still left unchanged."""
        default = main_module.create_default_config()
        default_root = str(Path(default.data_path) / default.dataset.name)
        yaml_path = tmp_path / "config.yaml"
        _write_yaml_with_datasets_root(main_module, yaml_path, default_root)

        config = run_main("--config", str(yaml_path), "--dataset-name", "foo")

        assert config.dataset.datasets_root == default_root
