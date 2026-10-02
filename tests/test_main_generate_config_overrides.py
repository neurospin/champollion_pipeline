#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for how main() applies CLI overrides to the YAML written by
``--generate-config`` (REQ-GENCFG-01, REQ-GENCFG-02).

The observable is the YAML file main() writes: sys.argv drives
parse_arguments(), check_for_updates() is stubbed (no network), the output path
is under tmp_path, and the file is read back with yaml.safe_load. The
--generate-config path never reaches PipelineOrchestrator, so no stage runs.

main.py lives at the repo root (not inside the installed package), so it is
loaded by file path under a private module name, as in test_main_stage_argv.py.
"""

import importlib.util
from pathlib import Path

import pytest
import yaml

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_generate_config_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def main_module():
    return _load_main_module()


@pytest.fixture
def generate_config(main_module, monkeypatch, tmp_path):
    """Run main() with --generate-config <tmp>/out.yaml plus the given args; return the written YAML as a dict."""
    monkeypatch.setattr(main_module, "check_for_updates", lambda: None)

    def _run(*cli_args):
        output = tmp_path / "out.yaml"
        monkeypatch.setattr("sys.argv", ["main.py", "--generate-config", str(output), *cli_args])
        assert main_module.main() == 0
        with open(output, "r") as f:
            return yaml.safe_load(f)

    return _run


@pytest.mark.unit
class TestGenerateConfigDatasetName:
    """REQ-GENCFG-01: --generate-config with --dataset-name writes the name and its re-derived datasets_root."""

    def test_generate_config_with_dataset_name_writes_name_and_rederived_datasets_root(self, generate_config):
        """REQ-GENCFG-01: dataset.name == foo and dataset.datasets_root == Path(data_path) / foo."""
        written = generate_config("--dataset-name", "foo")
        assert written["dataset"]["name"] == "foo"
        assert Path(written["dataset"]["datasets_root"]) == Path(written["data_path"]) / "foo"


@pytest.mark.unit
class TestGenerateConfigLeafOverrides:
    """REQ-GENCFG-02: --generate-config writes the values the leaf-field CLI overrides assign in a run."""

    def test_generate_config_writes_models_path_override(self, generate_config, tmp_path):
        """REQ-GENCFG-02: --models-path sets models_path in the written YAML."""
        models = str(tmp_path / "my_models")
        written = generate_config("--models-path", models)
        assert written["models_path"] == models

    def test_generate_config_writes_stages_override(self, main_module, generate_config):
        """REQ-GENCFG-02: --stages enables only the named stage(s) and disables every other stage."""
        written = generate_config("--stages", "run_cortical_tiles")
        expected = {stage: stage == "run_cortical_tiles" for stage in main_module.create_default_config().stages}
        assert written["stages"] == expected

    def test_generate_config_writes_enable_all_stages_override(self, main_module, generate_config):
        """REQ-GENCFG-02: --enable-all-stages enables every stage."""
        written = generate_config("--enable-all-stages")
        expected = {stage: True for stage in main_module.create_default_config().stages}
        assert written["stages"] == expected

    def test_generate_config_writes_verbose_override(self, generate_config):
        """REQ-GENCFG-02: --verbose sets verbose True and log_level DEBUG."""
        written = generate_config("--verbose")
        assert written["verbose"] is True
        assert written["log_level"] == "DEBUG"

    def test_generate_config_writes_bids_override(self, generate_config):
        """REQ-GENCFG-02: --bids sets dataset.bids True."""
        written = generate_config("--bids")
        assert written["dataset"]["bids"] is True

    def test_generate_config_writes_mode_override(self, generate_config):
        """REQ-GENCFG-02: --mode streaming sets mode to streaming."""
        written = generate_config("--mode", "streaming")
        assert written["mode"] == "streaming"
