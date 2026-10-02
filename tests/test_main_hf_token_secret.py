#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests that main.py never persists the HuggingFace token to disk
(REQ-HFTOKEN-01), still loads legacy config YAMLs that carry one
(REQ-HFTOKEN-02), and leaves the ambient HF_TOKEN environment variable
(the preferred token source) untouched when no token is configured
(REQ-HFTOKEN-03).

main.py lives at the repo root (not inside the installed package), so it is
loaded by file path under a private module name, as in
test_main_stage_argv.py.
"""

import importlib.util
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"
SECRET = "hf_SECRETtokenVALUE1234567890"


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_hf_token_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def main_module():
    return _load_main_module()


def _config_with_token(main_module):
    config = main_module.PipelineConfig()
    config.dataset.hf_enabled = True  # noqa: V101
    config.dataset.hf_repo_id = "neurospin/champollion"  # noqa: V101
    config.dataset.hf_token = SECRET  # noqa: V101
    return config


class TestSaveToYamlOmitsToken:
    """REQ-HFTOKEN-01: save_to_yaml never writes the HF token."""

    def test_saved_file_does_not_contain_token_string(self, main_module, tmp_path):
        """REQ-HFTOKEN-01: the token value appears nowhere in the saved file's text."""
        out = tmp_path / "config.yaml"
        main_module.ConfigLoader.save_to_yaml(_config_with_token(main_module), str(out))
        assert SECRET not in out.read_text(), "REQ-HFTOKEN-01: saved config YAML contains the hf_token value"

    def test_saved_file_has_no_non_empty_hf_token_entry(self, main_module, tmp_path):
        """REQ-HFTOKEN-01: dataset.hf_token in the saved file is absent or empty."""
        out = tmp_path / "config.yaml"
        main_module.ConfigLoader.save_to_yaml(_config_with_token(main_module), str(out))
        dataset = yaml.safe_load(out.read_text()).get("dataset") or {}
        assert not dataset.get("hf_token"), (
            f"REQ-HFTOKEN-01: saved config YAML has non-empty dataset.hf_token: {dataset.get('hf_token')!r}"
        )


class TestLoadFromYamlAcceptsLegacyToken:
    """REQ-HFTOKEN-02: load_from_yaml still reads hf_token from existing configs."""

    def test_legacy_hf_token_entry_is_loaded(self, main_module, tmp_path):
        """REQ-HFTOKEN-02: dataset.hf_token in a YAML file loads into config.dataset.hf_token."""
        legacy = tmp_path / "legacy.yaml"
        legacy.write_text(
            yaml.safe_dump({"dataset": {"name": "legacy", "hf_enabled": True, "hf_repo_id": "x/y", "hf_token": SECRET}})
        )
        config = main_module.ConfigLoader.load_from_yaml(str(legacy))
        assert config.dataset.hf_token == SECRET


class TestAmbientHfTokenPreserved:
    """REQ-HFTOKEN-03: with no configured token, the ambient HF_TOKEN is left as-is."""

    @pytest.mark.parametrize("empty_token", [None, ""])
    def test_env_hf_token_unchanged_during_run(self, main_module, monkeypatch, empty_token):
        """REQ-HFTOKEN-03: HF_TOKEN seen by GenerateEmbeddings.run equals the pre-existing value."""
        monkeypatch.setenv("HF_TOKEN", "hf_from_environment")
        config = main_module.PipelineConfig(models_path="/models")
        config.dataset.hf_enabled = True  # noqa: V101
        config.dataset.hf_repo_id = "neurospin/champollion"  # noqa: V101
        config.dataset.hf_token = empty_token  # noqa: V101
        config.dataset.datasets_root = "/data/MYDATASET"  # noqa: V101

        seen = {}

        def _run():
            seen["HF_TOKEN"] = os.environ.get("HF_TOKEN")
            return 0

        script_cls = MagicMock()
        script_cls.return_value.run.side_effect = _run  # noqa: V101
        with patch.object(main_module, "GenerateEmbeddings", script_cls):
            stage = main_module.GenerateEmbeddingsStage(
                "generate_embeddings", config, logging.getLogger("test_main_hf_token_secret")
            )
            result = stage.execute()

        assert result.success, result.message
        assert seen["HF_TOKEN"] == "hf_from_environment"
        assert os.environ.get("HF_TOKEN") == "hf_from_environment"
