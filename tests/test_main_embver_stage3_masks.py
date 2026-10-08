#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests that main.py's GenerateChampollionConfigStage hands dataset.masks_version
to generate_champollion_config as its --masks argument (epic "Versioned
embeddings output paths", task 3/8).

- REQ-EMBVER-BDRABCZUK-D7F47F708D6E: The GenerateChampollionConfigStage in
  main.py shall pass the value of dataset.masks_version as the --masks argument
  of generate_champollion_config.

The GenerateChampollionConfig class referenced by main.py is replaced by a
MagicMock to capture the argv the stage builds without running anything; that
argv is then parsed with the real GenerateChampollionConfig parser, so the
test is indifferent to ``--masks=X`` vs ``--masks X``.

main.py lives at the repo root (not inside the installed package), so it is
loaded by file path under a private module name, as in test_main_stage_argv.py.
"""

import importlib.util
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline.generate_champollion_config import GenerateChampollionConfig

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"

DEFAULT_MASKS_VERSION = "canonical_25"
OTHER_MASKS_VERSION = "canonical_corrected_26_1"


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_embver_stage3_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def main_module():
    return _load_main_module()


def _config(main_module, masks_version=None):
    """PipelineConfig with crops_path and dataset name set; masks_version overridden when given."""
    config = main_module.PipelineConfig(models_path="/models", outputs_path="/outputs")
    config.dataset.crops_path = "/crops"  # noqa: V101
    config.dataset.name = "MYDATASET"
    if masks_version is not None:
        config.dataset.masks_version = masks_version
    return config


def _captured_argv(main_module, config):
    """Run GenerateChampollionConfigStage.execute() with the script class mocked; return the argv it built."""
    script_cls = MagicMock()
    script_cls.return_value.run.return_value = 0
    with patch.object(main_module, "GenerateChampollionConfig", script_cls):
        stage = main_module.GenerateChampollionConfigStage(
            "generate_champollion_config", config, logging.getLogger("test_main_embver_stage3_masks")
        )
        result = stage.execute()
    assert result.success, result.message
    script_cls.return_value.parse_args.assert_called_once()
    return list(script_cls.return_value.parse_args.call_args[0][0])


def _parse_with_real_parser(argv):
    """Parse ``argv`` with GenerateChampollionConfig's real parser; fail (not error) on a rejection."""
    try:
        return GenerateChampollionConfig().parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"generate_champollion_config parser rejected argv {argv!r} (exit {exc.code})")


def _has_masks_option(argv):
    return any(token == "--masks" or token.startswith("--masks=") for token in argv)


@pytest.mark.unit
class TestGenerateChampollionConfigStageMasks:
    """REQ-EMBVER-BDRABCZUK-D7F47F708D6E: stage 3 passes dataset.masks_version as --masks."""

    def test_non_default_masks_version_reaches_generate_champollion_config(self, main_module):
        """A non-default dataset.masks_version is the --masks value generate_champollion_config parses."""
        argv = _captured_argv(main_module, _config(main_module, OTHER_MASKS_VERSION))
        parsed = _parse_with_real_parser(argv)
        assert parsed.masks == OTHER_MASKS_VERSION, (
            f"expected --masks={OTHER_MASKS_VERSION} from dataset.masks_version, stage argv was {argv!r}"
        )

    def test_default_masks_version_is_passed_explicitly(self, main_module):
        """With the default dataset.masks_version, the stage still passes --masks explicitly with that value."""
        config = _config(main_module)
        assert config.dataset.masks_version == DEFAULT_MASKS_VERSION
        argv = _captured_argv(main_module, config)
        assert _has_masks_option(argv), f"stage argv has no --masks option: {argv!r}"
        assert _parse_with_real_parser(argv).masks == DEFAULT_MASKS_VERSION
