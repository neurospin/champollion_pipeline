#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests that main.py's GenerateEmbeddingsStage forwards DatasetConfig.masks_version
to generate_embeddings.py (epic "Versioned embeddings output paths", task 4/8).

- REQ-EMBVER-BDRABCZUK-2086C0283B34: masks_version is passed as ``--masks``
  (crops subdirectory, generate_embeddings.py ``--masks``).
- REQ-EMBVER-BDRABCZUK-0839AF367ACD: when hf_enabled is true, masks_version is
  passed as ``--masks-version`` (Hugging Face model subfolder).

The GenerateEmbeddings class referenced by main.py is replaced by a MagicMock
to capture the argv the stage builds, and that argv is then parsed with the
*real* GenerateEmbeddings parser. A non-default mask version is used so the
script's own ``--masks`` default (canonical_25) cannot satisfy the assertion.

main.py lives at the repo root (not inside the installed package), so it is
loaded by file path under a private module name.
"""

import importlib.util
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"

NON_DEFAULT_MASKS_VERSION = "canonical_corrected_26_1"


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_embver_stage4_masks_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def main_module():
    return _load_main_module()


def _config(main_module, hf_enabled):
    """PipelineConfig with a non-default masks_version and the requested model source."""
    config = main_module.PipelineConfig(models_path="/models", outputs_path="/outputs")
    dataset = config.dataset
    dataset.datasets_root = "/data/MYDATASET"  # noqa: V101
    dataset.masks_version = NON_DEFAULT_MASKS_VERSION  # noqa: V101
    dataset.hf_enabled = hf_enabled  # noqa: V101
    if hf_enabled:
        dataset.hf_repo_id = "neurospin/Champollion_V1"  # noqa: V101
    return config


def _embeddings_namespace(main_module, config):
    """Run GenerateEmbeddingsStage.execute() with GenerateEmbeddings mocked; parse its argv for real."""
    script_cls = MagicMock()
    script_cls.return_value.run.return_value = 0
    with patch.object(main_module, "GenerateEmbeddings", script_cls, create=True):
        stage = main_module.GenerateEmbeddingsStage(
            "GenerateEmbeddingsStage", config, logging.getLogger("test_main_embver_stage4_masks")
        )
        result = stage.execute()
    assert result.success, result.message
    script_cls.return_value.parse_args.assert_called_once()
    argv = list(script_cls.return_value.parse_args.call_args[0][0])
    try:
        return GenerateEmbeddings().parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"GenerateEmbeddings parser rejected argv {argv!r} (exit {exc.code})")


@pytest.mark.unit
class TestGenerateEmbeddingsStageMasks:
    """REQ-EMBVER-BDRABCZUK-2086C0283B34: masks_version -> --masks."""

    def test_masks_argument_carries_dataset_masks_version(self, main_module):
        """GenerateEmbeddingsStage passes DatasetConfig.masks_version as --masks."""
        config = _config(main_module, hf_enabled=False)
        namespace = _embeddings_namespace(main_module, config)
        assert namespace.masks == NON_DEFAULT_MASKS_VERSION


@pytest.mark.unit
class TestGenerateEmbeddingsStageMasksVersion:
    """REQ-EMBVER-BDRABCZUK-0839AF367ACD: hf_enabled -> masks_version as --masks-version."""

    def test_hf_enabled_masks_version_argument_carries_dataset_masks_version(self, main_module):
        """With hf_enabled true, GenerateEmbeddingsStage passes DatasetConfig.masks_version as --masks-version."""
        config = _config(main_module, hf_enabled=True)
        namespace = _embeddings_namespace(main_module, config)
        assert namespace.masks_version == NON_DEFAULT_MASKS_VERSION
