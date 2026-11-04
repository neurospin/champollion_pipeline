#!/usr/bin/env python3
"""
Tests that main.py's GenerateEmbeddingsStage forwards DatasetConfig.masks_version
to generate_embeddings.py (epic "Versioned embeddings output paths", task 4/8).

- REQ-EMBVER-BDRABCZUK-2086C0283B34: masks_version is passed as ``--masks``
  (crops subdirectory, generate_embeddings.py ``--masks``).
- REQ-EMBVER-BDRABCZUK-EA5EB289888B (supersedes -52570A1DFB33 / -0839AF367ACD):
  when hf_enabled is true, masks_version is passed as ``--model-version``
  (Hugging Face model-set subfolder; ``--masks-version`` is a deprecated alias).

The GenerateEmbeddings class referenced by main.py is replaced by a MagicMock
to capture the argv the stage builds, and that argv is then parsed with the
*real* GenerateEmbeddings parser. A non-default mask version is used so the
script's own ``--masks`` default (canonical_25) cannot satisfy the assertion.

main.py lives at the repo root (not inside the installed package), so it is
loaded by file path under a private module name.
"""

from itertools import pairwise

import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from tests.main_stage_helpers import embeddings_stage_argv, load_main_module, masks_version_config

NON_DEFAULT_MASKS_VERSION = "canonical_corrected_26_1"


@pytest.fixture(scope="module")
def main_module():
    return load_main_module("_champollion_main_embver_stage4_masks_under_test")


def _config(main_module, hf_enabled):
    """PipelineConfig with a non-default masks_version and the requested model source."""
    return masks_version_config(main_module, NON_DEFAULT_MASKS_VERSION, hf_enabled)


def _embeddings_argv(main_module, config):
    """Run GenerateEmbeddingsStage.execute() with GenerateEmbeddings mocked; return the argv it built."""
    argv = embeddings_stage_argv(main_module, config, "GenerateEmbeddingsStage", "test_main_embver_stage4_masks")
    return [str(token) for token in argv]


def _parse(argv):
    """Parse argv with the real GenerateEmbeddings parser."""
    try:
        return GenerateEmbeddings().parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"GenerateEmbeddings parser rejected argv {argv!r} (exit {exc.code})")


def _embeddings_namespace(main_module, config):
    """Run GenerateEmbeddingsStage.execute() with GenerateEmbeddings mocked; parse its argv for real."""
    return _parse(_embeddings_argv(main_module, config))


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
    """REQ-EMBVER-BDRABCZUK-EA5EB289888B: hf_enabled -> masks_version as --model-version."""

    def test_hf_enabled_model_version_argument_carries_dataset_masks_version(self, main_module):
        """With hf_enabled true, GenerateEmbeddingsStage passes DatasetConfig.masks_version as --model-version."""
        config = _config(main_module, hf_enabled=True)
        argv = _embeddings_argv(main_module, config)
        assert f"--model-version={NON_DEFAULT_MASKS_VERSION}" in argv or any(
            first == "--model-version" and second == NON_DEFAULT_MASKS_VERSION for first, second in pairwise(argv)
        ), f"argv does not pass --model-version {NON_DEFAULT_MASKS_VERSION}: {argv!r}"
        namespace = _parse(argv)
        assert namespace.model_version == NON_DEFAULT_MASKS_VERSION
