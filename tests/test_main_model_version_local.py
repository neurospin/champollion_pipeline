#!/usr/bin/env python3
"""
Tests that main.py's GenerateEmbeddingsStage forwards the model version for a local model set.

- REQ-INPUTKIND (local model version): while hf_enabled is false, GenerateEmbeddingsStage
  shall pass dataset.masks_version to generate_embeddings.py as --model-version, so that
  provenance.json records a model_version for a local model set.

The GenerateEmbeddings class is mocked to capture the argv the stage builds; that argv is then
parsed with the real GenerateEmbeddings parser.
"""

from itertools import pairwise

import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from tests.main_stage_helpers import embeddings_stage_argv, load_main_module, masks_version_config

LOCAL_MASKS_VERSION = "canonical_corrected_26_1"


@pytest.fixture(scope="module")
def main_module():
    return load_main_module("_champollion_main_model_version_local_under_test")


def _local_argv(main_module):
    config = masks_version_config(main_module, LOCAL_MASKS_VERSION, hf_enabled=False)
    argv = embeddings_stage_argv(main_module, config, "GenerateEmbeddingsStage", "test_main_model_version_local")
    return [str(token) for token in argv]


@pytest.mark.unit
class TestLocalModelVersion:
    """While hf_enabled is false, the stage passes masks_version as --model-version."""

    def test_local_argv_passes_model_version(self, main_module):
        """The local-model argv carries --model-version with the configured masks_version."""
        argv = _local_argv(main_module)
        assert f"--model-version={LOCAL_MASKS_VERSION}" in argv or any(
            first == "--model-version" and second == LOCAL_MASKS_VERSION for first, second in pairwise(argv)
        ), f"argv does not pass --model-version {LOCAL_MASKS_VERSION}: {argv!r}"

    def test_local_argv_parses_to_model_version(self, main_module):
        """The real generate_embeddings parser reads the forwarded value into args.model_version."""
        argv = _local_argv(main_module)
        script = GenerateEmbeddings()
        script.parse_args(argv)
        assert script.args.model_version == LOCAL_MASKS_VERSION
