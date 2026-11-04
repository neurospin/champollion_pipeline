#!/usr/bin/env python3
"""
Tests that PutTogetherEmbeddingsStage writes the combined embeddings under a
mask-version folder: --output_path is
<datasets_root>/derivatives/champollion_V1/<masks_version>/embeddings
(REQ-EMBVER-BDRABCZUK-EB2EF83C785C, superseding REQ-COMBOUT-01).

The PutTogetherEmbeddings class referenced by main.py is replaced by a
MagicMock to capture the argv the stage builds, and that argv is parsed with
the real PutTogetherEmbeddings parser. main.py lives at the repo root, so it is
loaded by file path under a private module name.
"""

from pathlib import Path

import pytest

from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings
from tests.main_stage_helpers import capture_stage_argv, load_main_module

DATASETS_ROOT = "/data/MYDATASET"


@pytest.fixture(scope="module")
def main_module():
    return load_main_module("_champollion_main_embver_combined_dir_under_test")


def _config(main_module):
    """PipelineConfig with a non-empty datasets_root and distinct fallback locations."""
    config = main_module.PipelineConfig(models_path="/models", outputs_path="/outputs")
    dataset = config.dataset
    dataset.cortical_tiles_output = "/tiles"  # noqa: V101
    dataset.datasets_root = DATASETS_ROOT  # noqa: V101
    dataset.embeddings_path = "/data/MYDATASET/embeddings_raw"  # noqa: V101
    return config


def _combine_output_path(main_module, config):
    """Run PutTogetherEmbeddingsStage with the script mocked; return the parsed --output_path."""
    argv = capture_stage_argv(
        main_module.PutTogetherEmbeddingsStage,
        "champollion_pipeline.put_together_embeddings.PutTogetherEmbeddings",
        "PutTogetherEmbeddingsStage",
        config,
        "test_main_embver_combined_dir",
    )
    try:
        namespace = PutTogetherEmbeddings().parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"PutTogetherEmbeddings parser rejected the stage argv {argv!r} (SystemExit {exc.code})")
    return Path(namespace.output_path)


@pytest.mark.unit
class TestCombineOutputPathVersioned:
    """REQ-EMBVER-BDRABCZUK-EB2EF83C785C: combine --output_path carries the masks_version folder."""

    def test_combine_output_path_uses_default_masks_version(self, main_module):
        """Unset masks_version -> datasets_root/derivatives/champollion_V1/canonical_25/embeddings."""
        config = _config(main_module)
        expected = Path(DATASETS_ROOT) / "derivatives" / "champollion_V1" / "canonical_25" / "embeddings"
        assert _combine_output_path(main_module, config) == expected

    def test_combine_output_path_uses_configured_masks_version(self, main_module):
        """masks_version=canonical_corrected_26_1 -> that folder sits between champollion_V1 and embeddings."""
        config = _config(main_module)
        config.dataset.masks_version = "canonical_corrected_26_1"  # noqa: V101
        expected = Path(DATASETS_ROOT) / "derivatives" / "champollion_V1" / "canonical_corrected_26_1" / "embeddings"
        assert _combine_output_path(main_module, config) == expected
