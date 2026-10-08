#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-4 default output is versioned by mask version.

REQ-EMBVER-BDRABCZUK-BFACE95AE3AF: when --output is absent, generate_embeddings.py
writes region embeddings under
<datasets_root>/derivatives/champollion_V1/<masks>/region_embeddings, where
<masks> is its --masks argument.

REQ-EMBVER-BDRABCZUK-26D4C6214910: when dataset.embeddings_path is empty,
GenerateEmbeddingsStage (main.py) passes
--output=<datasets_root>/derivatives/champollion_V1/<masks_version>/region_embeddings,
with <datasets_root> = dataset.datasets_root and <masks_version> = dataset.masks_version.

main.py lives at the repo root (not inside the installed package), so it is
loaded by file path under a private module name, as in test_main_stage_argv.py.
"""

import importlib.util
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"
DERIVATIVES = "derivatives"
CHAMPOLLION_FOLDER = "champollion_V1"
REGION_EMBEDDINGS = "region_embeddings"
DEFAULT_MASKS = "canonical_25"
OTHER_MASKS = "canonical_corrected_26_1"
DATASETS_ROOT = "/data/MYDATASET"


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_embver_stage4_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def main_module():
    return _load_main_module()


def _expected_region_embeddings(datasets_root, masks):
    return Path(datasets_root) / DERIVATIVES / CHAMPOLLION_FOLDER / masks / REGION_EMBEDDINGS


def _standalone_output_base(tmp_path, extra_args):
    """Run GenerateEmbeddings.run() with fetch/per-region stubbed; return (output_base, datasets_root)."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    datasets_root = tmp_path / "MYDATASET"
    datasets_root.mkdir()

    script = GenerateEmbeddings()
    script.parse_args([str(models_dir), str(datasets_root), *extra_args])

    with patch.object(script, "fetch_models", return_value=str(models_dir)):
        with patch.object(script, "_run_per_region", return_value=0) as mock_per:
            script.run()
    mock_per.assert_called_once()
    return mock_per.call_args.args[2], datasets_root


@pytest.mark.unit
class TestStandaloneDefaultOutput:
    """REQ-EMBVER-BDRABCZUK-BFACE95AE3AF: generate_embeddings.py default output_base."""

    def test_default_output_uses_default_masks_canonical_25(self, tmp_path):
        """No --output, no --masks -> <datasets_root>/derivatives/champollion_V1/canonical_25/region_embeddings."""
        output_base, datasets_root = _standalone_output_base(tmp_path, [])
        assert Path(output_base) == _expected_region_embeddings(datasets_root, DEFAULT_MASKS)

    def test_default_output_follows_masks_argument(self, tmp_path):
        """No --output, --masks X -> <datasets_root>/derivatives/champollion_V1/X/region_embeddings."""
        output_base, datasets_root = _standalone_output_base(tmp_path, ["--masks", OTHER_MASKS])
        assert Path(output_base) == _expected_region_embeddings(datasets_root, OTHER_MASKS)


def _orchestrated_output(main_module, masks_version):
    """Run GenerateEmbeddingsStage.execute() with GenerateEmbeddings mocked; return the parsed --output."""
    config = main_module.PipelineConfig(models_path="/models", outputs_path="/outputs")
    config.dataset.datasets_root = DATASETS_ROOT  # noqa: V101
    config.dataset.embeddings_path = ""  # noqa: V101
    config.dataset.hf_enabled = False  # noqa: V101
    if masks_version is not None:
        config.dataset.masks_version = masks_version  # noqa: V101

    script_cls = MagicMock()
    script_cls.return_value.run.return_value = 0
    with patch.object(main_module, "GenerateEmbeddings", script_cls):
        stage = main_module.GenerateEmbeddingsStage(
            "GenerateEmbeddingsStage", config, logging.getLogger("test_embver_stage4_default_output")
        )
        result = stage.execute()
    assert result.success, result.message
    script_cls.return_value.parse_args.assert_called_once()
    argv = list(script_cls.return_value.parse_args.call_args[0][0])
    namespace = GenerateEmbeddings().parse_args(argv)
    assert namespace.output is not None, f"no --output passed to generate_embeddings; argv={argv!r}"
    return namespace.output


@pytest.mark.unit
class TestOrchestratedDefaultOutput:
    """REQ-EMBVER-BDRABCZUK-26D4C6214910: GenerateEmbeddingsStage --output when embeddings_path is empty."""

    def test_empty_embeddings_path_passes_default_masks_region_embeddings(self, main_module):
        """Default masks_version -> --output=<root>/derivatives/champollion_V1/canonical_25/region_embeddings."""
        output = _orchestrated_output(main_module, None)
        assert Path(output) == _expected_region_embeddings(DATASETS_ROOT, DEFAULT_MASKS)

    def test_empty_embeddings_path_follows_masks_version(self, main_module):
        """masks_version X -> --output=<datasets_root>/derivatives/champollion_V1/X/region_embeddings."""
        output = _orchestrated_output(main_module, OTHER_MASKS)
        assert Path(output) == _expected_region_embeddings(DATASETS_ROOT, OTHER_MASKS)
