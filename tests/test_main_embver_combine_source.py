#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-5 embeddings source follows stage 4's versioned default output.

REQ-EMBVER-BDRABCZUK-EF890636D83B: when dataset.embeddings_path is empty and
dataset.datasets_root is non-empty, PutTogetherEmbeddingsStage (main.py) uses
<datasets_root>/derivatives/champollion_V1/<masks_version>/region_embeddings as
the embeddings source checked by validate() and passed to
put_together_embeddings, with <datasets_root> = dataset.datasets_root and
<masks_version> = dataset.masks_version.

An explicit embeddings_path still wins; that case is pinned by
test_main_stage_argv.py::TestPutTogetherEmbeddingsStageArgv (REQ-STAGEARGV-06).

main.py lives at the repo root (not inside the installed package), so it is
loaded by file path under a private module name, as in test_main_stage_argv.py.
"""

import importlib.util
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"
DERIVATIVES = "derivatives"
CHAMPOLLION_FOLDER = "champollion_V1"
REGION_EMBEDDINGS = "region_embeddings"
DEFAULT_MASKS = "canonical_25"
OTHER_MASKS = "canonical_corrected_26_1"
DATASETS_ROOT = "/data/MYDATASET"


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_embver_combine_source_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def main_module():
    return _load_main_module()


def _expected_region_embeddings(datasets_root, masks):
    return Path(datasets_root) / DERIVATIVES / CHAMPOLLION_FOLDER / masks / REGION_EMBEDDINGS


def _config(main_module, datasets_root, masks_version):
    """PipelineConfig with an empty embeddings_path, a set datasets_root and an optional masks_version."""
    config = main_module.PipelineConfig(models_path="/models", outputs_path="/outputs")
    dataset = config.dataset
    dataset.cortical_tiles_output = "/tiles"  # noqa: V101
    dataset.datasets_root = str(datasets_root)  # noqa: V101
    dataset.embeddings_path = ""  # noqa: V101
    if masks_version is not None:
        dataset.masks_version = masks_version  # noqa: V101
    return config


def _stage(main_module, config):
    return main_module.PutTogetherEmbeddingsStage(
        "PutTogetherEmbeddingsStage", config, logging.getLogger("test_main_embver_combine_source")
    )


def _execute_source(main_module, masks_version):
    """Run PutTogetherEmbeddingsStage.execute() with the script mocked; return the parsed embeddings_source."""
    config = _config(main_module, DATASETS_ROOT, masks_version)
    script_cls = MagicMock()
    script_cls.return_value.run.return_value = 0
    with patch.object(main_module, "PutTogetherEmbeddings", script_cls):
        result = _stage(main_module, config).execute()
    assert result.success, result.message
    script_cls.return_value.parse_args.assert_called_once()
    argv = list(script_cls.return_value.parse_args.call_args[0][0])
    try:
        namespace = PutTogetherEmbeddings().parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"PutTogetherEmbeddings parser rejected the stage argv {argv!r} (SystemExit {exc.code})")
    return Path(namespace.embeddings_source)


@pytest.mark.unit
class TestCombineSourceDefault:
    """REQ-EMBVER-BDRABCZUK-EF890636D83B: stage-5 source when embeddings_path is empty."""

    def test_execute_empty_embeddings_path_passes_default_masks_region_embeddings(self, main_module):
        """Default masks_version -> source=<root>/derivatives/champollion_V1/canonical_25/region_embeddings."""
        source = _execute_source(main_module, None)
        assert source == _expected_region_embeddings(DATASETS_ROOT, DEFAULT_MASKS)

    def test_execute_empty_embeddings_path_follows_masks_version(self, main_module):
        """masks_version X -> embeddings_source=<root>/derivatives/champollion_V1/X/region_embeddings."""
        source = _execute_source(main_module, OTHER_MASKS)
        assert source == _expected_region_embeddings(DATASETS_ROOT, OTHER_MASKS)

    def test_validate_empty_embeddings_path_checks_default_masks_region_embeddings(self, main_module, tmp_path):
        """validate() is False until <root>/derivatives/champollion_V1/canonical_25/region_embeddings exists."""
        datasets_root = tmp_path / "MYDATASET"
        datasets_root.mkdir()
        stage = _stage(main_module, _config(main_module, datasets_root, None))
        assert stage.validate() is False, "validate() accepted a missing versioned region_embeddings directory"
        _expected_region_embeddings(datasets_root, DEFAULT_MASKS).mkdir(parents=True)
        assert stage.validate() is True

    def test_validate_empty_embeddings_path_follows_masks_version(self, main_module, tmp_path):
        """masks_version X: an existing canonical_25 folder does not satisfy validate(); the X folder does."""
        datasets_root = tmp_path / "MYDATASET"
        _expected_region_embeddings(datasets_root, DEFAULT_MASKS).mkdir(parents=True)
        stage = _stage(main_module, _config(main_module, datasets_root, OTHER_MASKS))
        assert stage.validate() is False, "validate() did not check the masks_version region_embeddings directory"
        _expected_region_embeddings(datasets_root, OTHER_MASKS).mkdir(parents=True)
        assert stage.validate() is True
