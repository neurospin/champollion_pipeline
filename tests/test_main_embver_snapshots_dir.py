#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests that GenerateSnapshotsStage reads and writes under a mask-version folder:

- --embeddings_dir is <datasets_root>/derivatives/champollion_V1/<masks_version>/embeddings
  (REQ-EMBVER-BDRABCZUK-F0EE5542FAB9, superseding REQ-COMBOUT-02);
- with an empty snapshots_path, --output_dir defaults to
  <datasets_root>/derivatives/champollion_V1/<masks_version>/snapshots
  (REQ-EMBVER-BDRABCZUK-03D4262668CF);
- with a non-empty datasets_root, validate() accepts an empty snapshots_path
  (REQ-EMBVER-BDRABCZUK-1686C2A756C8).

The GenerateSnapshots class referenced by main.py is replaced by a MagicMock to
capture the argv the stage builds, and that argv is parsed with the real
GenerateSnapshots parser. main.py lives at the repo root, so it is loaded by
file path under a private module name.
"""

import importlib.util
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline.generate_snapshots import GenerateSnapshots

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"
DATASETS_ROOT = "/data/MYDATASET"
DEFAULT_MASKS = "canonical_25"
OTHER_MASKS = "canonical_corrected_26_1"
LOGGER = logging.getLogger("test_main_embver_snapshots_dir")


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_embver_snapshots_dir_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def main_module():
    return _load_main_module()


def _config(main_module, snapshots_path="", embeddings_path=""):
    """PipelineConfig with a non-empty datasets_root and the given snapshots/embeddings paths."""
    config = main_module.PipelineConfig(models_path="/models", outputs_path="/outputs")
    dataset = config.dataset
    dataset.datasets_root = DATASETS_ROOT  # noqa: V101
    dataset.crops_path = "/crops"  # noqa: V101
    dataset.snapshots_path = snapshots_path  # noqa: V101
    dataset.embeddings_path = embeddings_path  # noqa: V101
    return config


def _snapshots_namespace(main_module, config):
    """Run GenerateSnapshotsStage.execute() with the script mocked; return the argv parsed by the real parser."""
    script_cls = MagicMock()
    script_cls.return_value.run.return_value = 0
    with patch.object(main_module, "GenerateSnapshots", script_cls):
        stage = main_module.GenerateSnapshotsStage("GenerateSnapshotsStage", config, LOGGER)
        result = stage.execute()
    assert result.success, result.message
    script_cls.return_value.parse_args.assert_called_once()
    argv = list(script_cls.return_value.parse_args.call_args[0][0])
    try:
        return GenerateSnapshots().parse_args(argv), argv
    except SystemExit as exc:
        pytest.fail(f"GenerateSnapshots parser rejected the stage argv {argv!r} (SystemExit {exc.code})")


def _versioned_dir(masks_version, leaf):
    return Path(DATASETS_ROOT) / "derivatives" / "champollion_V1" / masks_version / leaf


@pytest.mark.unit
class TestSnapshotsEmbeddingsDirVersioned:
    """REQ-EMBVER-BDRABCZUK-F0EE5542FAB9: --embeddings_dir is <root>/derivatives/champollion_V1/<masks>/embeddings."""

    def test_embeddings_dir_uses_default_masks_version(self, main_module):
        """Default masks_version (canonical_25) -> --embeddings_dir ends in canonical_25/embeddings."""
        config = _config(main_module, snapshots_path="/snapshots", embeddings_path="/data/MYDATASET/embeddings_raw")
        namespace, argv = _snapshots_namespace(main_module, config)
        assert namespace.embeddings_dir is not None, f"no --embeddings_dir in {argv!r}"
        assert Path(namespace.embeddings_dir) == _versioned_dir(DEFAULT_MASKS, "embeddings")

    def test_embeddings_dir_uses_configured_masks_version(self, main_module):
        """masks_version canonical_corrected_26_1 -> --embeddings_dir ends in canonical_corrected_26_1/embeddings."""
        config = _config(main_module, snapshots_path="/snapshots", embeddings_path="/data/MYDATASET/embeddings_raw")
        config.dataset.masks_version = OTHER_MASKS  # noqa: V101
        namespace, argv = _snapshots_namespace(main_module, config)
        assert namespace.embeddings_dir is not None, f"no --embeddings_dir in {argv!r}"
        assert Path(namespace.embeddings_dir) == _versioned_dir(OTHER_MASKS, "embeddings")


@pytest.mark.unit
class TestSnapshotsOutputDirDefault:
    """REQ-EMBVER-BDRABCZUK-03D4262668CF: empty snapshots_path -> --output_dir is <...>/<masks>/snapshots."""

    def test_output_dir_defaults_under_default_masks_version(self, main_module):
        """Empty snapshots_path, default masks_version -> --output_dir ends in canonical_25/snapshots."""
        config = _config(main_module)
        namespace, argv = _snapshots_namespace(main_module, config)
        assert namespace.output_dir, f"empty --output_dir in {argv!r}"
        assert Path(namespace.output_dir) == _versioned_dir(DEFAULT_MASKS, "snapshots")

    def test_output_dir_defaults_under_configured_masks_version(self, main_module):
        """Empty snapshots_path, masks_version canonical_corrected_26_1 -> --output_dir ends in that folder."""
        config = _config(main_module)
        config.dataset.masks_version = OTHER_MASKS  # noqa: V101
        namespace, argv = _snapshots_namespace(main_module, config)
        assert namespace.output_dir, f"empty --output_dir in {argv!r}"
        assert Path(namespace.output_dir) == _versioned_dir(OTHER_MASKS, "snapshots")


@pytest.mark.unit
class TestSnapshotsValidateEmptySnapshotsPath:
    """REQ-EMBVER-BDRABCZUK-1686C2A756C8: datasets_root set, snapshots_path empty -> validate() returns True."""

    def test_validate_accepts_empty_snapshots_path_with_empty_embeddings_path(self, main_module):
        """Empty snapshots_path and empty embeddings_path -> validate() is True."""
        stage = main_module.GenerateSnapshotsStage("GenerateSnapshotsStage", _config(main_module), LOGGER)
        assert stage.validate() is True

    def test_validate_accepts_empty_snapshots_path_with_existing_embeddings_path(self, main_module, tmp_path):
        """Empty snapshots_path and an existing embeddings_path directory -> validate() is True."""
        config = _config(main_module, embeddings_path=str(tmp_path))
        stage = main_module.GenerateSnapshotsStage("GenerateSnapshotsStage", config, LOGGER)
        assert stage.validate() is True
