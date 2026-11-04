#!/usr/bin/env python3
"""
Tests that main.py's RunCorticalTilesStage forwards dataset.masks_version to
run_cortical_tiles as its --masks option (epic "Versioned embeddings output
paths", task 2/8).

- REQ-EMBVER-BDRABCZUK-C0F4A13F662F: RunCorticalTilesStage passes the dataset
  masks_version setting as the value of the --masks option of
  run_cortical_tiles (depends on REQ-EMBVER-BDRABCZUK-C56942772AB4,
  DatasetConfig.masks_version).

The RunCorticalTiles class referenced by main.py is replaced by a MagicMock to
capture the argv the stage builds without running anything; that argv is then
parsed with the real RunCorticalTiles parser, so the asserted value is the one
run_cortical_tiles would actually receive as ``args.masks``.

main.py lives at the repo root (not inside the installed package), so it is
loaded by file path under a private module name.
"""

import pytest

from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from tests.main_stage_helpers import load_main_module, tiles_config, tiles_stage_argv

DEFAULT_MASKS_VERSION = "canonical_25"
OTHER_MASKS_VERSION = "canonical_corrected_26_1"


@pytest.fixture(scope="module")
def main_module():
    return load_main_module("_champollion_main_embver_stage2_under_test")


def _stage_masks_value(main_module, config):
    """Run RunCorticalTilesStage.execute() with RunCorticalTiles mocked; return the parsed --masks value."""
    argv = tiles_stage_argv(main_module, config, "test_main_embver_stage2_masks")
    try:
        namespace = RunCorticalTiles().parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"RunCorticalTiles parser rejected the stage argv {argv!r} (SystemExit {exc.code})")
    return namespace.masks, argv


@pytest.mark.unit
class TestRunCorticalTilesStageMasksArgv:
    """REQ-EMBVER-BDRABCZUK-C0F4A13F662F: dataset.masks_version reaches run_cortical_tiles as --masks."""

    def test_default_masks_version_passed_as_masks(self, main_module):
        """Unset dataset.masks_version -> run_cortical_tiles receives --masks canonical_25."""
        config = tiles_config(main_module)
        masks, argv = _stage_masks_value(main_module, config)
        assert masks == DEFAULT_MASKS_VERSION, (
            "REQ-EMBVER-BDRABCZUK-C0F4A13F662F: run_cortical_tiles --masks is "
            f"{masks!r}, expected {DEFAULT_MASKS_VERSION!r}; stage argv was {argv!r}"
        )

    def test_configured_masks_version_passed_as_masks(self, main_module):
        """dataset.masks_version set to a non-default tag -> run_cortical_tiles receives that tag as --masks."""
        config = tiles_config(main_module)
        config.dataset.masks_version = OTHER_MASKS_VERSION  # noqa: V101
        masks, argv = _stage_masks_value(main_module, config)
        assert masks == OTHER_MASKS_VERSION, (
            "REQ-EMBVER-BDRABCZUK-C0F4A13F662F: run_cortical_tiles --masks is "
            f"{masks!r}, expected {OTHER_MASKS_VERSION!r}; stage argv was {argv!r}"
        )
