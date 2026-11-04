#!/usr/bin/env python3
"""
Tests for main.py's GenerateMorphologistGraphsStage argument wiring (REQ-BIDS-04).

main.py lives at the repo root (not inside the installed package), so it is
loaded here by file path under a private module name. The stage imports
GenerateMorphologistGraphs when it runs, so each test patches the class in its
own module and inspects the argument list handed to parse_args().
"""

import pytest

from tests.main_stage_helpers import capture_stage_argv, load_main_module


@pytest.fixture(scope="module")
def main_module():
    return load_main_module("_champollion_main_under_test")


def _stage_args(main_module, bids, parallel=None):
    """Run GenerateMorphologistGraphsStage.execute() and return the args passed to parse_args()."""
    config = main_module.PipelineConfig()
    config.dataset.input_path = "/input"  # noqa: V101
    config.dataset.morphologist_graphs = "/output"  # noqa: V101
    config.dataset.bids = bids
    if parallel is not None:
        config.parallel = parallel

    return capture_stage_argv(
        main_module.GenerateMorphologistGraphsStage,
        "champollion_pipeline.generate_morphologist_graphs.GenerateMorphologistGraphs",
        "generate_morphologist_graphs",
        config,
        "test_main_morphologist_stage",
    )


@pytest.mark.unit
class TestGenerateMorphologistGraphsStageBids:
    """REQ-BIDS-04: config.dataset.bids maps to --bids, never to --parallel."""

    def test_bids_config_passes_bids_flag(self, main_module):
        """With config.dataset.bids True, the stage passes --bids to GenerateMorphologistGraphs."""
        args = _stage_args(main_module, bids=True)
        assert "--bids" in args

    def test_bids_config_does_not_pass_parallel_flag(self, main_module):
        """With config.dataset.bids True and no truthy config.parallel, --parallel is not passed."""
        args = _stage_args(main_module, bids=True)
        assert "--parallel" not in args

    def test_no_bids_config_omits_bids_flag(self, main_module):
        """Regression guard: with config.dataset.bids False, --bids is not passed."""
        args = _stage_args(main_module, bids=False)
        assert "--bids" not in args


@pytest.mark.unit
class TestGenerateMorphologistGraphsStageParallel:
    """REQ-BIDS-04 context: --parallel stays gated on config.parallel, independent of bids."""

    def test_parallel_config_passes_parallel_flag_without_bids(self, main_module):
        """Regression guard: truthy config.parallel alone still passes --parallel (and not --bids)."""
        args = _stage_args(main_module, bids=False, parallel=True)
        assert "--parallel" in args
        assert "--bids" not in args

    def test_bids_and_parallel_config_pass_both_flags(self, main_module):
        """With both config.dataset.bids and config.parallel truthy, both --bids and --parallel are passed."""
        args = _stage_args(main_module, bids=True, parallel=True)
        assert "--bids" in args
        assert "--parallel" in args
