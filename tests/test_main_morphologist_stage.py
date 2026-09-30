#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for main.py's GenerateMorphologistGraphsStage argument wiring (REQ-BIDS-04).

main.py lives at the repo root (not inside the installed package), so it is
loaded here by file path under a private module name. Its top-level imports
of the pipeline scripts are wrapped in try/except, so GenerateMorphologistGraphs
may be absent from the loaded module; each test patches it in (create=True)
and inspects the argument list handed to parse_args().
"""

import importlib.util
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def main_module():
    return _load_main_module()


def _stage_args(main_module, bids, parallel=None):
    """Run GenerateMorphologistGraphsStage.execute() and return the args passed to parse_args()."""
    config = main_module.PipelineConfig()
    config.dataset.input_path = "/input"
    config.dataset.morphologist_graphs = "/output"
    config.dataset.bids = bids
    if parallel is not None:
        config.parallel = parallel

    script_cls = MagicMock()
    script_cls.return_value.run.return_value = 0
    with patch.object(main_module, "GenerateMorphologistGraphs", script_cls, create=True):
        stage = main_module.GenerateMorphologistGraphsStage(
            "generate_morphologist_graphs", config, logging.getLogger("test_main_morphologist_stage")
        )
        result = stage.execute()

    assert result.success, result.message
    script_cls.return_value.parse_args.assert_called_once()
    return list(script_cls.return_value.parse_args.call_args[0][0])


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
