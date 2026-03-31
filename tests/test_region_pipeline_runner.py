#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for RegionPipelineRunner in generate_sulcal_regions.py.
"""

import pytest
from unittest.mock import patch, call

# RegionPipelineRunner lives in the brainvisa subdirectory; conftest adds src/
# to sys.path but not brainvisa/. Add it here so the import works in tests.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '../external/cortical_tiles/deep_folding/brainvisa'
)))

from generate_sulcal_regions import RegionPipelineRunner


RESOLVED_CONFIG = {
    "save_behavior": "best",
    "out_voxel_size": 2.0,
    "brain_regions_json": "/some/path/sulci_regions.json",
    "parallel": True,
    "nb_subjects": -1,
    "labeled_subjects_dir": "/data/labeled",
    "path_to_graph_supervised": "t1mri/folds/3.3",
    "supervised_output_dir": "/data/supervised",
    "nb_subjects_mask": -1,
    "graphs_dir": "/data/graphs",
    "path_to_graph": "t1mri/folds/3.1",
    "path_to_skeleton_with_hull": "t1mri/segmentation",
    "skel_qc_path": "",
    "output_dir": "/data/output",
    "junction": "thin",
    "bids": False,
    "new_sulcus": None,
    "resampled_skel": False,
    "cropping_type": "mask",
    "combine_type": False,
    "no_mask": False,
    "threshold": 0,
    "dilation": 5,
    "skip_distbottom": False,
}


class TestRegionPipelineRunnerInit:

    def test_region_name_set(self):
        runner = RegionPipelineRunner(RESOLVED_CONFIG, "S.C.-sylv.")
        assert runner.config["region_name"] == "S.C.-sylv."

    def test_config_is_a_copy(self):
        """Mutating the runner's config must not affect the original."""
        runner = RegionPipelineRunner(RESOLVED_CONFIG, "S.C.-sylv.")
        runner.config["region_name"] = "MUTATED"
        assert RESOLVED_CONFIG.get("region_name") != "MUTATED"

    def test_combine_type_false_for_normal_region(self):
        runner = RegionPipelineRunner(RESOLVED_CONFIG, "S.C.-sylv.")
        assert runner.config["combine_type"] is False

    def test_combine_type_true_for_cingulate(self):
        runner = RegionPipelineRunner(RESOLVED_CONFIG, "CINGULATE.")
        assert runner.config["combine_type"] is True

    def test_threshold_zero_for_normal_region(self):
        runner = RegionPipelineRunner(RESOLVED_CONFIG, "S.C.-sylv.")
        assert runner.config["threshold"] == 0

    def test_threshold_one_for_occipital(self):
        runner = RegionPipelineRunner(RESOLVED_CONFIG, "OCCIPITAL")
        assert runner.config["threshold"] == 1

    def test_threshold_one_for_insula(self):
        runner = RegionPipelineRunner(
            RESOLVED_CONFIG, "F.C.L.p.-subsc.-F.C.L.a.-INSULA.")
        assert runner.config["threshold"] == 1


class TestRegionPipelineRunnerRun:

    def test_run_with_params_called_once_per_side_input_type(self):
        """run() must call run_with_params exactly sides × input_types times."""
        runner = RegionPipelineRunner(RESOLVED_CONFIG, "S.C.-sylv.")
        sides = ["L", "R"]
        input_types = ["skeleton", "foldlabel"]

        with patch(
            "generate_sulcal_regions.run_with_params"
        ) as mock_run:
            runner.run(sides, input_types, njobs=4)

        assert mock_run.call_count == len(sides) * len(input_types)

    def test_run_passes_correct_side_and_input_type(self):
        """Each run_with_params call receives the right side and input_type."""
        runner = RegionPipelineRunner(RESOLVED_CONFIG, "S.C.-sylv.")

        with patch(
            "generate_sulcal_regions.run_with_params"
        ) as mock_run:
            runner.run(["L"], ["skeleton"], njobs=2)

        cfg_passed = mock_run.call_args[0][0]
        assert cfg_passed["side"] == "L"
        assert cfg_passed["input_type"] == "skeleton"

    def test_run_passes_njobs(self):
        """njobs is forwarded to run_with_params."""
        runner = RegionPipelineRunner(RESOLVED_CONFIG, "S.C.-sylv.")

        with patch(
            "generate_sulcal_regions.run_with_params"
        ) as mock_run:
            runner.run(["L"], ["skeleton"], njobs=8)

        cfg_passed = mock_run.call_args[0][0]
        assert cfg_passed["njobs"] == 8

    def test_run_does_not_mutate_runner_config(self):
        """Calls to run() must not permanently mutate runner.config."""
        runner = RegionPipelineRunner(RESOLVED_CONFIG, "S.C.-sylv.")
        original_side = runner.config.get("side")

        with patch("generate_sulcal_regions.run_with_params"):
            runner.run(["L", "R"], ["skeleton"], njobs=2)

        assert runner.config.get("side") == original_side

    def test_insula_left_gets_threshold_one(self):
        """F.C.L.p.-subsc.-F.C.L.a.-INSULA. on side L must use threshold=1."""
        runner = RegionPipelineRunner(
            RESOLVED_CONFIG, "F.C.L.p.-subsc.-F.C.L.a.-INSULA.")

        configs_seen = []
        with patch(
            "generate_sulcal_regions.run_with_params",
            side_effect=lambda cfg: configs_seen.append(dict(cfg))
        ):
            runner.run(["L", "R"], ["skeleton"], njobs=2)

        left_cfg = next(c for c in configs_seen if c["side"] == "L")
        right_cfg = next(c for c in configs_seen if c["side"] == "R")
        assert left_cfg["threshold"] == 1
        assert right_cfg["threshold"] == 1  # already 1 from __init__

    def test_configs_are_independent_between_calls(self):
        """Each run_with_params call must receive its own config copy."""
        runner = RegionPipelineRunner(RESOLVED_CONFIG, "S.C.-sylv.")
        configs_seen = []

        with patch(
            "generate_sulcal_regions.run_with_params",
            side_effect=lambda cfg: configs_seen.append(cfg)
        ):
            runner.run(["L", "R"], ["skeleton", "foldlabel"], njobs=2)

        # All 4 configs must be distinct objects
        assert len(configs_seen) == 4
        ids = [id(c) for c in configs_seen]
        assert len(set(ids)) == 4
