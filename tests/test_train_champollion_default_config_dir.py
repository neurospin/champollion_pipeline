#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Default --config-dir of train_champollion.py (REQ-CFGLOC-04).

Without --config-dir, training must resolve dataset configs exactly as if
``--config-dir <P>/data/<dataset>/derivatives/champollion_V1/configs`` were
given (``<P>`` = champollion_pipeline repo root), i.e. where stage 3 now writes
them by default (REQ-CFGLOC-01/02).

Nothing is written under ``<P>/data``: the dataset name is unique to this
module, ``exists`` / ``chdir`` / ``execute_command`` are patched, and the
built-in champollion_V1 config dir is redirected to an empty tmp_path.
"""

import os
from unittest.mock import patch

import pytest

from champollion_pipeline import train_champollion
from champollion_pipeline.train_champollion import TrainChampollion

DATASET = "CFGLOC_DEFAULT_TEST"
REGION = "cingulate_left"
PIPELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(train_champollion.__file__), "..", ".."))
DEFAULT_CONFIG_ROOT = os.path.join(PIPELINE_ROOT, "data", DATASET, "derivatives", "champollion_V1", "configs")


def _make_script(*extra):
    script = TrainChampollion()
    script.parse_args(["--dataset", DATASET, "--region", REGION, *extra])
    return script


@pytest.mark.unit
class TestDefaultConfigDir:
    """REQ-CFGLOC-04: omitted --config-dir acts as <P>/data/<dataset>/derivatives/champollion_V1/configs."""

    def test_validation_accepts_config_present_only_at_derivatives_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr(train_champollion, "_CONTRASTIVE_DIR", str(tmp_path / "builtin_missing"))
        default_yaml = os.path.join(DEFAULT_CONFIG_ROOT, "dataset", DATASET, f"{REGION}.yaml")
        monkeypatch.setattr(train_champollion, "exists", lambda p: os.path.abspath(p) == default_yaml)
        _make_script()._validate_inputs()  # must not raise

    def test_hydra_searchpath_points_at_derivatives_default(self, tmp_path):
        captured = {}

        def fake_execute(cmd, shell=False):
            captured["cmd"] = cmd
            return 0

        script = _make_script("--output_dir", str(tmp_path / "models"))
        script.execute_command = fake_execute
        with patch.object(train_champollion.os, "chdir"):
            script._run_training(str(tmp_path))
        expected = f"hydra.searchpath=[file://{DEFAULT_CONFIG_ROOT}]"
        assert expected in captured["cmd"], f"{expected!r} missing from train.py overrides {captured['cmd']}"
