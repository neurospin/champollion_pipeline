#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for ConfigLoader.save_to_yaml / load_from_yaml round-tripping the
pipeline configuration (REQ-CFGSAVE-01, REQ-CFGSAVE-02).

REQ-CFGSAVE-01 is checked directly on ConfigLoader: a PipelineConfig with
non-default values is saved under tmp_path and loaded back, and the written
YAML keys are compared against the PipelineConfig/DatasetConfig dataclass
fields. dataset.hf_token is excluded: it is a secret that save_to_yaml never
writes (REQ-HFTOKEN-01).

REQ-CFGSAVE-02 goes through main() with --generate-config, using the same seam
as test_main_generate_config_overrides.py: sys.argv drives parse_arguments(),
check_for_updates() is stubbed (no network), and the YAML is written under
tmp_path and read back with yaml.safe_load.

main.py lives at the repo root (not inside the installed package), so it is
loaded by file path under a private module name.
"""

import dataclasses
import importlib.util
from pathlib import Path

import pytest
import yaml

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"

# Secret field never written to YAML (REQ-HFTOKEN-01).
_UNSAVED_DATASET_FIELDS = {"hf_token"}


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_config_roundtrip_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def main_module():
    return _load_main_module()


@pytest.mark.unit
class TestConfigSaveLoadRoundTrip:
    """REQ-CFGSAVE-01: save_to_yaml then load_from_yaml returns the saved field values (hf_token excepted)."""

    def test_save_load_round_trip_preserves_n_workers_and_worker_timeout(self, main_module, tmp_path):
        """REQ-CFGSAVE-01: non-default n_workers and worker_timeout survive a save/load round trip."""
        config = main_module.PipelineConfig()
        config.n_workers = 7
        config.worker_timeout = 99
        output = tmp_path / "saved.yaml"

        main_module.ConfigLoader.save_to_yaml(config, str(output))
        loaded = main_module.ConfigLoader.load_from_yaml(str(output))

        assert (loaded.n_workers, loaded.worker_timeout) == (7, 99), (
            "REQ-CFGSAVE-01: n_workers/worker_timeout lost on save/load round trip: "
            f"got ({loaded.n_workers}, {loaded.worker_timeout}), expected (7, 99)"
        )

    def test_save_to_yaml_writes_a_key_for_each_config_field_except_hf_token(self, main_module, tmp_path):
        """REQ-CFGSAVE-01: the written YAML has a key per PipelineConfig and DatasetConfig field, hf_token excepted.

        A field with no key reloads as its dataclass default, so a missing key
        means a non-default value cannot round-trip.
        """
        output = tmp_path / "saved.yaml"
        main_module.ConfigLoader.save_to_yaml(main_module.PipelineConfig(), str(output))
        with open(output, "r") as f:
            written = yaml.safe_load(f)

        pipeline_fields = {f.name for f in dataclasses.fields(main_module.PipelineConfig)}
        dataset_fields = {f.name for f in dataclasses.fields(main_module.DatasetConfig)} - _UNSAVED_DATASET_FIELDS

        missing_top = sorted(pipeline_fields - set(written))
        missing_dataset = sorted(dataset_fields - set(written["dataset"]))
        assert (missing_top, missing_dataset) == ([], []), (
            f"REQ-CFGSAVE-01: save_to_yaml omits config fields: top-level {missing_top}, dataset {missing_dataset}"
        )


@pytest.mark.unit
class TestGenerateConfigWorkerSettings:
    """REQ-CFGSAVE-02: --generate-config writes the --n-workers and --worker-timeout values."""

    def test_generate_config_writes_n_workers_and_worker_timeout_overrides(self, main_module, monkeypatch, tmp_path):
        """REQ-CFGSAVE-02: --n-workers 7 --worker-timeout 99 appear as n_workers 7 and worker_timeout 99."""
        monkeypatch.setattr(main_module, "check_for_updates", lambda: None)
        output = tmp_path / "out.yaml"
        monkeypatch.setattr(
            "sys.argv",
            ["main.py", "--generate-config", str(output), "--n-workers", "7", "--worker-timeout", "99"],
        )

        assert main_module.main() == 0
        with open(output, "r") as f:
            written = yaml.safe_load(f)

        assert (written.get("n_workers"), written.get("worker_timeout")) == (7, 99), (
            "REQ-CFGSAVE-02: --generate-config YAML lacks the worker overrides: "
            f"n_workers={written.get('n_workers')!r}, worker_timeout={written.get('worker_timeout')!r}"
        )
