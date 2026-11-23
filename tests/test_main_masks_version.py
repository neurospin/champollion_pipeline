#!/usr/bin/env python3
"""
Tests for the dataset.masks_version setting of the main.py pipeline config
(epic "Versioned embeddings output paths", task 1/8).

- REQ-EMBVER-BDRABCZUK-91A2A590197E: ConfigLoader.load_from_yaml reads the
  dataset mapping's masks_version key.
- REQ-EMBVER-BDRABCZUK-A3CE0AFED232: ConfigLoader.save_to_yaml writes
  dataset.masks_version as the masks_version key of the dataset mapping.

The default value (DatasetConfig.masks_version, and a YAML without the key) is
tested in tests/test_masks_default_corrected.py::TestSupersededDefaults
(REQ-EMBVER-BDRABCZUK-3219994C034F, REQ-EMBVER-BDRABCZUK-BF32F3E9A1C3, which
supersede the canonical_25 defaults; TASK-MODELVER-BDRABCZUK-39D727185A86).
OTHER_MASKS_VERSION is a version other than the default. YAML files are written
under tmp_path and read with yaml.safe_load / yaml.safe_dump.

main.py lives at the repo root (not inside the installed package), so it is
loaded by file path under a private module name.
"""

import importlib.util
from pathlib import Path

import pytest
import yaml

CLI_PY = Path(__file__).resolve().parent.parent / "src" / "champollion_pipeline" / "cli.py"

OTHER_MASKS_VERSION = "canonical_25"


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_masks_version_under_test", CLI_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def main_module():
    return _load_main_module()


def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


@pytest.mark.unit
class TestLoadFromYamlMasksVersion:
    """REQ-EMBVER-BDRABCZUK-91A2A590197E: loading dataset.masks_version."""

    def test_masks_version_key_is_loaded(self, main_module, tmp_path):
        """REQ-EMBVER-BDRABCZUK-91A2A590197E: dataset.masks_version in the YAML is the loaded value."""
        config_file = tmp_path / "config.yaml"
        _write_yaml(config_file, {"dataset": {"name": "example_dataset", "masks_version": OTHER_MASKS_VERSION}})

        loaded = main_module.ConfigLoader.load_from_yaml(str(config_file))

        assert loaded.dataset.masks_version == OTHER_MASKS_VERSION, (
            "REQ-EMBVER-BDRABCZUK-91A2A590197E: load_from_yaml returned dataset.masks_version="
            f"{loaded.dataset.masks_version!r}, expected {OTHER_MASKS_VERSION!r}"
        )


@pytest.mark.unit
class TestSaveToYamlMasksVersion:
    """REQ-EMBVER-BDRABCZUK-A3CE0AFED232: save_to_yaml writes dataset.masks_version."""

    def test_masks_version_is_written_under_dataset(self, main_module, tmp_path):
        """A non-default dataset.masks_version is written as the dataset mapping's masks_version key."""
        config = main_module.PipelineConfig()
        config.dataset.masks_version = OTHER_MASKS_VERSION
        output = tmp_path / "saved.yaml"

        main_module.ConfigLoader.save_to_yaml(config, str(output))
        with open(output) as f:
            written = yaml.safe_load(f)

        assert written["dataset"].get("masks_version") == OTHER_MASKS_VERSION, (
            "REQ-EMBVER-BDRABCZUK-A3CE0AFED232: saved dataset mapping has masks_version="
            f"{written['dataset'].get('masks_version')!r}, expected {OTHER_MASKS_VERSION!r}"
        )
