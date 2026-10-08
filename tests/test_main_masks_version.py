#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the dataset.masks_version setting of the main.py pipeline config
(epic "Versioned embeddings output paths", task 1/8).

- REQ-EMBVER-BDRABCZUK-C56942772AB4: DatasetConfig declares a masks_version
  field whose default value is canonical_25.
- REQ-EMBVER-BDRABCZUK-91A2A590197E: ConfigLoader.load_from_yaml reads the
  dataset mapping's masks_version key.
- REQ-EMBVER-BDRABCZUK-CC2C03FB545E: ConfigLoader.load_from_yaml yields
  canonical_25 when the dataset mapping has no masks_version key.
- REQ-EMBVER-BDRABCZUK-A3CE0AFED232: ConfigLoader.save_to_yaml writes
  dataset.masks_version as the masks_version key of the dataset mapping.

canonical_25 is the default of the --masks option of run_cortical_tiles and
generate_embeddings. YAML files are written under tmp_path and read with
yaml.safe_load / yaml.safe_dump.

main.py lives at the repo root (not inside the installed package), so it is
loaded by file path under a private module name.
"""

import dataclasses
import importlib.util
from pathlib import Path

import pytest
import yaml

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"

DEFAULT_MASKS_VERSION = "canonical_25"
OTHER_MASKS_VERSION = "canonical_corrected_26_1"


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_masks_version_under_test", MAIN_PY)
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
class TestDatasetConfigMasksVersionDefault:
    """REQ-EMBVER-BDRABCZUK-C56942772AB4: DatasetConfig.masks_version defaults to canonical_25."""

    def test_masks_version_field_defaults_to_canonical_25(self, main_module):
        """The DatasetConfig dataclass declares masks_version with default canonical_25."""
        fields = {f.name: f for f in dataclasses.fields(main_module.DatasetConfig)}

        assert "masks_version" in fields, (
            "REQ-EMBVER-BDRABCZUK-C56942772AB4: DatasetConfig declares no masks_version field; "
            f"fields are {sorted(fields)}"
        )
        assert fields["masks_version"].default == DEFAULT_MASKS_VERSION, (
            "REQ-EMBVER-BDRABCZUK-C56942772AB4: DatasetConfig.masks_version default is "
            f"{fields['masks_version'].default!r}, expected {DEFAULT_MASKS_VERSION!r}"
        )


@pytest.mark.unit
class TestLoadFromYamlMasksVersion:
    """REQ-EMBVER-BDRABCZUK-91A2A590197E / REQ-EMBVER-BDRABCZUK-CC2C03FB545E: loading dataset.masks_version."""

    def test_masks_version_key_is_loaded(self, main_module, tmp_path):
        """REQ-EMBVER-BDRABCZUK-91A2A590197E: dataset.masks_version in the YAML is the loaded value."""
        config_file = tmp_path / "config.yaml"
        _write_yaml(config_file, {"dataset": {"name": "example_dataset", "masks_version": OTHER_MASKS_VERSION}})

        loaded = main_module.ConfigLoader.load_from_yaml(str(config_file))

        assert loaded.dataset.masks_version == OTHER_MASKS_VERSION, (
            "REQ-EMBVER-BDRABCZUK-91A2A590197E: load_from_yaml returned dataset.masks_version="
            f"{loaded.dataset.masks_version!r}, expected {OTHER_MASKS_VERSION!r}"
        )

    def test_missing_masks_version_key_loads_canonical_25(self, main_module, tmp_path):
        """REQ-EMBVER-BDRABCZUK-CC2C03FB545E: a dataset mapping without masks_version loads canonical_25."""
        config_file = tmp_path / "config.yaml"
        _write_yaml(config_file, {"dataset": {"name": "example_dataset"}})

        loaded = main_module.ConfigLoader.load_from_yaml(str(config_file))

        assert getattr(loaded.dataset, "masks_version", None) == DEFAULT_MASKS_VERSION, (
            "REQ-EMBVER-BDRABCZUK-CC2C03FB545E: load_from_yaml without dataset.masks_version returned "
            f"{getattr(loaded.dataset, 'masks_version', '<no attribute>')!r}, expected {DEFAULT_MASKS_VERSION!r}"
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
        with open(output, "r") as f:
            written = yaml.safe_load(f)

        assert written["dataset"].get("masks_version") == OTHER_MASKS_VERSION, (
            "REQ-EMBVER-BDRABCZUK-A3CE0AFED232: saved dataset mapping has masks_version="
            f"{written['dataset'].get('masks_version')!r}, expected {OTHER_MASKS_VERSION!r}"
        )
