#!/usr/bin/env python3
"""
Removal of the dead dataset.use_best_model config field, and loading of YAML files that still
carry it (REQ-LASTCKPT-01, -02, -10, -13, -14; TASK-209). The --use_last_checkpoint flag these
tests also covered was removed with dataset.checkpoint_epoch (epic EPOCH80).

The main.py tests reuse test_main_stage_argv.py's module loader.
"""

import dataclasses
import logging
from pathlib import Path

import pytest
import yaml

from tests.test_main_stage_argv import _load_main_module

pytestmark = pytest.mark.unit  # noqa: V107


@pytest.fixture(scope="module")
def main_module():
    return _load_main_module()


# --------------------------------------------------------------------------- #
# REQ-LASTCKPT-01 / 02: dead use_best_model field removed
# --------------------------------------------------------------------------- #


class TestDatasetConfigHasNoUseBestModel:
    """REQ-LASTCKPT-01: main.py DatasetConfig declares no use_best_model field."""

    def test_dataset_config_declares_no_use_best_model_field(self, main_module):
        names = {f.name for f in dataclasses.fields(main_module.DatasetConfig)}
        assert "use_best_model" not in names, "DatasetConfig still declares the dead use_best_model field"


class TestSavedConfigHasNoUseBestModel:
    """REQ-LASTCKPT-02: save_to_yaml writes no dataset.use_best_model key."""

    def test_saved_dataset_mapping_has_no_use_best_model_key(self, main_module, tmp_path):
        output = tmp_path / "saved.yaml"
        main_module.ConfigLoader.save_to_yaml(main_module.PipelineConfig(), str(output))
        with open(output) as f:
            written = yaml.safe_load(f)
        assert "use_best_model" not in written["dataset"], "save_to_yaml still writes dataset.use_best_model"


# --------------------------------------------------------------------------- #
# REQ-LASTCKPT-10 / 13 / 14: loading a YAML that still carries use_best_model
# --------------------------------------------------------------------------- #

# A dataset mapping with only live DatasetConfig fields.
_LIVE_DATASET = {"input_path": "/raw", "datasets_root": "/data/MYDATASET"}


def _write_yaml(tmp_path: Path, name: str, dataset: dict) -> str:
    path = tmp_path / name
    path.write_text(yaml.safe_dump({"models_path": "/models", "dataset": dataset}))
    return str(path)


class TestLoadIgnoresRemovedUseBestModel:
    """REQ-LASTCKPT-10: a dataset.use_best_model key loads to the same PipelineConfig as without it."""

    @pytest.mark.parametrize("value", [False, True])
    def test_use_best_model_key_loads_equal_to_config_without_it(self, main_module, tmp_path, value):
        without = _write_yaml(tmp_path, "without.yaml", dict(_LIVE_DATASET))
        with_key = _write_yaml(tmp_path, "with.yaml", {**_LIVE_DATASET, "use_best_model": value})

        expected = main_module.ConfigLoader.load_from_yaml(without)
        try:
            loaded = main_module.ConfigLoader.load_from_yaml(with_key)
        except TypeError as exc:
            pytest.fail(f"load_from_yaml rejected dataset.use_best_model: {value}: {exc}")

        assert loaded == expected


class TestLoadWarnsOnRemovedUseBestModel:
    """REQ-LASTCKPT-13: loading dataset.use_best_model logs one WARNING record naming dataset.use_best_model."""

    def test_one_warning_names_dataset_use_best_model(self, main_module, tmp_path, caplog):
        path = _write_yaml(tmp_path, "with.yaml", {**_LIVE_DATASET, "use_best_model": False})
        # Also listen on the pipeline logger directly, in case another test turned propagation off.
        pipeline_logger = logging.getLogger("champollion_pipeline")
        pipeline_logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.WARNING), caplog.at_level(logging.WARNING, logger="champollion_pipeline"):
                try:
                    main_module.ConfigLoader.load_from_yaml(path)
                except TypeError as exc:
                    pytest.fail(f"load_from_yaml rejected dataset.use_best_model: {exc}")
        finally:
            pipeline_logger.removeHandler(caplog.handler)

        unique = {id(r): r for r in caplog.records}.values()
        naming = [r for r in unique if r.levelno == logging.WARNING and "dataset.use_best_model" in r.getMessage()]
        assert len(naming) == 1, [(r.levelname, r.getMessage()) for r in unique]


class TestLoadRejectsOtherUnknownDatasetKey:
    """REQ-LASTCKPT-14: any other unknown dataset key still raises TypeError (characterization)."""

    def test_unknown_dataset_key_raises_type_error(self, main_module, tmp_path):
        path = _write_yaml(tmp_path, "typo.yaml", {**_LIVE_DATASET, "use_last_chekpoint": True})
        with pytest.raises(TypeError):
            main_module.ConfigLoader.load_from_yaml(path)


# --------------------------------------------------------------------------- #
# REQ-LASTCKPT-11 / 12: flag with best weights and no native ckpt in version_0
# --------------------------------------------------------------------------- #
