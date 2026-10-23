#!/usr/bin/env python3
"""
Tests for the dataset.checkpoint_epoch setting of the main.py pipeline config,
which replaces dataset.use_last_checkpoint (TASK-EPOCH80-BDRABCZUK-28E2E3809251).

- REQ-EPOCH80-BDRABCZUK-53D867A8EE79: DatasetConfig declares a checkpoint_epoch
  field of type int whose default value is 80.
- REQ-EPOCH80-BDRABCZUK-FB8CAF3D89F8: DatasetConfig declares no use_last_checkpoint field.
- REQ-EPOCH80-BDRABCZUK-F7CF9DE7F056: ConfigLoader.load_from_yaml reads the dataset
  mapping's checkpoint_epoch key.
- REQ-EPOCH80-BDRABCZUK-9D30ADB37EB2: ConfigLoader.save_to_yaml writes
  config.dataset.checkpoint_epoch as the dataset mapping's checkpoint_epoch key.
- REQ-EPOCH80-BDRABCZUK-827DF88D1467: ConfigLoader.save_to_yaml writes no
  use_last_checkpoint key.
- REQ-EPOCH80-BDRABCZUK-53FF590C899A: GenerateEmbeddingsStage passes
  GenerateEmbeddings the argv element --checkpoint-epoch=<n>.
- REQ-EPOCH80-BDRABCZUK-642AE9008237: a YAML dataset.use_last_checkpoint key loads
  to the same PipelineConfig as without it.
- REQ-EPOCH80-BDRABCZUK-8CC60F128FF8: loading dataset.use_last_checkpoint logs one
  WARNING record naming dataset.use_last_checkpoint and dataset.checkpoint_epoch.
- REQ-EPOCH80-BDRABCZUK-96DDF8C5A24A: loading dataset.use_best_model logs no record
  mentioning use_last_checkpoint.
- REQ-EPOCH80-BDRABCZUK-99F714DBA4B7: loading dataset.use_best_model logs one
  WARNING record naming dataset.checkpoint_epoch.

main.py lives at the repo root (not inside the installed package), so it is
loaded by file path (this checkout's main.py) under a private module name. The
stage test replaces the GenerateEmbeddings class referenced by main.py with a
MagicMock to capture the argv GenerateEmbeddingsStage builds without running
anything.
"""

import dataclasses
import logging
import typing
from pathlib import Path

import pytest
import yaml

from tests.test_main_stage_argv import _base_config, _captured_argv, _load_main_module

pytestmark = pytest.mark.unit  # noqa: V107

DEFAULT_EPOCH = 80

# A dataset mapping with only live DatasetConfig fields.
_LIVE_DATASET = {"input_path": "/raw", "datasets_root": "/data/MYDATASET"}


@pytest.fixture(scope="module")
def main_module():
    return _load_main_module()


def _write_yaml(tmp_path: Path, name: str, dataset: dict) -> str:
    path = tmp_path / name
    path.write_text(yaml.safe_dump({"models_path": "/models", "dataset": dataset}))
    return str(path)


def _saved_dataset_mapping(main_module, tmp_path: Path, config) -> dict:
    output = tmp_path / "saved.yaml"
    main_module.ConfigLoader.save_to_yaml(config, str(output))
    with open(output) as f:
        return yaml.safe_load(f)["dataset"]


def _load_records(main_module, path: str, caplog) -> list[logging.LogRecord]:
    """Load ``path`` with ConfigLoader.load_from_yaml; return the distinct log records emitted."""
    # Also listen on the pipeline logger directly, in case another test turned propagation off.
    pipeline_logger = logging.getLogger("champollion_pipeline")
    pipeline_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.DEBUG), caplog.at_level(logging.DEBUG, logger="champollion_pipeline"):
            try:
                main_module.ConfigLoader.load_from_yaml(path)
            except TypeError as exc:
                pytest.fail(f"load_from_yaml rejected {path}: {exc}")
    finally:
        pipeline_logger.removeHandler(caplog.handler)
    return list({id(r): r for r in caplog.records}.values())


def _describe(records: list[logging.LogRecord]) -> list[tuple[str, str]]:
    return [(r.levelname, r.getMessage()) for r in records]


# --------------------------------------------------------------------------- #
# DatasetConfig fields
# --------------------------------------------------------------------------- #


class TestDatasetConfigCheckpointEpoch:
    """REQ-EPOCH80-BDRABCZUK-53D867A8EE79: checkpoint_epoch: int = 80."""

    def test_field_is_int_defaulting_to_80(self, main_module):
        fields = {f.name: f for f in dataclasses.fields(main_module.DatasetConfig)}
        assert "checkpoint_epoch" in fields, "DatasetConfig declares no checkpoint_epoch field"
        assert typing.get_type_hints(main_module.DatasetConfig)["checkpoint_epoch"] is int
        default = main_module.DatasetConfig().checkpoint_epoch
        assert type(default) is int and default == DEFAULT_EPOCH


class TestDatasetConfigHasNoUseLastCheckpoint:
    """REQ-EPOCH80-BDRABCZUK-FB8CAF3D89F8: no use_last_checkpoint field."""

    def test_declares_no_use_last_checkpoint_field(self, main_module):
        names = {f.name for f in dataclasses.fields(main_module.DatasetConfig)}
        assert "use_last_checkpoint" not in names, "DatasetConfig still declares use_last_checkpoint"


# --------------------------------------------------------------------------- #
# YAML load / save
# --------------------------------------------------------------------------- #


class TestLoadReadsCheckpointEpoch:
    """REQ-EPOCH80-BDRABCZUK-F7CF9DE7F056: load_from_yaml reads dataset.checkpoint_epoch."""

    @pytest.mark.parametrize("epoch", [0, 37, 80])
    def test_loaded_value_equals_yaml_value(self, main_module, tmp_path, epoch):
        path = _write_yaml(tmp_path, "epoch.yaml", {**_LIVE_DATASET, "checkpoint_epoch": epoch})
        try:
            loaded = main_module.ConfigLoader.load_from_yaml(path)
        except TypeError as exc:
            pytest.fail(f"load_from_yaml rejected dataset.checkpoint_epoch: {epoch}: {exc}")
        assert loaded.dataset.checkpoint_epoch == epoch


class TestSaveWritesCheckpointEpoch:
    """REQ-EPOCH80-BDRABCZUK-9D30ADB37EB2: save_to_yaml writes dataset.checkpoint_epoch."""

    @pytest.mark.parametrize("epoch", [0, 37, 80])
    def test_saved_value_equals_config_value(self, main_module, tmp_path, epoch):
        config = main_module.PipelineConfig()
        config.dataset.checkpoint_epoch = epoch  # noqa: V101
        dataset = _saved_dataset_mapping(main_module, tmp_path, config)
        assert "checkpoint_epoch" in dataset, f"save_to_yaml wrote no checkpoint_epoch key: {sorted(dataset)}"
        assert dataset["checkpoint_epoch"] == epoch


class TestSaveWritesNoUseLastCheckpoint:
    """REQ-EPOCH80-BDRABCZUK-827DF88D1467: save_to_yaml writes no dataset.use_last_checkpoint key."""

    def test_saved_dataset_mapping_has_no_use_last_checkpoint_key(self, main_module, tmp_path):
        dataset = _saved_dataset_mapping(main_module, tmp_path, main_module.PipelineConfig())
        assert "use_last_checkpoint" not in dataset, "save_to_yaml still writes dataset.use_last_checkpoint"


# --------------------------------------------------------------------------- #
# Stage argv
# --------------------------------------------------------------------------- #


class TestStageForwardsCheckpointEpoch:
    """REQ-EPOCH80-BDRABCZUK-53FF590C899A: argv carries --checkpoint-epoch=<n>."""

    @pytest.mark.parametrize("epoch", [0, 37, 80])
    def test_argv_has_checkpoint_epoch_element(self, main_module, epoch):
        config = _base_config(main_module)
        config.dataset.checkpoint_epoch = epoch  # noqa: V101
        argv = _captured_argv(main_module, "GenerateEmbeddingsStage", "GenerateEmbeddings", config)
        assert f"--checkpoint-epoch={epoch}" in argv, argv


# --------------------------------------------------------------------------- #
# Removed dataset.use_last_checkpoint key
# --------------------------------------------------------------------------- #


class TestLoadIgnoresRemovedUseLastCheckpoint:
    """REQ-EPOCH80-BDRABCZUK-642AE9008237: use_last_checkpoint loads equal to the config without it."""

    @pytest.mark.parametrize("value", [False, True])
    def test_use_last_checkpoint_key_loads_equal_to_config_without_it(self, main_module, tmp_path, value):
        without = _write_yaml(tmp_path, "without.yaml", dict(_LIVE_DATASET))
        with_key = _write_yaml(tmp_path, "with.yaml", {**_LIVE_DATASET, "use_last_checkpoint": value})

        expected = main_module.ConfigLoader.load_from_yaml(without)
        try:
            loaded = main_module.ConfigLoader.load_from_yaml(with_key)
        except TypeError as exc:
            pytest.fail(f"load_from_yaml rejected dataset.use_last_checkpoint: {value}: {exc}")

        assert loaded == expected


class TestLoadWarnsOnRemovedUseLastCheckpoint:
    """REQ-EPOCH80-BDRABCZUK-8CC60F128FF8: one WARNING naming the removed key and its replacement."""

    def test_one_warning_names_removed_key_and_replacement(self, main_module, tmp_path, caplog):
        path = _write_yaml(tmp_path, "with.yaml", {**_LIVE_DATASET, "use_last_checkpoint": True})
        records = _load_records(main_module, path, caplog)
        naming = [
            r
            for r in records
            if r.levelno == logging.WARNING
            and "dataset.use_last_checkpoint" in r.getMessage()
            and "dataset.checkpoint_epoch" in r.getMessage()
        ]
        assert len(naming) == 1, _describe(records)


# --------------------------------------------------------------------------- #
# Updated dataset.use_best_model removed-key message
# --------------------------------------------------------------------------- #


class TestUseBestModelWarningUpdated:
    """REQ-EPOCH80-BDRABCZUK-96DDF8C5A24A / -99F714DBA4B7: the use_best_model warning text."""

    def test_no_record_mentions_use_last_checkpoint(self, main_module, tmp_path, caplog):
        path = _write_yaml(tmp_path, "with.yaml", {**_LIVE_DATASET, "use_best_model": False})
        records = _load_records(main_module, path, caplog)
        assert any("dataset.use_best_model" in r.getMessage() for r in records), (
            "sanity check: no record about dataset.use_best_model was captured",
            _describe(records),
        )
        mentioning = [r for r in records if "use_last_checkpoint" in r.getMessage()]
        assert mentioning == [], _describe(mentioning)

    def test_one_warning_names_checkpoint_epoch(self, main_module, tmp_path, caplog):
        path = _write_yaml(tmp_path, "with.yaml", {**_LIVE_DATASET, "use_best_model": False})
        records = _load_records(main_module, path, caplog)
        naming = [r for r in records if r.levelno == logging.WARNING and "dataset.checkpoint_epoch" in r.getMessage()]
        assert len(naming) == 1, _describe(records)
