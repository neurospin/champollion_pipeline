#!/usr/bin/env python3
"""
Stage 4 model options of the main.py pipeline config: dataset.per_model_folders and dataset.checkpoint_epochs
(TASK-MULTIMODEL-BDRABCZUK-8AB92FCF3DDD, epic TASK-MULTIMODEL-BDRABCZUK-A02369CED8AE).

Both are DatasetConfig fields, so cli_options derives their main.py arguments; their values reach generate_embeddings
through GenerateEmbeddingsStage and survive the replay file round trip. Unset, they keep today's stage 4 argv.

Requirements (wording in the ledger; parents S1 REQ-MULTIMODEL-BDRABCZUK-25A9FF6998AA, S4
REQ-MULTIMODEL-BDRABCZUK-7A5BD71C6EE7):
    TestPerModelFoldersField: DatasetConfig.per_model_folders, bool, default False.
    TestCheckpointEpochsField: DatasetConfig.checkpoint_epochs, list[str], default empty.
    TestYamlLoad: ConfigLoader.load_from_yaml reads both keys.
    TestMainCommandLine: main.py --per-model-folders and the dataset.checkpoint_epochs argument set the fields.
    TestStageArgv: GenerateEmbeddingsStage passes --per-model-folders when set and one --checkpoint-epoch <id>=N per
        value; unset, no --per-model-folders and exactly one --checkpoint-epoch.
    TestReplayRoundTrip: write_replay_config then load_from_yaml keeps both values.

The pass-through is tested at the main.py -> stage 4 argv boundary: GenerateEmbeddings is a MagicMock and the argv
it receives is read in either the '--opt=V' or the '--opt V' form; stage 4 internals are not run. The flag spelling
and the list-or-repeat shape of the checkpoint_epochs argument are read from the cli_options registry and its parser,
not written here.
"""

import argparse
import collections
import dataclasses
import typing
from pathlib import Path

import pytest
import yaml

from champollion_pipeline import replay_config
from champollion_pipeline.cli_options import CONFIG_ARGUMENTS, add_config_arguments
from champollion_pipeline.config import ConfigLoader, DatasetConfig, PipelineConfig
from tests.main_stage_helpers import (
    embeddings_stage_argv,
    enter_work_dir,
    load_main_module,
    main_runner,
    option_values,
    ran_config,
    stage_paths_config,
)

pytestmark = pytest.mark.unit  # noqa: V107

MAIN = load_main_module("_champollion_main_multi_model_config_under_test")

PER_MODEL_FOLDERS = "per_model_folders"
CHECKPOINT_EPOCHS = "checkpoint_epochs"
PER_MODEL_FOLDERS_FLAG = "--per-model-folders"
CHECKPOINT_EPOCH_OPTION = "--checkpoint-epoch"
# '<id>=N' values: a plain id, a nested id, an id holding '=' (split at the last '=').
ID_EPOCHS = ["run_3=60", "exp/v2=90", "lr=0.1=70"]
LOGGER_NAME = "test_main_multi_model_config"

# A dataset mapping with only live DatasetConfig fields.
_LIVE_DATASET = {"input_path": "/raw", "datasets_root": "/data/MYDATASET"}


def _dataset_field(name: str) -> dataclasses.Field:
    """The DatasetConfig field called name; fails the test when DatasetConfig declares none."""
    fields = {f.name: f for f in dataclasses.fields(DatasetConfig)}
    if name not in fields:
        pytest.fail(f"DatasetConfig declares no {name} field")
    return fields[name]


def _with_dataset(config: PipelineConfig, **values) -> PipelineConfig:
    """config with the given DatasetConfig field values; fails the test when a field is not declared."""
    for name in values:
        _dataset_field(name)
    config.dataset = dataclasses.replace(config.dataset, **values)  # noqa: V101
    return config


def _write_yaml(tmp_path: Path, dataset: dict) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"models_path": "/models", "dataset": dataset}))
    return str(path)


def _load(path: str) -> PipelineConfig:
    try:
        return ConfigLoader.load_from_yaml(path)
    except TypeError as exc:
        pytest.fail(f"load_from_yaml rejected {path}: {exc}")


def _loaded_value(loaded: PipelineConfig, name: str):
    if not hasattr(loaded.dataset, name):
        pytest.fail(f"the loaded DatasetConfig has no {name} attribute")
    return getattr(loaded.dataset, name)


def _stage_argv(config: PipelineConfig) -> list[str]:
    return embeddings_stage_argv(MAIN, config, "generate_embeddings", LOGGER_NAME)


# --------------------------------------------------------------------------- #
# DatasetConfig fields
# --------------------------------------------------------------------------- #


class TestPerModelFoldersField:
    """DatasetConfig.per_model_folders: bool, default False; a YAML file without the key loads False."""

    def test_field_is_bool_defaulting_to_false(self):
        _dataset_field(PER_MODEL_FOLDERS)
        assert typing.get_type_hints(DatasetConfig)[PER_MODEL_FOLDERS] is bool
        assert DatasetConfig().per_model_folders is False

    def test_yaml_without_the_key_loads_false(self, tmp_path):
        loaded = _load(_write_yaml(tmp_path, dict(_LIVE_DATASET)))
        assert _loaded_value(loaded, PER_MODEL_FOLDERS) is False


class TestCheckpointEpochsField:
    """DatasetConfig.checkpoint_epochs: list[str], default empty; a YAML file without the key loads empty."""

    def test_field_is_str_list_defaulting_to_empty(self):
        _dataset_field(CHECKPOINT_EPOCHS)
        assert typing.get_type_hints(DatasetConfig)[CHECKPOINT_EPOCHS] == list[str]
        first, second = DatasetConfig(), DatasetConfig()
        assert first.checkpoint_epochs == []
        assert first.checkpoint_epochs is not second.checkpoint_epochs, "the default list is shared between configs"

    def test_yaml_without_the_key_loads_empty(self, tmp_path):
        loaded = _load(_write_yaml(tmp_path, dict(_LIVE_DATASET)))
        assert _loaded_value(loaded, CHECKPOINT_EPOCHS) == []


# --------------------------------------------------------------------------- #
# YAML load
# --------------------------------------------------------------------------- #


class TestYamlLoad:
    """ConfigLoader.load_from_yaml reads dataset.per_model_folders and dataset.checkpoint_epochs."""

    @pytest.mark.parametrize("value", [True, False])
    def test_per_model_folders_key_is_read(self, tmp_path, value):
        loaded = _load(_write_yaml(tmp_path, {**_LIVE_DATASET, PER_MODEL_FOLDERS: value}))
        assert _loaded_value(loaded, PER_MODEL_FOLDERS) is value

    def test_checkpoint_epochs_key_is_read(self, tmp_path):
        loaded = _load(_write_yaml(tmp_path, {**_LIVE_DATASET, CHECKPOINT_EPOCHS: list(ID_EPOCHS)}))
        assert _loaded_value(loaded, CHECKPOINT_EPOCHS) == ID_EPOCHS


# --------------------------------------------------------------------------- #
# main.py command line
# --------------------------------------------------------------------------- #


def _registry_row(field_name: str):
    rows = [row for row in CONFIG_ARGUMENTS if row.is_dataset_field and row.field_name == field_name]
    if not rows:
        pytest.fail(f"the cli_options registry has no argument for dataset.{field_name}")
    return rows[0]


def _list_arguments(field_name: str, values: list[str]) -> list[str]:
    """Command-line arguments giving values to the field's argument: one list, or the flag repeated per value."""
    row = _registry_row(field_name)
    parser = argparse.ArgumentParser(add_help=False)
    add_config_arguments(parser)
    action = parser._option_string_actions[row.flag]
    if action.nargs in ("+", "*"):
        return [row.flag, *values]
    return [token for value in values for token in (row.flag, value)]


@pytest.fixture
def run_main(monkeypatch, tmp_path):
    """Run main() with the given arguments; return the one PipelineConfig it handed to the orchestrator."""
    enter_work_dir(monkeypatch, tmp_path)
    runner = main_runner(MAIN, monkeypatch)

    def _run(*cli_args: str) -> PipelineConfig:
        return ran_config(runner(*cli_args), cli_args)

    return _run


class TestMainCommandLine:
    """main.py --per-model-folders and the dataset.checkpoint_epochs argument set the config fields."""

    def test_per_model_folders_flag_sets_the_field(self, run_main):
        config = run_main(PER_MODEL_FOLDERS_FLAG)
        assert getattr(config.dataset, PER_MODEL_FOLDERS, None) is True

    def test_checkpoint_epochs_argument_sets_the_field(self, run_main):
        cli_args = _list_arguments(CHECKPOINT_EPOCHS, ID_EPOCHS)
        config = run_main(*cli_args)
        assert getattr(config.dataset, CHECKPOINT_EPOCHS, None) == ID_EPOCHS, cli_args


# --------------------------------------------------------------------------- #
# Stage 4 argv
# --------------------------------------------------------------------------- #


class TestStageArgv:
    """GenerateEmbeddingsStage forwards the two fields to GenerateEmbeddings; unset, the argv is today's."""

    def test_per_model_folders_true_passes_the_flag(self):
        config = _with_dataset(stage_paths_config(MAIN), per_model_folders=True)
        argv = _stage_argv(config)
        assert argv.count(PER_MODEL_FOLDERS_FLAG) == 1, argv

    def test_per_model_folders_false_passes_no_flag(self):
        config = _with_dataset(stage_paths_config(MAIN), per_model_folders=False)
        argv = _stage_argv(config)
        assert not [token for token in argv if token.startswith(PER_MODEL_FOLDERS_FLAG)], argv

    def test_each_checkpoint_epochs_value_is_one_checkpoint_epoch_option(self):
        config = _with_dataset(stage_paths_config(MAIN), checkpoint_epochs=list(ID_EPOCHS))
        values = collections.Counter(option_values(_stage_argv(config), CHECKPOINT_EPOCH_OPTION))
        assert {value: values[value] for value in ID_EPOCHS} == dict.fromkeys(ID_EPOCHS, 1), values

    def test_default_epoch_option_is_kept_beside_the_id_values(self):
        config = _with_dataset(stage_paths_config(MAIN), checkpoint_epoch=37, checkpoint_epochs=list(ID_EPOCHS))
        values = option_values(_stage_argv(config), CHECKPOINT_EPOCH_OPTION)
        assert sorted(values) == sorted(["37", *ID_EPOCHS]), values

    def test_empty_checkpoint_epochs_passes_one_checkpoint_epoch_option(self):
        config = _with_dataset(stage_paths_config(MAIN), checkpoint_epoch=37, checkpoint_epochs=[])
        values = option_values(_stage_argv(config), CHECKPOINT_EPOCH_OPTION)
        assert values == ["37"], values


# --------------------------------------------------------------------------- #
# Replay round trip
# --------------------------------------------------------------------------- #


def _replayed(tmp_path: Path, **dataset_values) -> PipelineConfig:
    """write_replay_config of a config holding dataset_values, read back by ConfigLoader.load_from_yaml."""
    config = _with_dataset(PipelineConfig(log_dir=str(tmp_path / "logs")), **dataset_values)
    return _load(replay_config.write_replay_config(config))


class TestReplayRoundTrip:
    """The replay file keeps dataset.per_model_folders and dataset.checkpoint_epochs."""

    @pytest.mark.parametrize("value", [True, False])
    def test_per_model_folders_survives_the_replay(self, tmp_path, value):
        replayed = _replayed(tmp_path, per_model_folders=value)
        assert _loaded_value(replayed, PER_MODEL_FOLDERS) is value

    @pytest.mark.parametrize("values", [[], ID_EPOCHS], ids=["empty", "three_ids"])
    def test_checkpoint_epochs_survive_the_replay(self, tmp_path, values):
        replayed = _replayed(tmp_path, checkpoint_epochs=list(values))
        assert _loaded_value(replayed, CHECKPOINT_EPOCHS) == values
