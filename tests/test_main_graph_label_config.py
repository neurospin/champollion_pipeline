#!/usr/bin/env python3
"""
Tests for the dataset.graph_label setting of the main.py pipeline config
(ARGDISC epic: the labelling set that run_cortical_tiles --graph_label picks when a subject folder holds the graphs
of several labelling sets; an empty value passes no --graph_label, so the default behaviour is unchanged).

The graph_label field is a labelling set name (for example "default_session_auto"), not a labelling session:
--labelling_session keeps its own setting.

- DatasetConfig declares graph_label: str = "".
- ConfigLoader.load_from_yaml reads, and save_to_yaml writes, the dataset mapping's graph_label key.
- main.py's --graph-label argument (one argument per config field, from the cli_options registry) sets
  dataset.graph_label; the field is not set when the argument is absent.
- RunCorticalTilesStage forwards a non-empty value as --graph_label and nothing for an empty one; the argv is parsed
  with the real RunCorticalTiles parser.
- The streaming per-scan stage 2 command (streaming.stages._cortical_tiles_argv) follows the same rule.

main.py lives at the repo root (not inside the installed package): cli.py is loaded by file path under a private name.
"""

import argparse
import dataclasses
import typing

import pytest
import yaml

from champollion_pipeline.cli_options import add_config_arguments, apply_config_arguments
from champollion_pipeline.config import DatasetConfig
from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from champollion_pipeline.streaming.settings import StreamingSettings
from champollion_pipeline.streaming.stages import _cortical_tiles_argv
from tests.main_stage_helpers import load_main_module, option_values, tiles_config, tiles_stage_argv

GRAPH_LABEL_OPTION = "--graph_label"
CONFIGURED_LABEL = "default_session_auto"

REQ_FIELD = "REQ-ARGDISC-BDRABCZUK-FIELD"
REQ_LOAD = "REQ-ARGDISC-BDRABCZUK-LOAD"
REQ_SAVE = "REQ-ARGDISC-BDRABCZUK-SAVE"
REQ_CLI = "REQ-ARGDISC-BDRABCZUK-CLI"
REQ_FORWARD = "REQ-ARGDISC-BDRABCZUK-FORWARD"
REQ_NO_FORWARD = "REQ-ARGDISC-BDRABCZUK-NOFORWARD"
REQ_STREAM = "REQ-ARGDISC-BDRABCZUK-STREAM"
REQ_STREAM_EMPTY = "REQ-ARGDISC-BDRABCZUK-STREAMEMPTY"


@pytest.fixture
def main_module():
    return load_main_module("_champollion_main_graph_label_config_under_test")


def _loaded_graph_label(main_module, tmp_path, dataset_mapping):
    """dataset.graph_label that load_from_yaml returns for a YAML file holding dataset_mapping."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump({"dataset": dataset_mapping}))
    dataset = main_module.ConfigLoader.load_from_yaml(str(config_file)).dataset
    return getattr(dataset, "graph_label", "<no attribute>")


def _saved_dataset(main_module, tmp_path, config):
    """The dataset mapping that save_to_yaml writes for config."""
    output = tmp_path / "saved.yaml"
    main_module.ConfigLoader.save_to_yaml(config, str(output))
    return yaml.safe_load(output.read_text())["dataset"]


@pytest.mark.unit
class TestDatasetConfigGraphLabelDefault:
    def test_graph_label_field_defaults_to_empty_string(self):
        """The DatasetConfig dataclass declares graph_label: str = ""."""
        fields = {f.name: f for f in dataclasses.fields(DatasetConfig)}
        assert "graph_label" in fields, f"{REQ_FIELD}: DatasetConfig declares no graph_label field"
        assert fields["graph_label"].default == "", f"{REQ_FIELD}: default is {fields['graph_label'].default!r}"
        assert typing.get_type_hints(DatasetConfig)["graph_label"] is str


@pytest.mark.unit
class TestYamlGraphLabel:
    def test_graph_label_key_is_loaded(self, main_module, tmp_path):
        assert _loaded_graph_label(main_module, tmp_path, {"graph_label": CONFIGURED_LABEL}) == CONFIGURED_LABEL, (
            REQ_LOAD
        )

    def test_graph_label_is_written_under_dataset(self, main_module, tmp_path):
        config = main_module.PipelineConfig()
        config.dataset.graph_label = CONFIGURED_LABEL

        assert _saved_dataset(main_module, tmp_path, config).get("graph_label", "<no key>") == CONFIGURED_LABEL, (
            REQ_SAVE
        )

    def test_default_config_writes_empty_graph_label(self, main_module, tmp_path):
        saved = _saved_dataset(main_module, tmp_path, main_module.PipelineConfig())

        assert saved.get("graph_label", "<no key>") == "", REQ_SAVE


@pytest.mark.unit
class TestGraphLabelArgument:
    """The --graph-label argument of main.py sets dataset.graph_label (CLI first; the YAML is for replay)."""

    @staticmethod
    def _apply(main_module, argv):
        parser = argparse.ArgumentParser(add_help=False)
        add_config_arguments(parser)
        config = main_module.PipelineConfig()
        apply_config_arguments(config, parser.parse_args(argv))
        return config

    def test_argument_sets_the_field(self, main_module):
        config = self._apply(main_module, ["--graph-label", CONFIGURED_LABEL])

        assert getattr(config.dataset, "graph_label", "<no attribute>") == CONFIGURED_LABEL, REQ_CLI

    def test_absent_argument_keeps_the_default(self, main_module):
        config = self._apply(main_module, [])

        assert getattr(config.dataset, "graph_label", "<no attribute>") == "", REQ_CLI


def _graph_label_parsed(argv):
    """graph_label of the real RunCorticalTiles parser for argv; a rejected argv is a failure, not an error."""
    try:
        return RunCorticalTiles().parse_args(argv).graph_label
    except SystemExit as exc:
        pytest.fail(f"RunCorticalTiles parser rejected {argv!r} (SystemExit {exc.code})")


@pytest.mark.unit
class TestRunCorticalTilesStageGraphLabelArgv:
    def test_non_empty_graph_label_passed_as_option(self, main_module):
        config = tiles_config(main_module)
        config.dataset.graph_label = CONFIGURED_LABEL  # noqa: V101

        argv = [str(t) for t in tiles_stage_argv(main_module, config, "test_main_graph_label_config")]

        assert option_values(argv, GRAPH_LABEL_OPTION) == [CONFIGURED_LABEL], f"{REQ_FORWARD}: argv was {argv!r}"
        assert _graph_label_parsed(argv) == CONFIGURED_LABEL

    def test_empty_graph_label_passes_no_option(self, main_module):
        config = tiles_config(main_module)
        assert getattr(config.dataset, "graph_label", "<no attribute>") == "", REQ_NO_FORWARD

        argv = [str(t) for t in tiles_stage_argv(main_module, config, "test_main_graph_label_config")]

        assert option_values(argv, GRAPH_LABEL_OPTION) == [], f"{REQ_NO_FORWARD}: argv was {argv!r}"
        assert _graph_label_parsed(argv) is None


def _streaming_argv(tmp_path, dataset):
    settings = StreamingSettings(
        subjects_dir=str(tmp_path / "subjects"),
        output_dir=str(tmp_path / "out"),
        log_dir=str(tmp_path / "logs"),
        n_workers=1,
    )
    return _cortical_tiles_argv(str(tmp_path / "in"), str(tmp_path / "out"), settings, dataset)


@pytest.mark.unit
class TestStreamingStageGraphLabelArgv:
    def test_non_empty_graph_label_reaches_the_scan_command(self, tmp_path):
        dataset = DatasetConfig()
        dataset.graph_label = CONFIGURED_LABEL  # noqa: V101

        argv = _streaming_argv(tmp_path, dataset)

        assert option_values(argv, GRAPH_LABEL_OPTION) == [CONFIGURED_LABEL], f"{REQ_STREAM}: argv was {argv!r}"

    def test_empty_graph_label_passes_no_option(self, tmp_path):
        argv = _streaming_argv(tmp_path, DatasetConfig())

        assert option_values(argv, GRAPH_LABEL_OPTION) == [], f"{REQ_STREAM_EMPTY}: argv was {argv!r}"
