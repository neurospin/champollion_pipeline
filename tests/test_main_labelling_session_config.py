#!/usr/bin/env python3
"""
Tests for the dataset.labelling_session setting of the main.py pipeline config
(whole-brain epic: the labelling session reaches run_cortical_tiles from the
YAML pipeline config; an empty value keeps the per-subject auto-detection of
REQ-LABELSESSION-BDRABCZUK-DFC6751DB4CA).

- REQ-WHOLEBRAIN-BDRABCZUK-B6C5701E5530: DatasetConfig declares a
  labelling_session field of type str whose default value is "".
- REQ-WHOLEBRAIN-BDRABCZUK-E0C750353087: ConfigLoader.load_from_yaml reads the
  dataset mapping's labelling_session key.
- REQ-WHOLEBRAIN-BDRABCZUK-3DBF0C9F422F: ConfigLoader.save_to_yaml writes
  dataset.labelling_session as the labelling_session key of the dataset mapping.
- REQ-WHOLEBRAIN-BDRABCZUK-D657457C56CF: a non-empty labelling_session S
  reaches run_cortical_tiles as --labelling_session S.
- REQ-WHOLEBRAIN-BDRABCZUK-96FF8A402A81: an empty labelling_session passes no
  --labelling_session option to run_cortical_tiles.

The stage tests replace the RunCorticalTiles class referenced by main.py with a
MagicMock to capture the argv RunCorticalTilesStage builds without running
anything; that argv is then parsed with the real RunCorticalTiles parser, so
the asserted values (labelling_session, labelling_session_explicit) are the
ones run_cortical_tiles would receive.

YAML files are written under tmp_path with yaml.safe_dump and read with
yaml.safe_load. main.py lives at the repo root (not inside the installed
package), so it is loaded by file path under a private module name.
"""

import dataclasses
import importlib.util
import logging
import typing
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from champollion_pipeline.run_cortical_tiles import RunCorticalTiles

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"

LABELLING_SESSION_OPTION = "--labelling_session"
CONFIGURED_SESSION = "0_auto"

REQ_FIELD = "REQ-WHOLEBRAIN-BDRABCZUK-B6C5701E5530"
REQ_LOAD = "REQ-WHOLEBRAIN-BDRABCZUK-E0C750353087"
REQ_SAVE = "REQ-WHOLEBRAIN-BDRABCZUK-3DBF0C9F422F"
REQ_FORWARD = "REQ-WHOLEBRAIN-BDRABCZUK-D657457C56CF"
REQ_NO_FORWARD = "REQ-WHOLEBRAIN-BDRABCZUK-96FF8A402A81"


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_labelling_session_config_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def main_module():
    return _load_main_module()


def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def _load_dataset(main_module, path, req_id):
    """load_from_yaml(path).dataset; a rejected dataset key is a test failure, not an error."""
    try:
        return main_module.ConfigLoader.load_from_yaml(str(path)).dataset
    except TypeError as exc:
        pytest.fail(f"{req_id}: load_from_yaml rejected the dataset mapping: {exc}")


# ---------------------------------------------------------------- DatasetConfig


@pytest.mark.unit
class TestDatasetConfigLabellingSessionDefault:
    """REQ-WHOLEBRAIN-BDRABCZUK-B6C5701E5530: DatasetConfig.labelling_session is str, default ""."""

    def test_labelling_session_field_defaults_to_empty_string(self, main_module):
        """The DatasetConfig dataclass declares labelling_session: str = ""."""
        fields = {f.name: f for f in dataclasses.fields(main_module.DatasetConfig)}
        assert "labelling_session" in fields, (
            f"{REQ_FIELD}: DatasetConfig declares no labelling_session field; fields are {sorted(fields)}"
        )
        field = fields["labelling_session"]

        assert field.default == "", (
            f"{REQ_FIELD}: DatasetConfig.labelling_session default is {field.default!r}, expected ''"
        )
        hints = typing.get_type_hints(main_module.DatasetConfig)
        assert hints["labelling_session"] is str, (
            f"{REQ_FIELD}: DatasetConfig.labelling_session is annotated {hints['labelling_session']!r}, expected str"
        )


# ---------------------------------------------------------------- load_from_yaml


@pytest.mark.unit
class TestLoadFromYamlLabellingSession:
    """REQ-WHOLEBRAIN-BDRABCZUK-E0C750353087 / -B6C5701E5530: loading dataset.labelling_session."""

    def test_labelling_session_key_is_loaded(self, main_module, tmp_path):
        """REQ-WHOLEBRAIN-BDRABCZUK-E0C750353087: dataset.labelling_session in the YAML is the loaded value."""
        config_file = tmp_path / "config.yaml"
        _write_yaml(config_file, {"dataset": {"name": "example_dataset", "labelling_session": CONFIGURED_SESSION}})

        dataset = _load_dataset(main_module, config_file, REQ_LOAD)
        loaded = getattr(dataset, "labelling_session", "<no attribute>")

        assert loaded == CONFIGURED_SESSION, (
            f"{REQ_LOAD}: load_from_yaml returned dataset.labelling_session={loaded!r}, expected {CONFIGURED_SESSION!r}"
        )

    def test_missing_labelling_session_key_loads_empty_string(self, main_module, tmp_path):
        """REQ-WHOLEBRAIN-BDRABCZUK-B6C5701E5530: a dataset mapping without labelling_session loads the default ''."""
        config_file = tmp_path / "config.yaml"
        _write_yaml(config_file, {"dataset": {"name": "example_dataset"}})

        dataset = _load_dataset(main_module, config_file, REQ_FIELD)
        loaded = getattr(dataset, "labelling_session", "<no attribute>")

        assert loaded == "", (
            f"{REQ_FIELD}: load_from_yaml without dataset.labelling_session returned {loaded!r}, expected ''"
        )


# ---------------------------------------------------------------- save_to_yaml


@pytest.mark.unit
class TestSaveToYamlLabellingSession:
    """REQ-WHOLEBRAIN-BDRABCZUK-3DBF0C9F422F: save_to_yaml writes dataset.labelling_session."""

    def test_labelling_session_is_written_under_dataset(self, main_module, tmp_path):
        """A configured dataset.labelling_session is written as the dataset mapping's labelling_session key."""
        config = main_module.PipelineConfig()
        config.dataset.labelling_session = CONFIGURED_SESSION
        output = tmp_path / "saved.yaml"

        main_module.ConfigLoader.save_to_yaml(config, str(output))
        with open(output) as f:
            written = yaml.safe_load(f)

        assert written["dataset"].get("labelling_session", "<no key>") == CONFIGURED_SESSION, (
            f"{REQ_SAVE}: saved dataset mapping has labelling_session="
            f"{written['dataset'].get('labelling_session', '<no key>')!r}, expected {CONFIGURED_SESSION!r}"
        )

    def test_default_config_writes_empty_labelling_session(self, main_module, tmp_path):
        """A default PipelineConfig (what --generate-config saves) writes "labelling_session: ''" under dataset."""
        output = tmp_path / "saved.yaml"

        main_module.ConfigLoader.save_to_yaml(main_module.PipelineConfig(), str(output))
        with open(output) as f:
            written = yaml.safe_load(f)

        assert written["dataset"].get("labelling_session", "<no key>") == "", (
            f"{REQ_SAVE}: saved default dataset mapping has labelling_session="
            f"{written['dataset'].get('labelling_session', '<no key>')!r}, expected ''"
        )


# ---------------------------------------------------------------- RunCorticalTilesStage


def _tiles_config(main_module):
    """PipelineConfig with the paths RunCorticalTilesStage reads set to distinct values."""
    config = main_module.PipelineConfig()
    config.dataset.morphologist_graphs = "/graphs"  # noqa: V101
    config.dataset.cortical_tiles_output = "/tiles"  # noqa: V101
    return config


def _stage_argv(main_module, config):
    """Run RunCorticalTilesStage.execute() with RunCorticalTiles mocked; return the argv it built."""
    script_cls = MagicMock()
    script_cls.return_value.run.return_value = 0
    with patch.object(main_module, "RunCorticalTiles", script_cls):
        stage = main_module.RunCorticalTilesStage(
            "run_cortical_tiles", config, logging.getLogger("test_main_labelling_session_config")
        )
        result = stage.execute()
    assert result.success, result.message
    script_cls.return_value.parse_args.assert_called_once()
    return [str(token) for token in script_cls.return_value.parse_args.call_args[0][0]]


def _labelling_session_tokens(argv):
    """Values given to --labelling_session in argv, in either the '--opt=S' or the '--opt S' form."""
    values = []
    for i, token in enumerate(argv):
        if token == LABELLING_SESSION_OPTION:
            values.append(argv[i + 1] if i + 1 < len(argv) else None)
        elif token.startswith(LABELLING_SESSION_OPTION + "="):
            values.append(token.split("=", 1)[1])
    return values


def _parse_with_real_parser(argv, req_id):
    try:
        return RunCorticalTiles().parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"{req_id}: RunCorticalTiles parser rejected the stage argv {argv!r} (SystemExit {exc.code})")


@pytest.mark.unit
class TestRunCorticalTilesStageLabellingSessionArgv:
    """REQ-WHOLEBRAIN-BDRABCZUK-D657457C56CF / -96FF8A402A81: dataset.labelling_session reaches run_cortical_tiles."""

    def test_non_empty_labelling_session_passed_as_option(self, main_module):
        """REQ-WHOLEBRAIN-BDRABCZUK-D657457C56CF: labelling_session '0_auto' -> --labelling_session 0_auto."""
        config = _tiles_config(main_module)
        config.dataset.labelling_session = CONFIGURED_SESSION  # noqa: V101

        argv = _stage_argv(main_module, config)

        values = _labelling_session_tokens(argv)
        assert values == [CONFIGURED_SESSION], (
            f"{REQ_FORWARD}: stage argv passes --labelling_session values {values!r}, "
            f"expected [{CONFIGURED_SESSION!r}]; argv was {argv!r}"
        )
        namespace = _parse_with_real_parser(argv, REQ_FORWARD)
        assert namespace.labelling_session == CONFIGURED_SESSION, (
            f"{REQ_FORWARD}: run_cortical_tiles parses labelling_session={namespace.labelling_session!r}, "
            f"expected {CONFIGURED_SESSION!r}; argv was {argv!r}"
        )
        assert namespace.labelling_session_explicit is True, (
            f"{REQ_FORWARD}: run_cortical_tiles does not see --labelling_session as given; argv was {argv!r}"
        )

    def test_empty_labelling_session_passes_no_option(self, main_module):
        """REQ-WHOLEBRAIN-BDRABCZUK-96FF8A402A81: default ('') labelling_session -> no --labelling_session option.

        The '' comes from the DatasetConfig default (REQ-WHOLEBRAIN-BDRABCZUK-B6C5701E5530), not from an
        attribute set by the test, so the check is meaningful only once that setting exists.
        """
        config = _tiles_config(main_module)
        assert getattr(config.dataset, "labelling_session", "<no attribute>") == "", (
            f"{REQ_NO_FORWARD}: the default dataset config has no labelling_session setting of '' "
            f"(got {getattr(config.dataset, 'labelling_session', '<no attribute>')!r})"
        )

        argv = _stage_argv(main_module, config)

        offending = [t for t in argv if t == LABELLING_SESSION_OPTION or t.startswith(LABELLING_SESSION_OPTION + "=")]
        assert not offending, (
            f"{REQ_NO_FORWARD}: stage argv passes {offending!r} with labelling_session ''; argv was {argv!r}"
        )
        namespace = _parse_with_real_parser(argv, REQ_NO_FORWARD)
        assert namespace.labelling_session_explicit is False, (
            f"{REQ_NO_FORWARD}: run_cortical_tiles sees an explicit labelling session (auto-detect disabled); "
            f"argv was {argv!r}"
        )
