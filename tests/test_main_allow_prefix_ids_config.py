#!/usr/bin/env python3
"""
Tests for the dataset.allow_prefix_ids setting of the main.py pipeline config
(TEMPORARY: removed together with the run_cortical_tiles prefix subject-ID
guard once DEF-CORTILES-BDRABCZUK-8A44DAD7F312 is fixed upstream in
cortical_tiles). The setting reaches run_cortical_tiles as --allow-prefix-ids,
following the --bids precedent; it is not forwarded by default.

- REQ-CORTILES-BDRABCZUK-A2E8D130693C: DatasetConfig declares an
  allow_prefix_ids field of type bool whose default value is False.
- REQ-CORTILES-BDRABCZUK-2E1666FC340B: ConfigLoader.load_from_yaml reads the
  dataset mapping's allow_prefix_ids key.
- REQ-CORTILES-BDRABCZUK-A20F9F75095C: ConfigLoader.save_to_yaml writes
  dataset.allow_prefix_ids as the allow_prefix_ids key of the dataset mapping.
- REQ-CORTILES-BDRABCZUK-EFAEAA66C3A1: allow_prefix_ids True reaches
  run_cortical_tiles as --allow-prefix-ids.
- REQ-CORTILES-BDRABCZUK-A560F75F4196: allow_prefix_ids False passes no
  --allow-prefix-ids option to run_cortical_tiles.
- REQ-CORTILES-BDRABCZUK-C6B3C49D25E4: main.py --allow-prefix-ids sets
  dataset.allow_prefix_ids True through _apply_cli_overrides.

The stage tests replace the RunCorticalTiles class referenced by main.py with a
MagicMock to capture the argv RunCorticalTilesStage builds without running
anything; that argv is then parsed with the real RunCorticalTiles parser, so
the asserted allow_prefix_ids value is the one run_cortical_tiles would receive.

YAML files are written under tmp_path with yaml.safe_dump and read with
yaml.safe_load. main.py lives at the repo root (not inside the installed
package), so it is loaded by file path under a private module name.
"""

import dataclasses
import importlib.util
import logging
import sys
import typing
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from champollion_pipeline.run_cortical_tiles import RunCorticalTiles

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"

ALLOW_PREFIX_IDS_OPTION = "--allow-prefix-ids"

REQ_FIELD = "REQ-CORTILES-BDRABCZUK-A2E8D130693C"
REQ_LOAD = "REQ-CORTILES-BDRABCZUK-2E1666FC340B"
REQ_SAVE = "REQ-CORTILES-BDRABCZUK-A20F9F75095C"
REQ_FORWARD = "REQ-CORTILES-BDRABCZUK-EFAEAA66C3A1"
REQ_NO_FORWARD = "REQ-CORTILES-BDRABCZUK-A560F75F4196"
REQ_CLI = "REQ-CORTILES-BDRABCZUK-C6B3C49D25E4"


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_allow_prefix_ids_config_under_test", MAIN_PY)
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
class TestDatasetConfigAllowPrefixIdsDefault:
    """REQ-CORTILES-BDRABCZUK-A2E8D130693C: DatasetConfig.allow_prefix_ids is bool, default False."""

    def test_allow_prefix_ids_field_defaults_to_false(self, main_module):
        """The DatasetConfig dataclass declares allow_prefix_ids: bool = False."""
        fields = {f.name: f for f in dataclasses.fields(main_module.DatasetConfig)}
        assert "allow_prefix_ids" in fields, (
            f"{REQ_FIELD}: DatasetConfig declares no allow_prefix_ids field; fields are {sorted(fields)}"
        )
        field = fields["allow_prefix_ids"]

        assert field.default is False, (
            f"{REQ_FIELD}: DatasetConfig.allow_prefix_ids default is {field.default!r}, expected False"
        )
        hints = typing.get_type_hints(main_module.DatasetConfig)
        assert hints["allow_prefix_ids"] is bool, (
            f"{REQ_FIELD}: DatasetConfig.allow_prefix_ids is annotated {hints['allow_prefix_ids']!r}, expected bool"
        )


# ---------------------------------------------------------------- load_from_yaml


@pytest.mark.unit
class TestLoadFromYamlAllowPrefixIds:
    """REQ-CORTILES-BDRABCZUK-2E1666FC340B / -A2E8D130693C: loading dataset.allow_prefix_ids."""

    def test_allow_prefix_ids_key_is_loaded(self, main_module, tmp_path):
        """REQ-CORTILES-BDRABCZUK-2E1666FC340B: dataset.allow_prefix_ids true in the YAML is the loaded value."""
        config_file = tmp_path / "config.yaml"
        _write_yaml(config_file, {"dataset": {"name": "example_dataset", "allow_prefix_ids": True}})

        dataset = _load_dataset(main_module, config_file, REQ_LOAD)
        loaded = getattr(dataset, "allow_prefix_ids", "<no attribute>")

        assert loaded is True, f"{REQ_LOAD}: load_from_yaml returned dataset.allow_prefix_ids={loaded!r}, expected True"

    def test_missing_allow_prefix_ids_key_loads_false(self, main_module, tmp_path):
        """REQ-CORTILES-BDRABCZUK-A2E8D130693C: a dataset mapping without allow_prefix_ids loads the default False."""
        config_file = tmp_path / "config.yaml"
        _write_yaml(config_file, {"dataset": {"name": "example_dataset"}})

        dataset = _load_dataset(main_module, config_file, REQ_FIELD)
        loaded = getattr(dataset, "allow_prefix_ids", "<no attribute>")

        assert loaded is False, (
            f"{REQ_FIELD}: load_from_yaml without dataset.allow_prefix_ids returned {loaded!r}, expected False"
        )


# ---------------------------------------------------------------- save_to_yaml


def _saved_dataset(main_module, config, path):
    main_module.ConfigLoader.save_to_yaml(config, str(path))
    with open(path) as f:
        return yaml.safe_load(f)["dataset"]


@pytest.mark.unit
class TestSaveToYamlAllowPrefixIds:
    """REQ-CORTILES-BDRABCZUK-A20F9F75095C: save_to_yaml writes dataset.allow_prefix_ids."""

    def test_allow_prefix_ids_is_written_under_dataset(self, main_module, tmp_path):
        """A dataset.allow_prefix_ids of True is written as the dataset mapping's allow_prefix_ids key."""
        config = main_module.PipelineConfig()
        config.dataset.allow_prefix_ids = True  # noqa: V101

        written = _saved_dataset(main_module, config, tmp_path / "saved.yaml")

        assert written.get("allow_prefix_ids", "<no key>") is True, (
            f"{REQ_SAVE}: saved dataset mapping has allow_prefix_ids="
            f"{written.get('allow_prefix_ids', '<no key>')!r}, expected True"
        )

    def test_default_config_writes_false_allow_prefix_ids(self, main_module, tmp_path):
        """A default PipelineConfig (what --generate-config saves) writes 'allow_prefix_ids: false' under dataset."""
        written = _saved_dataset(main_module, main_module.PipelineConfig(), tmp_path / "saved.yaml")

        assert written.get("allow_prefix_ids", "<no key>") is False, (
            f"{REQ_SAVE}: saved default dataset mapping has allow_prefix_ids="
            f"{written.get('allow_prefix_ids', '<no key>')!r}, expected False"
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
            "run_cortical_tiles", config, logging.getLogger("test_main_allow_prefix_ids_config")
        )
        result = stage.execute()
    assert result.success, result.message
    script_cls.return_value.parse_args.assert_called_once()
    return [str(token) for token in script_cls.return_value.parse_args.call_args[0][0]]


def _parse_with_real_parser(argv, req_id):
    try:
        return RunCorticalTiles().parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"{req_id}: RunCorticalTiles parser rejected the stage argv {argv!r} (SystemExit {exc.code})")


def _allow_prefix_ids_tokens(argv):
    return [t for t in argv if t == ALLOW_PREFIX_IDS_OPTION or t.startswith(ALLOW_PREFIX_IDS_OPTION + "=")]


@pytest.mark.unit
class TestRunCorticalTilesStageAllowPrefixIdsArgv:
    """REQ-CORTILES-BDRABCZUK-EFAEAA66C3A1 / -A560F75F4196: dataset.allow_prefix_ids reaches run_cortical_tiles."""

    def test_true_allow_prefix_ids_passes_flag(self, main_module):
        """REQ-CORTILES-BDRABCZUK-EFAEAA66C3A1: allow_prefix_ids True -> --allow-prefix-ids."""
        config = _tiles_config(main_module)
        config.dataset.allow_prefix_ids = True  # noqa: V101

        argv = _stage_argv(main_module, config)

        tokens = _allow_prefix_ids_tokens(argv)
        assert tokens == [ALLOW_PREFIX_IDS_OPTION], (
            f"{REQ_FORWARD}: stage argv passes {tokens!r}, expected [{ALLOW_PREFIX_IDS_OPTION!r}]; argv was {argv!r}"
        )
        namespace = _parse_with_real_parser(argv, REQ_FORWARD)
        assert namespace.allow_prefix_ids is True, (
            f"{REQ_FORWARD}: run_cortical_tiles parses allow_prefix_ids={namespace.allow_prefix_ids!r}, "
            f"expected True; argv was {argv!r}"
        )

    def test_false_allow_prefix_ids_passes_no_flag(self, main_module):
        """REQ-CORTILES-BDRABCZUK-A560F75F4196: default (False) allow_prefix_ids -> no --allow-prefix-ids option.

        The False comes from the DatasetConfig default (REQ-CORTILES-BDRABCZUK-A2E8D130693C), not from an
        attribute set by the test, so the check is meaningful only once that setting exists.
        """
        config = _tiles_config(main_module)
        assert getattr(config.dataset, "allow_prefix_ids", "<no attribute>") is False, (
            f"{REQ_NO_FORWARD}: the default dataset config has no allow_prefix_ids setting of False "
            f"(got {getattr(config.dataset, 'allow_prefix_ids', '<no attribute>')!r})"
        )

        argv = _stage_argv(main_module, config)

        tokens = _allow_prefix_ids_tokens(argv)
        assert not tokens, (
            f"{REQ_NO_FORWARD}: stage argv passes {tokens!r} with allow_prefix_ids False; argv was {argv!r}"
        )
        namespace = _parse_with_real_parser(argv, REQ_NO_FORWARD)
        assert namespace.allow_prefix_ids is False, (
            f"{REQ_NO_FORWARD}: run_cortical_tiles parses allow_prefix_ids={namespace.allow_prefix_ids!r}, "
            f"expected False; argv was {argv!r}"
        )


# ---------------------------------------------------------------- main.py command line


def _config_after_cli(main_module, monkeypatch, cli_args, req_id):
    """Default config with parse_arguments() of main.py for *cli_args* applied by _apply_cli_overrides."""
    monkeypatch.setattr(sys, "argv", ["main.py", *cli_args])
    try:
        args = main_module.parse_arguments()
    except SystemExit as exc:
        pytest.fail(f"{req_id}: main.py parse_arguments() refused {cli_args!r} (SystemExit {exc.code})")
    config = main_module.create_default_config()
    main_module._apply_cli_overrides(config, args, from_default=True)
    return config


@pytest.mark.unit
class TestCliAllowPrefixIds:
    """REQ-CORTILES-BDRABCZUK-C6B3C49D25E4: main.py --allow-prefix-ids sets dataset.allow_prefix_ids."""

    def test_flag_sets_allow_prefix_ids(self, main_module, monkeypatch):
        """--allow-prefix-ids on the main.py command line gives dataset.allow_prefix_ids True."""
        config = _config_after_cli(main_module, monkeypatch, [ALLOW_PREFIX_IDS_OPTION], REQ_CLI)

        value = getattr(config.dataset, "allow_prefix_ids", "<no attribute>")
        assert value is True, (
            f"{REQ_CLI}: after --allow-prefix-ids, dataset.allow_prefix_ids is {value!r}, expected True"
        )

    def test_no_flag_keeps_default_false(self, main_module, monkeypatch):
        """Without --allow-prefix-ids, the default config keeps dataset.allow_prefix_ids False."""
        config = _config_after_cli(main_module, monkeypatch, [], REQ_FIELD)

        value = getattr(config.dataset, "allow_prefix_ids", "<no attribute>")
        assert value is False, (
            f"{REQ_FIELD}: without --allow-prefix-ids, dataset.allow_prefix_ids is {value!r}, expected False"
        )
