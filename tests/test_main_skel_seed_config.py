#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the dataset.skel_seed setting of the main.py pipeline config
(skeleton seed epic, task 8/9: skel_seed settable in the main.py YAML
pipeline config).

- REQ-SKELSEED-BDRABCZUK-136B4964314B: DatasetConfig declares a skel_seed
  field of type Optional[int] whose default value is None.
- REQ-SKELSEED-BDRABCZUK-6140C5853158: ConfigLoader.load_from_yaml reads the
  dataset mapping's skel_seed key.
- REQ-SKELSEED-BDRABCZUK-B1AB6FB2B3C2: ConfigLoader.load_from_yaml yields None
  when the dataset mapping has no skel_seed key.
- REQ-SKELSEED-BDRABCZUK-8F68E80A401F: ConfigLoader.save_to_yaml writes
  dataset.skel_seed as the skel_seed key of the dataset mapping (so
  --generate-config, which saves a default PipelineConfig, emits
  ``skel_seed: null``).
- REQ-SKELSEED-BDRABCZUK-A2EC0E672F57: an integer skel_seed N (0 included)
  reaches run_cortical_tiles as --skel-seed N.
- REQ-SKELSEED-BDRABCZUK-BF867A0E5A14: a None skel_seed passes no --skel-seed
  option to run_cortical_tiles.

The stage tests replace the RunCorticalTiles class referenced by main.py with a
MagicMock to capture the argv RunCorticalTilesStage builds without running
anything. When the real RunCorticalTiles parser already declares --skel-seed
(REQ-SKELSEED-BDRABCZUK-16447BAF3125), that argv is also parsed with it, so the
asserted value is the one run_cortical_tiles would receive; otherwise the raw
argv tokens are asserted (``--skel-seed=N`` or ``--skel-seed N``).

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

SKEL_SEED_OPTION = "--skel-seed"
CONFIGURED_SKEL_SEED = 7

REQ_FIELD = "REQ-SKELSEED-BDRABCZUK-136B4964314B"
REQ_LOAD = "REQ-SKELSEED-BDRABCZUK-6140C5853158"
REQ_LOAD_MISSING = "REQ-SKELSEED-BDRABCZUK-B1AB6FB2B3C2"
REQ_SAVE = "REQ-SKELSEED-BDRABCZUK-8F68E80A401F"
REQ_FORWARD = "REQ-SKELSEED-BDRABCZUK-A2EC0E672F57"
REQ_NO_FORWARD = "REQ-SKELSEED-BDRABCZUK-BF867A0E5A14"


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_skel_seed_config_under_test", MAIN_PY)
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


def _skel_seed_field(main_module, req_id):
    fields = {f.name: f for f in dataclasses.fields(main_module.DatasetConfig)}
    assert "skel_seed" in fields, f"{req_id}: DatasetConfig declares no skel_seed field; fields are {sorted(fields)}"
    return fields["skel_seed"]


# ---------------------------------------------------------------- DatasetConfig


@pytest.mark.unit
class TestDatasetConfigSkelSeedDefault:
    """REQ-SKELSEED-BDRABCZUK-136B4964314B: DatasetConfig.skel_seed is Optional[int], default None."""

    def test_skel_seed_field_defaults_to_none(self, main_module):
        """The DatasetConfig dataclass declares skel_seed: Optional[int] = None."""
        field = _skel_seed_field(main_module, REQ_FIELD)

        assert field.default is None, (
            f"{REQ_FIELD}: DatasetConfig.skel_seed default is {field.default!r}, expected None"
        )
        hints = typing.get_type_hints(main_module.DatasetConfig)
        assert hints["skel_seed"] == typing.Optional[int], (
            f"{REQ_FIELD}: DatasetConfig.skel_seed is annotated {hints['skel_seed']!r}, expected Optional[int]"
        )


# ---------------------------------------------------------------- load_from_yaml


@pytest.mark.unit
class TestLoadFromYamlSkelSeed:
    """REQ-SKELSEED-BDRABCZUK-6140C5853158 / -B1AB6FB2B3C2: loading dataset.skel_seed."""

    def test_skel_seed_key_is_loaded(self, main_module, tmp_path):
        """REQ-SKELSEED-BDRABCZUK-6140C5853158: dataset.skel_seed in the YAML is the loaded value."""
        config_file = tmp_path / "config.yaml"
        _write_yaml(config_file, {"dataset": {"name": "example_dataset", "skel_seed": CONFIGURED_SKEL_SEED}})

        dataset = _load_dataset(main_module, config_file, REQ_LOAD)

        assert getattr(dataset, "skel_seed", "<no attribute>") == CONFIGURED_SKEL_SEED, (
            f"{REQ_LOAD}: load_from_yaml returned dataset.skel_seed="
            f"{getattr(dataset, 'skel_seed', '<no attribute>')!r}, expected {CONFIGURED_SKEL_SEED!r}"
        )

    def test_zero_skel_seed_key_is_loaded(self, main_module, tmp_path):
        """REQ-SKELSEED-BDRABCZUK-6140C5853158: skel_seed 0 is a valid seed and loads as 0, not None."""
        config_file = tmp_path / "config.yaml"
        _write_yaml(config_file, {"dataset": {"name": "example_dataset", "skel_seed": 0}})

        dataset = _load_dataset(main_module, config_file, REQ_LOAD)
        loaded = getattr(dataset, "skel_seed", "<no attribute>")

        assert loaded == 0 and loaded is not None and not isinstance(loaded, bool), (
            f"{REQ_LOAD}: load_from_yaml returned dataset.skel_seed={loaded!r}, expected 0"
        )

    def test_missing_skel_seed_key_loads_none(self, main_module, tmp_path):
        """REQ-SKELSEED-BDRABCZUK-B1AB6FB2B3C2: a dataset mapping without skel_seed loads None."""
        config_file = tmp_path / "config.yaml"
        _write_yaml(config_file, {"dataset": {"name": "example_dataset"}})

        dataset = _load_dataset(main_module, config_file, REQ_LOAD_MISSING)

        assert hasattr(dataset, "skel_seed"), (
            f"{REQ_LOAD_MISSING}: load_from_yaml without dataset.skel_seed returned a dataset with no "
            "skel_seed attribute, expected skel_seed=None"
        )
        assert dataset.skel_seed is None, (
            f"{REQ_LOAD_MISSING}: load_from_yaml without dataset.skel_seed returned "
            f"{dataset.skel_seed!r}, expected None"
        )


# ---------------------------------------------------------------- save_to_yaml


@pytest.mark.unit
class TestSaveToYamlSkelSeed:
    """REQ-SKELSEED-BDRABCZUK-8F68E80A401F: save_to_yaml writes dataset.skel_seed."""

    def test_skel_seed_is_written_under_dataset(self, main_module, tmp_path):
        """A configured dataset.skel_seed is written as the dataset mapping's skel_seed key."""
        config = main_module.PipelineConfig()
        config.dataset.skel_seed = CONFIGURED_SKEL_SEED
        output = tmp_path / "saved.yaml"

        main_module.ConfigLoader.save_to_yaml(config, str(output))
        with open(output, "r") as f:
            written = yaml.safe_load(f)

        assert written["dataset"].get("skel_seed", "<no key>") == CONFIGURED_SKEL_SEED, (
            f"{REQ_SAVE}: saved dataset mapping has skel_seed="
            f"{written['dataset'].get('skel_seed', '<no key>')!r}, expected {CONFIGURED_SKEL_SEED!r}"
        )

    def test_default_config_writes_null_skel_seed(self, main_module, tmp_path):
        """A default PipelineConfig (what --generate-config saves) writes 'skel_seed: null' under dataset."""
        output = tmp_path / "saved.yaml"

        main_module.ConfigLoader.save_to_yaml(main_module.PipelineConfig(), str(output))
        with open(output, "r") as f:
            written = yaml.safe_load(f)

        assert "skel_seed" in written["dataset"], (
            f"{REQ_SAVE}: saved default dataset mapping has no skel_seed key; keys are {sorted(written['dataset'])}"
        )
        assert written["dataset"]["skel_seed"] is None, (
            f"{REQ_SAVE}: saved default dataset mapping has skel_seed={written['dataset']['skel_seed']!r}, "
            "expected null"
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
            "run_cortical_tiles", config, logging.getLogger("test_main_skel_seed_config")
        )
        result = stage.execute()
    assert result.success, result.message
    script_cls.return_value.parse_args.assert_called_once()
    return [str(token) for token in script_cls.return_value.parse_args.call_args[0][0]]


def _skel_seed_tokens(argv):
    """Values given to --skel-seed in argv, in either the '--skel-seed=N' or the '--skel-seed N' form."""
    values = []
    for i, token in enumerate(argv):
        if token == SKEL_SEED_OPTION:
            values.append(argv[i + 1] if i + 1 < len(argv) else None)
        elif token.startswith(SKEL_SEED_OPTION + "="):
            values.append(token.split("=", 1)[1])
    return values


def _parser_declares_skel_seed():
    return SKEL_SEED_OPTION in RunCorticalTiles().parser._option_string_actions


def _assert_forwarded(argv, seed):
    """argv passes --skel-seed with the value seed to run_cortical_tiles."""
    values = _skel_seed_tokens(argv)
    assert values == [str(seed)], (
        f"{REQ_FORWARD}: stage argv passes --skel-seed values {values!r}, expected [{str(seed)!r}]; argv was {argv!r}"
    )
    if _parser_declares_skel_seed():
        try:
            namespace = RunCorticalTiles().parse_args(argv)
        except SystemExit as exc:
            pytest.fail(
                f"{REQ_FORWARD}: RunCorticalTiles parser rejected the stage argv {argv!r} (SystemExit {exc.code})"
            )
        assert namespace.skel_seed == seed, (
            f"{REQ_FORWARD}: run_cortical_tiles parses skel_seed={namespace.skel_seed!r}, expected {seed!r}; "
            f"argv was {argv!r}"
        )


@pytest.mark.unit
class TestRunCorticalTilesStageSkelSeedArgv:
    """REQ-SKELSEED-BDRABCZUK-A2EC0E672F57 / -BF867A0E5A14: dataset.skel_seed reaches run_cortical_tiles."""

    def test_integer_skel_seed_passed_as_skel_seed_option(self, main_module):
        """REQ-SKELSEED-BDRABCZUK-A2EC0E672F57: dataset.skel_seed 7 -> run_cortical_tiles receives --skel-seed 7."""
        config = _tiles_config(main_module)
        config.dataset.skel_seed = CONFIGURED_SKEL_SEED  # noqa: V101

        _assert_forwarded(_stage_argv(main_module, config), CONFIGURED_SKEL_SEED)

    def test_zero_skel_seed_passed_as_skel_seed_option(self, main_module):
        """REQ-SKELSEED-BDRABCZUK-A2EC0E672F57: 0 is a valid seed and is forwarded (not dropped as falsy)."""
        config = _tiles_config(main_module)
        config.dataset.skel_seed = 0  # noqa: V101

        _assert_forwarded(_stage_argv(main_module, config), 0)

    def test_none_skel_seed_passes_no_skel_seed_option(self, main_module):
        """REQ-SKELSEED-BDRABCZUK-BF867A0E5A14: default (None) dataset.skel_seed -> no --skel-seed in the argv.

        The None comes from the DatasetConfig default (REQ-SKELSEED-BDRABCZUK-136B4964314B), not from an
        attribute set by the test, so the check is meaningful only once that setting exists.
        """
        config = _tiles_config(main_module)
        assert hasattr(config.dataset, "skel_seed") and config.dataset.skel_seed is None, (
            f"{REQ_NO_FORWARD}: the default dataset config has no skel_seed setting of None "
            f"(got {getattr(config.dataset, 'skel_seed', '<no attribute>')!r})"
        )

        argv = _stage_argv(main_module, config)

        offending = [t for t in argv if t == SKEL_SEED_OPTION or t.startswith(SKEL_SEED_OPTION + "=")]
        assert not offending, (
            f"{REQ_NO_FORWARD}: stage argv passes {offending!r} with skel_seed None; argv was {argv!r}"
        )
