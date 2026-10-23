#!/usr/bin/env python3
"""
Tests for the dataset.models_cache setting of the main.py pipeline config
(models_cache epic, task 4/6: the generate_embeddings --models-cache override
settable from the main.py YAML pipeline config).

- REQ-DATASETS-BDRABCZUK-C41A0F37ADD7: DatasetConfig declares a models_cache
  field of type Optional[str] whose default value is None.
- REQ-DATASETS-BDRABCZUK-F6B8406A7A12: ConfigLoader.load_from_yaml reads the
  dataset mapping's models_cache key.
- REQ-DATASETS-BDRABCZUK-95923A49EDCB: ConfigLoader.load_from_yaml yields None
  when the dataset mapping has no models_cache key.
- REQ-DATASETS-BDRABCZUK-9AC1E998B868: ConfigLoader.save_to_yaml writes
  dataset.models_cache as the models_cache key of the dataset mapping (so
  --generate-config, which saves a default PipelineConfig, emits
  ``models_cache: null``).
- REQ-DATASETS-BDRABCZUK-5B12C31094F2: a models_cache path string P reaches
  generate_embeddings unchanged as --models-cache P.
- REQ-DATASETS-BDRABCZUK-770A696B10A0: a None models_cache passes no
  --models-cache option to generate_embeddings.

Relative paths (decision recorded on the requirements): main.py neither
rewrites the YAML value at load time nor at forward time, so
generate_embeddings resolves a relative models_cache against the current
working directory of the main.py process, exactly as a --models-cache given
on the command line from that directory (REQ-DATASETS-BDRABCZUK-BDE1ED8B4720).

The stage tests replace the GenerateEmbeddings class referenced by main.py with
a MagicMock to capture the argv GenerateEmbeddingsStage builds without running
anything. That argv is then parsed by the real GenerateEmbeddings parser, so
the asserted value is the one generate_embeddings would receive.

YAML files are written under tmp_path with yaml.safe_dump and read with
yaml.safe_load. main.py lives at the repo root (not inside the installed
package), so it is loaded by file path under a private module name.
"""

import dataclasses
import importlib.util
import logging
import os
import typing
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from champollion_pipeline.generate_embeddings import GenerateEmbeddings

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"

MODELS_CACHE_OPTION = "--models-cache"
ABSOLUTE_MODELS_CACHE = "/scratch/user/champollion_models_cache"
RELATIVE_MODELS_CACHE = os.path.join("relative", "models_cache")

REQ_FIELD = "REQ-DATASETS-BDRABCZUK-C41A0F37ADD7"
REQ_LOAD = "REQ-DATASETS-BDRABCZUK-F6B8406A7A12"
REQ_LOAD_MISSING = "REQ-DATASETS-BDRABCZUK-95923A49EDCB"
REQ_SAVE = "REQ-DATASETS-BDRABCZUK-9AC1E998B868"
REQ_FORWARD = "REQ-DATASETS-BDRABCZUK-5B12C31094F2"
REQ_NO_FORWARD = "REQ-DATASETS-BDRABCZUK-770A696B10A0"


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_models_cache_config_under_test", MAIN_PY)
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


def _models_cache_field(main_module, req_id):
    fields = {f.name: f for f in dataclasses.fields(main_module.DatasetConfig)}
    assert "models_cache" in fields, (
        f"{req_id}: DatasetConfig declares no models_cache field; fields are {sorted(fields)}"
    )
    return fields["models_cache"]


def _saved_dataset_mapping(main_module, config, tmp_path):
    output = tmp_path / "saved.yaml"
    main_module.ConfigLoader.save_to_yaml(config, str(output))
    with open(output) as f:
        return yaml.safe_load(f)["dataset"]


# ---------------------------------------------------------------- DatasetConfig


@pytest.mark.unit
class TestDatasetConfigModelsCacheDefault:
    """REQ-DATASETS-BDRABCZUK-C41A0F37ADD7: DatasetConfig.models_cache is Optional[str], default None."""

    def test_models_cache_field_defaults_to_none(self, main_module):
        """The DatasetConfig dataclass declares models_cache: Optional[str] = None."""
        field = _models_cache_field(main_module, REQ_FIELD)

        assert field.default is None, (
            f"{REQ_FIELD}: DatasetConfig.models_cache default is {field.default!r}, expected None"
        )
        hints = typing.get_type_hints(main_module.DatasetConfig)
        assert hints["models_cache"] == str | None, (
            f"{REQ_FIELD}: DatasetConfig.models_cache is annotated {hints['models_cache']!r}, expected Optional[str]"
        )


# ---------------------------------------------------------------- load_from_yaml


@pytest.mark.unit
class TestLoadFromYamlModelsCache:
    """REQ-DATASETS-BDRABCZUK-F6B8406A7A12 / -95923A49EDCB: loading dataset.models_cache."""

    def test_models_cache_key_is_loaded(self, main_module, tmp_path):
        """REQ-DATASETS-BDRABCZUK-F6B8406A7A12: dataset.models_cache in the YAML is the loaded value."""
        config_file = tmp_path / "config.yaml"
        _write_yaml(config_file, {"dataset": {"name": "example_dataset", "models_cache": ABSOLUTE_MODELS_CACHE}})

        dataset = _load_dataset(main_module, config_file, REQ_LOAD)

        loaded = getattr(dataset, "models_cache", "<no attribute>")
        assert loaded == ABSOLUTE_MODELS_CACHE, (
            f"{REQ_LOAD}: load_from_yaml returned dataset.models_cache={loaded!r}, expected {ABSOLUTE_MODELS_CACHE!r}"
        )

    def test_relative_models_cache_key_is_loaded_verbatim(self, main_module, tmp_path, monkeypatch):
        """REQ-DATASETS-BDRABCZUK-F6B8406A7A12: a relative models_cache loads as written, not made absolute.

        Resolution against the cwd is generate_embeddings' job (REQ-DATASETS-BDRABCZUK-BDE1ED8B4720);
        resolving it earlier (e.g. against the YAML file's directory) would break the CLI equivalence.
        """
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        _write_yaml(config_file, {"dataset": {"name": "example_dataset", "models_cache": RELATIVE_MODELS_CACHE}})
        monkeypatch.chdir(tmp_path)

        dataset = _load_dataset(main_module, config_file, REQ_LOAD)

        loaded = getattr(dataset, "models_cache", "<no attribute>")
        assert loaded == RELATIVE_MODELS_CACHE, (
            f"{REQ_LOAD}: load_from_yaml returned dataset.models_cache={loaded!r}, "
            f"expected the YAML value {RELATIVE_MODELS_CACHE!r} unchanged"
        )

    def test_missing_models_cache_key_loads_none(self, main_module, tmp_path):
        """REQ-DATASETS-BDRABCZUK-95923A49EDCB: a dataset mapping without models_cache loads None."""
        config_file = tmp_path / "config.yaml"
        _write_yaml(config_file, {"dataset": {"name": "example_dataset"}})

        dataset = _load_dataset(main_module, config_file, REQ_LOAD_MISSING)

        assert hasattr(dataset, "models_cache"), (
            f"{REQ_LOAD_MISSING}: load_from_yaml without dataset.models_cache returned a dataset with no "
            "models_cache attribute, expected models_cache=None"
        )
        assert dataset.models_cache is None, (
            f"{REQ_LOAD_MISSING}: load_from_yaml without dataset.models_cache returned "
            f"{dataset.models_cache!r}, expected None"
        )


# ---------------------------------------------------------------- save_to_yaml


@pytest.mark.unit
class TestSaveToYamlModelsCache:
    """REQ-DATASETS-BDRABCZUK-9AC1E998B868: save_to_yaml writes dataset.models_cache."""

    def test_models_cache_is_written_under_dataset(self, main_module, tmp_path):
        """A configured dataset.models_cache is written as the dataset mapping's models_cache key."""
        config = main_module.PipelineConfig()
        config.dataset.models_cache = ABSOLUTE_MODELS_CACHE  # noqa: V101

        written = _saved_dataset_mapping(main_module, config, tmp_path)

        assert written.get("models_cache", "<no key>") == ABSOLUTE_MODELS_CACHE, (
            f"{REQ_SAVE}: saved dataset mapping has models_cache="
            f"{written.get('models_cache', '<no key>')!r}, expected {ABSOLUTE_MODELS_CACHE!r}"
        )

    def test_default_config_writes_null_models_cache(self, main_module, tmp_path):
        """A default PipelineConfig (what --generate-config saves) writes 'models_cache: null' under dataset."""
        written = _saved_dataset_mapping(main_module, main_module.PipelineConfig(), tmp_path)

        assert "models_cache" in written, (
            f"{REQ_SAVE}: saved default dataset mapping has no models_cache key; keys are {sorted(written)}"
        )
        assert written["models_cache"] is None, (
            f"{REQ_SAVE}: saved default dataset mapping has models_cache={written['models_cache']!r}, expected null"
        )


# ---------------------------------------------------------------- GenerateEmbeddingsStage


def _embeddings_config(main_module):
    """PipelineConfig with the local-models inputs GenerateEmbeddingsStage reads set to distinct values."""
    config = main_module.PipelineConfig()
    config.models_path = "/models"  # noqa: V101
    config.dataset.datasets_root = "/datasets/example"  # noqa: V101
    return config


def _stage_argv(main_module, config):
    """Run GenerateEmbeddingsStage.execute() with GenerateEmbeddings mocked; return the argv it built."""
    script_cls = MagicMock()
    script_cls.return_value.run.return_value = 0
    with patch.object(main_module, "GenerateEmbeddings", script_cls):
        stage = main_module.GenerateEmbeddingsStage(
            "generate_embeddings", config, logging.getLogger("test_main_models_cache_config")
        )
        result = stage.execute()
    assert result.success, result.message
    script_cls.return_value.parse_args.assert_called_once()
    return [str(token) for token in script_cls.return_value.parse_args.call_args[0][0]]


def _models_cache_tokens(argv):
    """Values given to --models-cache in argv, in either the '--models-cache=P' or the '--models-cache P' form."""
    values = []
    for i, token in enumerate(argv):
        if token == MODELS_CACHE_OPTION:
            values.append(argv[i + 1] if i + 1 < len(argv) else None)
        elif token.startswith(MODELS_CACHE_OPTION + "="):
            values.append(token.split("=", 1)[1])
    return values


def _parse_with_generate_embeddings(argv):
    """Parse the stage argv with the real generate_embeddings parser; a rejected argv is a test failure."""
    script = GenerateEmbeddings()
    try:
        script.parse_args(argv)
    except SystemExit as exc:
        pytest.fail(
            f"{REQ_FORWARD}: GenerateEmbeddings parser rejected the stage argv {argv!r} (SystemExit {exc.code})"
        )
    return script


def _assert_forwarded(argv, models_cache):
    """argv passes --models-cache with the value models_cache, unchanged, to generate_embeddings."""
    values = _models_cache_tokens(argv)
    assert values == [models_cache], (
        f"{REQ_FORWARD}: stage argv passes --models-cache values {values!r}, expected [{models_cache!r}]; "
        f"argv was {argv!r}"
    )
    script = _parse_with_generate_embeddings(argv)
    assert script.args.models_cache == models_cache, (
        f"{REQ_FORWARD}: generate_embeddings parses models_cache={script.args.models_cache!r}, "
        f"expected {models_cache!r}; argv was {argv!r}"
    )
    return script


@pytest.mark.unit
class TestGenerateEmbeddingsStageModelsCacheArgv:
    """REQ-DATASETS-BDRABCZUK-5B12C31094F2 / -770A696B10A0: dataset.models_cache reaches generate_embeddings."""

    def test_absolute_models_cache_passed_as_models_cache_option(self, main_module):
        """REQ-DATASETS-BDRABCZUK-5B12C31094F2: models_cache P -> generate_embeddings receives --models-cache P."""
        config = _embeddings_config(main_module)
        config.dataset.models_cache = ABSOLUTE_MODELS_CACHE  # noqa: V101

        _assert_forwarded(_stage_argv(main_module, config), ABSOLUTE_MODELS_CACHE)

    def test_relative_models_cache_passed_unchanged_and_resolves_against_cwd(self, main_module, tmp_path, monkeypatch):
        """REQ-DATASETS-BDRABCZUK-5B12C31094F2: a relative models_cache is forwarded as written.

        generate_embeddings then resolves it against the cwd of the main.py process, the same
        directory a --models-cache given on the command line from there would resolve against.
        """
        monkeypatch.chdir(tmp_path)
        config = _embeddings_config(main_module)
        config.dataset.models_cache = RELATIVE_MODELS_CACHE  # noqa: V101

        script = _assert_forwarded(_stage_argv(main_module, config), RELATIVE_MODELS_CACHE)

        extract_dir = script._compute_extract_dir()
        expected = os.path.join(os.getcwd(), RELATIVE_MODELS_CACHE)
        assert extract_dir == expected, (
            f"{REQ_FORWARD}: relative dataset.models_cache resolves to {extract_dir!r} in generate_embeddings, "
            f"expected {expected!r} (the main.py cwd)"
        )

    def test_none_models_cache_passes_no_models_cache_option(self, main_module):
        """REQ-DATASETS-BDRABCZUK-770A696B10A0: default (None) dataset.models_cache -> no --models-cache in the argv.

        The None comes from the DatasetConfig default (REQ-DATASETS-BDRABCZUK-C41A0F37ADD7), not from an
        attribute set by the test, so the check is meaningful only once that setting exists.
        """
        config = _embeddings_config(main_module)
        assert hasattr(config.dataset, "models_cache") and config.dataset.models_cache is None, (
            f"{REQ_NO_FORWARD}: the default dataset config has no models_cache setting of None "
            f"(got {getattr(config.dataset, 'models_cache', '<no attribute>')!r})"
        )

        argv = _stage_argv(main_module, config)

        offending = [t for t in argv if t == MODELS_CACHE_OPTION or t.startswith(MODELS_CACHE_OPTION + "=")]
        assert not offending, (
            f"{REQ_NO_FORWARD}: stage argv passes {offending!r} with models_cache None; argv was {argv!r}"
        )
