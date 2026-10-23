#!/usr/bin/env python3
"""
Tests that main.py's GenerateEmbeddingsStage no longer forwards the deprecated
``--masks-version`` option to generate_embeddings.py (epic MODELVER, main.py side).

- REQ-MODELVER-BDRABCZUK-03C21D7A4298: GenerateEmbeddingsStage in main.py shall
  pass no --masks-version argument to generate_embeddings.py.

The companion requirement REQ-EMBVER-BDRABCZUK-EA5EB289888B (hf_enabled ->
masks_version as ``--model-version``) is tested in test_main_embver_stage4_masks.py.

The GenerateEmbeddings class referenced by main.py is replaced by a MagicMock to
capture the argv the stage builds. The last test then parses that argv with the
*real* GenerateEmbeddings parser and checks that no FutureWarning naming
--masks-version is raised, which is what an orchestrated pipeline run sees.

The YAML/DatasetConfig key stays ``dataset.masks_version``; only the option name
on the generate_embeddings command line changes.
"""

import importlib.util
import logging
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"

NON_DEFAULT_MASKS_VERSION = "canonical_corrected_26_1"
DEPRECATED_OPTION = "--masks-version"


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_modelver_stage4_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def main_module():
    return _load_main_module()


def _config(main_module, hf_enabled):
    """PipelineConfig with a non-default masks_version and the requested model source."""
    config = main_module.PipelineConfig(models_path="/models", outputs_path="/outputs")
    dataset = config.dataset
    dataset.datasets_root = "/data/MYDATASET"  # noqa: V101
    dataset.masks_version = NON_DEFAULT_MASKS_VERSION  # noqa: V101
    dataset.hf_enabled = hf_enabled  # noqa: V101
    if hf_enabled:
        dataset.hf_repo_id = "neurospin/Champollion_V1"  # noqa: V101
    return config


def _stage_argv(main_module, config):
    """Run GenerateEmbeddingsStage.execute() with GenerateEmbeddings mocked; return the argv it built."""
    script_cls = MagicMock()
    script_cls.return_value.run.return_value = 0
    with patch.object(main_module, "GenerateEmbeddings", script_cls, create=True):
        stage = main_module.GenerateEmbeddingsStage(
            "GenerateEmbeddingsStage", config, logging.getLogger("test_main_modelver_stage4_model_version")
        )
        result = stage.execute()
    assert result.success, result.message
    script_cls.return_value.parse_args.assert_called_once()
    return [str(token) for token in script_cls.return_value.parse_args.call_args[0][0]]


def _deprecated_tokens(argv):
    """Tokens of argv that pass --masks-version, in either ``--opt value`` or ``--opt=value`` form."""
    return [token for token in argv if token == DEPRECATED_OPTION or token.startswith(f"{DEPRECATED_OPTION}=")]


@pytest.mark.unit
class TestGenerateEmbeddingsStageNoMasksVersion:
    """REQ-MODELVER-BDRABCZUK-03C21D7A4298: GenerateEmbeddingsStage passes no --masks-version."""

    def test_hf_enabled_argv_has_no_masks_version(self, main_module):
        """With hf_enabled true, the argv given to generate_embeddings carries no --masks-version option."""
        argv = _stage_argv(main_module, _config(main_module, hf_enabled=True))
        assert _deprecated_tokens(argv) == [], f"argv still passes {DEPRECATED_OPTION}: {argv!r}"

    def test_hf_disabled_argv_has_no_masks_version(self, main_module):
        """With hf_enabled false, the argv given to generate_embeddings carries no --masks-version option."""
        argv = _stage_argv(main_module, _config(main_module, hf_enabled=False))
        assert _deprecated_tokens(argv) == [], f"argv passes {DEPRECATED_OPTION}: {argv!r}"

    def test_hf_enabled_argv_parses_without_masks_version_future_warning(self, main_module):
        """The stage's hf_enabled argv, parsed by the real generate_embeddings parser, raises no FutureWarning
        naming --masks-version (the warning an orchestrated pipeline run used to print)."""
        argv = _stage_argv(main_module, _config(main_module, hf_enabled=True))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                GenerateEmbeddings().parse_args(argv)
            except SystemExit as exc:
                pytest.fail(f"GenerateEmbeddings parser rejected argv {argv!r} (exit {exc.code})")
        deprecations = [
            str(w.message)
            for w in caught
            if issubclass(w.category, FutureWarning) and DEPRECATED_OPTION in str(w.message)
        ]
        assert deprecations == [], f"parsing the stage argv warned: {deprecations!r}"
