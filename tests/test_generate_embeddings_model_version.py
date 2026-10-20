#!/usr/bin/env python3
"""
generate_embeddings: the --model-version option and its deprecated alias --masks-version
(MODELVER epic, first task).

--model-version names the Hugging Face model-set subfolder (the role --masks-version
used to have). --masks-version stays accepted as a deprecated alias: same parsed value,
plus a FutureWarning (the file's existing deprecation convention, see --subjects).

No model, no evaluate.py, no network: the real GenerateEmbeddings parser and
fetch_models with HuggingFaceStrategy replaced by a mock.
"""

import warnings
from unittest.mock import MagicMock, patch

import pytest

import champollion_pipeline.generate_embeddings as ge
from champollion_pipeline.generate_embeddings import GenerateEmbeddings

pytestmark = pytest.mark.unit  # noqa: V107

HF_REPO = "neurospin/Champollion_V1"
VERSION = "canonical_corrected_26_1"
_ABSENT = "<no model_version attribute>"


def _positionals(tmp_path) -> list[str]:
    """models_path and a writable datasets_root (fetch_models creates its models cache there)."""
    return ["/models", str(tmp_path / "TEST01")]


def _parse(argv: list[str]) -> GenerateEmbeddings:
    """Return a GenerateEmbeddings whose args are argv parsed by the real parser; fail on rejection."""
    script = GenerateEmbeddings()
    try:
        script.parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"GenerateEmbeddings parser rejected {argv!r} (SystemExit {exc.code})")
    return script


def _parse_and_fetch(argv: list[str]) -> tuple[MagicMock, list[warnings.WarningMessage]]:
    """Parse argv, run fetch_models on a HF repo id with a mocked HuggingFaceStrategy.

    Returns the mocked strategy class and every warning raised by parsing or fetching,
    so a warning emitted at either point is seen.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        script = _parse(argv)
        strategy = MagicMock(can_handle=MagicMock(return_value=True), fetch=MagicMock(return_value="/hf"))
        with patch.object(ge, "HuggingFaceStrategy", return_value=strategy) as strategy_cls:
            script.fetch_models(HF_REPO)
    return strategy_cls, list(caught)


def _future_warning_messages(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]


class TestModelVersionSelectsHfSubfolder:
    """REQ-MODELVER-BDRABCZUK-EFDB859C8303: --model-version <V> fetches HF subfolder <V>."""

    def test_model_version_is_passed_as_hf_subfolder(self, tmp_path):
        strategy_cls, _ = _parse_and_fetch([*_positionals(tmp_path), "--model-version", VERSION])
        assert strategy_cls.call_args is not None, "HuggingFaceStrategy was not constructed"
        assert strategy_cls.call_args.kwargs.get("subfolder") == VERSION, strategy_cls.call_args


class TestMasksVersionAlias:
    """REQ-MODELVER-BDRABCZUK-3C9C56B7B9E8: --masks-version <V> alone yields model_version <V>."""

    def test_masks_version_yields_model_version(self, tmp_path):
        script = _parse([*_positionals(tmp_path), "--masks-version", VERSION])
        assert vars(script.args).get("model_version", _ABSENT) == VERSION, vars(script.args)


class TestMasksVersionDeprecationWarning:
    """REQ-MODELVER-BDRABCZUK-BAA0164B78C1: --masks-version emits a FutureWarning naming --model-version."""

    def test_masks_version_warns_naming_model_version(self, tmp_path):
        _, caught = _parse_and_fetch([*_positionals(tmp_path), "--masks-version", VERSION])
        messages = _future_warning_messages(caught)
        assert any("--model-version" in message for message in messages), messages


class TestModelVersionNoDeprecationWarning:
    """REQ-MODELVER-BDRABCZUK-A64E64BDDE8B: --model-version alone emits no FutureWarning naming --masks-version."""

    def test_model_version_alone_does_not_warn(self, tmp_path):
        strategy_cls, caught = _parse_and_fetch([*_positionals(tmp_path), "--model-version", VERSION])
        messages = _future_warning_messages(caught)
        assert not any("--masks-version" in message for message in messages), messages
        # Guard against a vacuous pass: the option must have reached the HF strategy.
        assert strategy_cls.call_args is not None, "HuggingFaceStrategy was not constructed"
        assert strategy_cls.call_args.kwargs.get("subfolder") == VERSION, strategy_cls.call_args


class TestModelVersionMasksVersionConflict:
    """REQ-MODELVER-BDRABCZUK-163B193212A4: different --model-version and --masks-version exit 2 naming both."""

    def test_different_values_exit_2_naming_both(self, tmp_path, capsys):
        model_version, masks_version = "canonical_corrected_26_1", "canonical_25"
        argv = [*_positionals(tmp_path), "--model-version", model_version, "--masks-version", masks_version]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with pytest.raises(SystemExit) as exc_info:
                GenerateEmbeddings().parse_args(argv)
        assert exc_info.value.code == 2, exc_info.value.code
        stderr = capsys.readouterr().err
        assert model_version in stderr, stderr
        assert masks_version in stderr, stderr
