#!/usr/bin/env python3
"""
generate_embeddings: --masks / --model-version mismatch policy (MODELVER epic).

--masks picks the input crops (and the default output folder); --model-version picks the
model set. When both are given and differ, the parser exits with status 2 naming both
values, unless --allow-version-mismatch is given, in which case it emits a UserWarning
naming both and keeps both values.

Decisions settled by these requirements:
- The check is a parse-time check: it applies whether models_path is a local directory
  (where --model-version selects nothing today) or a Hugging Face repository id.
- The deprecated alias --masks-version counts as --model-version (alias resolved first),
  so an explicit, different --masks mismatches it too. --masks-version alone defaults
  --masks to its value and never mismatches (REQ-MODELVER-BDRABCZUK-59D9506761B0).

No model, no evaluate.py, no network: only the real GenerateEmbeddings parser.
"""

import warnings

import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings

pytestmark = pytest.mark.unit  # noqa: V107

HF_REPO = "neurospin/Champollion_V1"
MODEL_VERSION = "canonical_corrected_26_1"
MASKS = "canonical_25"


def _datasets_root(tmp_path) -> str:
    return str(tmp_path / "TEST01")


def _parse(argv: list[str]) -> GenerateEmbeddings:
    """Return a GenerateEmbeddings whose args are argv parsed by the real parser; fail on rejection."""
    script = GenerateEmbeddings()
    try:
        script.parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"GenerateEmbeddings parser rejected {argv!r} (SystemExit {exc.code})")
    return script


def _expect_usage_error(argv: list[str], capsys) -> str:
    """Parse argv with the real parser, assert SystemExit 2, return stderr."""
    script = GenerateEmbeddings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with pytest.raises(SystemExit) as exc_info:
            script.parse_args(argv)
    assert exc_info.value.code == 2
    return capsys.readouterr().err


class TestMasksModelVersionMismatchError:
    """REQ-MODELVER-BDRABCZUK-C87FF389BD8F: --masks <M> != --model-version <V> exits 2 naming both."""

    def test_mismatch_exits_2_naming_both(self, tmp_path, capsys):
        argv = [HF_REPO, _datasets_root(tmp_path), "--masks", MASKS, "--model-version", MODEL_VERSION]
        stderr = _expect_usage_error(argv, capsys)
        assert MASKS in stderr, stderr
        assert MODEL_VERSION in stderr, stderr

    def test_mismatch_with_local_models_path_exits_2(self, tmp_path, capsys):
        local_models = tmp_path / "models"
        local_models.mkdir()
        argv = [str(local_models), _datasets_root(tmp_path), "--masks", MASKS, "--model-version", MODEL_VERSION]
        stderr = _expect_usage_error(argv, capsys)
        assert MASKS in stderr, stderr
        assert MODEL_VERSION in stderr, stderr

    def test_equal_values_parse_without_warning(self, tmp_path):
        """Regression guard: --masks equal to --model-version is no mismatch."""
        argv = [HF_REPO, _datasets_root(tmp_path), "--masks", MODEL_VERSION, "--model-version", MODEL_VERSION]
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            script = _parse(argv)
        assert script.args.masks == MODEL_VERSION


class TestMasksAliasMismatchError:
    """REQ-MODELVER-BDRABCZUK-00BF1BEA8AF2: --masks <M> != --masks-version <V> exits 2 naming both."""

    def test_alias_mismatch_exits_2_naming_both(self, tmp_path, capsys):
        argv = [HF_REPO, _datasets_root(tmp_path), "--masks", MASKS, "--masks-version", MODEL_VERSION]
        stderr = _expect_usage_error(argv, capsys)
        assert MASKS in stderr, stderr
        assert MODEL_VERSION in stderr, stderr


class TestAllowVersionMismatch:
    """REQ-MODELVER-BDRABCZUK-55C6CBEED962 and REQ-MODELVER-BDRABCZUK-93C601727466: the override flag."""

    def _argv(self, tmp_path) -> list[str]:
        return [
            HF_REPO,
            _datasets_root(tmp_path),
            "--masks",
            MASKS,
            "--model-version",
            MODEL_VERSION,
            "--allow-version-mismatch",
        ]

    def test_override_warns_naming_both(self, tmp_path):
        """REQ-MODELVER-BDRABCZUK-55C6CBEED962: a UserWarning names <M> and <V>."""
        with pytest.warns(UserWarning) as record:
            _parse(self._argv(tmp_path))
        messages = [str(w.message) for w in record if issubclass(w.category, UserWarning)]
        assert any(MASKS in m and MODEL_VERSION in m for m in messages), messages

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_override_keeps_both_versions(self, tmp_path):
        """REQ-MODELVER-BDRABCZUK-93C601727466: masks <M> with model_version <V>."""
        script = _parse(self._argv(tmp_path))
        assert script.args.masks == MASKS, vars(script.args)
        assert script.args.model_version == MODEL_VERSION, vars(script.args)
