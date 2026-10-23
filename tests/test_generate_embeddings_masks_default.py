#!/usr/bin/env python3
"""
generate_embeddings: --masks defaults to --model-version (MODELVER epic, second task).

When only --model-version <V> is given, the cortical_tiles crops mask version (--masks)
defaults to <V>, so the crops directory and the default output folder follow the model
set. With neither option, --masks stays canonical_25 (today's default). An explicit
--masks is kept as given. The deprecated --masks-version <V> alone also yields masks <V>
(alias resolved first). The --masks / --model-version mismatch error belongs to the
next task and is not tested here.

No model, no evaluate.py, no network: only the real GenerateEmbeddings parser.
"""

from os.path import join

import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings

pytestmark = pytest.mark.unit  # noqa: V107

VERSION = "canonical_corrected_26_1"
EXPLICIT_MASKS = "canonical_24"
DEFAULT_MASKS = "canonical_25"
_ABSENT = "<no masks attribute>"


def _positionals(tmp_path) -> list[str]:
    return ["/models", str(tmp_path / "TEST01")]


def _parse(argv: list[str]) -> GenerateEmbeddings:
    """Return a GenerateEmbeddings whose args are argv parsed by the real parser; fail on rejection."""
    script = GenerateEmbeddings()
    try:
        script.parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"GenerateEmbeddings parser rejected {argv!r} (SystemExit {exc.code})")
    return script


def _masks(script: GenerateEmbeddings) -> str:
    return vars(script.args).get("masks", _ABSENT)


class TestMasksDefaultsToModelVersion:
    """REQ-MODELVER-BDRABCZUK-1B99CEB3A3D7: --model-version <V> without --masks yields masks <V>."""

    def test_model_version_alone_yields_masks(self, tmp_path):
        script = _parse([*_positionals(tmp_path), "--model-version", VERSION])
        assert _masks(script) == VERSION, vars(script.args)

    def test_model_version_alone_selects_crops_dir(self, tmp_path):
        datasets_root = str(tmp_path / "TEST01")
        script = _parse([*_positionals(tmp_path), "--model-version", VERSION])
        expected = join(datasets_root, "derivatives", script.args.cortical_version, "crops", VERSION, "2mm")
        assert script._get_crops_2mm_dir(datasets_root) == expected


class TestMasksDefaultWithoutVersions:
    """REQ-MODELVER-BDRABCZUK-6307AC2A6A2B: neither --model-version nor --masks yields masks canonical_25."""

    def test_no_version_options_yield_canonical_25(self, tmp_path):
        script = _parse(_positionals(tmp_path))
        assert _masks(script) == DEFAULT_MASKS, vars(script.args)


class TestExplicitMasksKept:
    """REQ-MODELVER-BDRABCZUK-645F10C1ADBC: --masks <M> without --model-version yields masks <M>."""

    def test_explicit_masks_kept(self, tmp_path):
        script = _parse([*_positionals(tmp_path), "--masks", EXPLICIT_MASKS])
        assert _masks(script) == EXPLICIT_MASKS, vars(script.args)


class TestMasksDefaultsToMasksVersionAlias:
    """REQ-MODELVER-BDRABCZUK-59D9506761B0: --masks-version <V> alone yields masks <V>."""

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_masks_version_alone_yields_masks(self, tmp_path):
        script = _parse([*_positionals(tmp_path), "--masks-version", VERSION])
        assert _masks(script) == VERSION, vars(script.args)
