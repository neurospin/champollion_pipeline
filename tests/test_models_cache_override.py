#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The --models-cache PATH override of generate_embeddings.py.

For datasets mounted read-only, ``--models-cache PATH`` replaces the default
``<datasets_root>/derivatives/champollion_V1/models_cache`` as the extraction
directory GenerateEmbeddings.fetch_models hands to every model fetch strategy.
PATH is made absolute against the current working directory. The same
create/writability check as the default location applies to the override: an
unusable override directory fails before any strategy runs, with an OSError
naming the absolute override path.

Every fetch strategy is mocked: no network access, no model weights.
"""

import os
from os.path import join
from unittest.mock import patch

import pytest

from champollion_pipeline import generate_embeddings as ge
from champollion_pipeline.generate_embeddings import GenerateEmbeddings

STRATEGY_CLASSES = (
    "HuggingFaceStrategy",
    "RemoteArchiveStrategy",
    "LocalPathStrategy",
    "InteractiveFallbackStrategy",
)

RUNNING_AS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


def make_script(datasets_root, models_cache):
    """Return a GenerateEmbeddings parsed with --models-cache for one dataset root."""
    script = GenerateEmbeddings()
    script.parse_args(["/models", str(datasets_root), "--models-cache", str(models_cache)])
    return script


@pytest.fixture(autouse=True)  # noqa: V103 - autouse fixture, used by pytest
def keep_legacy_cache_out_of_repo(tmp_path, monkeypatch):
    """Point the module's __file__ into tmp_path.

    The legacy implementation derives the cache from the module location
    (<repo>/data/...); redirecting it keeps any write made by that code path
    inside tmp_path instead of the real repository.
    """
    fake_module = tmp_path / "fake_repo" / "src" / "champollion_pipeline" / "generate_embeddings.py"
    monkeypatch.setattr(ge, "__file__", str(fake_module))


@pytest.fixture
def mocked_strategies():
    """Patch every strategy's fetch; return a dict name -> mock."""
    patchers = {
        name: patch.object(getattr(ge, name), "fetch", return_value=f"/fetched/{name}") for name in STRATEGY_CLASSES
    }
    mocks = {name: p.start() for name, p in patchers.items()}
    yield mocks
    for p in patchers.values():
        p.stop()


def _models_path_for(strategy, tmp_path):
    """Return a models_path that fetch_models routes to the given strategy."""
    if strategy == "HuggingFaceStrategy":
        return "neurospin/Champollion_V1"
    if strategy == "RemoteArchiveStrategy":
        return "https://example.com/models.tar.gz"
    if strategy == "LocalPathStrategy":
        local = tmp_path / "local_models"
        local.mkdir()
        return str(local)
    return str(tmp_path / "definitely_missing_models")


class TestModelsCacheOverride:
    """--models-cache PATH is the extraction directory handed to the strategies."""

    @pytest.mark.parametrize("strategy", STRATEGY_CLASSES)
    def test_strategy_receives_absolute_override_dir(self, strategy, tmp_path, mocked_strategies):
        datasets_root = tmp_path / "DATASET"
        datasets_root.mkdir()
        override = tmp_path / "shared_cache" / "models"
        override.mkdir(parents=True)
        script = make_script(datasets_root, override)

        result = script.fetch_models(_models_path_for(strategy, tmp_path))

        assert result == f"/fetched/{strategy}"
        fetch = mocked_strategies[strategy]
        fetch.assert_called_once()
        extract_to = fetch.call_args.args[1]
        assert extract_to == os.path.abspath(str(override))
        assert not extract_to.startswith(str(datasets_root) + os.sep)

    def test_relative_override_resolves_against_cwd(self, tmp_path, monkeypatch, mocked_strategies):
        datasets_root = tmp_path / "DATASET"
        datasets_root.mkdir()
        (tmp_path / "rel_cache").mkdir()
        monkeypatch.chdir(tmp_path)
        script = make_script(datasets_root, "rel_cache")

        script.fetch_models(_models_path_for("InteractiveFallbackStrategy", tmp_path))

        extract_to = mocked_strategies["InteractiveFallbackStrategy"].call_args.args[1]
        assert os.path.isabs(extract_to)
        assert extract_to == join(os.getcwd(), "rel_cache")

    @pytest.mark.skipif(RUNNING_AS_ROOT, reason="root ignores directory permissions")
    def test_override_used_when_datasets_root_is_read_only(self, tmp_path, mocked_strategies):
        datasets_root = tmp_path / "READONLY_DATASET"
        datasets_root.mkdir()
        override = tmp_path / "writable_cache"
        override.mkdir()
        datasets_root.chmod(0o500)
        try:
            script = make_script(datasets_root, override)

            result = script.fetch_models(_models_path_for("HuggingFaceStrategy", tmp_path))

            assert result == "/fetched/HuggingFaceStrategy"
            extract_to = mocked_strategies["HuggingFaceStrategy"].call_args.args[1]
            assert extract_to == os.path.abspath(str(override))
            assert not (datasets_root / "derivatives").exists()
        finally:
            datasets_root.chmod(0o700)


@pytest.mark.skipif(RUNNING_AS_ROOT, reason="root ignores directory permissions")
class TestUnwritableModelsCacheOverride:
    """An unusable --models-cache directory fails fast with an OSError naming it."""

    @staticmethod
    def _assert_fails_before_strategies(script, expected_path, mocked_strategies, tmp_path):
        models_path = _models_path_for("HuggingFaceStrategy", tmp_path)
        with pytest.raises(OSError) as excinfo:
            script.fetch_models(models_path)
        assert expected_path in str(excinfo.value)
        for name, fetch in mocked_strategies.items():
            assert not fetch.called, f"{name}.fetch ran before the --models-cache directory error"

    @staticmethod
    def _writable_datasets_root(tmp_path):
        datasets_root = tmp_path / "DATASET"
        datasets_root.mkdir()
        return datasets_root

    def test_uncreatable_override_raises_oserror_naming_path(self, tmp_path, mocked_strategies):
        datasets_root = self._writable_datasets_root(tmp_path)
        readonly_parent = tmp_path / "readonly_parent"
        readonly_parent.mkdir()
        override = readonly_parent / "models_cache"
        readonly_parent.chmod(0o500)
        try:
            script = make_script(datasets_root, override)
            self._assert_fails_before_strategies(script, os.path.abspath(str(override)), mocked_strategies, tmp_path)
        finally:
            readonly_parent.chmod(0o700)

    def test_unwritable_existing_override_raises_oserror_naming_path(self, tmp_path, mocked_strategies):
        datasets_root = self._writable_datasets_root(tmp_path)
        override = tmp_path / "readonly_cache"
        override.mkdir()
        override.chmod(0o500)
        try:
            script = make_script(datasets_root, override)
            self._assert_fails_before_strategies(script, os.path.abspath(str(override)), mocked_strategies, tmp_path)
        finally:
            override.chmod(0o700)

    def test_relative_unwritable_override_error_names_absolute_path(self, tmp_path, monkeypatch, mocked_strategies):
        datasets_root = self._writable_datasets_root(tmp_path)
        override = tmp_path / "readonly_rel_cache"
        override.mkdir()
        override.chmod(0o500)
        monkeypatch.chdir(tmp_path)
        try:
            script = make_script(datasets_root, "readonly_rel_cache")
            expected = join(os.getcwd(), "readonly_rel_cache")
            self._assert_fails_before_strategies(script, expected, mocked_strategies, tmp_path)
        finally:
            override.chmod(0o700)
