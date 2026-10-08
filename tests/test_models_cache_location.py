#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Where GenerateEmbeddings.fetch_models caches downloaded/extracted models.

The models cache lives inside the dataset tree, at
``<datasets_root>/derivatives/champollion_V1/models_cache``, and no longer
under ``<pipeline repo>/data/<datasets_root without leading />/...``.
When that directory cannot be created or written, fetch_models fails before
any model fetch strategy runs, with an OSError naming the directory.

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


def expected_cache_dir(datasets_root):
    """The cache location the requirement names, spelled out literally."""
    return os.path.abspath(join(datasets_root, "derivatives", "champollion_V1", "models_cache"))


def make_script(datasets_root, extra_argv=None):
    """Return a GenerateEmbeddings with parsed arguments for one dataset root."""
    script = GenerateEmbeddings()
    script.parse_args(["/models", str(datasets_root)] + (extra_argv or []))
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


class TestModelsCacheInsideDatasetTree:
    """The cache directory handed to the strategies sits inside datasets_root."""

    @pytest.mark.parametrize("strategy", STRATEGY_CLASSES)
    def test_strategy_receives_cache_dir_inside_absolute_datasets_root(self, strategy, tmp_path, mocked_strategies):
        datasets_root = tmp_path / "DATASET"
        datasets_root.mkdir()
        script = make_script(datasets_root)

        result = script.fetch_models(_models_path_for(strategy, tmp_path))

        assert result == f"/fetched/{strategy}"
        fetch = mocked_strategies[strategy]
        fetch.assert_called_once()
        extract_to = fetch.call_args.args[1]
        assert extract_to == expected_cache_dir(datasets_root)

    def test_relative_datasets_root_cache_dir_is_absolute_under_cwd(self, tmp_path, monkeypatch, mocked_strategies):
        (tmp_path / "DATASET").mkdir()
        monkeypatch.chdir(tmp_path)
        script = make_script("DATASET")

        script.fetch_models(str(tmp_path / "definitely_missing_models"))

        extract_to = mocked_strategies["InteractiveFallbackStrategy"].call_args.args[1]
        assert os.path.isabs(extract_to)
        assert extract_to == join(str(tmp_path), "DATASET", "derivatives", "champollion_V1", "models_cache")


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory permissions")
class TestUnwritableModelsCache:
    """An unusable cache directory fails fast with an OSError naming it."""

    @staticmethod
    def _assert_fails_before_strategies(script, datasets_root, mocked_strategies, tmp_path):
        cache_dir = expected_cache_dir(datasets_root)
        with pytest.raises(OSError) as excinfo:
            script.fetch_models(_models_path_for("HuggingFaceStrategy", tmp_path))
        assert cache_dir in str(excinfo.value)
        for name, fetch in mocked_strategies.items():
            assert not fetch.called, f"{name}.fetch ran before the cache directory error"

    def test_uncreatable_cache_dir_raises_oserror_naming_path(self, tmp_path, mocked_strategies):
        datasets_root = tmp_path / "READONLY_DATASET"
        datasets_root.mkdir()
        datasets_root.chmod(0o500)
        try:
            script = make_script(datasets_root)
            self._assert_fails_before_strategies(script, datasets_root, mocked_strategies, tmp_path)
        finally:
            datasets_root.chmod(0o700)

    def test_unwritable_existing_cache_dir_raises_oserror_naming_path(self, tmp_path, mocked_strategies):
        datasets_root = tmp_path / "DATASET"
        cache_dir = datasets_root / "derivatives" / "champollion_V1" / "models_cache"
        cache_dir.mkdir(parents=True)
        cache_dir.chmod(0o500)
        try:
            script = make_script(datasets_root)
            self._assert_fails_before_strategies(script, datasets_root, mocked_strategies, tmp_path)
        finally:
            cache_dir.chmod(0o700)
