#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the champollion_pipeline package __init__.

Each script is imported behind a ``try/except ImportError`` guard so the
package stays importable in partial pixi environments (embeddings-only on
Jean-Zay, no brainvisa, no torch...).  These tests exercise both the happy
path and the degraded path by re-executing the package body with submodule
imports forced to fail.
"""

import builtins
import importlib

import pytest

import champollion_pipeline

# Names imported by champollion_pipeline/__init__.py, in declaration order.
GUARDED_SUBMODULES = [
    "generate_morphologist_graphs",
    "run_cortical_tiles",
    "generate_champollion_config",
    "generate_embeddings",
    "put_together_embeddings",
    "generate_snapshots",
    "train_champollion",
    "prune_failed_subjects",
    "purge_subject",
    "generate_masks",
]

EXPORTED_CLASSES = [
    "GenerateMorphologistGraphs",
    "RunCorticalTiles",
    "GenerateChampollionConfig",
    "GenerateEmbeddings",
    "PutTogetherEmbeddings",
    "GenerateSnapshots",
    "TrainChampollion",
    "PruneFailedSubjects",
    "PurgeSubject",
    "GenerateMasks",
]


def reload_with_blocked_submodules(blocked):
    """Re-execute the package body with the named relative imports failing."""
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level > 0 and name in blocked:
            raise ImportError(f"simulated missing optional dependency for {name}")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    try:
        importlib.reload(champollion_pipeline)
    finally:
        builtins.__import__ = real_import


@pytest.fixture
def restore_package():
    """Reload the package normally after a test has degraded it."""
    yield
    importlib.reload(champollion_pipeline)


class TestPackageMetadata:
    """Test package level metadata."""

    def test_version_is_exposed(self):
        assert isinstance(champollion_pipeline.__version__, str)
        assert champollion_pipeline.__version__.count(".") == 2

    def test_docstring_mentions_the_pipeline(self):
        assert "pipeline" in champollion_pipeline.__doc__.lower()


class TestPackageExports:
    """Test that every script class is re-exported when its deps are present."""

    @pytest.mark.parametrize("class_name", EXPORTED_CLASSES)
    def test_class_is_exported(self, class_name):
        assert hasattr(champollion_pipeline, class_name)


class TestGuardedImports:
    """Test the ImportError fallbacks for partial environments."""

    def test_package_still_imports_when_every_submodule_fails(self, restore_package):
        reload_with_blocked_submodules(set(GUARDED_SUBMODULES))
        assert champollion_pipeline.__version__

    @pytest.mark.parametrize(
        "submodule,class_name", list(zip(GUARDED_SUBMODULES, EXPORTED_CLASSES))
    )
    def test_single_failing_submodule_is_tolerated(self, restore_package, submodule, class_name):
        reload_with_blocked_submodules({submodule})
        # The package imported successfully even though this one script did not.
        assert champollion_pipeline.__version__
        others = [c for c in EXPORTED_CLASSES if c != class_name]
        assert all(hasattr(champollion_pipeline, c) for c in others)

    def test_package_recovers_after_a_degraded_reload(self, restore_package):
        reload_with_blocked_submodules(set(GUARDED_SUBMODULES))
        importlib.reload(champollion_pipeline)
        assert all(hasattr(champollion_pipeline, c) for c in EXPORTED_CLASSES)
