#!/usr/bin/env python3
"""
Unit tests for the champollion_pipeline package __init__.

Stage classes are exported lazily through a PEP 562 module ``__getattr__``
so the package stays importable in partial pixi environments (embeddings-only
on Jean-Zay, no brainvisa, no torch...).  These tests exercise both the happy
path and the degraded path by making a stage submodule unimportable.
"""

import sys

import pytest

import champollion_pipeline

# Lazy exports of champollion_pipeline/__init__.py (_LAZY_EXPORTS), in declaration order.
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
    "generate_labelling_qc",
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
    "GenerateLabellingQC",
]


def block_submodules(monkeypatch, blocked):
    """Make the named stage submodules unimportable and drop their cached classes."""
    for submodule in blocked:
        monkeypatch.setitem(sys.modules, f"champollion_pipeline.{submodule}", None)
    for submodule, class_name in zip(GUARDED_SUBMODULES, EXPORTED_CLASSES, strict=True):
        if submodule in blocked:
            monkeypatch.delitem(vars(champollion_pipeline), class_name, raising=False)


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


class TestLazyExportsMatchTests:
    """Keep the test lists in step with the package's lazy export table."""

    def test_lists_equal_lazy_exports(self):
        assert dict(zip(EXPORTED_CLASSES, GUARDED_SUBMODULES, strict=True)) == champollion_pipeline._LAZY_EXPORTS


class TestGuardedImports:
    """Test that a missing optional dependency only hides its own class."""

    def test_package_still_works_when_every_submodule_fails(self, monkeypatch):
        block_submodules(monkeypatch, set(GUARDED_SUBMODULES))
        assert champollion_pipeline.__version__
        assert not any(hasattr(champollion_pipeline, c) for c in EXPORTED_CLASSES)

    @pytest.mark.parametrize("submodule,class_name", list(zip(GUARDED_SUBMODULES, EXPORTED_CLASSES, strict=True)))
    def test_single_failing_submodule_is_tolerated(self, monkeypatch, submodule, class_name):
        block_submodules(monkeypatch, {submodule})
        assert champollion_pipeline.__version__
        assert not hasattr(champollion_pipeline, class_name)
        others = [c for c in EXPORTED_CLASSES if c != class_name]
        assert all(hasattr(champollion_pipeline, c) for c in others)

    def test_package_recovers_after_a_failed_access(self, monkeypatch):
        with monkeypatch.context() as m:
            block_submodules(m, set(GUARDED_SUBMODULES))
            assert not hasattr(champollion_pipeline, "GenerateEmbeddings")
        assert all(hasattr(champollion_pipeline, c) for c in EXPORTED_CLASSES)

    def test_unknown_name_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            champollion_pipeline.NoSuchStage  # noqa: B018
