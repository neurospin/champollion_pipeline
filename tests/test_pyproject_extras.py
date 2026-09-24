#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optional-dependency extras declared in pyproject.toml.

Run with: pixi run test-specific tests/test_pyproject_extras.py

REQ-INSTALL-04: `pip install -e .` in a clean virtual environment leaves
`champollion_pipeline.generate_embeddings` unimportable — it imports `torch`
at module level, and the champollion_V1 CKA module it loads
(`champollion.metrics.cka_coherence`) imports `pandas` at module level.
Neither is declared in `[project].dependencies`. `pyproject.toml` must
declare an `embeddings` extra under `[project.optional-dependencies]` that
lists both, so `pip install -e .[embeddings]` makes the module importable.
"""

import re
from pathlib import Path

import pytest
import tomllib

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _optional_dependencies() -> dict:
    """Return the `[project.optional-dependencies]` table (empty if absent)."""
    with PYPROJECT.open("rb") as handle:
        data = tomllib.load(handle)
    return data["project"].get("optional-dependencies", {})


def _distribution_name(requirement: str) -> str:
    """Return the normalized (PEP 503) distribution name of a PEP 508 entry."""
    name = re.split(r"[\s<>=!~;\[@(]", requirement.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def _embeddings_extra_names() -> set[str]:
    extras = _optional_dependencies()
    assert "embeddings" in extras, (
        "pyproject.toml declares no 'embeddings' extra under "
        f"[project.optional-dependencies] (found extras: {sorted(extras)!r}); "
        "`pip install -e .[embeddings]` cannot pull in the embeddings-stage dependencies."
    )
    return {_distribution_name(entry) for entry in extras["embeddings"]}


class TestEmbeddingsExtra:
    """`pip install -e .[embeddings]` must cover generate_embeddings' imports."""

    @pytest.mark.smoke
    def test_embeddings_extra_is_declared(self):
        """An extra named exactly `embeddings` exists."""
        assert "embeddings" in _optional_dependencies(), (
            "pyproject.toml declares no 'embeddings' extra under [project.optional-dependencies]"
        )

    @pytest.mark.smoke
    def test_embeddings_extra_lists_torch(self):
        """`torch` is imported at module level by generate_embeddings.py."""
        assert "torch" in _embeddings_extra_names(), (
            "the 'embeddings' extra does not list torch; "
            "`import champollion_pipeline.generate_embeddings` raises ModuleNotFoundError: torch"
        )

    @pytest.mark.smoke
    def test_embeddings_extra_lists_pandas(self):
        """`pandas` is imported at module level by champollion.metrics.cka_coherence."""
        assert "pandas" in _embeddings_extra_names(), (
            "the 'embeddings' extra does not list pandas; "
            "champollion.metrics.cka_coherence (loaded by generate_embeddings) "
            "raises ModuleNotFoundError: pandas"
        )
