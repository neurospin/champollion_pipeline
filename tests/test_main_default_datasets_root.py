#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for main.py's default dataset root (REQ-DEFROOT-01) and the absence of
any champollion_V1 ``contrastive`` directory remnant in main.py (REQ-DEFROOT-02).

``GenerateEmbeddingsStage`` hands ``dataset.datasets_root`` to
generate_embeddings.py as its ``datasets_root`` positional: the dataset root
that holds ``derivatives/`` and ``participants.tsv``. The default config must
therefore point at ``<data_path>/<dataset.name>``, never at a configs folder
inside the champollion_V1 submodule.

main.py lives at the repo root (not inside the installed package), so it is
loaded by file path under a private module name, as in test_main_stage_argv.py.
"""

import ast
import importlib.util
from pathlib import Path

import pytest

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_default_root_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def default_config():
    return _load_main_module().create_default_config()


@pytest.mark.unit
class TestDefaultDatasetsRoot:
    """REQ-DEFROOT-01: default datasets_root is Path(data_path) / dataset.name."""

    def test_default_datasets_root_is_data_path_joined_with_dataset_name(self, default_config):
        """REQ-DEFROOT-01: datasets_root equals Path(data_path) / dataset.name of the same config."""
        expected = Path(default_config.data_path) / default_config.dataset.name
        assert Path(default_config.dataset.datasets_root) == expected

    def test_default_datasets_root_not_under_champollion_v1(self, default_config):
        """REQ-DEFROOT-01: the default dataset root is not inside the champollion_V1 submodule."""
        datasets_root = Path(default_config.dataset.datasets_root)
        champollion_v1 = Path(default_config.champollion_v1_path)
        assert not datasets_root.is_relative_to(champollion_v1)


@pytest.mark.unit
class TestNoContrastiveRemnant:
    """REQ-DEFROOT-02: main.py names no champollion_V1 ``contrastive`` directory."""

    def test_main_py_has_no_contrastive_path_substring(self):
        """REQ-DEFROOT-02: the substring 'contrastive/' appears nowhere in main.py."""
        offending = [
            f"{lineno}: {line.strip()}"
            for lineno, line in enumerate(MAIN_PY.read_text(encoding="utf-8").splitlines(), start=1)
            if "contrastive/" in line
        ]
        assert offending == []

    def test_main_py_has_no_contrastive_string_literal(self):
        """REQ-DEFROOT-02: no Python string literal in main.py equals 'contrastive'."""
        tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"), filename=str(MAIN_PY))
        offending = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value == "contrastive"
        ]
        assert offending == []
