#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Source-text guards for REQ-REFACTOR-01.

Julien renamed the ``contrastive/`` Python package in neurospin/champollion to
``champollion/``. Every path and import in champollion_pipeline that still
names ``contrastive`` breaks against the renamed upstream checkout.

These tests read the source files as text (no runtime import, so no
champollion_utils / torch / hydra dependency, and no dependency on
``external/champollion_V1`` actually being checked out).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PKG = PROJECT_ROOT / "src" / "champollion_pipeline"
SRC = PROJECT_ROOT / "src"


def _read(path: Path) -> str:
    assert path.is_file(), f"expected source file to exist: {path}"
    return path.read_text(encoding="utf-8")


def _code_lines(text: str) -> list[str]:
    """Lines with whole-line comments and blank lines removed."""
    return [line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


@pytest.mark.smoke
class TestGenerateEmbeddingsRename:
    """generate_embeddings.py must point at the renamed package."""

    def test_champollion_dir_constant_targets_champollion_package(self):
        """_CHAMPOLLION_DIR is the champollion_V1 parent dir (sys.path parent for champollion pkg)."""
        lines = [ln for ln in _code_lines(_read(PKG / "generate_embeddings.py")) if "_CHAMPOLLION_DIR" in ln and "=" in ln]
        assert lines, "no _CHAMPOLLION_DIR assignment found in generate_embeddings.py"
        assignment = lines[0]
        assert '"champollion_V1"' in assignment, (
            f"_CHAMPOLLION_DIR must reference champollion_V1, got: {assignment.strip()}"
        )
        assert "contrastive" not in assignment

    def test_no_contrastive_package_import(self):
        """No `from contrastive.` / `import contrastive` remains."""
        text = _read(PKG / "generate_embeddings.py")
        assert not re.search(r"^\s*from\s+contrastive[\s.]", text, re.MULTILINE), (
            "generate_embeddings.py still imports from the contrastive package"
        )
        assert not re.search(r"^\s*import\s+contrastive\b", text, re.MULTILINE)
        assert "from champollion.evaluation.cka_coherence import" in text, (
            "expected the CKA import to come from champollion.evaluation.cka_coherence"
        )

    def test_subprocess_cwd_targets_champollion_package(self):
        """The chdir target used for the inference subprocess is the renamed dir."""
        lines = [ln for ln in _code_lines(_read(PKG / "generate_embeddings.py")) if "champollion_V1" in ln]
        assert lines, "no champollion_V1 path join found in generate_embeddings.py"
        offenders = [ln.strip() for ln in lines if "contrastive" in ln]
        assert not offenders, f"champollion_V1 paths still naming contrastive: {offenders}"

    def test_user_facing_config_path_message_renamed(self):
        """The error string pointing users at the configs dir names champollion/."""
        text = _read(PKG / "generate_embeddings.py")
        assert "external/champollion_V1/contrastive/configs/" not in text, (
            "error message still directs users to champollion_V1/contrastive/configs/"
        )
        assert "external/champollion_V1/champollion/configs/" in text


@pytest.mark.smoke
class TestPutTogetherEmbeddingsRename:
    def test_sys_path_append_targets_champollion_utils(self):
        """put_together_embeddings.py appends champollion_V1/champollion/utils."""
        lines = [ln for ln in _code_lines(_read(PKG / "put_together_embeddings.py")) if "champollion_V1" in ln]
        assert lines, "no champollion_V1 path found in put_together_embeddings.py"
        joined = "\n".join(lines)
        assert '"champollion_V1", "champollion", "utils"' in joined, (
            f"expected champollion_V1/champollion/utils, got: {joined.strip()}"
        )
        assert "contrastive" not in joined


@pytest.mark.smoke
class TestTrainChampollionRename:
    def test_module_dir_constant_targets_champollion_package(self):
        """train_champollion.py's champollion checkout dir constant is renamed."""
        lines = [
            ln
            for ln in _code_lines(_read(PKG / "train_champollion.py"))
            if "champollion_V1" in ln and "=" in ln
        ]
        assert lines, "no champollion_V1 directory constant found in train_champollion.py"
        assignment = lines[0]
        assert '"champollion_V1", "champollion"' in assignment, (
            f"module dir constant must end with champollion_V1/champollion, got: {assignment.strip()}"
        )
        assert "contrastive" not in assignment


@pytest.mark.smoke
class TestGenerateChampollionConfigRename:
    def test_config_path_joins_use_champollion(self):
        """join(champollion_loc, ...) config paths no longer name contrastive."""
        lines = [
            ln
            for ln in _code_lines(_read(PKG / "generate_champollion_config.py"))
            if "champollion_loc" in ln and "join(" in ln
        ]
        assert lines, "no join(champollion_loc, ...) found in generate_champollion_config.py"
        offenders = [ln.strip() for ln in lines if '"contrastive"' in ln]
        assert not offenders, f"config path joins still using the contrastive package dir: {offenders}"
        assert any('"champollion", "configs"' in ln for ln in lines), (
            'expected at least one join(champollion_loc, "champollion", "configs", ...)'
        )


@pytest.mark.smoke
class TestRunCkaRename:
    def test_cka_module_constant_renamed(self):
        """src/run_cka.py invokes champollion.evaluation.cka_coherence."""
        lines = [ln for ln in _code_lines(_read(SRC / "run_cka.py")) if "CKA_MODULE" in ln and "=" in ln]
        assert lines, "no CKA_MODULE assignment found in src/run_cka.py"
        assignment = lines[0]
        assert "champollion.evaluation.cka_coherence" in assignment, (
            f"CKA_MODULE must name the champollion package, got: {assignment.strip()}"
        )
        assert "contrastive" not in assignment


@pytest.mark.smoke
class TestGenerateUmapReferenceRename:
    def test_sys_path_append_targets_champollion_utils(self):
        """src/generate_umap_reference.py appends champollion_V1/champollion/utils."""
        lines = [ln for ln in _code_lines(_read(SRC / "generate_umap_reference.py")) if "champollion_V1" in ln]
        assert lines, "no champollion_V1 path found in src/generate_umap_reference.py"
        joined = "\n".join(lines)
        assert '"champollion_V1", "champollion", "utils"' in joined, (
            f"expected champollion_V1/champollion/utils, got: {joined.strip()}"
        )
        assert "contrastive" not in joined
