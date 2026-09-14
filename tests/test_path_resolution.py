#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for REQ-PATH-01 — external/ path resolution.

Every script under ``src/champollion_pipeline/`` must reach the ``external/``
directory (and the repo-root ``reference.yaml``) by ascending *two* levels
from its own directory: ``src/champollion_pipeline/`` -> ``src/`` -> repo root.

Each test below pins down one of the five affected call sites.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

import champollion_pipeline

# Repo root = parent of src/, i.e. two levels above the package directory.
PACKAGE_DIR = Path(champollion_pipeline.__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
EXTERNAL_DIR = REPO_ROOT / "external"


@pytest.mark.smoke
class TestExternalPathResolution:
    """REQ-PATH-01: external/ must resolve to the repository root."""

    def test_repo_root_is_two_levels_above_package(self):
        """Sanity anchor: the real external/ dir lives two levels up, next to src/."""
        assert (REPO_ROOT / "src").is_dir()
        assert EXTERNAL_DIR.is_dir()

    def test_default_champollion_loc_points_at_repo_root_external(self):
        """generate_champollion_config._DEFAULT_CHAMPOLLION_LOC -> <root>/external/champollion_V1."""
        from champollion_pipeline.generate_champollion_config import _DEFAULT_CHAMPOLLION_LOC

        assert Path(_DEFAULT_CHAMPOLLION_LOC) == EXTERNAL_DIR / "champollion_V1"

    def test_reference_yaml_source_is_repo_root_reference_yaml(self):
        """generate_champollion_config copies reference.yaml from the repo root, not from src/."""
        from champollion_pipeline import generate_champollion_config as gcc

        # Verify that ascending two levels from _SCRIPT_DIR reaches the repo root.
        actual = Path(gcc.dirname(gcc.dirname(gcc._SCRIPT_DIR))) / "reference.yaml"
        assert actual == REPO_ROOT / "reference.yaml"

    def test_contrastive_dir_points_at_repo_root_external(self):
        """train_champollion._CONTRASTIVE_DIR -> <root>/external/champollion_V1/contrastive."""
        from champollion_pipeline.train_champollion import _CONTRASTIVE_DIR

        assert Path(_CONTRASTIVE_DIR) == EXTERNAL_DIR / "champollion_V1" / "champollion"

    def test_put_together_embeddings_copies_csvs_to_output(self, tmp_path):
        """put_together_embeddings.run() copies per-region CSVs to output_path."""
        from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings

        source = tmp_path / "mydata_embeddings"
        source.mkdir()
        region_dir = source / "SC-sylv_left"
        region_dir.mkdir()
        (region_dir / "full_embeddings.csv").write_text("subject,emb\nS01,0.1")

        output = tmp_path / "combined"
        script = PutTogetherEmbeddings()
        script.parse_args([str(source), "--output_path", str(output)])
        script.run()

        assert (output / "SC-sylv_left_embeddings.csv").exists()

    def test_run_cortical_tiles_invokes_repo_root_external_script(self, temp_dir):
        """run_cortical_tiles.run() invokes <root>/external/cortical_tiles/.../generate_sulcal_regions.py."""
        from champollion_pipeline.run_cortical_tiles import RunCorticalTiles

        expected = EXTERNAL_DIR / "cortical_tiles" / "deep_folding" / "brainvisa" / "generate_sulcal_regions.py"

        script = RunCorticalTiles()
        script.parse_args(
            [
                temp_dir,
                temp_dir,
                "--path_to_graph",
                "graphs",
                "--path_sk_with_hull",
                "skeleton",
            ]
        )

        with (
            patch("champollion_pipeline.run_cortical_tiles.chdir"),
            patch("champollion_pipeline.run_cortical_tiles.getcwd", return_value="/original"),
            patch.object(script, "validate_paths", return_value=True),
            patch.object(script, "execute_command", return_value=0) as mock_exec,
        ):
            script.run()

        invoked = [
            arg
            for call_args in mock_exec.call_args_list
            for arg in call_args[0][0]
            if str(arg).endswith("generate_sulcal_regions.py")
        ]
        assert invoked, "generate_sulcal_regions.py was never invoked"
        assert Path(invoked[0]) == expected
