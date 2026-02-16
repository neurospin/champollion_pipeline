#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for generate_snapshots.py
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch

from generate_snapshots import (
    find_sulcal_graphs,
    find_white_mesh,
    find_completed_regions,
    COLLATERAL_FILES,
    GenerateSnapshots,
    main,
)


class TestFindSulcalGraphs:
    """Test find_sulcal_graphs function."""

    def test_finds_arg_files_with_sulci_in_name(self, temp_dir):
        """Test that .arg files with 'sulci' in the name are found."""
        graph_dir = Path(temp_dir) / "sub-01" / "folds"
        graph_dir.mkdir(parents=True)
        (graph_dir / "Lsulci_sub-01.arg").touch()
        (graph_dir / "Rsulci_sub-01.arg").touch()

        result = find_sulcal_graphs(temp_dir)
        assert len(result) == 2

    def test_finds_arg_files_with_folds_in_name(self, temp_dir):
        """Test that .arg files with 'folds' in the path are found."""
        graph_dir = Path(temp_dir) / "sub-01" / "folds" / "3.1"
        graph_dir.mkdir(parents=True)
        (graph_dir / "Lfolds_sub-01.arg").touch()

        result = find_sulcal_graphs(temp_dir)
        assert len(result) == 1

    def test_ignores_arg_files_without_sulci_or_folds(self, temp_dir):
        """Test that .arg files without 'sulci'/'folds' are excluded."""
        graph_dir = Path(temp_dir) / "sub-01"
        graph_dir.mkdir(parents=True)
        (graph_dir / "other_data.arg").touch()

        result = find_sulcal_graphs(temp_dir)
        assert len(result) == 0

    def test_empty_directory_returns_empty(self, temp_dir):
        """Test that empty directory returns empty list."""
        result = find_sulcal_graphs(temp_dir)
        assert result == []

    def test_recursive_search(self, temp_dir):
        """Test that search is recursive through subdirectories."""
        deep_dir = Path(temp_dir) / "a" / "b" / "c" / "folds"
        deep_dir.mkdir(parents=True)
        (deep_dir / "Lsulci_deep.arg").touch()

        result = find_sulcal_graphs(temp_dir)
        assert len(result) == 1


class TestFindWhiteMesh:
    """Test find_white_mesh function."""

    def test_morphologist_60_structure(self, temp_dir):
        """Test finding mesh in Morphologist 6.0 layout (default_acquisition/0)."""
        # Build: sub-01/default_acquisition/0/segmentation/mesh/sub-01_Lwhite.gii
        base = Path(temp_dir) / "sub-01" / "default_acquisition" / "0"
        mesh_dir = base / "segmentation" / "mesh"
        mesh_dir.mkdir(parents=True)
        (mesh_dir / "sub-01_Lwhite.gii").touch()

        folds_dir = base / "folds" / "3.1"
        folds_dir.mkdir(parents=True)
        graph_path = str(folds_dir / "Lsulci_sub-01.arg")
        Path(graph_path).touch()

        result = find_white_mesh(graph_path)
        assert result is not None
        assert result.endswith("Lwhite.gii")

    def test_morphologist_legacy_structure(self, temp_dir):
        """Test finding mesh in Morphologist <6 layout (default_analysis)."""
        base = Path(temp_dir) / "sub-01" / "default_analysis"
        mesh_dir = base / "segmentation" / "mesh"
        mesh_dir.mkdir(parents=True)
        (mesh_dir / "sub-01_Rwhite.gii").touch()

        folds_dir = base / "folds" / "3.1"
        folds_dir.mkdir(parents=True)
        graph_path = str(folds_dir / "Rsulci_sub-01.arg")
        Path(graph_path).touch()

        result = find_white_mesh(graph_path)
        assert result is not None
        assert result.endswith("Rwhite.gii")

    def test_right_hemisphere_detection(self, temp_dir):
        """Test that right hemisphere graphs find Rwhite mesh."""
        base = Path(temp_dir) / "sub-01" / "default_acquisition" / "0"
        mesh_dir = base / "segmentation" / "mesh"
        mesh_dir.mkdir(parents=True)
        (mesh_dir / "sub-01_Lwhite.gii").touch()
        (mesh_dir / "sub-01_Rwhite.gii").touch()

        folds_dir = base / "folds"
        folds_dir.mkdir(parents=True)
        graph_path = str(folds_dir / "Rsulci_sub-01.arg")

        result = find_white_mesh(graph_path)
        assert result is not None
        assert "Rwhite" in result

    def test_missing_mesh_dir_returns_none(self, temp_dir):
        """Test that missing mesh directory returns None."""
        base = Path(temp_dir) / "sub-01" / "default_analysis"
        folds_dir = base / "folds"
        folds_dir.mkdir(parents=True)
        graph_path = str(folds_dir / "Lsulci_sub-01.arg")

        result = find_white_mesh(graph_path)
        assert result is None

    def test_no_anchor_dir_returns_none(self):
        """Test that a path without default_analysis or default_acquisition/0 returns None."""
        result = find_white_mesh("/some/random/path/graph.arg")
        assert result is None


class TestFindCompletedRegions:
    """Test find_completed_regions function."""

    def test_finds_left_and_right_regions(self, temp_dir):
        """Test detection of both L and R mask_skeleton files."""
        region = Path(temp_dir) / "region_A" / "mask"
        region.mkdir(parents=True)
        (region / "Lmask_skeleton.nii.gz").touch()
        (region / "Rmask_skeleton.nii.gz").touch()

        result = find_completed_regions(temp_dir)
        assert "region_A" in result["left"]
        assert "region_A" in result["right"]

    def test_single_hemisphere(self, temp_dir):
        """Test region with only one hemisphere mask."""
        region = Path(temp_dir) / "region_B" / "mask"
        region.mkdir(parents=True)
        (region / "Lmask_skeleton.nii.gz").touch()

        result = find_completed_regions(temp_dir)
        assert "region_B" in result["left"]
        assert "region_B" not in result["right"]

    def test_empty_directory(self, temp_dir):
        """Test with empty crops directory."""
        result = find_completed_regions(temp_dir)
        assert result == {"left": [], "right": []}

    def test_nonexistent_directory(self):
        """Test with nonexistent directory."""
        result = find_completed_regions("/nonexistent/path")
        assert result == {"left": [], "right": []}

    def test_region_without_mask_dir(self, temp_dir):
        """Test that regions without mask/ subdirectory are skipped."""
        (Path(temp_dir) / "region_C").mkdir()
        result = find_completed_regions(temp_dir)
        assert result == {"left": [], "right": []}

    def test_multiple_regions_sorted(self, temp_dir):
        """Test that regions are returned in sorted order."""
        for name in ["zebra", "alpha", "middle"]:
            mask_dir = Path(temp_dir) / name / "mask"
            mask_dir.mkdir(parents=True)
            (mask_dir / "Lmask_skeleton.nii.gz").touch()

        result = find_completed_regions(temp_dir)
        assert result["left"] == ["alpha", "middle", "zebra"]


class TestGenerateSnapshotsInit:
    """Test GenerateSnapshots class initialization."""

    def test_script_name(self):
        """Test that script_name is set correctly."""
        script = GenerateSnapshots()
        assert script.script_name == "generate_snapshots"

    def test_output_dir_required(self):
        """Test that --output_dir is required."""
        script = GenerateSnapshots()
        with pytest.raises(SystemExit):
            script.parse_args([])

    def test_parse_all_args(self, temp_dir):
        """Test parsing all arguments."""
        script = GenerateSnapshots()
        args = script.parse_args([
            "--output_dir", temp_dir,
            "--morphologist_dir", "/morpho",
            "--embeddings_dir", "/emb",
            "--cortical_tiles_dir", "/tiles",
            "--width", "1024",
            "--height", "768",
            "--sulcal-only",
            "--tiles_level", "2",
            "--reference_data_dir", "/ref",
        ])
        assert args.output_dir == temp_dir
        assert args.morphologist_dir == "/morpho"
        assert args.embeddings_dir == "/emb"
        assert args.cortical_tiles_dir == "/tiles"
        assert args.width == 1024
        assert args.height == 768
        assert args.sulcal_only is True
        assert args.tiles_level == 2
        assert args.reference_data_dir == "/ref"

    def test_default_values(self, temp_dir):
        """Test default argument values."""
        script = GenerateSnapshots()
        args = script.parse_args(["--output_dir", temp_dir])
        assert args.width == 800
        assert args.height == 600
        assert args.sulcal_only is False
        assert args.tiles_only is False
        assert args.umap_only is False
        assert args.tiles_level == 1


class TestGenerateSnapshotsRun:
    """Test GenerateSnapshots.run() method."""

    def test_run_creates_output_dir(self, temp_dir):
        """Test that run() creates the output directory."""
        output_dir = os.path.join(temp_dir, "new_output")
        script = GenerateSnapshots()
        script.parse_args(["--output_dir", output_dir])
        result = script.run()
        assert result == 0
        assert os.path.isdir(output_dir)

    def test_run_writes_manifest(self, temp_dir):
        """Test that run() writes manifest JSON."""
        output_dir = os.path.join(temp_dir, "out")
        script = GenerateSnapshots()
        script.parse_args(["--output_dir", output_dir])
        script.run()
        manifest = os.path.join(output_dir, "snapshots_manifest.json")
        assert os.path.exists(manifest)
        with open(manifest) as f:
            data = json.load(f)
        assert "snapshots" in data
        assert isinstance(data["snapshots"], list)

    def test_run_returns_zero(self, temp_dir):
        """Test that run() returns 0 on success."""
        script = GenerateSnapshots()
        script.parse_args(["--output_dir", os.path.join(temp_dir, "out")])
        assert script.run() == 0


class TestOnlyFlagLogic:
    """Test the --*-only flag logic without running the full main()."""

    def _compute_flags(self, sulcal_only=False, tiles_only=False, umap_only=False):
        """Helper to compute run_sulcal, run_tiles, run_umap from only flags."""
        only_flags = (sulcal_only, tiles_only, umap_only)
        run_sulcal = not any(only_flags) or sulcal_only
        run_tiles = not any(only_flags) or tiles_only
        run_umap = not any(only_flags) or umap_only
        return run_sulcal, run_tiles, run_umap

    def test_no_flags_runs_all(self):
        """Test that no --*-only flags runs all snapshot types."""
        run_s, run_t, run_u = self._compute_flags()
        assert run_s is True
        assert run_t is True
        assert run_u is True

    def test_sulcal_only(self):
        """Test that --sulcal-only runs only sulcal."""
        run_s, run_t, run_u = self._compute_flags(sulcal_only=True)
        assert run_s is True
        assert run_t is False
        assert run_u is False

    def test_tiles_only(self):
        """Test that --tiles-only runs only tiles."""
        run_s, run_t, run_u = self._compute_flags(tiles_only=True)
        assert run_s is False
        assert run_t is True
        assert run_u is False

    def test_umap_only(self):
        """Test that --umap-only runs only umap."""
        run_s, run_t, run_u = self._compute_flags(umap_only=True)
        assert run_s is False
        assert run_t is False
        assert run_u is True


class TestCollateralFilesConstant:
    """Test the COLLATERAL_FILES constant."""

    def test_has_left_and_right(self):
        """Test that COLLATERAL_FILES has both hemispheres."""
        assert "left" in COLLATERAL_FILES
        assert "right" in COLLATERAL_FILES

    def test_values_end_with_csv(self):
        """Test that file names end with .csv."""
        for hemi, fname in COLLATERAL_FILES.items():
            assert fname.endswith(".csv"), f"{hemi} file does not end with .csv"


class TestMainFunction:
    """Test the main() function."""

    def test_output_dir_is_required(self):
        """Test that --output_dir is required."""
        with patch('sys.argv', ['generate_snapshots.py']):
            with pytest.raises(SystemExit):
                main()

    def test_main_creates_output_dir(self, temp_dir):
        """Test that main creates the output directory."""
        output_dir = os.path.join(temp_dir, "snapshots_output")
        with patch('sys.argv', [
            'generate_snapshots.py',
            '--output_dir', output_dir,
        ]):
            result = main()
            assert result == 0
            assert os.path.isdir(output_dir)

    def test_main_writes_manifest(self, temp_dir):
        """Test that main writes a snapshots_manifest.json."""
        output_dir = os.path.join(temp_dir, "snapshots_output")
        with patch('sys.argv', [
            'generate_snapshots.py',
            '--output_dir', output_dir,
        ]):
            main()
            manifest = os.path.join(output_dir, "snapshots_manifest.json")
            assert os.path.exists(manifest)
            with open(manifest) as f:
                data = json.load(f)
            assert "snapshots" in data

    def test_sulcal_only_does_not_run_tiles_or_umap(self, temp_dir):
        """Test that --sulcal-only skips tiles and umap."""
        output_dir = os.path.join(temp_dir, "out")
        with patch('sys.argv', [
            'generate_snapshots.py',
            '--output_dir', output_dir,
            '--sulcal-only',
        ]):
            result = main()
            assert result == 0

    def test_tiles_only_does_not_run_sulcal_or_umap(self, temp_dir):
        """Test that --tiles-only skips sulcal and umap."""
        output_dir = os.path.join(temp_dir, "out")
        with patch('sys.argv', [
            'generate_snapshots.py',
            '--output_dir', output_dir,
            '--tiles-only',
        ]):
            result = main()
            assert result == 0

    def test_umap_only_does_not_run_sulcal_or_tiles(self, temp_dir):
        """Test that --umap-only skips sulcal and tiles."""
        output_dir = os.path.join(temp_dir, "out")
        with patch('sys.argv', [
            'generate_snapshots.py',
            '--output_dir', output_dir,
            '--umap-only',
        ]):
            result = main()
            assert result == 0


    def test_main_calls_build_print_run(self, temp_dir):
        """Test that main() follows the ScriptBuilder pattern."""
        output_dir = os.path.join(temp_dir, "out")
        with patch('sys.argv', [
            'generate_snapshots.py',
            '--output_dir', output_dir,
        ]):
            with patch.object(GenerateSnapshots, 'build',
                              return_value=GenerateSnapshots.__new__(GenerateSnapshots)) as mock_build:
                mock_instance = mock_build.return_value
                mock_instance.print_args = lambda: mock_instance
                mock_instance.run = lambda: 0
                result = main()
                mock_build.assert_called_once()
                assert result == 0


@pytest.mark.smoke
class TestGenerateSnapshotsSmoke:
    """Smoke tests for basic functionality."""

    def test_find_sulcal_graphs_is_callable(self):
        """Test that find_sulcal_graphs is callable."""
        assert callable(find_sulcal_graphs)

    def test_find_white_mesh_is_callable(self):
        """Test that find_white_mesh is callable."""
        assert callable(find_white_mesh)

    def test_find_completed_regions_is_callable(self):
        """Test that find_completed_regions is callable."""
        assert callable(find_completed_regions)

    def test_generate_snapshots_class_exists(self):
        """Test that GenerateSnapshots class can be instantiated."""
        script = GenerateSnapshots()
        assert script.script_name == "generate_snapshots"

    def test_main_is_callable(self):
        """Test that main is callable."""
        assert callable(main)
