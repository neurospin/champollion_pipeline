#!/usr/bin/env python3
"""
Unit tests for generate_snapshots.py
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from champollion_pipeline.generate_snapshots import (
    GenerateSnapshots,
    discover_umap_pairs,
    find_completed_regions,
    find_sulcal_graphs,
    find_white_mesh,
    main,
)
from champollion_pipeline.utils.qc_report import read_qc_tsv


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
        """Test finding mesh in Morphologist 6.0 layout."""
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
        """Test that a path without known anchors returns None."""
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
        args = script.parse_args(
            [
                "--output_dir",
                temp_dir,
                "--morphologist_dir",
                "/morpho",
                "--subject",
                "sub_0001",
                "--acquisition",
                "wk30",
                "--embeddings_dir",
                "/emb",
                "--cortical_tiles_dir",
                "/tiles",
                "--width",
                "1024",
                "--height",
                "768",
                "--sulcal-only",
                "--tiles_level",
                "2",
                "--reference_data_dir",
                "/ref",
                "--umap_region",
                "FColl-SRh,S.Or.",
            ]
        )
        assert args.output_dir == temp_dir
        assert args.morphologist_dir == "/morpho"
        assert args.subject == "sub_0001"
        assert args.acquisition == "wk30"
        assert args.embeddings_dir == "/emb"
        assert args.cortical_tiles_dir == "/tiles"
        assert args.width == 1024
        assert args.height == 768
        assert args.sulcal_only is True
        assert args.tiles_level == 2
        assert args.reference_data_dir == "/ref"
        assert args.umap_region == "FColl-SRh,S.Or."

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
        """Helper to compute run_sulcal, run_tiles, run_umap from flags."""
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


class TestDiscoverUmapPairs:
    """Test the discover_umap_pairs function."""

    def test_returns_empty_when_no_csvs(self, temp_dir):
        """Test that an empty embeddings dir returns no pairs."""
        result = discover_umap_pairs(temp_dir, temp_dir)
        assert result == []

    def test_returns_pair_when_model_exists(self, temp_dir):
        """Test that a matching CSV + model pair is returned."""
        emb_dir = Path(temp_dir) / "embeddings"
        ref_dir = Path(temp_dir) / "reference"
        emb_dir.mkdir()
        ref_dir.mkdir()

        (emb_dir / "FColl-SRh_left_name01_embeddings.csv").touch()
        (ref_dir / "umap_FColl-SRh_left.pkl").touch()
        (ref_dir / "umap_FColl-SRh_left_coords.npy").touch()

        result = discover_umap_pairs(str(emb_dir), str(ref_dir))
        assert len(result) == 1
        _, _, _, region, hemi = result[0]
        assert region == "FColl-SRh"
        assert hemi == "left"

    def test_skips_csv_without_model(self, temp_dir):
        """Test that a CSV without a matching model is skipped."""
        emb_dir = Path(temp_dir) / "embeddings"
        ref_dir = Path(temp_dir) / "reference"
        emb_dir.mkdir()
        ref_dir.mkdir()

        (emb_dir / "S.Or._left_name01_embeddings.csv").touch()
        # No model files created

        result = discover_umap_pairs(str(emb_dir), str(ref_dir))
        assert result == []

    def test_region_filter(self, temp_dir):
        """Test that the regions filter excludes unwanted regions."""
        emb_dir = Path(temp_dir) / "embeddings"
        ref_dir = Path(temp_dir) / "reference"
        emb_dir.mkdir()
        ref_dir.mkdir()

        for region in ("FColl-SRh", "S.Or."):
            (emb_dir / f"{region}_left_name01_embeddings.csv").touch()
            (ref_dir / f"umap_{region}_left.pkl").touch()
            (ref_dir / f"umap_{region}_left_coords.npy").touch()

        result = discover_umap_pairs(str(emb_dir), str(ref_dir), regions=["FColl-SRh"])
        assert len(result) == 1
        assert result[0][3] == "FColl-SRh"

    def test_ignores_files_with_bad_hemi(self, temp_dir):
        """Test that CSVs whose second token is not left/right are ignored."""
        emb_dir = Path(temp_dir) / "embeddings"
        ref_dir = Path(temp_dir) / "reference"
        emb_dir.mkdir()
        ref_dir.mkdir()

        (emb_dir / "FColl-SRh_both_name01_embeddings.csv").touch()

        result = discover_umap_pairs(str(emb_dir), str(ref_dir))
        assert result == []


class TestMainFunction:
    """Test the main() function."""

    def test_output_dir_is_required(self):
        """Test that --output_dir is required."""
        with patch("sys.argv", ["generate_snapshots.py"]), pytest.raises(SystemExit):
            main()

    def test_main_creates_output_dir(self, temp_dir):
        """Test that main creates the output directory."""
        output_dir = os.path.join(temp_dir, "snapshots_output")
        with patch(
            "sys.argv",
            [
                "generate_snapshots.py",
                "--output_dir",
                output_dir,
            ],
        ):
            result = main()
            assert result == 0
            assert os.path.isdir(output_dir)

    def test_main_writes_manifest(self, temp_dir):
        """Test that main writes a snapshots_manifest.json."""
        output_dir = os.path.join(temp_dir, "snapshots_output")
        with patch(
            "sys.argv",
            [
                "generate_snapshots.py",
                "--output_dir",
                output_dir,
            ],
        ):
            main()
            manifest = os.path.join(output_dir, "snapshots_manifest.json")
            assert os.path.exists(manifest)
            with open(manifest) as f:
                data = json.load(f)
            assert "snapshots" in data

    def test_sulcal_only_does_not_run_tiles_or_umap(self, temp_dir):
        """Test that --sulcal-only skips tiles and umap."""
        output_dir = os.path.join(temp_dir, "out")
        with patch(
            "sys.argv",
            [
                "generate_snapshots.py",
                "--output_dir",
                output_dir,
                "--sulcal-only",
            ],
        ):
            result = main()
            assert result == 0

    def test_tiles_only_does_not_run_sulcal_or_umap(self, temp_dir):
        """Test that --tiles-only skips sulcal and umap."""
        output_dir = os.path.join(temp_dir, "out")
        with patch(
            "sys.argv",
            [
                "generate_snapshots.py",
                "--output_dir",
                output_dir,
                "--tiles-only",
            ],
        ):
            result = main()
            assert result == 0

    def test_umap_only_does_not_run_sulcal_or_tiles(self, temp_dir):
        """Test that --umap-only skips sulcal and tiles."""
        output_dir = os.path.join(temp_dir, "out")
        with patch(
            "sys.argv",
            [
                "generate_snapshots.py",
                "--output_dir",
                output_dir,
                "--umap-only",
            ],
        ):
            result = main()
            assert result == 0

    def test_main_inits_then_returns_script_main(self):
        """main() calls init_pipeline_process(), then returns the script's ScriptBuilder.main() result."""
        calls = []
        with patch(
            "champollion_pipeline.generate_snapshots.init_pipeline_process", side_effect=lambda: calls.append("init")
        ):
            script_main = {"return_value.main.side_effect": lambda: calls.append("main") or 3}
            with patch("champollion_pipeline.generate_snapshots.GenerateSnapshots", **script_main) as MockScript:
                result = main()

        MockScript.assert_called_once_with()
        MockScript.return_value.main.assert_called_once_with()
        assert calls == ["init", "main"]
        assert result == 3


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


# --- Stage 6 input QC (TASK-NTUI-BDRABCZUK-8C1AE0C0AD63) ---

QC_FILE = "snapshots_input_qc.tsv"
_RENDER_SULCAL = "champollion_pipeline.generate_snapshots.generate_sulcal_graph_snapshot"
_RENDER_TILES = "champollion_pipeline.generate_snapshots.generate_tiles_snapshot"
_LOAD_UMAP_MODEL = "champollion_pipeline.snapshot_types.umap.joblib.load"


def _morphologist_subject(morph_dir, subject, graphs=("L", "R"), meshes=("L", "R")):
    """Create one Morphologist subject with the given sulcal graph and white mesh sides."""
    analysis = Path(morph_dir) / subject / "t1mri" / "default_acquisition" / "0"
    (analysis / "folds" / "3.1").mkdir(parents=True)
    (analysis / "segmentation" / "mesh").mkdir(parents=True)
    for side in graphs:
        (analysis / "folds" / "3.1" / f"{side}sulci_{subject}.arg").touch()
    for side in meshes:
        (analysis / "segmentation" / "mesh" / f"{subject}_{side}white.gii").touch()


def _embeddings_csv(embeddings_dir, region, hemi, rows):
    """Write {region}_{hemi}_embeddings.csv with an ID column and two dimensions."""
    Path(embeddings_dir).mkdir(parents=True, exist_ok=True)
    lines = ["ID,dim1,dim2", *(f"{subject},{d1},{d2}" for subject, d1, d2 in rows)]
    (Path(embeddings_dir) / f"{region}_{hemi}_embeddings.csv").write_text("\n".join(lines) + "\n")


def _umap_reference(reference_dir, region, hemi, model=True, coords=True):
    """Create the UMAP model and reference coordinates files of one region and hemisphere."""
    Path(reference_dir).mkdir(parents=True, exist_ok=True)
    if model:
        (Path(reference_dir) / f"umap_{region}_{hemi}.pkl").touch()
    if coords:
        np.save(Path(reference_dir) / f"umap_{region}_{hemi}_coords.npy", np.zeros((3, 2)))


class _RecordingUmapModel:
    """UMAP model stand-in that records each array passed to transform."""

    def __init__(self, on_transform=None):
        self.inputs = []
        self.on_transform = on_transform

    def transform(self, values):
        if self.on_transform is not None:
            self.on_transform()
        self.inputs.append(np.array(values))
        return np.zeros((len(values), 2))


def _run_snapshots(args):
    """Parse args and run GenerateSnapshots, returning its exit value."""
    script = GenerateSnapshots()
    script.parse_args(args)
    return script.run()


def _qc_rows(output_dir):
    """Read snapshots_input_qc.tsv of output_dir, keyed by participant_id."""
    return {row["participant_id"]: row for row in read_qc_tsv(os.path.join(output_dir, QC_FILE))}


def _lines(captured):
    """Return the stdout then stderr lines of a pytest capture."""
    return (captured.out + captured.err).splitlines()


class TestSnapshotsInputQC:
    """Stage 6 input QC: per-subject QC file, exclusions, and up-front missing-input warnings."""

    def test_qc_file_rows_written_before_rendering(self, temp_dir):
        """REQ C7139FD8D38A: one row per sulcal subject and embeddings ID, written before rendering."""
        morph = os.path.join(temp_dir, "morphologist")
        _morphologist_subject(morph, "sub-01")
        _morphologist_subject(morph, "sub-02")
        embeddings = os.path.join(temp_dir, "embeddings")
        _embeddings_csv(embeddings, "S.Or.", "left", [("sub-02", 0.1, 0.2), ("sub-03", 0.3, 0.4)])
        reference = os.path.join(temp_dir, "reference")
        _umap_reference(reference, "S.Or.", "left")
        output_dir = os.path.join(temp_dir, "out")
        qc_present_at_render = []

        def render(graph_path, out, *args, **kwargs):
            qc_present_at_render.append(os.path.exists(os.path.join(output_dir, QC_FILE)))
            return out

        with patch(_RENDER_SULCAL, side_effect=render), patch(_LOAD_UMAP_MODEL, return_value=_RecordingUmapModel()):
            _run_snapshots(
                [
                    "--output_dir",
                    output_dir,
                    "--morphologist_dir",
                    morph,
                    "--embeddings_dir",
                    embeddings,
                    "--reference_data_dir",
                    reference,
                ]
            )

        assert set(_qc_rows(output_dir)) == {"sub-01", "sub-02", "sub-03"}
        assert qc_present_at_render and all(qc_present_at_render)

    def test_missing_right_graph_excludes_subject(self, temp_dir):
        """REQ 02280180FCF2: a subject with no right sulcal graph gets qc 0, reason missing_sulcal_graph_right."""
        morph = os.path.join(temp_dir, "morphologist")
        _morphologist_subject(morph, "sub-01", graphs=("L",), meshes=("L",))
        _morphologist_subject(morph, "sub-02")
        output_dir = os.path.join(temp_dir, "out")

        with patch(_RENDER_SULCAL, side_effect=lambda graph_path, out, *a, **k: out):
            _run_snapshots(["--output_dir", output_dir, "--morphologist_dir", morph, "--sulcal-only"])

        rows = _qc_rows(output_dir)
        assert (rows["sub-01"]["qc"], rows["sub-01"]["reason"]) == (0, "missing_sulcal_graph_right")
        assert (rows["sub-02"]["qc"], rows["sub-02"]["reason"]) == (1, "")

    def test_missing_white_mesh_column_keeps_subject(self, temp_dir):
        """REQ C191487CA1E0: a missing right white mesh is listed in missing_white_mesh, qc stays 1."""
        morph = os.path.join(temp_dir, "morphologist")
        _morphologist_subject(morph, "sub-01", meshes=("L",))
        output_dir = os.path.join(temp_dir, "out")

        with patch(_RENDER_SULCAL, side_effect=lambda graph_path, out, *a, **k: out):
            _run_snapshots(["--output_dir", output_dir, "--morphologist_dir", morph, "--sulcal-only"])

        row = _qc_rows(output_dir)["sub-01"]
        assert (row["qc"], row.get("missing_white_mesh")) == (1, "right")

    def test_subject_missing_from_one_csv_is_excluded(self, temp_dir):
        """REQ 912F3EB96130: no row in one selected embeddings CSV gives missing_embedding_<region>_<hemi>."""
        embeddings = os.path.join(temp_dir, "embeddings")
        _embeddings_csv(embeddings, "S.Or.", "left", [("sub-01", 0.1, 0.2), ("sub-02", 0.3, 0.4)])
        _embeddings_csv(embeddings, "S.Or.", "right", [("sub-01", 0.5, 0.6)])
        reference = os.path.join(temp_dir, "reference")
        _umap_reference(reference, "S.Or.", "left")
        _umap_reference(reference, "S.Or.", "right")
        output_dir = os.path.join(temp_dir, "out")

        with patch(_LOAD_UMAP_MODEL, side_effect=lambda path: _RecordingUmapModel()):
            _run_snapshots(
                [
                    "--output_dir",
                    output_dir,
                    "--embeddings_dir",
                    embeddings,
                    "--reference_data_dir",
                    reference,
                    "--umap-only",
                ]
            )

        rows = _qc_rows(output_dir)
        assert (rows["sub-02"]["qc"], rows["sub-02"]["reason"]) == (0, "missing_embedding_S.Or._right")
        assert (rows["sub-01"]["qc"], rows["sub-01"]["reason"]) == (1, "")

    def test_nan_embedding_row_excludes_subject(self, temp_dir):
        """REQ D68BA15849DA: an empty embeddings cell gives qc 0, reason nonfinite_embedding_<region>_<hemi>."""
        embeddings = os.path.join(temp_dir, "embeddings")
        _embeddings_csv(embeddings, "S.Or.", "left", [("sub-01", 0.1, 0.2), ("sub-02", "", 0.4)])
        reference = os.path.join(temp_dir, "reference")
        _umap_reference(reference, "S.Or.", "left")
        output_dir = os.path.join(temp_dir, "out")

        with patch(_LOAD_UMAP_MODEL, return_value=_RecordingUmapModel()):
            _run_snapshots(
                [
                    "--output_dir",
                    output_dir,
                    "--embeddings_dir",
                    embeddings,
                    "--reference_data_dir",
                    reference,
                    "--umap-only",
                ]
            )

        rows = _qc_rows(output_dir)
        assert (rows["sub-02"]["qc"], rows["sub-02"]["reason"]) == (0, "nonfinite_embedding_S.Or._left")

    def test_several_reasons_joined_by_semicolons(self, temp_dir):
        """REQ 6FC91D82C46C: a subject failing two checks gets both reasons joined by semicolons."""
        morph = os.path.join(temp_dir, "morphologist")
        (Path(morph) / "sub-01" / "t1mri").mkdir(parents=True)
        output_dir = os.path.join(temp_dir, "out")

        with patch(_RENDER_SULCAL, side_effect=lambda graph_path, out, *a, **k: out):
            _run_snapshots(["--output_dir", output_dir, "--morphologist_dir", morph, "--sulcal-only"])

        row = _qc_rows(output_dir)["sub-01"]
        assert row["qc"] == 0
        assert set(row["reason"].split(";")) == {"missing_sulcal_graph_left", "missing_sulcal_graph_right"}

    def test_excluded_subjects_named_in_one_warning_and_exit_zero(self, temp_dir, capsys):
        """REQ 100D13B8E6A5: excluded subjects are named on one warning line and the run returns 0."""
        morph = os.path.join(temp_dir, "morphologist")
        _morphologist_subject(morph, "sub-01")
        _morphologist_subject(morph, "sub-02", graphs=("R",), meshes=("R",))
        _morphologist_subject(morph, "sub-03", graphs=("L",), meshes=("L",))
        output_dir = os.path.join(temp_dir, "out")

        with patch(_RENDER_SULCAL, side_effect=lambda graph_path, out, *a, **k: out):
            result = _run_snapshots(["--output_dir", output_dir, "--morphologist_dir", morph, "--sulcal-only"])

        naming_lines = [line for line in _lines(capsys.readouterr()) if "sub-02" in line and "sub-03" in line]
        assert result == 0
        assert len(naming_lines) == 1
        assert os.path.exists(os.path.join(output_dir, QC_FILE))

    def test_sulcal_step_skips_excluded_subject(self, temp_dir):
        """REQ BE44D168C384: the sulcal step renders no graph of a subject with qc 0."""
        morph = os.path.join(temp_dir, "morphologist")
        _morphologist_subject(morph, "sub-01", graphs=("L",), meshes=("L",))
        output_dir = os.path.join(temp_dir, "out")

        with patch(_RENDER_SULCAL, side_effect=lambda graph_path, out, *a, **k: out) as render:
            _run_snapshots(
                ["--output_dir", output_dir, "--morphologist_dir", morph, "--subject", "sub-01", "--sulcal-only"]
            )

        assert _qc_rows(output_dir)["sub-01"]["qc"] == 0
        render.assert_not_called()

    def test_umap_projects_only_kept_rows(self, temp_dir):
        """REQ 61D89615C8AF: the UMAP model receives only the embeddings rows of qc 1 subjects."""
        embeddings = os.path.join(temp_dir, "embeddings")
        _embeddings_csv(embeddings, "S.Or.", "left", [("sub-01", 0.1, 0.2), ("sub-02", "", 0.4)])
        reference = os.path.join(temp_dir, "reference")
        _umap_reference(reference, "S.Or.", "left")
        output_dir = os.path.join(temp_dir, "out")
        model = _RecordingUmapModel()

        with patch(_LOAD_UMAP_MODEL, return_value=model):
            _run_snapshots(
                [
                    "--output_dir",
                    output_dir,
                    "--embeddings_dir",
                    embeddings,
                    "--reference_data_dir",
                    reference,
                    "--umap-only",
                ]
            )

        assert len(model.inputs) == 1
        np.testing.assert_allclose(model.inputs[0], np.array([[0.1, 0.2]]), rtol=1e-6)

    def test_missing_umap_reference_files_in_one_warning(self, temp_dir, capsys):
        """REQ 08776847068C: missing UMAP model and coords files are named on one line before any projection."""
        embeddings = os.path.join(temp_dir, "embeddings")
        for hemi in ("left", "right"):
            _embeddings_csv(embeddings, "S.Or.", hemi, [("sub-01", 0.1, 0.2)])
        _embeddings_csv(embeddings, "S.Pe.C.", "left", [("sub-01", 0.1, 0.2)])
        reference = os.path.join(temp_dir, "reference")
        _umap_reference(reference, "S.Or.", "left", coords=False)
        _umap_reference(reference, "S.Pe.C.", "left")
        output_dir = os.path.join(temp_dir, "out")
        before_projection = []
        model = _RecordingUmapModel(on_transform=lambda: before_projection.extend(_lines(capsys.readouterr())))

        with patch(_LOAD_UMAP_MODEL, return_value=model):
            _run_snapshots(
                [
                    "--output_dir",
                    output_dir,
                    "--embeddings_dir",
                    embeddings,
                    "--reference_data_dir",
                    reference,
                    "--umap-only",
                ]
            )

        missing = ("umap_S.Or._left_coords.npy", "umap_S.Or._right.pkl", "umap_S.Or._right_coords.npy")
        assert model.inputs, "the complete S.Pe.C. left pair was not projected"
        assert len([line for line in before_projection if all(name in line for name in missing)]) == 1

    def test_crop_region_without_mask_in_one_warning(self, temp_dir, capsys):
        """REQ 1EDE1833EB60: crop regions with no mask skeleton are named on one line before the tiles render."""
        crops = Path(temp_dir) / "cortical_tiles" / "crops" / "2mm"
        (crops / "RegionA" / "mask").mkdir(parents=True)
        (crops / "RegionB" / "mask").mkdir(parents=True)
        (crops / "RegionB" / "mask" / "Lmask_skeleton.nii.gz").touch()
        (crops / "RegionC").mkdir(parents=True)
        output_dir = os.path.join(temp_dir, "out")
        before_render = []

        def render(*args, **kwargs):
            before_render.extend(_lines(capsys.readouterr()))
            return []

        with patch(_RENDER_TILES, side_effect=render):
            _run_snapshots(
                ["--output_dir", output_dir, "--cortical_tiles_dir", str(crops.parent.parent), "--tiles-only"]
            )

        naming = [line for line in before_render if "RegionA" in line and "RegionC" in line]
        assert len(naming) == 1
        assert "RegionB" not in naming[0]

    def test_missing_region_graph_in_one_warning(self, temp_dir, capsys):
        """REQ 5EC4D9061B22: a missing region graph of a completed hemisphere is named before the render."""
        crops = Path(temp_dir) / "cortical_tiles" / "crops" / "2mm"
        (crops / "RegionB" / "mask").mkdir(parents=True)
        (crops / "RegionB" / "mask" / "Lmask_skeleton.nii.gz").touch()
        data_root = Path(temp_dir) / "champollion_data"
        (data_root / "mask" / "2mm" / "regions" / "meshes").mkdir(parents=True)
        output_dir = os.path.join(temp_dir, "out")
        before_render = []

        def render(*args, **kwargs):
            before_render.extend(_lines(capsys.readouterr()))
            return []

        with patch(_RENDER_TILES, side_effect=render):
            _run_snapshots(
                [
                    "--output_dir",
                    output_dir,
                    "--cortical_tiles_dir",
                    str(crops.parent.parent),
                    "--champollion_data_root",
                    str(data_root),
                    "--tiles-only",
                ]
            )

        assert len([line for line in before_render if "Lregions_model_1.arg" in line]) == 1
        assert not [line for line in before_render if "Rregions_model_1.arg" in line]
