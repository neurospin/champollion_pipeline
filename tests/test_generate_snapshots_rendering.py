#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the rendering helpers and run-step dispatch of
generate_snapshots.py.

Anatomist, PyAIMS and deep_folding are injected as mocks through sys.modules,
so no BrainVISA install, X display or GPU is needed.  The UMAP plots are
rendered for real through matplotlib's Agg backend using tiny synthetic
coordinate arrays and a stubbed UMAP model.
"""

import os
import os.path as osp
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from champollion_pipeline import generate_snapshots as gs
from champollion_pipeline.generate_snapshots import (
    GenerateSnapshots,
    _detect_hemi,
    _parse_embedding_csv_name,
    find_completed_regions,
    find_sulcal_graphs,
    find_white_mesh,
    generate_sulcal_graph_snapshot,
    generate_tiles_snapshot,
    generate_umap_snapshot,
    list_acquisitions,
)


def anatomist_modules(anatomist_instance):
    """sys.modules entries providing a mocked anatomist.headless."""
    headless = MagicMock()
    headless.Anatomist.return_value = anatomist_instance
    package = MagicMock()
    package.headless = headless
    return {"anatomist": package, "anatomist.headless": headless}


def make_script(argv):
    """Return a GenerateSnapshots with parsed arguments."""
    script = GenerateSnapshots()
    script.parse_args(argv)
    return script


class TestListAcquisitions:
    """Test list_acquisitions."""

    def test_returns_empty_when_no_t1mri_dir(self, tmp_path):
        assert list_acquisitions(str(tmp_path)) == []

    def test_lists_acquisitions_containing_graphs(self, tmp_path):
        for acq in ("wk30", "wk40"):
            d = tmp_path / "t1mri" / acq / "default_analysis" / "folds"
            d.mkdir(parents=True)
            (d / "Lsulci_sub01.arg").touch()
        assert list_acquisitions(str(tmp_path)) == ["wk30", "wk40"]

    def test_acquisitions_without_graphs_are_excluded(self, tmp_path):
        good = tmp_path / "t1mri" / "wk30" / "folds"
        good.mkdir(parents=True)
        (good / "Lsulci_sub01.arg").touch()
        (tmp_path / "t1mri" / "wk40").mkdir(parents=True)
        assert list_acquisitions(str(tmp_path)) == ["wk30"]

    def test_unrelated_arg_files_do_not_count(self, tmp_path):
        # The acquisition only counts when a graph file is named for a sulcal
        # or folds graph; any other .arg product must be ignored.
        d = tmp_path / "t1mri" / "wk30"
        d.mkdir(parents=True)
        (d / "other.arg").touch()
        assert list_acquisitions(str(tmp_path)) == []

    def test_files_directly_under_t1mri_are_skipped(self, tmp_path):
        (tmp_path / "t1mri").mkdir()
        (tmp_path / "t1mri" / "stray.txt").write_text("x")
        assert list_acquisitions(str(tmp_path)) == []

    def test_result_is_sorted(self, tmp_path):
        for acq in ("zz", "aa"):
            d = tmp_path / "t1mri" / acq
            d.mkdir(parents=True)
            (d / "Lfolds.arg").touch()
        assert list_acquisitions(str(tmp_path)) == ["aa", "zz"]


class TestFindSulcalGraphsSubjectLookup:
    """Test the subject / acquisition filtering branches of find_sulcal_graphs."""

    def test_subject_found_via_recursive_search(self, tmp_path, capsys):
        nested = tmp_path / "cohort" / "sub_0001" / "folds"
        nested.mkdir(parents=True)
        (nested / "Lsulci.arg").touch()
        graphs = find_sulcal_graphs(str(tmp_path), subject="sub_0001")
        assert len(graphs) == 1
        assert "Subject found at:" in capsys.readouterr().out

    def test_missing_subject_returns_empty_list(self, tmp_path, capsys):
        graphs = find_sulcal_graphs(str(tmp_path), subject="sub_9999")
        assert graphs == []
        assert "Subject directory not found" in capsys.readouterr().out

    def test_acquisition_filter_keeps_only_matching_graphs(self, tmp_path):
        for acq in ("wk30", "wk40"):
            d = tmp_path / "sub_0001" / "t1mri" / acq / "folds"
            d.mkdir(parents=True)
            (d / "Lsulci.arg").touch()
        graphs = find_sulcal_graphs(str(tmp_path), subject="sub_0001", acquisition="wk30")
        assert len(graphs) == 1
        assert "/wk30/" in graphs[0]


class TestFindWhiteMesh:
    """Test find_white_mesh anchor resolution."""

    def test_returns_none_without_a_known_anchor(self, tmp_path):
        assert find_white_mesh(str(tmp_path / "a" / "b" / "Lsulci.arg")) is None

    def test_default_acquisition_zero_anchor_is_accepted(self, tmp_path):
        analysis = tmp_path / "t1mri" / "default_acquisition" / "0"
        mesh_dir = analysis / "segmentation" / "mesh"
        mesh_dir.mkdir(parents=True)
        (mesh_dir / "sub01_Lwhite.gii").touch()
        graph = analysis / "folds" / "Lsulci.arg"
        graph.parent.mkdir(parents=True)
        graph.touch()
        assert find_white_mesh(str(graph)).endswith("sub01_Lwhite.gii")

    def test_bare_zero_component_is_not_an_anchor(self, tmp_path):
        graph = tmp_path / "0" / "folds" / "Lsulci.arg"
        graph.parent.mkdir(parents=True)
        graph.touch()
        assert find_white_mesh(str(graph)) is None

    def test_returns_none_when_mesh_dir_missing(self, tmp_path):
        graph = tmp_path / "default_analysis" / "folds" / "Lsulci.arg"
        graph.parent.mkdir(parents=True)
        graph.touch()
        assert find_white_mesh(str(graph)) is None

    def test_right_hemisphere_mesh_is_selected(self, tmp_path):
        analysis = tmp_path / "default_analysis"
        mesh_dir = analysis / "segmentation" / "mesh"
        mesh_dir.mkdir(parents=True)
        (mesh_dir / "sub01_Rwhite.gii").touch()
        graph = analysis / "folds" / "Rsulci.arg"
        graph.parent.mkdir(parents=True)
        graph.touch()
        assert find_white_mesh(str(graph)).endswith("_Rwhite.gii")

    def test_returns_none_when_no_matching_mesh_file(self, tmp_path):
        analysis = tmp_path / "default_analysis"
        (analysis / "segmentation" / "mesh").mkdir(parents=True)
        graph = analysis / "folds" / "Lsulci.arg"
        graph.parent.mkdir(parents=True)
        graph.touch()
        assert find_white_mesh(str(graph)) is None


class TestFindCompletedRegions:
    """Test find_completed_regions."""

    def test_non_directory_entries_are_skipped(self, tmp_path):
        (tmp_path / "stray.txt").write_text("x")
        assert find_completed_regions(str(tmp_path)) == {"left": [], "right": []}

    def test_region_without_mask_dir_is_skipped(self, tmp_path):
        (tmp_path / "S.Or.").mkdir()
        assert find_completed_regions(str(tmp_path)) == {"left": [], "right": []}

    def test_both_hemispheres_are_detected(self, tmp_path):
        mask = tmp_path / "S.Or." / "mask"
        mask.mkdir(parents=True)
        (mask / "Lmask_skeleton.nii.gz").touch()
        (mask / "Rmask_skeleton.nii.gz").touch()
        assert find_completed_regions(str(tmp_path)) == {"left": ["S.Or."], "right": ["S.Or."]}


class TestDetectHemi:
    """Test _detect_hemi."""

    @pytest.mark.parametrize("name", ["Rsulci.arg", "sub_Rwhite.arg", "right_folds.arg"])
    def test_right_is_detected(self, name):
        assert _detect_hemi(f"/a/b/{name}") == "right"

    def test_left_is_the_default(self):
        assert _detect_hemi("/a/b/Lsulci.arg") == "left"


class TestParseEmbeddingCsvName:
    """Test _parse_embedding_csv_name."""

    def test_non_embedding_filename_is_rejected(self):
        assert _parse_embedding_csv_name("results.csv") == (None, None)

    def test_old_naming_convention(self):
        assert _parse_embedding_csv_name("SOr_left_123_embeddings.csv") == ("SOr", "left")

    def test_new_naming_convention(self):
        region, hemi = _parse_embedding_csv_name("_FColl--right--modelA_embeddings.csv")
        assert (region, hemi) == ("FColl", "right")

    def test_unparseable_name_returns_none(self):
        assert _parse_embedding_csv_name("weird_embeddings.csv") == (None, None)


class TestGenerateSulcalGraphSnapshot:
    """Test generate_sulcal_graph_snapshot with a mocked Anatomist."""

    def _anatomist(self):
        a = MagicMock()
        a.createWindow.return_value = MagicMock()
        return a

    def test_returns_output_path_and_saves_image(self, tmp_path):
        a = self._anatomist()
        out = tmp_path / "snap.png"
        with patch.dict(sys.modules, anatomist_modules(a)):
            result = generate_sulcal_graph_snapshot("/graph.arg", str(out))
        assert result == str(out)
        a.createWindow.return_value.snapshotImage.return_value.save.assert_called_once_with(str(out))

    def test_default_quaternion_is_the_left_side_view(self, tmp_path):
        a = self._anatomist()
        with patch.dict(sys.modules, anatomist_modules(a)):
            generate_sulcal_graph_snapshot("/graph.arg", str(tmp_path / "s.png"))
        win = a.createWindow.return_value
        assert win.camera.call_args.kwargs["view_quaternion"] == (0.5, 0.5, 0.5, 0.5)

    def test_explicit_quaternion_is_used(self, tmp_path):
        a = self._anatomist()
        with patch.dict(sys.modules, anatomist_modules(a)):
            generate_sulcal_graph_snapshot(
                "/graph.arg", str(tmp_path / "s.png"), view_quaternion=(1, 0, 0, 0)
            )
        assert a.createWindow.return_value.camera.call_args.kwargs["view_quaternion"] == (1, 0, 0, 0)

    def test_existing_mesh_is_added_as_transparent_surface(self, tmp_path):
        a = self._anatomist()
        mesh = tmp_path / "Lwhite.gii"
        mesh.touch()
        with patch.dict(sys.modules, anatomist_modules(a)):
            generate_sulcal_graph_snapshot(
                "/graph.arg", str(tmp_path / "s.png"), mesh_path=str(mesh)
            )
        assert a.loadObject.call_count == 2
        a.loadObject.return_value.setMaterial.assert_called_with(diffuse=[0.8, 0.8, 0.8, 0.37])

    def test_missing_mesh_is_ignored(self, tmp_path):
        a = self._anatomist()
        with patch.dict(sys.modules, anatomist_modules(a)):
            generate_sulcal_graph_snapshot(
                "/graph.arg", str(tmp_path / "s.png"), mesh_path=str(tmp_path / "nope.gii")
            )
        assert a.loadObject.call_count == 1

    def test_existing_anatomist_instance_is_reused(self, tmp_path):
        a = self._anatomist()
        modules = anatomist_modules(MagicMock())
        with patch.dict(sys.modules, modules):
            generate_sulcal_graph_snapshot("/graph.arg", str(tmp_path / "s.png"), a=a)
        modules["anatomist.headless"].Anatomist.assert_not_called()
        a.createWindow.assert_called_once()

    def test_objects_and_windows_are_released(self, tmp_path):
        a = self._anatomist()
        with patch.dict(sys.modules, anatomist_modules(a)):
            generate_sulcal_graph_snapshot("/graph.arg", str(tmp_path / "s.png"))
        a.removeObjects.assert_called_once()
        a.closeWindows.assert_called_once()
        a.deleteObjects.assert_called_once()


class TestGenerateTilesSnapshot:
    """Test generate_tiles_snapshot with mocked Anatomist / PyAIMS / deep_folding."""

    @pytest.fixture
    def env(self, tmp_path):
        """Return (modules, anatomist_mock, data_root) with graphs and meshes on disk."""
        a = MagicMock()
        a.createWindow.return_value = MagicMock()

        data_root = tmp_path / "champollion_data"
        meshes = data_root / "mask" / "2mm" / "regions" / "meshes"
        meshes.mkdir(parents=True)
        (meshes / "Lregions_model_1.arg").touch()
        (meshes / "Rregions_model_1.arg").touch()

        icbm = tmp_path / "icbm"
        icbm.mkdir()
        (icbm / "mni_icbm152_nlin_asym_09c_Lhemi.gii").touch()
        (icbm / "mni_icbm152_nlin_asym_09c_Rhemi.gii").touch()

        aims = MagicMock()
        aims.carto.Paths.findResourceFile.side_effect = lambda *args, **kwargs: (
            str(icbm) if "icbm152" in args[0] else "/nomenclature.hie"
        )

        config_mod = MagicMock()
        config_mod.config.return_value.get_champollion_data_root_dir.return_value = str(data_root)
        cortical_tiles = MagicMock()
        cortical_tiles.config = config_mod

        modules = anatomist_modules(a)
        modules.update(
            {
                "cortical_tiles": cortical_tiles,
                "cortical_tiles.config": config_mod,
                "soma": MagicMock(aims=aims),
                "soma.aims": aims,
            }
        )
        return modules, a, data_root, icbm

    def _crops(self, tmp_path, hemis=("L", "R"), region="S.Or."):
        crops = tmp_path / "crops"
        mask = crops / region / "mask"
        mask.mkdir(parents=True)
        for prefix in hemis:
            (mask / f"{prefix}mask_skeleton.nii.gz").touch()
        return crops

    def test_no_completed_regions_returns_empty(self, tmp_path, env, capsys):
        modules, _, _, _ = env
        crops = tmp_path / "empty_crops"
        crops.mkdir()
        with patch.dict(sys.modules, modules):
            assert generate_tiles_snapshot(str(crops), str(tmp_path / "t.png")) == []
        assert "No completed regions found" in capsys.readouterr().out

    def test_one_snapshot_per_hemisphere(self, tmp_path, env):
        modules, _, data_root, _ = env
        crops = self._crops(tmp_path)
        with patch.dict(sys.modules, modules):
            snaps = generate_tiles_snapshot(
                str(crops), str(tmp_path / "tiles.png"), champollion_data_root=str(data_root)
            )
        assert snaps == [str(tmp_path / "tiles_left.png"), str(tmp_path / "tiles_right.png")]

    def test_single_hemisphere_produces_one_snapshot(self, tmp_path, env):
        modules, _, data_root, _ = env
        crops = self._crops(tmp_path, hemis=("L",))
        with patch.dict(sys.modules, modules):
            snaps = generate_tiles_snapshot(
                str(crops), str(tmp_path / "tiles.png"), champollion_data_root=str(data_root)
            )
        assert snaps == [str(tmp_path / "tiles_left.png")]

    def test_data_root_falls_back_to_cortical_tiles_config(self, tmp_path, env):
        modules, _, data_root, _ = env
        crops = self._crops(tmp_path)
        with patch.dict(sys.modules, modules):
            snaps = generate_tiles_snapshot(str(crops), str(tmp_path / "tiles.png"))
        assert len(snaps) == 2
        modules["cortical_tiles.config"].config.assert_called()

    def test_missing_region_graph_is_reported_and_skipped(self, tmp_path, env, capsys):
        modules, _, data_root, _ = env
        crops = self._crops(tmp_path)
        with patch.dict(sys.modules, modules):
            snaps = generate_tiles_snapshot(
                str(crops), str(tmp_path / "tiles.png"), level=3, champollion_data_root=str(data_root)
            )
        assert snaps == []
        assert "Region graph not found" in capsys.readouterr().out

    def test_missing_icbm_mesh_is_reported_and_skipped(self, tmp_path, env, capsys):
        modules, _, data_root, icbm = env
        for gii in icbm.glob("*.gii"):
            gii.unlink()
        crops = self._crops(tmp_path)
        with patch.dict(sys.modules, modules):
            snaps = generate_tiles_snapshot(
                str(crops), str(tmp_path / "tiles.png"), champollion_data_root=str(data_root)
            )
        assert snaps == []
        assert "ICBM mesh not found" in capsys.readouterr().out

    def test_fallback_mesh_dir_used_when_resource_lookup_fails(self, tmp_path, env, capsys):
        modules, _, data_root, _ = env
        modules["soma.aims"].carto.Paths.findResourceFile.side_effect = (
            lambda *args, **kwargs: None if "icbm152" in args[0] else "/nomenclature.hie"
        )
        crops = self._crops(tmp_path)
        with patch.dict(sys.modules, modules):
            generate_tiles_snapshot(
                str(crops), str(tmp_path / "tiles.png"), champollion_data_root=str(data_root)
            )
        assert gs.ICBM_MESH_DIR_FALLBACK in capsys.readouterr().out

    def test_extension_defaults_to_png(self, tmp_path, env):
        modules, _, data_root, _ = env
        crops = self._crops(tmp_path, hemis=("L",))
        with patch.dict(sys.modules, modules):
            snaps = generate_tiles_snapshot(
                str(crops), str(tmp_path / "tiles"), champollion_data_root=str(data_root)
            )
        assert snaps == [str(tmp_path / "tiles_left.png")]

    def test_nomenclature_object_is_released(self, tmp_path, env):
        modules, a, data_root, _ = env
        crops = self._crops(tmp_path, hemis=("L",))
        with patch.dict(sys.modules, modules):
            generate_tiles_snapshot(
                str(crops), str(tmp_path / "tiles.png"), champollion_data_root=str(data_root)
            )
        assert a.deleteObjects.call_args.args[0] == [a.toAObject.return_value]


@pytest.fixture
def umap_inputs(tmp_path):
    """Build an embeddings CSV plus matching UMAP model/coords artefacts."""
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    (embeddings / "SOr_left_001_embeddings.csv").write_text("ID,f0,f1\nsub01,0.1,0.2\n")

    reference = tmp_path / "reference_data"
    reference.mkdir()
    (reference / "umap_SOr_left.pkl").touch()
    np.save(reference / "umap_SOr_left_coords.npy", np.zeros((5, 2), dtype=np.float32))
    return embeddings, reference


class TestGenerateUmapSnapshot:
    """Test generate_umap_snapshot (rendered through matplotlib's Agg backend)."""

    def _model(self):
        model = MagicMock()
        model.transform.return_value = np.array([[0.5, 0.5]], dtype=np.float32)
        return model

    def test_no_pairs_returns_empty(self, tmp_path, capsys):
        embeddings = tmp_path / "embeddings"
        embeddings.mkdir()
        result = generate_umap_snapshot(str(embeddings), str(tmp_path), str(tmp_path / "u.png"))
        assert result == []
        assert "No *_embeddings.csv files found" in capsys.readouterr().out

    def test_plot_is_written_per_region(self, tmp_path, umap_inputs):
        embeddings, reference = umap_inputs
        with patch("joblib.load", return_value=self._model()):
            snaps = generate_umap_snapshot(
                str(embeddings), str(reference), str(tmp_path / "umap.png")
            )
        assert snaps == [str(tmp_path / "umap_SOr_left.png")]
        assert osp.exists(snaps[0])

    def test_extension_defaults_to_png(self, tmp_path, umap_inputs):
        embeddings, reference = umap_inputs
        with patch("joblib.load", return_value=self._model()):
            snaps = generate_umap_snapshot(str(embeddings), str(reference), str(tmp_path / "umap"))
        assert snaps == [str(tmp_path / "umap_SOr_left.png")]

    def test_id_column_is_dropped_before_projection(self, tmp_path, umap_inputs):
        embeddings, reference = umap_inputs
        model = self._model()
        with patch("joblib.load", return_value=model):
            generate_umap_snapshot(str(embeddings), str(reference), str(tmp_path / "umap.png"))
        assert model.transform.call_args.args[0].shape == (1, 2)

    def test_unmatched_pairs_are_reported(self, tmp_path, capsys):
        embeddings = tmp_path / "embeddings"
        embeddings.mkdir()
        (embeddings / "SOr_left_001_embeddings.csv").write_text("ID,f0\nsub01,0.1\n")
        reference = tmp_path / "reference_data"
        reference.mkdir()
        assert generate_umap_snapshot(str(embeddings), str(reference), str(tmp_path / "u.png")) == []
        out = capsys.readouterr().out
        assert "UMAP model not found for SOr left" in out
        assert "No matching (embedding CSV, UMAP model) pairs found" in out

    def test_region_filter_excludes_other_regions(self, tmp_path, umap_inputs):
        embeddings, reference = umap_inputs
        with patch("joblib.load", return_value=self._model()):
            snaps = generate_umap_snapshot(
                str(embeddings), str(reference), str(tmp_path / "u.png"), regions=["Other"]
            )
        assert snaps == []

    def test_unrecognized_filename_is_reported(self, tmp_path, capsys):
        embeddings = tmp_path / "embeddings"
        embeddings.mkdir()
        (embeddings / "weird_embeddings.csv").write_text("ID\n")
        generate_umap_snapshot(str(embeddings), str(tmp_path), str(tmp_path / "u.png"))
        assert "Unrecognized filename format" in capsys.readouterr().out


class TestRunSulcalStep:
    """Test GenerateSnapshots._run_sulcal."""

    def _script(self, tmp_path, extra=None):
        return make_script(["--output_dir", str(tmp_path / "out")] + (extra or []))

    def test_no_morphologist_dir_returns_empty(self, tmp_path):
        assert self._script(tmp_path)._run_sulcal((800, 600)) == []

    def test_missing_morphologist_dir_is_reported(self, tmp_path, capsys):
        script = self._script(tmp_path, ["--morphologist_dir", str(tmp_path / "nope")])
        assert script._run_sulcal((800, 600)) == []
        assert "Morphologist directory not found" in capsys.readouterr().out

    def test_snapshot_is_generated_per_hemisphere(self, tmp_path):
        morpho = tmp_path / "morphologist"
        folds = morpho / "sub_0001" / "t1mri" / "wk30" / "default_analysis" / "folds"
        folds.mkdir(parents=True)
        (folds / "Lsulci_sub_0001.arg").touch()
        (folds / "Rsulci_sub_0001.arg").touch()

        script = self._script(tmp_path, ["--morphologist_dir", str(morpho)])
        with patch.object(gs, "generate_sulcal_graph_snapshot", side_effect=lambda *a, **k: a[1]):
            snaps = script._run_sulcal((800, 600))
        assert len(snaps) == 2
        assert any(s.endswith("sulcal_graph_left.png") for s in snaps)

    def test_acquisition_tag_is_added_to_filename(self, tmp_path):
        morpho = tmp_path / "morphologist"
        folds = morpho / "sub_0001" / "t1mri" / "wk30" / "default_analysis" / "folds"
        folds.mkdir(parents=True)
        (folds / "Lsulci_sub_0001.arg").touch()

        script = self._script(
            tmp_path,
            ["--morphologist_dir", str(morpho), "--acquisition", "wk30"],
        )
        with patch.object(gs, "generate_sulcal_graph_snapshot", side_effect=lambda *a, **k: a[1]):
            snaps = script._run_sulcal((800, 600))
        assert snaps[0].endswith("sulcal_graph_wk30_left.png")

    def test_multiple_acquisitions_trigger_a_warning(self, tmp_path, capsys):
        morpho = tmp_path / "morphologist"
        for acq in ("wk30", "wk40"):
            folds = morpho / "t1mri" / acq / "default_analysis" / "folds"
            folds.mkdir(parents=True)
            (folds / "Lsulci.arg").touch()

        script = self._script(tmp_path, ["--morphologist_dir", str(morpho)])
        with patch.object(gs, "generate_sulcal_graph_snapshot", side_effect=lambda *a, **k: a[1]):
            script._run_sulcal((800, 600))
        out = capsys.readouterr().out
        assert "Warning: 2 left graphs found" in out
        assert "wk30, wk40" in out

    def test_white_mesh_is_reported_when_found(self, tmp_path, capsys):
        morpho = tmp_path / "morphologist"
        analysis = morpho / "sub_0001" / "t1mri" / "wk30" / "default_analysis"
        folds = analysis / "folds"
        folds.mkdir(parents=True)
        (folds / "Lsulci.arg").touch()
        mesh_dir = analysis / "segmentation" / "mesh"
        mesh_dir.mkdir(parents=True)
        (mesh_dir / "sub_0001_Lwhite.gii").touch()

        script = self._script(tmp_path, ["--morphologist_dir", str(morpho)])
        with patch.object(gs, "generate_sulcal_graph_snapshot", side_effect=lambda *a, **k: a[1]):
            script._run_sulcal((800, 600))
        assert "White mesh:" in capsys.readouterr().out

    def test_rendering_error_is_caught(self, tmp_path, capsys):
        morpho = tmp_path / "morphologist"
        folds = morpho / "sub_0001" / "folds"
        folds.mkdir(parents=True)
        (folds / "Lsulci.arg").touch()

        script = self._script(tmp_path, ["--morphologist_dir", str(morpho)])
        with patch.object(
            gs, "generate_sulcal_graph_snapshot", side_effect=RuntimeError("no display")
        ):
            assert script._run_sulcal((800, 600)) == []
        assert "Error processing" in capsys.readouterr().out


class TestRunTilesStep:
    """Test GenerateSnapshots._run_tiles."""

    def _script(self, tmp_path, extra=None):
        return make_script(["--output_dir", str(tmp_path / "out")] + (extra or []))

    def test_no_tiles_dir_returns_empty(self, tmp_path):
        assert self._script(tmp_path)._run_tiles((800, 600)) == []

    def test_missing_tiles_dir_is_reported(self, tmp_path, capsys):
        script = self._script(tmp_path, ["--cortical_tiles_dir", str(tmp_path / "nope")])
        assert script._run_tiles((800, 600)) == []
        assert "Cortical tiles directory not found" in capsys.readouterr().out

    def test_crops_2mm_subdirectory_is_auto_detected(self, tmp_path):
        root = tmp_path / "cortical_tiles"
        (root / "crops" / "2mm").mkdir(parents=True)
        script = self._script(tmp_path, ["--cortical_tiles_dir", str(root)])
        with patch.object(gs, "generate_tiles_snapshot", return_value=["/snap.png"]) as tiles:
            assert script._run_tiles((800, 600)) == ["/snap.png"]
        assert tiles.call_args.args[0] == str(root / "crops" / "2mm")

    def test_direct_crops_path_is_used_as_is(self, tmp_path):
        crops = tmp_path / "crops2mm"
        crops.mkdir()
        script = self._script(tmp_path, ["--cortical_tiles_dir", str(crops)])
        with patch.object(gs, "generate_tiles_snapshot", return_value=[]) as tiles:
            script._run_tiles((800, 600))
        assert tiles.call_args.args[0] == str(crops)

    def test_tiles_level_and_data_root_are_forwarded(self, tmp_path):
        crops = tmp_path / "crops2mm"
        crops.mkdir()
        script = self._script(
            tmp_path,
            [
                "--cortical_tiles_dir",
                str(crops),
                "--tiles_level",
                "3",
                "--champollion_data_root",
                "/my/data",
            ],
        )
        with patch.object(gs, "generate_tiles_snapshot", return_value=[]) as tiles:
            script._run_tiles((800, 600))
        assert tiles.call_args.kwargs["level"] == 3
        assert tiles.call_args.kwargs["champollion_data_root"] == "/my/data"

    def test_missing_anatomist_is_reported(self, tmp_path, capsys):
        crops = tmp_path / "crops2mm"
        crops.mkdir()
        script = self._script(tmp_path, ["--cortical_tiles_dir", str(crops)])
        with patch.object(gs, "generate_tiles_snapshot", side_effect=ImportError("no anatomist")):
            assert script._run_tiles((800, 600)) == []
        assert "Anatomist not available" in capsys.readouterr().out

    def test_other_errors_are_caught(self, tmp_path, capsys):
        crops = tmp_path / "crops2mm"
        crops.mkdir()
        script = self._script(tmp_path, ["--cortical_tiles_dir", str(crops)])
        with patch.object(gs, "generate_tiles_snapshot", side_effect=RuntimeError("boom")):
            assert script._run_tiles((800, 600)) == []
        assert "Error generating tiles snapshot" in capsys.readouterr().out


class TestRunUmapStep:
    """Test GenerateSnapshots._run_umap."""

    def _script(self, tmp_path, extra=None):
        return make_script(["--output_dir", str(tmp_path / "out")] + (extra or []))

    def test_no_embeddings_dir_returns_empty(self, tmp_path):
        assert self._script(tmp_path)._run_umap((800, 600)) == []

    def test_missing_embeddings_dir_is_reported(self, tmp_path, capsys):
        script = self._script(tmp_path, ["--embeddings_dir", str(tmp_path / "nope")])
        assert script._run_umap((800, 600)) == []
        assert "Embeddings directory not found" in capsys.readouterr().out

    def test_no_reference_data_dir_returns_empty(self, tmp_path):
        embeddings = tmp_path / "embeddings"
        embeddings.mkdir()
        script = self._script(tmp_path, ["--embeddings_dir", str(embeddings)])
        assert script._run_umap((800, 600)) == []

    def test_missing_reference_data_dir_is_reported(self, tmp_path, capsys):
        embeddings = tmp_path / "embeddings"
        embeddings.mkdir()
        script = self._script(
            tmp_path,
            ["--embeddings_dir", str(embeddings), "--reference_data_dir", str(tmp_path / "nope")],
        )
        assert script._run_umap((800, 600)) == []
        assert "Reference data directory not found" in capsys.readouterr().out

    def test_snapshots_are_returned(self, tmp_path):
        embeddings = tmp_path / "embeddings"
        embeddings.mkdir()
        reference = tmp_path / "reference"
        reference.mkdir()
        script = self._script(
            tmp_path,
            ["--embeddings_dir", str(embeddings), "--reference_data_dir", str(reference)],
        )
        with patch.object(gs, "generate_umap_snapshot", return_value=["/u.png"]):
            assert script._run_umap((800, 600)) == ["/u.png"]

    def test_region_filter_is_parsed_and_forwarded(self, tmp_path, capsys):
        embeddings = tmp_path / "embeddings"
        embeddings.mkdir()
        reference = tmp_path / "reference"
        reference.mkdir()
        script = self._script(
            tmp_path,
            [
                "--embeddings_dir",
                str(embeddings),
                "--reference_data_dir",
                str(reference),
                "--umap_region",
                "FColl-SRh, S.Or.",
            ],
        )
        with patch.object(gs, "generate_umap_snapshot", return_value=[]) as umap:
            script._run_umap((800, 600))
        assert umap.call_args.kwargs["regions"] == ["FColl-SRh", "S.Or."]
        assert "UMAP region filter: FColl-SRh, S.Or." in capsys.readouterr().out

    def test_errors_are_caught(self, tmp_path, capsys):
        embeddings = tmp_path / "embeddings"
        embeddings.mkdir()
        reference = tmp_path / "reference"
        reference.mkdir()
        script = self._script(
            tmp_path,
            ["--embeddings_dir", str(embeddings), "--reference_data_dir", str(reference)],
        )
        with patch.object(gs, "generate_umap_snapshot", side_effect=RuntimeError("boom")):
            assert script._run_umap((800, 600)) == []
        assert "Error generating UMAP snapshot" in capsys.readouterr().out


class TestRunDispatch:
    """Test GenerateSnapshots.run() step selection and manifest."""

    def _script(self, tmp_path, extra=None):
        return make_script(["--output_dir", str(tmp_path / "out")] + (extra or []))

    def _patched(self, script):
        return (
            patch.object(script, "_run_sulcal", return_value=["s.png"]),
            patch.object(script, "_run_tiles", return_value=["t.png"]),
            patch.object(script, "_run_umap", return_value=["u.png"]),
        )

    def test_all_steps_run_by_default(self, tmp_path):
        script = self._script(tmp_path)
        sulcal, tiles, umap = self._patched(script)
        with sulcal as s, tiles as t, umap as u:
            assert script.run() == 0
        s.assert_called_once()
        t.assert_called_once()
        u.assert_called_once()

    @pytest.mark.parametrize(
        "flag,expected",
        [("--sulcal-only", "sulcal"), ("--tiles-only", "tiles"), ("--umap-only", "umap")],
    )
    def test_only_flags_restrict_the_steps(self, tmp_path, flag, expected):
        script = self._script(tmp_path, [flag])
        sulcal, tiles, umap = self._patched(script)
        with sulcal as s, tiles as t, umap as u:
            script.run()
        called = {"sulcal": s.called, "tiles": t.called, "umap": u.called}
        assert called[expected] is True
        assert sum(called.values()) == 1

    def test_output_directory_is_created(self, tmp_path):
        script = self._script(tmp_path)
        sulcal, tiles, umap = self._patched(script)
        with sulcal, tiles, umap:
            script.run()
        assert (tmp_path / "out").is_dir()

    def test_manifest_lists_every_snapshot(self, tmp_path):
        import json

        script = self._script(tmp_path)
        sulcal, tiles, umap = self._patched(script)
        with sulcal, tiles, umap:
            script.run()
        manifest = json.loads((tmp_path / "out" / "snapshots_manifest.json").read_text())
        assert manifest == {"snapshots": ["s.png", "t.png", "u.png"]}

    def test_snapshot_count_is_reported(self, tmp_path, capsys):
        script = self._script(tmp_path)
        sulcal, tiles, umap = self._patched(script)
        with sulcal, tiles, umap:
            script.run()
        assert "Generated 3 snapshot(s)" in capsys.readouterr().out

    def test_custom_size_is_forwarded_to_each_step(self, tmp_path):
        script = self._script(tmp_path, ["--width", "1024", "--height", "768"])
        sulcal, tiles, umap = self._patched(script)
        with sulcal as s, tiles, umap:
            script.run()
        assert s.call_args.args[0] == (1024, 768)


def test_module_exposes_icbm_fallback_path():
    """The hard-coded ICBM mesh fallback must stay an absolute neurospin path."""
    assert os.path.isabs(gs.ICBM_MESH_DIR_FALLBACK)
    assert gs.ICBM_MESH_DIR_FALLBACK.endswith("segmentation/mesh")
