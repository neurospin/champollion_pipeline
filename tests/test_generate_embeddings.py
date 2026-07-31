#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for generate_embeddings.py
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings, HuggingFaceStrategy


class TestGenerateEmbeddingsInit:
    """Test initialization."""

    def test_init_creates_script(self):
        """Test that script is initialized correctly."""
        script = GenerateEmbeddings()
        assert script.script_name == "generate_embeddings"
        assert "embeddings" in script.description.lower()


class TestGenerateEmbeddingsArguments:
    """Test argument parsing."""

    def test_parse_required_arguments(self):
        """Test parsing required positional arguments."""
        script = GenerateEmbeddings()
        args = script.parse_args([
            "/models",
            "local",
            "/datasets",
            "test_run"
        ])
        assert args.models_path == "/models"
        assert args.dataset_localization == "local"
        assert args.datasets_root == "/datasets"
        assert args.short_name == "test_run"

    def test_datasets_default(self):
        """Test that datasets has default value."""
        script = GenerateEmbeddings()
        args = script.parse_args(["/m", "loc", "/d", "name"])
        assert args.datasets == ["toto"]

    def test_labels_default(self):
        """Test that labels has default value."""
        script = GenerateEmbeddings()
        args = script.parse_args(["/m", "loc", "/d", "name"])
        assert args.labels == ["Sex"]

    def test_classifier_name_default(self):
        """Test that classifier_name has default value."""
        script = GenerateEmbeddings()
        args = script.parse_args(["/m", "loc", "/d", "name"])
        assert args.classifier_name == "svm"

    def test_cv_default(self):
        """Test that cv has default value."""
        script = GenerateEmbeddings()
        args = script.parse_args(["/m", "loc", "/d", "name"])
        assert args.cv == 5

    def test_flags_default_to_false(self):
        """Test that boolean flags default to False."""
        script = GenerateEmbeddings()
        args = script.parse_args(["/m", "loc", "/d", "name"])
        assert args.overwrite is False
        assert args.embeddings_only is False
        assert args.use_best_model is False
        assert args.verbose is False
        assert args.cpu is False
        assert args.profiling is False
        assert args.run_cka is False
        assert args.no_cache is False

    def test_flags_can_be_set(self):
        """Test that flags can be set to True."""
        script = GenerateEmbeddings()
        args = script.parse_args([
            "/m", "loc", "/d", "name",
            "--overwrite",
            "--embeddings_only",
            "--use_best_model",
            "--verbose",
            "--cpu",
            "--profiling",
            "--run-cka",
            "--no-cache"
        ])
        assert args.overwrite is True
        assert args.embeddings_only is True
        assert args.use_best_model is True
        assert args.verbose is True
        assert args.cpu is True
        assert args.profiling is True
        assert args.run_cka is True
        assert args.no_cache is True

    def test_nb_jobs_default_none(self):
        """Test that nb_jobs defaults to None."""
        script = GenerateEmbeddings()
        args = script.parse_args(["/m", "loc", "/d", "name"])
        assert args.nb_jobs is None

    def test_nb_jobs_can_be_set(self):
        """Test that --nb_jobs can be set."""
        script = GenerateEmbeddings()
        args = script.parse_args(["/m", "loc", "/d", "name", "--nb_jobs", "8"])
        assert args.nb_jobs == 8

    def test_list_arguments(self):
        """Test list arguments with multiple values."""
        script = GenerateEmbeddings()
        args = script.parse_args([
            "/m", "loc", "/d", "name",
            "--datasets", "ds1", "ds2", "ds3",
            "--labels", "Age", "Gender",
            "--subsets", "train", "test",
            "--epochs", "10", "20", "30"
        ])
        assert args.datasets == ["ds1", "ds2", "ds3"]
        assert args.labels == ["Age", "Gender"]
        assert args.subsets == ["train", "test"]
        assert args.epochs == ["10", "20", "30"]


class TestBuildCommand:
    """Test build_command usage."""

    def test_build_command_called_with_correct_args(self, temp_dir):
        """Test that build_command is called correctly."""
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, "loc", temp_dir, "test"])

        with patch('os.chdir'):
            with patch('os.getcwd', return_value="/original"):
                with patch.object(script, 'build_command', return_value=["python", "script.py"]) as mock_build:
                    with patch.object(script, 'execute_command', return_value=0):
                        script.run()

                        mock_build.assert_called_once()
                        call_kwargs = mock_build.call_args[1]

                        assert call_kwargs['script_path'] == "evaluation/embeddings_pipeline.py"
                        assert set(call_kwargs['required_args']) == {
                            "models_path", "dataset_localization", "datasets_root", "short_name"
                        }
                        assert 'defaults' in call_kwargs


class TestRunMethod:
    """Test the run method."""

    def test_run_changes_to_champollion_directory(self, temp_dir):
        """Test that run changes to champollion directory."""
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, "loc", temp_dir, "test"])

        with patch('os.chdir') as mock_chdir:
            with patch('os.getcwd', return_value="/original"):
                with patch.object(script, 'build_command', return_value=["cmd"]):
                    with patch.object(script, 'execute_command', return_value=0):
                        script.run()

                        # Should change to champollion and back
                        assert any("champollion" in str(call) for call in mock_chdir.call_args_list)

    def test_run_executes_command_without_shell(self, temp_dir):
        """Test that command is executed with shell=False."""
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, "loc", temp_dir, "test"])

        with patch('os.chdir'):
            with patch('os.getcwd', return_value="/original"):
                with patch.object(script, 'build_command', return_value=["cmd"]):
                    with patch.object(script, 'execute_command', return_value=0) as mock_exec:
                        script.run()

                        assert mock_exec.call_args[1]['shell'] is False

    def test_run_returns_result(self, temp_dir):
        """Test that run returns command result."""
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, "loc", temp_dir, "test"])

        with patch('os.chdir'):
            with patch('os.getcwd', return_value="/original"):
                with patch.object(script, 'build_command', return_value=["cmd"]):
                    with patch.object(script, 'execute_command', return_value=99):
                        result = script.run()
                        assert result == 99

    def test_run_restores_directory(self, temp_dir):
        """Test that original directory is restored."""
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, "loc", temp_dir, "test"])
        original = "/original/dir"

        with patch('os.getcwd', return_value=original):
            with patch('os.chdir') as mock_chdir:
                with patch.object(script, 'build_command', return_value=["cmd"]):
                    with patch.object(script, 'execute_command', return_value=0):
                        script.run()

                        from unittest.mock import call
                        assert call(original) in mock_chdir.call_args_list


@pytest.mark.integration
class TestGenerateEmbeddingsIntegration:
    """Integration tests."""

    def test_full_workflow(self, temp_dir):
        """Test complete workflow."""
        script = GenerateEmbeddings()
        script.parse_args([
            temp_dir, "local", temp_dir, "test_embeddings",
            "--datasets", "ds1", "ds2",
            "--labels", "Age", "Sex",
            "--overwrite"
        ])

        with patch.object(script, 'execute_command', return_value=0) as mock_exec:
            with patch('os.chdir'):
                with patch('os.getcwd', return_value="/original"):
                    result = script.run()

                    assert result == 0
                    mock_exec.assert_called_once()


class TestCorticalVersionFlags:
    """Tests for --cortical_version and --legacy flags."""

    def test_cortical_version_default(self):
        from champollion_pipeline.utils.lib import CORTICAL_TILES_VERSION
        script = GenerateEmbeddings()
        args = script.parse_args(["/m", "loc", "/d", "name"])
        assert args.cortical_version == f"cortical_tiles-{CORTICAL_TILES_VERSION}"

    def test_cortical_version_can_be_overridden(self):
        script = GenerateEmbeddings()
        args = script.parse_args(["/m", "loc", "/d", "name", "--cortical_version", "deep_folding-2025"])
        assert args.cortical_version == "deep_folding-2025"

    def test_legacy_flag_defaults_false(self):
        script = GenerateEmbeddings()
        args = script.parse_args(["/m", "loc", "/d", "name"])
        assert args.legacy is False

    def test_legacy_flag_can_be_set(self):
        script = GenerateEmbeddings()
        args = script.parse_args(["/m", "loc", "/d", "name", "--legacy"])
        assert args.legacy is True

    def test_get_derivatives_folder_returns_cortical_version_by_default(self):
        from champollion_pipeline.utils.lib import CORTICAL_TILES_VERSION
        script = GenerateEmbeddings()
        script.args = script.parse_args(["/m", "loc", "/d", "name"])
        assert script._get_derivatives_folder() == f"cortical_tiles-{CORTICAL_TILES_VERSION}"

    def test_get_derivatives_folder_legacy_overrides(self):
        script = GenerateEmbeddings()
        script.args = script.parse_args(["/m", "loc", "/d", "name", "--legacy"])
        assert script._get_derivatives_folder() == "deep_folding-2025"

    def test_get_derivatives_folder_cortical_version_overrides(self):
        script = GenerateEmbeddings()
        script.args = script.parse_args(["/m", "loc", "/d", "name", "--cortical_version", "cortical_tiles-2027"])
        assert script._get_derivatives_folder() == "cortical_tiles-2027"

    def test_patch_config_paths_rewrites_derivatives_folder(self, tmp_path):
        yaml_file = tmp_path / "SC-sylv_left.yaml"
        yaml_file.write_text(
            "numpy_all: /data/derivatives/deep_folding-2025/crops/2mm/S.C.-sylv./mask/Lskeleton.npy\n"
        )

        script = GenerateEmbeddings()
        script._patch_config_paths(str(tmp_path), "cortical_tiles-2026")

        content = yaml_file.read_text()
        assert "cortical_tiles-2026" in content
        assert "deep_folding-2025" not in content

    def test_patch_config_paths_leaves_unrelated_lines_intact(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        original = (
            "numpy_all: /data/derivatives/deep_folding-2025/crops/2mm/region/mask/L.npy\n"
            "dataset_folder: /some/unrelated/path\n"
        )
        yaml_file.write_text(original)

        script = GenerateEmbeddings()
        script._patch_config_paths(str(tmp_path), "cortical_tiles-2026")

        content = yaml_file.read_text()
        assert "dataset_folder: /some/unrelated/path" in content

    def test_patch_config_paths_skips_already_correct_files(self, tmp_path):
        from champollion_pipeline.utils.lib import CORTICAL_TILES_VERSION
        yaml_file = tmp_path / "config.yaml"
        original = f"numpy_all: /data/derivatives/cortical_tiles-{CORTICAL_TILES_VERSION}/crops/2mm/L.npy\n"
        yaml_file.write_text(original)
        mtime_before = yaml_file.stat().st_mtime

        script = GenerateEmbeddings()
        script._patch_config_paths(str(tmp_path), f"cortical_tiles-{CORTICAL_TILES_VERSION}")

        assert yaml_file.stat().st_mtime == mtime_before

    def test_patch_config_paths_recurses_into_subdirs(self, tmp_path):
        subdir = tmp_path / "dataset" / "mydata"
        subdir.mkdir(parents=True)
        yaml_file = subdir / "region.yaml"
        yaml_file.write_text("path: /x/derivatives/old_folder/crops/2mm/r.npy\n")

        script = GenerateEmbeddings()
        script._patch_config_paths(str(tmp_path), "new_folder")

        assert "new_folder" in yaml_file.read_text()


class TestRegionsFilter:
    """Tests for --regions filtering."""

    def test_regions_default_none(self):
        script = GenerateEmbeddings()
        args = script.parse_args(["/m", "loc", "/d", "name"])
        assert args.regions is None

    def test_regions_single(self):
        script = GenerateEmbeddings()
        args = script.parse_args(["/m", "loc", "/d", "name", "--regions", "SC-sylv_left"])
        assert args.regions == ["SC-sylv_left"]

    def test_regions_multiple(self):
        script = GenerateEmbeddings()
        args = script.parse_args([
            "/m", "loc", "/d", "name",
            "--regions", "SC-sylv_left", "SC-sylv_right", "FIP-FIPPoCinf_left"
        ])
        assert args.regions == ["SC-sylv_left", "SC-sylv_right", "FIP-FIPPoCinf_left"]

    def test_make_regions_tmpdir_creates_symlinks(self, tmp_path):
        (tmp_path / "SC-sylv_left").mkdir()
        (tmp_path / "SC-sylv_right").mkdir()
        (tmp_path / "other_region").mkdir()

        script = GenerateEmbeddings()
        script.args = script.parse_args([
            str(tmp_path), "loc", "/d", "name",
            "--regions", "SC-sylv_left", "SC-sylv_right"
        ])

        tmpdir = script._make_regions_tmpdir(str(tmp_path))
        try:
            entries = sorted(os.listdir(tmpdir))
            assert entries == ["SC-sylv_left", "SC-sylv_right"]
            assert os.path.islink(os.path.join(tmpdir, "SC-sylv_left"))
            assert "other_region" not in entries
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_make_regions_tmpdir_raises_on_missing_region(self, tmp_path):
        (tmp_path / "SC-sylv_left").mkdir()

        script = GenerateEmbeddings()
        script.args = script.parse_args([
            str(tmp_path), "loc", "/d", "name",
            "--regions", "SC-sylv_left", "nonexistent_region"
        ])

        with pytest.raises(FileNotFoundError, match="nonexistent_region"):
            script._make_regions_tmpdir(str(tmp_path))

    def test_run_pipeline_uses_tmpdir_when_regions_set(self, tmp_path):
        (tmp_path / "SC-sylv_left").mkdir()
        (tmp_path / "SC-sylv_right").mkdir()

        script = GenerateEmbeddings()
        script.parse_args([
            str(tmp_path), "loc", str(tmp_path), "name",
            "--regions", "SC-sylv_left"
        ])

        # Capture tmpdir state during execute_command, before cleanup
        snapshot = {}

        def fake_execute(cmd, **kwargs):
            path = script.args.models_path
            snapshot['path'] = path
            snapshot['entries'] = sorted(os.listdir(path))
            snapshot['is_symlink'] = os.path.islink(os.path.join(path, "SC-sylv_left"))
            return 0

        with patch('os.chdir'):
            with patch('os.getcwd', return_value="/original"):
                with patch.object(script, 'fetch_models', return_value=str(tmp_path)):
                    with patch.object(script, 'build_command', return_value=["cmd"]):
                        with patch.object(script, 'execute_command', side_effect=fake_execute):
                            script.run()

        assert snapshot['path'] != str(tmp_path), "should use a tmpdir, not the original path"
        assert snapshot['entries'] == ["SC-sylv_left"], "only requested region should appear"
        assert snapshot['is_symlink'], "region entry should be a symlink"
        assert not os.path.exists(snapshot['path']), "tmpdir should be cleaned up after run"

    def test_run_pipeline_cleans_up_tmpdir_on_error(self, tmp_path):
        (tmp_path / "SC-sylv_left").mkdir()

        script = GenerateEmbeddings()
        script.parse_args([
            str(tmp_path), "loc", str(tmp_path), "name",
            "--regions", "SC-sylv_left"
        ])

        created_tmpdirs = []

        original_make = script._make_regions_tmpdir

        def tracking_make(models_path):
            d = original_make(models_path)
            created_tmpdirs.append(d)
            return d

        with patch('os.chdir'):
            with patch('os.getcwd', return_value="/original"):
                with patch.object(script, 'fetch_models', return_value=str(tmp_path)):
                    with patch.object(script, '_make_regions_tmpdir', side_effect=tracking_make):
                        with patch.object(script, 'build_command', return_value=["cmd"]):
                            with patch.object(script, 'execute_command', side_effect=RuntimeError("boom")):
                                with pytest.raises(RuntimeError):
                                    script.run()

        assert len(created_tmpdirs) == 1
        assert not os.path.exists(created_tmpdirs[0]), "tmpdir must be cleaned up even on error"

    def test_run_pipeline_no_tmpdir_without_regions(self, temp_dir):
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, "loc", temp_dir, "name"])

        with patch('os.chdir'):
            with patch('os.getcwd', return_value="/original"):
                with patch.object(script, 'build_command', return_value=["cmd"]):
                    with patch.object(script, 'execute_command', return_value=0):
                        with patch.object(script, '_make_regions_tmpdir') as mock_make:
                            script.run()
                            mock_make.assert_not_called()


class TestProfiling:
    """Tests for --profiling / cProfile behavior."""

    def test_run_dispatches_to_profiling_when_flag_set(self, temp_dir):
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, "loc", temp_dir, "name", "--profiling"])

        with patch.object(script, '_validate_inputs'):
            with patch.object(script, '_run_with_profiling', return_value=0) as mock_prof:
                script.run()
                mock_prof.assert_called_once()

    def test_run_dispatches_to_normal_without_profiling_flag(self, temp_dir):
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, "loc", temp_dir, "name"])

        with patch.object(script, '_validate_inputs'):
            with patch.object(script, '_run_normal', return_value=0) as mock_normal:
                script.run()
                mock_normal.assert_called_once()

    def test_run_with_profiling_calls_run_normal(self, temp_dir):
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, "loc", temp_dir, "name", "--profiling"])

        with patch('champollion_pipeline.generate_embeddings.cProfile.Profile'):
            with patch('champollion_pipeline.generate_embeddings.pstats.Stats', return_value=MagicMock()):
                with patch.object(script, '_run_normal', return_value=42) as mock_normal:
                    result = script._run_with_profiling()
                    mock_normal.assert_called_once()
                    assert result == 42

    def test_run_with_profiling_dumps_profile_file(self, temp_dir):
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, "loc", temp_dir, "name", "--profiling"])

        mock_stats = MagicMock()

        with patch('champollion_pipeline.generate_embeddings.cProfile.Profile'):
            with patch('champollion_pipeline.generate_embeddings.pstats.Stats', return_value=mock_stats):
                with patch.object(script, '_run_normal', return_value=0):
                    script._run_with_profiling()
                    mock_stats.dump_stats.assert_called_once_with('embeddings_profile.prof')

    def test_run_with_profiling_dumps_on_error(self, temp_dir):
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, "loc", temp_dir, "name", "--profiling"])

        mock_stats = MagicMock()

        with patch('champollion_pipeline.generate_embeddings.cProfile.Profile'):
            with patch('champollion_pipeline.generate_embeddings.pstats.Stats', return_value=mock_stats):
                with patch.object(script, '_run_normal', side_effect=RuntimeError("boom")):
                    with pytest.raises(RuntimeError):
                        script._run_with_profiling()
                    mock_stats.dump_stats.assert_called_once_with('embeddings_profile.prof')


@pytest.mark.smoke
class TestGenerateEmbeddingsSmoke:
    """Smoke tests."""

    def test_script_can_be_instantiated(self):
        """Test that script can be created."""
        script = GenerateEmbeddings()
        assert script is not None

    def test_script_has_run_method(self):
        """Test that script has run method."""
        script = GenerateEmbeddings()
        assert hasattr(script, 'run')
        assert callable(script.run)


@pytest.mark.unit
class TestHuggingFaceStrategy:
    """Unit tests for HuggingFaceStrategy.fetch() — regression for snapshot_download bugs."""

    def _make_strategy(self, subfolder=None):
        return HuggingFaceStrategy(subfolder=subfolder)

    def test_fetch_without_subfolder_calls_snapshot_download_no_allow_patterns(self, tmp_path):
        strategy = self._make_strategy(subfolder=None)
        with patch('huggingface_hub.snapshot_download',
                   return_value=str(tmp_path)) as mock_dl:
            strategy.fetch("neurospin/Champollion_V1", str(tmp_path))
        call_kwargs = mock_dl.call_args[1]
        assert 'subfolder' not in call_kwargs
        assert call_kwargs.get('allow_patterns') is None

    def test_fetch_with_subfolder_uses_allow_patterns(self, tmp_path):
        strategy = self._make_strategy(subfolder="canonical_corrected_26_1")
        with patch('huggingface_hub.snapshot_download',
                   return_value=str(tmp_path)) as mock_dl:
            strategy.fetch("neurospin/Champollion_V1", str(tmp_path))
        call_kwargs = mock_dl.call_args[1]
        assert 'subfolder' not in call_kwargs
        assert call_kwargs['allow_patterns'] == ["canonical_corrected_26_1/*"]

    def test_fetch_never_passes_subfolder_kwarg(self, tmp_path):
        for subfolder in [None, "canonical_25", "canonical_corrected_26_1"]:
            strategy = self._make_strategy(subfolder=subfolder)
            with patch('huggingface_hub.snapshot_download',
                       return_value=str(tmp_path)) as mock_dl:
                strategy.fetch("neurospin/Champollion_V1", str(tmp_path))
            assert 'subfolder' not in mock_dl.call_args[1], \
                f"subfolder kwarg must never be passed (subfolder={subfolder!r})"


@pytest.mark.unit
class TestPixiTaskPaths:
    """Verify pixi task script paths resolve to existing files after relocation."""

    PIXI_TASKS = [
        ("champollion-config", "src/champollion_pipeline/generate_champollion_config.py"),
        ("embeddings", "src/champollion_pipeline/generate_embeddings.py"),
        ("combine", "src/champollion_pipeline/put_together_embeddings.py"),
        ("train", "src/champollion_pipeline/train_champollion.py"),
        ("generate-umap-reference", "src/generate_umap_reference.py"),
    ]

    def test_all_pixi_task_scripts_exist(self):
        import pathlib
        repo_root = pathlib.Path(__file__).parent.parent
        for task_name, rel_path in self.PIXI_TASKS:
            script = repo_root / rel_path
            assert script.exists(), (
                f"pixi task '{task_name}' points to '{rel_path}' which does not exist"
            )
