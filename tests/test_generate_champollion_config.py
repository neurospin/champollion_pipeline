#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for generate_champollion_config.py
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from champollion_pipeline.generate_champollion_config import GenerateChampollionConfig


class TestGenerateChampollionConfigInit:
    """Test initialization of GenerateChampollionConfig."""

    def test_init_creates_script(self):
        """Test that script is initialized correctly."""
        script = GenerateChampollionConfig()
        assert script.script_name == "generate_champollion_config"
        assert "champollion" in script.description.lower()

    def test_init_configures_arguments(self):
        """Test that arguments are configured."""
        script = GenerateChampollionConfig()
        with pytest.raises(SystemExit):
            script.parse_args([])  # Missing required args


class TestGenerateChampollionConfigArguments:
    """Test argument parsing."""

    def test_parse_crop_path_argument(self):
        """Test parsing crop_path argument."""
        script = GenerateChampollionConfig()
        args = script.parse_args(["/path/to/crops", "--dataset", "test_dataset"])
        assert args.crop_path == "/path/to/crops"
        assert args.dataset == "test_dataset"

    def test_dataset_required(self):
        """Test that dataset is required."""
        script = GenerateChampollionConfig()
        with pytest.raises(SystemExit):
            script.parse_args(["/path/to/crops"])

    def test_champollion_loc_has_default(self):
        """Test that champollion_loc has a default value."""
        script = GenerateChampollionConfig()
        with patch('os.getcwd', return_value="/current"):
            args = script.parse_args(["/path/to/crops", "--dataset", "test"])
            assert args.champollion_loc is not None

    def test_external_config_default_none(self):
        """Test that --external-config defaults to None."""
        script = GenerateChampollionConfig()
        args = script.parse_args(["/path/to/crops", "--dataset", "test"])
        assert args.external_config is None

    def test_external_config_can_be_set(self):
        """Test that --external-config can be set."""
        script = GenerateChampollionConfig()
        args = script.parse_args([
            "/path/to/crops", "--dataset", "test",
            "--external-config", "/writable/path/local.yaml"
        ])
        assert args.external_config == "/writable/path/local.yaml"

    def test_external_crops_default_false(self):
        """Test that --external_crops defaults to False."""
        script = GenerateChampollionConfig()
        args = script.parse_args(["/path/to/crops", "--dataset", "test"])
        assert args.external_crops is False

    def test_external_crops_can_be_set(self):
        """Test that --external_crops can be set to True."""
        script = GenerateChampollionConfig()
        args = script.parse_args([
            "/path/to/crops", "--dataset", "test",
            "--external_crops"
        ])
        assert args.external_crops is True

    def test_localization_defaults_to_local(self):
        """Test that --localization defaults to 'local'."""
        script = GenerateChampollionConfig()
        args = script.parse_args(["/path/to/crops", "--dataset", "test"])
        assert args.localization == "local"

    def test_localization_can_be_set(self):
        """Test that --localization can be set to a custom value."""
        script = GenerateChampollionConfig()
        args = script.parse_args([
            "/path/to/crops", "--dataset", "test",
            "--localization", "jean-zay"
        ])
        assert args.localization == "jean-zay"


class TestValidateInputs:
    """Test _validate_inputs method."""

    def test_validate_inputs_valid_path(self, temp_dir):
        """Test validation with valid crop path."""
        script = GenerateChampollionConfig()
        script.parse_args([temp_dir, "--dataset", "test"])
        # Should not raise
        script._validate_inputs()

    def test_validate_inputs_invalid_path(self):
        """Test validation with invalid crop path."""
        script = GenerateChampollionConfig()
        script.parse_args(["/nonexistent/path", "--dataset", "test"])
        with pytest.raises(ValueError, match="does not exist"):
            script._validate_inputs()


class TestWriteLocalizationYaml:
    """Test _write_localization_yaml method."""

    def test_writes_dataset_folder(self, temp_dir):
        """Test that dataset_folder is written correctly."""
        script = GenerateChampollionConfig()
        script.parse_args([temp_dir, "--dataset", "test"])

        dest = Path(temp_dir) / "local.yaml"
        script._write_localization_yaml(str(dest), "/my/dataset/folder")

        content = dest.read_text()
        assert "dataset_folder: /my/dataset/folder" in content
        assert "# @package _global_" in content

    def test_creates_parent_directories(self, temp_dir):
        """Test that missing parent directories are created."""
        script = GenerateChampollionConfig()
        script.parse_args([temp_dir, "--dataset", "test"])

        dest = Path(temp_dir) / "nested" / "dir" / "jean-zay.yaml"
        script._write_localization_yaml(str(dest), "/some/path")

        assert dest.exists()

    def test_overwrites_existing_file(self, temp_dir):
        """Test that an existing file is overwritten."""
        script = GenerateChampollionConfig()
        script.parse_args([temp_dir, "--dataset", "test"])

        dest = Path(temp_dir) / "local.yaml"
        dest.write_text("dataset_folder: /old/path\n")

        script._write_localization_yaml(str(dest), "/new/path")

        content = dest.read_text()
        assert "/new/path" in content
        assert "/old/path" not in content


class TestRunMethod:
    """Test the run method."""

    def test_run_validates_inputs(self, temp_dir):
        """Test that run calls _validate_inputs."""
        script = GenerateChampollionConfig()
        script.parse_args([temp_dir, "--dataset", "test"])

        with patch.object(script, '_validate_inputs') as mock_validate:
            with patch.object(script, 'execute_command', return_value=0):
                with patch.object(script, '_write_localization_yaml'):
                    with patch('champollion_pipeline.generate_champollion_config.find_dataset_folder', return_value="/parent"):
                        with patch('builtins.open', mock_open(read_data="data: TESTXX")):
                            with patch('champollion_pipeline.generate_champollion_config.exists', return_value=True):
                                script.run()
                                mock_validate.assert_called_once()

    def test_run_creates_dataset_directory(self, temp_dir):
        """Test that run creates dataset directory if it doesn't exist."""
        script = GenerateChampollionConfig()
        script.parse_args([temp_dir, "--dataset", "test_dataset"])

        # exists() must return True for crop_path validation, False for dataset_loc check
        with patch('champollion_pipeline.generate_champollion_config.exists', side_effect=[True, False]):
            with patch.object(script, 'execute_command', return_value=0) as mock_exec:
                with patch.object(script, '_write_localization_yaml'):
                    with patch('champollion_pipeline.generate_champollion_config.find_dataset_folder', return_value="/parent"):
                        with patch('builtins.open', mock_open(read_data="data: TESTXX")):
                            script.run()

                            mkdir_calls = [c for c in mock_exec.call_args_list
                                           if "mkdir" in str(c)]
                            assert len(mkdir_calls) > 0

    def test_run_copies_reference_yaml(self, temp_dir):
        """Test that run always copies reference.yaml."""
        script = GenerateChampollionConfig()
        script.parse_args([temp_dir, "--dataset", "test_dataset"])

        with patch('champollion_pipeline.generate_champollion_config.exists', return_value=True):
            with patch.object(script, 'execute_command', return_value=0) as mock_exec:
                with patch.object(script, '_write_localization_yaml'):
                    with patch('champollion_pipeline.generate_champollion_config.find_dataset_folder', return_value="/parent"):
                        with patch('builtins.open', mock_open(read_data="data: TESTXX")):
                            script.run()

                            cp_calls = [c for c in mock_exec.call_args_list
                                        if "cp" in str(c)]
                            assert len(cp_calls) > 0

    def test_run_updates_reference_yaml(self, temp_dir):
        """Test that run updates TESTXX in reference.yaml."""
        script = GenerateChampollionConfig()
        dataset_name = "my_dataset"
        script.parse_args([temp_dir, "--dataset", dataset_name])

        yaml_content = "crop_dir: ${dataset_folder}/TESTXX/crops/2mm/SC-sylv/mask/Lcrops\n"

        with patch('champollion_pipeline.generate_champollion_config.exists', return_value=True):
            with patch.object(script, 'execute_command', return_value=0):
                with patch.object(script, '_write_localization_yaml'):
                    with patch('champollion_pipeline.generate_champollion_config.find_dataset_folder', return_value="/parent"):
                        m = mock_open(read_data=yaml_content)
                        with patch('builtins.open', m):
                            script.run()

                            write_calls = [c for c in m().writelines.call_args_list]
                            if write_calls:
                                written_lines = write_calls[0][0][0]
                                written_content = ''.join(written_lines)
                                assert dataset_name in written_content
                                assert "canonical_25" in written_content
                                assert "TESTXX" not in written_content

    def test_run_calls_write_localization_yaml(self, temp_dir):
        """Test that run calls _write_localization_yaml once."""
        script = GenerateChampollionConfig()
        script.parse_args([temp_dir, "--dataset", "test"])

        with patch('champollion_pipeline.generate_champollion_config.exists', return_value=True):
            with patch.object(script, 'execute_command', return_value=0):
                with patch.object(script, '_write_localization_yaml') as mock_write:
                    with patch('champollion_pipeline.generate_champollion_config.find_dataset_folder', return_value="/parent"):
                        with patch('builtins.open', mock_open(read_data="data: TESTXX")):
                            script.run()
                            mock_write.assert_called_once()

    def test_run_passes_localization_name_to_yaml_path(self, temp_dir):
        """Test that the localization name drives which YAML file is written."""
        script = GenerateChampollionConfig()
        script.parse_args([temp_dir, "--dataset", "test", "--localization", "jean-zay"])

        with patch('champollion_pipeline.generate_champollion_config.exists', return_value=True):
            with patch.object(script, 'execute_command', return_value=0):
                with patch.object(script, '_write_localization_yaml') as mock_write:
                    with patch('champollion_pipeline.generate_champollion_config.find_dataset_folder', return_value="/parent"):
                        with patch('builtins.open', mock_open(read_data="data: TESTXX")):
                            script.run()
                            dest_path = mock_write.call_args[0][0]
                            assert "jean-zay.yaml" in dest_path

    def test_run_returns_zero(self, temp_dir):
        """Test that run returns 0 on success."""
        script = GenerateChampollionConfig()
        script.parse_args([temp_dir, "--dataset", "test"])

        with patch('champollion_pipeline.generate_champollion_config.exists', return_value=True):
            with patch.object(script, 'execute_command', return_value=0):
                with patch.object(script, '_write_localization_yaml'):
                    with patch('champollion_pipeline.generate_champollion_config.find_dataset_folder', return_value="/parent"):
                        with patch('builtins.open', mock_open(read_data="data: TESTXX")):
                            result = script.run()
                            assert result == 0


class TestMainFunction:
    """Test the main entry point."""

    def test_main_creates_script_and_runs(self, temp_dir):
        """Test that main creates script and calls build().print_args().run()."""
        with patch('champollion_pipeline.generate_champollion_config.GenerateChampollionConfig') as MockScript:
            mock_instance = MagicMock()
            mock_instance.build.return_value = mock_instance
            mock_instance.print_args.return_value = mock_instance
            mock_instance.run.return_value = 0
            MockScript.return_value = mock_instance

            from champollion_pipeline.generate_champollion_config import main

            with patch('sys.argv', ['script', temp_dir, '--dataset', 'test']):
                result = main()

                MockScript.assert_called_once()
                mock_instance.build.assert_called_once()
                mock_instance.print_args.assert_called_once()
                mock_instance.run.assert_called_once()
                assert result == 0


@pytest.mark.integration
class TestGenerateChampollionConfigIntegration:
    """Integration tests for GenerateChampollionConfig."""

    def test_full_workflow(self, temp_dir):
        """Test complete workflow."""
        script = GenerateChampollionConfig()
        script.parse_args([temp_dir, "--dataset", "test_dataset"])

        with patch.object(script, 'execute_command', return_value=0):
            with patch('champollion_pipeline.generate_champollion_config.exists', return_value=True):
                with patch('champollion_pipeline.generate_champollion_config.find_dataset_folder', return_value="/parent"):
                    with patch('builtins.open', mock_open(read_data="data: TESTXX\n")):
                        with patch.object(script, '_write_localization_yaml'):
                            result = script.run()
                            assert result == 0


@pytest.mark.smoke
class TestGenerateChampollionConfigSmoke:
    """Smoke tests for basic functionality."""

    def test_script_can_be_instantiated(self):
        """Test that script can be created."""
        script = GenerateChampollionConfig()
        assert script is not None

    def test_script_has_required_methods(self):
        """Test that script has all required methods."""
        script = GenerateChampollionConfig()
        assert hasattr(script, 'run')
        assert hasattr(script, '_validate_inputs')
        assert hasattr(script, '_write_localization_yaml')
        assert callable(script.run)
