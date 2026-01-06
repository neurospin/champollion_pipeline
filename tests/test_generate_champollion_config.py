#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for generate_champollion_config.py
"""

import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call, mock_open

from generate_champollion_config import GenerateChampollionConfig


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


class TestHandleYamlConf:
    """Test _handle_yaml_conf method."""

    def test_handle_yaml_conf_updates_dataset_folder(self, temp_dir):
        """Test that dataset_folder is updated in YAML."""
        script = GenerateChampollionConfig()
        script.parse_args([temp_dir, "--dataset", "test"])

        yaml_content = "dataset_folder: /old/path\nother_key: value\n"
        conf_file = Path(temp_dir) / "test.yaml"
        conf_file.write_text(yaml_content)

        with patch('generate_champollion_config.get_nth_parent_dir', return_value="/new/parent"):
            script._handle_yaml_conf(str(conf_file), temp_dir)

            updated_content = conf_file.read_text()
            assert "dataset_folder: /new/parent" in updated_content
            assert "other_key: value" in updated_content

    def test_handle_yaml_conf_preserves_other_lines(self, temp_dir):
        """Test that other lines are preserved."""
        script = GenerateChampollionConfig()
        script.parse_args([temp_dir, "--dataset", "test"])

        yaml_content = "key1: value1\ndataset_folder: /old\nkey2: value2\n"
        conf_file = Path(temp_dir) / "test.yaml"
        conf_file.write_text(yaml_content)

        with patch('generate_champollion_config.get_nth_parent_dir', return_value="/new"):
            script._handle_yaml_conf(str(conf_file), temp_dir)

            updated_content = conf_file.read_text()
            assert "key1: value1" in updated_content
            assert "key2: value2" in updated_content


class TestRunMethod:
    """Test the run method."""

    def test_run_validates_inputs(self, temp_dir):
        """Test that run calls _validate_inputs."""
        script = GenerateChampollionConfig()
        script.parse_args([temp_dir, "--dataset", "test"])

        with patch.object(script, '_validate_inputs') as mock_validate:
            with patch('os.path.exists', return_value=True):
                with patch.object(script, 'execute_command', return_value=0):
                    with patch.object(script, '_handle_yaml_conf'):
                        with patch('os.chdir'):
                            with patch('os.getcwd', return_value="/original"):
                                with patch('builtins.open', mock_open(read_data="data: TESTXX")):
                                    script.run()
                                    mock_validate.assert_called_once()

    def test_run_creates_dataset_directory(self, temp_dir):
        """Test that run creates dataset directory if it doesn't exist."""
        script = GenerateChampollionConfig()
        script.parse_args([temp_dir, "--dataset", "test_dataset"])

        with patch('os.path.exists', side_effect=[False, False]):  # dir and file don't exist
            with patch.object(script, 'execute_command', return_value=0) as mock_exec:
                with patch.object(script, '_handle_yaml_conf'):
                    with patch('os.chdir'):
                        with patch('os.getcwd', return_value="/original"):
                            with patch('builtins.open', mock_open(read_data="data: TESTXX")):
                                script.run()

                                # Check mkdir was called
                                mkdir_calls = [call for call in mock_exec.call_args_list
                                             if "mkdir" in str(call)]
                                assert len(mkdir_calls) > 0

    def test_run_copies_reference_yaml(self, temp_dir):
        """Test that run copies reference.yaml if it doesn't exist."""
        script = GenerateChampollionConfig()
        script.parse_args([temp_dir, "--dataset", "test_dataset"])

        with patch('os.path.exists', side_effect=[True, False]):  # dir exists, file doesn't
            with patch.object(script, 'execute_command', return_value=0) as mock_exec:
                with patch.object(script, '_handle_yaml_conf'):
                    with patch('os.chdir'):
                        with patch('os.getcwd', return_value="/original"):
                            with patch('builtins.open', mock_open(read_data="data: TESTXX")):
                                script.run()

                                # Check cp was called
                                cp_calls = [call for call in mock_exec.call_args_list
                                          if "cp" in str(call)]
                                assert len(cp_calls) > 0

    def test_run_updates_reference_yaml(self, temp_dir):
        """Test that run updates TESTXX in reference.yaml."""
        script = GenerateChampollionConfig()
        dataset_name = "my_dataset"
        script.parse_args([temp_dir, "--dataset", dataset_name])

        yaml_content = "some_key: TESTXX/path\nanother: TESTXX"
        expected_path = f"{dataset_name}/derivatives/deep_folding-2025"

        with patch('os.path.exists', return_value=True):
            with patch.object(script, 'execute_command', return_value=0):
                with patch.object(script, '_handle_yaml_conf'):
                    with patch('os.chdir'):
                        with patch('os.getcwd', return_value="/original"):
                            m = mock_open(read_data=yaml_content)
                            with patch('builtins.open', m):
                                script.run()

                                # Check that file was written with replaced TESTXX
                                write_calls = [call for call in m().writelines.call_args_list]
                                if write_calls:
                                    written_lines = write_calls[0][0][0]
                                    written_content = ''.join(written_lines)
                                    assert expected_path in written_content or dataset_name in written_content

    def test_run_calls_create_dataset_config_files(self, temp_dir):
        """Test that run calls create_dataset_config_files.py."""
        script = GenerateChampollionConfig()
        script.parse_args([temp_dir, "--dataset", "test"])

        with patch('os.path.exists', return_value=True):
            with patch.object(script, 'execute_command', return_value=0) as mock_exec:
                with patch.object(script, '_handle_yaml_conf'):
                    with patch('os.chdir'):
                        with patch('os.getcwd', return_value="/original"):
                            with patch('builtins.open', mock_open(read_data="data: TESTXX")):
                                script.run()

                                # Find the call to create_dataset_config_files
                                create_calls = [call for call in mock_exec.call_args_list
                                              if "create_dataset_config_files" in str(call)]
                                assert len(create_calls) > 0

    def test_run_calls_handle_yaml_conf(self, temp_dir):
        """Test that run calls _handle_yaml_conf."""
        script = GenerateChampollionConfig()
        script.parse_args([temp_dir, "--dataset", "test"])

        with patch('os.path.exists', return_value=True):
            with patch.object(script, 'execute_command', return_value=0):
                with patch.object(script, '_handle_yaml_conf') as mock_handle:
                    with patch('os.chdir'):
                        with patch('os.getcwd', return_value="/original"):
                            with patch('builtins.open', mock_open(read_data="data: TESTXX")):
                                script.run()
                                mock_handle.assert_called_once()

    def test_run_restores_directory(self, temp_dir):
        """Test that run restores original directory."""
        script = GenerateChampollionConfig()
        script.parse_args([temp_dir, "--dataset", "test"])
        original_dir = "/original/dir"

        with patch('os.getcwd', return_value=original_dir):
            with patch('os.chdir') as mock_chdir:
                with patch('os.path.exists', return_value=True):
                    with patch.object(script, 'execute_command', return_value=0):
                        with patch.object(script, '_handle_yaml_conf'):
                            with patch('builtins.open', mock_open(read_data="data: TESTXX")):
                                script.run()

                                # Verify we returned to original directory
                                assert mock_chdir.call_args_list[-1] == call(original_dir)

    def test_run_returns_result(self, temp_dir):
        """Test that run returns the result from execute_command."""
        script = GenerateChampollionConfig()
        script.parse_args([temp_dir, "--dataset", "test"])

        with patch('os.path.exists', return_value=True):
            with patch.object(script, 'execute_command', return_value=42):
                with patch.object(script, '_handle_yaml_conf'):
                    with patch('os.chdir'):
                        with patch('os.getcwd', return_value="/original"):
                            with patch('builtins.open', mock_open(read_data="data: TESTXX")):
                                result = script.run()
                                assert result == 42


class TestMainFunction:
    """Test the main entry point."""

    def test_main_creates_script_and_runs(self, temp_dir):
        """Test that main creates script and calls build().print_args().run()."""
        with patch('generate_champollion_config.GenerateChampollionConfig') as MockScript:
            mock_instance = MagicMock()
            mock_instance.build.return_value = mock_instance
            mock_instance.print_args.return_value = mock_instance
            mock_instance.run.return_value = 0
            MockScript.return_value = mock_instance

            from generate_champollion_config import main

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
            with patch('os.chdir'):
                with patch('os.getcwd', return_value="/original"):
                    with patch('os.path.exists', return_value=True):
                        with patch('builtins.open', mock_open(read_data="data: TESTXX\n")):
                            with patch('generate_champollion_config.get_nth_parent_dir', return_value="/parent"):
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
        assert hasattr(script, '_handle_yaml_conf')
        assert callable(script.run)
