#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for generate_morphologist_graphs.py
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from generate_morphologist_graphs import GenerateMorphologistGraphs


class TestGenerateMorphologistGraphsInit:
    """Test initialization of GenerateMorphologistGraphs."""

    def test_init_creates_script(self):
        """Test that script is initialized correctly."""
        script = GenerateMorphologistGraphs()
        assert script.script_name == "morphologist_graphs_generator"
        assert "morphologist" in script.description.lower()

    def test_init_configures_arguments(self):
        """Test that arguments are configured."""
        script = GenerateMorphologistGraphs()
        # Parse with help to see if arguments are set up
        with pytest.raises(SystemExit):
            script.parse_args([])  # Should fail due to missing required args


class TestGenerateMorphologistGraphsArguments:
    """Test argument parsing."""

    def test_parse_input_output_arguments(self):
        """Test parsing input and output arguments."""
        script = GenerateMorphologistGraphs()
        args = script.parse_args(["/path/to/input", "/path/to/output"])
        assert args.input == "/path/to/input"
        assert args.output == "/path/to/output"

    def test_missing_arguments_raises_error(self):
        """Test that missing arguments raise SystemExit."""
        script = GenerateMorphologistGraphs()
        with pytest.raises(SystemExit):
            script.parse_args([])

    def test_only_one_argument_raises_error(self):
        """Test that providing only one argument raises error."""
        script = GenerateMorphologistGraphs()
        with pytest.raises(SystemExit):
            script.parse_args(["/path/to/input"])

    def test_parallel_flag_default_false(self):
        """Test that --parallel defaults to False."""
        script = GenerateMorphologistGraphs()
        args = script.parse_args(["/input", "/output"])
        assert args.parallel is False

    def test_parallel_flag_can_be_set(self):
        """Test that --parallel can be set to True."""
        script = GenerateMorphologistGraphs()
        args = script.parse_args(["/input", "/output", "--parallel"])
        assert args.parallel is True

    def test_enable_sulcal_recognition_default_false(self):
        """Test that --enable-sulcal-recognition defaults to False."""
        script = GenerateMorphologistGraphs()
        args = script.parse_args(["/input", "/output"])
        assert args.enable_sulcal_recognition is False

    def test_enable_sulcal_recognition_can_be_set(self):
        """Test that --enable-sulcal-recognition can be set to True."""
        script = GenerateMorphologistGraphs()
        args = script.parse_args(["/input", "/output", "--enable-sulcal-recognition"])
        assert args.enable_sulcal_recognition is True


class TestGetInputFiles:
    """Test _get_input_files method."""

    def test_get_input_files_with_nii_gz(self, temp_dir):
        """Test getting input files with .nii.gz extension."""
        script = GenerateMorphologistGraphs()
        script.parse_args([temp_dir, temp_dir])

        # Create test files
        (Path(temp_dir) / "test1.nii.gz").touch()
        (Path(temp_dir) / "test2.nii.gz").touch()
        (Path(temp_dir) / "ignore.txt").touch()

        files = script._get_input_files()
        assert len(files) == 2
        assert all(f.endswith((".nii.gz", ".nii", ".gz")) for f in files)

    def test_get_input_files_with_nii(self, temp_dir):
        """Test getting input files with .nii extension."""
        script = GenerateMorphologistGraphs()
        script.parse_args([temp_dir, temp_dir])

        # Create test files
        (Path(temp_dir) / "test1.nii").touch()
        (Path(temp_dir) / "test2.nii").touch()

        files = script._get_input_files()
        assert len(files) == 2

    def test_get_input_files_mixed_extensions(self, temp_dir):
        """Test getting input files with mixed valid extensions."""
        script = GenerateMorphologistGraphs()
        script.parse_args([temp_dir, temp_dir])

        # Create test files
        (Path(temp_dir) / "test1.nii.gz").touch()
        (Path(temp_dir) / "test2.nii").touch()
        (Path(temp_dir) / "test3.gz").touch()
        (Path(temp_dir) / "ignore.txt").touch()
        (Path(temp_dir) / "ignore.json").touch()

        files = script._get_input_files()
        assert len(files) == 3
        assert "ignore.txt" not in files
        assert "ignore.json" not in files

    def test_get_input_files_no_valid_files(self, temp_dir):
        """Test getting input files when no valid files exist."""
        script = GenerateMorphologistGraphs()
        script.parse_args([temp_dir, temp_dir])

        # Create only invalid files
        (Path(temp_dir) / "test.txt").touch()
        (Path(temp_dir) / "test.json").touch()

        files = script._get_input_files()
        assert len(files) == 0

    def test_get_input_files_empty_directory(self, temp_dir):
        """Test getting input files from empty directory."""
        script = GenerateMorphologistGraphs()
        script.parse_args([temp_dir, temp_dir])

        files = script._get_input_files()
        assert len(files) == 0


class TestValidatePaths:
    """Test path validation in run method."""

    def test_validate_paths_success(self, temp_dir):
        """Test successful path validation."""
        script = GenerateMorphologistGraphs()
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        script.parse_args([str(input_dir), str(output_dir)])

        assert script.validate_paths([str(input_dir), str(output_dir)]) is True

    def test_validate_paths_failure_raises_error(self, temp_dir):
        """Test that invalid paths raise ValueError."""
        script = GenerateMorphologistGraphs()
        script.parse_args(["/nonexistent/input", "/nonexistent/output"])

        with patch.object(script, 'validate_paths', return_value=False):
            with pytest.raises(ValueError, match="Please input valid paths"):
                script.run()


class TestRunMethod:
    """Test the run method."""

    @patch('generate_morphologist_graphs.chdir')
    @patch('generate_morphologist_graphs.getcwd', return_value="/original/dir")
    def test_run_changes_directory(self, mock_getcwd, mock_chdir, temp_dir):
        """Test that run changes to input directory."""
        script = GenerateMorphologistGraphs()
        script.parse_args([temp_dir, temp_dir])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, '_get_input_files', return_value=["test.nii.gz"]):
                with patch.object(script, 'execute_command', return_value=0):
                    script.run()

                    # Check that chdir was called to input and back
                    assert call(temp_dir) in mock_chdir.call_args_list
                    assert call("/original/dir") in mock_chdir.call_args_list

    def test_run_builds_correct_command(self, temp_dir):
        """Test that run builds correct morphologist-cli command."""
        script = GenerateMorphologistGraphs()
        input_dir = temp_dir
        output_dir = str(Path(temp_dir) / "output")
        Path(output_dir).mkdir()

        script.parse_args([input_dir, output_dir])

        test_files = ["subject1.nii.gz", "subject2.nii.gz"]

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, '_get_input_files', return_value=test_files):
                with patch.object(script, 'execute_command', return_value=0) as mock_exec:
                    with patch('generate_morphologist_graphs.chdir'):
                        with patch('generate_morphologist_graphs.getcwd', return_value="/original"):
                            script.run()

                            # Check the command
                            cmd = mock_exec.call_args[0][0]
                            assert "morphologist-cli" in cmd
                            assert "subject1.nii.gz" in cmd
                            assert "subject2.nii.gz" in cmd
                            assert output_dir in cmd
                            assert "--" in cmd
                            assert "--of" in cmd
                            assert "morphologist-auto-nonoverlap-1.0" in cmd

    def test_run_with_shell_true(self, temp_dir):
        """Test that run executes command with shell=True."""
        script = GenerateMorphologistGraphs()
        script.parse_args([temp_dir, temp_dir])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, '_get_input_files', return_value=["test.nii"]):
                with patch.object(script, 'execute_command', return_value=0) as mock_exec:
                    with patch('generate_morphologist_graphs.chdir'):
                        with patch('generate_morphologist_graphs.getcwd', return_value="/original"):
                            script.run()

                            # Check that shell=True was passed
                            assert mock_exec.call_args[1]['shell'] is True

    def test_run_returns_command_result(self, temp_dir):
        """Test that run returns the result from execute_command."""
        script = GenerateMorphologistGraphs()
        script.parse_args([temp_dir, temp_dir])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, '_get_input_files', return_value=["test.nii"]):
                with patch.object(script, 'execute_command', return_value=42):
                    with patch('generate_morphologist_graphs.chdir'):
                        with patch('generate_morphologist_graphs.getcwd', return_value="/original"):
                            result = script.run()
                            assert result == 42

    def test_run_restores_directory_on_success(self, temp_dir):
        """Test that original directory is restored after successful run."""
        script = GenerateMorphologistGraphs()
        script.parse_args([temp_dir, temp_dir])
        original_dir = "/original/dir"

        with patch('generate_morphologist_graphs.getcwd', return_value=original_dir):
            with patch('generate_morphologist_graphs.chdir') as mock_chdir:
                with patch.object(script, 'validate_paths', return_value=True):
                    with patch.object(script, '_get_input_files', return_value=["test.nii"]):
                        with patch.object(script, 'execute_command', return_value=0):
                            script.run()

                            # Verify we returned to original directory
                            assert mock_chdir.call_args_list[-1] == call(original_dir)


class TestMainFunction:
    """Test the main entry point."""

    def test_main_creates_script_and_runs(self, temp_dir):
        """Test that main creates script and calls build().print_args().run()."""
        with patch('generate_morphologist_graphs.GenerateMorphologistGraphs') as MockScript:
            mock_instance = MagicMock()
            mock_instance.build.return_value = mock_instance
            mock_instance.print_args.return_value = mock_instance
            mock_instance.run.return_value = 0
            MockScript.return_value = mock_instance

            from generate_morphologist_graphs import main

            with patch('sys.argv', ['script', temp_dir, temp_dir]):
                result = main()

                MockScript.assert_called_once()
                mock_instance.build.assert_called_once()
                mock_instance.print_args.assert_called_once()
                mock_instance.run.assert_called_once()
                assert result == 0


class TestCommandFlags:
    """Test that --parallel and --enable-sulcal-recognition affect the command."""

    def test_parallel_adds_swf_to_command(self, temp_dir):
        """Test that --parallel adds --swf to the command."""
        script = GenerateMorphologistGraphs()
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        (input_dir / "subject.nii.gz").touch()

        script.parse_args([str(input_dir), str(output_dir), "--parallel"])

        with patch.object(script, 'execute_command', return_value=0) as mock_exec:
            script.run()
            cmd = mock_exec.call_args[0][0]
            assert "--swf" in cmd

    def test_no_parallel_does_not_add_swf(self, temp_dir):
        """Test that without --parallel, --swf is not in the command."""
        script = GenerateMorphologistGraphs()
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        (input_dir / "subject.nii.gz").touch()

        script.parse_args([str(input_dir), str(output_dir)])

        with patch.object(script, 'execute_command', return_value=0) as mock_exec:
            script.run()
            cmd = mock_exec.call_args[0][0]
            assert "--swf" not in cmd

    def test_default_skips_sulcal_recognition(self, temp_dir):
        """Test that without --enable-sulcal-recognition, sulci_labelling=False is in cmd."""
        script = GenerateMorphologistGraphs()
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        (input_dir / "subject.nii.gz").touch()

        script.parse_args([str(input_dir), str(output_dir)])

        with patch.object(script, 'execute_command', return_value=0) as mock_exec:
            script.run()
            cmd = mock_exec.call_args[0][0]
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            assert "sulci_labelling=False" in cmd_str

    def test_enable_sulcal_recognition_removes_disable_step(self, temp_dir):
        """Test that --enable-sulcal-recognition removes the pipeline_steps disable."""
        script = GenerateMorphologistGraphs()
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        (input_dir / "subject.nii.gz").touch()

        script.parse_args([str(input_dir), str(output_dir), "--enable-sulcal-recognition"])

        with patch.object(script, 'execute_command', return_value=0) as mock_exec:
            script.run()
            cmd = mock_exec.call_args[0][0]
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            assert "sulci_labelling=False" not in cmd_str


@pytest.mark.integration
class TestGenerateMorphologistGraphsIntegration:
    """Integration tests for GenerateMorphologistGraphs."""

    def test_full_workflow_with_valid_files(self, temp_dir):
        """Test complete workflow with valid input files."""
        script = GenerateMorphologistGraphs()
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Create test files
        (input_dir / "subject1.nii.gz").touch()
        (input_dir / "subject2.nii.gz").touch()

        script.parse_args([str(input_dir), str(output_dir)])

        with patch.object(script, 'execute_command', return_value=0) as mock_exec:
            result = script.run()

            assert result == 0
            mock_exec.assert_called_once()
            # Verify command structure
            cmd = mock_exec.call_args[0][0]
            assert len(cmd) > 0
            assert "morphologist-cli" in cmd


@pytest.mark.smoke
class TestGenerateMorphologistGraphsSmoke:
    """Smoke tests for basic functionality."""

    def test_script_can_be_instantiated(self):
        """Test that script can be created."""
        script = GenerateMorphologistGraphs()
        assert script is not None

    def test_script_has_required_methods(self):
        """Test that script has all required methods."""
        script = GenerateMorphologistGraphs()
        assert hasattr(script, 'run')
        assert hasattr(script, '_get_input_files')
        assert callable(script.run)
        assert callable(script._get_input_files)
