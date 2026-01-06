#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for run_deep_folding.py
"""

import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from run_deep_folding import RunDeepFolding


class TestRunDeepFoldingInit:
    """Test initialization of RunDeepFolding."""

    def test_init_creates_script(self):
        """Test that script is initialized correctly."""
        script = RunDeepFolding()
        assert script.script_name == "run_deep_folding"
        assert "deep_folding" in script.description.lower()

    def test_init_configures_arguments(self):
        """Test that arguments are configured."""
        script = RunDeepFolding()
        with pytest.raises(SystemExit):
            script.parse_args([])  # Should fail due to missing required args


class TestRunDeepFoldingArguments:
    """Test argument parsing."""

    def test_parse_required_arguments(self):
        """Test parsing required arguments."""
        script = RunDeepFolding()
        args = script.parse_args([
            "/input",
            "/output",
            "--path_to_graph", "graphs/path",
            "--path_sk_with_hull", "skeleton/path"
        ])
        assert args.input == "/input"
        assert args.output == "/output"
        assert args.path_to_graph == "graphs/path"
        assert args.path_sk_with_hull == "skeleton/path"

    def test_optional_region_file(self):
        """Test optional region-file argument."""
        script = RunDeepFolding()
        args = script.parse_args([
            "/input",
            "/output",
            "--path_to_graph", "graphs/path",
            "--path_sk_with_hull", "skeleton/path",
            "--region-file", "/path/to/regions.json"
        ])
        assert args.region_file == "/path/to/regions.json"

    def test_optional_sk_qc_path_default(self):
        """Test that sk_qc_path defaults to empty string."""
        script = RunDeepFolding()
        args = script.parse_args([
            "/input",
            "/output",
            "--path_to_graph", "graphs/path",
            "--path_sk_with_hull", "skeleton/path"
        ])
        assert args.sk_qc_path == ""

    def test_optional_njobs_default(self):
        """Test that njobs defaults to None."""
        script = RunDeepFolding()
        args = script.parse_args([
            "/input",
            "/output",
            "--path_to_graph", "graphs/path",
            "--path_sk_with_hull", "skeleton/path"
        ])
        assert args.njobs is None

    def test_njobs_custom_value(self):
        """Test setting custom njobs value."""
        script = RunDeepFolding()
        args = script.parse_args([
            "/input",
            "/output",
            "--path_to_graph", "graphs/path",
            "--path_sk_with_hull", "skeleton/path",
            "--njobs", "10"
        ])
        assert args.njobs == 10

    def test_missing_required_arguments(self):
        """Test that missing required arguments raise error."""
        script = RunDeepFolding()
        with pytest.raises(SystemExit):
            script.parse_args(["/input", "/output"])  # Missing path_to_graph and path_sk_with_hull


class TestNjobsHandling:
    """Test njobs calculation and validation."""

    @patch('run_deep_folding.cpu_count', return_value=24)
    def test_njobs_none_uses_default_calculation(self, mock_cpu):
        """Test that njobs=None calculates min(22, cpu_count-2)."""
        script = RunDeepFolding()
        script.parse_args([
            "/input",
            "/output",
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton"
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'execute_command', return_value=0):
                with patch('run_deep_folding.chdir'):
                    with patch('run_deep_folding.getcwd', return_value="/original"):
                        script.run()

                        # njobs should be min(22, 24-2) = 22
                        assert script.args.njobs == 22

    @patch('run_deep_folding.cpu_count', return_value=8)
    def test_njobs_none_with_low_cpu_count(self, mock_cpu):
        """Test njobs calculation with low CPU count."""
        script = RunDeepFolding()
        script.parse_args([
            "/input",
            "/output",
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton"
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'execute_command', return_value=0):
                with patch('run_deep_folding.chdir'):
                    with patch('run_deep_folding.getcwd', return_value="/original"):
                        script.run()

                        # njobs should be min(22, 8-2) = 6
                        assert script.args.njobs == 6

    @patch('run_deep_folding.cpu_count', return_value=8)
    @patch('builtins.print')
    def test_njobs_exceeds_cpu_count_prints_warning(self, mock_print, mock_cpu):
        """Test that warning is printed when njobs >= cpu_count."""
        script = RunDeepFolding()
        script.parse_args([
            "/input",
            "/output",
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton",
            "--njobs", "10"  # More than cpu_count
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'execute_command', return_value=0):
                with patch('run_deep_folding.chdir'):
                    with patch('run_deep_folding.getcwd', return_value="/original"):
                        script.run()

                        # Check warning was printed
                        warning_printed = any(
                            "Warning" in str(call_args)
                            for call_args in mock_print.call_args_list
                        )
                        assert warning_printed

    @patch('joblib.cpu_count', return_value=24)
    def test_njobs_custom_value_preserved(self, mock_cpu):
        """Test that custom njobs value is preserved."""
        script = RunDeepFolding()
        script.parse_args([
            "/input",
            "/output",
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton",
            "--njobs", "5"
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'build_command', return_value=["cmd"]):
                with patch.object(script, 'execute_command', return_value=0):
                    with patch('os.chdir'):
                        with patch('os.getcwd', return_value="/original"):
                            script.run()

                            assert script.args.njobs == 5


class TestValidatePaths:
    """Test path validation."""

    def test_validate_paths_success(self, temp_dir):
        """Test successful path validation."""
        script = RunDeepFolding()
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        script.parse_args([
            str(input_dir),
            str(output_dir),
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton"
        ])

        assert script.validate_paths([str(input_dir), str(output_dir)]) is True

    def test_validate_paths_failure_raises_error(self):
        """Test that invalid paths raise ValueError."""
        script = RunDeepFolding()
        script.parse_args([
            "/nonexistent/input",
            "/nonexistent/output",
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton"
        ])

        with patch.object(script, 'validate_paths', return_value=False):
            with pytest.raises(ValueError, match="Please input valid paths"):
                script.run()


class TestBuildCommand:
    """Test command building."""

    def test_command_includes_required_parameters(self, temp_dir):
        """Test that command includes all required parameters."""
        script = RunDeepFolding()
        script.parse_args([
            temp_dir,
            temp_dir,
            "--path_to_graph", "custom/graph/path",
            "--path_sk_with_hull", "custom/skeleton/path"
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'execute_command', return_value=0) as mock_exec:
                with patch('run_deep_folding.chdir'):
                    with patch('run_deep_folding.getcwd', return_value="/original"):
                        script.run()

                        # Check execute_command was called with proper command
                        mock_exec.assert_called_once()
                        cmd = mock_exec.call_args[0][0]

                        # Verify command structure
                        assert "-d" in cmd
                        assert "--path_to_graph" in cmd
                        assert "--path_sk_with_hull" in cmd
                        assert "--njobs" in cmd


class TestCommandModification:
    """Test command modification for -d flag."""

    def test_command_replaces_input_with_d_flag(self, temp_dir):
        """Test that --input= is replaced with -d."""
        script = RunDeepFolding()
        script.parse_args([
            temp_dir,
            temp_dir,
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton"
        ])

        # Mock build_command to return command with --input=
        mock_cmd = ["python", "script.py", f"--input={temp_dir}", "--other=value"]

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'build_command', return_value=mock_cmd):
                with patch.object(script, 'execute_command', return_value=0) as mock_exec:
                    with patch('os.chdir'):
                        with patch('os.getcwd', return_value="/original"):
                            script.run()

                            # Check the modified command
                            executed_cmd = mock_exec.call_args[0][0]

                            # Should not contain --input=
                            assert not any("--input=" in arg for arg in executed_cmd)

                            # Should contain -d with space separation
                            cmd_str = " ".join(executed_cmd)
                            assert "-d " in cmd_str


class TestRunMethod:
    """Test the run method."""

    @patch('run_deep_folding.chdir')
    @patch('run_deep_folding.getcwd', return_value="/original")
    def test_run_changes_to_deep_folding_directory(self, mock_getcwd, mock_chdir, temp_dir):
        """Test that run changes to deep_folding directory."""
        script = RunDeepFolding()
        script.parse_args([
            temp_dir,
            temp_dir,
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton"
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'execute_command', return_value=0):
                script.run()

                # Check that chdir was called
                assert mock_chdir.called
                # Should change to deep_folding path and back to original
                assert call("/original") in mock_chdir.call_args_list

    def test_run_executes_with_shell_false(self, temp_dir):
        """Test that command is executed with shell=False."""
        script = RunDeepFolding()
        script.parse_args([
            temp_dir,
            temp_dir,
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton"
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'execute_command', return_value=0) as mock_exec:
                with patch('run_deep_folding.chdir'):
                    with patch('run_deep_folding.getcwd', return_value="/original"):
                        script.run()

                        # Check shell=False
                        assert mock_exec.call_args[1]['shell'] is False

    def test_run_returns_command_result(self, temp_dir):
        """Test that run returns the result from execute_command."""
        script = RunDeepFolding()
        script.parse_args([
            temp_dir,
            temp_dir,
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton"
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'execute_command', return_value=42):
                with patch('run_deep_folding.chdir'):
                    with patch('run_deep_folding.getcwd', return_value="/original"):
                        result = script.run()
                        assert result == 42

    @patch('builtins.print')
    def test_run_prints_input_output(self, mock_print, temp_dir):
        """Test that run prints input and output paths."""
        script = RunDeepFolding()
        script.parse_args([
            temp_dir,
            temp_dir,
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton"
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'execute_command', return_value=0):
                with patch('run_deep_folding.chdir'):
                    with patch('run_deep_folding.getcwd', return_value="/original"):
                        script.run()

                        # Check that paths were printed
                        print_calls = [str(call) for call in mock_print.call_args_list]
                        assert any("input" in call.lower() for call in print_calls)
                        assert any("output" in call.lower() for call in print_calls)


class TestMainFunction:
    """Test the main entry point."""

    def test_main_creates_script_and_runs(self, temp_dir):
        """Test that main creates script and calls build().print_args().run()."""
        with patch('run_deep_folding.RunDeepFolding') as MockScript:
            mock_instance = MagicMock()
            mock_instance.build.return_value = mock_instance
            mock_instance.print_args.return_value = mock_instance
            mock_instance.run.return_value = 0
            MockScript.return_value = mock_instance

            from run_deep_folding import main

            with patch('sys.argv', [
                'script',
                temp_dir,
                temp_dir,
                '--path_to_graph', 'graphs',
                '--path_sk_with_hull', 'skeleton'
            ]):
                result = main()

                MockScript.assert_called_once()
                mock_instance.build.assert_called_once()
                mock_instance.print_args.assert_called_once()
                mock_instance.run.assert_called_once()
                assert result == 0


@pytest.mark.integration
class TestRunDeepFoldingIntegration:
    """Integration tests for RunDeepFolding."""

    def test_full_workflow(self, temp_dir):
        """Test complete workflow."""
        script = RunDeepFolding()
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        script.parse_args([
            str(input_dir),
            str(output_dir),
            "--path_to_graph", "graphs/path",
            "--path_sk_with_hull", "skeleton/path",
            "--njobs", "4"
        ])

        with patch.object(script, 'execute_command', return_value=0) as mock_exec:
            with patch('os.chdir'):
                with patch('os.getcwd', return_value="/original"):
                    result = script.run()

                    assert result == 0
                    mock_exec.assert_called_once()


@pytest.mark.smoke
class TestRunDeepFoldingSmoke:
    """Smoke tests for basic functionality."""

    def test_script_can_be_instantiated(self):
        """Test that script can be created."""
        script = RunDeepFolding()
        assert script is not None

    def test_script_has_required_methods(self):
        """Test that script has all required methods."""
        script = RunDeepFolding()
        assert hasattr(script, 'run')
        assert callable(script.run)
