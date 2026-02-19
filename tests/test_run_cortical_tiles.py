#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for run_cortical_tiles.py
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from run_cortical_tiles import RunCorticalTiles


class TestRunCorticalTilesInit:
    """Test initialization of RunCorticalTiles."""

    def test_init_creates_script(self):
        """Test that script is initialized correctly."""
        script = RunCorticalTiles()
        assert script.script_name == "run_cortical_tiles"
        assert "cortical_tiles" in script.description.lower() or "sulcal" in script.description.lower()

    def test_init_configures_arguments(self):
        """Test that arguments are configured."""
        script = RunCorticalTiles()
        with pytest.raises(SystemExit):
            script.parse_args([])  # Should fail due to missing required args


class TestRunCorticalTilesArguments:
    """Test argument parsing."""

    def test_parse_required_arguments(self):
        """Test parsing required arguments."""
        script = RunCorticalTiles()
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
        script = RunCorticalTiles()
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
        script = RunCorticalTiles()
        args = script.parse_args([
            "/input",
            "/output",
            "--path_to_graph", "graphs/path",
            "--path_sk_with_hull", "skeleton/path"
        ])
        assert args.sk_qc_path == ""

    def test_optional_njobs_default(self):
        """Test that njobs defaults to None."""
        script = RunCorticalTiles()
        args = script.parse_args([
            "/input",
            "/output",
            "--path_to_graph", "graphs/path",
            "--path_sk_with_hull", "skeleton/path"
        ])
        assert args.njobs is None

    def test_njobs_custom_value(self):
        """Test setting custom njobs value."""
        script = RunCorticalTiles()
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
        script = RunCorticalTiles()
        with pytest.raises(SystemExit):
            script.parse_args(["/input", "/output"])  # Missing path_to_graph and path_sk_with_hull

    def test_input_types_default_none(self):
        """Test that input_types defaults to None."""
        script = RunCorticalTiles()
        args = script.parse_args([
            "/input", "/output",
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton"
        ])
        assert args.input_types is None

    def test_input_types_can_be_set(self):
        """Test that --input-types accepts multiple values."""
        script = RunCorticalTiles()
        args = script.parse_args([
            "/input", "/output",
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton",
            "--input-types", "skeleton", "foldlabel", "extremities"
        ])
        assert args.input_types == ["skeleton", "foldlabel", "extremities"]

    def test_skip_distbottom_default_false(self):
        """Test that --skip-distbottom defaults to False."""
        script = RunCorticalTiles()
        args = script.parse_args([
            "/input", "/output",
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton"
        ])
        assert args.skip_distbottom is False

    def test_skip_distbottom_can_be_set(self):
        """Test that --skip-distbottom can be set to True."""
        script = RunCorticalTiles()
        args = script.parse_args([
            "/input", "/output",
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton",
            "--skip-distbottom"
        ])
        assert args.skip_distbottom is True


class TestNjobsHandling:
    """Test njobs calculation and validation."""

    @patch('run_cortical_tiles.cpu_count', return_value=24)
    def test_njobs_none_uses_default_calculation(self, mock_cpu):
        """Test that njobs=None calculates min(22, cpu_count-2)."""
        script = RunCorticalTiles()
        script.parse_args([
            "/input",
            "/output",
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton"
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'execute_command', return_value=0):
                with patch('run_cortical_tiles.chdir'):
                    with patch('run_cortical_tiles.getcwd', return_value="/original"):
                        script.run()

                        # njobs should be min(22, 24-2) = 22
                        assert script.args.njobs == 22

    @patch('run_cortical_tiles.cpu_count', return_value=8)
    def test_njobs_none_with_low_cpu_count(self, mock_cpu):
        """Test njobs calculation with low CPU count."""
        script = RunCorticalTiles()
        script.parse_args([
            "/input",
            "/output",
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton"
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'execute_command', return_value=0):
                with patch('run_cortical_tiles.chdir'):
                    with patch('run_cortical_tiles.getcwd', return_value="/original"):
                        script.run()

                        # njobs should be min(22, 8-2) = 6
                        assert script.args.njobs == 6

    @patch('run_cortical_tiles.cpu_count', return_value=8)
    @patch('builtins.print')
    def test_njobs_exceeds_cpu_count_prints_warning(self, mock_print, mock_cpu):
        """Test that warning is printed when njobs >= cpu_count."""
        script = RunCorticalTiles()
        script.parse_args([
            "/input",
            "/output",
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton",
            "--njobs", "10"  # More than cpu_count
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'execute_command', return_value=0):
                with patch('run_cortical_tiles.chdir'):
                    with patch('run_cortical_tiles.getcwd', return_value="/original"):
                        script.run()

                        # Check warning was printed
                        warning_printed = any(
                            "Warning" in str(call_args)
                            for call_args in mock_print.call_args_list
                        )
                        assert warning_printed


class TestValidatePaths:
    """Test path validation."""

    def test_validate_paths_success(self, temp_dir):
        """Test successful path validation."""
        script = RunCorticalTiles()
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
        script = RunCorticalTiles()
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
        script = RunCorticalTiles()
        script.parse_args([
            temp_dir,
            temp_dir,
            "--path_to_graph", "custom/graph/path",
            "--path_sk_with_hull", "custom/skeleton/path"
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'execute_command', return_value=0) as mock_exec:
                with patch('run_cortical_tiles.chdir'):
                    with patch('run_cortical_tiles.getcwd', return_value="/original"):
                        script.run()

                        # Check execute_command was called with proper command
                        mock_exec.assert_called()
                        cmd = mock_exec.call_args[0][0]

                        # Verify command structure
                        assert "-d" in cmd
                        assert "--path_to_graph" in cmd
                        assert "--path_sk_with_hull" in cmd
                        assert "--njobs" in cmd

    def test_command_includes_input_types_when_set(self, temp_dir):
        """Test that --input-types adds -y flag to command."""
        script = RunCorticalTiles()
        script.parse_args([
            temp_dir, temp_dir,
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton",
            "--input-types", "skeleton", "foldlabel"
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'execute_command', return_value=0) as mock_exec:
                with patch('run_cortical_tiles.chdir'):
                    with patch('run_cortical_tiles.getcwd', return_value="/original"):
                        script.run()

                        cmd = mock_exec.call_args[0][0]
                        assert "-y" in cmd
                        assert "skeleton" in cmd
                        assert "foldlabel" in cmd

    def test_command_includes_sk_qc_path_when_set(self, temp_dir):
        """Test that --sk_qc_path is included when set."""
        script = RunCorticalTiles()
        script.parse_args([
            temp_dir, temp_dir,
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton",
            "--sk_qc_path", "/path/to/qc.tsv"
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'execute_command', return_value=0) as mock_exec:
                with patch('run_cortical_tiles.chdir'):
                    with patch('run_cortical_tiles.getcwd', return_value="/original"):
                        script.run()

                        cmd = mock_exec.call_args[0][0]
                        assert "--sk_qc_path" in cmd
                        assert "/path/to/qc.tsv" in cmd


class TestSkipDistbottom:
    """Test skip_distbottom modifying pipeline_loop_2mm.json."""

    def test_skip_distbottom_modifies_config(self, temp_dir):
        """Test that --skip-distbottom sets skip_distbottom in pipeline JSON."""
        # Config lives inside the input directory
        config_data = {"graphs_dir": "", "other_key": "value"}
        config_path = Path(temp_dir) / "pipeline_loop_2mm.json"
        config_path.write_text(json.dumps(config_data))

        script = RunCorticalTiles()
        script.parse_args([
            temp_dir, temp_dir,
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton",
            "--skip-distbottom"
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'execute_command', return_value=0):
                with patch('run_cortical_tiles.chdir'):
                    with patch('run_cortical_tiles.getcwd', return_value="/original"):
                        script.run()

                        # Read back config and check skip_distbottom was set
                        updated_config = json.loads(config_path.read_text())
                        assert updated_config.get('skip_distbottom') is True

    def test_no_skip_distbottom_does_not_modify_config(self, temp_dir):
        """Test that without --skip-distbottom, config is unchanged for that key."""
        config_data = {"graphs_dir": "", "other_key": "value"}
        config_path = Path(temp_dir) / "pipeline_loop_2mm.json"
        config_path.write_text(json.dumps(config_data))

        script = RunCorticalTiles()
        script.parse_args([
            temp_dir, temp_dir,
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton",
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'execute_command', return_value=0):
                with patch('run_cortical_tiles.chdir'):
                    with patch('run_cortical_tiles.getcwd', return_value="/original"):
                        script.run()

                        updated_config = json.loads(config_path.read_text())
                        assert 'skip_distbottom' not in updated_config

    def test_graphs_dir_set_to_input(self, temp_dir):
        """Test that graphs_dir in config is set to input path."""
        config_data = {"graphs_dir": "$local", "other_key": "value"}
        config_path = Path(temp_dir) / "pipeline_loop_2mm.json"
        config_path.write_text(json.dumps(config_data))

        script = RunCorticalTiles()
        script.parse_args([
            temp_dir, temp_dir,
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton",
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'execute_command', return_value=0):
                with patch('run_cortical_tiles.chdir'):
                    with patch('run_cortical_tiles.getcwd', return_value="/original"):
                        script.run()

                        updated = json.loads(config_path.read_text())
                        assert updated['graphs_dir'] == str(
                            Path(temp_dir).resolve()
                        )

    def test_output_dir_set_from_output_arg(self, temp_dir):
        """Test that output_dir in config is set to output/DERIVATIVES_FOLDER."""
        from utils.lib import DERIVATIVES_FOLDER
        config_data = {"graphs_dir": "$local", "output_dir": "$local"}
        config_path = Path(temp_dir) / "pipeline_loop_2mm.json"
        config_path.write_text(json.dumps(config_data))

        output_dir = Path(temp_dir) / "derivatives"
        output_dir.mkdir()

        script = RunCorticalTiles()
        script.parse_args([
            temp_dir, str(output_dir),
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton",
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'execute_command', return_value=0):
                with patch('run_cortical_tiles.chdir'):
                    with patch('run_cortical_tiles.getcwd', return_value="/original"):
                        script.run()

                        updated = json.loads(config_path.read_text())
                        expected = str(output_dir.resolve() / DERIVATIVES_FOLDER)
                        assert updated['output_dir'] == expected


class TestRunMethod:
    """Test the run method."""

    @patch('run_cortical_tiles.chdir')
    @patch('run_cortical_tiles.getcwd', return_value="/original")
    def test_run_changes_to_script_directory(self, mock_getcwd, mock_chdir, temp_dir):
        """Test that run changes to the cortical_tiles script directory."""
        script = RunCorticalTiles()
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
                # Should change back to original directory
                assert call("/original") in mock_chdir.call_args_list

    def test_run_executes_with_shell_false(self, temp_dir):
        """Test that command is executed with shell=False."""
        script = RunCorticalTiles()
        script.parse_args([
            temp_dir,
            temp_dir,
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton"
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'execute_command', return_value=0) as mock_exec:
                with patch('run_cortical_tiles.chdir'):
                    with patch('run_cortical_tiles.getcwd', return_value="/original"):
                        script.run()

                        # Check shell=False
                        assert mock_exec.call_args[1]['shell'] is False

    def test_run_returns_command_result(self, temp_dir):
        """Test that run returns the result from execute_command."""
        script = RunCorticalTiles()
        script.parse_args([
            temp_dir,
            temp_dir,
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton"
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'execute_command', return_value=42):
                with patch('run_cortical_tiles.chdir'):
                    with patch('run_cortical_tiles.getcwd', return_value="/original"):
                        result = script.run()
                        assert result == 42

    @patch('builtins.print')
    def test_run_prints_input_output(self, mock_print, temp_dir):
        """Test that run prints input and output paths."""
        script = RunCorticalTiles()
        script.parse_args([
            temp_dir,
            temp_dir,
            "--path_to_graph", "graphs",
            "--path_sk_with_hull", "skeleton"
        ])

        with patch.object(script, 'validate_paths', return_value=True):
            with patch.object(script, 'execute_command', return_value=0):
                with patch('run_cortical_tiles.chdir'):
                    with patch('run_cortical_tiles.getcwd', return_value="/original"):
                        script.run()

                        # Check that paths were printed
                        print_calls = [str(call) for call in mock_print.call_args_list]
                        assert any("input" in call.lower() for call in print_calls)
                        assert any("output" in call.lower() for call in print_calls)


class TestMainFunction:
    """Test the main entry point."""

    def test_main_creates_script_and_runs(self, temp_dir):
        """Test that main creates script and calls build().print_args().run()."""
        with patch('run_cortical_tiles.RunCorticalTiles') as MockScript:
            mock_instance = MagicMock()
            mock_instance.build.return_value = mock_instance
            mock_instance.print_args.return_value = mock_instance
            mock_instance.run.return_value = 0
            MockScript.return_value = mock_instance

            from run_cortical_tiles import main

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
class TestRunCorticalTilesIntegration:
    """Integration tests for RunCorticalTiles."""

    def test_full_workflow(self, temp_dir):
        """Test complete workflow."""
        script = RunCorticalTiles()
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
            with patch('run_cortical_tiles.chdir'):
                with patch('run_cortical_tiles.getcwd', return_value="/original"):
                    result = script.run()

                    assert result == 0
                    mock_exec.assert_called()


@pytest.mark.smoke
class TestRunCorticalTilesSmoke:
    """Smoke tests for basic functionality."""

    def test_script_can_be_instantiated(self):
        """Test that script can be created."""
        script = RunCorticalTiles()
        assert script is not None

    def test_script_has_required_methods(self):
        """Test that script has all required methods."""
        script = RunCorticalTiles()
        assert hasattr(script, 'run')
        assert callable(script.run)
