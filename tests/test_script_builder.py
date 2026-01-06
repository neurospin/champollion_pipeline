#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the ScriptBuilder base class.
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch, call
from pathlib import Path

from script_builder import ScriptBuilder


class ConcreteScriptBuilder(ScriptBuilder):
    """Concrete implementation of ScriptBuilder for testing."""

    def __init__(self):
        super().__init__(
            script_name="test_script",
            description="Test script"
        )

    def run(self):
        """Minimal run implementation."""
        return 0


class TestScriptBuilderInit:
    """Test ScriptBuilder initialization."""

    def test_init_creates_parser(self):
        """Test that initialization creates an argument parser."""
        script = ConcreteScriptBuilder()
        assert script.parser is not None
        assert script.script_name == "test_script"
        assert script.description == "Test script"
        assert script.args is None

    def test_parser_has_correct_attributes(self):
        """Test that parser has correct prog and description."""
        script = ConcreteScriptBuilder()
        assert script.parser.prog == "test_script"
        assert script.parser.description == "Test script"


class TestScriptBuilderChaining:
    """Test method chaining functionality."""

    def test_add_argument_returns_self(self):
        """Test that add_argument returns self for chaining."""
        script = ConcreteScriptBuilder()
        result = script.add_argument("input", help="Input path")
        assert result is script

    def test_add_required_argument_returns_self(self):
        """Test that add_required_argument returns self for chaining."""
        script = ConcreteScriptBuilder()
        result = script.add_required_argument("--dataset", "Dataset name")
        assert result is script

    def test_add_optional_argument_returns_self(self):
        """Test that add_optional_argument returns self for chaining."""
        script = ConcreteScriptBuilder()
        result = script.add_optional_argument("--output", "Output path", default="/tmp")
        assert result is script

    def test_add_flag_returns_self(self):
        """Test that add_flag returns self for chaining."""
        script = ConcreteScriptBuilder()
        result = script.add_flag("--verbose", "Enable verbose output")
        assert result is script

    def test_print_args_returns_self(self):
        """Test that print_args returns self for chaining."""
        script = ConcreteScriptBuilder()
        script.add_argument("input", help="Input path")
        script.parse_args(["test_input"])
        result = script.print_args()
        assert result is script

    def test_build_returns_self(self):
        """Test that build returns self for chaining."""
        script = ConcreteScriptBuilder()
        script.add_argument("input", help="Input path")
        with patch.object(script.parser, 'parse_args', return_value=MagicMock(input="test")):
            result = script.build()
            assert result is script

    def test_full_chaining(self):
        """Test complete method chaining."""
        script = ConcreteScriptBuilder()
        result = (script.add_argument("input", help="Input")
                  .add_optional_argument("--output", "Output", default="/tmp")
                  .add_flag("--verbose", "Verbose"))
        assert result is script


class TestScriptBuilderArguments:
    """Test argument configuration methods."""

    def test_add_argument(self):
        """Test adding a positional argument."""
        script = ConcreteScriptBuilder()
        script.add_argument("input", help="Input path", type=str)
        args = script.parse_args(["test_input"])
        assert args.input == "test_input"

    def test_add_required_argument(self):
        """Test adding a required argument."""
        script = ConcreteScriptBuilder()
        script.add_required_argument("--dataset", "Dataset name")

        # Should fail without the argument
        with pytest.raises(SystemExit):
            script.parse_args([])

        # Should succeed with the argument
        args = script.parse_args(["--dataset", "test_dataset"])
        assert args.dataset == "test_dataset"

    def test_add_optional_argument_with_default(self):
        """Test adding an optional argument with default value."""
        script = ConcreteScriptBuilder()
        script.add_optional_argument("--output", "Output path", default="/tmp")
        args = script.parse_args([])
        assert args.output == "/tmp"

    def test_add_optional_argument_override_default(self):
        """Test overriding default value of optional argument."""
        script = ConcreteScriptBuilder()
        script.add_optional_argument("--output", "Output path", default="/tmp")
        args = script.parse_args(["--output", "/custom"])
        assert args.output == "/custom"

    def test_add_flag_default_false(self):
        """Test that flag defaults to False."""
        script = ConcreteScriptBuilder()
        script.add_flag("--verbose", "Verbose output")
        args = script.parse_args([])
        assert args.verbose is False

    def test_add_flag_set_true(self):
        """Test setting flag to True."""
        script = ConcreteScriptBuilder()
        script.add_flag("--verbose", "Verbose output")
        args = script.parse_args(["--verbose"])
        assert args.verbose is True


class TestScriptBuilderBuild:
    """Test the build method."""

    def test_build_parses_args(self):
        """Test that build parses arguments."""
        script = ConcreteScriptBuilder()
        script.add_argument("input", help="Input path")
        with patch.object(script, 'parse_args') as mock_parse:
            mock_parse.return_value = MagicMock(input="test")
            script.build()
            mock_parse.assert_called_once()

    def test_build_sets_args(self):
        """Test that build sets self.args."""
        script = ConcreteScriptBuilder()
        script.add_argument("input", help="Input path")
        with patch('sys.argv', ['script', 'test_input']):
            script.build()
            assert script.args is not None
            assert script.args.input == "test_input"


class TestScriptBuilderValidatePaths:
    """Test path validation."""

    def test_validate_paths_all_exist(self, temp_dir):
        """Test validation when all paths exist."""
        script = ConcreteScriptBuilder()
        path1 = Path(temp_dir) / "path1"
        path2 = Path(temp_dir) / "path2"
        path1.mkdir()
        path2.mkdir()

        assert script.validate_paths([str(path1), str(path2)]) is True

    def test_validate_paths_one_missing(self, temp_dir):
        """Test validation when one path is missing."""
        script = ConcreteScriptBuilder()
        path1 = Path(temp_dir) / "path1"
        path2 = Path(temp_dir) / "nonexistent"
        path1.mkdir()

        assert script.validate_paths([str(path1), str(path2)]) is False

    def test_validate_paths_all_missing(self):
        """Test validation when all paths are missing."""
        script = ConcreteScriptBuilder()
        assert script.validate_paths(["/nonexistent1", "/nonexistent2"]) is False

    def test_validate_paths_empty_list(self):
        """Test validation with empty list."""
        script = ConcreteScriptBuilder()
        assert script.validate_paths([]) is True


class TestScriptBuilderBuildCommand:
    """Test command building functionality."""

    def test_build_command_required_args_only(self):
        """Test building command with only required arguments."""
        script = ConcreteScriptBuilder()
        script.add_argument("input", help="Input")
        script.add_argument("output", help="Output")
        script.parse_args(["test_input", "test_output"])

        cmd = script.build_command(
            script_path="test_script.py",
            required_args=["input", "output"]
        )

        assert sys.executable in cmd
        assert "test_script.py" in cmd
        assert "--input=test_input" in cmd
        assert "--output=test_output" in cmd

    def test_build_command_with_defaults(self):
        """Test building command with default values."""
        script = ConcreteScriptBuilder()
        script.add_argument("input", help="Input")
        script.add_optional_argument("--verbose", "Verbose", default=False)
        script.add_optional_argument("--output", "Output", default="/tmp")
        script.parse_args(["test_input"])

        defaults = {"verbose": False, "output": "/tmp"}
        cmd = script.build_command(
            script_path="test_script.py",
            required_args=["input"],
            defaults=defaults
        )

        # Default values should not be included
        assert "--verbose" not in " ".join(cmd)
        assert "--output=/tmp" not in " ".join(cmd)

    def test_build_command_override_defaults(self):
        """Test building command with overridden defaults."""
        script = ConcreteScriptBuilder()
        script.add_argument("input", help="Input")
        script.add_optional_argument("--output", "Output", default="/tmp")
        script.parse_args(["test_input", "--output", "/custom"])

        defaults = {"output": "/tmp"}
        cmd = script.build_command(
            script_path="test_script.py",
            required_args=["input"],
            defaults=defaults
        )

        assert "--output=/custom" in cmd

    def test_build_command_with_boolean_flag(self):
        """Test building command with boolean flag."""
        script = ConcreteScriptBuilder()
        script.add_argument("input", help="Input")
        script.add_flag("--verbose", "Verbose")
        script.parse_args(["test_input", "--verbose"])

        defaults = {"verbose": False}
        cmd = script.build_command(
            script_path="test_script.py",
            required_args=["input"],
            defaults=defaults
        )

        assert "--verbose" in cmd

    def test_build_command_with_list_arguments(self):
        """Test building command with list arguments."""
        script = ConcreteScriptBuilder()
        script.add_argument("input", help="Input")
        script.add_argument("--datasets", nargs="+", default=["default"])
        script.parse_args(["test_input", "--datasets", "ds1", "ds2", "ds3"])

        defaults = {"datasets": ["default"]}
        cmd = script.build_command(
            script_path="test_script.py",
            required_args=["input"],
            defaults=defaults
        )

        assert "--datasets=ds1" in cmd
        assert "--datasets=ds2" in cmd
        assert "--datasets=ds3" in cmd


class TestScriptBuilderExecuteCommand:
    """Test command execution."""

    def test_execute_command_success_no_shell(self, mock_subprocess_success):
        """Test successful command execution without shell."""
        script = ConcreteScriptBuilder()
        cmd = ["echo", "test"]
        result = script.execute_command(cmd, shell=False)
        assert result == 0

    def test_execute_command_success_with_shell(self):
        """Test successful command execution with shell."""
        script = ConcreteScriptBuilder()
        cmd = ["echo", "test"]
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = script.execute_command(cmd, shell=True)
            assert result == 0
            mock_run.assert_called_once()

    def test_execute_command_failure(self):
        """Test command execution failure."""
        script = ConcreteScriptBuilder()
        cmd = ["nonexistent_command"]
        with patch('subprocess.check_call', side_effect=Exception("Command failed")):
            result = script.execute_command(cmd, shell=False)
            assert result == 1

    @patch('builtins.print')
    def test_execute_command_prints_command(self, mock_print):
        """Test that execute_command prints the command."""
        script = ConcreteScriptBuilder()
        cmd = ["echo", "test"]
        with patch('subprocess.check_call', return_value=0):
            script.execute_command(cmd, shell=False)
            # Check that print was called with command info
            assert mock_print.called


class TestScriptBuilderMain:
    """Test the main entry point."""

    def test_main_calls_build_print_run(self):
        """Test that main calls build, print_args, and run in sequence."""
        script = ConcreteScriptBuilder()
        script.add_argument("input", help="Input")

        with patch.object(script, 'build', return_value=script) as mock_build:
            with patch.object(script, 'print_args', return_value=script) as mock_print:
                with patch.object(script, 'run', return_value=0) as mock_run:
                    with patch('sys.argv', ['script', 'test_input']):
                        result = script.main()

                        mock_build.assert_called_once()
                        mock_print.assert_called_once()
                        mock_run.assert_called_once()
                        assert result == 0

    def test_main_returns_run_result(self):
        """Test that main returns the result from run."""
        script = ConcreteScriptBuilder()
        script.add_argument("input", help="Input")

        with patch.object(script, 'build', return_value=script):
            with patch.object(script, 'print_args', return_value=script):
                with patch.object(script, 'run', return_value=42):
                    with patch('sys.argv', ['script', 'test_input']):
                        result = script.main()
                        assert result == 42


@pytest.mark.unit
class TestScriptBuilderAbstract:
    """Test abstract method enforcement."""

    def test_cannot_instantiate_base_class(self):
        """Test that ScriptBuilder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ScriptBuilder("test", "test description")

    def test_subclass_must_implement_run(self):
        """Test that subclass must implement run method."""

        class IncompleteScript(ScriptBuilder):
            def __init__(self):
                super().__init__("test", "test")

        with pytest.raises(TypeError):
            IncompleteScript()
