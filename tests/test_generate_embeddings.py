#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for generate_embeddings.py
"""

import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from generate_embeddings import GenerateEmbeddings


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

    def test_flags_can_be_set(self):
        """Test that flags can be set to True."""
        script = GenerateEmbeddings()
        args = script.parse_args([
            "/m", "loc", "/d", "name",
            "--overwrite",
            "--embeddings_only",
            "--use_best_model",
            "--verbose"
        ])
        assert args.overwrite is True
        assert args.embeddings_only is True
        assert args.use_best_model is True
        assert args.verbose is True

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
