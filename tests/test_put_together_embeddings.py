#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for put_together_embeddings.py
"""

import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from put_together_embeddings import PutTogetherEmbeddings


class TestPutTogetherEmbeddingsInit:
    """Test initialization."""

    def test_init_creates_script(self):
        """Test that script is initialized correctly."""
        script = PutTogetherEmbeddings()
        assert script.script_name == "put_together_embeddings"
        assert "embeddings" in script.description.lower()


class TestPutTogetherEmbeddingsArguments:
    """Test argument parsing."""

    def test_parse_required_arguments(self):
        """Test parsing required arguments."""
        script = PutTogetherEmbeddings()
        args = script.parse_args([
            "--embeddings_subpath", "models/embeddings",
            "--output_path", "/output"
        ])
        assert args.embeddings_subpath == "models/embeddings"
        assert args.output_path == "/output"

    def test_path_models_has_default(self):
        """Test that path_models has default value."""
        script = PutTogetherEmbeddings()
        args = script.parse_args([
            "--embeddings_subpath", "emb",
            "--output_path", "/out"
        ])
        assert args.path_models is not None
        assert "Champollion" in args.path_models

    def test_path_models_can_be_overridden(self):
        """Test that path_models can be overridden."""
        script = PutTogetherEmbeddings()
        args = script.parse_args([
            "--embeddings_subpath", "emb",
            "--output_path", "/out",
            "--path_models", "/custom/models"
        ])
        assert args.path_models == "/custom/models"

    def test_missing_required_args_raises_error(self):
        """Test that missing required arguments raise error."""
        script = PutTogetherEmbeddings()
        with pytest.raises(SystemExit):
            script.parse_args(["--embeddings_subpath", "emb"])


class TestRunMethod:
    """Test the run method."""

    @patch('builtins.print')
    def test_run_prints_arguments(self, mock_print, temp_dir):
        """Test that run prints the arguments."""
        script = PutTogetherEmbeddings()
        script.parse_args([
            "--embeddings_subpath", "emb/path",
            "--output_path", temp_dir
        ])

        with patch('os.makedirs'):
            with patch.object(script, 'validate_paths', return_value=True):
                with patch('os.chdir'):
                    with patch('os.getcwd', return_value="/original"):
                        with patch.object(script, 'build_command', return_value=["cmd"]):
                            with patch.object(script, 'execute_command', return_value=0):
                                script.run()

                                # Check that paths were printed
                                print_calls = [str(c) for c in mock_print.call_args_list]
                                assert any("embeddings_subpath" in c for c in print_calls)
                                assert any("path_models" in c for c in print_calls)
                                assert any("output_path" in c for c in print_calls)

    def test_run_creates_output_directory(self, temp_dir):
        """Test that run creates output directory."""
        script = PutTogetherEmbeddings()
        output_dir = Path(temp_dir) / "new_output"
        script.parse_args([
            "--embeddings_subpath", "emb",
            "--output_path", str(output_dir)
        ])

        with patch('os.makedirs') as mock_makedirs:
            with patch.object(script, 'validate_paths', return_value=True):
                with patch('os.chdir'):
                    with patch('os.getcwd', return_value="/original"):
                        with patch.object(script, 'build_command', return_value=["cmd"]):
                            with patch.object(script, 'execute_command', return_value=0):
                                script.run()

                                mock_makedirs.assert_called_once_with(str(output_dir), exist_ok=True)

    def test_run_validates_paths(self, temp_dir):
        """Test that run validates paths."""
        script = PutTogetherEmbeddings()
        script.parse_args([
            "--embeddings_subpath", "emb",
            "--output_path", temp_dir
        ])

        with patch('os.makedirs'):
            with patch.object(script, 'validate_paths', return_value=False):
                with pytest.raises(ValueError, match="Please input valid paths"):
                    script.run()

    def test_run_changes_to_champollion_directory(self, temp_dir):
        """Test that run changes to champollion utils directory."""
        script = PutTogetherEmbeddings()
        script.parse_args([
            "--embeddings_subpath", "emb",
            "--output_path", temp_dir
        ])

        with patch('os.makedirs'):
            with patch.object(script, 'validate_paths', return_value=True):
                with patch('os.chdir') as mock_chdir:
                    with patch('os.getcwd', return_value="/original"):
                        with patch.object(script, 'build_command', return_value=["cmd"]):
                            with patch.object(script, 'execute_command', return_value=0):
                                script.run()

                                # Should change directory to champollion path
                                chdir_calls = [str(c) for c in mock_chdir.call_args_list]
                                assert any("champollion" in c.lower() for c in chdir_calls)

    def test_run_uses_build_command(self, temp_dir):
        """Test that run uses build_command."""
        script = PutTogetherEmbeddings()
        script.parse_args([
            "--embeddings_subpath", "emb/sub",
            "--output_path", temp_dir,
            "--path_models", "/models"
        ])

        with patch('os.makedirs'):
            with patch.object(script, 'validate_paths', return_value=True):
                with patch('os.chdir'):
                    with patch('os.getcwd', return_value="/original"):
                        with patch.object(script, 'build_command', return_value=["python", "script.py"]) as mock_build:
                            with patch.object(script, 'execute_command', return_value=0):
                                script.run()

                                mock_build.assert_called_once()
                                call_kwargs = mock_build.call_args[1]

                                assert call_kwargs['script_path'] == "put_together_embeddings_files.py"
                                assert "embeddings_subpath" in call_kwargs['required_args']
                                assert "output_path" in call_kwargs['required_args']
                                assert 'defaults' in call_kwargs

    def test_run_executes_command_without_shell(self, temp_dir):
        """Test that command is executed with shell=False."""
        script = PutTogetherEmbeddings()
        script.parse_args([
            "--embeddings_subpath", "emb",
            "--output_path", temp_dir
        ])

        with patch('os.makedirs'):
            with patch.object(script, 'validate_paths', return_value=True):
                with patch('os.chdir'):
                    with patch('os.getcwd', return_value="/original"):
                        with patch.object(script, 'build_command', return_value=["cmd"]):
                            with patch.object(script, 'execute_command', return_value=0) as mock_exec:
                                script.run()

                                assert mock_exec.call_args[1]['shell'] is False

    def test_run_returns_result(self, temp_dir):
        """Test that run returns command result."""
        script = PutTogetherEmbeddings()
        script.parse_args([
            "--embeddings_subpath", "emb",
            "--output_path", temp_dir
        ])

        with patch('os.makedirs'):
            with patch.object(script, 'validate_paths', return_value=True):
                with patch('os.chdir'):
                    with patch('os.getcwd', return_value="/original"):
                        with patch.object(script, 'build_command', return_value=["cmd"]):
                            with patch.object(script, 'execute_command', return_value=123):
                                result = script.run()
                                assert result == 123

    def test_run_restores_directory(self, temp_dir):
        """Test that original directory is restored."""
        script = PutTogetherEmbeddings()
        script.parse_args([
            "--embeddings_subpath", "emb",
            "--output_path", temp_dir
        ])
        original = "/original/dir"

        with patch('os.makedirs'):
            with patch.object(script, 'validate_paths', return_value=True):
                with patch('os.getcwd', return_value=original):
                    with patch('os.chdir') as mock_chdir:
                        with patch.object(script, 'build_command', return_value=["cmd"]):
                            with patch.object(script, 'execute_command', return_value=0):
                                script.run()

                                assert call(original) in mock_chdir.call_args_list


class TestMainFunction:
    """Test the main entry point."""

    def test_main_creates_script_and_runs(self, temp_dir):
        """Test that main creates script and calls build().print_args().run()."""
        with patch('put_together_embeddings.PutTogetherEmbeddings') as MockScript:
            mock_instance = MagicMock()
            mock_instance.build.return_value = mock_instance
            mock_instance.print_args.return_value = mock_instance
            mock_instance.run.return_value = 0
            MockScript.return_value = mock_instance

            from put_together_embeddings import main

            with patch('sys.argv', ['script', '--embeddings_subpath', 'emb', '--output_path', temp_dir]):
                result = main()

                MockScript.assert_called_once()
                mock_instance.build.assert_called_once()
                mock_instance.print_args.assert_called_once()
                mock_instance.run.assert_called_once()
                assert result == 0


@pytest.mark.integration
class TestPutTogetherEmbeddingsIntegration:
    """Integration tests."""

    def test_full_workflow(self, temp_dir):
        """Test complete workflow."""
        script = PutTogetherEmbeddings()
        output_dir = Path(temp_dir) / "output"
        models_dir = Path(temp_dir) / "models"
        models_dir.mkdir()

        script.parse_args([
            "--embeddings_subpath", "embeddings/v1",
            "--output_path", str(output_dir),
            "--path_models", str(models_dir)
        ])

        with patch.object(script, 'execute_command', return_value=0) as mock_exec:
            with patch('os.chdir'):
                with patch('os.getcwd', return_value="/original"):
                    result = script.run()

                    assert result == 0
                    mock_exec.assert_called_once()


@pytest.mark.smoke
class TestPutTogetherEmbeddingsSmoke:
    """Smoke tests."""

    def test_script_can_be_instantiated(self):
        """Test that script can be created."""
        script = PutTogetherEmbeddings()
        assert script is not None

    def test_script_has_run_method(self):
        """Test that script has run method."""
        script = PutTogetherEmbeddings()
        assert hasattr(script, 'run')
        assert callable(script.run)
