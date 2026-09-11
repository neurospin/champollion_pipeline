#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for put_together_embeddings.py"""

from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings


class TestPutTogetherEmbeddingsInit:
    def test_init_creates_script(self):
        script = PutTogetherEmbeddings()
        assert script.script_name == "put_together_embeddings"
        assert "embeddings" in script.description.lower()


class TestPutTogetherEmbeddingsArguments:
    def test_parse_required_arguments(self, tmp_path):
        script = PutTogetherEmbeddings()
        args = script.parse_args([str(tmp_path), "--output_path", "/output"])
        assert args.embeddings_source == str(tmp_path)
        assert args.output_path == "/output"

    def test_missing_output_path_raises(self, tmp_path):
        script = PutTogetherEmbeddings()
        with pytest.raises(SystemExit):
            script.parse_args([str(tmp_path)])

    def test_missing_source_raises(self):
        script = PutTogetherEmbeddings()
        with pytest.raises(SystemExit):
            script.parse_args(["--output_path", "/out"])


class TestRunMethod:
    def test_run_copies_per_region_csvs(self, tmp_path):
        """run() copies {region}/full_embeddings.csv → {output}/{region}_embeddings.csv."""
        source = tmp_path / "source"
        source.mkdir()
        for region in ("SC-sylv_left", "SC-sylv_right"):
            d = source / region
            d.mkdir()
            (d / "full_embeddings.csv").write_text(f"data_{region}")

        output = tmp_path / "output"
        script = PutTogetherEmbeddings()
        script.parse_args([str(source), "--output_path", str(output)])
        result = script.run()

        assert result == 0
        assert (output / "SC-sylv_left_embeddings.csv").exists()
        assert (output / "SC-sylv_right_embeddings.csv").exists()
        assert (output / "SC-sylv_left_embeddings.csv").read_text() == "data_SC-sylv_left"

    def test_run_skips_regions_without_csv(self, tmp_path):
        """Regions with no full_embeddings.csv are skipped silently."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "SC-sylv_left").mkdir()  # no CSV
        (source / "SC-sylv_right").mkdir()
        (source / "SC-sylv_right" / "full_embeddings.csv").write_text("data")

        output = tmp_path / "output"
        script = PutTogetherEmbeddings()
        script.parse_args([str(source), "--output_path", str(output)])
        script.run()

        assert not (output / "SC-sylv_left_embeddings.csv").exists()
        assert (output / "SC-sylv_right_embeddings.csv").exists()

    def test_run_creates_output_directory(self, tmp_path):
        """run() creates the output directory if it does not exist."""
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "new_output"
        script = PutTogetherEmbeddings()
        script.parse_args([str(source), "--output_path", str(output)])
        script.run()
        assert output.is_dir()

    def test_run_returns_zero(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        script = PutTogetherEmbeddings()
        script.parse_args([str(source), "--output_path", str(tmp_path / "out")])
        assert script.run() == 0

    def test_run_ignores_non_directory_entries(self, tmp_path):
        """Files at the top level of embeddings_source are ignored."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "some_file.txt").write_text("noise")
        (source / "SC-sylv_left").mkdir()
        (source / "SC-sylv_left" / "full_embeddings.csv").write_text("data")

        output = tmp_path / "output"
        script = PutTogetherEmbeddings()
        script.parse_args([str(source), "--output_path", str(output)])
        script.run()

        assert (output / "SC-sylv_left_embeddings.csv").exists()
        assert not (output / "some_file_embeddings.csv").exists()


class TestMainFunction:
    def test_main_creates_script_and_runs(self, tmp_path):
        with patch("champollion_pipeline.put_together_embeddings.PutTogetherEmbeddings") as MockScript:
            mock_instance = MagicMock()
            mock_instance.build.return_value = mock_instance
            mock_instance.print_args.return_value = mock_instance
            mock_instance.run.return_value = 0
            MockScript.return_value = mock_instance

            from champollion_pipeline.put_together_embeddings import main

            with patch("sys.argv", ["script", str(tmp_path), "--output_path", str(tmp_path / "out")]):
                result = main()

            MockScript.assert_called_once()
            mock_instance.build.assert_called_once()
            mock_instance.print_args.assert_called_once()
            mock_instance.run.assert_called_once()
            assert result == 0


@pytest.mark.integration
class TestPutTogetherEmbeddingsIntegration:
    def test_full_workflow(self, tmp_path):
        source = tmp_path / "dataset_name_embeddings"
        source.mkdir()
        for region in ("SC-sylv_left", "SC-sylv_right", "FIP_left"):
            (source / region).mkdir()
            (source / region / "full_embeddings.csv").write_text("subject,emb\nS01,0.1")

        output = tmp_path / "combined"
        script = PutTogetherEmbeddings()
        script.parse_args([str(source), "--output_path", str(output)])
        result = script.run()

        assert result == 0
        csvs = list(output.glob("*_embeddings.csv"))
        assert len(csvs) == 3


@pytest.mark.smoke
class TestPutTogetherEmbeddingsSmoke:
    def test_script_can_be_instantiated(self):
        script = PutTogetherEmbeddings()
        assert script is not None

    def test_script_has_run_method(self):
        script = PutTogetherEmbeddings()
        assert hasattr(script, "run")
        assert callable(script.run)
