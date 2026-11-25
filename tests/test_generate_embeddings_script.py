#!/usr/bin/env python3
"""
Unit tests for generate_embeddings.py: script set-up, run flow, legacy crops path,
integration, profiling and pixi task paths.

Split from tests/test_generate_embeddings.py (TASK-CODE-BDRABCZUK-73BD87657EA2).
"""

import os
import pathlib
import re
from unittest.mock import MagicMock, patch

import pytest
import tomllib

from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from champollion_pipeline.utils.lib import CORTICAL_TILES_VERSION
from tests.embeddings_parse_helpers import MASKS_ARGS


class TestGenerateEmbeddingsInit:
    """Test initialization."""

    def test_init_creates_script(self):
        """Test that script is initialized correctly."""
        script = GenerateEmbeddings()
        assert script.script_name == "generate_embeddings"
        assert "embeddings" in script.description.lower()


class TestRunMethod:
    """Test the run method."""

    def test_run_calls_run_per_region(self, tmp_path):
        """run() delegates to _run_per_region."""
        (tmp_path / "SC-sylv_left").mkdir()

        script = GenerateEmbeddings()
        script.parse_args([str(tmp_path), str(tmp_path), *MASKS_ARGS])

        with (
            patch.object(script, "fetch_models", return_value=str(tmp_path)),
            patch.object(script, "_run_per_region", return_value=0) as mock_per,
        ):
            script.run()
        mock_per.assert_called_once()

    def test_run_returns_result(self, tmp_path):
        """run() returns the value from _run_per_region."""
        (tmp_path / "SC-sylv_left").mkdir()

        script = GenerateEmbeddings()
        script.parse_args([str(tmp_path), str(tmp_path), *MASKS_ARGS])

        with (
            patch.object(script, "fetch_models", return_value=str(tmp_path)),
            patch.object(script, "_run_per_region", return_value=99),
        ):
            result = script.run()
        assert result == 99


class TestLegacyCropsPath:
    """REQ-LEGACYPATH-01: --legacy reads crops from the flat deep_folding-2025/crops/2mm layout."""

    def _captured_crops_dir(self, tmp_path, extra_args):
        """Run run() with _run_per_region stubbed; return the crops_2mm_dir it received."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        datasets_root = tmp_path / "dataset"
        datasets_root.mkdir()

        script = GenerateEmbeddings()
        script.parse_args([str(models_dir), str(datasets_root), *MASKS_ARGS, *extra_args])

        with (
            patch.object(script, "fetch_models", return_value=str(models_dir)),
            patch.object(script, "_run_per_region", return_value=0) as mock_per,
        ):
            script.run()
        mock_per.assert_called_once()
        return mock_per.call_args.args[1], str(datasets_root)

    def test_legacy_crops_dir_has_no_mask_version_segment(self, tmp_path):
        crops_dir, root = self._captured_crops_dir(tmp_path, ["--legacy"])
        assert crops_dir == os.path.join(root, "derivatives", "deep_folding-2025", "crops", "2mm")

    def test_legacy_crops_dir_ignores_explicit_masks_option(self, tmp_path):
        crops_dir, root = self._captured_crops_dir(tmp_path, ["--legacy", "--masks", "canonical_25"])
        assert crops_dir == os.path.join(root, "derivatives", "deep_folding-2025", "crops", "2mm")

    def test_non_legacy_crops_dir_keeps_mask_version_segment(self, tmp_path):
        crops_dir, root = self._captured_crops_dir(tmp_path, [])
        assert crops_dir == os.path.join(
            root, "derivatives", f"cortical_tiles-{CORTICAL_TILES_VERSION}", "crops", "canonical_25", "2mm"
        )


@pytest.mark.integration
class TestGenerateEmbeddingsIntegration:
    """Integration tests."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow."""
        (tmp_path / "SC-sylv_left").mkdir()

        script = GenerateEmbeddings()
        script.parse_args([str(tmp_path), str(tmp_path), *MASKS_ARGS, "--overwrite"])

        with (
            patch.object(script, "fetch_models", return_value=str(tmp_path)),
            patch.object(script, "_run_per_region", return_value=0) as mock_run,
        ):
            result = script.run()

        assert result == 0
        mock_run.assert_called_once()


class TestProfiling:
    """Tests for --profiling / cProfile behavior."""

    def test_run_dispatches_to_profiling_when_flag_set(self, temp_dir):
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, temp_dir, *MASKS_ARGS, "--profiling"])

        with (
            patch.object(script, "_validate_inputs"),
            patch.object(script, "_run_with_profiling", return_value=0) as mock_prof,
        ):
            script.run()
            mock_prof.assert_called_once()

    def test_run_dispatches_to_normal_without_profiling_flag(self, temp_dir):
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, temp_dir, *MASKS_ARGS])

        with (
            patch.object(script, "_validate_inputs"),
            patch.object(script, "_run_normal", return_value=0) as mock_normal,
        ):
            script.run()
            mock_normal.assert_called_once()

    def test_run_with_profiling_calls_run_normal(self, temp_dir):
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, temp_dir, *MASKS_ARGS, "--profiling"])

        with (
            patch("champollion_pipeline.generate_embeddings.cProfile.Profile"),
            patch("champollion_pipeline.generate_embeddings.pstats.Stats", return_value=MagicMock()),
            patch.object(script, "_run_normal", return_value=42) as mock_normal,
        ):
            result = script._run_with_profiling()
            mock_normal.assert_called_once()
            assert result == 42

    def test_run_with_profiling_dumps_profile_file(self, temp_dir):
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, temp_dir, *MASKS_ARGS, "--profiling"])

        mock_stats = MagicMock()

        with (
            patch("champollion_pipeline.generate_embeddings.cProfile.Profile"),
            patch("champollion_pipeline.generate_embeddings.pstats.Stats", return_value=mock_stats),
            patch.object(script, "_run_normal", return_value=0),
        ):
            script._run_with_profiling()
            mock_stats.dump_stats.assert_called_once_with("embeddings_profile.prof")

    def test_run_with_profiling_dumps_on_error(self, temp_dir):
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, temp_dir, *MASKS_ARGS, "--profiling"])

        mock_stats = MagicMock()

        with (
            patch("champollion_pipeline.generate_embeddings.cProfile.Profile"),
            patch("champollion_pipeline.generate_embeddings.pstats.Stats", return_value=mock_stats),
            patch.object(script, "_run_normal", side_effect=RuntimeError("boom")),
        ):
            with pytest.raises(RuntimeError):
                script._run_with_profiling()
            mock_stats.dump_stats.assert_called_once_with("embeddings_profile.prof")


@pytest.mark.unit
class TestPixiTaskPaths:
    """REQ-PIXITASKS-01 — every pixi task invoking a `python3 src/...` script
    must point at a file that actually exists.

    Rather than hardcoding a handful of known task names (which silently stops
    covering a task the moment it's renamed or a new one is added), this walks
    the whole parsed ``pixi.toml`` — top-level ``[tasks]`` and every
    ``[feature.<name>.tasks]`` table — and extracts every ``python3 src/...``
    invocation automatically.
    """

    @staticmethod
    def _task_command(task):
        """Return a task's shell command, whether it is a string or a table."""
        if isinstance(task, str):
            return task
        if isinstance(task, dict):
            return task.get("cmd", "")
        raise TypeError(f"unexpected pixi task type: {type(task)!r}")

    @classmethod
    def _iter_task_commands(cls, pixi_config):
        """Yield (location, task_name, command) for every task in pixi.toml."""
        for task_name, task in pixi_config.get("tasks", {}).items():
            yield "[tasks]", task_name, cls._task_command(task)
        for feature_name, feature in pixi_config.get("feature", {}).items():
            for task_name, task in feature.get("tasks", {}).items():
                yield f"[feature.{feature_name}.tasks]", task_name, cls._task_command(task)

    @classmethod
    def _iter_src_script_paths(cls, pixi_config):
        """Yield (location, task_name, rel_path) for every `python3 src/...py` invocation."""
        pattern = re.compile(r"python3\s+(src/\S+\.py)")
        for location, task_name, command in cls._iter_task_commands(pixi_config):
            for rel_path in pattern.findall(command):
                yield location, task_name, rel_path

    def test_all_pixi_task_scripts_exist(self):

        repo_root = pathlib.Path(__file__).parent.parent
        with (repo_root / "pixi.toml").open("rb") as handle:
            pixi_config = tomllib.load(handle)

        checked = 0
        for location, task_name, rel_path in self._iter_src_script_paths(pixi_config):
            checked += 1
            script = repo_root / rel_path
            assert script.exists(), (
                f"pixi task '{task_name}' under {location} points to '{rel_path}' which does not exist"
            )
        assert checked >= 5, "expected to find at least the known python3 src/... pixi tasks; parsing regressed"
