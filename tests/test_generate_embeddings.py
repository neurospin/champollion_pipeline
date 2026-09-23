#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for generate_embeddings.py
"""

import os
import re
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings, HuggingFaceStrategy


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
        args = script.parse_args(["/models", "/datasets"])
        assert args.models_path == "/models"
        assert args.datasets_root == "/datasets"

    def test_flags_default_to_false(self):
        """Test that boolean flags default to False."""
        script = GenerateEmbeddings()
        args = script.parse_args(["/m", "/d"])
        assert args.overwrite is False
        assert args.cpu is False
        assert args.profiling is False
        assert args.run_cka is False
        assert args.no_cache is False

    def test_flags_can_be_set(self):
        """Test that flags can be set to True."""
        script = GenerateEmbeddings()
        args = script.parse_args(
            [
                "/m",
                "/d",
                "--overwrite",
                "--cpu",
                "--profiling",
                "--run-cka",
                "--no-cache",
            ]
        )
        assert args.overwrite is True
        assert args.cpu is True
        assert args.profiling is True
        assert args.run_cka is True
        assert args.no_cache is True


class TestPerRegionInvocation:
    """Test per-region evaluate.py invocation."""

    def test_run_per_region_calls_execute_for_each_region(self, tmp_path):
        """_run_per_region calls execute_command once per region directory."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "SC-sylv_left" / "logs").mkdir(parents=True)
        (models_dir / "SC-sylv_right" / "logs").mkdir(parents=True)

        script = GenerateEmbeddings()
        script.args = script.parse_args([str(models_dir), str(tmp_path)])

        executed_cmds = []

        def fake_execute(cmd, **kwargs):
            executed_cmds.append(cmd)
            return 0

        with patch.object(script, "execute_command", side_effect=fake_execute):
            script._run_per_region(
                evaluate_script="/eval.py",
                crops_2mm_dir=str(tmp_path / "crops"),
                subjects_path=str(tmp_path / "participants.tsv"),
                output_base=str(tmp_path / "out"),
            )

        assert len(executed_cmds) == 2

    def test_run_per_region_skips_existing_output_without_overwrite(self, tmp_path):
        """_run_per_region skips a region when output CSV already exists."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "SC-sylv_left" / "logs").mkdir(parents=True)
        out = tmp_path / "out" / "SC-sylv_left"
        out.mkdir(parents=True)
        (out / "full_embeddings.csv").write_text("existing")

        script = GenerateEmbeddings()
        script.args = script.parse_args([str(models_dir), str(tmp_path)])

        executed = []
        with patch.object(script, "execute_command", side_effect=lambda c, **k: executed.append(c) or 0):
            script._run_per_region(
                evaluate_script="/eval.py",
                crops_2mm_dir=str(tmp_path / "crops"),
                subjects_path=str(tmp_path / "participants.tsv"),
                output_base=str(tmp_path / "out"),
            )

        assert len(executed) == 0

    def test_run_per_region_runs_when_overwrite_set(self, tmp_path):
        """_run_per_region runs even when output exists if --overwrite is set."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "SC-sylv_left" / "logs").mkdir(parents=True)
        out = tmp_path / "out" / "SC-sylv_left"
        out.mkdir(parents=True)
        (out / "full_embeddings.csv").write_text("existing")

        script = GenerateEmbeddings()
        script.args = script.parse_args([str(models_dir), str(tmp_path), "--overwrite"])

        executed = []
        with patch.object(script, "execute_command", side_effect=lambda c, **k: executed.append(c) or 0):
            script._run_per_region(
                evaluate_script="/eval.py",
                crops_2mm_dir=str(tmp_path / "crops"),
                subjects_path=str(tmp_path / "participants.tsv"),
                output_base=str(tmp_path / "out"),
            )

        assert len(executed) == 1


@pytest.mark.unit
class TestModelDiscovery:
    """REQ-MODELDISCOVERY-01 — a directory is a region model iff it holds `logs/`.

    Real model folds on disk look like
    `<models_path>/<mask_version>/<REGION>_<side>/logs/best_model_weights.pt`
    (plus `.hydra/config.yaml`), so discovery must recurse to any depth and
    select only directories carrying the `logs/` marker.
    """

    @staticmethod
    def _make_model(path):
        """Create a directory shaped like a real Champollion model fold."""
        (path / "logs").mkdir(parents=True)
        (path / ".hydra").mkdir(parents=True)
        (path / ".hydra" / "config.yaml").write_text("model: {}\n")
        return path

    @staticmethod
    def _collect_model_args(script, tmp_path):
        """Run _run_per_region with execute_command stubbed; return the `-m` values."""
        executed = []

        def fake_execute(cmd, **kwargs):
            executed.append(cmd)
            return 0

        with patch.object(script, "execute_command", side_effect=fake_execute):
            script._run_per_region(
                evaluate_script="/eval.py",
                crops_2mm_dir=str(tmp_path / "crops"),
                subjects_path=str(tmp_path / "participants.tsv"),
                output_base=str(tmp_path / "out"),
            )

        return {os.path.realpath(cmd[cmd.index("-m") + 1]) for cmd in executed}

    def test_discovery_descends_into_mask_version_parent_directory(self, tmp_path):
        """Nested per-region model folds under a mask-version parent are each run."""
        models_dir = tmp_path / "models_cache"
        parent = models_dir / "canonical_corrected_26_1"
        left = self._make_model(parent / "FCLp-subsc-FCLa-INSULA_left")
        right = self._make_model(parent / "FCLp-subsc-FCLa-INSULA_right")

        script = GenerateEmbeddings()
        script.args = script.parse_args([str(models_dir), str(tmp_path)])

        model_args = self._collect_model_args(script, tmp_path)

        assert model_args == {os.path.realpath(str(left)), os.path.realpath(str(right))}

    def test_discovery_ignores_directory_without_logs_marker(self, tmp_path):
        """A stray `.cache/` leftover carries no `logs/`, so it is not a region."""
        models_dir = tmp_path / "models_cache"
        (models_dir / ".cache" / "huggingface").mkdir(parents=True)
        real = self._make_model(models_dir / "FCMpost-SpC_left")

        script = GenerateEmbeddings()
        script.args = script.parse_args([str(models_dir), str(tmp_path)])

        model_args = self._collect_model_args(script, tmp_path)

        assert model_args == {os.path.realpath(str(real))}

    @pytest.mark.unit
    def test_discovery_skips_dot_directory_subtree(self, tmp_path):
        """REQ-MODELDISCOVERY-02 — a dot-directory and its whole subtree are skipped.

        The Hugging Face Hub download-staging cache
        (`<models_path>/.cache/huggingface/download/<mask_version>/<REGION>/`)
        mirrors the real model tree, `logs/` marker included, but holds only
        `.pt.metadata` bookkeeping files. Matching it yields stale duplicates
        that later blow up in evaluate.py.
        """
        models_dir = tmp_path / "models_cache"
        real = self._make_model(models_dir / "canonical_corrected_26_1" / "FCMpost-SpC_left")
        mirror = self._make_model(
            models_dir / ".cache" / "huggingface" / "download" / "canonical_corrected_26_1" / "FCMpost-SpC_left"
        )
        (mirror / "logs" / "best_model_weights.pt.metadata").write_text("{}\n")

        script = GenerateEmbeddings()
        script.args = script.parse_args([str(models_dir), str(tmp_path)])

        found = script._find_region_model_dirs(str(models_dir))

        assert found == [str(real)]


class TestRunMethod:
    """Test the run method."""

    def test_run_calls_run_per_region(self, tmp_path):
        """run() delegates to _run_per_region."""
        (tmp_path / "SC-sylv_left").mkdir()

        script = GenerateEmbeddings()
        script.parse_args([str(tmp_path), str(tmp_path)])

        with patch.object(script, "fetch_models", return_value=str(tmp_path)):
            with patch.object(script, "_run_per_region", return_value=0) as mock_per:
                with patch.object(script, "_find_subjects_file", return_value=str(tmp_path / "participants.tsv")):
                    script.run()
        mock_per.assert_called_once()

    def test_run_returns_result(self, tmp_path):
        """run() returns the value from _run_per_region."""
        (tmp_path / "SC-sylv_left").mkdir()

        script = GenerateEmbeddings()
        script.parse_args([str(tmp_path), str(tmp_path)])

        with patch.object(script, "fetch_models", return_value=str(tmp_path)):
            with patch.object(script, "_run_per_region", return_value=99):
                with patch.object(script, "_find_subjects_file", return_value=str(tmp_path / "p.tsv")):
                    result = script.run()
        assert result == 99


@pytest.mark.integration
class TestGenerateEmbeddingsIntegration:
    """Integration tests."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow."""
        (tmp_path / "SC-sylv_left").mkdir()

        script = GenerateEmbeddings()
        script.parse_args([str(tmp_path), str(tmp_path), "--overwrite"])

        with patch.object(script, "fetch_models", return_value=str(tmp_path)):
            with patch.object(script, "_run_per_region", return_value=0) as mock_run:
                with patch.object(script, "_find_subjects_file", return_value=str(tmp_path / "participants.tsv")):
                    result = script.run()

        assert result == 0
        mock_run.assert_called_once()


class TestCorticalVersionFlags:
    """Tests for --cortical_version and --legacy flags."""

    def test_cortical_version_default(self):
        from champollion_pipeline.utils.lib import CORTICAL_TILES_VERSION

        script = GenerateEmbeddings()
        args = script.parse_args(["/m", "/d"])
        assert args.cortical_version == f"cortical_tiles-{CORTICAL_TILES_VERSION}"

    def test_cortical_version_can_be_overridden(self):
        script = GenerateEmbeddings()
        args = script.parse_args(["/m", "/d", "--cortical_version", "deep_folding-2025"])
        assert args.cortical_version == "deep_folding-2025"

    def test_legacy_flag_defaults_false(self):
        script = GenerateEmbeddings()
        args = script.parse_args(["/m", "/d"])
        assert args.legacy is False

    def test_legacy_flag_can_be_set(self):
        script = GenerateEmbeddings()
        args = script.parse_args(["/m", "/d", "--legacy"])
        assert args.legacy is True

    def test_get_derivatives_folder_returns_cortical_version_by_default(self):
        from champollion_pipeline.utils.lib import CORTICAL_TILES_VERSION

        script = GenerateEmbeddings()
        script.args = script.parse_args(["/m", "/d"])
        assert script._get_derivatives_folder() == f"cortical_tiles-{CORTICAL_TILES_VERSION}"

    def test_get_derivatives_folder_legacy_overrides(self):
        script = GenerateEmbeddings()
        script.args = script.parse_args(["/m", "/d", "--legacy"])
        assert script._get_derivatives_folder() == "deep_folding-2025"

    def test_get_derivatives_folder_cortical_version_overrides(self):
        script = GenerateEmbeddings()
        script.args = script.parse_args(["/m", "/d", "--cortical_version", "cortical_tiles-2027"])
        assert script._get_derivatives_folder() == "cortical_tiles-2027"

    def test_patch_config_paths_rewrites_derivatives_folder(self, tmp_path):
        yaml_file = tmp_path / "SC-sylv_left.yaml"
        yaml_file.write_text("numpy_all: /data/derivatives/deep_folding-2025/crops/2mm/S.C.-sylv./mask/Lskeleton.npy\n")

        script = GenerateEmbeddings()
        script._patch_config_paths(str(tmp_path), "cortical_tiles-2026")

        content = yaml_file.read_text()
        assert "cortical_tiles-2026" in content
        assert "deep_folding-2025" not in content

    def test_patch_config_paths_leaves_unrelated_lines_intact(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        original = (
            "numpy_all: /data/derivatives/deep_folding-2025/crops/2mm/region/mask/L.npy\n"
            "dataset_folder: /some/unrelated/path\n"
        )
        yaml_file.write_text(original)

        script = GenerateEmbeddings()
        script._patch_config_paths(str(tmp_path), "cortical_tiles-2026")

        content = yaml_file.read_text()
        assert "dataset_folder: /some/unrelated/path" in content

    def test_patch_config_paths_skips_already_correct_files(self, tmp_path):
        from champollion_pipeline.utils.lib import CORTICAL_TILES_VERSION

        yaml_file = tmp_path / "config.yaml"
        original = f"numpy_all: /data/derivatives/cortical_tiles-{CORTICAL_TILES_VERSION}/crops/2mm/L.npy\n"
        yaml_file.write_text(original)
        mtime_before = yaml_file.stat().st_mtime

        script = GenerateEmbeddings()
        script._patch_config_paths(str(tmp_path), f"cortical_tiles-{CORTICAL_TILES_VERSION}")

        assert yaml_file.stat().st_mtime == mtime_before

    def test_patch_config_paths_recurses_into_subdirs(self, tmp_path):
        subdir = tmp_path / "dataset" / "mydata"
        subdir.mkdir(parents=True)
        yaml_file = subdir / "region.yaml"
        yaml_file.write_text("path: /x/derivatives/old_folder/crops/2mm/r.npy\n")

        script = GenerateEmbeddings()
        script._patch_config_paths(str(tmp_path), "new_folder")

        assert "new_folder" in yaml_file.read_text()


class TestRegionsFilter:
    """Tests for --regions filtering."""

    def test_regions_default_none(self):
        script = GenerateEmbeddings()
        args = script.parse_args(["/m", "/d"])
        assert args.regions is None

    def test_regions_single(self):
        script = GenerateEmbeddings()
        args = script.parse_args(["/m", "/d", "--regions", "SC-sylv_left"])
        assert args.regions == ["SC-sylv_left"]

    def test_regions_multiple(self):
        script = GenerateEmbeddings()
        args = script.parse_args(
            ["/m", "/d", "--regions", "SC-sylv_left", "SC-sylv_right", "FIP-FIPPoCinf_left"]
        )
        assert args.regions == ["SC-sylv_left", "SC-sylv_right", "FIP-FIPPoCinf_left"]

    def test_make_regions_tmpdir_creates_symlinks(self, tmp_path):
        (tmp_path / "SC-sylv_left").mkdir()
        (tmp_path / "SC-sylv_right").mkdir()
        (tmp_path / "other_region").mkdir()

        script = GenerateEmbeddings()
        script.args = script.parse_args(
            [str(tmp_path), "/d", "--regions", "SC-sylv_left", "SC-sylv_right"]
        )

        tmpdir = script._make_regions_tmpdir(str(tmp_path))
        try:
            entries = sorted(os.listdir(tmpdir))
            assert entries == ["SC-sylv_left", "SC-sylv_right"]
            assert os.path.islink(os.path.join(tmpdir, "SC-sylv_left"))
            assert "other_region" not in entries
        finally:
            import shutil

            shutil.rmtree(tmpdir)

    def test_make_regions_tmpdir_raises_on_missing_region(self, tmp_path):
        (tmp_path / "SC-sylv_left").mkdir()

        script = GenerateEmbeddings()
        script.args = script.parse_args(
            [str(tmp_path), "/d", "--regions", "SC-sylv_left", "nonexistent_region"]
        )

        with pytest.raises(FileNotFoundError, match="nonexistent_region"):
            script._make_regions_tmpdir(str(tmp_path))

    def test_run_pipeline_uses_tmpdir_when_regions_set(self, tmp_path):
        (tmp_path / "SC-sylv_left").mkdir()
        (tmp_path / "SC-sylv_right").mkdir()

        script = GenerateEmbeddings()
        script.parse_args([str(tmp_path), str(tmp_path), "--regions", "SC-sylv_left"])

        # Capture models_path seen by _run_per_region
        snapshot = {}

        def fake_per_region(evaluate_script, crops_2mm_dir, subjects_path, output_base):
            snapshot["path"] = script.args.models_path
            snapshot["entries"] = sorted(os.listdir(script.args.models_path))
            snapshot["is_symlink"] = os.path.islink(os.path.join(script.args.models_path, "SC-sylv_left"))
            return 0

        with patch.object(script, "fetch_models", return_value=str(tmp_path)):
            with patch.object(script, "_run_per_region", side_effect=fake_per_region):
                with patch.object(script, "_find_subjects_file", return_value=str(tmp_path / "p.tsv")):
                    script.run()

        assert snapshot["path"] != str(tmp_path), "should use a tmpdir, not the original path"
        assert snapshot["entries"] == ["SC-sylv_left"], "only requested region should appear"
        assert snapshot["is_symlink"], "region entry should be a symlink"
        assert not os.path.exists(snapshot["path"]), "tmpdir should be cleaned up after run"

    def test_run_pipeline_cleans_up_tmpdir_on_error(self, tmp_path):
        (tmp_path / "SC-sylv_left").mkdir()

        script = GenerateEmbeddings()
        script.parse_args([str(tmp_path), str(tmp_path), "--regions", "SC-sylv_left"])

        created_tmpdirs = []

        original_make = script._make_regions_tmpdir

        def tracking_make(models_path):
            d = original_make(models_path)
            created_tmpdirs.append(d)
            return d

        with patch.object(script, "fetch_models", return_value=str(tmp_path)):
            with patch.object(script, "_make_regions_tmpdir", side_effect=tracking_make):
                with patch.object(script, "_run_per_region", side_effect=RuntimeError("boom")):
                    with patch.object(script, "_find_subjects_file", return_value=str(tmp_path / "p.tsv")):
                        with pytest.raises(RuntimeError):
                            script.run()

        assert len(created_tmpdirs) == 1
        assert not os.path.exists(created_tmpdirs[0]), "tmpdir must be cleaned up even on error"

    def test_run_pipeline_no_tmpdir_without_regions(self, temp_dir):
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, temp_dir])

        with patch.object(script, "fetch_models", return_value=temp_dir):
            with patch.object(script, "_run_per_region", return_value=0):
                with patch.object(script, "_find_subjects_file", return_value=temp_dir + "/p.tsv"):
                    with patch.object(script, "_make_regions_tmpdir") as mock_make:
                        script.run()
                        mock_make.assert_not_called()


class TestProfiling:
    """Tests for --profiling / cProfile behavior."""

    def test_run_dispatches_to_profiling_when_flag_set(self, temp_dir):
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, temp_dir, "--profiling"])

        with patch.object(script, "_validate_inputs"):
            with patch.object(script, "_run_with_profiling", return_value=0) as mock_prof:
                script.run()
                mock_prof.assert_called_once()

    def test_run_dispatches_to_normal_without_profiling_flag(self, temp_dir):
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, temp_dir])

        with patch.object(script, "_validate_inputs"):
            with patch.object(script, "_run_normal", return_value=0) as mock_normal:
                script.run()
                mock_normal.assert_called_once()

    def test_run_with_profiling_calls_run_normal(self, temp_dir):
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, temp_dir, "--profiling"])

        with patch("champollion_pipeline.generate_embeddings.cProfile.Profile"):
            with patch("champollion_pipeline.generate_embeddings.pstats.Stats", return_value=MagicMock()):
                with patch.object(script, "_run_normal", return_value=42) as mock_normal:
                    result = script._run_with_profiling()
                    mock_normal.assert_called_once()
                    assert result == 42

    def test_run_with_profiling_dumps_profile_file(self, temp_dir):
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, temp_dir, "--profiling"])

        mock_stats = MagicMock()

        with patch("champollion_pipeline.generate_embeddings.cProfile.Profile"):
            with patch("champollion_pipeline.generate_embeddings.pstats.Stats", return_value=mock_stats):
                with patch.object(script, "_run_normal", return_value=0):
                    script._run_with_profiling()
                    mock_stats.dump_stats.assert_called_once_with("embeddings_profile.prof")

    def test_run_with_profiling_dumps_on_error(self, temp_dir):
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, temp_dir, "--profiling"])

        mock_stats = MagicMock()

        with patch("champollion_pipeline.generate_embeddings.cProfile.Profile"):
            with patch("champollion_pipeline.generate_embeddings.pstats.Stats", return_value=mock_stats):
                with patch.object(script, "_run_normal", side_effect=RuntimeError("boom")):
                    with pytest.raises(RuntimeError):
                        script._run_with_profiling()
                    mock_stats.dump_stats.assert_called_once_with("embeddings_profile.prof")


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
        assert hasattr(script, "run")
        assert callable(script.run)


@pytest.mark.unit
class TestHuggingFaceStrategy:
    """Unit tests for HuggingFaceStrategy.fetch() — regression for snapshot_download bugs."""

    def _make_strategy(self, subfolder=None):
        return HuggingFaceStrategy(subfolder=subfolder)

    def test_fetch_without_subfolder_calls_snapshot_download_no_allow_patterns(self, tmp_path):
        strategy = self._make_strategy(subfolder=None)
        with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as mock_dl:
            strategy.fetch("neurospin/Champollion_V1", str(tmp_path))
        call_kwargs = mock_dl.call_args[1]
        assert "subfolder" not in call_kwargs
        assert call_kwargs.get("allow_patterns") is None

    def test_fetch_with_subfolder_uses_allow_patterns(self, tmp_path):
        strategy = self._make_strategy(subfolder="canonical_corrected_26_1")
        with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as mock_dl:
            strategy.fetch("neurospin/Champollion_V1", str(tmp_path))
        call_kwargs = mock_dl.call_args[1]
        assert "subfolder" not in call_kwargs
        assert call_kwargs["allow_patterns"] == ["canonical_corrected_26_1/*"]

    def test_fetch_never_passes_subfolder_kwarg(self, tmp_path):
        for subfolder in [None, "canonical_25", "canonical_corrected_26_1"]:
            strategy = self._make_strategy(subfolder=subfolder)
            with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as mock_dl:
                strategy.fetch("neurospin/Champollion_V1", str(tmp_path))
            assert "subfolder" not in mock_dl.call_args[1], (
                f"subfolder kwarg must never be passed (subfolder={subfolder!r})"
            )

    def test_fetch_with_subfolder_returns_path_including_subfolder(self, tmp_path):
        """REQ-HF-01: with a non-empty subfolder, fetch() returns snapshot_root/subfolder."""
        snapshot_root = str(tmp_path / "fake_root")
        strategy = self._make_strategy(subfolder="canonical_25")
        with patch("huggingface_hub.snapshot_download", return_value=snapshot_root):
            resolved = strategy.fetch("neurospin/Champollion_V1", str(tmp_path))
        assert resolved == os.path.join(snapshot_root, "canonical_25")

    def test_fetch_without_subfolder_returns_snapshot_root_unchanged(self, tmp_path):
        """REQ-HF-01: with no subfolder, fetch() returns the snapshot root unchanged."""
        snapshot_root = str(tmp_path / "fake_root")
        strategy = self._make_strategy(subfolder=None)
        with patch("huggingface_hub.snapshot_download", return_value=snapshot_root):
            resolved = strategy.fetch("neurospin/Champollion_V1", str(tmp_path))
        assert resolved == snapshot_root


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
        import pathlib
        import tomllib

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


@pytest.mark.unit
class TestEnsureCkptShape:
    """REQ-CKPTSHAPE-01 — `_ensure_ckpt` must not double-wrap an already-wrapped `.pt`.

    Real `logs/best_model_weights.pt` files ship as Lightning-style
    `{"state_dict": {...<backbones.0.encoder.*> weights...}}` dicts. Wrapping one
    again yields `{"state_dict": {"state_dict": {...}}}`, so
    `external/champollion_V1/champollion/evaluate.py`'s prefix filter over
    `checkpoint["state_dict"]` sees only the literal key `"state_dict"` and
    `model.load_state_dict({})` raises `RuntimeError: Missing key(s)`.
    """

    ENCODER_KEYS = (
        "backbones.0.encoder.layer1.0.weight",
        "backbones.0.encoder.layer1.0.bias",
    )

    @staticmethod
    def _make_model_dir(tmp_path, payload):
        """Write `logs/best_model_weights.pt` holding `payload`; return the model dir."""
        import torch

        model_dir = tmp_path / "canonical_corrected_26_1" / "SsP-SPaint_left"
        (model_dir / "logs").mkdir(parents=True)
        torch.save(payload, str(model_dir / "logs" / "best_model_weights.pt"))
        return model_dir

    @staticmethod
    def _load_converted_ckpt(model_dir):
        """Load the `.ckpt` `_ensure_ckpt` was expected to produce."""
        import torch

        ckpt_path = model_dir / "logs" / "lightning_logs" / "version_0" / "checkpoints" / "best_model.ckpt"
        assert ckpt_path.exists(), f"_ensure_ckpt produced no checkpoint at {ckpt_path}"
        return torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    def _weights(self):
        import torch

        return {key: torch.zeros(2) for key in self.ENCODER_KEYS}

    def test_already_wrapped_pt_is_not_wrapped_again(self, tmp_path):
        """A `{"state_dict": ...}`-shaped `.pt` converts to a ckpt with no nested wrap."""
        model_dir = self._make_model_dir(tmp_path, {"state_dict": self._weights()})

        script = GenerateEmbeddings()
        script._ensure_ckpt(str(model_dir))

        ckpt = self._load_converted_ckpt(model_dir)
        assert "state_dict" not in ckpt["state_dict"], (
            "ckpt['state_dict'] is itself wrapped in another 'state_dict' key; "
            "evaluate.py's prefix filter will match nothing"
        )

    def test_already_wrapped_pt_keeps_encoder_keys_at_top_level(self, tmp_path):
        """The converted ckpt exposes the real `backbones.0.encoder.*` keys directly."""
        model_dir = self._make_model_dir(tmp_path, {"state_dict": self._weights()})

        script = GenerateEmbeddings()
        script._ensure_ckpt(str(model_dir))

        ckpt = self._load_converted_ckpt(model_dir)
        assert set(ckpt["state_dict"]) == set(self.ENCODER_KEYS)

    def test_bare_state_dict_pt_is_still_wrapped_once(self, tmp_path):
        """A `.pt` holding bare weights (no `state_dict` key) is still wrapped exactly once."""
        model_dir = self._make_model_dir(tmp_path, self._weights())

        script = GenerateEmbeddings()
        script._ensure_ckpt(str(model_dir))

        ckpt = self._load_converted_ckpt(model_dir)
        assert set(ckpt["state_dict"]) == set(self.ENCODER_KEYS)
