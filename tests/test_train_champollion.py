#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for train_champollion.py

All external side effects (subprocess execution, chdir, directory creation)
are mocked: no champollion_V1 checkout, GPU or cluster is required.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline import train_champollion
from champollion_pipeline.train_champollion import TrainChampollion, main


def make_script(argv):
    """Return a TrainChampollion with parsed arguments (no update check)."""
    script = TrainChampollion()
    script.parse_args(argv)
    return script


BASE_ARGS = ["--dataset", "TEST01", "--region", "cingulate_left"]


class TestTrainChampollionInit:
    """Test initialization and argument registration."""

    def test_init_sets_script_name(self):
        script = TrainChampollion()
        assert script.script_name == "train_champollion"
        assert "champollion_V1" in script.description

    def test_required_arguments_are_parsed(self):
        args = make_script(BASE_ARGS).args
        assert args.dataset == "TEST01"
        assert args.region == "cingulate_left"

    def test_defaults(self):
        args = make_script(BASE_ARGS).args
        assert args.mode == "encoder"
        assert args.localization == "local"
        assert args.config_dir is None
        assert args.output_dir is None
        assert args.njobs is None
        assert args.cpu is False
        assert args.overwrite is False
        assert args.load_sparse is False
        assert args.swf is False

    def test_flags_can_be_set(self):
        args = make_script(BASE_ARGS + ["--cpu", "--overwrite", "--load-sparse", "--swf"]).args
        assert args.cpu is True
        assert args.overwrite is True
        assert args.load_sparse is True
        assert args.swf is True

    def test_njobs_is_int(self):
        """n_jobs/soma-workflow compatibility: --njobs must parse as an int."""
        args = make_script(BASE_ARGS + ["--njobs", "12"]).args
        assert args.njobs == 12


class TestResolveOutputDir:
    """Test _resolve_output_dir."""

    def test_explicit_output_dir_is_made_absolute(self, temp_dir):
        script = make_script(BASE_ARGS + ["--output_dir", temp_dir])
        assert script._resolve_output_dir() == os.path.abspath(temp_dir)

    def test_relative_output_dir_is_made_absolute(self):
        script = make_script(BASE_ARGS + ["--output_dir", "some/relative/dir"])
        resolved = script._resolve_output_dir()
        assert os.path.isabs(resolved)
        assert resolved.endswith("some/relative/dir")

    def test_default_output_dir_uses_derivatives_tree(self):
        resolved = make_script(BASE_ARGS)._resolve_output_dir()
        assert resolved.endswith("data/TEST01/derivatives/champollion_V1/models/cingulate_left")
        assert os.path.isabs(resolved)


class TestValidateInputs:
    """Test _validate_inputs."""

    def _write_builtin_config(self, tmp_path, dataset="TEST01", region="cingulate_left"):
        path = Path(tmp_path) / "configs" / "dataset" / dataset
        path.mkdir(parents=True)
        (path / f"{region}.yaml").write_text("x: 1\n")

    def test_passes_when_builtin_config_exists(self, tmp_path, monkeypatch):
        self._write_builtin_config(tmp_path)
        monkeypatch.setattr(train_champollion, "_CONTRASTIVE_DIR", str(tmp_path))
        make_script(BASE_ARGS)._validate_inputs()  # must not raise

    def test_raises_when_builtin_config_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(train_champollion, "_CONTRASTIVE_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Dataset config not found"):
            make_script(BASE_ARGS)._validate_inputs()

    def test_error_mentions_config_dir_hint_when_not_supplied(self, tmp_path, monkeypatch):
        monkeypatch.setattr(train_champollion, "_CONTRASTIVE_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="--config-dir"):
            make_script(BASE_ARGS)._validate_inputs()

    def test_passes_when_local_config_dir_has_the_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(train_champollion, "_CONTRASTIVE_DIR", str(tmp_path / "missing"))
        config_dir = tmp_path / "myconfigs"
        (config_dir / "dataset" / "TEST01").mkdir(parents=True)
        (config_dir / "dataset" / "TEST01" / "cingulate_left.yaml").write_text("x: 1\n")
        make_script(BASE_ARGS + ["--config-dir", str(config_dir)])._validate_inputs()

    def test_passes_when_config_dir_given_but_builtin_has_the_config(self, tmp_path, monkeypatch):
        self._write_builtin_config(tmp_path / "builtin")
        monkeypatch.setattr(train_champollion, "_CONTRASTIVE_DIR", str(tmp_path / "builtin"))
        make_script(BASE_ARGS + ["--config-dir", str(tmp_path / "empty")])._validate_inputs()

    def test_raises_listing_both_locations_when_neither_has_the_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(train_champollion, "_CONTRASTIVE_DIR", str(tmp_path / "builtin"))
        with pytest.raises(FileNotFoundError, match="either location"):
            make_script(BASE_ARGS + ["--config-dir", str(tmp_path / "empty")])._validate_inputs()

    @pytest.mark.parametrize("mode", ["encoder", "classifier", "regresser"])
    def test_accepts_valid_modes(self, tmp_path, monkeypatch, mode):
        self._write_builtin_config(tmp_path)
        monkeypatch.setattr(train_champollion, "_CONTRASTIVE_DIR", str(tmp_path))
        make_script(BASE_ARGS + ["--mode", mode])._validate_inputs()

    def test_rejects_invalid_mode(self, tmp_path, monkeypatch):
        self._write_builtin_config(tmp_path)
        monkeypatch.setattr(train_champollion, "_CONTRASTIVE_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="Invalid --mode"):
            make_script(BASE_ARGS + ["--mode", "diffusion"])._validate_inputs()


class TestRunCudaHandling:
    """Test run() and its CUDA_VISIBLE_DEVICES save/restore contract."""

    def _prepare(self, script):
        script._validate_inputs = lambda: None
        script._run_training = lambda local_dir: 0
        return script

    def test_returns_training_result(self):
        script = self._prepare(make_script(BASE_ARGS))
        script._run_training = lambda local_dir: 7
        assert script.run() == 7

    def test_cpu_flag_sets_cuda_visible_devices_empty(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        seen = {}
        script = self._prepare(make_script(BASE_ARGS + ["--cpu"]))
        script._run_training = lambda local_dir: seen.setdefault("cuda", os.environ["CUDA_VISIBLE_DEVICES"])
        script.run()
        assert seen["cuda"] == ""

    def test_cuda_env_removed_again_when_absent_before(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        self._prepare(make_script(BASE_ARGS + ["--cpu"])).run()
        assert "CUDA_VISIBLE_DEVICES" not in os.environ

    def test_cuda_env_restored_to_original_value(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
        self._prepare(make_script(BASE_ARGS + ["--cpu"])).run()
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"

    def test_cuda_env_restored_even_when_training_raises(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

        def boom(local_dir):
            raise RuntimeError("training exploded")

        script = self._prepare(make_script(BASE_ARGS + ["--cpu"]))
        script._run_training = boom
        with pytest.raises(RuntimeError, match="training exploded"):
            script.run()
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"

    def test_no_cpu_flag_leaves_cuda_untouched(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        self._prepare(make_script(BASE_ARGS)).run()
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"


class TestRunTraining:
    """Test _run_training command construction."""

    def _run(self, script, tmp_path, output_dir=None):
        """Call _run_training with chdir and subprocess mocked; return argv list."""
        captured = {}

        def fake_execute(cmd, shell=False):
            captured["cmd"] = cmd
            captured["shell"] = shell
            return 0

        script.execute_command = fake_execute
        with patch.object(train_champollion.os, "chdir") as chdir:
            result = script._run_training(str(tmp_path))
        captured["result"] = result
        captured["chdir_calls"] = [c.args[0] for c in chdir.call_args_list]
        return captured

    def test_skips_when_output_exists_and_no_overwrite(self, tmp_path, capsys):
        out = tmp_path / "models"
        out.mkdir()
        script = make_script(BASE_ARGS + ["--output_dir", str(out)])
        script.execute_command = lambda cmd, shell=False: pytest.fail("must not run training")
        assert script._run_training(str(tmp_path)) == 0
        assert "Use --overwrite to force retraining" in capsys.readouterr().out

    def test_overwrite_reruns_even_when_output_exists(self, tmp_path):
        out = tmp_path / "models"
        out.mkdir()
        script = make_script(BASE_ARGS + ["--output_dir", str(out), "--overwrite"])
        captured = self._run(script, tmp_path)
        assert captured["cmd"][:2] == ["python3", "train.py"]

    def test_creates_output_dir_and_returns_command_result(self, tmp_path):
        out = tmp_path / "new" / "models"
        script = make_script(BASE_ARGS + ["--output_dir", str(out)])
        captured = self._run(script, tmp_path)
        assert out.is_dir()
        assert captured["result"] == 0
        assert captured["shell"] is False

    def test_hydra_overrides_contain_dataset_region_and_mode(self, tmp_path):
        out = tmp_path / "models"
        script = make_script(BASE_ARGS + ["--output_dir", str(out), "--mode", "classifier"])
        cmd = self._run(script, tmp_path)["cmd"]
        assert "+dataset/TEST01=cingulate_left" in cmd
        assert "+dataset_localization=local" in cmd
        assert "mode=classifier" in cmd
        assert f"hydra.run.dir={os.path.abspath(str(out))}" in cmd

    def test_platform_is_cuda_by_default(self, tmp_path):
        script = make_script(BASE_ARGS + ["--output_dir", str(tmp_path / "m")])
        assert "platform=cuda_not_brainvisa" in self._run(script, tmp_path)["cmd"]

    def test_platform_is_cpu_with_cpu_flag(self, tmp_path):
        script = make_script(BASE_ARGS + ["--output_dir", str(tmp_path / "m"), "--cpu"])
        assert "platform=cpu" in self._run(script, tmp_path)["cmd"]

    def test_localization_override_is_forwarded(self, tmp_path):
        script = make_script(BASE_ARGS + ["--output_dir", str(tmp_path / "m"), "--localization", "jean-zay"])
        assert "+dataset_localization=jean-zay" in self._run(script, tmp_path)["cmd"]

    def test_load_sparse_defaults_to_false(self, tmp_path):
        script = make_script(BASE_ARGS + ["--output_dir", str(tmp_path / "m")])
        assert "load_sparse=false" in self._run(script, tmp_path)["cmd"]

    def test_load_sparse_flag_sets_true(self, tmp_path):
        script = make_script(BASE_ARGS + ["--output_dir", str(tmp_path / "m"), "--load-sparse"])
        assert "load_sparse=true" in self._run(script, tmp_path)["cmd"]

    def test_njobs_becomes_num_cpu_workers_override(self, tmp_path):
        script = make_script(BASE_ARGS + ["--output_dir", str(tmp_path / "m"), "--njobs", "8"])
        assert "num_cpu_workers=8" in self._run(script, tmp_path)["cmd"]

    def test_no_num_cpu_workers_override_without_njobs(self, tmp_path):
        script = make_script(BASE_ARGS + ["--output_dir", str(tmp_path / "m")])
        cmd = self._run(script, tmp_path)["cmd"]
        assert not any(c.startswith("num_cpu_workers=") for c in cmd)

    def test_config_dir_adds_hydra_searchpath(self, tmp_path):
        config_dir = tmp_path / "cfg"
        config_dir.mkdir()
        script = make_script(BASE_ARGS + ["--output_dir", str(tmp_path / "m"), "--config-dir", str(config_dir)])
        cmd = self._run(script, tmp_path)["cmd"]
        assert f"hydra.searchpath=[file://{os.path.abspath(str(config_dir))}]" in cmd
        assert not any(c.startswith("++dataset_folder=") for c in cmd)

    def test_config_dir_localization_yaml_forces_dataset_folder(self, tmp_path):
        config_dir = tmp_path / "cfg"
        loc = config_dir / "dataset_localization"
        loc.mkdir(parents=True)
        (loc / "local.yaml").write_text("# @package _global_\ndataset_folder: /data/root\n")
        script = make_script(BASE_ARGS + ["--output_dir", str(tmp_path / "m"), "--config-dir", str(config_dir)])
        assert "++dataset_folder=/data/root" in self._run(script, tmp_path)["cmd"]

    def test_localization_yaml_without_dataset_folder_adds_no_override(self, tmp_path):
        config_dir = tmp_path / "cfg"
        loc = config_dir / "dataset_localization"
        loc.mkdir(parents=True)
        (loc / "local.yaml").write_text("# @package _global_\nother_key: value\n")
        script = make_script(BASE_ARGS + ["--output_dir", str(tmp_path / "m"), "--config-dir", str(config_dir)])
        cmd = self._run(script, tmp_path)["cmd"]
        assert not any(c.startswith("++dataset_folder=") for c in cmd)

    def test_returns_to_original_directory(self, tmp_path):
        script = make_script(BASE_ARGS + ["--output_dir", str(tmp_path / "m")])
        chdir_calls = self._run(script, tmp_path)["chdir_calls"]
        assert chdir_calls[0] == train_champollion._CONTRASTIVE_DIR
        assert chdir_calls[-1] == str(tmp_path)


class TestMain:
    """Test the main() entry point."""

    def test_main_builds_prints_and_runs(self):
        with patch.object(train_champollion, "TrainChampollion") as cls:
            cls.return_value.build.return_value.print_args.return_value.run.return_value = 0
            assert main() == 0
            cls.return_value.build.assert_called_once()
