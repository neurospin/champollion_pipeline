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

    # REQ-TRAIN-PLATFORM-01: the GPU-mode platform override must name a config that exists
    # in the pinned champollion_V1 submodule (guards against upstream renames, e.g. the
    # removal of cuda_not_brainvisa.yaml in champollion_V1 cde9a184).
    _PLATFORM_DIR = (
        Path(__file__).resolve().parents[1] / "external" / "champollion_V1" / "champollion" / "configs" / "platform"
    )

    def _platform_override(self, script, tmp_path):
        values = [c.split("=", 1)[1] for c in self._run(script, tmp_path)["cmd"] if c.startswith("platform=")]
        assert len(values) == 1, f"expected exactly one platform= override, got {values}"
        return values[0]

    def _platform_yaml(self, name):
        if not self._PLATFORM_DIR.is_dir():
            pytest.skip(f"champollion_V1 submodule not checked out: {self._PLATFORM_DIR}")
        path = self._PLATFORM_DIR / f"{name}.yaml"
        available = sorted(p.stem for p in self._PLATFORM_DIR.glob("*.yaml"))
        assert path.is_file(), f"platform={name} has no {path.name} in pinned champollion_V1 (available: {available})"
        return path.read_text()

    def test_gpu_platform_config_exists_in_pinned_champollion_v1(self, tmp_path):
        script = make_script(BASE_ARGS + ["--output_dir", str(tmp_path / "m")])
        self._platform_yaml(self._platform_override(script, tmp_path))

    def test_gpu_platform_config_sets_cuda_device(self, tmp_path):
        script = make_script(BASE_ARGS + ["--output_dir", str(tmp_path / "m")])
        text = self._platform_yaml(self._platform_override(script, tmp_path))
        assert any(line.split("#", 1)[0].split() == ["device:", "cuda"] for line in text.splitlines())

    def test_cpu_platform_config_exists_in_pinned_champollion_v1(self, tmp_path):
        script = make_script(BASE_ARGS + ["--output_dir", str(tmp_path / "m"), "--cpu"])
        self._platform_yaml(self._platform_override(script, tmp_path))

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


class TestSwfUnsupported:
    """REQ-SWF-01: --swf fails fast with NotImplementedError instead of being a silent no-op.

    The flag stays parseable (champollion_sulcal_mcp's start_training forwards it), but
    run() must refuse it before any side effect: no output directory, no chdir, no command.
    """

    def _prepare(self, tmp_path, calls):
        out = tmp_path / "models"
        script = make_script(BASE_ARGS + ["--output_dir", str(out), "--swf"])
        script._validate_inputs = lambda: None

        def fake_execute(cmd, shell=False):
            calls["execute"].append(cmd)
            return 0

        script.execute_command = fake_execute
        return script, out

    def _run_ignoring_not_implemented(self, script, calls):
        with patch.object(train_champollion.os, "chdir") as chdir:
            try:
                script.run()
            except NotImplementedError:
                pass
        calls["chdir"] = [c.args[0] for c in chdir.call_args_list]

    def test_swf_raises_not_implemented_naming_the_flag(self, tmp_path):
        calls = {"execute": []}
        script, _ = self._prepare(tmp_path, calls)
        with patch.object(train_champollion.os, "chdir"):
            with pytest.raises(NotImplementedError, match="--swf"):
                script.run()

    def test_swf_does_not_create_output_directory(self, tmp_path):
        calls = {"execute": []}
        script, out = self._prepare(tmp_path, calls)
        self._run_ignoring_not_implemented(script, calls)
        assert not out.exists()

    def test_swf_does_not_change_working_directory(self, tmp_path):
        calls = {"execute": []}
        script, _ = self._prepare(tmp_path, calls)
        self._run_ignoring_not_implemented(script, calls)
        assert calls["chdir"] == []

    def test_swf_does_not_execute_any_command(self, tmp_path):
        calls = {"execute": []}
        script, _ = self._prepare(tmp_path, calls)
        self._run_ignoring_not_implemented(script, calls)
        assert calls["execute"] == []


class TestMain:
    """Test the main() entry point."""

    def test_main_builds_prints_and_runs(self):
        with patch.object(train_champollion, "TrainChampollion") as cls:
            cls.return_value.build.return_value.print_args.return_value.run.return_value = 0
            assert main() == 0
            cls.return_value.build.assert_called_once()
