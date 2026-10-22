#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for main.py's run log of each in-process pipeline stage (TASK-RUNLOG-BDRABCZUK-C76B691F9497).

main.py runs every stage in-process (script.parse_args + script.run, never ScriptBuilder.main), so it
wraps each stage run itself with champollion_utils.run_log.RunLogCapture:

- REQ-RUNLOG-BDRABCZUK-F2F6A21FE6BB: PipelineConfig.log == "" -> the stage script's default_log_path().
- REQ-RUNLOG-BDRABCZUK-1859A582D928: PipelineConfig.log == DIR -> DIR/<stage>_<YYYYmmdd-HHMMSS>.log.
- REQ-RUNLOG-BDRABCZUK-B43595D9D816: PipelineConfig.log is None -> no run-log file.
- REQ-RUNLOG-BDRABCZUK-D6494D56F8C7: footer exit code = the script's return code, 1 on an exception.
- REQ-RUNLOG-BDRABCZUK-CE60EAEC74A5 / -4C92EF7F7AE4: main.py --log [DIR] sets PipelineConfig.log.
- REQ-RUNLOG-BDRABCZUK-4C2F09533A2D: PipelineConfig.log survives a YAML save/load round trip.

No real stage runs: each stage script's run() is patched. main.py lives at the repo root (not inside
the installed package), so it is loaded by file path under a private module name.
"""

import importlib.util
import logging
import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"

RUN_LOG_HEADER = "==== champollion run log ===="
RUN_LOG_FOOTER = "==== end of run ===="
RUN_OUTPUT_MARKER = "stage-run-output-marker"


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_stage_run_log_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def main_module():
    return _load_main_module()


STAGE_NAMES = [
    "generate_morphologist_graphs",
    "run_cortical_tiles",
    "generate_champollion_config",
    "generate_embeddings",
    "put_together_embeddings",
    "generate_snapshots",
]


def _config(main_module, tmp_path, log):
    """PipelineConfig whose paths all live under tmp_path, with PipelineConfig.log set to *log*."""
    config = main_module.PipelineConfig(
        root_path=str(tmp_path),
        data_path=str(tmp_path / "data"),
        models_path=str(tmp_path / "models"),
        outputs_path=str(tmp_path / "outputs"),
    )
    dataset = config.dataset
    dataset.input_path = str(tmp_path / "input")  # noqa: V101
    dataset.morphologist_graphs = str(tmp_path / "graphs")  # noqa: V101
    dataset.cortical_tiles_output = str(tmp_path / "tiles")  # noqa: V101
    dataset.crops_path = str(tmp_path / "tiles" / "crops")  # noqa: V101
    dataset.datasets_root = str(tmp_path / "data" / "DS")  # noqa: V101
    dataset.hf_enabled = False  # noqa: V101
    config.log = log
    return config


def _execute_stage(main_module, config, stage_name, run=None, default_log_path=None):
    """Run one stage's execute() with its script's run() (and optionally default_log_path()) patched."""
    stage = main_module.PipelineOrchestrator.STAGE_REGISTRY[stage_name](
        stage_name, config, logging.getLogger("test_main_stage_run_log")
    )
    script_cls = type(stage.create_script())

    def _default_run(*_args, **_kwargs):
        print(RUN_OUTPUT_MARKER, flush=True)
        return 0

    with patch.object(script_cls, "run", side_effect=run or _default_run):
        if default_log_path is None:
            return stage.execute()
        with patch.object(script_cls, "default_log_path", return_value=str(default_log_path)) as mocked:
            result = stage.execute()
        mocked.assert_called()
        return result


def _assert_run_log(path: Path):
    assert path.is_file(), f"no run log at {path}"
    text = path.read_text()
    assert text.startswith(RUN_LOG_HEADER), text[:200]
    assert RUN_LOG_FOOTER in text, text[-200:]
    return text


@pytest.mark.unit
class TestBareLogDefaultPath:
    """REQ-RUNLOG-BDRABCZUK-F2F6A21FE6BB: PipelineConfig.log == "" writes to default_log_path()."""

    @pytest.mark.parametrize("stage_name", STAGE_NAMES)
    def test_stage_writes_run_log_at_default_log_path(self, main_module, tmp_path, stage_name):
        default_path = tmp_path / "default_logs" / f"{stage_name}.log"
        config = _config(main_module, tmp_path, log="")
        result = _execute_stage(main_module, config, stage_name, default_log_path=default_path)
        assert result.success, result.message
        text = _assert_run_log(default_path)
        assert RUN_OUTPUT_MARKER in text


@pytest.mark.unit
class TestLogDirectory:
    """REQ-RUNLOG-BDRABCZUK-1859A582D928: PipelineConfig.log == DIR writes DIR/<stage>_<stamp>.log."""

    @pytest.mark.parametrize("stage_name", STAGE_NAMES)
    def test_stage_writes_timestamped_run_log_in_log_directory(self, main_module, tmp_path, stage_name):
        log_dir = tmp_path / "run_logs"
        config = _config(main_module, tmp_path, log=str(log_dir))
        result = _execute_stage(main_module, config, stage_name)
        assert result.success, result.message
        logs = sorted(log_dir.glob("*.log")) if log_dir.is_dir() else []
        assert len(logs) == 1, f"expected one run log in {log_dir}, found {logs}"
        assert re.fullmatch(rf"{stage_name}_\d{{8}}-\d{{6}}\.log", logs[0].name), logs[0].name
        text = _assert_run_log(logs[0])
        assert RUN_OUTPUT_MARKER in text


@pytest.mark.unit
class TestNoLog:
    """REQ-RUNLOG-BDRABCZUK-B43595D9D816: PipelineConfig.log None creates no run-log file."""

    @pytest.mark.parametrize("stage_name", STAGE_NAMES)
    def test_stage_writes_no_run_log(self, main_module, tmp_path, stage_name):
        default_path = tmp_path / "default_logs" / f"{stage_name}.log"
        config = _config(main_module, tmp_path, log=None)
        stage = main_module.PipelineOrchestrator.STAGE_REGISTRY[stage_name](
            stage_name, config, logging.getLogger("test_main_stage_run_log")
        )
        script_cls = type(stage.create_script())
        with (
            patch.object(script_cls, "run", return_value=0),
            patch.object(script_cls, "default_log_path", return_value=str(default_path)),
        ):
            result = stage.execute()
        assert result.success, result.message
        assert list(tmp_path.rglob("*.log")) == []


@pytest.mark.unit
class TestFooterExitCode:
    """REQ-RUNLOG-BDRABCZUK-D6494D56F8C7: footer exit code = script return code, 1 on an exception."""

    def test_footer_gives_script_return_code(self, main_module, tmp_path):
        default_path = tmp_path / "default_logs" / "combine.log"
        config = _config(main_module, tmp_path, log="")
        result = _execute_stage(
            main_module, config, "put_together_embeddings", run=lambda *a, **k: 3, default_log_path=default_path
        )
        assert result.return_code == 3
        text = _assert_run_log(default_path)
        assert "exit code: 3\n" in text

    def test_footer_gives_one_on_exception(self, main_module, tmp_path):
        default_path = tmp_path / "default_logs" / "combine.log"
        config = _config(main_module, tmp_path, log="")

        def _raise(*_args, **_kwargs):
            raise RuntimeError("stage blew up")

        result = _execute_stage(
            main_module, config, "put_together_embeddings", run=_raise, default_log_path=default_path
        )
        assert not result.success
        text = _assert_run_log(default_path)
        assert "exit code: 1\n" in text


def _cli_config(main_module, argv, monkeypatch, yaml_log):
    """Apply main.py's CLI overrides for *argv* to a config whose YAML gave log == *yaml_log*."""
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    args = main_module.parse_arguments()
    config = main_module.PipelineConfig()
    config.log = yaml_log
    main_module._apply_cli_overrides(config, args, from_default=False)
    return config


@pytest.mark.unit
class TestCliLogOption:
    """REQ-RUNLOG-BDRABCZUK-CE60EAEC74A5 / -4C92EF7F7AE4: main.py --log [DIR] sets PipelineConfig.log."""

    def test_bare_log_sets_empty_string(self, main_module, monkeypatch):
        config = _cli_config(main_module, ["--log"], monkeypatch, yaml_log="/from/yaml")
        assert config.log == ""

    def test_log_dir_sets_dir(self, main_module, monkeypatch, tmp_path):
        log_dir = str(tmp_path / "cli_logs")
        config = _cli_config(main_module, [f"--log={log_dir}"], monkeypatch, yaml_log="/from/yaml")
        assert config.log == log_dir

    def test_no_log_option_keeps_config_file_value(self, main_module, monkeypatch):
        config = _cli_config(main_module, [], monkeypatch, yaml_log="/from/yaml")
        assert config.log == "/from/yaml"


@pytest.mark.unit
class TestConfigLogRoundTrip:
    """REQ-RUNLOG-BDRABCZUK-4C2F09533A2D: PipelineConfig.log defaults to null and survives YAML."""

    def test_log_defaults_to_none(self, main_module):
        assert main_module.PipelineConfig().log is None

    @pytest.mark.parametrize("value", [None, "", "/run/logs"])
    def test_log_round_trips_through_yaml(self, main_module, tmp_path, value):
        config = main_module.PipelineConfig()
        config.log = value
        out = tmp_path / "config.yaml"
        main_module.ConfigLoader.save_to_yaml(config, str(out))
        saved = yaml.safe_load(out.read_text())
        assert "log" in saved, "PipelineConfig.log not written to the YAML file"
        assert saved["log"] == value
        loaded = main_module.ConfigLoader.load_from_yaml(str(out))
        assert loaded.log == value
