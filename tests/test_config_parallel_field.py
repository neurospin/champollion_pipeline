#!/usr/bin/env python3
"""
PipelineConfig carries a parallel field, so --parallel and a --config YAML parallel value reach stage 1
(TASK-INPUTKIND-BDRABCZUK-0913621954E5).

REQ-INPUTKIND-BDRABCZUK-9AD1D6F6E0B7: a PipelineConfig built without a parallel value holds False in parallel.
REQ-INPUTKIND-BDRABCZUK-D6CF6B3A2BCC: main() with --parallel hands --parallel to the GenerateMorphologistGraphs script.
REQ-INPUTKIND-BDRABCZUK-C7A6A6282F64: main() with a --config YAML whose parallel is true hands --parallel to the
GenerateMorphologistGraphs script.

The stage 1 script class is replaced by a MagicMock in its own module; the argument list it receives through
parse_args is the observable. main() runs in a temporary working directory with the update check and the process
setup stubbed, as in test_main_morphologist_chain.py.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from champollion_pipeline.config import PipelineConfig
from tests.main_stage_helpers import enter_work_dir, load_main_module, stub_main_setup

MORPHOLOGIST_SCRIPT = "champollion_pipeline.generate_morphologist_graphs.GenerateMorphologistGraphs"
STAGE1 = "generate_morphologist_graphs"
PARALLEL_FLAG = "--parallel"


@pytest.fixture
def main_module(monkeypatch, tmp_path):
    """main loaded under its own name, run from tmp_path with the update check and process setup stubbed."""
    enter_work_dir(monkeypatch, tmp_path)
    return stub_main_setup(load_main_module("_champollion_main_config_parallel_field_under_test"), monkeypatch)


def _raw_input(tmp_path: Path) -> Path:
    """A raw T1 input folder holding one synthetic scan."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "sub-01.nii.gz").write_bytes(b"synthetic")
    return raw


def _default_config(main_module, tmp_path: Path):
    """PipelineConfig built without a parallel value, whose stage 1 paths live under tmp_path."""
    config = main_module.PipelineConfig(
        root_path=str(tmp_path),
        data_path=str(tmp_path / "data"),
        models_path=str(tmp_path / "models"),
        outputs_path=str(tmp_path / "outputs"),
    )
    config.dataset.input_path = str(_raw_input(tmp_path))  # noqa: V101
    config.dataset.datasets_root = str(tmp_path / "data" / "DS")  # noqa: V101
    config.log_to_file = False  # noqa: V101
    return config


def _stage1_argv_of_main(main_module, monkeypatch, *args: str) -> list[str]:
    """Run main() with args and the stage 1 script mocked; return the argument list stage 1 parsed."""
    script_cls = MagicMock()
    script_cls.return_value.run.return_value = 0
    monkeypatch.setattr("sys.argv", ["main.py", *args])
    with patch(MORPHOLOGIST_SCRIPT, script_cls):
        exit_code = main_module.main()
    assert exit_code == 0, f"main() exit status {exit_code}"
    script_cls.return_value.parse_args.assert_called_once()
    return list(script_cls.return_value.parse_args.call_args[0][0])


@pytest.mark.unit
def test_pipeline_config_parallel_defaults_to_false():
    assert PipelineConfig().parallel is False


@pytest.mark.unit
def test_parallel_option_reaches_stage1_script(main_module, monkeypatch, tmp_path):
    config = _default_config(main_module, tmp_path)
    monkeypatch.setattr(main_module, "create_default_config", lambda: config)
    argv = _stage1_argv_of_main(main_module, monkeypatch, "--stages", STAGE1, PARALLEL_FLAG)
    assert PARALLEL_FLAG in argv, f"stage 1 argument list {argv}"


@pytest.mark.unit
def test_yaml_parallel_reaches_stage1_script(main_module, monkeypatch, tmp_path):
    config_file = tmp_path / "pipeline.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "parallel": True,
                "log_to_file": False,
                "dataset": {
                    "input_path": str(_raw_input(tmp_path)),
                    "datasets_root": str(tmp_path / "data" / "DS"),
                },
            }
        )
    )
    argv = _stage1_argv_of_main(main_module, monkeypatch, "--config", str(config_file), "--stages", STAGE1)
    assert PARALLEL_FLAG in argv, f"stage 1 argument list {argv}"
