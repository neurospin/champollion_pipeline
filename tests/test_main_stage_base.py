#!/usr/bin/env python3
"""
Tests for the template-method base class of main.py's six pipeline stages
(TASK-REFACTOR-BDRABCZUK-F57652FA703A).

Two groups:

* structural (REQ-STAGEBASE-BDRABCZUK-3B0F9C04A40D, -1214AC64925F): the
  registered stages inherit ``execute`` from
  ``champollion_pipeline._stage_base.PipelineStage`` and main.py's
  ``StageResult`` is the one ``_stage_base`` defines;
* behaviour pins (REQ-STAGEBASE-BDRABCZUK-AA25F0BE4331, -6DAAFCC8C5DA,
  -85434902211A): what ``execute`` returns and logs, as main.py did at commit
  136054f, so the refactor cannot change it.

Each stage's script class is replaced in main.py's namespace by a MagicMock, as
in test_main_stage_argv.py; main.py is loaded by file path under a private
module name.
"""

import importlib.util
import logging
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from champollion_pipeline import _stage_base

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"

# Registry key -> (stage class name, script attribute in main.py, success message, failure message),
# as main.py produced them at commit 136054f.
STAGES = {
    "generate_morphologist_graphs": (
        "GenerateMorphologistGraphsStage",
        "GenerateMorphologistGraphs",
        "Morphologist graphs generated successfully",
        "Morphologist failed",
    ),
    "run_cortical_tiles": ("RunCorticalTilesStage", "RunCorticalTiles", "Cortical tiles completed", "Failed"),
    "generate_champollion_config": (
        "GenerateChampollionConfigStage",
        "GenerateChampollionConfig",
        "Champollion config generated",
        "Config generation failed",
    ),
    "generate_embeddings": (
        "GenerateEmbeddingsStage",
        "GenerateEmbeddings",
        "Embeddings generated",
        "Embedding generation failed",
    ),
    "put_together_embeddings": (
        "PutTogetherEmbeddingsStage",
        "PutTogetherEmbeddings",
        "Embeddings combined successfully",
        "Combine failed",
    ),
    "generate_snapshots": (
        "GenerateSnapshotsStage",
        "GenerateSnapshots",
        "Snapshots generated",
        "Snapshot generation failed",
    ),
}


@pytest.fixture(scope="module")
def main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_stage_base_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_config(main_module):
    """PipelineConfig with every path the six stages read set to a distinct value."""
    config = main_module.PipelineConfig(models_path="/models", outputs_path="/outputs")
    dataset = config.dataset
    dataset.input_path = "/raw"  # noqa: V101
    dataset.morphologist_graphs = "/graphs"  # noqa: V101
    dataset.cortical_tiles_output = "/tiles"  # noqa: V101
    dataset.crops_path = "/crops"  # noqa: V101
    dataset.datasets_root = "/data/MYDATASET"  # noqa: V101
    dataset.snapshots_path = "/snapshots"  # noqa: V101
    return config


def _make_stage(main_module, key, config=None):
    stage_cls = main_module.PipelineOrchestrator.STAGE_REGISTRY[key]
    return stage_cls(key, config or _base_config(main_module), MagicMock(spec=logging.Logger))


def _script_mock(run_return=0, run_error=None):
    script_cls = MagicMock()
    if run_error is not None:
        script_cls.return_value.run.side_effect = run_error  # noqa: V101
    else:
        script_cls.return_value.run.return_value = run_return
    return script_cls


@pytest.mark.unit
class TestExecuteInherited:
    """REQ-STAGEBASE-BDRABCZUK-3B0F9C04A40D: registered stages inherit _stage_base.PipelineStage.execute."""

    @pytest.mark.parametrize("key", sorted(STAGES))
    def test_stage_execute_inherited_from_stage_base(self, main_module, key):
        """REQ-STAGEBASE-BDRABCZUK-3B0F9C04A40D: the stage's execute is the base class's method."""
        stage_cls = main_module.PipelineOrchestrator.STAGE_REGISTRY[key]
        assert stage_cls.execute is _stage_base.PipelineStage.execute, (
            f"{stage_cls.__name__}.execute is not inherited from champollion_pipeline._stage_base.PipelineStage"
        )


@pytest.mark.unit
class TestStageResultReexport:
    """REQ-STAGEBASE-BDRABCZUK-1214AC64925F: main.StageResult is _stage_base.StageResult."""

    def test_main_stage_result_is_stage_base_class(self, main_module):
        """REQ-STAGEBASE-BDRABCZUK-1214AC64925F: one StageResult class, defined in _stage_base."""
        assert main_module.StageResult is _stage_base.StageResult


@pytest.mark.unit
class TestScriptException:
    """REQ-STAGEBASE-BDRABCZUK-AA25F0BE4331: a raising script yields a failed StageResult."""

    @pytest.mark.parametrize("key", sorted(STAGES))
    def test_exception_returns_failed_result(self, main_module, key):
        """REQ-STAGEBASE-BDRABCZUK-AA25F0BE4331: success False, return code 1, message 'Failed: <text>'."""
        _, script_attr, _, _ = STAGES[key]
        stage = _make_stage(main_module, key)
        with patch.object(main_module, script_attr, _script_mock(run_error=RuntimeError("boom 42"))):
            result = stage.execute()
        assert (result.stage_name, result.success, result.return_code, result.message) == (
            key,
            False,
            1,
            "Failed: boom 42",
        )


@pytest.mark.unit
class TestReturnCodeResult:
    """REQ-STAGEBASE-BDRABCZUK-6DAAFCC8C5DA: the script's return code maps to the stage's own result."""

    @pytest.mark.parametrize("return_code", [0, 3])
    @pytest.mark.parametrize("key", sorted(STAGES))
    def test_return_code_maps_to_stage_message(self, main_module, key, return_code):
        """REQ-STAGEBASE-BDRABCZUK-6DAAFCC8C5DA: return code kept, success only on 0, message as at 136054f."""
        _, script_attr, ok_message, failed_message = STAGES[key]
        stage = _make_stage(main_module, key)
        with patch.object(main_module, script_attr, _script_mock(run_return=return_code)):
            result = stage.execute()
        expected_message = ok_message if return_code == 0 else failed_message
        assert (result.stage_name, result.success, result.return_code, result.message) == (
            key,
            return_code == 0,
            return_code,
            expected_message,
        )


@pytest.mark.unit
class TestStageLogging:
    """REQ-STAGEBASE-BDRABCZUK-85434902211A: log_start before the script, log_end with the result."""

    @pytest.mark.parametrize("raises", [False, True], ids=["returns", "raises"])
    @pytest.mark.parametrize("key", sorted(STAGES))
    def test_log_calls_bracket_script_run(self, main_module, key, raises):
        """REQ-STAGEBASE-BDRABCZUK-85434902211A: log_start, run, [log_failure], log_end(result), once each."""
        _, script_attr, _, _ = STAGES[key]
        stage = _make_stage(main_module, key)
        recorder = MagicMock()
        stage.log_start = recorder.log_start
        stage.log_failure = recorder.log_failure
        stage.log_end = recorder.log_end
        script_cls = MagicMock()
        script_cls.return_value.run = recorder.run
        if raises:
            recorder.run.side_effect = RuntimeError("boom")  # noqa: V101
        else:
            recorder.run.return_value = 0
        with patch.object(main_module, script_attr, script_cls):
            result = stage.execute()
        expected = [call.log_start(), call.run()]
        if raises:
            expected.append(call.log_failure())
        expected.append(call.log_end(result))
        assert recorder.mock_calls == expected

    def test_embeddings_missing_repo_id_returns_before_log_start(self, main_module):
        """REQ-STAGEBASE-BDRABCZUK-85434902211A: the hf guard returns before the script, with no log call."""
        config = _base_config(main_module)
        config.dataset.hf_enabled = True  # noqa: V101
        config.dataset.hf_repo_id = ""  # noqa: V101
        stage = _make_stage(main_module, "generate_embeddings", config)
        recorder = MagicMock()
        stage.log_start = recorder.log_start
        stage.log_failure = recorder.log_failure
        stage.log_end = recorder.log_end
        script_cls = MagicMock()
        with patch.object(main_module, "GenerateEmbeddings", script_cls):
            result = stage.execute()
        assert recorder.mock_calls == []
        script_cls.assert_not_called()
        assert (result.success, result.return_code, result.message) == (
            False,
            1,
            "hf_enabled requires dataset.hf_repo_id",
        )
