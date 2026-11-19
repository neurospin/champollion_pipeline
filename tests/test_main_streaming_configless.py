"""main.py config-less streaming mode (TASK-INPUTKIND-BDRABCZUK-A844DE5C3F1F).

Requirements (wording in the ledger):
    test_embeddings_only_argument_passes_guard (REQ-INPUTKIND-BDRABCZUK-B66B44B4C906): --mode streaming with
        --embeddings-only and no --config calls run_parallel_pipeline.
    test_subjects_dir_is_inferred_subjects_folder (REQ-INPUTKIND-BDRABCZUK-9DDC69EA81EA): a config-less Morphologist
        input gives StreamingSettings.subjects_dir the inferred Morphologist subjects folder.
    test_guard_error_names_argument_and_yaml_key (REQ-INPUTKIND-BDRABCZUK-79CA711A56D0): the embeddings_only guard
        ERROR names the --embeddings-only argument and the dataset.embeddings_only YAML key.

run_parallel_pipeline is replaced by a recorder; the update check, the process setup and profiling are stubbed.
"""

import logging

import pytest

import champollion_pipeline.streaming.runner as streaming_runner
from champollion_pipeline.config import PipelineConfig
from champollion_pipeline.path_inference import infer_paths
from champollion_pipeline.streaming import StreamingSettings
from champollion_pipeline.utils.input_kind import InputKind
from tests.main_stage_helpers import enter_work_dir, load_main_module, stub_main_setup
from tests.test_main_configless_run import _morphologist_input

MAIN = load_main_module("_champollion_main_streaming_configless_under_test")


@pytest.fixture
def run_streaming(monkeypatch, tmp_path):
    """Run main() with the arguments; return (exit code, the run_parallel_pipeline calls)."""
    enter_work_dir(monkeypatch, tmp_path)
    stub_main_setup(MAIN, monkeypatch)
    monkeypatch.setattr(MAIN, "start_profiling", lambda *a, **k: None)
    monkeypatch.setattr("champollion_utils.script_builder.start_profiling", lambda *a, **k: None)
    calls: list = []

    def _recorder(settings, *_args):
        calls.append(settings)
        return 0

    monkeypatch.setattr(streaming_runner, "run_parallel_pipeline", _recorder)

    def _run(*cli_args: str):
        monkeypatch.setattr("sys.argv", ["main.py", *cli_args])
        calls.clear()
        return MAIN.main(), list(calls)

    return _run


def test_embeddings_only_argument_passes_guard(run_streaming):
    code, calls = run_streaming("--mode", "streaming", "--embeddings-only")

    assert code == 0
    assert len(calls) == 1
    assert isinstance(calls[0], StreamingSettings)


def test_subjects_dir_is_inferred_subjects_folder(run_streaming, tmp_path):
    root = _morphologist_input(tmp_path)
    expected = PipelineConfig()
    expected.dataset.input_path = str(root)  # noqa: V101
    expected = infer_paths(expected, str(root), InputKind.MORPHOLOGIST).dataset.morphologist_graphs
    assert expected.endswith("morphologist-6.0")

    code, calls = run_streaming(str(root), "--mode", "streaming", "--embeddings-only")

    assert code == 0
    assert [call.subjects_dir for call in calls] == [expected]


def test_guard_error_names_argument_and_yaml_key(run_streaming, pipeline_caplog):
    code, calls = run_streaming("--mode", "streaming")

    assert code == 1
    assert calls == []
    errors = [r.getMessage() for r in pipeline_caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1, errors
    assert "--embeddings-only" in errors[0]
    assert "dataset.embeddings_only" in errors[0]
