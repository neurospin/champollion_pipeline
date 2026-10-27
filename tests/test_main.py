"""main() messages go through the champollion_pipeline loggers, not print() (TASK-LOGGING-BDRABCZUK-41659FA95B6C).

REQ-LOGGING-BDRABCZUK-BF7CAFA17DF1: --generate-config logs the template path at OUTPUT.
REQ-LOGGING-BDRABCZUK-09E807184505: --config logs the configuration file path at LOG.
REQ-LOGGING-BDRABCZUK-B650F7D4E810: no --config logs "Using default configuration" at LOG.
REQ-LOGGING-BDRABCZUK-0A7F0CE5FFB2: streaming without dataset.embeddings_only logs at ERROR.
REQ-LOGGING-BDRABCZUK-A9D14195A854: streaming with a failed parallel_runner import logs at ERROR.
REQ-LOGGING-BDRABCZUK-84D9CEFCF3F8: main.py contains no built-in print call.

main.py lives at the repo root, so it is loaded by file path under a private
module name, as in test_main_stage_argv.py. check_for_updates (network) and
init_pipeline_process (root handlers) are stubbed; no stage runs.
"""

import ast
import importlib.util
import logging
from pathlib import Path
from typing import ClassVar

import pytest

from tests.log_assertions import LOG, OUTPUT, print_call_lines

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"


class _NoRunOrchestrator:
    """Stands in for PipelineOrchestrator: main() builds it and runs nothing."""

    STAGE_REGISTRY: ClassVar[dict] = {}

    def __init__(self, config):
        self.config = config

    def run(self) -> int:
        return 0


@pytest.fixture
def main_module(monkeypatch):
    spec = importlib.util.spec_from_file_location("_champollion_main_logging_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "check_for_updates", lambda: None)
    monkeypatch.setattr(module, "init_pipeline_process", lambda: logging.getLogger())
    # parse_arguments() reads the stage names from PipelineOrchestrator.STAGE_REGISTRY.
    monkeypatch.setattr(_NoRunOrchestrator, "STAGE_REGISTRY", module.PipelineOrchestrator.STAGE_REGISTRY)
    monkeypatch.setattr(module, "PipelineOrchestrator", _NoRunOrchestrator)
    return module


def _run_main(main_module, monkeypatch, *args: str) -> int:
    monkeypatch.setattr("sys.argv", ["main.py", *args])
    return main_module.main()


def _with_embeddings_only(main_module, monkeypatch, embeddings_only: bool) -> None:
    """Make main()'s default configuration carry dataset.embeddings_only = embeddings_only."""
    default_config = main_module.create_default_config

    def config_factory():
        config = default_config()
        config.dataset.embeddings_only = embeddings_only
        return config

    monkeypatch.setattr(main_module, "create_default_config", config_factory)


def test_generate_config_logs_template_path_at_output(main_module, monkeypatch, tmp_path, pipeline_caplog):
    template = tmp_path / "template.yaml"

    assert _run_main(main_module, monkeypatch, "--generate-config", str(template)) == 0

    pipeline_caplog.assert_logged(OUTPUT, str(template))


def test_config_file_load_logged_at_log(main_module, monkeypatch, tmp_path, pipeline_caplog):
    config_file = tmp_path / "pipeline.yaml"
    main_module.ConfigLoader.save_to_yaml(main_module.create_default_config(), str(config_file))

    assert _run_main(main_module, monkeypatch, "--config", str(config_file)) == 0

    pipeline_caplog.assert_logged(LOG, str(config_file))


def test_default_configuration_logged_at_log(main_module, monkeypatch, pipeline_caplog):
    assert _run_main(main_module, monkeypatch) == 0

    pipeline_caplog.assert_logged(LOG, "Using default configuration")


def test_streaming_without_embeddings_only_logged_at_error(main_module, monkeypatch, pipeline_caplog):
    _with_embeddings_only(main_module, monkeypatch, False)

    assert _run_main(main_module, monkeypatch, "--mode", "streaming") == 1

    pipeline_caplog.assert_logged(logging.ERROR, "--mode streaming requires --embeddings-only")


def test_parallel_runner_import_failure_logged_at_error(main_module, monkeypatch, pipeline_caplog):
    _with_embeddings_only(main_module, monkeypatch, True)
    monkeypatch.setattr(main_module, "run_parallel_pipeline", None)
    monkeypatch.setattr(main_module, "_PARALLEL_RUNNER_IMPORT_ERROR", "No module named 'fake_parallel_runner_dep'")

    assert _run_main(main_module, monkeypatch, "--mode", "streaming") == 1

    pipeline_caplog.assert_logged(logging.ERROR, "No module named 'fake_parallel_runner_dep'")


def test_main_py_has_no_print_call():
    assert print_call_lines(ast.parse(MAIN_PY.read_text())) == []
