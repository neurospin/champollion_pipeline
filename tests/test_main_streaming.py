"""main.py wiring of the streaming mode and of the file_indexer checker (TASK-STREAMING-BDRABCZUK-CA8E292B33D0).

REQ-STREAMING-BDRABCZUK-6D651731635B: run_parallel_pipeline imported from champollion_pipeline.streaming.runner.
REQ-STREAMING-BDRABCZUK-17D1E8A44C11: SubjectEligibilityChecker imported from champollion_pipeline.file_indexer.
REQ-STREAMING-BDRABCZUK-B7A8E951F74C: main.py's only ImportError handler guards the streaming runner import alone.
REQ-STREAMING-BDRABCZUK-AA54FBD84A93: streaming main() returns 1 when the streaming runner failed to import.
REQ-STREAMING-BDRABCZUK-D1258610AC92: main.py has no sys.path reference.
REQ-STREAMING-BDRABCZUK-86C8EA3AF69D: streaming main() returns run_parallel_pipeline's exit code.
REQ-STREAMING-BDRABCZUK-FD554C40F883: streaming without dataset.embeddings_only returns 1, no orchestrator call.
REQ-STREAMING-BDRABCZUK-6A11F9277D6B: --n-workers N -> StreamingSettings.n_workers.
REQ-STREAMING-BDRABCZUK-AB1478A56267: --worker-timeout SECONDS -> StreamingSettings.worker_timeout.
REQ-STREAMING-BDRABCZUK-7160F02B7DDF: --dry-run -> StreamingSettings.dry_run.
REQ-STREAMING-BDRABCZUK-F8E09DB5668B: --bids -> DatasetConfig.bids.
REQ-STREAMING-BDRABCZUK-BC7AEA3AD661: --log DIR -> StreamingSettings.log_dir.
REQ-STREAMING-BDRABCZUK-7DB06C49B059: --log DIR -> no run log of the main.py run.
REQ-STREAMING-BDRABCZUK-1E28715FF411: bare --log -> StreamingSettings.log_dir is main.py's run-log path without .log.
REQ-STREAMING-BDRABCZUK-C3AE3291EEF2: bare --log -> main.py's run log holds the fd 1 and 2 text of the streaming call.
REQ-STREAMING-BDRABCZUK-EC47AC1B9F8D: hf_enabled false -> models_path argument is PipelineConfig.models_path.
REQ-STREAMING-BDRABCZUK-DCA7928D3675: hf_enabled true -> models_path argument is dataset.hf_repo_id.
REQ-STREAMING-BDRABCZUK-1A49B653DBC5: sequential run_cortical_tiles emits the Pre-check OUTPUT record first.

main.py lives at the repo root, so it is loaded by file path under a private module name. check_for_updates
(network), init_pipeline_process and ScriptBuilder's init_process (root handlers) are stubbed, and every test runs in a
temporary working directory. run_parallel_pipeline is replaced by a recorder whose arguments are bound to the real
orchestrator's signature, so a call that does not match it fails the test. default_log_path() of main.py's
ScriptBuilder subclass is replaced by a fake returning a new path under the working directory on each call, so the
tests see which of its paths main.py's run log was written to.
"""

import ast
import importlib.util
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest
from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.config import DatasetConfig
from champollion_pipeline.streaming import StreamingSettings
from champollion_pipeline.streaming.runner import run_parallel_pipeline as real_run_parallel_pipeline
from tests.log_assertions import OUTPUT

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"
MAIN_TREE = ast.parse(MAIN_PY.read_text())

STDOUT_MARKER = "streaming-parent-stdout-marker"
STDERR_MARKER = "streaming-parent-stderr-marker"
PRE_CHECK_PREFIX = "Pre-check run_cortical_tiles"
STREAMING_RUNNER = "champollion_pipeline.streaming.runner"
IMPORT_ERRORS = {"ImportError", "ModuleNotFoundError"}


class _NoRunOrchestrator:
    """Stands in for PipelineOrchestrator in the streaming tests: a sequential run there is a test failure."""

    STAGE_REGISTRY: ClassVar[dict] = {}

    def __init__(self, config):
        self.config = config

    def run(self) -> int:
        pytest.fail("main() ran the sequential orchestrator instead of the streaming one")


class _OrchestratorRecorder:
    """Replaces run_parallel_pipeline: records the arguments bound to the real signature, returns exit_code."""

    def __init__(self, exit_code: int = 0, on_call=None):
        self.exit_code = exit_code
        self.on_call = on_call
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs) -> int:
        try:
            bound = inspect.signature(real_run_parallel_pipeline).bind(*args, **kwargs)
        except TypeError as error:
            expected = inspect.signature(real_run_parallel_pipeline)
            pytest.fail(f"main() called run_parallel_pipeline with arguments that do not match {expected}: {error}")
        self.calls.append(dict(bound.arguments))
        if self.on_call is not None:
            self.on_call()
        return self.exit_code


class _CountingLogPaths:
    """Fake default_log_path(): a new run-log path under *root* on each call, all of them kept in ``paths``."""

    def __init__(self, root: Path):
        self.root = root
        self.paths: list[str] = []

    def install(self, module, monkeypatch) -> "_CountingLogPaths":
        """Make this fake the default_log_path() of main.py's ScriptBuilder subclass."""
        monkeypatch.setattr(_script_class(module), "default_log_path", lambda _script: self.next_path())
        return self

    def next_path(self) -> str:
        path = str(self.root / ".log" / f"champollion_pipeline_{len(self.paths)}.log")
        self.paths.append(path)
        return path

    def written(self) -> list[str]:
        return [path for path in self.paths if Path(path).is_file()]


def _load_main():
    spec = importlib.util.spec_from_file_location("_champollion_main_streaming_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _script_class(module) -> type:
    """The one ScriptBuilder subclass defined in main.py."""
    classes = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, ScriptBuilder)
        and value is not ScriptBuilder
        and value.__module__ == module.__name__
    ]
    assert len(classes) == 1, f"main.py defines {classes!r}, expected one ScriptBuilder subclass"
    return classes[0]


def _write_fd_markers() -> None:
    os.write(1, f"{STDOUT_MARKER}\n".encode())
    os.write(2, f"{STDERR_MARKER}\n".encode())


def _files_holding_markers(root: Path) -> list[Path]:
    files = sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else []
    return [p for p in files if STDOUT_MARKER in p.read_text(errors="replace")]


def _tmp_config(module, tmp_path: Path):
    """PipelineConfig whose paths all live under tmp_path, embeddings-only, no Hugging Face, no log file."""
    config = module.PipelineConfig(
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
    dataset.embeddings_only = True
    dataset.hf_enabled = False  # noqa: V101
    config.log_to_file = False  # noqa: V101
    return config


@pytest.fixture
def work_dir(monkeypatch, tmp_path) -> Path:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


@pytest.fixture
def main_module(monkeypatch, work_dir):
    module = _load_main()
    monkeypatch.setattr(module, "check_for_updates", lambda: None)
    monkeypatch.setattr(module, "init_pipeline_process", lambda: logging.getLogger())
    monkeypatch.setattr("champollion_utils.script_builder.init_process", lambda *a, **k: logging.getLogger())
    monkeypatch.setattr("champollion_utils.script_builder.check_for_updates", lambda *a, **k: None)
    return module


@pytest.fixture
def streaming_main(main_module, monkeypatch, tmp_path):
    """main.py whose default configuration is _tmp_config in streaming mode and whose orchestrators are fakes."""
    config = _tmp_config(main_module, tmp_path)
    config.mode = "streaming"
    monkeypatch.setattr(main_module, "create_default_config", lambda: config)
    monkeypatch.setattr(_NoRunOrchestrator, "STAGE_REGISTRY", main_module.PipelineOrchestrator.STAGE_REGISTRY)
    monkeypatch.setattr(main_module, "PipelineOrchestrator", _NoRunOrchestrator)
    main_module.test_config = config
    return main_module


def _run_main(module, monkeypatch, *args: str) -> int:
    monkeypatch.setattr("sys.argv", ["main.py", *args])
    return module.main()


def _streaming_call(module, monkeypatch, *args: str, recorder: _OrchestratorRecorder | None = None) -> dict:
    """Run main() in streaming mode with *args and return the one run_parallel_pipeline call's bound arguments."""
    recorder = recorder or _OrchestratorRecorder()
    monkeypatch.setattr(module, "run_parallel_pipeline", recorder)
    _run_main(module, monkeypatch, *args)
    assert len(recorder.calls) == 1, f"expected one run_parallel_pipeline call, got {len(recorder.calls)}"
    return recorder.calls[0]


def _settings(call: dict) -> StreamingSettings:
    settings = call["settings"]
    assert isinstance(settings, StreamingSettings), type(settings)
    return settings


def _module_level_statements() -> list[ast.stmt]:
    """Top-level statements of main.py, plus the statements of the ``try`` bodies of top-level ``try`` blocks."""
    statements: list[ast.stmt] = []
    for node in MAIN_TREE.body:
        statements.append(node)
        if isinstance(node, ast.Try):
            statements.extend(node.body)
    return statements


def _module_level_imports(module_name: str) -> set[str]:
    """Names imported by module-level ``from <module_name> import ...`` statements of main.py (top-level try too)."""
    return {
        alias.name
        for node in _module_level_statements()
        if isinstance(node, ast.ImportFrom) and node.module == module_name
        for alias in node.names
    }


def _handler_type_names(handler: ast.ExceptHandler) -> set[str]:
    if handler.type is None:
        return set()
    nodes = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    names = set()
    for node in nodes:
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


# ====================== Imports (static checks on main.py) ======================


def test_run_parallel_pipeline_imported_from_streaming_runner():
    assert "run_parallel_pipeline" in _module_level_imports("champollion_pipeline.streaming.runner")


def test_subject_eligibility_checker_imported_from_file_indexer():
    assert "SubjectEligibilityChecker" in _module_level_imports("champollion_pipeline.file_indexer")


def test_main_py_import_error_handler_guards_only_streaming_runner():
    guards = [
        (node, handler)
        for node in ast.walk(MAIN_TREE)
        if isinstance(node, ast.Try)
        for handler in node.handlers
        if _handler_type_names(handler) & IMPORT_ERRORS
    ]
    assert len(guards) == 1, (
        f"expected one ImportError/ModuleNotFoundError handler, found lines {[h.lineno for _, h in guards]}"
    )
    guarded = guards[0][0].body
    assert len(guarded) == 1, f"the guarded try block holds {len(guarded)} statements, expected the runner import alone"
    statement = guarded[0]
    assert isinstance(statement, ast.ImportFrom), ast.dump(statement)
    assert statement.module == STREAMING_RUNNER, f"the guarded import is from {statement.module!r}"


def test_main_py_has_no_sys_path_reference():
    lines = [
        node.lineno
        for node in ast.walk(MAIN_TREE)
        if isinstance(node, ast.Attribute)
        and node.attr == "path"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    ]
    assert lines == []


# ====================== Streaming mode ======================


@pytest.mark.parametrize("exit_code", [0, 1])
def test_streaming_returns_orchestrator_exit_code(streaming_main, monkeypatch, exit_code):
    recorder = _OrchestratorRecorder(exit_code=exit_code)
    monkeypatch.setattr(streaming_main, "run_parallel_pipeline", recorder)

    assert _run_main(streaming_main, monkeypatch, "--mode", "streaming") == exit_code
    assert len(recorder.calls) == 1


def test_streaming_runner_import_failure_returns_1(monkeypatch, work_dir, tmp_path):
    runner = sys.modules[STREAMING_RUNNER]
    monkeypatch.setitem(sys.modules, STREAMING_RUNNER, None)  # any import of the runner now raises ImportError
    module = _load_main()
    monkeypatch.setitem(sys.modules, STREAMING_RUNNER, runner)
    monkeypatch.setattr(module, "check_for_updates", lambda: None)
    monkeypatch.setattr(module, "init_pipeline_process", lambda: logging.getLogger())
    monkeypatch.setattr("champollion_utils.script_builder.init_process", lambda *a, **k: logging.getLogger())
    config = _tmp_config(module, tmp_path)
    config.mode = "streaming"
    monkeypatch.setattr(module, "create_default_config", lambda: config)
    monkeypatch.setattr(_NoRunOrchestrator, "STAGE_REGISTRY", module.PipelineOrchestrator.STAGE_REGISTRY)
    monkeypatch.setattr(module, "PipelineOrchestrator", _NoRunOrchestrator)

    assert _run_main(module, monkeypatch, "--mode", "streaming") == 1


def test_streaming_without_embeddings_only_skips_orchestrator(streaming_main, monkeypatch):
    streaming_main.test_config.dataset.embeddings_only = False
    recorder = _OrchestratorRecorder()
    monkeypatch.setattr(streaming_main, "run_parallel_pipeline", recorder)

    assert _run_main(streaming_main, monkeypatch, "--mode", "streaming") == 1
    assert recorder.calls == []


def test_n_workers_reaches_streaming_settings(streaming_main, monkeypatch):
    call = _streaming_call(streaming_main, monkeypatch, "--mode", "streaming", "--n-workers", "3")

    assert _settings(call).n_workers == 3


def test_worker_timeout_reaches_streaming_settings(streaming_main, monkeypatch):
    call = _streaming_call(streaming_main, monkeypatch, "--mode", "streaming", "--worker-timeout", "42")

    assert _settings(call).worker_timeout == 42


@pytest.mark.parametrize(("extra_args", "expected"), [((), False), (("--dry-run",), True)])
def test_dry_run_reaches_streaming_settings(streaming_main, monkeypatch, extra_args, expected):
    call = _streaming_call(streaming_main, monkeypatch, "--mode", "streaming", *extra_args)

    assert _settings(call).dry_run is expected


def test_bids_reaches_streaming_dataset(streaming_main, monkeypatch):
    call = _streaming_call(streaming_main, monkeypatch, "--mode", "streaming", "--bids")

    dataset = call["dataset"]
    assert isinstance(dataset, DatasetConfig), type(dataset)
    assert dataset.bids is True


def test_log_dir_reaches_streaming_settings(streaming_main, monkeypatch, tmp_path):
    log_dir = tmp_path / "run_logs"

    call = _streaming_call(streaming_main, monkeypatch, "--mode", "streaming", "--log", str(log_dir))

    assert _settings(call).log_dir == str(log_dir)


def test_streaming_log_dir_creates_no_main_run_log(streaming_main, monkeypatch, tmp_path, work_dir):
    log_paths = _CountingLogPaths(work_dir).install(streaming_main, monkeypatch)
    log_dir = tmp_path / "run_logs"

    _streaming_call(
        streaming_main,
        monkeypatch,
        "--mode",
        "streaming",
        "--log",
        str(log_dir),
        recorder=_OrchestratorRecorder(on_call=_write_fd_markers),
    )

    assert log_paths.written() == [], f"main.py --log DIR wrote a run log at default_log_path(): {log_paths.written()}"
    assert _files_holding_markers(log_dir) == [], "main.py --log DIR wrote a run log of the main.py run into DIR"
    assert _files_holding_markers(work_dir) == [], "main.py --log DIR wrote a run log under the working directory"


def test_bare_log_streaming_log_dir_beside_main_run_log(streaming_main, monkeypatch, work_dir):
    log_paths = _CountingLogPaths(work_dir).install(streaming_main, monkeypatch)

    call = _streaming_call(streaming_main, monkeypatch, "--mode", "streaming", "--log")

    written = log_paths.written()
    assert len(written) == 1, f"expected one main.py run log among the default_log_path() paths {log_paths.paths}"
    assert _settings(call).log_dir == written[0].removesuffix(".log")


def test_bare_log_main_run_log_holds_streaming_output(streaming_main, monkeypatch, work_dir):
    log_paths = _CountingLogPaths(work_dir).install(streaming_main, monkeypatch)

    _streaming_call(
        streaming_main,
        monkeypatch,
        "--mode",
        "streaming",
        "--log",
        recorder=_OrchestratorRecorder(on_call=_write_fd_markers),
    )

    written = log_paths.written()
    assert len(written) == 1, f"expected one main.py run log among the default_log_path() paths {log_paths.paths}"
    text = Path(written[0]).read_text()
    assert STDOUT_MARKER in text, "the main.py run log lacks the fd 1 text of the run_parallel_pipeline call"
    assert STDERR_MARKER in text, "the main.py run log lacks the fd 2 text of the run_parallel_pipeline call"


def test_streaming_models_path_without_hf(streaming_main, monkeypatch):
    call = _streaming_call(streaming_main, monkeypatch, "--mode", "streaming")

    assert call["models_path"] == streaming_main.test_config.models_path


def test_streaming_models_path_with_hf(streaming_main, monkeypatch):
    dataset = streaming_main.test_config.dataset
    dataset.hf_enabled = True  # noqa: V101
    dataset.hf_repo_id = "neurospin/champollion-test-models"  # noqa: V101

    call = _streaming_call(streaming_main, monkeypatch, "--mode", "streaming")

    assert call["models_path"] == "neurospin/champollion-test-models"


# ====================== Sequential pre-check through the package checker ======================


def test_sequential_pre_check_summary_logged(main_module, monkeypatch, tmp_path, pipeline_caplog):
    config = _tmp_config(main_module, tmp_path)
    Path(config.dataset.morphologist_graphs).mkdir(parents=True)
    monkeypatch.setattr(main_module, "create_default_config", lambda: config)
    seen_before_run: list[bool] = []

    def run_tiles(*_args, **_kwargs) -> int:
        seen_before_run.append(
            any(
                record.levelno == OUTPUT and record.getMessage().startswith(PRE_CHECK_PREFIX)
                for record in pipeline_caplog.records
            )
        )
        return 0

    with patch.object(main_module.RunCorticalTiles, "run", side_effect=run_tiles):
        assert _run_main(main_module, monkeypatch, "--stages", "run_cortical_tiles") == 0

    assert seen_before_run == [True], f"no OUTPUT record starting with {PRE_CHECK_PREFIX!r} before the stage script ran"
