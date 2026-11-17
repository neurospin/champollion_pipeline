"""Shared helpers of the main.py stage tests.

main.py lives at the repo root, not in the installed package, so each test module loads it by
file path under its own private module name. The stage helpers replace the script class in its
own module by a MagicMock, run the stage and return the argv handed to ``parse_args``.
"""

import dataclasses
import importlib.util
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"


def load_main_module(module_name):
    """Load main.py under the private name ``module_name``."""
    spec = importlib.util.spec_from_file_location(module_name, MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def option_values(argv, option):
    """Values given to ``option`` in argv, in either the '--opt=V' or the '--opt V' form."""
    values = []
    for i, token in enumerate(argv):
        if token == option:
            values.append(argv[i + 1] if i + 1 < len(argv) else None)
        elif token.startswith(option + "="):
            values.append(token.split("=", 1)[1])
    return values


def stage_paths_config(main_module):
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


def tiles_config(main_module):
    """PipelineConfig with the paths RunCorticalTilesStage reads set to distinct values."""
    config = main_module.PipelineConfig()
    config.dataset.morphologist_graphs = "/graphs"  # noqa: V101
    config.dataset.cortical_tiles_output = "/tiles"  # noqa: V101
    return config


def masks_version_config(main_module, masks_version, hf_enabled):
    """PipelineConfig with the given masks_version and the requested model source."""
    config = main_module.PipelineConfig(models_path="/models", outputs_path="/outputs")
    dataset = config.dataset
    dataset.datasets_root = "/data/MYDATASET"  # noqa: V101
    dataset.masks_version = masks_version  # noqa: V101
    dataset.hf_enabled = hf_enabled  # noqa: V101
    if hf_enabled:
        dataset.hf_repo_id = "neurospin/Champollion_V1"  # noqa: V101
    return config


def capture_stage_argv(stage_cls, script_path, stage_name, config, logger_name):
    """Run ``stage_cls.execute()`` with the class at ``script_path`` mocked; return the argv it built."""
    script_cls = MagicMock()
    script_cls.return_value.run.return_value = 0
    with patch(script_path, script_cls):
        stage = stage_cls(stage_name, config, logging.getLogger(logger_name))
        result = stage.execute()
    assert result.success, result.message
    script_cls.return_value.parse_args.assert_called_once()
    return list(script_cls.return_value.parse_args.call_args[0][0])


def tiles_stage_argv(main_module, config, logger_name):
    """Argv RunCorticalTilesStage builds, with RunCorticalTiles mocked."""
    return capture_stage_argv(
        main_module.RunCorticalTilesStage,
        "champollion_pipeline.run_cortical_tiles.RunCorticalTiles",
        "run_cortical_tiles",
        config,
        logger_name,
    )


def embeddings_stage_argv(main_module, config, stage_name, logger_name):
    """Argv GenerateEmbeddingsStage builds, with GenerateEmbeddings mocked."""
    return capture_stage_argv(
        main_module.GenerateEmbeddingsStage,
        "champollion_pipeline.generate_embeddings.GenerateEmbeddings",
        stage_name,
        config,
        logger_name,
    )


def stub_main_setup(module, monkeypatch):
    """Stub the update check and the process setup of a loaded main module and of ScriptBuilder; return module."""
    monkeypatch.setattr(module, "check_for_updates", lambda: None)
    monkeypatch.setattr(module, "init_pipeline_process", lambda: logging.getLogger())
    monkeypatch.setattr("champollion_utils.script_builder.init_process", lambda *a, **k: logging.getLogger())
    monkeypatch.setattr("champollion_utils.script_builder.check_for_updates", lambda *a, **k: None)
    return module


def record_logged_before_run(caplog, level, prefix):
    """(run, seen): run() appends to seen whether a caplog record at level starting with prefix was logged."""
    seen: list[bool] = []

    def run(*_args, **_kwargs) -> int:
        seen.append(
            any(record.levelno == level and record.getMessage().startswith(prefix) for record in caplog.records)
        )
        return 0

    return run, seen


def enter_work_dir(monkeypatch, tmp_path):
    """Create tmp_path/cwd, make it the working directory for the test, and return it."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


def fake_script(argv, name, on_run=None):
    """Script class mock recording the argv of ``name`` in ``argv``; ``on_run`` gets that argv when it runs."""
    script_cls = MagicMock()
    script = script_cls.return_value

    def parse_args(args) -> None:
        argv[name] = list(args)

    def run(*_args, **_kwargs) -> int:
        if on_run is not None:
            on_run(argv[name])
        return 0

    script.parse_args.side_effect = parse_args  # noqa: V101
    script.run.side_effect = run  # noqa: V101
    return script_cls


@dataclasses.dataclass
class RunOutcome:
    """Exit status of one main() run and the PipelineConfigs it handed to the orchestrator."""

    status: int | str | None
    configs: list


def main_runner(main_module, monkeypatch, before_run=None):
    """Return a function running main_module.main() with the given arguments.

    The orchestrator is replaced by a recorder and the update check, process setup and profiling are
    switched off; the function returns the exit status and the configurations main() handed over.
    ``before_run`` is called with the arguments before each run.
    """
    configs: list = []

    class _RecordingOrchestrator:
        STAGE_REGISTRY = main_module.PipelineOrchestrator.STAGE_REGISTRY

        def __init__(self, config) -> None:
            configs.append(config)

        def run(self) -> int:
            return 0

    monkeypatch.setattr(main_module, "check_for_updates", lambda: None)
    monkeypatch.setattr(main_module, "init_pipeline_process", lambda: logging.getLogger())
    monkeypatch.setattr(main_module, "start_profiling", lambda *a, **k: None)
    monkeypatch.setattr(main_module, "PipelineOrchestrator", _RecordingOrchestrator)
    monkeypatch.setattr("champollion_utils.script_builder.check_for_updates", lambda *a, **k: None)
    monkeypatch.setattr("champollion_utils.script_builder.init_process", lambda *a, **k: logging.getLogger())
    monkeypatch.setattr("champollion_utils.script_builder.start_profiling", lambda *a, **k: None)

    def _run(*cli_args: str) -> RunOutcome:
        if before_run is not None:
            before_run(cli_args)
        configs.clear()
        monkeypatch.setattr("sys.argv", ["main.py", *cli_args])
        try:
            status = main_module.main()
        except SystemExit as stop:
            status = stop.code
        return RunOutcome(status=status, configs=list(configs))

    return _run


def ran_config(outcome: RunOutcome, cli_args: tuple[str, ...]):
    """The one configuration main() ran; fails when main() refused the command line or ran none."""
    assert outcome.status == 0, f"main() exit status {outcome.status} for {list(cli_args)}, expected 0"
    assert len(outcome.configs) == 1, f"main() ran {len(outcome.configs)} configurations for {list(cli_args)}"
    return outcome.configs[0]
