"""main.py as a champollion_utils ScriptBuilder script (TASK-REFACTOR-BDRABCZUK-752C6242C0F9).

main.py becomes a ScriptBuilder script with the script_name champollion_pipeline. A bare --log gives the
main.py run its own run log at default_log_path(); --log=DIR keeps its per-stage meaning and gives the
main.py run no run log; PipelineConfig.log keeps its values ("" for a bare --log, DIR for --log=DIR).
The existing command line, help text, update check and exit code stay as they were at commit cbe51d7.

main.py lives at the repo root, so it is loaded by file path under a private module name, as in
test_main.py. The process-wide setup (init_pipeline_process, ScriptBuilder's init_process) and the
update check (network) are replaced by recorders; the working directory is a temporary directory, so a
default run-log path can only land there. Except in the end-to-end stage tests, PipelineOrchestrator is
replaced by a stub that prints a marker and returns a chosen exit code.
"""

import importlib.util
import inspect
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import pytest
from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"

SCRIPT_NAME = "champollion_pipeline"
RUN_LOG_HEADER = "==== champollion run log ===="
ORCHESTRATOR_MARKER = "orchestrator-run-output-marker"
STAGE_MARKER = "combine-stage-run-output-marker"
STAMP_FORMAT = "%Y%m%d-%H%M%S"
HELP_COLUMNS = "100"

# Golden command lines and what parse_arguments() of main.py at commit cbe51d7 gave for each:
# the option values (vars of the namespace) or, for a refused command line, the exit status.
_CBE51D7_DEFAULTS = {
    "config": None,
    "stages": None,
    "enable_all_stages": False,
    "dataset_name": None,
    "models_path": None,
    "verbose": False,
    "generate_config": None,
    "bids": False,
    "mode": None,
    "n_workers": None,
    "worker_timeout": None,
    "log": None,
    "dry_run": False,
}
GOLDEN_COMMAND_LINES = [
    ([], dict(_CBE51D7_DEFAULTS)),
    (["--log"], {**_CBE51D7_DEFAULTS, "log": ""}),
    (["--log=/runs/logs"], {**_CBE51D7_DEFAULTS, "log": "/runs/logs"}),
    (["--log", "/runs/logs"], {**_CBE51D7_DEFAULTS, "log": "/runs/logs"}),
    (
        [
            "--config",
            "pipeline.yaml",
            "--stages",
            "generate_embeddings",
            "put_together_embeddings",
            "--dataset-name",
            "DS",
            "--models-path",
            "/models",
            "--verbose",
            "--bids",
            "--mode",
            "streaming",
            "--n-workers",
            "4",
            "--worker-timeout",
            "60",
            "--dry-run",
        ],
        {
            **_CBE51D7_DEFAULTS,
            "config": "pipeline.yaml",
            "stages": ["generate_embeddings", "put_together_embeddings"],
            "dataset_name": "DS",
            "models_path": "/models",
            "verbose": True,
            "bids": True,
            "mode": "streaming",
            "n_workers": 4,
            "worker_timeout": 60,
            "dry_run": True,
        },
    ),
    (
        ["--enable-all-stages", "--generate-config", "out.yaml"],
        {**_CBE51D7_DEFAULTS, "enable_all_stages": True, "generate_config": "out.yaml"},
    ),
    (["--mode", "bogus"], 2),
    (["--stages", "nope"], 2),
    (["--n-workers", "x"], 2),
    (["--unknown"], 2),
]

# main.py --help at commit cbe51d7 (Python 3.11 argparse, COLUMNS=100); the --mode and --worker-timeout
# entries carry the texts of TASK-STREAMING-BDRABCZUK-7CD70A508D2E.
CBE51D7_HELP = """\
usage: main.py [-h] [--config CONFIG]
               [--stages {generate_morphologist_graphs,run_cortical_tiles,generate_champollion_config,generate_embeddings,put_together_embeddings,generate_snapshots} [{generate_morphologist_graphs,run_cortical_tiles,generate_champollion_config,generate_embeddings,put_together_embeddings,generate_snapshots} ...]]
               [--enable-all-stages] [--dataset-name DATASET_NAME] [--models-path MODELS_PATH]
               [--verbose] [--generate-config OUTPUT_PATH] [--bids]
               [--mode {sequential,streaming}] [--n-workers N] [--worker-timeout SECONDS]
               [--log [DIR]] [--dry-run]

Champollion Pipeline Orchestrator

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to configuration YAML file
  --stages {generate_morphologist_graphs,run_cortical_tiles,generate_champollion_config,generate_embeddings,put_together_embeddings,generate_snapshots} [{generate_morphologist_graphs,run_cortical_tiles,generate_champollion_config,generate_embeddings,put_together_embeddings,generate_snapshots} ...]
                        Specific stages to run (overrides config)
  --enable-all-stages   Enable all pipeline stages
  --dataset-name DATASET_NAME
                        Dataset name (overrides config)
  --models-path MODELS_PATH
                        Path to models directory (overrides config)
  --verbose             Enable verbose output
  --generate-config OUTPUT_PATH
                        Generate a template configuration file and exit
  --bids                Input database follows BIDS layout (subjects/sub-*/ses-*/...). Activates
                        BIDS-aware scan enumeration in eligibility checks and passes --bids to
                        cortical_tiles.
  --mode {sequential,streaming}
                        Execution mode: 'sequential' (default, stage-centric batch) or 'streaming'
                        (scan-centric parallel, one worker per scan). Streaming mode requires
                        dataset.embeddings_only: true in the --config YAML.
  --n-workers N         Number of parallel scan workers (streaming mode only). Defaults to
                        os.cpu_count().
  --worker-timeout SECONDS
                        Timeout in seconds of the whole producer-worker run of one scan; a longer
                        run ends FAILED (default: 7200).
  --log [DIR]           Write a run log per stage: DIR/<stage>_<YYYYmmdd-HHMMSS>.log; bare --log:
                        each stage's default log path (its derivatives .log folder). Overrides the
                        config file's log.
  --dry-run             Enumerate scans and report without processing (streaming mode only).

Examples:
  # Run with default config
  python main.py

  # Run with custom config
  python main.py --config configs/my_experiment.yaml

  # Run specific stages
  python main.py --stages generate_embeddings

  # Enable all stages
  python main.py --enable-all-stages

  # Generate template config
  python main.py --generate-config my_config.yaml
"""  # noqa: E501 - verbatim argparse help text at cbe51d7


class _StubOrchestrator:
    """Stands in for PipelineOrchestrator: records the config, prints a marker, returns exit_code."""

    STAGE_REGISTRY: ClassVar[dict] = {}
    exit_code: ClassVar[int] = 0
    configs: ClassVar[list] = []

    def __init__(self, config):
        type(self).configs.append(config)

    def run(self) -> int:
        print(ORCHESTRATOR_MARKER, flush=True)
        return type(self).exit_code


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_script_builder_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _script_classes(module) -> list[type]:
    """ScriptBuilder subclasses defined in main.py itself (not imported into it)."""
    return [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, ScriptBuilder) and obj is not ScriptBuilder and obj.__module__ == module.__name__
    ]


def _script_class(module) -> type:
    classes = _script_classes(module)
    assert len(classes) == 1, f"main.py defines {classes!r}, expected one ScriptBuilder subclass"
    return classes[0]


class _Harness:
    """main.py loaded with recorded process setup and update check, run in a temporary directory."""

    def __init__(self, module, monkeypatch, tmp_path: Path):
        self.module = module
        self.monkeypatch = monkeypatch
        self.tmp_path = tmp_path
        self.cwd = tmp_path / "cwd"
        self.cwd.mkdir()
        self.calls: list[str] = []

    def stub_orchestrator(self, exit_code: int = 0) -> type:
        stub = type("StubOrchestrator", (_StubOrchestrator,), {})
        stub.STAGE_REGISTRY = self.module.PipelineOrchestrator.STAGE_REGISTRY
        stub.exit_code = exit_code
        stub.configs = []
        self.monkeypatch.setattr(self.module, "PipelineOrchestrator", stub)
        return stub

    def patch_main_default_log_path(self, path: Path, *, required: bool = True) -> None:
        """Point default_log_path() of main.py's ScriptBuilder subclass to *path*.

        With required=False and no subclass yet, nothing is patched (the working directory is
        temporary, so an unpatched default path still lands under tmp_path).
        """
        if not required and len(_script_classes(self.module)) != 1:
            return
        self.monkeypatch.setattr(_script_class(self.module), "default_log_path", lambda _self: str(path))

    def run(self, *args: str) -> int:
        self.monkeypatch.setattr(sys, "argv", ["main.py", *args])
        return self.module.main()

    def files_under(self, root: Path) -> list[Path]:
        return sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else []


@pytest.fixture
def harness(monkeypatch, tmp_path):
    module = _load_main_module()
    h = _Harness(module, monkeypatch, tmp_path)

    def _record_init(*_args, **_kwargs):
        h.calls.append("init_pipeline_process")
        return logging.getLogger()

    def _record_update(*_args, **_kwargs):
        h.calls.append("check_for_updates")

    monkeypatch.setattr(module, "init_pipeline_process", _record_init)
    monkeypatch.setattr(module, "check_for_updates", _record_update, raising=False)
    monkeypatch.setattr("champollion_utils.script_builder.check_for_updates", _record_update)
    monkeypatch.setattr("champollion_utils.script_builder.init_process", lambda *a, **k: logging.getLogger())
    monkeypatch.chdir(h.cwd)
    return h


def _parse(module, monkeypatch, args: list[str]):
    """parse_arguments() of main.py for *args*: the namespace, or the SystemExit it raised."""
    monkeypatch.setattr(sys, "argv", ["main.py", *args])
    try:
        return module.parse_arguments()
    except SystemExit as exc:
        return exc


def _help_text(harness, monkeypatch, capsys) -> str:
    monkeypatch.setenv("COLUMNS", HELP_COLUMNS)
    with pytest.raises(SystemExit) as exc:
        harness.run("--help")
    assert exc.value.code == 0
    return capsys.readouterr().out


def _comparable_help(text: str) -> str:
    """Help text without the usage lines, the --log entry, the "profiling" option group and trailing blanks."""
    lines = text.splitlines()
    first_blank = lines.index("")
    kept: list[str] = []
    skipping = None
    for line in lines[first_blank + 1 :]:
        if line == "profiling:":
            skipping = "group"
            continue
        if skipping == "group":
            if line == "":
                skipping = None
            continue
        if line.startswith("  --log"):
            skipping = "entry"
            continue
        if skipping == "entry":
            if line.startswith("    "):
                continue
            skipping = None
        kept.append(line)
    return "\n".join(kept).rstrip()


def _write_config(module, path: Path, log) -> Path:
    config = module.create_default_config()
    config.log = log
    path.parent.mkdir(parents=True, exist_ok=True)
    module.ConfigLoader.save_to_yaml(config, str(path))
    return path


# --- ScriptBuilder subclass ---------------------------------------------------------------------------


# REQ-REFACTOR-BDRABCZUK-763BD3E85E5E
def test_main_defines_one_script_builder_subclass():
    classes = _script_classes(_load_main_module())
    assert len(classes) == 1, f"main.py defines {classes!r}, expected exactly one ScriptBuilder subclass"


# REQ-REFACTOR-BDRABCZUK-46E516AB195C
def test_default_log_path_names_champollion_pipeline():
    script = _script_class(_load_main_module())()
    before = datetime.now().astimezone().replace(microsecond=0)
    path = script.default_log_path()
    after = datetime.now().astimezone()
    match = re.fullmatch(r"\./\.log/champollion_pipeline_(\d{8}-\d{6})\.log", path)
    assert match, f"default_log_path() returned {path!r}"
    stamp = datetime.strptime(match.group(1), STAMP_FORMAT).replace(tzinfo=before.tzinfo)
    assert before <= stamp <= after, f"stamp {stamp} outside the call window [{before}, {after}]"


# --- main.py's own run log ----------------------------------------------------------------------------


# REQ-REFACTOR-BDRABCZUK-46AF64BFAC56
def test_bare_log_writes_main_run_log_at_default_log_path(harness):
    harness.stub_orchestrator()
    main_log = harness.tmp_path / "main_logs" / "main.log"
    harness.patch_main_default_log_path(main_log)
    assert harness.run("--log") == 0
    assert main_log.is_file(), f"main.py --log wrote no run log at default_log_path() {main_log}"
    assert main_log.read_text().startswith(RUN_LOG_HEADER)


# REQ-REFACTOR-BDRABCZUK-08F78D6AA486
def test_bare_log_main_run_log_contains_orchestrator_output(harness):
    harness.stub_orchestrator()
    main_log = harness.tmp_path / "main_logs" / "main.log"
    harness.patch_main_default_log_path(main_log)
    assert harness.run("--log") == 0
    assert main_log.is_file(), f"main.py --log wrote no run log at default_log_path() {main_log}"
    assert ORCHESTRATOR_MARKER in main_log.read_text()


# REQ-REFACTOR-BDRABCZUK-25ABF46F0A6A
def test_log_dir_creates_no_main_run_log(harness):
    harness.stub_orchestrator()
    main_log = harness.tmp_path / "main_logs" / "main.log"
    harness.patch_main_default_log_path(main_log, required=False)
    log_dir = harness.tmp_path / "run_logs"
    assert harness.run(f"--log={log_dir}") == 0
    assert harness.files_under(log_dir) == [], "main.py --log=DIR wrote a run log of its own into DIR"
    assert not main_log.exists(), "main.py --log=DIR wrote a run log at default_log_path()"
    assert harness.files_under(harness.cwd) == []


# REQ-REFACTOR-BDRABCZUK-517850E82143
@pytest.mark.parametrize("config_log", [None, "", "/from/yaml"])
def test_no_log_option_creates_no_main_run_log(harness, config_log):
    harness.stub_orchestrator()
    main_log = harness.tmp_path / "main_logs" / "main.log"
    harness.patch_main_default_log_path(main_log, required=False)
    config_file = _write_config(harness.module, harness.tmp_path / "cfg" / "pipeline.yaml", config_log)
    assert harness.run("--config", str(config_file)) == 0
    assert not main_log.exists(), "main.py without --log wrote a run log at default_log_path()"
    assert harness.files_under(harness.cwd) == []
    assert harness.files_under(harness.tmp_path / "main_logs") == []


# --- Process setup, update check, exit code ---------------------------------------------------------


# REQ-REFACTOR-BDRABCZUK-8A97252746D5
def test_init_pipeline_process_runs_before_update_check(harness):
    harness.stub_orchestrator()
    assert harness.run() == 0
    assert "init_pipeline_process" in harness.calls
    assert "check_for_updates" in harness.calls
    assert harness.calls.index("init_pipeline_process") < harness.calls.index("check_for_updates"), harness.calls


# REQ-REFACTOR-BDRABCZUK-C1849A047B57
def test_update_check_runs_once(harness):
    harness.stub_orchestrator()
    assert harness.run() == 0
    assert harness.calls.count("check_for_updates") == 1, harness.calls


# REQ-REFACTOR-BDRABCZUK-807086B7ABD0
@pytest.mark.parametrize("exit_code", [0, 1, 3])
def test_main_returns_orchestrator_exit_code(harness, exit_code):
    harness.stub_orchestrator(exit_code=exit_code)
    assert harness.run() == exit_code


# --- Command line and help ------------------------------------------------------------------------------


# REQ-REFACTOR-BDRABCZUK-52348F7590DE
@pytest.mark.parametrize(
    ("argv", "expected"), GOLDEN_COMMAND_LINES, ids=[" ".join(a) or "<none>" for a, _ in GOLDEN_COMMAND_LINES]
)
def test_golden_command_line_parses_as_at_cbe51d7(monkeypatch, argv, expected):
    result = _parse(_load_main_module(), monkeypatch, argv)
    if isinstance(expected, int):
        assert isinstance(result, SystemExit), f"{argv} parsed to {result!r}, expected exit status {expected}"
        assert result.code == expected
        return
    assert not isinstance(result, SystemExit), f"{argv} exited with status {result.code}"
    values = vars(result)
    assert {key: values.get(key, "<missing>") for key in expected} == expected


# REQ-REFACTOR-BDRABCZUK-97F9D32446CC
def test_help_usage_names_champollion_pipeline(harness, monkeypatch, capsys):
    text = _help_text(harness, monkeypatch, capsys)
    assert text.startswith(f"usage: {SCRIPT_NAME} "), text.splitlines()[0]


# REQ-REFACTOR-BDRABCZUK-09DD7C8F5B7E
def test_help_text_unchanged_since_cbe51d7(harness, monkeypatch, capsys):
    text = _help_text(harness, monkeypatch, capsys)
    assert _comparable_help(text) == _comparable_help(CBE51D7_HELP)


# REQ-REFACTOR-BDRABCZUK-80A48D228105
def test_parser_accepts_profile_dir(monkeypatch, tmp_path):
    profile_dir = str(tmp_path / "profile")
    result = _parse(_load_main_module(), monkeypatch, ["--profile", profile_dir])
    assert not isinstance(result, SystemExit), f"--profile DIR refused with exit status {result.code}"
    assert result.profile == profile_dir


# REQ-REFACTOR-BDRABCZUK-13E086BC7AE2
def test_parser_rejects_profile_mode_without_profile(monkeypatch, capsys):
    result = _parse(_load_main_module(), monkeypatch, ["--profile-mode", "cpu"])
    assert isinstance(result, SystemExit), f"--profile-mode without --profile parsed to {result!r}"
    assert result.code == 2
    assert "--profile-mode requires --profile" in capsys.readouterr().err


# --- PipelineConfig.log through main() (reused RUNLOG requirements) -------------------------------------


# REQ-RUNLOG-BDRABCZUK-CE60EAEC74A5
def test_bare_log_sets_config_log_empty_through_main(harness):
    stub = harness.stub_orchestrator()
    harness.patch_main_default_log_path(harness.tmp_path / "main_logs" / "main.log", required=False)
    config_file = _write_config(harness.module, harness.tmp_path / "cfg" / "pipeline.yaml", "/from/yaml")
    assert harness.run("--config", str(config_file), "--log") == 0
    assert [config.log for config in stub.configs] == [""]


# REQ-RUNLOG-BDRABCZUK-4C92EF7F7AE4
def test_log_dir_sets_config_log_through_main(harness):
    stub = harness.stub_orchestrator()
    log_dir = str(harness.tmp_path / "run_logs")
    config_file = _write_config(harness.module, harness.tmp_path / "cfg" / "pipeline.yaml", "/from/yaml")
    assert harness.run("--config", str(config_file), f"--log={log_dir}") == 0
    assert [config.log for config in stub.configs] == [log_dir]


def _combine_only_config(module, tmp_path: Path):
    """Default config with only put_together_embeddings enabled, its input present, all paths in tmp_path."""
    config = module.create_default_config()
    config.root_path = str(tmp_path)
    config.data_path = str(tmp_path / "data")
    config.models_path = str(tmp_path / "models")
    config.outputs_path = str(tmp_path / "outputs")  # noqa: V101
    config.log_to_file = False  # noqa: V101
    for stage in config.stages:
        config.stages[stage] = False
    config.stages["put_together_embeddings"] = True
    embeddings = tmp_path / "region_embeddings"
    embeddings.mkdir()
    config.dataset.embeddings_path = str(embeddings)
    config.dataset.datasets_root = str(tmp_path / "data" / "DS")
    return config


@pytest.fixture
def combine_stage(harness, monkeypatch):
    """Real orchestrator running only put_together_embeddings, whose script prints a marker."""
    config = _combine_only_config(harness.module, harness.tmp_path)
    monkeypatch.setattr(harness.module, "create_default_config", lambda: config)

    def _run(_self):
        print(STAGE_MARKER, flush=True)
        return 0

    monkeypatch.setattr(PutTogetherEmbeddings, "run", _run)
    return harness


# REQ-RUNLOG-BDRABCZUK-F2F6A21FE6BB
def test_bare_log_writes_stage_run_log_at_stage_default_log_path(combine_stage, monkeypatch):
    harness = combine_stage
    stage_log = harness.tmp_path / "stage_logs" / "combine.log"
    monkeypatch.setattr(PutTogetherEmbeddings, "default_log_path", lambda _self: str(stage_log))
    harness.patch_main_default_log_path(harness.tmp_path / "main_logs" / "main.log", required=False)
    assert harness.run("--log") == 0
    assert stage_log.is_file(), f"main.py --log wrote no stage run log at {stage_log}"
    text = stage_log.read_text()
    assert text.startswith(RUN_LOG_HEADER)
    assert STAGE_MARKER in text


# REQ-RUNLOG-BDRABCZUK-1859A582D928
def test_log_dir_writes_stage_run_log_in_dir(combine_stage):
    harness = combine_stage
    log_dir = harness.tmp_path / "run_logs"
    assert harness.run(f"--log={log_dir}") == 0
    logs = harness.files_under(log_dir)
    assert len(logs) == 1, f"expected one run log in {log_dir}, found {logs}"
    assert re.fullmatch(r"put_together_embeddings_\d{8}-\d{6}\.log", logs[0].name), logs[0].name
    assert STAGE_MARKER in logs[0].read_text()
