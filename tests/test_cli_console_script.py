"""The installed console script champollion-pipeline (TASK-CLICMD-BDRABCZUK-0FD3652CD077).

The orchestrator and the command line of main.py move to src/champollion_pipeline/cli.py, declared in
pyproject.toml's [project.scripts] as champollion-pipeline = "champollion_pipeline.cli:main"; main.py stays
a thin shim running the same main().

The command itself is not installed in the test environment until the package is reinstalled, so the
console script is run the way the generated wrapper runs it: ``sys.exit(champollion_pipeline.cli.main())``
in a new interpreter, with src/ on the path. python main.py is run as a script from the repo root.
Both runs stop at argument parsing (--help or a refused command line), so no stage and no update check run.

Requirements:
REQ-CLICMD-BDRABCZUK-0282A356B5BE (stakeholder: an installed champollion-pipeline command),
REQ-CLICMD-BDRABCZUK-964B81082CD1 ([project.scripts] maps champollion-pipeline to champollion_pipeline.cli:main),
REQ-CLICMD-BDRABCZUK-77ED45C8339F (--help equals python main.py --help),
REQ-CLICMD-BDRABCZUK-A5C5DAA64E37 (exit status equals python main.py's),
REQ-CLICMD-BDRABCZUK-2E52CDCD561A (main.py defines zero classes),
REQ-CLICMD-BDRABCZUK-DAEA1FD7FDB7 (main.py defines zero functions),
REQ-CLICMD-BDRABCZUK-80A5E739979B (cli binds the module-level names of main.py at da693ee),
REQ-CLICMD-BDRABCZUK-8D525167B449 (importing cli adds no toolchain module),
REQ-CLICMD-BDRABCZUK-9B2E86A01DD3 (--log=DIR sets PipelineConfig.log),
REQ-CLICMD-BDRABCZUK-9234E2F02F03 (--log=DIR writes no run log of the champollion-pipeline run),
REQ-CLICMD-BDRABCZUK-0F57CD9E94BB (README Console scripts row),
REQ-CLICMD-BDRABCZUK-33A6CF5E2DA5 (README Run logs row),
REQ-CLICMD-BDRABCZUK-E4B229E2CE70 (README Run logs states champollion-pipeline --log=DIR).
"""

import ast
import importlib
import inspect
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_PY = REPO_ROOT / "main.py"
README = REPO_ROOT / "README.md"
COMMAND = "champollion-pipeline"
CLI_MODULE = "champollion_pipeline.cli"
ENTRY_POINT = f"{CLI_MODULE}:main"
TOOLCHAIN_PACKAGES = ("soma", "cortical_tiles", "champollion")
MODULES_MARKER = "MODULES="
HELP_COLUMNS = "100"
LOG_DIR_STAMP = "DIR/<stage>_<YYYYmmdd-HHMMSS>.log"

# What the generated console-script wrapper does.
_RUN_CONSOLE_SCRIPT = "import sys\nfrom champollion_pipeline.cli import main\nsys.exit(main())\n"

# Module-level names main.py bound at commit da693ee (imports, classes, functions, assignments).
MAIN_PY_DA693EE_NAMES = (
    "AbstractContextManager",
    "AmbiguousGraphFolders",
    "ArgumentConflictError",
    "ChampollionPipeline",
    "ClassVar",
    "ConfigLoader",
    "DERIVATIVES_FOLDER",
    "DatasetConfig",
    "DatasetRootNotInferable",
    "EnvFileError",
    "FileIndexStageIndexer",
    "GenerateChampollionConfigStage",
    "GenerateEmbeddingsStage",
    "GenerateMorphologistGraphsStage",
    "GenerateSnapshotsStage",
    "InputChoiceError",
    "InputKind",
    "Iterator",
    "ModuleType",
    "PARTIAL_STATUS",
    "Path",
    "PipelineConfig",
    "PipelineOrchestrator",
    "PipelineStage",
    "PutTogetherEmbeddingsStage",
    "RunCorticalTilesStage",
    "SCRIPT_DESCRIPTION",
    "SCRIPT_NAME",
    "SKELETON_SIDES",
    "ScriptBuilder",
    "StageIndexer",
    "StageResult",
    "StreamingSettings",
    "_HELP_EPILOG",
    "_HF_REPO_ID_REQUIRED",
    "_MORPHOLOGIST_STAGE",
    "_USAGE_ERROR_STATUS",
    "_apply_cli_overrides",
    "_apply_input_kind",
    "_apply_morphologist_input",
    "_apply_t1_input",
    "_build_streaming_settings",
    "_compute_combined_output_dir",
    "_compute_configs_dir",
    "_compute_default_datasets_root",
    "_compute_models_source",
    "_compute_morphologist_subjects_dir",
    "_compute_region_embeddings_source",
    "_compute_snapshots_embeddings_dir",
    "_compute_snapshots_output_dir",
    "_compute_stage4_crops_dir",
    "_default_stage_indexer",
    "_infer_configless_paths",
    "_load_config",
    "_prepare_configless_input_run",
    "_profiling_session",
    "_resolve_streaming_input",
    "_run_streaming_mode",
    "_save_config_template",
    "_scoped_env",
    "_should_rederive_datasets_root",
    "_skip_morphologist_stage",
    "_stage_segment",
    "_start_config_profiling",
    "active_profiling",
    "add_config_arguments",
    "add_env_argument",
    "add_path_arguments",
    "apply_config_arguments",
    "apply_env_values",
    "argparse",
    "check_for_updates",
    "check_input_kind_value",
    "choose_input_kind",
    "choose_subjects_folder",
    "compute_champollion_configs_root",
    "compute_combined_embeddings_dir",
    "compute_region_embeddings_dir",
    "compute_region_model_names",
    "compute_skeleton_pattern",
    "compute_snapshots_dir",
    "compute_versioned_crops_dir",
    "contextmanager",
    "create_default_config",
    "existing_dir",
    "find_dataset_folder",
    "fold_path_arguments",
    "infer_paths",
    "init_pipeline_process",
    "logger",
    "logging",
    "main",
    "nullcontext",
    "os",
    "parse_arguments",
    "read_env_values",
    "stage1_subjects_dir",
    "start_profiling",
    "sys",
    "write_replay_config",
)

# Command lines that both programs refuse or answer during argument parsing, before any stage runs.
PARSE_TIME_COMMAND_LINES = [
    pytest.param(["--help"], id="help"),
    pytest.param(["--no-such-option"], id="unknown-option"),
    pytest.param(["--profile-mode", "cprofile"], id="profile-mode-without-profile"),
    pytest.param(["--verb"], id="abbreviated-option"),
    pytest.param(["--mode", "no-such-mode"], id="invalid-mode-choice"),
]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT / "src"), env.get("PYTHONPATH")) if p)
    env["COLUMNS"] = HELP_COLUMNS
    return env


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=str(REPO_ROOT), env=_env(), capture_output=True, text=True, timeout=300)


def _run_console_script(*args: str) -> subprocess.CompletedProcess:
    return _run([sys.executable, "-c", _RUN_CONSOLE_SCRIPT, *args])


def _run_main_py(*args: str) -> subprocess.CompletedProcess:
    return _run([sys.executable, str(MAIN_PY), *args])


def _console_scripts() -> dict[str, str]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]


def _main_py_tree() -> ast.Module:
    return ast.parse(MAIN_PY.read_text(encoding="utf-8"), filename=str(MAIN_PY))


def _readme_section(heading: str) -> str:
    """Text of the README section under ``heading`` (a full heading line), up to the next heading of any level."""
    lines = README.read_text(encoding="utf-8").splitlines()
    assert heading in lines, f"README.md has no {heading!r} heading"
    body: list[str] = []
    for line in lines[lines.index(heading) + 1 :]:
        if line.startswith("#"):
            break
        body.append(line)
    return "\n".join(body)


def _table_rows(section: str) -> list[list[str]]:
    """Cells of each markdown table row of ``section``, header and separator rows left out, backticks removed."""
    rows = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip().replace("`", "") for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    return rows[1:]


# ====================== pyproject.toml and the entry point ======================


@pytest.mark.unit
def test_pyproject_maps_champollion_pipeline_to_cli_main():
    """REQ-CLICMD-BDRABCZUK-964B81082CD1, REQ-CLICMD-BDRABCZUK-0282A356B5BE."""
    scripts = _console_scripts()
    assert scripts.get(COMMAND) == ENTRY_POINT, (
        f"[project.scripts] maps {COMMAND} to {scripts.get(COMMAND)!r}, expected {ENTRY_POINT!r}"
    )


@pytest.mark.unit
def test_cli_main_takes_no_argument():
    """REQ-CLICMD-BDRABCZUK-964B81082CD1: the console-script wrapper calls main() with no argument."""
    main = importlib.import_module(CLI_MODULE).main
    assert inspect.isfunction(main)
    inspect.signature(main).bind()


# ====================== Same command line as python main.py ======================


@pytest.mark.integration
def test_help_output_equals_main_py_help():
    """REQ-CLICMD-BDRABCZUK-77ED45C8339F, REQ-CLICMD-BDRABCZUK-0282A356B5BE."""
    console = _run_console_script("--help")
    reference = _run_main_py("--help")
    assert reference.returncode == 0, f"python main.py --help exited {reference.returncode}:\n{reference.stderr}"
    assert console.returncode == 0, f"{COMMAND} --help exited {console.returncode}:\n{console.stderr}"
    assert console.stdout == reference.stdout, f"{COMMAND} --help differs from python main.py --help"


@pytest.mark.integration
@pytest.mark.parametrize("args", PARSE_TIME_COMMAND_LINES)
def test_exit_status_equals_main_py_exit_status(args):
    """REQ-CLICMD-BDRABCZUK-A5C5DAA64E37."""
    reference = _run_main_py(*args)
    console = _run_console_script(*args)
    assert "ModuleNotFoundError" not in console.stderr, f"{COMMAND} {args} did not start:\n{console.stderr}"
    assert console.returncode == reference.returncode, (
        f"{COMMAND} {args} exited {console.returncode}, python main.py exited {reference.returncode}\n{console.stderr}"
    )


# ====================== main.py is a thin shim ======================


@pytest.mark.unit
def test_main_py_defines_no_class():
    """REQ-CLICMD-BDRABCZUK-2E52CDCD561A."""
    classes = [node.name for node in ast.walk(_main_py_tree()) if isinstance(node, ast.ClassDef)]
    assert classes == [], f"main.py defines the classes {classes}"


@pytest.mark.unit
def test_main_py_defines_no_function():
    """REQ-CLICMD-BDRABCZUK-DAEA1FD7FDB7."""
    nodes = list(ast.walk(_main_py_tree()))
    functions = [node.name for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    lambdas = sum(isinstance(node, ast.Lambda) for node in nodes)
    assert functions == [], f"main.py defines the functions {functions}"
    assert lambdas == 0, f"main.py defines {lambdas} lambda function(s)"


# ====================== The orchestrator lives in champollion_pipeline.cli ======================


@pytest.mark.unit
def test_cli_binds_main_py_module_level_names():
    """REQ-CLICMD-BDRABCZUK-80A5E739979B."""
    cli = importlib.import_module(CLI_MODULE)
    missing = [name for name in MAIN_PY_DA693EE_NAMES if not hasattr(cli, name)]
    assert missing == [], f"{CLI_MODULE} binds none of {missing}"


@pytest.mark.integration
def test_importing_cli_adds_no_toolchain_module():
    """REQ-CLICMD-BDRABCZUK-8D525167B449."""
    probe = (
        "import json, sys\n"
        "before = set(sys.modules)\n"
        f"import {CLI_MODULE}\n"
        f"print({MODULES_MARKER!r} + json.dumps(sorted(set(sys.modules) - before)))\n"
    )
    result = _run([sys.executable, "-c", probe])
    assert result.returncode == 0, f"importing {CLI_MODULE} failed:\n{result.stderr}"
    lines = [line for line in result.stdout.splitlines() if line.startswith(MODULES_MARKER)]
    assert len(lines) == 1, f"probe printed no {MODULES_MARKER} line:\n{result.stdout}"
    added = json.loads(lines[0][len(MODULES_MARKER) :])
    toolchain = [name for name in added if name.split(".")[0] in TOOLCHAIN_PACKAGES]
    assert toolchain == [], f"importing {CLI_MODULE} loaded {toolchain}"


# ====================== --log=DIR keeps the main.py folder meaning ======================


class _StubOrchestrator:
    """Records the PipelineConfig it is built with; run() returns 0."""

    configs: ClassVar[list] = []

    def __init__(self, config):
        type(self).configs.append(config)

    def run(self) -> int:
        return 0


class _CliHarness:
    """champollion_pipeline.cli.main() run in-process as champollion-pipeline.

    Built inside the test body, so a missing name of the cli module fails the test rather than a fixture.
    The process-wide setup and the update check are no-ops, the working directory is temporary and
    PipelineOrchestrator is a stub recording its config.
    """

    def __init__(self, monkeypatch, tmp_path: Path):
        self.monkeypatch = monkeypatch
        self.tmp_path = tmp_path
        self.cli = importlib.import_module(CLI_MODULE)
        self.stub = type("StubOrchestrator", (_StubOrchestrator,), {"configs": []})
        self.stub.STAGE_REGISTRY = self.cli.PipelineOrchestrator.STAGE_REGISTRY
        monkeypatch.setattr(self.cli, "PipelineOrchestrator", self.stub)
        monkeypatch.setattr(self.cli, "init_pipeline_process", lambda *a, **k: logging.getLogger())
        monkeypatch.setattr(self.cli, "check_for_updates", lambda *a, **k: None, raising=False)
        monkeypatch.setattr("champollion_utils.script_builder.check_for_updates", lambda *a, **k: None)
        monkeypatch.setattr("champollion_utils.script_builder.init_process", lambda *a, **k: logging.getLogger())
        self.cwd = tmp_path / "cwd"
        self.cwd.mkdir()
        monkeypatch.chdir(self.cwd)

    def config_file(self, log_value: str | None) -> Path:
        config = self.cli.create_default_config()
        config.log = log_value
        path = self.tmp_path / "cfg" / "pipeline.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.cli.ConfigLoader.save_to_yaml(config, str(path))
        return path

    def run(self, *args: str) -> int:
        self.monkeypatch.setattr(sys, "argv", [COMMAND, *args])
        return self.cli.main()


def _files_under(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else []


@pytest.mark.unit
def test_log_dir_sets_pipeline_config_log(monkeypatch, tmp_path):
    """REQ-CLICMD-BDRABCZUK-9B2E86A01DD3."""
    harness = _CliHarness(monkeypatch, tmp_path)
    log_dir = tmp_path / "run_logs"
    exit_code = harness.run("--config", str(harness.config_file("/from/yaml")), f"--log={log_dir}")
    assert exit_code == 0
    configs = harness.stub.configs
    assert len(configs) == 1, f"{COMMAND} built {len(configs)} orchestrators"
    assert configs[0].log == str(log_dir), f"{COMMAND} --log={log_dir} set PipelineConfig.log to {configs[0].log!r}"


@pytest.mark.unit
def test_log_dir_creates_no_own_run_log(monkeypatch, tmp_path):
    """REQ-CLICMD-BDRABCZUK-9234E2F02F03."""
    harness = _CliHarness(monkeypatch, tmp_path)
    own_log = tmp_path / "own_logs" / "champollion_pipeline.log"
    monkeypatch.setattr(harness.cli.ChampollionPipeline, "default_log_path", lambda _self: str(own_log))
    log_dir = tmp_path / "run_logs"
    exit_code = harness.run("--config", str(harness.config_file(None)), f"--log={log_dir}")
    assert exit_code == 0
    assert _files_under(log_dir) == [], f"{COMMAND} --log=DIR wrote a run log of its own into DIR"
    assert not own_log.exists(), f"{COMMAND} --log=DIR wrote a run log at default_log_path()"
    assert _files_under(harness.cwd) == [], f"{COMMAND} --log=DIR wrote files in the working directory"


# ====================== README ======================


@pytest.mark.smoke
def test_readme_console_scripts_table_lists_champollion_pipeline():
    """REQ-CLICMD-BDRABCZUK-0F57CD9E94BB."""
    commands = [row[0] for row in _table_rows(_readme_section("### Console scripts"))]
    assert COMMAND in commands, f"README Console scripts table has no {COMMAND} row: {commands}"


@pytest.mark.smoke
def test_readme_run_logs_table_row_for_champollion_pipeline():
    """REQ-CLICMD-BDRABCZUK-33A6CF5E2DA5."""
    rows = [row for row in _table_rows(_readme_section("## Run logs")) if row[0] == COMMAND]
    assert len(rows) == 1, f"README Run logs table has {len(rows)} {COMMAND} rows"
    assert "./.log" in rows[0][1], f"README Run logs row of {COMMAND} gives the folder {rows[0][1]!r}, not ./.log"


@pytest.mark.smoke
def test_readme_run_logs_states_log_dir_for_champollion_pipeline():
    """REQ-CLICMD-BDRABCZUK-E4B229E2CE70."""
    paragraphs = [" ".join(p.split()).replace("`", "") for p in _readme_section("## Run logs").split("\n\n")]
    stating = [p for p in paragraphs if f"{COMMAND} --log=DIR" in p and LOG_DIR_STAMP in p]
    assert stating, f"README Run logs states no {COMMAND} --log=DIR ... {LOG_DIR_STAMP}"
