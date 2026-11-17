#!/usr/bin/env python3
"""main.py --env: the four exempt fields read from a .env file and from the process environment.

The exempt fields hf_token, dependencies, config_path and root_path have no argument of their own; main.py reads
them from the file named by --env (./.env for a bare --env) and from the CHAMPOLLION_* variables of the process
environment (TASK-INPUTKIND-BDRABCZUK-740058459CA5, user answers 2026-11-05 and 2026-11-16). Keys: HF_TOKEN,
CHAMPOLLION_ROOT_PATH, CHAMPOLLION_CONFIG_PATH and CHAMPOLLION_DEPENDENCIES (a JSON object), plus
CHAMPOLLION_DATA_ROOT for dataset.champollion_data_root (user answer 2026-11-16 21:40; that field also has the
--champollion-data-root flag, never passed here). Precedence: process environment > .env file > --config YAML. A
missing file, an unknown CHAMPOLLION_ key or a bad CHAMPOLLION_DEPENDENCIES stops main.py with status 2 before any
stage. The file follows the python-dotenv syntax, without ${VAR} expansion. The token never reaches a log record.
With --generate-config both sources are still read and checked, but none of their values reaches the template
(user answer 2026-11-16 21:40); --env is listed in main.py --help.

The observable is the PipelineConfig main() runs: sys.argv drives the parse, the update check, the process setup and
profiling are stubbed, and PipelineOrchestrator is replaced by a recorder whose run() returns 0, so no stage executes.
Every run that passes --env first checks that main.py's parser declares --env, so a status 2 is never argparse's
own refusal of an unknown option.

Inputs:
    main.py: loaded by file path under a private module name (tests/main_stage_helpers.load_main_module)
    a temporary working folder holding the .env files and --config YAML files each test writes
    the process environment, with HF_TOKEN and every CHAMPOLLION_* variable removed, then set per test
Outputs:
    pass or fail per requirement; nothing is written outside the temporary folder

```mermaid
flowchart LR
    envfile[(.env file)] --> main[main.py main]
    procenv[(process environment)] --> main
    yaml[(--config YAML)] --> main
    main --> recorder[recording orchestrator]
    main --> logs[/log records/]
    recorder --> checks[assertions on the PipelineConfig]
    logs --> checks
```
"""

import dataclasses
import json
import logging
import os
from pathlib import Path

import pytest
import yaml

from tests.main_stage_helpers import RunOutcome, load_main_module, main_runner, ran_config

MAIN = load_main_module("_champollion_main_env_file_under_test")

REQ_FILE_ROOT_PATH = "REQ-INPUTKIND-BDRABCZUK-564A1121B511: --env sets root_path from CHAMPOLLION_ROOT_PATH"
REQ_FILE_CONFIG_PATH = "REQ-INPUTKIND-BDRABCZUK-271625471667: --env sets dataset.config_path"
REQ_FILE_DEPENDENCIES = "REQ-INPUTKIND-BDRABCZUK-CE93CCA04557: --env sets dependencies from JSON"
REQ_FILE_HF_TOKEN = "REQ-INPUTKIND-BDRABCZUK-5844FD7ECA7E: --env sets dataset.hf_token from HF_TOKEN"
REQ_TOKEN_NOT_LOGGED = "REQ-INPUTKIND-BDRABCZUK-46019E42FDD9: a token read by --env stays out of the logs"
REQ_PROCESS_ROOT_PATH = "REQ-INPUTKIND-BDRABCZUK-8AC0C7092CBA: process environment CHAMPOLLION_ROOT_PATH"
REQ_PROCESS_CONFIG_PATH = "REQ-INPUTKIND-BDRABCZUK-0CA4176CBC69: process environment CHAMPOLLION_CONFIG_PATH"
REQ_PROCESS_DEPENDENCIES = "REQ-INPUTKIND-BDRABCZUK-DC9751B68DED: process environment CHAMPOLLION_DEPENDENCIES"
REQ_PROCESS_OVER_FILE = "REQ-INPUTKIND-BDRABCZUK-0873EF3BD6D1: process environment wins over the --env file"
REQ_FILE_OVER_YAML = "REQ-INPUTKIND-BDRABCZUK-C2BD255F6D4F: the --env file wins over the --config YAML"
REQ_PROCESS_OVER_YAML = "REQ-INPUTKIND-BDRABCZUK-838D79B157EC: process environment wins over the --config YAML"
REQ_BARE_ENV = "REQ-INPUTKIND-BDRABCZUK-0AD3153614BB: a bare --env reads .env in the working directory"
REQ_MISSING_STOPS = "REQ-INPUTKIND-BDRABCZUK-81ED65D8A899: a missing --env file stops main.py with status 2"
REQ_MISSING_ERROR = "REQ-INPUTKIND-BDRABCZUK-AFFF34577A36: a missing --env file logs one ERROR naming --env, PATH"
REQ_UNKNOWN_STOPS = "REQ-INPUTKIND-BDRABCZUK-846490C94002: an unknown CHAMPOLLION_ key stops main.py with status 2"
REQ_UNKNOWN_ERROR = "REQ-INPUTKIND-BDRABCZUK-6A9CADD14F5A: an unknown CHAMPOLLION_ key is named in an ERROR"
REQ_FOREIGN_IGNORED = "REQ-INPUTKIND-BDRABCZUK-13467703DD12: a non-CHAMPOLLION_ key other than HF_TOKEN is ignored"
REQ_BAD_FILE_STOPS = "REQ-INPUTKIND-BDRABCZUK-E998682626D7: a bad CHAMPOLLION_DEPENDENCIES in the file: status 2"
REQ_BAD_ERROR = "REQ-INPUTKIND-BDRABCZUK-DF345FB3A760: a bad CHAMPOLLION_DEPENDENCIES is named in an ERROR"
REQ_BAD_PROCESS_STOPS = "REQ-INPUTKIND-BDRABCZUK-231B5E3F4A5D: a bad CHAMPOLLION_DEPENDENCIES in the environment: 2"
REQ_DOTENV_SYNTAX = "REQ-INPUTKIND-BDRABCZUK-8FE9DF3C4629: the --env file follows the python-dotenv syntax"
REQ_NO_EXPANSION = "REQ-INPUTKIND-BDRABCZUK-5FF1D1C09E01: the --env file expands no variable"
REQ_UNKNOWN_PROCESS_WARNING = "REQ-INPUTKIND-BDRABCZUK-70C125F1A7B0: an unknown CHAMPOLLION_ variable: one WARNING"
REQ_UNKNOWN_PROCESS_IGNORED = "REQ-INPUTKIND-BDRABCZUK-91967CFBDF99: an unknown CHAMPOLLION_ variable is ignored"
REQ_PROCESS_TOKEN_OVER_YAML = "REQ-INPUTKIND-BDRABCZUK-7ACB4F99987A: process environment HF_TOKEN wins over the YAML"
REQ_TEMPLATE_NO_FILE_VALUE = "REQ-INPUTKIND-BDRABCZUK-DD970776FE2A: --generate-config writes no --env file value"
REQ_TEMPLATE_NO_PROCESS_VALUE = "REQ-INPUTKIND-BDRABCZUK-57A2BAC10FE6: --generate-config writes no environment value"
REQ_HELP_LISTS_ENV = "REQ-INPUTKIND-BDRABCZUK-865919F7E691: main.py --help lists the --env option"
REQ_PROCESS_DATA_ROOT = "REQ-INPUTKIND-BDRABCZUK-EF00718D15E1: process environment CHAMPOLLION_DATA_ROOT"
REQ_FILE_DATA_ROOT = "REQ-INPUTKIND-BDRABCZUK-862D46E04350: --env sets dataset.champollion_data_root"
REQ_DATA_ROOT_PROCESS_OVER_FILE = "REQ-INPUTKIND-BDRABCZUK-D1638C0B0D7C: CHAMPOLLION_DATA_ROOT, environment over file"
REQ_DATA_ROOT_FILE_OVER_YAML = "REQ-INPUTKIND-BDRABCZUK-4A47956A625D: CHAMPOLLION_DATA_ROOT, --env file over YAML"
REQ_DATA_ROOT_PROCESS_OVER_YAML = "REQ-INPUTKIND-BDRABCZUK-C2BD436EB8F4: CHAMPOLLION_DATA_ROOT, environment over YAML"
REQ_DATA_ROOT_NO_WARNING = "REQ-INPUTKIND-BDRABCZUK-6D6C8E11D0D6: CHAMPOLLION_DATA_ROOT draws no WARNING"
REQ_DATA_ROOT_FLAG_OVER_PROCESS = "REQ-INPUTKIND-BDRABCZUK-305A0B34C0B0: --champollion-data-root over the environment"

USAGE_ERROR_STATUS = 2
ENV_FILE_NAME = "run.env"
TOKEN_IN_FILE = "hf_fileTokenD41D8CD98F00B204"
TOKEN_IN_PROCESS = "hf_processTokenE9800998ECF8427E"
TOKEN_IN_YAML = "hf_yamlTokenC4CA4238A0B9"
UNKNOWN_PROCESS_VARIABLE = "CHAMPOLLION_FOO"
FILE_DEPENDENCIES = {"generate_embeddings": [], "put_together_embeddings": ["generate_embeddings"]}
PROCESS_DEPENDENCIES = {"generate_embeddings": [], "generate_snapshots": ["generate_embeddings"]}
YAML_DEPENDENCIES = {"generate_embeddings": [], "generate_snapshots": []}
SECRET_VARIABLES = ("HF_TOKEN",)


@dataclasses.dataclass(frozen=True)
class FieldVariable:
    """One exempt field: the variable naming it and three distinct values, one per source."""

    variable: str
    field_path: tuple[str, ...]
    in_file: object
    in_process: object
    in_yaml: object


FIELD_VARIABLES = [
    FieldVariable("HF_TOKEN", ("dataset", "hf_token"), TOKEN_IN_FILE, TOKEN_IN_PROCESS, TOKEN_IN_YAML),
    FieldVariable("CHAMPOLLION_ROOT_PATH", ("root_path",), "/file/root", "/process/root", "/yaml/root"),
    FieldVariable(
        "CHAMPOLLION_CONFIG_PATH",
        ("dataset", "config_path"),
        "/file/config.yaml",
        "/process/config.yaml",
        "/yaml/config.yaml",
    ),
    FieldVariable(
        "CHAMPOLLION_DEPENDENCIES", ("dependencies",), FILE_DEPENDENCIES, PROCESS_DEPENDENCIES, YAML_DEPENDENCIES
    ),
]
CHAMPOLLION_FIELD_VARIABLES = [case for case in FIELD_VARIABLES if case.variable.startswith("CHAMPOLLION_")]
DATA_ROOT = FieldVariable(
    "CHAMPOLLION_DATA_ROOT",
    ("dataset", "champollion_data_root"),
    "/file/data-root",
    "/process/data-root",
    "/yaml/data-root",
)
# Every known variable: the four exempt fields and CHAMPOLLION_DATA_ROOT.
KNOWN_VARIABLES = [*FIELD_VARIABLES, DATA_ROOT]

# main.py command lines a check of the sources applies to: a stage run, and a --generate-config template run.
RUN_KINDS = [
    pytest.param((), id="run"),
    pytest.param(("--generate-config", "template.yaml"), id="generate-config"),
]

# Files that hold CHAMPOLLION_ROOT_PATH=/dotenv/root in one python-dotenv form each.
DOTENV_FORMS = [
    pytest.param('CHAMPOLLION_ROOT_PATH="/dotenv/root"\n', id="double-quotes"),
    pytest.param("CHAMPOLLION_ROOT_PATH='/dotenv/root'\n", id="single-quotes"),
    pytest.param("export CHAMPOLLION_ROOT_PATH=/dotenv/root\n", id="export-prefix"),
    pytest.param("# the root of this run\nCHAMPOLLION_ROOT_PATH=/dotenv/root\n", id="comment-line"),
    pytest.param("\n\nCHAMPOLLION_ROOT_PATH=/dotenv/root\n\n", id="blank-lines"),
    pytest.param("CHAMPOLLION_ROOT_PATH=/dotenv/root # the root of this run\n", id="inline-comment"),
    pytest.param("CHAMPOLLION_ROOT_PATH = /dotenv/root\n", id="spaces-around-equals"),
]

# CHAMPOLLION_DEPENDENCIES values that are not a JSON object of stage names to stage name lists.
BAD_DEPENDENCIES = [
    pytest.param("not json at all", id="not-json"),
    pytest.param('["generate_embeddings"]', id="json-list"),
    pytest.param('{"generate_embeddings": "generate_champollion_config"}', id="value-not-a-list"),
    pytest.param('{"generate_embeddings": [1]}', id="list-of-numbers"),
]


@pytest.fixture
def work_dir(monkeypatch, tmp_path) -> Path:
    """tmp_path as the working folder, with HF_TOKEN and every CHAMPOLLION_* variable removed from the environment."""
    monkeypatch.chdir(tmp_path)
    for name in list(os.environ):
        if name in SECRET_VARIABLES or name.startswith("CHAMPOLLION_"):
            monkeypatch.delenv(name)
    return tmp_path


def _require_env_if_named(cli_args) -> None:
    """Check --env is declared before a run that names it."""
    if "--env" in cli_args:
        _require_env_option()


def _require_env_option() -> None:
    """Fail unless main.py's parser declares --env, so a status 2 is never argparse's unknown-option refusal."""
    options = MAIN.ChampollionPipeline().parser._option_string_actions
    assert "--env" in options, "main.py has no --env option"


@pytest.fixture
def run_main(monkeypatch, work_dir):
    """Run main() with the given arguments; return its exit status and the configurations it ran."""
    return main_runner(MAIN, monkeypatch, before_run=_require_env_if_named)


def _run_config(run_main, *cli_args: str):
    return ran_config(run_main(*cli_args), cli_args)


def _assert_stopped_before_stages(outcome: RunOutcome, message: str) -> None:
    """main() ended with the usage-error status and handed no configuration to the orchestrator."""
    assert outcome.status == USAGE_ERROR_STATUS, f"{message} (exit status {outcome.status})"
    assert outcome.configs == [], f"{message} (a stage run started)"


def _as_text(value: object) -> str:
    """The text a .env line or an environment variable holds for value: JSON for a dependencies object."""
    return value if isinstance(value, str) else json.dumps(value)


def _write_env_file(folder: Path, lines: dict[str, object], name: str = ENV_FILE_NAME) -> str:
    """Write one KEY=value line per entry of lines into folder/name; return the file name."""
    text = "".join(f"{key}={_as_text(value)}\n" for key, value in lines.items())
    (folder / name).write_text(text)
    return name


def _write_yaml_config(path: Path, field_path: tuple[str, ...], value: object) -> str:
    """Write the default configuration with the field at field_path set to value as a --config YAML file."""
    config_dict = dataclasses.asdict(MAIN.create_default_config())
    target = config_dict
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = value
    path.write_text(yaml.safe_dump(config_dict))
    return str(path)


def _field(config, field_path: tuple[str, ...]) -> object:
    value = config
    for name in field_path:
        value = getattr(value, name)
    return value


def _run_succeeds(run_main, *cli_args: str) -> None:
    """Run main(): a --generate-config run must end with status 0, any other run must run one configuration."""
    if "--generate-config" in cli_args:
        outcome = run_main(*cli_args)
        assert outcome.status == 0, f"main() exit status {outcome.status} for {list(cli_args)}, expected 0"
    else:
        _run_config(run_main, *cli_args)


def _template(run_main, folder: Path, name: str, *cli_args: str) -> dict:
    """Run main.py --generate-config folder/name with cli_args; return the template it wrote, as a dict."""
    outcome = run_main("--generate-config", name, *cli_args)
    assert outcome.status == 0, f"--generate-config {name} {list(cli_args)}: exit status {outcome.status}"
    return yaml.safe_load((folder / name).read_text())


def _records_naming(caplog, text: str) -> list[logging.LogRecord]:
    """The log records whose message contains text; caplog sees a propagating record once per logger: keep one."""
    records = {id(record): record for record in caplog.records}.values()
    return [record for record in records if text in record.getMessage()]


def _error_messages(pipeline_caplog) -> list[str]:
    return [record.getMessage() for record in pipeline_caplog.records if record.levelno >= logging.ERROR]


# ====================== The --env file sets each exempt field ======================


@pytest.mark.unit
def test_env_sets_root_path(run_main, work_dir):
    """--env PATH without --config, PATH holding CHAMPOLLION_ROOT_PATH=<value>: root_path is <value>."""
    env_file = _write_env_file(work_dir, {"CHAMPOLLION_ROOT_PATH": "/file/root"})
    config = _run_config(run_main, "--env", env_file)
    assert config.root_path == "/file/root", REQ_FILE_ROOT_PATH


@pytest.mark.unit
def test_env_sets_config_path(run_main, work_dir):
    """--env PATH without --config, PATH holding CHAMPOLLION_CONFIG_PATH=<value>: dataset.config_path is <value>."""
    env_file = _write_env_file(work_dir, {"CHAMPOLLION_CONFIG_PATH": "/file/config.yaml"})
    config = _run_config(run_main, "--env", env_file)
    assert config.dataset.config_path == "/file/config.yaml", REQ_FILE_CONFIG_PATH


@pytest.mark.unit
def test_env_sets_dependencies(run_main, work_dir):
    """--env PATH without --config, PATH holding CHAMPOLLION_DEPENDENCIES=<json>: dependencies is that object."""
    env_file = _write_env_file(work_dir, {"CHAMPOLLION_DEPENDENCIES": FILE_DEPENDENCIES})
    config = _run_config(run_main, "--env", env_file)
    assert config.dependencies == FILE_DEPENDENCIES, REQ_FILE_DEPENDENCIES


@pytest.mark.unit
def test_env_sets_hf_token(run_main, work_dir):
    """--env PATH without --config, no HF_TOKEN in the environment, PATH holding HF_TOKEN=<token>: hf_token is it."""
    env_file = _write_env_file(work_dir, {"HF_TOKEN": TOKEN_IN_FILE})
    config = _run_config(run_main, "--env", env_file)
    assert config.dataset.hf_token == TOKEN_IN_FILE, REQ_FILE_HF_TOKEN


@pytest.mark.unit
def test_token_absent_from_log_records(run_main, work_dir, caplog, pipeline_caplog):
    """--env PATH, PATH holding HF_TOKEN=<token>: no log record of the run contains <token>."""
    caplog.set_level(logging.DEBUG)
    env_file = _write_env_file(work_dir, {"HF_TOKEN": TOKEN_IN_FILE})
    _run_config(run_main, "--env", env_file, "--log-level", "DEBUG")
    records = [*caplog.records, *pipeline_caplog.records]
    leaking = [record.getMessage() for record in records if TOKEN_IN_FILE in record.getMessage()]
    assert leaking == [], f"{REQ_TOKEN_NOT_LOGGED}: {len(leaking)} log record(s) contain the token"


# ====================== The process environment sets each CHAMPOLLION_ field ======================


@pytest.mark.unit
def test_process_environment_sets_root_path(run_main, monkeypatch):
    """No --config, CHAMPOLLION_ROOT_PATH=<value> in the process environment: root_path is <value>."""
    monkeypatch.setenv("CHAMPOLLION_ROOT_PATH", "/process/root")
    config = _run_config(run_main)
    assert config.root_path == "/process/root", REQ_PROCESS_ROOT_PATH


@pytest.mark.unit
def test_process_environment_sets_config_path(run_main, monkeypatch):
    """No --config, CHAMPOLLION_CONFIG_PATH=<value> in the process environment: dataset.config_path is <value>."""
    monkeypatch.setenv("CHAMPOLLION_CONFIG_PATH", "/process/config.yaml")
    config = _run_config(run_main)
    assert config.dataset.config_path == "/process/config.yaml", REQ_PROCESS_CONFIG_PATH


@pytest.mark.unit
def test_process_environment_sets_dependencies(run_main, monkeypatch):
    """No --config, CHAMPOLLION_DEPENDENCIES=<json> in the process environment: dependencies is that object."""
    monkeypatch.setenv("CHAMPOLLION_DEPENDENCIES", json.dumps(PROCESS_DEPENDENCIES))
    config = _run_config(run_main)
    assert config.dependencies == PROCESS_DEPENDENCIES, REQ_PROCESS_DEPENDENCIES


# ====================== Precedence: process environment > .env file > --config YAML ======================


@pytest.mark.unit
@pytest.mark.parametrize("case", FIELD_VARIABLES, ids=lambda case: case.variable)
def test_process_environment_wins_over_env_file(run_main, work_dir, monkeypatch, case):
    """--env PATH, the process environment and PATH giving one variable different values: the environment's wins."""
    monkeypatch.setenv(case.variable, _as_text(case.in_process))
    env_file = _write_env_file(work_dir, {case.variable: case.in_file})
    config = _run_config(run_main, "--env", env_file)
    assert _field(config, case.field_path) == case.in_process, f"{REQ_PROCESS_OVER_FILE} ({case.variable})"


@pytest.mark.unit
@pytest.mark.parametrize("case", FIELD_VARIABLES, ids=lambda case: case.variable)
def test_env_file_wins_over_config_file(run_main, work_dir, case):
    """--config FILE and --env PATH both setting one field: the value of PATH wins."""
    config_file = _write_yaml_config(work_dir / "run.yaml", case.field_path, case.in_yaml)
    env_file = _write_env_file(work_dir, {case.variable: case.in_file})
    config = _run_config(run_main, "--config", config_file, "--env", env_file)
    assert _field(config, case.field_path) == case.in_file, f"{REQ_FILE_OVER_YAML} ({case.variable})"


@pytest.mark.unit
@pytest.mark.parametrize("case", CHAMPOLLION_FIELD_VARIABLES, ids=lambda case: case.variable)
def test_process_environment_wins_over_config_file(run_main, work_dir, monkeypatch, case):
    """--config FILE setting a field whose CHAMPOLLION_ variable the process environment holds: the environment wins."""
    config_file = _write_yaml_config(work_dir / "run.yaml", case.field_path, case.in_yaml)
    monkeypatch.setenv(case.variable, _as_text(case.in_process))
    config = _run_config(run_main, "--config", config_file)
    assert _field(config, case.field_path) == case.in_process, f"{REQ_PROCESS_OVER_YAML} ({case.variable})"


@pytest.mark.unit
def test_process_environment_hf_token_wins_over_config_file(run_main, work_dir, monkeypatch):
    """--config FILE setting dataset.hf_token, no --env, another HF_TOKEN in the process environment: it wins."""
    config_file = _write_yaml_config(work_dir / "run.yaml", ("dataset", "hf_token"), TOKEN_IN_YAML)
    monkeypatch.setenv("HF_TOKEN", TOKEN_IN_PROCESS)
    config = _run_config(run_main, "--config", config_file)
    assert config.dataset.hf_token == TOKEN_IN_PROCESS, REQ_PROCESS_TOKEN_OVER_YAML


# ====================== An unknown CHAMPOLLION_ variable in the process environment ======================


@pytest.mark.unit
@pytest.mark.parametrize("run_kind", RUN_KINDS)
def test_unknown_champollion_variable_in_process_environment_logs_one_warning(
    run_main, monkeypatch, caplog, pipeline_caplog, run_kind
):
    """An unknown CHAMPOLLION_<NAME> in the process environment: exactly one WARNING record names it.

    The generate-config case: the process environment is still checked under --generate-config.
    """
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv(UNKNOWN_PROCESS_VARIABLE, "1")
    _run_succeeds(run_main, *run_kind)
    naming = _records_naming(caplog, UNKNOWN_PROCESS_VARIABLE)
    assert len(naming) == 1, f"{REQ_UNKNOWN_PROCESS_WARNING}: {len(naming)} records name {UNKNOWN_PROCESS_VARIABLE}"
    assert naming[0].levelno == logging.WARNING, f"{REQ_UNKNOWN_PROCESS_WARNING}: level {naming[0].levelname}"


@pytest.mark.unit
def test_unknown_champollion_variable_in_process_environment_is_ignored(run_main, monkeypatch):
    """An unknown CHAMPOLLION_<NAME> in the process environment: the run goes on with the same configuration.

    CHAMPOLLION_ROOT_PATH is set in both runs, so the comparison is between runs that read the process environment.
    """
    monkeypatch.setenv("CHAMPOLLION_ROOT_PATH", "/process/root")
    without = _run_config(run_main)
    assert without.root_path == "/process/root", f"{REQ_UNKNOWN_PROCESS_IGNORED}: process environment not read"
    monkeypatch.setenv(UNKNOWN_PROCESS_VARIABLE, "1")
    with_unknown = _run_config(run_main)
    assert dataclasses.asdict(with_unknown) == dataclasses.asdict(without), REQ_UNKNOWN_PROCESS_IGNORED


# ====================== Bare --env ======================


@pytest.mark.unit
def test_bare_env_reads_working_directory_env_file(run_main, work_dir):
    """--env with no path reads .env in the current working directory."""
    _write_env_file(work_dir, {"CHAMPOLLION_ROOT_PATH": "/working-directory/root"}, name=".env")
    config = _run_config(run_main, "--env")
    assert config.root_path == "/working-directory/root", REQ_BARE_ENV


# ====================== Refusals: status 2 before any stage, one ERROR ======================


@pytest.mark.unit
@pytest.mark.parametrize("run_kind", RUN_KINDS)
def test_missing_env_file_exits_with_status_2(run_main, work_dir, run_kind):
    """--env PATH with no file at PATH: status 2, no stage runs (with --generate-config too)."""
    outcome = run_main(*run_kind, "--env", "missing.env")
    _assert_stopped_before_stages(outcome, REQ_MISSING_STOPS)


@pytest.mark.unit
def test_missing_env_file_logs_one_error(run_main, work_dir, pipeline_caplog):
    """--env PATH with no file at PATH: exactly one ERROR record, naming --env and PATH."""
    run_main("--env", "missing.env")
    errors = _error_messages(pipeline_caplog)
    assert len(errors) == 1, f"{REQ_MISSING_ERROR}: {len(errors)} ERROR records: {errors}"
    assert "--env" in errors[0], f"{REQ_MISSING_ERROR}: ERROR does not name --env: {errors[0]!r}"
    assert "missing.env" in errors[0], f"{REQ_MISSING_ERROR}: ERROR does not name the path: {errors[0]!r}"


@pytest.mark.unit
@pytest.mark.parametrize("run_kind", RUN_KINDS)
def test_unknown_champollion_key_exits_with_status_2(run_main, work_dir, run_kind):
    """--env PATH, PATH holding an unknown CHAMPOLLION_<NAME>: status 2, no stage runs (with --generate-config too)."""
    env_file = _write_env_file(work_dir, {"CHAMPOLLION_ROOT_PAHT": "/typo/root"})
    outcome = run_main(*run_kind, "--env", env_file)
    _assert_stopped_before_stages(outcome, REQ_UNKNOWN_STOPS)


@pytest.mark.unit
def test_unknown_champollion_key_error_names_key(run_main, work_dir, pipeline_caplog):
    """--env PATH, PATH holding an unknown CHAMPOLLION_<NAME>: an ERROR record names CHAMPOLLION_<NAME>."""
    env_file = _write_env_file(work_dir, {"CHAMPOLLION_ROOT_PAHT": "/typo/root"})
    run_main("--env", env_file)
    errors = _error_messages(pipeline_caplog)
    assert any("CHAMPOLLION_ROOT_PAHT" in message for message in errors), f"{REQ_UNKNOWN_ERROR}: {errors}"


@pytest.mark.unit
@pytest.mark.parametrize("foreign_variable", ["OTHER_TOOL_SETTING", "HF_HOME"])
def test_foreign_key_is_ignored(run_main, work_dir, foreign_variable):
    """--env PATH, PATH holding a variable without the CHAMPOLLION_ prefix other than HF_TOKEN: same configuration."""
    plain_file = _write_env_file(work_dir, {"CHAMPOLLION_ROOT_PATH": "/file/root"}, name="plain.env")
    foreign_file = _write_env_file(
        work_dir, {"CHAMPOLLION_ROOT_PATH": "/file/root", foreign_variable: "/foreign/value"}, name="foreign.env"
    )
    without = dataclasses.asdict(_run_config(run_main, "--env", plain_file))
    with_foreign = dataclasses.asdict(_run_config(run_main, "--env", foreign_file))
    assert with_foreign == without, f"{REQ_FOREIGN_IGNORED} ({foreign_variable})"


@pytest.mark.unit
@pytest.mark.parametrize("run_kind", RUN_KINDS)
@pytest.mark.parametrize("bad_value", BAD_DEPENDENCIES)
def test_bad_dependencies_in_env_file_exits_with_status_2(run_main, work_dir, bad_value, run_kind):
    """--env PATH, PATH holding a CHAMPOLLION_DEPENDENCIES that is no stage-name object: status 2, no stage runs."""
    env_file = _write_env_file(work_dir, {"CHAMPOLLION_DEPENDENCIES": bad_value})
    outcome = run_main(*run_kind, "--env", env_file)
    _assert_stopped_before_stages(outcome, REQ_BAD_FILE_STOPS)


@pytest.mark.unit
@pytest.mark.parametrize("bad_value", BAD_DEPENDENCIES)
def test_bad_dependencies_error_names_variable(run_main, work_dir, pipeline_caplog, bad_value):
    """--env PATH, PATH holding a bad CHAMPOLLION_DEPENDENCIES: an ERROR record names CHAMPOLLION_DEPENDENCIES."""
    env_file = _write_env_file(work_dir, {"CHAMPOLLION_DEPENDENCIES": bad_value})
    run_main("--env", env_file)
    errors = _error_messages(pipeline_caplog)
    assert any("CHAMPOLLION_DEPENDENCIES" in message for message in errors), f"{REQ_BAD_ERROR}: {errors}"


@pytest.mark.unit
@pytest.mark.parametrize("run_kind", RUN_KINDS)
@pytest.mark.parametrize("bad_value", BAD_DEPENDENCIES)
def test_bad_dependencies_in_process_environment_exits_with_status_2(run_main, monkeypatch, bad_value, run_kind):
    """A bad CHAMPOLLION_DEPENDENCIES in the process environment: status 2, no stage runs (--generate-config too)."""
    monkeypatch.setenv("CHAMPOLLION_DEPENDENCIES", bad_value)
    outcome = run_main(*run_kind)
    _assert_stopped_before_stages(outcome, REQ_BAD_PROCESS_STOPS)


# ====================== python-dotenv syntax, no expansion ======================


@pytest.mark.unit
@pytest.mark.parametrize("file_text", DOTENV_FORMS)
def test_env_file_follows_dotenv_syntax(run_main, work_dir, file_text):
    """--env PATH: PATH is read in the python-dotenv syntax (quotes, export, comments, blank lines, spaces)."""
    (work_dir / ENV_FILE_NAME).write_text(file_text)
    config = _run_config(run_main, "--env", ENV_FILE_NAME)
    assert config.root_path == "/dotenv/root", f"{REQ_DOTENV_SYNTAX}: {file_text!r} gave {config.root_path!r}"


@pytest.mark.unit
def test_env_file_expands_no_variable(run_main, work_dir, monkeypatch):
    """--env PATH, PATH holding CHAMPOLLION_ROOT_PATH=<text> with ${HOME}: root_path is <text> unchanged."""
    monkeypatch.setenv("HOME", "/home/someone")
    (work_dir / ENV_FILE_NAME).write_text("CHAMPOLLION_ROOT_PATH=${HOME}/champollion\n")
    config = _run_config(run_main, "--env", ENV_FILE_NAME)
    assert config.root_path == "${HOME}/champollion", f"{REQ_NO_EXPANSION}: root_path {config.root_path!r}"


# ====================== --generate-config: sources checked, no value in the template ======================


@pytest.mark.unit
@pytest.mark.parametrize("case", KNOWN_VARIABLES, ids=lambda case: case.variable)
def test_generate_config_writes_no_env_file_value(run_main, work_dir, case):
    """--generate-config FILE --env PATH: FILE equals the template written for the same command line without --env."""
    env_file = _write_env_file(work_dir, {case.variable: case.in_file})
    with_env = _template(run_main, work_dir, "with_env.yaml", "--env", env_file)
    without_env = _template(run_main, work_dir, "without_env.yaml")
    assert with_env == without_env, f"{REQ_TEMPLATE_NO_FILE_VALUE} ({case.variable})"


@pytest.mark.unit
@pytest.mark.parametrize("case", KNOWN_VARIABLES, ids=lambda case: case.variable)
def test_generate_config_writes_no_process_environment_value(run_main, work_dir, monkeypatch, case):
    """--generate-config FILE, a known variable in the process environment: FILE equals the template without it.

    A stage run first checks that main.py reads the variable at all, so the comparison is between runs of a main.py
    that reads the process environment.
    """
    monkeypatch.setenv(case.variable, _as_text(case.in_process))
    ran = _run_config(run_main)
    read = _field(ran, case.field_path)
    assert read == case.in_process, f"{REQ_TEMPLATE_NO_PROCESS_VALUE}: {case.variable} not read (field {read!r})"
    with_variable = _template(run_main, work_dir, "with_variable.yaml")
    monkeypatch.delenv(case.variable)
    without_variable = _template(run_main, work_dir, "without_variable.yaml")
    assert with_variable == without_variable, f"{REQ_TEMPLATE_NO_PROCESS_VALUE} ({case.variable})"


# ====================== --help ======================


@pytest.mark.unit
def test_help_lists_env_option():
    """main.py --help holds an entry whose first word is --env."""
    help_text = MAIN.ChampollionPipeline().parser.format_help()
    first_words = [line.split()[0].rstrip(",") for line in help_text.splitlines() if line.strip()]
    assert "--env" in first_words, f"{REQ_HELP_LISTS_ENV}: no --env entry in main.py --help"


# ====================== CHAMPOLLION_DATA_ROOT: dataset.champollion_data_root ======================


@pytest.mark.unit
def test_process_environment_sets_champollion_data_root(run_main, monkeypatch):
    """No --config, CHAMPOLLION_DATA_ROOT=<value> in the process environment: champollion_data_root is <value>."""
    monkeypatch.setenv(DATA_ROOT.variable, DATA_ROOT.in_process)
    config = _run_config(run_main)
    assert config.dataset.champollion_data_root == DATA_ROOT.in_process, REQ_PROCESS_DATA_ROOT


@pytest.mark.unit
def test_env_sets_champollion_data_root(run_main, work_dir):
    """--env PATH without --config, PATH holding CHAMPOLLION_DATA_ROOT=<value>: champollion_data_root is <value>."""
    env_file = _write_env_file(work_dir, {DATA_ROOT.variable: DATA_ROOT.in_file})
    config = _run_config(run_main, "--env", env_file)
    assert config.dataset.champollion_data_root == DATA_ROOT.in_file, REQ_FILE_DATA_ROOT


@pytest.mark.unit
def test_process_environment_wins_over_env_file_for_champollion_data_root(run_main, work_dir, monkeypatch):
    """--env PATH, the process environment and PATH giving CHAMPOLLION_DATA_ROOT different values: the environment's."""
    monkeypatch.setenv(DATA_ROOT.variable, DATA_ROOT.in_process)
    env_file = _write_env_file(work_dir, {DATA_ROOT.variable: DATA_ROOT.in_file})
    config = _run_config(run_main, "--env", env_file)
    assert config.dataset.champollion_data_root == DATA_ROOT.in_process, REQ_DATA_ROOT_PROCESS_OVER_FILE


@pytest.mark.unit
def test_env_file_wins_over_config_file_for_champollion_data_root(run_main, work_dir):
    """--config FILE setting dataset.champollion_data_root, --env PATH holding CHAMPOLLION_DATA_ROOT: PATH wins."""
    config_file = _write_yaml_config(work_dir / "run.yaml", DATA_ROOT.field_path, DATA_ROOT.in_yaml)
    env_file = _write_env_file(work_dir, {DATA_ROOT.variable: DATA_ROOT.in_file})
    config = _run_config(run_main, "--config", config_file, "--env", env_file)
    assert config.dataset.champollion_data_root == DATA_ROOT.in_file, REQ_DATA_ROOT_FILE_OVER_YAML


@pytest.mark.unit
def test_process_environment_wins_over_config_file_for_champollion_data_root(run_main, work_dir, monkeypatch):
    """--config FILE setting dataset.champollion_data_root, CHAMPOLLION_DATA_ROOT in the environment: it wins."""
    config_file = _write_yaml_config(work_dir / "run.yaml", DATA_ROOT.field_path, DATA_ROOT.in_yaml)
    monkeypatch.setenv(DATA_ROOT.variable, DATA_ROOT.in_process)
    config = _run_config(run_main, "--config", config_file)
    assert config.dataset.champollion_data_root == DATA_ROOT.in_process, REQ_DATA_ROOT_PROCESS_OVER_YAML


@pytest.mark.unit
def test_champollion_data_root_in_process_environment_logs_no_warning(run_main, monkeypatch, caplog, pipeline_caplog):
    """CHAMPOLLION_DATA_ROOT in the process environment: no WARNING record names it.

    The run first has to read the variable, so the check is on a main.py that reads the process environment.
    """
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv(DATA_ROOT.variable, DATA_ROOT.in_process)
    config = _run_config(run_main)
    read = config.dataset.champollion_data_root
    assert read == DATA_ROOT.in_process, f"{REQ_DATA_ROOT_NO_WARNING}: variable not read (field {read!r})"
    warnings = [record for record in _records_naming(caplog, DATA_ROOT.variable) if record.levelno == logging.WARNING]
    assert warnings == [], f"{REQ_DATA_ROOT_NO_WARNING}: {[record.getMessage() for record in warnings]}"


@pytest.mark.unit
def test_champollion_data_root_flag_wins_over_process_environment(run_main, monkeypatch):
    """--champollion-data-root <value>, CHAMPOLLION_DATA_ROOT in the environment with another value: <value> wins.

    A main.py that never reads the variable would let the flag win by default, so the run without the flag first
    checks that the variable is read.
    """
    flag_value = "/flag/data-root"
    monkeypatch.setenv(DATA_ROOT.variable, DATA_ROOT.in_process)
    unflagged = _run_config(run_main).dataset.champollion_data_root
    assert unflagged == DATA_ROOT.in_process, f"{REQ_DATA_ROOT_FLAG_OVER_PROCESS}: variable not read ({unflagged!r})"
    read = _run_config(run_main, "--champollion-data-root", flag_value).dataset.champollion_data_root
    assert read == flag_value, f"{REQ_DATA_ROOT_FLAG_OVER_PROCESS}: field {read!r}, expected {flag_value!r}"
