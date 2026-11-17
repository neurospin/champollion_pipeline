#!/usr/bin/env python3
"""
main.py <input> <output>: the input and output positionals, and --output-path (TASK-INPUTKIND-BDRABCZUK-95028C1E6116).

The first positional sets dataset.input_path, like --input-path (the registry argument of input_path); the second
positional sets dataset.datasets_root, like --output-path (user answer 2026-11-05: the folder holding derivatives/).
Two forms of one value that resolve to different paths (pathlib Path.resolve()) stop main.py with status 2, the
usage-error status; forms resolving to the same path run (user answers 2026-11-16). The second positional argument
and --output-path also stop beside a --datasets-root resolving elsewhere, naming both values. main.py --help lists
INPUT, OUTPUT and --output-path.

The observable is the PipelineConfig main() runs: sys.argv drives the parse, the update check, the process setup and
profiling are stubbed, and PipelineOrchestrator is replaced by a recorder whose run() returns 0, so no stage executes.
With --generate-config the observable is the YAML file main() writes. The input folder holds one T1 file, so the
input kind detection reads it as t1 and changes no field the tests read.

main.py lives at the repo root, so it is loaded by file path under a private module name.
"""

import dataclasses
import logging
from pathlib import Path

import pytest
import yaml

from tests.main_stage_helpers import load_main_module, main_runner, ran_config

MAIN = load_main_module("_champollion_main_input_output_arguments_under_test")

REQ_FIRST_POSITIONAL = "REQ-INPUTKIND-BDRABCZUK-126F13C25628: first positional argument sets input_path"
REQ_SECOND_POSITIONAL = "REQ-INPUTKIND-BDRABCZUK-0937DDB84617: second positional argument sets datasets_root"
REQ_OUTPUT_PATH = "REQ-INPUTKIND-BDRABCZUK-285931731F72: --output-path sets datasets_root"
REQ_FIRST_OVER_YAML = "REQ-INPUTKIND-BDRABCZUK-B98C76A73E02: first positional argument wins over the --config file"
REQ_SECOND_OVER_YAML = "REQ-INPUTKIND-BDRABCZUK-AF67C53D28C8: second positional argument wins over the --config file"
REQ_SECOND_OVER_NAME = "REQ-INPUTKIND-BDRABCZUK-13D31B09EDDC: second positional wins over the --dataset-name root"
REQ_OUTPUT_PATH_OVER_NAME = "REQ-INPUTKIND-BDRABCZUK-9D634F9746A4: --output-path wins over the --dataset-name root"
REQ_EQUAL_INPUT = "REQ-INPUTKIND-BDRABCZUK-B712CDEBFDEC: equal first positional and --input-path run"
REQ_EQUAL_OUTPUT = "REQ-INPUTKIND-BDRABCZUK-0B2E796B3341: equal second positional and --output-path run"
REQ_DIFFERENT_INPUT_STOPS = "REQ-INPUTKIND-BDRABCZUK-E822CED14B5F: first positional and --input-path apart: status 2"
REQ_DIFFERENT_INPUT_ERROR = "REQ-INPUTKIND-BDRABCZUK-F53FF0A4B753: differing input error names --input-path"
REQ_DIFFERENT_OUTPUT_STOPS = "REQ-INPUTKIND-BDRABCZUK-51241CA626FC: second positional, --output-path apart: status 2"
REQ_DIFFERENT_OUTPUT_ERROR = "REQ-INPUTKIND-BDRABCZUK-F27EF561DFB9: differing output error names --output-path"
REQ_SAME_INPUT = "REQ-INPUTKIND-BDRABCZUK-55F3FDB250B4: first positional and --input-path on the same path run"
REQ_SAME_OUTPUT = "REQ-INPUTKIND-BDRABCZUK-813DA8E97A28: second positional and --output-path on the same path run"
REQ_POSITIONAL_ROOT_STOPS = (
    "REQ-INPUTKIND-BDRABCZUK-6B012C3011DB: second positional and --datasets-root apart: status 2"
)
REQ_OPTION_ROOT_STOPS = "REQ-INPUTKIND-BDRABCZUK-441BCC6A7A00: --output-path and --datasets-root apart: status 2"
REQ_POSITIONAL_ROOT_ERROR = "REQ-INPUTKIND-BDRABCZUK-6A7AC5D7C92E: second positional/--datasets-root error names both"
REQ_OPTION_ROOT_ERROR = "REQ-INPUTKIND-BDRABCZUK-180BA5E719F9: --output-path/--datasets-root error names both values"
REQ_POSITIONAL_ROOT_SAME = "REQ-INPUTKIND-BDRABCZUK-440A99F081E9: second positional and --datasets-root same path run"
REQ_OPTION_ROOT_SAME = "REQ-INPUTKIND-BDRABCZUK-D64800FB011A: --output-path and --datasets-root on the same path run"
REQ_HELP_INPUT = "REQ-INPUTKIND-BDRABCZUK-88597346266F: main.py --help lists the INPUT positional argument"
REQ_HELP_OUTPUT = "REQ-INPUTKIND-BDRABCZUK-255766C641D5: main.py --help lists the OUTPUT positional argument"
REQ_HELP_OUTPUT_PATH = "REQ-INPUTKIND-BDRABCZUK-211D164B703B: main.py --help lists the --output-path flag"
REQ_GENERATE_FIRST = "REQ-INPUTKIND-BDRABCZUK-944824E44EEA: --generate-config writes the first positional"
REQ_GENERATE_SECOND = "REQ-INPUTKIND-BDRABCZUK-305B3C9395A0: --generate-config writes the second positional"

INPUT_NAME = "t1-input"
OTHER_INPUT_NAME = "other-t1-input"
INPUT_LINK_NAME = "t1-input-link"
OUTPUT_NAME = "cli-output"
OUTPUT_LINK_NAME = "cli-output-link"
USAGE_ERROR_STATUS = 2

# Other spellings of one folder. Each resolves, with pathlib Path.resolve() in the working folder, to the folder the
# plain name gives: INPUT_NAME for the first list, OUTPUT_NAME for the second.
SAME_INPUT_SPELLINGS = [
    pytest.param(INPUT_NAME + "/", id="trailing-slash"),
    pytest.param("./" + INPUT_NAME, id="dot"),
    pytest.param(f"{OTHER_INPUT_NAME}/../{INPUT_NAME}", id="dot-dot"),
    pytest.param(INPUT_LINK_NAME, id="symlink"),
]
SAME_OUTPUT_SPELLINGS = [
    pytest.param(OUTPUT_NAME + "/", id="trailing-slash"),
    pytest.param("./" + OUTPUT_NAME, id="dot"),
    pytest.param(f"{OTHER_INPUT_NAME}/../{OUTPUT_NAME}", id="dot-dot"),
    pytest.param(OUTPUT_LINK_NAME, id="symlink"),
]


@pytest.fixture
def work_dir(monkeypatch, tmp_path) -> Path:
    """tmp_path as the working folder: two T1 input folders, an output folder and a symlink to each of the two."""
    monkeypatch.chdir(tmp_path)
    for name in (INPUT_NAME, OTHER_INPUT_NAME):
        (tmp_path / name).mkdir()
        (tmp_path / name / "sub-01.nii.gz").write_bytes(b"synthetic")
    (tmp_path / OUTPUT_NAME).mkdir()
    (tmp_path / INPUT_LINK_NAME).symlink_to(tmp_path / INPUT_NAME, target_is_directory=True)
    (tmp_path / OUTPUT_LINK_NAME).symlink_to(tmp_path / OUTPUT_NAME, target_is_directory=True)
    return tmp_path


@pytest.fixture
def run_main(monkeypatch, work_dir):
    """Run main() with the given arguments; return its exit status and the configurations it ran."""
    return main_runner(MAIN, monkeypatch)


def _run_config(run_main, *cli_args: str):
    return ran_config(run_main(*cli_args), cli_args)


def _write_yaml_config(path: Path, input_path: str, datasets_root: str) -> str:
    """Write the default configuration with input_path and datasets_root set as a --config YAML file."""
    config_dict = dataclasses.asdict(MAIN.create_default_config())
    config_dict["dataset"]["input_path"] = input_path
    config_dict["dataset"]["datasets_root"] = datasets_root
    path.write_text(yaml.safe_dump(config_dict))
    return str(path)


def _error_messages(pipeline_caplog) -> list[str]:
    return [record.getMessage() for record in pipeline_caplog.records if record.levelno >= logging.ERROR]


# ====================== Each form sets its field ======================


@pytest.mark.unit
def test_first_positional_sets_input_path(run_main):
    """When main() runs with a first positional argument, without --input-path, input_path holds that argument."""
    config = _run_config(run_main, INPUT_NAME)
    assert config.dataset.input_path == INPUT_NAME, REQ_FIRST_POSITIONAL


@pytest.mark.unit
def test_second_positional_sets_datasets_root(run_main):
    """When main() runs with a second positional argument, datasets_root holds that argument."""
    config = _run_config(run_main, INPUT_NAME, "/cli/output")
    assert config.dataset.datasets_root == "/cli/output", REQ_SECOND_POSITIONAL


@pytest.mark.unit
def test_output_path_option_sets_datasets_root(run_main):
    """When main() runs with --output-path, without --datasets-root, datasets_root holds the --output-path value."""
    config = _run_config(run_main, "--input-path", INPUT_NAME, "--output-path", "/cli/output")
    assert config.dataset.datasets_root == "/cli/output", REQ_OUTPUT_PATH


# ====================== Precedence over the --config file and the default root ======================


@pytest.mark.unit
def test_first_positional_wins_over_config_file(run_main, work_dir):
    """With --config plus a first positional argument, input_path holds the positional argument."""
    config_file = _write_yaml_config(work_dir / "run.yaml", OTHER_INPUT_NAME, "/yaml/root")
    config = _run_config(run_main, "--config", config_file, INPUT_NAME)
    assert config.dataset.input_path == INPUT_NAME, REQ_FIRST_OVER_YAML


@pytest.mark.unit
def test_second_positional_wins_over_config_file(run_main, work_dir):
    """With --config plus a second positional argument, datasets_root holds the positional argument."""
    config_file = _write_yaml_config(work_dir / "run.yaml", INPUT_NAME, "/yaml/root")
    config = _run_config(run_main, "--config", config_file, INPUT_NAME, "/cli/output")
    assert config.dataset.datasets_root == "/cli/output", REQ_SECOND_OVER_YAML


@pytest.mark.unit
def test_second_positional_wins_over_dataset_name_default_root(run_main):
    """With --dataset-name plus a second positional argument, datasets_root holds the positional argument."""
    config = _run_config(run_main, "--dataset-name", "foo", INPUT_NAME, "/cli/output")
    assert config.dataset.datasets_root == "/cli/output", REQ_SECOND_OVER_NAME


@pytest.mark.unit
def test_output_path_option_wins_over_dataset_name_default_root(run_main):
    """With --dataset-name plus --output-path, datasets_root holds the --output-path value."""
    config = _run_config(run_main, "--dataset-name", "foo", "--input-path", INPUT_NAME, "--output-path", "/cli/output")
    assert config.dataset.datasets_root == "/cli/output", REQ_OUTPUT_PATH_OVER_NAME


# ====================== Both forms of one value ======================


@pytest.mark.unit
def test_equal_first_positional_and_input_path_run(run_main):
    """A first positional argument equal to --input-path runs with that value in input_path."""
    config = _run_config(run_main, INPUT_NAME, "--input-path", INPUT_NAME)
    assert config.dataset.input_path == INPUT_NAME, REQ_EQUAL_INPUT


@pytest.mark.unit
def test_equal_second_positional_and_output_path_run(run_main):
    """A second positional argument equal to --output-path runs with that value in datasets_root."""
    config = _run_config(run_main, INPUT_NAME, "/cli/output", "--output-path", "/cli/output")
    assert config.dataset.datasets_root == "/cli/output", REQ_EQUAL_OUTPUT


@pytest.mark.unit
def test_differing_first_positional_and_input_path_stop(run_main):
    """A first positional argument resolving apart from --input-path exits with status 2 before any stage runs.

    The same command line with equal values must run, so the stop comes from the difference, not from a parser
    that refuses the positional argument itself.
    """
    _run_config(run_main, INPUT_NAME, "--input-path", INPUT_NAME)
    outcome = run_main(INPUT_NAME, "--input-path", OTHER_INPUT_NAME)
    assert outcome.status == USAGE_ERROR_STATUS, f"main() exit status {outcome.status}; {REQ_DIFFERENT_INPUT_STOPS}"
    assert not outcome.configs, f"the pipeline ran despite the difference; {REQ_DIFFERENT_INPUT_STOPS}"


@pytest.mark.unit
def test_differing_input_error_names_input_path(run_main, pipeline_caplog):
    """A first positional argument differing from --input-path logs an ERROR record containing --input-path."""
    run_main(INPUT_NAME, "--input-path", OTHER_INPUT_NAME)
    messages = _error_messages(pipeline_caplog)
    assert any("--input-path" in message for message in messages), (
        f"no ERROR record containing --input-path; ERROR records: {messages}; {REQ_DIFFERENT_INPUT_ERROR}"
    )


@pytest.mark.unit
def test_differing_second_positional_and_output_path_stop(run_main):
    """A second positional argument resolving apart from --output-path exits with status 2 before any stage runs.

    The same command line with equal values must run, so the stop comes from the difference, not from a parser
    that refuses the positional argument or the option itself.
    """
    _run_config(run_main, INPUT_NAME, "/cli/output", "--output-path", "/cli/output")
    outcome = run_main(INPUT_NAME, "/cli/output", "--output-path", "/cli/other")
    assert outcome.status == USAGE_ERROR_STATUS, f"main() exit status {outcome.status}; {REQ_DIFFERENT_OUTPUT_STOPS}"
    assert not outcome.configs, f"the pipeline ran despite the difference; {REQ_DIFFERENT_OUTPUT_STOPS}"


@pytest.mark.unit
def test_differing_output_error_names_output_path(run_main, pipeline_caplog):
    """A second positional argument differing from --output-path logs an ERROR record containing --output-path."""
    run_main(INPUT_NAME, "/cli/output", "--output-path", "/cli/other")
    messages = _error_messages(pipeline_caplog)
    assert any("--output-path" in message for message in messages), (
        f"no ERROR record containing --output-path; ERROR records: {messages}; {REQ_DIFFERENT_OUTPUT_ERROR}"
    )


# ====================== Two forms naming the same folder ======================


def _resolved(value: str) -> Path:
    return Path(value).resolve()


@pytest.mark.unit
@pytest.mark.parametrize("spelling", SAME_INPUT_SPELLINGS)
def test_first_positional_and_input_path_on_same_path_run(run_main, spelling):
    """A first positional argument and an --input-path spelled apart but resolving to one folder run with it."""
    config = _run_config(run_main, INPUT_NAME, "--input-path", spelling)
    held = config.dataset.input_path
    assert _resolved(held) == _resolved(INPUT_NAME), f"{spelling!r}: input_path {held!r}; {REQ_SAME_INPUT}"


@pytest.mark.unit
@pytest.mark.parametrize("spelling", SAME_OUTPUT_SPELLINGS)
def test_second_positional_and_output_path_on_same_path_run(run_main, spelling):
    """A second positional argument and an --output-path spelled apart but resolving to one folder run with it."""
    config = _run_config(run_main, INPUT_NAME, OUTPUT_NAME, "--output-path", spelling)
    held = config.dataset.datasets_root
    assert _resolved(held) == _resolved(OUTPUT_NAME), f"{spelling!r}: datasets_root {held!r}; {REQ_SAME_OUTPUT}"


# ====================== The output forms beside --datasets-root ======================


@pytest.mark.unit
def test_second_positional_and_datasets_root_on_different_paths_stop(run_main):
    """A second positional argument resolving apart from --datasets-root exits with status 2 before any stage runs.

    The same command line with equal values must run, so the stop comes from the difference, not from a parser
    that refuses the positional argument itself.
    """
    _run_config(run_main, INPUT_NAME, "/cli/output", "--datasets-root", "/cli/output")
    outcome = run_main(INPUT_NAME, "/cli/output", "--datasets-root", "/cli/other")
    assert outcome.status == USAGE_ERROR_STATUS, f"main() exit status {outcome.status}; {REQ_POSITIONAL_ROOT_STOPS}"
    assert not outcome.configs, f"the pipeline ran despite the difference; {REQ_POSITIONAL_ROOT_STOPS}"


@pytest.mark.unit
def test_output_path_and_datasets_root_on_different_paths_stop(run_main):
    """An --output-path resolving apart from --datasets-root exits with status 2 before any stage runs.

    The same command line with equal values must run, so the stop comes from the difference, not from a parser
    that refuses --output-path itself.
    """
    _run_config(run_main, "--input-path", INPUT_NAME, "--output-path", "/cli/output", "--datasets-root", "/cli/output")
    outcome = run_main("--input-path", INPUT_NAME, "--output-path", "/cli/output", "--datasets-root", "/cli/other")
    assert outcome.status == USAGE_ERROR_STATUS, f"main() exit status {outcome.status}; {REQ_OPTION_ROOT_STOPS}"
    assert not outcome.configs, f"the pipeline ran despite the difference; {REQ_OPTION_ROOT_STOPS}"


def _assert_one_error_names(pipeline_caplog, values: tuple[str, ...], requirement: str) -> None:
    messages = _error_messages(pipeline_caplog)
    assert any(all(value in message for value in values) for message in messages), (
        f"no ERROR record containing {values}; ERROR records: {messages}; {requirement}"
    )


@pytest.mark.unit
def test_second_positional_and_datasets_root_error_names_both_values(run_main, pipeline_caplog):
    """A second positional argument resolving apart from --datasets-root logs an ERROR record naming both values."""
    run_main(INPUT_NAME, "/cli/output", "--datasets-root", "/cli/other")
    _assert_one_error_names(pipeline_caplog, ("/cli/output", "/cli/other"), REQ_POSITIONAL_ROOT_ERROR)


@pytest.mark.unit
def test_output_path_and_datasets_root_error_names_both_values(run_main, pipeline_caplog):
    """An --output-path resolving apart from --datasets-root logs an ERROR record naming both values."""
    run_main("--input-path", INPUT_NAME, "--output-path", "/cli/output", "--datasets-root", "/cli/other")
    _assert_one_error_names(pipeline_caplog, ("/cli/output", "/cli/other"), REQ_OPTION_ROOT_ERROR)


@pytest.mark.unit
@pytest.mark.parametrize("spelling", SAME_OUTPUT_SPELLINGS)
def test_second_positional_and_datasets_root_on_same_path_run(run_main, spelling):
    """A second positional argument and a --datasets-root resolving to one folder run with it in datasets_root."""
    config = _run_config(run_main, INPUT_NAME, OUTPUT_NAME, "--datasets-root", spelling)
    held = config.dataset.datasets_root
    assert _resolved(held) == _resolved(OUTPUT_NAME), (
        f"{spelling!r}: datasets_root {held!r}; {REQ_POSITIONAL_ROOT_SAME}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("spelling", SAME_OUTPUT_SPELLINGS)
def test_output_path_and_datasets_root_on_same_path_run(run_main, spelling):
    """An --output-path and a --datasets-root resolving to one folder run with it in datasets_root."""
    config = _run_config(
        run_main, "--input-path", INPUT_NAME, "--output-path", OUTPUT_NAME, "--datasets-root", spelling
    )
    held = config.dataset.datasets_root
    assert _resolved(held) == _resolved(OUTPUT_NAME), f"{spelling!r}: datasets_root {held!r}; {REQ_OPTION_ROOT_SAME}"


# ====================== --generate-config ======================


def _generated(run_main, work_dir: Path, *cli_args: str) -> dict:
    """Run main() with --generate-config <work_dir>/out.yaml plus cli_args; return the written YAML."""
    output = work_dir / "out.yaml"
    outcome = run_main("--generate-config", str(output), *cli_args)
    assert outcome.status == 0, f"main() exit status {outcome.status} for --generate-config with {list(cli_args)}"
    return yaml.safe_load(output.read_text())


@pytest.mark.unit
def test_generate_config_writes_first_positional(run_main, work_dir):
    """--generate-config plus a first positional argument writes that argument in dataset.input_path."""
    written = _generated(run_main, work_dir, INPUT_NAME)
    assert written["dataset"]["input_path"] == INPUT_NAME, REQ_GENERATE_FIRST


@pytest.mark.unit
def test_generate_config_writes_second_positional(run_main, work_dir):
    """--generate-config plus a second positional argument writes that argument in dataset.datasets_root."""
    written = _generated(run_main, work_dir, INPUT_NAME, "/cli/output")
    assert written["dataset"]["datasets_root"] == "/cli/output", REQ_GENERATE_SECOND


# ====================== --help ======================


def _help_section(run_main, monkeypatch, capsys, heading: str) -> list[str]:
    """The lines of one section of main.py --help (COLUMNS=100), heading excluded; [] when there is no such section."""
    monkeypatch.setenv("COLUMNS", "100")
    capsys.readouterr()
    outcome = run_main("--help")
    assert outcome.status == 0, f"main.py --help exit status {outcome.status}"
    text = capsys.readouterr().out
    if f"\n{heading}\n" not in text:
        return []
    return text.split(f"\n{heading}\n", 1)[1].split("\n\n", 1)[0].splitlines()


def _entries_named(lines: list[str], name: str) -> list[str]:
    return [line for line in lines if line.startswith("  ") and line.split()[0].rstrip(",") == name]


@pytest.mark.unit
def test_help_lists_input_positional(run_main, monkeypatch, capsys):
    """The positional arguments section of main.py --help lists INPUT."""
    section = _help_section(run_main, monkeypatch, capsys, "positional arguments:")
    assert _entries_named(section, "INPUT"), f"no INPUT entry in: {section}; {REQ_HELP_INPUT}"


@pytest.mark.unit
def test_help_lists_output_positional(run_main, monkeypatch, capsys):
    """The positional arguments section of main.py --help lists OUTPUT."""
    section = _help_section(run_main, monkeypatch, capsys, "positional arguments:")
    assert _entries_named(section, "OUTPUT"), f"no OUTPUT entry in: {section}; {REQ_HELP_OUTPUT}"


@pytest.mark.unit
def test_help_lists_output_path_option(run_main, monkeypatch, capsys):
    """The options section of main.py --help lists --output-path."""
    section = _help_section(run_main, monkeypatch, capsys, "options:")
    assert _entries_named(section, "--output-path"), f"no --output-path entry in: {section}; {REQ_HELP_OUTPUT_PATH}"
