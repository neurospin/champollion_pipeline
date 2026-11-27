#!/usr/bin/env python3
"""
main.py options --session, --run and --analysis (TASK-ARGDISC-BDRABCZUK-91F0268EFCDC, ARGDISC epic).

User decisions 2026-11-27: the three options narrow graph selection through the select_graphs filters, each an
anchored-start regular expression on the ses-, run- or ana- folder name with the prefix kept (ses-2, run-0, ana-1).
They are main.py arguments and DatasetConfig fields (CLI first; they reach the replay file). An invalid regular
expression stops main.py with exit status 2. With no option the filters are empty, so every sub/ses/run/ana folder
stays its own scan. This Task only parses, checks and carries the values; the pre-check and stage 2 Tasks read them.

Requirements (area ARGDISC, author BDRABCZUK), test per requirement:
* A400E8626495 user rule, a run narrowed by session, run or analysis - test_session_option_narrows_select_graphs
* 75488D459BFE DatasetConfig fields, empty by default - test_dataset_config_declares_the_filter_field
* DBC51341E09C the options set the fields - test_option_sets_the_dataset_field,
  test_absent_options_keep_the_empty_fields
* D000970F41B0 graph_selection_filters - test_selection_filters_hold_the_given_values,
  test_selection_filters_are_empty_without_values, test_session_option_narrows_select_graphs
* 96D06C92A2B3 invalid regex, exit status 2 - test_invalid_regex_option_stops_with_status_2,
  test_invalid_regex_in_config_file_stops_with_status_2
* E58D9CD49391 the stop names the option - test_invalid_regex_error_names_the_option
* 401CE6F3B346 replay file keeps the values - test_replay_file_holds_the_filter_values
* 71CD98115847 --help describes the options - test_help_describes_the_option

Inputs: synthetic trees from tests/morphologist_layout_fixtures.py and YAML files written under tmp_path; no real
data. main() runs with the orchestrator replaced by a recorder (tests/main_stage_helpers.main_runner), so no stage
executes. Expected values are written out here from the user decisions, independently of the code under test.
"""

import argparse
import dataclasses
import logging
import typing
from pathlib import Path

import pytest
import yaml

from champollion_pipeline.cli_options import add_config_arguments
from champollion_pipeline.config import DatasetConfig, graph_selection_filters
from champollion_pipeline.file_indexer.pipeline_checks import FileIndexStageIndexer
from tests.main_stage_helpers import enter_clean_work_dir, load_main_module, main_runner, ran_config
from tests.morphologist_layout_fixtures import CLI_SUBJECTS, HEMISPHERES, _create, build_morphologist_cli_tree

MAIN = load_main_module("_champollion_main_select_graphs_options_under_test")

# Field name -> its main.py option and the folder prefix its regular expression applies to.
FILTER_OPTIONS = {"session": "--session", "run": "--run", "analysis": "--analysis"}
FOLDER_PREFIXES = {"session": "ses-", "run": "run-", "analysis": "ana-"}
# Values as the user writes them: the folder prefix kept.
GIVEN_VALUES = {"session": "ses-2", "run": "run-0", "analysis": "ana-1"}
INVALID_REGEX = "ses-("
USAGE_ERROR_STATUS = 2
REPLAY_GLOB = "replay_*.yaml"


@pytest.fixture
def work_dir(monkeypatch, tmp_path) -> Path:
    """A private working folder; HF_TOKEN and the CHAMPOLLION_* variables of the process environment removed."""
    return enter_clean_work_dir(monkeypatch, tmp_path)


@pytest.fixture
def run_main(monkeypatch, work_dir):
    return main_runner(MAIN, monkeypatch)


def _base_args(tmp_path: Path) -> tuple[str, ...]:
    """A run main() accepts without any input: outputs and logs under tmp_path."""
    return ("--outputs-path", str(tmp_path / "out"), "--log-dir", str(tmp_path / "logs"))


def _registry_options() -> dict[str, argparse.Action]:
    """Option string -> action of the config arguments main.py adds."""
    parser = argparse.ArgumentParser(add_help=False)
    add_config_arguments(parser)
    return {flag: action for action in parser._actions for flag in action.option_strings}


def _assert_option_declared(option: str) -> None:
    """Fail when main.py has no such option, so that a status 2 below is never argparse's own refusal."""
    assert option in _registry_options(), f"main.py declares no {option} option"


def _filter_values(dataset: DatasetConfig) -> dict[str, object]:
    return {name: getattr(dataset, name, "<no attribute>") for name in FILTER_OPTIONS}


# ---- DatasetConfig fields ----


@pytest.mark.unit
@pytest.mark.parametrize("name", list(FILTER_OPTIONS))
def test_dataset_config_declares_the_filter_field(name: str) -> None:
    fields = {config_field.name: config_field for config_field in dataclasses.fields(DatasetConfig)}
    assert name in fields, f"DatasetConfig declares no {name} field"
    assert typing.get_type_hints(DatasetConfig)[name] is str, f"DatasetConfig.{name} is not of type str"
    assert fields[name].default == "", f"DatasetConfig.{name} defaults to {fields[name].default!r}, expected ''"


# ---- main.py options ----


@pytest.mark.unit
@pytest.mark.parametrize("name", list(FILTER_OPTIONS))
def test_option_sets_the_dataset_field(run_main, tmp_path, name: str) -> None:
    option, value = FILTER_OPTIONS[name], GIVEN_VALUES[name]
    _assert_option_declared(option)
    cli_args = (*_base_args(tmp_path), option, value)

    config = ran_config(run_main(*cli_args), cli_args)

    assert getattr(config.dataset, name, "<no attribute>") == value, f"{option} {value} left dataset.{name} unset"


@pytest.mark.unit
def test_absent_options_keep_the_empty_fields(run_main, tmp_path) -> None:
    cli_args = _base_args(tmp_path)

    config = ran_config(run_main(*cli_args), cli_args)

    assert _filter_values(config.dataset) == dict.fromkeys(FILTER_OPTIONS, "")


# ---- graph selection filters ----


@pytest.mark.unit
@pytest.mark.parametrize(
    "given",
    [
        {"session": "ses-2"},
        {"run": "run-0"},
        {"analysis": "ana-1"},
        {"session": "ses-(1|2)", "analysis": "ana-1"},
        GIVEN_VALUES,
    ],
    ids=["session", "run", "analysis", "session_and_analysis", "all_three"],
)
def test_selection_filters_hold_the_given_values(given: dict[str, str]) -> None:
    dataset = DatasetConfig()
    for name, value in given.items():
        setattr(dataset, name, value)

    assert graph_selection_filters(dataset) == given


@pytest.mark.unit
def test_selection_filters_are_empty_without_values() -> None:
    assert graph_selection_filters(DatasetConfig()) == {}


def _scan_graphs(subject: str, session: str, run: str, analysis: str) -> list[str]:
    """Unlabelled graph paths of one scan, as morphologist-cli names them."""
    folds = f"{subject}/{session}/{run}/{analysis}/folds/3.1"
    return [f"{folds}/{subject}_{session}_{run}_hemi-{hemi}.arg" for hemi in HEMISPHERES]


@pytest.mark.unit
def test_session_option_narrows_select_graphs(run_main, tmp_path) -> None:
    """--session ses-2, through the selection filters, keeps the ses-2 graphs and leaves the ses-1 graphs out."""
    tree = build_morphologist_cli_tree(tmp_path / "in")
    _create(tree, _scan_graphs(CLI_SUBJECTS[0], "ses-2", "run-0", "ana-0"))
    _assert_option_declared("--session")
    cli_args = (*_base_args(tmp_path), "--session", "ses-2")
    config = ran_config(run_main(*cli_args), cli_args)

    filters = graph_selection_filters(config.dataset)
    items = FileIndexStageIndexer(str(tmp_path / "indexes")).select_graphs(str(tree), filters)

    assert sorted(item.path for item in items) == sorted(_scan_graphs(CLI_SUBJECTS[0], "ses-2", "run-0", "ana-0"))


# ---- invalid regular expression ----


@pytest.mark.unit
@pytest.mark.parametrize("name", list(FILTER_OPTIONS))
def test_invalid_regex_option_stops_with_status_2(run_main, tmp_path, name: str) -> None:
    option = FILTER_OPTIONS[name]
    _assert_option_declared(option)

    outcome = run_main(*_base_args(tmp_path), option, INVALID_REGEX)

    assert outcome.status == USAGE_ERROR_STATUS, f"{option} {INVALID_REGEX!r}: exit status {outcome.status}"
    assert outcome.configs == [], f"{option} {INVALID_REGEX!r}: a stage run started"


@pytest.mark.unit
@pytest.mark.parametrize("name", list(FILTER_OPTIONS))
def test_invalid_regex_in_config_file_stops_with_status_2(run_main, tmp_path, name: str) -> None:
    assert name in {config_field.name for config_field in dataclasses.fields(DatasetConfig)}, (
        f"DatasetConfig declares no {name} field"
    )
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "outputs_path": str(tmp_path / "out"),
                "log_dir": str(tmp_path / "logs"),
                "dataset": {name: INVALID_REGEX},
            }
        )
    )

    outcome = run_main("--config", str(config_file))

    assert outcome.status == USAGE_ERROR_STATUS, f"dataset.{name} {INVALID_REGEX!r}: exit status {outcome.status}"
    assert outcome.configs == [], f"dataset.{name} {INVALID_REGEX!r}: a stage run started"


@pytest.mark.unit
@pytest.mark.parametrize("name", list(FILTER_OPTIONS))
def test_invalid_regex_error_names_the_option(run_main, tmp_path, caplog, name: str) -> None:
    option = FILTER_OPTIONS[name]
    _assert_option_declared(option)
    caplog.set_level(logging.ERROR)

    run_main(*_base_args(tmp_path), option, INVALID_REGEX)

    errors = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
    assert any(option in message for message in errors), f"no ERROR record names {option}: {errors!r}"


# ---- replay file ----


@pytest.mark.unit
def test_replay_file_holds_the_filter_values(run_main, tmp_path) -> None:
    for option in FILTER_OPTIONS.values():
        _assert_option_declared(option)
    given = (arg for name, option in FILTER_OPTIONS.items() for arg in (option, GIVEN_VALUES[name]))
    cli_args = (*_base_args(tmp_path), *given)
    ran_config(run_main(*cli_args), cli_args)
    replays = sorted((tmp_path / "logs").glob(REPLAY_GLOB))
    assert len(replays) == 1, f"expected one replay file in {tmp_path / 'logs'}, found {replays}"

    dataset = yaml.safe_load(replays[0].read_text())["dataset"]

    assert {name: dataset.get(name, "<no key>") for name in FILTER_OPTIONS} == GIVEN_VALUES


# ---- --help ----


@pytest.mark.unit
@pytest.mark.parametrize("name", list(FILTER_OPTIONS))
def test_help_describes_the_option(name: str) -> None:
    option = FILTER_OPTIONS[name]
    _assert_option_declared(option)

    text = " ".join(str(_registry_options()[option].help).split())

    assert "regular expression" in text, f"{option} help {text!r} does not say regular expression"
    assert "start" in text, f"{option} help {text!r} does not say the match is at the start"
    assert FOLDER_PREFIXES[name] in text, f"{option} help {text!r} does not name the {FOLDER_PREFIXES[name]} folder"
