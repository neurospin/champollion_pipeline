"""The DatasetConfig field input_kind and --input-kind answering the kind question in advance: default, YAML round
trip, no prompt, stage routing, and the stops on a missing kind, an unknown kind, an empty input_path or no
subjects folder.

Split from tests/test_main_input_autodetect_ambiguous.py (TASK-INPUTKIND-BDRABCZUK-830F5595E185);
behaviour TASK-INPUTKIND-BDRABCZUK-5ECD06E2023D.
Requirements: one per test below (alm links on the Task root).
Harness: tests/main_input_autodetect_harness.py.
"""

import dataclasses
from pathlib import Path

import pytest
import yaml

from champollion_pipeline.config import ConfigLoader, DatasetConfig, PipelineConfig
from champollion_pipeline.utils.input_kind import scan_input
from tests.main_input_autodetect_harness import (
    DATASETS,
    GENERATED_CONFIG,
    INPUT_KIND_OPTION,
    KINDS,
    MORPHOLOGIST_6,
    NO_SUBJECTS_FOLDER,
    STAGE1,
    STAGE2,
    UNKNOWN_KIND,
    _assert_error_contains,
    _assert_stopped,
    _batch,
    _config,
    _error_messages,
    _exit_status,
    _main_module_under_test,
    _run_stages,
    _single_subject_dataset,
    _stop_config,
    _t1_dataset,
    _terminal,
)


@pytest.fixture
def main_module(monkeypatch, tmp_path):
    return _main_module_under_test(monkeypatch, tmp_path)


# ====================== The input_kind field ======================


@pytest.mark.unit
def test_dataset_config_input_kind_defaults_to_none():
    assert DatasetConfig().input_kind is None


@pytest.mark.unit
@pytest.mark.parametrize("kind", KINDS)
def test_input_kind_round_trips_through_yaml(tmp_path, kind):
    path = tmp_path / "replay.yaml"
    ConfigLoader.save_to_yaml(PipelineConfig(dataset=DatasetConfig(input_kind=kind)), str(path))
    assert ConfigLoader.load_from_yaml(str(path)).dataset.input_kind == kind


# ====================== input_kind answers in advance ======================


@pytest.mark.unit
@pytest.mark.parametrize(("dataset", "kind"), [("both", "t1"), ("both", "morphologist"), ("sourcedata", "t1")])
def test_set_input_kind_asks_nothing(main_module, monkeypatch, tmp_path, dataset, kind):
    terminal = _terminal(monkeypatch, "t1")
    config = _config(main_module, tmp_path, DATASETS[dataset](tmp_path))
    status, argv = _run_stages(main_module, monkeypatch, config, INPUT_KIND_OPTION, kind)
    # REQ-INPUTKIND-BDRABCZUK-4C4FDD23A14B fixes no exit status (stage 2 may stop later on its own checks); a stage
    # call proves main() went past the input kind choice, so the read count below is not taken before it
    assert argv, f"no stage ran with input_kind {kind} (exit status {status})"
    assert terminal.stdin.reads == 0, f"standard input was read {terminal.stdin.reads} times with input_kind {kind}"


@pytest.mark.unit
@pytest.mark.parametrize("dataset", ["both", "sourcedata", "t1"])
def test_input_kind_t1_runs_stage1_on_input(main_module, monkeypatch, tmp_path, dataset):
    _batch(monkeypatch)
    root = DATASETS[dataset](tmp_path)
    status, argv = _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root), INPUT_KIND_OPTION, "t1")
    assert STAGE1 in argv, f"stage 1 did not run with input_kind t1 (exit status {status})"
    assert argv[STAGE1][1] == str(root), f"stage 1 output root {argv[STAGE1][1]!r}"


@pytest.mark.unit
@pytest.mark.parametrize("dataset", ["both", "morphologist"])
def test_input_kind_morphologist_stage2_reads_resolved_folder(main_module, monkeypatch, tmp_path, dataset):
    """input_kind comes from a --config YAML, as from a replay config."""
    _batch(monkeypatch)
    root = DATASETS[dataset](tmp_path)
    content = dataclasses.asdict(_config(main_module, tmp_path, root))
    content["dataset"]["input_kind"] = "morphologist"
    replay = tmp_path / "replay.yaml"
    replay.write_text(yaml.safe_dump(content))
    status, argv = _run_stages(main_module, monkeypatch, None, "--config", str(replay))
    assert STAGE2 in argv, f"stage 2 did not run with input_kind morphologist (exit status {status})"
    assert argv[STAGE2][0] == str(root.joinpath(*MORPHOLOGIST_6)), f"stage 2 subjects folder {argv[STAGE2][0]!r}"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("dataset", "kind"),
    [("morphologist", "t1"), ("t1", "morphologist"), ("neither", "t1"), ("neither", "morphologist")],
)
def test_missing_chosen_kind_exits_1(main_module, monkeypatch, tmp_path, dataset, kind):
    _batch(monkeypatch)
    config = _stop_config(main_module, tmp_path, DATASETS[dataset](tmp_path))
    _assert_stopped(*_run_stages(main_module, monkeypatch, config, INPUT_KIND_OPTION, kind))


@pytest.mark.unit
@pytest.mark.parametrize("case", [("morphologist", "t1"), ("t1", "morphologist")], ids=["kind_t1", "kind_morphologist"])
def test_missing_chosen_kind_error_names_other_kind_example(main_module, monkeypatch, tmp_path, pipeline_caplog, case):
    dataset, kind = case
    _batch(monkeypatch)
    root = DATASETS[dataset](tmp_path)
    scan = scan_input(str(root))
    example = scan.first_graph if kind == "t1" else scan.first_t1
    _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root), INPUT_KIND_OPTION, kind)
    _assert_error_contains(pipeline_caplog, example)


def _run_unknown_kind(main_module, monkeypatch, config, route: str):
    """Run main() with --input-kind UNKNOWN_KIND, through the pipeline run or --generate-config (in the work dir)."""
    if route == "generate_config":
        args = ("--generate-config", str(Path.cwd() / GENERATED_CONFIG), INPUT_KIND_OPTION, UNKNOWN_KIND)
        return _exit_status(main_module, monkeypatch, config, *args), {}
    return _run_stages(main_module, monkeypatch, config, INPUT_KIND_OPTION, UNKNOWN_KIND)


UNKNOWN_KIND_ROUTES = pytest.mark.parametrize("route", ["run", "generate_config"])


@pytest.mark.unit
@UNKNOWN_KIND_ROUTES
def test_unknown_input_kind_exits_1(main_module, monkeypatch, tmp_path, route):
    _batch(monkeypatch)
    config = _stop_config(main_module, tmp_path, _t1_dataset(tmp_path))
    _assert_stopped(*_run_unknown_kind(main_module, monkeypatch, config, route))


@pytest.mark.unit
@UNKNOWN_KIND_ROUTES
def test_unknown_input_kind_error_names_the_two_kinds(main_module, monkeypatch, tmp_path, pipeline_caplog, route):
    _batch(monkeypatch)
    raw = _t1_dataset(tmp_path)
    _run_unknown_kind(main_module, monkeypatch, _config(main_module, tmp_path, raw), route)
    messages = [message.replace(str(raw), "") for message in _error_messages(pipeline_caplog)]
    assert any(all(kind in message for kind in KINDS) for message in messages), (
        f"no ERROR record naming {KINDS} for input_kind {UNKNOWN_KIND!r}; ERROR records: {messages}"
    )


@pytest.mark.unit
def test_generate_config_unknown_input_kind_writes_no_file(main_module, monkeypatch, tmp_path):
    """dataset.input_kind is set on the default configuration, as --input-kind would set it."""
    _batch(monkeypatch)
    config = _config(main_module, tmp_path, _t1_dataset(tmp_path))
    config.dataset.input_kind = UNKNOWN_KIND  # noqa: V101
    target = Path.cwd() / GENERATED_CONFIG
    status = _exit_status(main_module, monkeypatch, config, "--generate-config", str(target))
    assert not target.exists(), (
        f"--generate-config wrote {target} for input_kind {UNKNOWN_KIND!r} (exit status {status})"
    )


@pytest.mark.unit
@pytest.mark.parametrize("kind", KINDS)
def test_input_kind_with_empty_input_path_exits_1(main_module, monkeypatch, tmp_path, kind):
    _batch(monkeypatch)
    config = _stop_config(main_module, tmp_path, "")
    _assert_stopped(*_run_stages(main_module, monkeypatch, config, INPUT_KIND_OPTION, kind))


@pytest.mark.unit
@pytest.mark.parametrize("kind", KINDS)
def test_input_kind_with_empty_input_path_error_names_option(main_module, monkeypatch, tmp_path, pipeline_caplog, kind):
    _batch(monkeypatch)
    _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, ""), INPUT_KIND_OPTION, kind)
    _assert_error_contains(pipeline_caplog, "--input-path")


SINGLE_SUBJECT_CASES = pytest.mark.parametrize("t1_first", [True, False], ids=["t1_sorts_first", "graphs_only"])


@pytest.mark.unit
@SINGLE_SUBJECT_CASES
def test_input_kind_morphologist_without_subjects_folder_exits_1(main_module, monkeypatch, tmp_path, t1_first):
    _batch(monkeypatch)
    config = _stop_config(main_module, tmp_path, _single_subject_dataset(tmp_path, t1_first))
    _assert_stopped(*_run_stages(main_module, monkeypatch, config, INPUT_KIND_OPTION, "morphologist"))


@pytest.mark.unit
@SINGLE_SUBJECT_CASES
def test_input_kind_morphologist_without_subjects_folder_error_names_input(
    main_module, monkeypatch, tmp_path, pipeline_caplog, t1_first
):
    _batch(monkeypatch)
    subject = _single_subject_dataset(tmp_path, t1_first)
    _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, subject), INPUT_KIND_OPTION, "morphologist")
    _assert_error_contains(pipeline_caplog, f"{NO_SUBJECTS_FOLDER} {subject}")
