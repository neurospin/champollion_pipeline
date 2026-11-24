#!/usr/bin/env python3
"""
main.py takes every config field argument of the registry, with the precedence command line over --config YAML
over default (TASK-INPUTKIND-BDRABCZUK-6642B10A5597).

The cases come from CONFIG_ARGUMENTS of champollion_pipeline.cli_options, the registry that owns the field set and
the value rules, so a new config field is tested here without an edit. The observable is the PipelineConfig main()
runs: sys.argv drives the parse, the update check, the process setup and profiling are stubbed, and
PipelineOrchestrator is replaced by a recorder whose run() returns 0, so no stage executes. A run whose mode is
streaming goes to the streaming runner instead, which is replaced by a recorder too: for the mode field, reaching it
is what "mode is streaming" means. With --generate-config the observable is the YAML file main() writes.

main.py lives at the repo root, so it is loaded by file path under a private module name, as in
test_main_dataset_name_override.py.
"""

import dataclasses
from pathlib import Path

import pytest
import yaml

from champollion_pipeline.cli_options import CONFIG_ARGUMENTS, EXEMPT_FIELDS, ArgumentKind, ConfigArgument
from tests.main_stage_helpers import load_main_module

MAIN = load_main_module("_champollion_main_cli_overrides_all_fields_under_test")

REQ_OVER_DEFAULT = "REQ-INPUTKIND-BDRABCZUK-2D667852F456: config field argument wins over the default config"
REQ_OVER_YAML = "REQ-INPUTKIND-BDRABCZUK-2D6361D75AF2: config field argument wins over the --config file"
REQ_YAML_OVER_DEFAULT = "REQ-INPUTKIND-BDRABCZUK-5857EDED9E61: --config file value wins over the default config"
REQ_DEFAULT = "REQ-INPUTKIND-BDRABCZUK-F0D02CD54CBD: no argument runs the default config"
REQ_ROOT_WINS = "REQ-INPUTKIND-BDRABCZUK-1246B7A946C0: --datasets-root wins over the --dataset-name default root"
REQ_GENERATE = "REQ-INPUTKIND-BDRABCZUK-894C3E1DEA21: --generate-config writes each config field argument"
REQ_DATA_PATH_ROOT = "REQ-INPUTKIND-BDRABCZUK-F9E8034E7565: --data-path re-derives the default datasets_root"
REQ_GENERATE_ROOT = "REQ-INPUTKIND-BDRABCZUK-7E9FAA84F332: --generate-config keeps --datasets-root over the derived one"
REQ_NO_ABBREV = "REQ-INPUTKIND-BDRABCZUK-A622708CCB4D: main.py refuses an abbreviated option name"

STREAMING = "streaming"
ROW_IDS = [row.flag for row in CONFIG_ARGUMENTS]
STAGE_PICKED = "run_cortical_tiles"
# input_kind cases need an input holding the chosen kind: KIND_INPUT holds a T1 file and one Morphologist graph.
KIND_INPUT = "kind-input"
KIND_INPUT_GRAPHS = Path("derivatives/morphologist-6.0/sub-01/t1mri/default_acquisition/default_analysis/folds/3.1")
# region_masks takes NAME=VERSION entries: champollion-pipeline exits 2 on any other text (REQ-MODELVER-BDRABCZUK-
# 9273939CFE5A, TASK-MODELVER-BDRABCZUK-6C6C79B0F6DE), so its cases are valid per-region Mask version maps.
REGION_MASKS_CLI = ["SC-sylv_left=canonical_25", "SC-sylv_right=canonical_25"]
REGION_MASKS_YAML = ["SOr_left=canonical_corrected_26_1"]


@dataclasses.dataclass
class RunRecord:
    """What one main() run handed on: the orchestrator's PipelineConfig, or the streaming runner's call."""

    configs: list = dataclasses.field(default_factory=list)
    streaming_calls: list = dataclasses.field(default_factory=list)


def _write_input_folders(root: Path) -> None:
    """The input folders the input_path and input_kind cases name: T1 inputs, and KIND_INPUT holding both kinds."""
    for name in ("cli-input_path", "yaml-input_path", KIND_INPUT):
        (root / name).mkdir()
        (root / name / "sub-01.nii.gz").write_bytes(b"synthetic")
    graphs = root / KIND_INPUT / KIND_INPUT_GRAPHS
    graphs.mkdir(parents=True)
    (graphs / "Lsub-01.arg").write_text("synthetic")


def _stub_main_side_effects(monkeypatch, tmp_path, record: RunRecord) -> None:
    """Run main() in tmp_path with no update check, process setup or profiling, recording what it would run."""

    class _RecordingOrchestrator:
        STAGE_REGISTRY = MAIN.PipelineOrchestrator.STAGE_REGISTRY

        def __init__(self, config):
            record.configs.append(config)

        def run(self):
            return 0

    def _record_streaming(*args, **kwargs):
        record.streaming_calls.append((args, kwargs))
        return 0

    monkeypatch.chdir(tmp_path)
    _write_input_folders(tmp_path)
    monkeypatch.setattr(MAIN, "check_for_updates", lambda: None)
    monkeypatch.setattr(MAIN, "init_pipeline_process", lambda: None)
    monkeypatch.setattr(MAIN, "start_profiling", lambda *a, **k: None)
    monkeypatch.setattr(MAIN, "PipelineOrchestrator", _RecordingOrchestrator)
    monkeypatch.setattr("champollion_utils.script_builder.check_for_updates", lambda *a, **k: None)
    monkeypatch.setattr("champollion_utils.script_builder.init_process", lambda *a, **k: None)
    monkeypatch.setattr("champollion_utils.script_builder.start_profiling", lambda *a, **k: None)
    monkeypatch.setattr("champollion_pipeline.streaming.runner.run_parallel_pipeline", _record_streaming)


@pytest.fixture
def run_main(monkeypatch, tmp_path):
    """Run main() with the given arguments; return the RunRecord, failing on a refused command line."""
    record = RunRecord()
    _stub_main_side_effects(monkeypatch, tmp_path, record)

    def _run(*cli_args: str) -> RunRecord:
        monkeypatch.setattr("sys.argv", ["main.py", *cli_args])
        try:
            exit_code = MAIN.main()
        except SystemExit as exc:
            pytest.fail(f"main.py refused the command line {list(cli_args)} with exit status {exc.code}")
        assert exit_code == 0, f"main() returned {exit_code} for {list(cli_args)}"
        runs = len(record.configs) + len(record.streaming_calls)
        assert runs == 1, f"main() ran {runs} configurations for {list(cli_args)}, expected 1"
        return record

    return _run


def _field_value(record: RunRecord, field_name: str, is_dataset_field: bool):
    """The value of the field in the configuration main() ran; mode reads streaming when the streaming runner ran."""
    if record.streaming_calls:
        assert field_name == "mode", f"main() ran in streaming mode, but the case sets {field_name}"
        return STREAMING
    config = record.configs[0]
    return getattr(config.dataset if is_dataset_field else config, field_name)


def _yaml_field_value(written: dict, row: ConfigArgument):
    return written["dataset"][row.field_name] if row.is_dataset_field else written[row.field_name]


def _named_field_case(row: ConfigArgument, current) -> tuple[list[str], object] | None:
    """Arguments and value for a field whose values are a fixed set of words or a map, or None for any other field."""
    if row.field_name == "mode":
        target = STREAMING if current != STREAMING else "sequential"
        return [row.flag, target], target
    if row.field_name == "log_level":
        target = "ERROR" if current != "ERROR" else "WARNING"
        return [row.flag, target], target
    if row.field_name == "input_kind":
        target = "t1" if current != "t1" else "morphologist"
        return [row.flag, target, "--input-path", KIND_INPUT], target
    if row.field_name == "region_masks":
        return [row.flag, *REGION_MASKS_CLI], list(REGION_MASKS_CLI)
    return None


def _argument_case(row: ConfigArgument, current) -> tuple[list[str], object]:
    """Command-line arguments setting the row's field to a value other than current, and that value."""
    named = _named_field_case(row, current)
    if named is not None:
        return named
    kind = row.kind
    if kind is ArgumentKind.FLAG:
        target = not current
        return [row.flag if target else "--no-" + row.flag.removeprefix("--")], target
    if kind is ArgumentKind.TEXT:
        target = f"cli-{row.field_name}"
        return [row.flag, target], target
    if kind is ArgumentKind.INTEGER:
        target = 7 if current != 7 else 8
        return [row.flag, str(target)], target
    if kind is ArgumentKind.TEXT_LIST:
        return [row.flag, "cli-a", "cli-b"], ["cli-a", "cli-b"]
    if kind is ArgumentKind.INTEGER_LIST:
        return [row.flag, "3", "4"], [3, 4]
    if kind is ArgumentKind.STAGE_SET:
        target = {stage: stage == STAGE_PICKED for stage in current}
        return [row.flag, STAGE_PICKED], target
    raise AssertionError(f"no test value for argument kind {kind} of {row.field_name}")


def _yaml_value(field_name: str, kind: ArgumentKind | None, default):
    """A value of the field, other than its default, for a --config YAML file."""
    if field_name == "mode":
        return STREAMING
    if field_name == "log_level":
        return "WARNING"
    if field_name == "input_kind":
        return "morphologist"
    if field_name == "region_masks":
        return list(REGION_MASKS_YAML)
    if kind is ArgumentKind.FLAG:
        return not default
    if kind is ArgumentKind.INTEGER:
        return 3
    if kind is ArgumentKind.TEXT_LIST:
        return ["yaml-a"]
    if kind is ArgumentKind.INTEGER_LIST:
        return [1]
    if kind is ArgumentKind.STAGE_SET:
        return dict.fromkeys(default, True)
    if field_name == "dependencies":
        return {stage: [] for stage in default}
    return f"yaml-{field_name}"


def _default_value(field_name: str, is_dataset_field: bool):
    config = MAIN.create_default_config()
    return getattr(config.dataset if is_dataset_field else config, field_name)


def _write_yaml(path: Path, field_name: str, is_dataset_field: bool, value) -> str:
    """Write the default configuration with one field set to value, hf_token included, as a --config YAML file."""
    config_dict = dataclasses.asdict(MAIN.create_default_config())
    section = config_dict["dataset"] if is_dataset_field else config_dict
    section[field_name] = value
    if field_name == "mode" and value == STREAMING:
        config_dict["dataset"]["embeddings_only"] = True  # the streaming guard needs it
    if field_name == "input_kind":
        config_dict["dataset"]["input_path"] = KIND_INPUT  # the chosen kind must be there
    path.write_text(yaml.safe_dump(config_dict, sort_keys=False))
    return str(path)


def _streaming_guard_arguments(row: ConfigArgument, target) -> list[str]:
    """--embeddings-only for a streaming run from the command line: main() refuses streaming without it."""
    return ["--embeddings-only"] if row.field_name == "mode" and target == STREAMING else []


_YAML_CASES = [(row.field_name, row.is_dataset_field, row.kind) for row in CONFIG_ARGUMENTS] + [
    (name, name in {"hf_token", "config_path"}, None) for name in sorted(EXEMPT_FIELDS)
]


@pytest.mark.unit
@pytest.mark.parametrize("row", CONFIG_ARGUMENTS, ids=ROW_IDS)
def test_argument_sets_field_over_default(run_main, row):
    """REQ-INPUTKIND-BDRABCZUK-2D667852F456: config field argument wins over the default config."""
    argv, target = _argument_case(row, _default_value(row.field_name, row.is_dataset_field))
    record = run_main(*argv, *_streaming_guard_arguments(row, target))
    assert _field_value(record, row.field_name, row.is_dataset_field) == target, REQ_OVER_DEFAULT


@pytest.mark.unit
@pytest.mark.parametrize("row", CONFIG_ARGUMENTS, ids=ROW_IDS)
def test_argument_sets_field_over_config_file(run_main, row, tmp_path):
    """REQ-INPUTKIND-BDRABCZUK-2D6361D75AF2: config field argument wins over the --config file."""
    default = _default_value(row.field_name, row.is_dataset_field)
    yaml_value = _yaml_value(row.field_name, row.kind, default)
    config_file = _write_yaml(tmp_path / "config.yaml", row.field_name, row.is_dataset_field, yaml_value)
    argv, target = _argument_case(row, yaml_value)
    record = run_main("--config", config_file, *argv, *_streaming_guard_arguments(row, target))
    assert target != yaml_value
    assert _field_value(record, row.field_name, row.is_dataset_field) == target, REQ_OVER_YAML


@pytest.mark.unit
@pytest.mark.parametrize(("field_name", "is_dataset_field", "kind"), _YAML_CASES, ids=[c[0] for c in _YAML_CASES])
def test_config_file_value_kept_without_argument(run_main, tmp_path, field_name, is_dataset_field, kind):
    """REQ-INPUTKIND-BDRABCZUK-5857EDED9E61: --config file value wins over the default config."""
    yaml_value = _yaml_value(field_name, kind, _default_value(field_name, is_dataset_field))
    config_file = _write_yaml(tmp_path / "config.yaml", field_name, is_dataset_field, yaml_value)
    record = run_main("--config", config_file)
    assert _field_value(record, field_name, is_dataset_field) == yaml_value, REQ_YAML_OVER_DEFAULT


@pytest.mark.unit
def test_no_argument_runs_default_config(run_main):
    """REQ-INPUTKIND-BDRABCZUK-F0D02CD54CBD: no argument runs the default config."""
    record = run_main()
    assert record.configs == [MAIN.create_default_config()], REQ_DEFAULT


@pytest.mark.unit
def test_datasets_root_argument_wins_over_dataset_name(run_main):
    """REQ-INPUTKIND-BDRABCZUK-1246B7A946C0: --datasets-root wins over the --dataset-name default root."""
    record = run_main("--dataset-name", "foo", "--datasets-root", "/cli/root")
    assert record.configs[0].dataset.datasets_root == "/cli/root", REQ_ROOT_WINS


@pytest.mark.unit
def test_data_path_rederives_default_datasets_root(run_main):
    """REQ-INPUTKIND-BDRABCZUK-F9E8034E7565: --data-path re-derives the default datasets_root."""
    default_name = MAIN.create_default_config().dataset.name
    record = run_main("--data-path", "/cli/data")
    assert record.configs[0].dataset.datasets_root == str(Path("/cli/data") / default_name), REQ_DATA_PATH_ROOT


@pytest.fixture
def generate_config(run_main, monkeypatch, tmp_path):
    """Run main() with --generate-config <tmp>/out.yaml plus the given arguments; return the written YAML."""

    def _run(*cli_args: str) -> dict:
        output = tmp_path / "out.yaml"
        monkeypatch.setattr("sys.argv", ["main.py", "--generate-config", str(output), *cli_args])
        try:
            exit_code = MAIN.main()
        except SystemExit as exc:
            pytest.fail(f"main.py refused --generate-config with {list(cli_args)} with exit status {exc.code}")
        assert exit_code == 0
        return yaml.safe_load(output.read_text())

    return _run


@pytest.mark.unit
@pytest.mark.parametrize("row", CONFIG_ARGUMENTS, ids=ROW_IDS)
def test_generate_config_writes_argument_value(generate_config, row):
    """REQ-INPUTKIND-BDRABCZUK-894C3E1DEA21: --generate-config writes each config field argument."""
    argv, target = _argument_case(row, _default_value(row.field_name, row.is_dataset_field))
    written = generate_config(*argv)
    assert _yaml_field_value(written, row) == target, REQ_GENERATE


@pytest.mark.unit
def test_generate_config_datasets_root_argument_wins_over_dataset_name(generate_config):
    """REQ-INPUTKIND-BDRABCZUK-7E9FAA84F332: --generate-config keeps --datasets-root over the derived one."""
    written = generate_config("--dataset-name", "foo", "--datasets-root", "/cli/root")
    assert written["dataset"]["datasets_root"] == "/cli/root", REQ_GENERATE_ROOT


# Each command line starts with a proper prefix of one or more CONFIG_ARGUMENTS flags that is no flag itself: --data
# reads like several flags, the others like exactly one, which argparse's default prefix matching would take.
ABBREVIATED_COMMAND_LINES = [
    ["--data", "/cli/data"],
    ["--datasets-r", "/cli/root"],
    ["--dataset-n", "foo"],
    ["--models-p", "/cli/models"],
    ["--verb"],
    ["--embeddings-o"],
]


@pytest.mark.unit
@pytest.mark.parametrize("argv", ABBREVIATED_COMMAND_LINES, ids=[argv[0] for argv in ABBREVIATED_COMMAND_LINES])
def test_option_name_prefix_exits_2(monkeypatch, tmp_path, argv):
    """REQ-INPUTKIND-BDRABCZUK-A622708CCB4D: main.py refuses an abbreviated option name."""
    flags = {row.flag for row in CONFIG_ARGUMENTS}
    abbreviation = argv[0]
    assert abbreviation not in flags
    assert any(flag.startswith(abbreviation) for flag in flags), f"{abbreviation} abbreviates no registry flag"
    record = RunRecord()
    _stub_main_side_effects(monkeypatch, tmp_path, record)
    monkeypatch.setattr("sys.argv", ["main.py", *argv])
    try:
        exit_code = MAIN.main()
    except SystemExit as exc:
        exit_code = exc.code
    else:
        pytest.fail(f"main.py accepted the abbreviated option {abbreviation} and returned {exit_code}; {REQ_NO_ABBREV}")
    assert exit_code == 2, REQ_NO_ABBREV
    assert record.configs == [] and record.streaming_calls == [], REQ_NO_ABBREV
