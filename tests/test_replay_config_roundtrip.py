"""Replay round trip: main.py --config <replay file> rebuilds the run (TASK-INPUTKIND-BDRABCZUK-34868188CC6F).

The replay file a main.py run leaves in its logs folder (tests/test_replay_config_written.py: where, when, what it
holds) is read back by main.py --config. These tests pin the round trip: the PipelineConfig the replay run hands to
PipelineOrchestrator equals the one of the original run, for a T1 start and for a Morphologist start, and the YAML
round trip of write_replay_config and ConfigLoader.load_from_yaml keeps each field's value and type (None apart from
the empty string, a bool apart from an int, list[int | None] items, dataset.input_kind t1, morphologist or None).
root_path, dependencies, dataset.config_path and dataset.hf_token stay out of the replay file by design; relative path
values are made absolute by design.

Requirements (wording in the ledger):
    test_replay_of_t1_start_rebuilds_the_run: Replay of a T1 start rebuilds the run configuration (system).
    test_replay_of_morphologist_start_rebuilds_the_run: Replay of a Morphologist start rebuilds the run
        configuration (system).
    test_replay_round_trip_keeps_each_field, test_replay_round_trip_keeps_input_kind,
    test_replay_round_trip_keeps_value_and_type: Replay file YAML round trip keeps each field value and type
        (component).

Equality is typed: each leaf value is compared with its type name, so True never equals 1 and None never equals "".
The config-less T1 start with dataset.input_kind None is already pinned by
tests/test_replay_config_written.py::test_replay_run_rebuilds_the_configuration and is not repeated here.

main() runs in a temporary working directory with the update check, the process setup and profiling stubbed and
PipelineOrchestrator replaced by a recorder (tests.main_stage_helpers.main_runner): no stage runs. The input kind prompt
is a terminal stand-in answering the planned kind.
"""

import dataclasses
import tempfile
import typing
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from champollion_pipeline import input_kind_choice, replay_config
from champollion_pipeline.config import ConfigLoader, DatasetConfig, PipelineConfig
from tests.main_stage_helpers import (
    enter_clean_work_dir,
    load_main_module,
    main_runner,
    make_morphologist_input,
    make_t1_input,
    ran_config,
)

MAIN = load_main_module("_champollion_main_replay_roundtrip_under_test")

T1 = "t1"
MORPHOLOGIST = "morphologist"
MORPHOLOGIST_STAGE = "generate_morphologist_graphs"
# Keys the replay file leaves out by design: (section, key); section None is the top level.
EXEMPT_KEYS = ((None, "root_path"), (None, "dependencies"), ("dataset", "config_path"), ("dataset", "hf_token"))
# Fields whose relative value the replay file makes absolute: the property generates them absolute, empty or None.
PATH_FIELDS = {
    PipelineConfig: {
        "data_path",
        "models_path",
        "outputs_path",
        "champollion_v1_path",
        "log_dir",
        "log",
        "profile_dir",
    },
    DatasetConfig: {
        "datasets_root",
        "input_path",
        "morphologist_graphs",
        "cortical_tiles_output",
        "crops_path",
        "embeddings_path",
        "sk_qc_path",
        "models_cache",
        "splits_basedir",
        "snapshots_path",
        "reference_data_path",
        "champollion_data_root",
    },
}


# ---- Typed comparison ----


def _typed(value):
    """value with each leaf replaced by (type name, value): True and 1, None and "" compare unequal."""
    if isinstance(value, dict):
        return ("dict", {key: _typed(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return (type(value).__name__, [_typed(item) for item in value])
    return (type(value).__name__, value)


def _comparable(config: PipelineConfig):
    """The typed fields of a PipelineConfig the replay file carries over: all but the exempt keys."""
    values = dataclasses.asdict(config)
    for section, key in EXEMPT_KEYS:
        (values[section] if section else values).pop(key, None)
    return _typed(values)


def _differences(replayed: PipelineConfig, original: PipelineConfig) -> list[str]:
    """'<field>: <original> -> <replayed>' for each top-level or dataset field the round trip changed."""
    before, after = _comparable(original)[1], _comparable(replayed)[1]
    lines = []
    for name in sorted(set(before) | set(after)):
        if name == "dataset":
            inner_before, inner_after = before[name][1], after.get(name, ("dict", {}))[1]
            for key in sorted(set(inner_before) | set(inner_after)):
                if inner_before.get(key) != inner_after.get(key):
                    lines.append(f"dataset.{key}: {inner_before.get(key)} -> {inner_after.get(key)}")
        elif before.get(name) != after.get(name):
            lines.append(f"{name}: {before.get(name)} -> {after.get(name)}")
    return lines


def _round_trip(config: PipelineConfig) -> PipelineConfig:
    """write_replay_config(config) read back by ConfigLoader.load_from_yaml."""
    return ConfigLoader.load_from_yaml(replay_config.write_replay_config(config))


# ---- main.py runs ----


@pytest.fixture
def work_dir(monkeypatch, tmp_path) -> Path:
    return enter_clean_work_dir(monkeypatch, tmp_path)


@pytest.fixture
def run_main(monkeypatch, work_dir):
    """Run main() with the given arguments; return the one PipelineConfig it handed to the orchestrator."""
    runner = main_runner(MAIN, monkeypatch)

    def _run(*cli_args: str) -> PipelineConfig:
        return ran_config(runner(*cli_args), cli_args)

    return _run


@pytest.fixture
def prompt(monkeypatch):
    """A terminal whose input kind prompt answers the kind set in the returned dict ('answer')."""
    stand_in = {"answer": MORPHOLOGIST}
    monkeypatch.setattr(input_kind_choice, "_is_terminal", lambda: True)
    monkeypatch.setattr(input_kind_choice, "_ask", lambda lines, question, choices, on_eof: stand_in["answer"])
    return stand_in


def _the_replay(config: PipelineConfig) -> Path:
    """The one replay_<YYYYmmdd-HHMMSS>.yaml file in the logs folder of config; fails when there is none or several."""
    folder = replay_config.logs_folder(config)
    found = sorted(folder.glob("replay_*.yaml")) if folder.is_dir() else []
    assert len(found) == 1, f"expected one replay file in {folder}, found {[path.name for path in found]}"
    return found[0]


def _write_yaml(path: Path, dataset: dict, **pipeline) -> Path:
    """A --config YAML: the default PipelineConfig with the given pipeline and dataset field values."""
    config = dataclasses.replace(PipelineConfig(), dataset=dataclasses.replace(DatasetConfig(), **dataset), **pipeline)
    ConfigLoader.save_to_yaml(config, str(path))
    return path


def _start_args(tmp_path: Path, input_folder: Path, how: str, kind: str | None) -> tuple[str, ...]:
    """main.py arguments of the first run: config-less (INPUT, --input-kind) or a --config YAML naming the input."""
    logs = ("--outputs-path", str(tmp_path / "out"), "--log-dir", str(tmp_path / "logs"))
    if how == "configless":
        return (str(input_folder), *logs, *(("--input-kind", kind) if kind else ()))
    yaml_file = _write_yaml(
        tmp_path / "start.yaml",
        outputs_path=str(tmp_path / "out"),
        log_dir=str(tmp_path / "logs"),
        stages=dict.fromkeys(PipelineConfig().stages, True),
        dataset={
            "input_path": str(input_folder),
            "input_kind": kind,
            "datasets_root": str(tmp_path / "derivatives_root"),
        },
    )
    return ("--config", str(yaml_file))


def _assert_replay_rebuilds(run_main, first_args: tuple[str, ...]) -> PipelineConfig:
    """Run first_args, then main.py --config <its replay file>; assert both runs hand on the same PipelineConfig."""
    first = run_main(*first_args)
    replay = _the_replay(first)
    second = run_main("--config", str(replay))
    changed = _differences(second, first)
    assert not changed, f"main.py --config {replay.name} changed: {changed}"
    return first


# ---- System: a replay of a T1 start rebuilds the run ----


@pytest.mark.unit
@pytest.mark.parametrize(
    ("how", "kind"),
    [("config", T1), ("config", None), ("configless", T1), ("prompted", T1)],
    ids=["config_kind_t1", "config_kind_none", "configless_kind_t1", "prompted_t1"],
)
def test_replay_of_t1_start_rebuilds_the_run(run_main, prompt, tmp_path, how, kind):
    if how == "prompted":
        prompt["answer"] = T1
        first_args = _start_args(tmp_path, make_morphologist_input(tmp_path, "both", with_t1=True), "configless", None)
    else:
        first_args = _start_args(tmp_path, make_t1_input(tmp_path, "raw"), how, kind)
    first = _assert_replay_rebuilds(run_main, first_args)
    assert first.dataset.input_kind == kind, f"the T1 start handed on input_kind {first.dataset.input_kind!r}"


# ---- System: a replay of a Morphologist start rebuilds the run ----


@pytest.mark.unit
@pytest.mark.parametrize(
    ("how", "kind"),
    [
        ("configless", None),
        ("configless", MORPHOLOGIST),
        ("config", MORPHOLOGIST),
        ("config", None),
        ("prompted", MORPHOLOGIST),
    ],
    ids=[
        "configless_kind_none",
        "configless_kind_morphologist",
        "config_kind_morphologist",
        "config_kind_none",
        "prompted_morphologist",
    ],
)
def test_replay_of_morphologist_start_rebuilds_the_run(run_main, prompt, tmp_path, how, kind):
    prompted = how == "prompted"
    graphs_root = make_morphologist_input(tmp_path, "morph", with_t1=prompted)
    first_args = _start_args(tmp_path, graphs_root, "configless", None) if prompted else None
    first = _assert_replay_rebuilds(run_main, first_args or _start_args(tmp_path, graphs_root, how, kind))
    assert first.dataset.input_kind == kind, f"the Morphologist start handed on input_kind {first.dataset.input_kind!r}"
    assert first.dataset.morphologist_graphs, "the Morphologist start handed on no dataset.morphologist_graphs"
    assert first.stages[MORPHOLOGIST_STAGE] is False, "the Morphologist start left stage 1 on"


# ---- Component: the YAML round trip keeps each field's value and type ----

_SEGMENT = st.text(st.characters(blacklist_characters="/\x00", blacklist_categories=("Cs",)), min_size=1).filter(
    lambda segment: segment not in (".", "..")
)
_ABSOLUTE_PATH = st.lists(_SEGMENT, min_size=1, max_size=3).map(lambda parts: "/" + "/".join(parts))
_TEXT = st.text(st.characters(blacklist_categories=("Cs",)))
_LEAF_STRATEGIES = {
    str: _TEXT,
    str | None: st.none() | _TEXT,
    int: st.integers(),
    int | None: st.none() | st.integers(),
    bool: st.booleans(),
    list[str]: st.lists(_TEXT, max_size=4),
    list[int | None]: st.lists(st.none() | st.integers(), max_size=4),
    dict[str, bool]: st.dictionaries(_TEXT, st.booleans(), max_size=6),
    dict[str, list[str]]: st.dictionaries(_TEXT, st.lists(_TEXT, max_size=3), max_size=3),
}


def _field_strategy(owner: type, name: str, hint):
    """The values one field of owner is generated from; a field type with no strategy fails loudly."""
    if hint is DatasetConfig:
        return _config_strategy(DatasetConfig)
    if name in PATH_FIELDS.get(owner, ()):
        optional = st.none() if hint == str | None else st.nothing()
        return st.just("") | _ABSOLUTE_PATH | optional
    if hint not in _LEAF_STRATEGIES:
        raise AssertionError(f"{owner.__name__}.{name}: no strategy for field type {hint}; add one to the property")
    return _LEAF_STRATEGIES[hint]


def _config_strategy(owner: type):
    """Instances of the PipelineConfig or DatasetConfig dataclass with every field generated from its type."""
    hints = typing.get_type_hints(owner)
    fields = {field.name: _field_strategy(owner, field.name, hints[field.name]) for field in dataclasses.fields(owner)}
    return st.builds(owner, **fields)


@pytest.mark.unit
@settings(max_examples=80, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(config=_config_strategy(PipelineConfig))
def test_replay_round_trip_keeps_each_field(config):
    with tempfile.TemporaryDirectory() as logs:
        config.log_dir = logs
        replayed = _round_trip(config)
    changed = _differences(replayed, config)
    assert not changed, f"the replay YAML round trip changed: {changed}"


@pytest.mark.unit
@pytest.mark.parametrize("kind", [T1, MORPHOLOGIST, None], ids=["t1", "morphologist", "none"])
def test_replay_round_trip_keeps_input_kind(tmp_path, kind):
    config = PipelineConfig(log_dir=str(tmp_path / "logs"))
    config.dataset.input_kind = kind
    replayed = _round_trip(config)
    assert _typed(replayed.dataset.input_kind) == _typed(kind), (
        f"dataset.input_kind {kind!r} read back as {replayed.dataset.input_kind!r}"
    )
    assert not _differences(replayed, config), f"the replay YAML round trip changed: {_differences(replayed, config)}"


VALUE_CASES = [
    ("dataset", "models_cache", None),
    ("dataset", "models_cache", ""),
    ("dataset", "champollion_data_root", None),
    ("dataset", "hf_repo_id", ""),
    ("dataset", "skel_seed", 0),
    ("dataset", "idx_region_evaluation", None),
    ("dataset", "checkpoint_epoch", 0),
    ("dataset", "epochs", [None, 3, None]),
    ("dataset", "epochs", []),
    ("dataset", "regions", []),
    ("dataset", "regions", ["True", "1.0", "null", "~"]),
    ("dataset", "labelling_session", ""),
    ("dataset", "name", "yes"),
    ("dataset", "short_name", "1"),
    ("dataset", "hf_enabled", True),
    ("dataset", "allow_prefix_ids", True),
    (None, "verbose", True),
    (None, "stop_on_error", False),
    (None, "log", None),
    (None, "log", ""),
    (None, "profile_dir", None),
    (None, "n_workers", 0),
    (None, "stages", dict.fromkeys(PipelineConfig().stages, False)),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("section", "name", "value"),
    VALUE_CASES,
    ids=[f"{section or 'pipeline'}.{name}={value!r}" for section, name, value in VALUE_CASES],
)
def test_replay_round_trip_keeps_value_and_type(tmp_path, section, name, value):
    config = PipelineConfig(log_dir=str(tmp_path / "logs"))
    setattr(config.dataset if section else config, name, value)
    replayed = _round_trip(config)
    read_back = getattr(replayed.dataset if section else replayed, name)
    assert _typed(read_back) == _typed(value), f"{section or 'pipeline'}.{name} {value!r} read back as {read_back!r}"
    assert not _differences(replayed, config), f"the replay YAML round trip changed: {_differences(replayed, config)}"
