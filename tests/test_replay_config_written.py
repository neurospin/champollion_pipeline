"""main.py replay config: every run writes replay_<timestamp>.yaml (TASK-INPUTKIND-BDRABCZUK-29C188034B6A).

User answers 2026-11-05: the config file is written only to replay a run without its arguments; the replay file is
timestamped and sits beside the run log, in the logs folder (PipelineConfig.log_dir, else <outputs_path>/logs). It holds
the effective configuration (command line, inference and the --config YAML merged) with absolute paths; root_path,
dependencies, dataset.config_path and dataset.hf_token stay out of it. A replay of a run whose input kind prompt was
answered never prompts again.

Requirements (wording in the ledger):
    test_replay_run_rebuilds_the_configuration: Sequential main.py run leaves a replay file that rebuilds its
        configuration (system).
    test_replay_file_in_log_dir_before_orchestrator: Replay file in PipelineConfig.log_dir before the orchestrator
        starts.
    test_replay_file_in_outputs_logs_without_log_dir: Replay file in <outputs_path>/logs when log_dir is empty.
    test_replay_holds_the_effective_configuration: Replay file holds the effective configuration.
    test_replay_leaves_out_exempt_keys: Replay file leaves out the exempt keys.
    test_replay_holds_no_hf_token_string: Replay file holds no Hugging Face token string.
    test_replay_makes_relative_path_absolute: Replay file holds relative paths made absolute.
    test_replay_keeps_subject_relative_paths: Replay file keeps the subject-relative graph paths.
    test_replay_of_prompted_run_shows_no_prompt: Replay of a prompted run shows no input kind prompt.
    test_streaming_replay_*: the same file, place and moment for a streaming run, which hands its configuration to
        run_parallel_pipeline (replaced by a recorder).
    test_replay_written_for_dry_run: Replay file written for a --dry-run run (sequential and streaming).
    test_unwritable_replay_file_warns_and_run_goes_on: Unwritable replay file only warns.

User decision 2026-11-19: every main.py run writes the replay file, sequential and streaming, --dry-run included; two
runs started in the same second share a name and the second overwrites the first. User decision 2026-11-19 12:05: when
the replay file cannot be written (OSError), main.py logs one warning naming the path and the error and the run goes on.
The autouse fixture replay_files_in_tmp of conftest.py sends the replay files of every test to a temporary folder; the
last test checks it.

main() runs in a temporary working directory with the update check, the process setup and profiling stubbed, and
PipelineOrchestrator replaced by a recorder: no stage runs. The observables are the replay file in the logs folder and
the PipelineConfig main() hands to the orchestrator. The input kind prompt is a terminal stand-in: _is_terminal answers
True and _ask returns the planned answer and counts its calls.
"""

import dataclasses
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from champollion_pipeline import input_kind_choice
from champollion_pipeline.config import ConfigLoader, PipelineConfig
from champollion_pipeline.replay_config import write_replay_config
from tests.main_stage_helpers import (
    enter_work_dir,
    load_main_module,
    make_morphologist_input,
    make_t1_input,
    stub_main_setup,
)

MAIN = load_main_module("_champollion_main_replay_config_under_test")

REPLAY_NAME = re.compile(r"replay_(\d{8}-\d{6})\.yaml")
REPLAY_STAMP_FORMAT = "%Y%m%d-%H%M%S"
# Keys the replay file leaves out: (section, key); section None is the top level.
EXEMPT_KEYS = ((None, "root_path"), (None, "dependencies"), ("dataset", "config_path"), ("dataset", "hf_token"))
# Path fields whose relative value the replay file makes absolute: (section, field).
PIPELINE_PATH_FIELDS = (
    "data_path",
    "models_path",
    "outputs_path",
    "champollion_v1_path",
    "log_dir",
    "log",
    "profile_dir",
)
DATASET_PATH_FIELDS = (
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
)
PATH_FIELDS = tuple((None, name) for name in PIPELINE_PATH_FIELDS) + tuple(
    ("dataset", name) for name in DATASET_PATH_FIELDS
)
SUBJECT_RELATIVE = {
    "path_to_graph": "t1mri/default_acquisition/default_analysis/folds/3.3/base",
    "path_sk_with_hull": "t1mri/default_acquisition/default_analysis/segmentation",
}
TOKEN = "hf_replayTestToken0123456789"
MORPHOLOGIST = "morphologist"


@dataclasses.dataclass
class Run:
    """One main() run: exit status, the PipelineConfigs handed to the orchestrator, what each saw at its start."""

    status: int | str | None
    configs: list
    replays_at_start: list


@dataclasses.dataclass
class Prompt:
    """Terminal stand-in of the input kind and subjects folder prompts: the planned answer and the questions asked."""

    answer: str
    asked: list = dataclasses.field(default_factory=list)


@pytest.fixture
def work_dir(monkeypatch, tmp_path) -> Path:
    for name in list(os.environ):
        if name == "HF_TOKEN" or name.startswith("CHAMPOLLION_"):
            monkeypatch.delenv(name)
    return enter_work_dir(monkeypatch, tmp_path)


@pytest.fixture
def prompt(monkeypatch) -> Prompt:
    """A terminal whose prompts answer morphologist; the questions asked are recorded."""
    stand_in = Prompt(MORPHOLOGIST)

    def _ask(lines, question, choices, on_eof):
        stand_in.asked.append(question)
        return stand_in.answer

    monkeypatch.setattr(input_kind_choice, "_is_terminal", lambda: True)
    monkeypatch.setattr(input_kind_choice, "_ask", _ask)
    return stand_in


@pytest.fixture
def run_main(monkeypatch, work_dir):
    """Run main() with the given arguments; the orchestrator records its config and the replay files it finds."""
    watched: list[Path] = []

    def _run(*cli_args: str, logs: Path | None = None) -> Run:
        run = Run(status=None, configs=[], replays_at_start=[])
        watched[:] = [logs] if logs is not None else []

        class _RecordingOrchestrator:
            STAGE_REGISTRY = MAIN.PipelineOrchestrator.STAGE_REGISTRY

            def __init__(self, config) -> None:
                run.configs.append(config)
                folder = watched[0] if watched else _logs_folder(config)
                run.replays_at_start.append(_replay_files(folder))

            def run(self) -> int:
                return 0

        stub_main_setup(MAIN, monkeypatch)
        monkeypatch.setattr(MAIN, "start_profiling", lambda *a, **k: None)
        monkeypatch.setattr(MAIN, "PipelineOrchestrator", _RecordingOrchestrator)
        monkeypatch.setattr("champollion_utils.script_builder.start_profiling", lambda *a, **k: None)
        monkeypatch.setattr("sys.argv", ["main.py", *cli_args])
        try:
            run.status = MAIN.main()
        except SystemExit as stop:
            run.status = stop.code
        assert run.status == 0, f"main() exit status {run.status} for {list(cli_args)}, expected 0"
        assert len(run.configs) == 1, f"main() ran {len(run.configs)} configurations for {list(cli_args)}"
        return run

    return _run


def _logs_folder(config: PipelineConfig) -> Path:
    """The run's logs folder: PipelineConfig.log_dir, else <outputs_path>/logs, against the working directory."""
    folder = Path(config.log_dir) if config.log_dir else Path(config.outputs_path) / "logs"
    return Path(os.path.abspath(folder))


def _replay_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if REPLAY_NAME.fullmatch(path.name))


def _the_replay(folder: Path) -> Path:
    """The one replay_<YYYYmmdd-HHMMSS>.yaml file of the logs folder; fails when there is none or several."""
    found = _replay_files(folder)
    listed = sorted(path.name for path in folder.iterdir()) if folder.is_dir() else "no such folder"
    assert len(found) == 1, f"expected one replay_<YYYYmmdd-HHMMSS>.yaml in {folder}, found {listed}"
    return found[0]


def _comparable(config: PipelineConfig) -> dict:
    """The fields of a PipelineConfig the replay file carries over: all but the exempt keys."""
    values = dataclasses.asdict(config)
    for section, key in EXEMPT_KEYS:
        (values[section] if section else values).pop(key, None)
    return values


def _write_config(path: Path, **fields) -> Path:
    """A --config YAML: the default PipelineConfig with the given top-level or dataset_<field> values."""
    config = PipelineConfig()
    for name, value in fields.items():
        if name.startswith("dataset_"):
            setattr(config.dataset, name.removeprefix("dataset_"), value)
        else:
            setattr(config, name, value)
    ConfigLoader.save_to_yaml(config, str(path))
    return path


def _t1_input(tmp_path: Path) -> Path:
    return make_t1_input(tmp_path, "raw")


def _both_input(tmp_path: Path) -> Path:
    """Dataset root holding a top-level T1 file and one Morphologist subjects folder: the kind prompt asks."""
    return make_morphologist_input(tmp_path, "both", with_t1=True)


def _cli_args(tmp_path: Path) -> tuple[str, ...]:
    return (
        "--outputs-path",
        str(tmp_path / "out"),
        "--log-dir",
        str(tmp_path / "logs"),
        "--dataset-name",
        "replayed",
        "--datasets-root",
        str(tmp_path / "data" / "replayed"),
        "--njobs",
        "3",
        "--stages",
        "generate_embeddings",
        "put_together_embeddings",
    )


# ---- System: the replay file rebuilds the run's configuration ----


@pytest.mark.unit
@pytest.mark.parametrize("case", ["command_line", "configless_input"])
def test_replay_run_rebuilds_the_configuration(run_main, tmp_path, case):
    args = _cli_args(tmp_path) if case == "command_line" else (str(_t1_input(tmp_path)),)
    first = run_main(*args).configs[0]
    replay = _the_replay(_logs_folder(first))
    second = run_main("--config", str(replay)).configs[0]
    assert _comparable(second) == _comparable(first), f"main.py --config {replay} rebuilt another configuration"


# ---- Where and when the replay file is written ----


@pytest.mark.unit
def test_replay_file_in_log_dir_before_orchestrator(run_main, tmp_path):
    logs = tmp_path / "logs"
    before = datetime.now().replace(microsecond=0)
    run = run_main("--outputs-path", str(tmp_path / "out"), "--log-dir", str(logs), logs=logs)
    after = datetime.now()
    found = run.replays_at_start[0]
    assert len(found) == 1, f"PipelineOrchestrator started with {[p.name for p in found]} in --log-dir {logs}"
    stamp = datetime.strptime(REPLAY_NAME.fullmatch(found[0].name).group(1), REPLAY_STAMP_FORMAT)
    assert before <= stamp <= after, f"replay file stamp {stamp} outside the run ({before} to {after})"


@pytest.mark.unit
def test_replay_file_in_outputs_logs_without_log_dir(run_main, tmp_path):
    logs = tmp_path / "out" / "logs"
    run = run_main("--outputs-path", str(tmp_path / "out"), logs=logs)
    assert run.configs[0].log_dir == "", f"log_dir {run.configs[0].log_dir!r} is set; this case needs it empty"
    found = run.replays_at_start[0]
    assert len(found) == 1, f"PipelineOrchestrator started with {[p.name for p in found]} in {logs}"


# ---- What the replay file holds ----


@pytest.mark.unit
@pytest.mark.parametrize("case", ["command_line", "yaml_and_command_line", "configless_t1", "prompted_kind"])
def test_replay_holds_the_effective_configuration(run_main, prompt, tmp_path, case):
    if case == "command_line":
        args = _cli_args(tmp_path)
    elif case == "yaml_and_command_line":
        yaml_file = _write_config(
            tmp_path / "in.yaml",
            outputs_path=str(tmp_path / "out"),
            log_dir=str(tmp_path / "logs"),
            dataset_njobs=5,
            dataset_labels=["Age", "Sex"],
            dataset_masks_version="canonical_25",
        )
        args = ("--config", str(yaml_file), "--njobs", "7")
    elif case == "configless_t1":
        args = (str(_t1_input(tmp_path)),)
    else:
        args = (str(_both_input(tmp_path)),)
    handed = run_main(*args).configs[0]
    if case == "yaml_and_command_line":
        assert (handed.dataset.njobs, handed.dataset.labels) == (7, ["Age", "Sex"]), (
            "the YAML and --njobs did not merge"
        )
    expected_kind = MORPHOLOGIST if case == "prompted_kind" else None
    assert handed.dataset.input_kind == expected_kind, f"input_kind {handed.dataset.input_kind!r} handed on"
    replayed = ConfigLoader.load_from_yaml(str(_the_replay(_logs_folder(handed))))
    assert _comparable(replayed) == _comparable(handed), "the replay file differs from the configuration of the run"


@pytest.mark.unit
def test_replay_leaves_out_exempt_keys(run_main, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    yaml_file = _write_config(
        tmp_path / "in.yaml",
        outputs_path=str(tmp_path / "out"),
        log_dir=str(tmp_path / "logs"),
        dataset_config_path=str(tmp_path / "unused.yaml"),
    )
    handed = run_main("--config", str(yaml_file)).configs[0]
    held = yaml.safe_load(_the_replay(_logs_folder(handed)).read_text())
    present = [
        f"{section}.{key}" if section else key
        for section, key in EXEMPT_KEYS
        if key in (held.get(section, {}) if section else held)
    ]
    assert present == [], f"the replay file holds the exempt keys {present}"


@pytest.mark.unit
@pytest.mark.parametrize("source", ["environment", "yaml"])
def test_replay_holds_no_hf_token_string(run_main, monkeypatch, tmp_path, source):
    yaml_file = _write_config(tmp_path / "in.yaml", outputs_path=str(tmp_path / "out"), log_dir=str(tmp_path / "logs"))
    if source == "environment":
        monkeypatch.setenv("HF_TOKEN", TOKEN)
    else:
        values = yaml.safe_load(yaml_file.read_text())
        values["dataset"]["hf_token"] = TOKEN
        yaml_file.write_text(yaml.safe_dump(values, sort_keys=False))
    handed = run_main("--config", str(yaml_file)).configs[0]
    assert handed.dataset.hf_token == TOKEN, f"the run's dataset.hf_token is {handed.dataset.hf_token!r}"
    assert TOKEN not in _the_replay(_logs_folder(handed)).read_text(), "the replay file holds the token string"


@pytest.mark.unit
@pytest.mark.parametrize(("section", "field"), PATH_FIELDS, ids=[f"{s or 'pipeline'}.{f}" for s, f in PATH_FIELDS])
def test_replay_makes_relative_path_absolute(run_main, work_dir, tmp_path, section, field):
    relative = os.path.join("rel", field)
    fields = {
        "outputs_path": str(tmp_path / "out"),
        "log_dir": str(tmp_path / "logs"),
        "dataset_datasets_root": str(tmp_path / "root"),
    }
    fields[f"dataset_{field}" if section else field] = relative
    if field == "input_path":
        (work_dir / relative).mkdir(parents=True)
        (work_dir / relative / "sub-01.nii.gz").write_bytes(b"synthetic")
    handed = run_main("--config", str(_write_config(tmp_path / "in.yaml", **fields))).configs[0]
    held = yaml.safe_load(_the_replay(_logs_folder(handed)).read_text())
    value = (held[section] if section else held)[field]
    assert value == str(work_dir / relative), f"replay {section or 'pipeline'}.{field} is {value!r}, expected absolute"


@pytest.mark.unit
def test_replay_keeps_subject_relative_paths(run_main, tmp_path):
    fields = {f"dataset_{name}": value for name, value in SUBJECT_RELATIVE.items()}
    yaml_file = _write_config(
        tmp_path / "in.yaml", outputs_path=str(tmp_path / "out"), log_dir=str(tmp_path / "logs"), **fields
    )
    handed = run_main("--config", str(yaml_file)).configs[0]
    held = yaml.safe_load(_the_replay(_logs_folder(handed)).read_text())["dataset"]
    kept = {name: held.get(name) for name in SUBJECT_RELATIVE}
    assert kept == SUBJECT_RELATIVE, f"the replay file changed the subject-relative paths to {kept}"


# ---- Replaying a prompted run ----


@pytest.mark.unit
def test_replay_of_prompted_run_shows_no_prompt(run_main, prompt, tmp_path):
    first = run_main(str(_both_input(tmp_path))).configs[0]
    assert prompt.asked, "the input kind prompt was not shown for an input holding both kinds"
    replay = _the_replay(_logs_folder(first))
    prompt.asked.clear()
    second = run_main("--config", str(replay)).configs[0]
    assert prompt.asked == [], f"main.py --config {replay} showed the prompts {prompt.asked}"
    assert second.dataset.input_kind == MORPHOLOGIST, f"the replayed input_kind is {second.dataset.input_kind!r}"


# ---- Streaming runs and --dry-run runs write the replay file too ----


@dataclasses.dataclass
class StreamingRun:
    """One streaming main() run: the DatasetConfigs given to run_parallel_pipeline, the replay files then present."""

    datasets: list
    replays_at_start: list


@pytest.fixture
def run_streaming(monkeypatch, work_dir):
    """Run main() in streaming mode; run_parallel_pipeline records its dataset and the replay files in `logs`."""

    def _run(*cli_args: str, logs: Path) -> StreamingRun:
        run = StreamingRun(datasets=[], replays_at_start=[])

        def _recorder(_settings, dataset, _models_source) -> int:
            run.datasets.append(dataset)
            run.replays_at_start.append(_replay_files(logs))
            return 0

        stub_main_setup(MAIN, monkeypatch)
        monkeypatch.setattr(MAIN, "start_profiling", lambda *a, **k: None)
        monkeypatch.setattr("champollion_utils.script_builder.start_profiling", lambda *a, **k: None)
        monkeypatch.setattr("champollion_pipeline.streaming.runner.run_parallel_pipeline", _recorder)
        monkeypatch.setattr("sys.argv", ["main.py", *cli_args])
        try:
            status = MAIN.main()
        except SystemExit as stop:
            status = stop.code
        assert status == 0, f"main() exit status {status} for {list(cli_args)}, expected 0"
        assert len(run.datasets) == 1, f"main() ran {len(run.datasets)} streaming runs for {list(cli_args)}"
        return run

    return _run


def _streaming_yaml(path: Path, tmp_path: Path, *, log_dir: bool) -> Path:
    fields = {"mode": "streaming", "outputs_path": str(tmp_path / "out"), "dataset_embeddings_only": True}
    if log_dir:
        fields["log_dir"] = str(tmp_path / "logs")
    return _write_config(path, **fields)


@pytest.mark.unit
@pytest.mark.parametrize("case", ["log_dir", "outputs_logs"])
def test_streaming_replay_before_runner(run_streaming, tmp_path, case):
    logs = tmp_path / "logs" if case == "log_dir" else tmp_path / "out" / "logs"
    yaml_file = _streaming_yaml(tmp_path / "in.yaml", tmp_path, log_dir=case == "log_dir")
    run = run_streaming("--config", str(yaml_file), "--njobs", "4", logs=logs)
    found = run.replays_at_start[0]
    assert len(found) == 1, f"run_parallel_pipeline started with {[p.name for p in found]} in {logs}"
    held = ConfigLoader.load_from_yaml(str(found[0]))
    assert (held.mode, held.dataset.njobs) == ("streaming", 4), "the streaming replay file lacks the effective values"


@pytest.mark.unit
def test_streaming_replay_rebuilds_the_run(run_streaming, tmp_path):
    logs = tmp_path / "logs"
    first = run_streaming("--config", str(_streaming_yaml(tmp_path / "in.yaml", tmp_path, log_dir=True)), logs=logs)
    replay = _the_replay(logs)
    second = run_streaming("--config", str(replay), logs=logs)
    exempt = {key for section, key in EXEMPT_KEYS if section}
    rebuilt = {k: v for k, v in dataclasses.asdict(second.datasets[0]).items() if k not in exempt}
    assert rebuilt == {k: v for k, v in dataclasses.asdict(first.datasets[0]).items() if k not in exempt}, (
        f"main.py --config {replay} rebuilt another streaming configuration"
    )


@pytest.mark.unit
def test_streaming_replay_keeps_secrets_and_exempt_keys_out(run_streaming, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    logs = tmp_path / "logs"
    run_streaming("--config", str(_streaming_yaml(tmp_path / "in.yaml", tmp_path, log_dir=True)), logs=logs)
    text = _the_replay(logs).read_text()
    held = yaml.safe_load(text)
    present = [
        f"{section}.{key}" if section else key
        for section, key in EXEMPT_KEYS
        if key in (held.get(section, {}) if section else held)
    ]
    assert present == [], f"the streaming replay file holds the exempt keys {present}"
    assert TOKEN not in text, "the streaming replay file holds the token string"


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["sequential", "streaming"])
def test_replay_written_for_dry_run(run_main, run_streaming, tmp_path, mode):
    logs = tmp_path / "logs"
    if mode == "streaming":
        yaml_file = _streaming_yaml(tmp_path / "in.yaml", tmp_path, log_dir=True)
        found = run_streaming("--config", str(yaml_file), "--dry-run", logs=logs).replays_at_start[0]
    else:
        found = run_main("--outputs-path", str(tmp_path / "out"), "--log-dir", str(logs), "--dry-run", logs=logs)
        found = found.replays_at_start[0]
    assert len(found) == 1, f"the --dry-run {mode} run started with {[p.name for p in found]} in {logs}"


# ---- A replay file that cannot be written only warns ----


@pytest.mark.unit
def test_unwritable_replay_file_warns_and_run_goes_on(run_main, tmp_path, caplog):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, so no folder can be created under it")
    logs = blocker / "logs"
    with pytest.raises(OSError) as refused:
        logs.mkdir()
    with caplog.at_level(logging.WARNING):
        run = run_main("--outputs-path", str(tmp_path / "out"), "--log-dir", str(logs), logs=logs)
    warned = [r for r in caplog.records if r.levelno == logging.WARNING and str(logs) in r.getMessage()]
    assert len(warned) == 1, f"expected one warning naming {logs}, found {[r.getMessage() for r in warned]}"
    assert refused.value.strerror in warned[0].getMessage(), f"the warning does not name the error: {warned[0]}"
    assert run.configs[0].log_dir == str(logs), "the run did not go on with its configuration"


@pytest.mark.unit
def test_replay_files_of_tests_stay_outside_the_repo(monkeypatch):
    repo = Path(__file__).resolve().parent.parent
    monkeypatch.chdir(repo)
    written = Path(write_replay_config(PipelineConfig()))
    assert written.is_file(), f"no replay file at {written}"
    assert not written.is_relative_to(repo), f"the replay file {written} sits in the repository {repo}"
    assert not (repo / "outputs" / "logs").exists(), "a test left outputs/logs in the repository"
