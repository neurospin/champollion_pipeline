"""main.py on an input path holding neither kind, both kinds, T1s below sourcedata or two Morphologist outputs.

TASK-INPUTKIND-BDRABCZUK-5ECD06E2023D. Neither kind stops main.py with exit status 1 and a clear error. Both kinds and
T1 files below a sourcedata folder ask which kind on a terminal and stop with exit status 1 and a clear error without
one. Two Morphologist subjects folders (AmbiguousSubjectsFolders) ask which folder on a terminal and stop without one.
The DatasetConfig field input_kind (t1 or morphologist; --input-kind through the config argument registry) answers the
kind question in advance; the prompt answer is kept in that field (the subjects folder answer in morphologist_graphs),
so a replay config written by ConfigLoader.save_to_yaml carries it. A set input_kind with an empty input_path, an
unknown input_kind under --generate-config, and input_kind morphologist on a subject folder with no subjects folder stop
main.py with exit status 1 (user answers of 2026-11-16).

Terminal: standard input and standard output both report isatty() True. The tests replace sys.stdin, sys.stdout and
sys.stderr by fakes, never a real prompt: the fake standard input hands out the planned answers line by line (an empty
line read means end of input) and counts its reads. builtins.input falls back to these streams because they have no
file descriptor. The prompt text is what main.py wrote to the fake standard output and standard error; print_args is
silenced so the argument dump does not count as prompt text.

Requirements: the ledger records one Requirement per test below (alm links on the Task root).

The stage scripts are replaced by MagicMock classes in their own modules. main() runs in a temporary working
directory with the update check and the process setup stubbed; create_default_config returns the test's config.
"""

import dataclasses
import io
import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from champollion_pipeline.config import ConfigLoader, DatasetConfig, PipelineConfig
from champollion_pipeline.utils.input_kind import (
    AmbiguousSubjectsFolders,
    InputKind,
    detect_input_kind,
    resolve_subjects_dir,
    scan_input,
    sourcedata_t1_files,
)
from tests.main_stage_helpers import enter_work_dir, fake_script, load_main_module, stub_main_setup

MORPHOLOGIST_SCRIPT = "champollion_pipeline.generate_morphologist_graphs.GenerateMorphologistGraphs"
TILES_SCRIPT = "champollion_pipeline.run_cortical_tiles.RunCorticalTiles"
STAGE1 = "generate_morphologist_graphs"
STAGE2 = "run_cortical_tiles"
GRAPH_FOLDER = ("t1mri", "default_acquisition", "default_analysis", "folds", "3.1")
MORPHOLOGIST_6 = ("derivatives", "morphologist-6.0")
MORPHOLOGIST_2021 = ("derivatives", "morphologist-2021")
SOURCEDATA_T1 = ("sourcedata", "sub-01", "anat", "sub-01_T1w.nii.gz")
INPUT_KIND_OPTION = "--input-kind"
KINDS = ("t1", "morphologist")
UNKNOWN_KIND = "dicom"
UNLISTED_FOLDER = "/no/such/subjects/folder"
SINGLE_SUBJECT = "subject01"
NO_SUBJECTS_FOLDER = "no subjects folder found below"
GENERATED_CONFIG = "generated.yaml"
BATCH_CASES = pytest.mark.parametrize("ttys", [(False, True), (True, False)], ids=["stdin_not_tty", "stdout_not_tty"])


# ====================== Fake terminal ======================


class FakeStdin:
    """Standard input stand-in: its isatty() answer, the planned answers line by line, then end of input."""

    def __init__(self, tty: bool, answers: tuple[str, ...]) -> None:
        self._tty = tty
        self._answers = list(answers)
        self.reads = 0

    def isatty(self) -> bool:
        return self._tty

    def fileno(self) -> int:
        raise io.UnsupportedOperation("fake standard input has no file descriptor")

    def readline(self, *_args) -> str:
        self.reads += 1
        return f"{self._answers.pop(0)}\n" if self._answers else ""

    def read(self, *_args) -> str:
        self.reads += 1
        return "".join(f"{answer}\n" for answer in self._answers)


class FakeOutput(io.StringIO):
    """Standard output or error stand-in recording the text written, with its isatty() answer."""

    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@dataclasses.dataclass
class Terminal:
    """The three fake streams of one test."""

    stdin: FakeStdin
    stdout: FakeOutput
    stderr: FakeOutput

    def written(self) -> str:
        """Text main.py wrote to standard output and standard error."""
        return self.stdout.getvalue() + self.stderr.getvalue()


def _streams(monkeypatch, stdin_tty: bool, stdout_tty: bool, answers: tuple[str, ...] = ()) -> Terminal:
    """Install fake standard streams; a terminal when both stdin_tty and stdout_tty are True."""
    terminal = Terminal(FakeStdin(stdin_tty, answers), FakeOutput(stdout_tty), FakeOutput(False))
    monkeypatch.setattr("sys.stdin", terminal.stdin)
    monkeypatch.setattr("sys.stdout", terminal.stdout)
    monkeypatch.setattr("sys.stderr", terminal.stderr)
    return terminal


def _terminal(monkeypatch, *answers: str) -> Terminal:
    return _streams(monkeypatch, True, True, answers)


def _batch(monkeypatch) -> Terminal:
    return _streams(monkeypatch, False, True)


# ====================== Datasets ======================


def _write_graphs(subjects: Path, subject: str = "sub-01") -> None:
    """Write the left and right graphs of one subject below the subjects folder ``subjects``."""
    graphs = subjects.joinpath(subject, *GRAPH_FOLDER)
    graphs.mkdir(parents=True)
    for side in ("L", "R"):
        (graphs / f"{side}{subject}.arg").write_text("synthetic")


def _write_t1(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic")


def _both_dataset(tmp_path: Path) -> Path:
    """Dataset root holding a top-level T1 file and a Morphologist tree: both kinds."""
    root = tmp_path / "both"
    _write_graphs(root.joinpath(*MORPHOLOGIST_6))
    _write_t1(root / "sub-01.nii.gz")
    assert detect_input_kind(str(root)) is InputKind.BOTH
    return root


def _sourcedata_dataset(tmp_path: Path) -> Path:
    """Dataset root whose only T1 file sits below sourcedata: a t1 input with sourcedata T1 files."""
    root = tmp_path / "bids"
    _write_t1(root.joinpath(*SOURCEDATA_T1))
    assert detect_input_kind(str(root)) is InputKind.T1
    assert sourcedata_t1_files(str(root)) == [os.path.join(*SOURCEDATA_T1)]
    return root


def _t1_dataset(tmp_path: Path) -> Path:
    """Folder holding one top-level T1 file and no graph."""
    raw = tmp_path / "raw"
    _write_t1(raw / "sub-01.nii.gz")
    assert detect_input_kind(str(raw)) is InputKind.T1
    return raw


def _morphologist_dataset(tmp_path: Path) -> Path:
    """Dataset root holding one Morphologist tree and no T1 file."""
    root = tmp_path / "morpho"
    _write_graphs(root.joinpath(*MORPHOLOGIST_6))
    assert detect_input_kind(str(root)) is InputKind.MORPHOLOGIST
    return root


def _empty_dataset(tmp_path: Path) -> Path:
    """Folder holding neither kind."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert detect_input_kind(str(empty)) is InputKind.NEITHER
    return empty


def _two_outputs_dataset(tmp_path: Path) -> Path:
    """Dataset root holding two Morphologist outputs, morphologist-2021 and morphologist-6.0."""
    root = tmp_path / "two"
    _write_graphs(root.joinpath(*MORPHOLOGIST_2021))
    _write_graphs(root.joinpath(*MORPHOLOGIST_6))
    assert detect_input_kind(str(root)) is InputKind.MORPHOLOGIST
    return root


def _candidates(root: Path) -> tuple[str, ...]:
    """The candidates AmbiguousSubjectsFolders reports for ``root``."""
    with pytest.raises(AmbiguousSubjectsFolders) as raised:
        resolve_subjects_dir(str(root))
    assert len(raised.value.candidates) == 2
    return raised.value.candidates


def _single_subject_dataset(tmp_path: Path, t1_first: bool) -> Path:
    """One subject folder: its graphs in its own t1mri folder, and with t1_first a T1 file sorting before t1mri."""
    _write_graphs(tmp_path, SINGLE_SUBJECT)
    subject = tmp_path / SINGLE_SUBJECT
    if t1_first:
        _write_t1(subject / "raw.nii.gz")
    scan = scan_input(str(subject))
    assert (scan.first_graph is None) is t1_first
    assert resolve_subjects_dir(str(subject)) is None
    return subject


DATASETS = {
    "both": _both_dataset,
    "sourcedata": _sourcedata_dataset,
    "t1": _t1_dataset,
    "morphologist": _morphologist_dataset,
    "neither": _empty_dataset,
    "two_outputs": _two_outputs_dataset,
}


# ====================== Running main() ======================


@pytest.fixture
def work_dir(monkeypatch, tmp_path) -> Path:
    return enter_work_dir(monkeypatch, tmp_path)


@pytest.fixture
def main_module(monkeypatch, work_dir):
    module = stub_main_setup(load_main_module("_champollion_main_input_autodetect_ambiguous_under_test"), monkeypatch)
    monkeypatch.setattr(module.ChampollionPipeline, "print_args", lambda _self: None)
    return module


def _config(main_module, tmp_path: Path, input_path: Path | str, datasets_root: str = ""):
    """PipelineConfig with dataset.input_path set and every inferable path of the stages left empty."""
    config = main_module.PipelineConfig(
        root_path=str(tmp_path),
        data_path=str(tmp_path / "data"),
        models_path=str(tmp_path / "models"),
        outputs_path=str(tmp_path / "outputs"),
    )
    dataset = config.dataset
    dataset.input_path = str(input_path)  # noqa: V101
    dataset.datasets_root = datasets_root  # noqa: V101
    dataset.morphologist_graphs = ""  # noqa: V101
    dataset.hf_enabled = False  # noqa: V101
    config.log_to_file = False  # noqa: V101
    return config


def _stop_config(main_module, tmp_path: Path, input_path: Path | str):
    """_config with datasets_root set, so stage 1 passes its own validation: only main.py's stop may keep it idle."""
    return _config(main_module, tmp_path, input_path, datasets_root=str(tmp_path / "out"))


def _exit_status(main_module, monkeypatch, config, *args: str) -> int | str | None:
    """Run main() on ``config`` with ``args``; return its exit status, from a return or a SystemExit."""
    monkeypatch.setattr(main_module, "create_default_config", lambda: config)
    monkeypatch.setattr("sys.argv", ["main.py", *args])
    try:
        return main_module.main()
    except SystemExit as stop:
        return stop.code


def _run_stages(main_module, monkeypatch, config, *args: str) -> tuple[int | str | None, dict[str, list[str]]]:
    """Run main() with stages 1 and 2 enabled and their scripts mocked; return the exit status and each built argv."""
    argv: dict[str, list[str]] = {}
    morphologist = fake_script(argv, STAGE1)
    tiles = fake_script(argv, STAGE2)
    with patch(MORPHOLOGIST_SCRIPT, morphologist), patch(TILES_SCRIPT, tiles):
        status = _exit_status(main_module, monkeypatch, config, "--stages", STAGE1, STAGE2, *args)
    return status, argv


def _effective_config(main_module, monkeypatch, config, *args: str):
    """Run main() with PipelineOrchestrator replaced; return the configuration main.py ran the pipeline with."""
    seen = []

    class RecordingOrchestrator:
        def __init__(self, run_config) -> None:
            seen.append(run_config)

        def run(self) -> int:
            return 0

    monkeypatch.setattr(main_module, "PipelineOrchestrator", RecordingOrchestrator)
    status = _exit_status(main_module, monkeypatch, config, "--stages", STAGE1, STAGE2, *args)
    assert status == 0, f"main() exit status {status}"
    assert len(seen) == 1, f"the pipeline ran {len(seen)} times"
    return seen[0]


def _error_messages(pipeline_caplog) -> list[str]:
    return [record.getMessage() for record in pipeline_caplog.records if record.levelno >= logging.ERROR]


def _assert_stopped(status, argv) -> None:
    assert status == 1, f"main() exit status {status}, expected 1"
    assert not argv, f"stages ran before the stop: {sorted(argv)}"


def _assert_error_contains(pipeline_caplog, *texts: str) -> None:
    messages = _error_messages(pipeline_caplog)
    assert any(all(text in message for text in texts) for message in messages), (
        f"no ERROR record containing {texts}; ERROR records: {messages}"
    )


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


# ====================== Neither kind ======================


@pytest.mark.unit
def test_neither_kind_exits_1_before_any_stage(main_module, monkeypatch, tmp_path):
    _batch(monkeypatch)
    config = _stop_config(main_module, tmp_path, _empty_dataset(tmp_path))
    _assert_stopped(*_run_stages(main_module, monkeypatch, config))


@pytest.mark.unit
def test_neither_kind_error_names_input_path_and_option(main_module, monkeypatch, tmp_path, pipeline_caplog):
    _batch(monkeypatch)
    empty = _empty_dataset(tmp_path)
    _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, empty))
    _assert_error_contains(pipeline_caplog, str(empty), INPUT_KIND_OPTION)


# ====================== Both kinds and sourcedata without a terminal ======================


def _assert_batch_stops(main_module, monkeypatch, tmp_path, ttys, dataset: Path) -> None:
    """Without a terminal on one stream, main() on the dataset exits 1 before any stage."""
    _streams(monkeypatch, *ttys)
    config = _stop_config(main_module, tmp_path, dataset)
    _assert_stopped(*_run_stages(main_module, monkeypatch, config))


@pytest.mark.unit
@BATCH_CASES
def test_both_kinds_in_batch_exits_1_before_any_stage(main_module, monkeypatch, tmp_path, ttys):
    _assert_batch_stops(main_module, monkeypatch, tmp_path, ttys, _both_dataset(tmp_path))


@pytest.mark.unit
def test_both_kinds_batch_error_names_examples(main_module, monkeypatch, tmp_path, pipeline_caplog):
    _batch(monkeypatch)
    root = _both_dataset(tmp_path)
    scan = scan_input(str(root))
    _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root))
    _assert_error_contains(pipeline_caplog, scan.first_t1, scan.first_graph, INPUT_KIND_OPTION)


@pytest.mark.unit
@BATCH_CASES
def test_sourcedata_in_batch_exits_1_before_any_stage(main_module, monkeypatch, tmp_path, ttys):
    _assert_batch_stops(main_module, monkeypatch, tmp_path, ttys, _sourcedata_dataset(tmp_path))


@pytest.mark.unit
def test_sourcedata_batch_error_names_first_sourcedata_t1(main_module, monkeypatch, tmp_path, pipeline_caplog):
    _batch(monkeypatch)
    root = _sourcedata_dataset(tmp_path)
    _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root))
    _assert_error_contains(pipeline_caplog, sourcedata_t1_files(str(root))[0], INPUT_KIND_OPTION)


@pytest.mark.unit
def test_input_inside_sourcedata_runs_stage1(main_module, monkeypatch, tmp_path):
    _batch(monkeypatch)
    sourcedata = _sourcedata_dataset(tmp_path) / "sourcedata"
    assert sourcedata_t1_files(str(sourcedata)) == []
    status, argv = _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, sourcedata))
    assert STAGE1 in argv, f"stage 1 did not run on an input inside sourcedata (exit status {status})"
    assert argv[STAGE1][1] == str(sourcedata), f"stage 1 output root {argv[STAGE1][1]!r}"


# ====================== The input kind prompt ======================


@pytest.mark.unit
def test_both_kinds_prompt_shows_kinds_and_examples(main_module, monkeypatch, tmp_path):
    terminal = _terminal(monkeypatch, "t1")
    root = _both_dataset(tmp_path)
    scan = scan_input(str(root))
    _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root))
    assert terminal.stdin.reads, "no answer was read on a terminal for an input holding both kinds"
    for text in (*KINDS, scan.first_t1, scan.first_graph):
        assert text in terminal.written(), f"prompt lacks {text!r}: {terminal.written()!r}"


@pytest.mark.unit
def test_sourcedata_prompt_shows_first_sourcedata_t1(main_module, monkeypatch, tmp_path):
    terminal = _terminal(monkeypatch, "t1")
    root = _sourcedata_dataset(tmp_path)
    _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root))
    assert terminal.stdin.reads, "no answer was read on a terminal for sourcedata T1 files"
    for text in (*KINDS, sourcedata_t1_files(str(root))[0]):
        assert text in terminal.written(), f"prompt lacks {text!r}: {terminal.written()!r}"


@pytest.mark.unit
@pytest.mark.parametrize("dataset", ["both", "sourcedata"])
def test_prompt_answer_t1_runs_stage1_on_input(main_module, monkeypatch, tmp_path, dataset):
    terminal = _terminal(monkeypatch, "t1")
    root = DATASETS[dataset](tmp_path)
    status, argv = _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root))
    assert terminal.stdin.reads, f"no answer was read on a terminal for the {dataset} input"
    assert STAGE1 in argv, f"stage 1 did not run after the answer t1 (exit status {status})"
    assert argv[STAGE1][1] == str(root), f"stage 1 output root {argv[STAGE1][1]!r}"


@pytest.mark.unit
def test_prompt_answer_morphologist_skips_stage1(main_module, monkeypatch, tmp_path):
    _terminal(monkeypatch, "morphologist")
    root = _both_dataset(tmp_path)
    # A set datasets_root lets stage 1 pass its own validation: only the answer may keep it from running.
    status, argv = _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root, datasets_root=str(root)))
    assert status == 0, f"main() exit status {status}"
    assert STAGE1 not in argv, f"stage 1 ran after the answer morphologist with argv {argv.get(STAGE1)}"


@pytest.mark.unit
def test_prompt_answer_morphologist_stage2_reads_resolved_folder(main_module, monkeypatch, tmp_path):
    _terminal(monkeypatch, "morphologist")
    root = _both_dataset(tmp_path)
    status, argv = _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root))
    assert STAGE2 in argv, f"stage 2 did not run after the answer morphologist (exit status {status})"
    assert argv[STAGE2][0] == str(root.joinpath(*MORPHOLOGIST_6)), f"stage 2 subjects folder {argv[STAGE2][0]!r}"


@pytest.mark.unit
@pytest.mark.parametrize(("dataset", "answer"), [("both", "t1"), ("both", "morphologist"), ("sourcedata", "t1")])
def test_prompt_answer_is_kept_in_input_kind(main_module, monkeypatch, tmp_path, dataset, answer):
    _terminal(monkeypatch, answer)
    config = _config(main_module, tmp_path, DATASETS[dataset](tmp_path))
    assert _effective_config(main_module, monkeypatch, config).dataset.input_kind == answer


@pytest.mark.unit
@pytest.mark.parametrize(
    ("dataset", "answer"),
    [("both", UNKNOWN_KIND), ("sourcedata", UNKNOWN_KIND), ("two_outputs", UNLISTED_FOLDER)],
    ids=["kind_both", "kind_sourcedata", "subjects_folder"],
)
def test_unlisted_prompt_answer_exits_1(main_module, monkeypatch, tmp_path, dataset, answer):
    terminal = _terminal(monkeypatch, answer)
    config = _stop_config(main_module, tmp_path, DATASETS[dataset](tmp_path))
    status, argv = _run_stages(main_module, monkeypatch, config)
    assert terminal.stdin.reads, "no answer was read on a terminal"
    _assert_stopped(status, argv)


@pytest.mark.unit
@pytest.mark.parametrize("dataset", ["both", "sourcedata", "two_outputs"])
def test_end_of_input_at_prompt_exits_1(main_module, monkeypatch, tmp_path, dataset):
    terminal = _terminal(monkeypatch)
    config = _stop_config(main_module, tmp_path, DATASETS[dataset](tmp_path))
    status, argv = _run_stages(main_module, monkeypatch, config)
    assert terminal.stdin.reads, "no answer was read on a terminal"
    _assert_stopped(status, argv)


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
def test_generate_config_unknown_input_kind_writes_no_file(main_module, monkeypatch, tmp_path, work_dir):
    """dataset.input_kind is set on the default configuration, as --input-kind would set it."""
    _batch(monkeypatch)
    config = _config(main_module, tmp_path, _t1_dataset(tmp_path))
    config.dataset.input_kind = UNKNOWN_KIND  # noqa: V101
    target = work_dir / GENERATED_CONFIG
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


# ====================== Two Morphologist outputs ======================


@pytest.mark.unit
@BATCH_CASES
@pytest.mark.parametrize("kind", [None, "morphologist"], ids=["kind_unset", "kind_morphologist"])
def test_two_subjects_folders_in_batch_exit_1(main_module, monkeypatch, tmp_path, ttys, kind):
    _streams(monkeypatch, *ttys)
    config = _stop_config(main_module, tmp_path, _two_outputs_dataset(tmp_path))
    args = (INPUT_KIND_OPTION, kind) if kind else ()
    _assert_stopped(*_run_stages(main_module, monkeypatch, config, *args))


@pytest.mark.unit
def test_two_subjects_folders_batch_error_names_folders(main_module, monkeypatch, tmp_path, pipeline_caplog):
    _batch(monkeypatch)
    root = _two_outputs_dataset(tmp_path)
    candidates = _candidates(root)
    _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root))
    _assert_error_contains(pipeline_caplog, *candidates)


@pytest.mark.unit
def test_subjects_folder_prompt_lists_folders(main_module, monkeypatch, tmp_path):
    root = _two_outputs_dataset(tmp_path)
    candidates = _candidates(root)
    terminal = _terminal(monkeypatch, candidates[1])
    _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root))
    assert terminal.stdin.reads, "no answer was read on a terminal for two subjects folders"
    for folder in candidates:
        assert folder in terminal.written(), f"prompt lacks {folder!r}: {terminal.written()!r}"


@pytest.mark.unit
@pytest.mark.parametrize("choice", [0, 1], ids=["first_listed", "second_listed"])
def test_subjects_folder_answer_feeds_stage2(main_module, monkeypatch, tmp_path, choice):
    root = _two_outputs_dataset(tmp_path)
    chosen = _candidates(root)[choice]
    _terminal(monkeypatch, chosen)
    status, argv = _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root))
    assert STAGE2 in argv, f"stage 2 did not run after a listed subjects folder was chosen (exit status {status})"
    assert argv[STAGE2][0] == os.path.abspath(chosen), f"stage 2 subjects folder {argv[STAGE2][0]!r}"


@pytest.mark.unit
def test_subjects_folder_answer_is_kept_in_morphologist_graphs(main_module, monkeypatch, tmp_path):
    root = _two_outputs_dataset(tmp_path)
    chosen = _candidates(root)[1]
    _terminal(monkeypatch, chosen)
    effective = _effective_config(main_module, monkeypatch, _config(main_module, tmp_path, root))
    assert effective.dataset.morphologist_graphs == os.path.abspath(chosen)
