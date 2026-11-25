"""Harness of the main.py input kind routing tests: fake terminal, datasets and main() runners.

Terminal: standard input and standard output both report isatty() True. The tests replace sys.stdin, sys.stdout and
sys.stderr by fakes, never a real prompt: the fake standard input hands out the planned answers line by line (an empty
line read means end of input) and counts its reads. builtins.input falls back to these streams because they have no
file descriptor. The prompt text is what main.py wrote to the fake standard output and standard error; print_args is
silenced so the argument dump does not count as prompt text.

The stage scripts are replaced by MagicMock classes in their own modules. main() runs in a temporary working
directory with the update check and the process setup stubbed; create_default_config returns the test's config.

Shared by the three tests/test_main_input_autodetect_*.py topic files.
"""

import dataclasses
import io
import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

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


def _main_module_under_test(monkeypatch, tmp_path):
    """main.py loaded in tmp_path/cwd with the update check, process setup and print_args stubbed."""
    enter_work_dir(monkeypatch, tmp_path)
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
