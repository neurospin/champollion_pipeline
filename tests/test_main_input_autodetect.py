"""main.py input autodetect for the two detected kinds (TASK-INPUTKIND-BDRABCZUK-CBD99FAED57E).

From dataset.input_path alone, main.py reads the input kind with detect_input_kind. A T1 input runs stage 1
(generate_morphologist_graphs) below the input itself, then the later stages on the subjects folder stage 1 writes.
A Morphologist input skips stage 1, logs the skip, and runs stage 2 onward on the folder resolve_subjects_dir
finds. Inputs holding neither kind or both kinds belong to the next Task (TASK-INPUTKIND-BDRABCZUK-5ECD06E2023D).

The Morphologist tree carries the legacy subjects level (derivatives/morphologist-6.0/subjects, as in the IXI tree), so
the resolved subjects folder differs from the folder stage 1 would write under the same root.

Requirements (wording in the ledger):
    test_morphologist_input_skips_stage1: Morphologist input skips stage 1.
    test_morphologist_input_logs_stage1_skip: Stage 1 skip is logged.
    test_morphologist_input_runs_stage2_after_skip: Stage 2 runs after the stage 1 skip.
    test_morphologist_input_stage2_reads_resolved_subjects_folder: Stage 2 reads the resolved subjects folder.
    test_morphologist_input_stage6_reads_resolved_subjects_folder: Stage 6 reads the resolved subjects folder.
    test_t1_input_stage1_output_root_is_input: Stage 1 output root of a T1 input.
    test_t1_input_chains_stage1_into_stage2: A T1 input chains stage 1 into stage 2.

The stage scripts are replaced by MagicMock classes in their own modules. main() runs in a temporary working
directory with the update check and the process setup stubbed; create_default_config returns the test's config.
"""

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.utils.input_kind import InputKind, detect_input_kind, resolve_subjects_dir
from tests.main_stage_helpers import enter_work_dir, fake_script, load_main_module, option_values, stub_main_setup

MORPHOLOGIST_SCRIPT = "champollion_pipeline.generate_morphologist_graphs.GenerateMorphologistGraphs"
TILES_SCRIPT = "champollion_pipeline.run_cortical_tiles.RunCorticalTiles"
SNAPSHOTS_SCRIPT = "champollion_pipeline.generate_snapshots.GenerateSnapshots"
STAGE1 = "generate_morphologist_graphs"
STAGE2 = "run_cortical_tiles"
STAGE6 = "generate_snapshots"
STAGE1_SUBJECTS = ("derivatives", "morphologist-6.0")
LEGACY_SUBJECTS = (*STAGE1_SUBJECTS, "subjects")
GRAPH_FOLDER = ("t1mri", "default_acquisition", "default_analysis", "folds", "3.1")
SKIPPED_WORD = "skipped"
DATASETS_ROOT_CASES = pytest.mark.parametrize("root_given", [False, True], ids=["root_empty", "root_given"])


@pytest.fixture
def work_dir(monkeypatch, tmp_path) -> Path:
    return enter_work_dir(monkeypatch, tmp_path)


@pytest.fixture
def main_module(monkeypatch, work_dir):
    return stub_main_setup(load_main_module("_champollion_main_input_autodetect_under_test"), monkeypatch)


def _write_graphs(subjects: Path, subject: str) -> None:
    """Write the left and right graphs of one subject below the subjects folder ``subjects``."""
    graphs = subjects.joinpath(subject, *GRAPH_FOLDER)
    graphs.mkdir(parents=True)
    for side in ("L", "R"):
        (graphs / f"{side}{subject}.arg").write_text("synthetic")


def _morphologist_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """(dataset root, subjects folder) of a Morphologist tree with the legacy subjects level."""
    root = tmp_path / "ds"
    subjects = root.joinpath(*LEGACY_SUBJECTS)
    _write_graphs(subjects, "sub-01")
    assert detect_input_kind(str(root)) is InputKind.MORPHOLOGIST
    assert os.path.abspath(resolve_subjects_dir(str(root)) or "") == str(subjects)
    return root, subjects


def _t1_dataset(tmp_path: Path) -> Path:
    """Folder holding one top-level T1 file."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "sub-01.nii.gz").write_bytes(b"synthetic")
    assert detect_input_kind(str(raw)) is InputKind.T1
    return raw


def _relative(path: Path) -> str:
    """``path`` relative to the working directory, so the absolute form differs from the given value."""
    return os.path.relpath(path)


def _config(main_module, tmp_path: Path, input_path: str, datasets_root: str = ""):
    """PipelineConfig with dataset.input_path set and every inferable path of the stages left empty."""
    config = main_module.PipelineConfig(
        root_path=str(tmp_path),
        data_path=str(tmp_path / "data"),
        models_path=str(tmp_path / "models"),
        outputs_path=str(tmp_path / "outputs"),
    )
    dataset = config.dataset
    dataset.input_path = input_path  # noqa: V101
    dataset.datasets_root = datasets_root  # noqa: V101
    dataset.morphologist_graphs = ""  # noqa: V101
    dataset.hf_enabled = False  # noqa: V101
    config.log_to_file = False  # noqa: V101
    return config


def _write_stage1_output(args: list[str]) -> None:
    """Write what stage 1 leaves under its output root, the second argument: one subject with graphs."""
    _write_graphs(Path(args[1]).joinpath(*STAGE1_SUBJECTS), "sub-01")


def _run_main(module, monkeypatch, config, *stages: str) -> None:
    """Run main() on ``config`` with only ``stages`` enabled."""
    monkeypatch.setattr(module, "create_default_config", lambda: config)
    monkeypatch.setattr("sys.argv", ["main.py", "--stages", *stages])
    module.main()


def _run_stages(main_module, monkeypatch, config, *stages: str) -> dict[str, list[str]]:
    """Run main() with the stage 1, 2 and 6 scripts mocked; return the argv of each script that was built."""
    argv: dict[str, list[str]] = {}
    morphologist = fake_script(argv, STAGE1, on_run=_write_stage1_output)
    tiles = fake_script(argv, STAGE2)
    snapshots = fake_script(argv, STAGE6)
    with patch(MORPHOLOGIST_SCRIPT, morphologist), patch(TILES_SCRIPT, tiles), patch(SNAPSHOTS_SCRIPT, snapshots):
        _run_main(main_module, monkeypatch, config, *stages)
    return argv


# ====================== Morphologist input: stage 1 skipped ======================


@pytest.mark.unit
def test_morphologist_input_skips_stage1(main_module, monkeypatch, tmp_path):
    root, _ = _morphologist_dataset(tmp_path)
    # A set datasets_root lets stage 1 pass its own validation: only the input kind may keep it from running.
    config = _config(main_module, tmp_path, str(root), datasets_root=str(root))
    argv = _run_stages(main_module, monkeypatch, config, STAGE1)
    assert STAGE1 not in argv, f"stage 1 ran on a Morphologist input with argv {argv.get(STAGE1)}"


@pytest.mark.unit
def test_morphologist_input_logs_stage1_skip(main_module, monkeypatch, tmp_path, pipeline_caplog):
    root, _ = _morphologist_dataset(tmp_path)
    config = _config(main_module, tmp_path, str(root), datasets_root=str(root))
    _run_stages(main_module, monkeypatch, config, STAGE1)
    skip_records = [
        record
        for record in pipeline_caplog.records
        if record.levelno >= logging.INFO
        and STAGE1 in record.getMessage()
        and str(root) in record.getMessage()
        and SKIPPED_WORD in record.getMessage().lower()
    ]
    assert skip_records, (
        f"no record at INFO or above naming {STAGE1}, {root} and the word {SKIPPED_WORD!r}; got "
        f"{[record.getMessage() for record in pipeline_caplog.records if record.levelno >= logging.INFO]}"
    )


@pytest.mark.unit
def test_morphologist_input_runs_stage2_after_skip(main_module, monkeypatch, tmp_path):
    root, _ = _morphologist_dataset(tmp_path)
    config = _config(main_module, tmp_path, str(root))
    argv = _run_stages(main_module, monkeypatch, config, STAGE1, STAGE2)
    assert STAGE2 in argv, "stage 2 did not run after stage 1 was skipped for a Morphologist input"


# ====================== Morphologist input: resolved subjects folder ======================


@pytest.mark.unit
@DATASETS_ROOT_CASES
def test_morphologist_input_stage2_reads_resolved_subjects_folder(main_module, monkeypatch, tmp_path, root_given):
    root, subjects = _morphologist_dataset(tmp_path)
    config = _config(main_module, tmp_path, _relative(root), datasets_root=str(root) if root_given else "")
    argv = _run_stages(main_module, monkeypatch, config, STAGE2)
    assert STAGE2 in argv, "stage 2 did not run on a Morphologist input"
    assert argv[STAGE2][0] == str(subjects), f"stage 2 subjects folder {argv[STAGE2][0]!r}"


@pytest.mark.unit
@DATASETS_ROOT_CASES
def test_morphologist_input_stage6_reads_resolved_subjects_folder(main_module, monkeypatch, tmp_path, root_given):
    root, subjects = _morphologist_dataset(tmp_path)
    config = _config(main_module, tmp_path, _relative(root), datasets_root=str(root) if root_given else "")
    config.dataset.snapshots_path = str(tmp_path / "snapshots")  # noqa: V101
    argv = _run_stages(main_module, monkeypatch, config, STAGE6)
    assert STAGE6 in argv, "stage 6 did not run on a Morphologist input"
    assert option_values(argv[STAGE6], "--morphologist_dir") == [str(subjects)]


# ====================== T1 input: stage 1 below the input, then stage 2 ======================


@pytest.mark.unit
def test_t1_input_stage1_output_root_is_input(main_module, monkeypatch, tmp_path):
    raw = _t1_dataset(tmp_path)
    config = _config(main_module, tmp_path, _relative(raw))
    argv = _run_stages(main_module, monkeypatch, config, STAGE1)
    assert STAGE1 in argv, "stage 1 did not run on a T1 input given alone"
    assert argv[STAGE1][1] == str(raw), f"stage 1 output root {argv[STAGE1][1]!r}"


@pytest.mark.unit
def test_t1_input_chains_stage1_into_stage2(main_module, monkeypatch, tmp_path):
    raw = _t1_dataset(tmp_path)
    config = _config(main_module, tmp_path, _relative(raw))
    argv = _run_stages(main_module, monkeypatch, config, STAGE1, STAGE2)
    assert STAGE1 in argv, "stage 1 did not run on a T1 input given alone"
    assert STAGE2 in argv, "stage 2 did not run after stage 1"
    assert argv[STAGE2][0] == str(raw.joinpath(*STAGE1_SUBJECTS)), f"stage 2 subjects folder {argv[STAGE2][0]!r}"
