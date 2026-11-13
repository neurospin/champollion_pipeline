"""main.py chain from stage 1 to stage 2 with one configuration (TASK-INPUTKIND-BDRABCZUK-E27CB8AD548C).

Stage 1 writes under the Morphologist output root, dataset.datasets_root, which puts the subject folders in
<datasets_root>/derivatives/morphologist-6.0. dataset.morphologist_graphs is only the explicit override of that
subjects folder. Stage 2, the stage 2 pre-check, stage 6 and the streaming mode read the subjects folder.

REQ-INPUTKIND-BDRABCZUK-98B5DA679E96: stage 1 output argument is dataset.datasets_root.
REQ-INPUTKIND-BDRABCZUK-50DC84776128: stage 1 validate refuses an empty dataset.datasets_root.
REQ-INPUTKIND-BDRABCZUK-2ACE3A2B3EDA: stage 1 validate refuses an override other than the folder stage 1 writes.
REQ-INPUTKIND-BDRABCZUK-4C1D9F2C4445: stage 2 input defaults to <datasets_root>/derivatives/morphologist-6.0.
REQ-INPUTKIND-BDRABCZUK-3D5F2E88CE1D: stage 2 input is dataset.morphologist_graphs when set.
REQ-INPUTKIND-BDRABCZUK-6E0547C94160: stage 2 validate refuses a missing stage 1 subjects folder.
REQ-INPUTKIND-BDRABCZUK-7F13AA4863B9: stage 2 validate refuses empty morphologist_graphs and datasets_root.
REQ-INPUTKIND-BDRABCZUK-E4D60854FB42: stage 2 pre-check folder defaults to the stage 1 subjects folder.
REQ-STREAMING-BDRABCZUK-60540738F557: the sequential Pre-check record follows subject_input_dir.
REQ-INPUTKIND-BDRABCZUK-5B961BF6390F: stage 6 --morphologist_dir defaults to the stage 1 subjects folder.
REQ-INPUTKIND-BDRABCZUK-F988DC1B4B61: streaming subjects_dir defaults to the stage 1 subjects folder.
REQ-INPUTKIND-BDRABCZUK-22990488E3F5: streaming subjects_dir is dataset.morphologist_graphs when set.
REQ-INPUTKIND-BDRABCZUK-A2B45B8848EC (invariant, regression of DEF-INPUTKIND-BDRABCZUK-9555EBD838DB): one config
chains stage 1 into stage 2.

The stage scripts are replaced by MagicMock classes in their own modules. main() runs in a temporary working
directory with the update check and the process setup stubbed.
"""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import champollion_pipeline.streaming.runner as streaming_runner
from champollion_pipeline.streaming import StreamingSettings
from tests.log_assertions import OUTPUT
from tests.main_stage_helpers import (
    capture_stage_argv,
    enter_work_dir,
    fake_script,
    load_main_module,
    option_values,
    record_logged_before_run,
    stub_main_setup,
    tiles_stage_argv,
)

MORPHOLOGIST_SCRIPT = "champollion_pipeline.generate_morphologist_graphs.GenerateMorphologistGraphs"
TILES_SCRIPT = "champollion_pipeline.run_cortical_tiles.RunCorticalTiles"
SNAPSHOTS_SCRIPT = "champollion_pipeline.generate_snapshots.GenerateSnapshots"
STAGE1_SUBJECTS = Path("derivatives", "morphologist-6.0")
GRAPH_FOLDER = Path("t1mri", "default_acquisition", "default_analysis", "folds", "3.1")
PRE_CHECK_PREFIX = "Pre-check run_cortical_tiles"
LOGGER_NAME = "test_main_morphologist_chain"


@pytest.fixture
def work_dir(monkeypatch, tmp_path) -> Path:
    return enter_work_dir(monkeypatch, tmp_path)


@pytest.fixture
def main_module(monkeypatch, work_dir):
    return stub_main_setup(load_main_module("_champollion_main_morphologist_chain_under_test"), monkeypatch)


def _config(main_module, tmp_path: Path, morphologist_graphs: str = ""):
    """PipelineConfig whose paths live under tmp_path; raw T1 input, no subjects folder override."""
    config = main_module.PipelineConfig(
        root_path=str(tmp_path),
        data_path=str(tmp_path / "data"),
        models_path=str(tmp_path / "models"),
        outputs_path=str(tmp_path / "outputs"),
    )
    dataset = config.dataset
    dataset.input_path = str(tmp_path / "raw")  # noqa: V101
    dataset.datasets_root = str(tmp_path / "data" / "DS")  # noqa: V101
    dataset.morphologist_graphs = morphologist_graphs  # noqa: V101
    dataset.cortical_tiles_output = str(tmp_path / "tiles")  # noqa: V101
    dataset.crops_path = str(tmp_path / "tiles" / "crops")  # noqa: V101
    dataset.hf_enabled = False  # noqa: V101
    config.log_to_file = False  # noqa: V101
    Path(dataset.input_path).mkdir(parents=True)
    (Path(dataset.input_path) / "sub-01.nii.gz").write_bytes(b"synthetic")
    return config


def _write_stage1_tree(output_root: str) -> Path:
    """Write what stage 1 leaves under output_root: one subject with graphs, the qc folder and input_qc.tsv."""
    subjects = Path(output_root) / STAGE1_SUBJECTS
    graphs = subjects / "sub-01" / GRAPH_FOLDER
    graphs.mkdir(parents=True)
    for side in ("L", "R"):
        (graphs / f"{side}sub-01.arg").write_text("synthetic")
    (subjects / "qc").mkdir()
    (subjects / "input_qc.tsv").write_text("participant_id\tqc\treason\nsub-01\t1\t\n")
    return subjects


def _stage(stage_class, stage_name: str, config):
    return stage_class(stage_name, config, logging.getLogger(LOGGER_NAME))


def _stage1_argv(main_module, config) -> list[str]:
    return capture_stage_argv(
        main_module.GenerateMorphologistGraphsStage,
        MORPHOLOGIST_SCRIPT,
        "generate_morphologist_graphs",
        config,
        LOGGER_NAME,
    )


def _run_main(module, monkeypatch, *args: str) -> int:
    monkeypatch.setattr("sys.argv", ["main.py", *args])
    return module.main()


# ====================== Stage 1: output root ======================


@pytest.mark.unit
@pytest.mark.parametrize("morphologist_graphs", ["", "/elsewhere/graphs"], ids=["no_override", "override"])
def test_stage1_output_is_datasets_root(main_module, tmp_path, morphologist_graphs):
    config = _config(main_module, tmp_path, morphologist_graphs)
    argv = _stage1_argv(main_module, config)
    assert argv[1] == config.dataset.datasets_root, f"stage 1 output argument {argv[1]!r}"


@pytest.mark.unit
def test_stage1_validate_refuses_empty_datasets_root(main_module, tmp_path):
    config = _config(main_module, tmp_path)
    config.dataset.datasets_root = ""  # noqa: V101
    stage = _stage(main_module.GenerateMorphologistGraphsStage, "generate_morphologist_graphs", config)
    assert stage.validate() is False


@pytest.mark.unit
def test_stage1_validate_refuses_other_subjects_folder(main_module, tmp_path):
    other = tmp_path / "other" / "graphs"
    other.mkdir(parents=True)
    config = _config(main_module, tmp_path, str(other))
    stage = _stage(main_module.GenerateMorphologistGraphsStage, "generate_morphologist_graphs", config)
    assert stage.validate() is False


# ====================== Stage 2: subjects folder ======================


@pytest.mark.unit
def test_stage2_input_defaults_to_stage1_subjects_folder(main_module, tmp_path):
    config = _config(main_module, tmp_path)
    argv = tiles_stage_argv(main_module, config, LOGGER_NAME)
    assert argv[0] == str(Path(config.dataset.datasets_root) / STAGE1_SUBJECTS)


@pytest.mark.unit
def test_stage2_input_is_subjects_folder_override(main_module, tmp_path):
    config = _config(main_module, tmp_path, "/explicit/subjects")
    argv = tiles_stage_argv(main_module, config, LOGGER_NAME)
    assert argv[0] == "/explicit/subjects"


@pytest.mark.unit
def test_stage2_validate_refuses_missing_stage1_subjects_folder(main_module, tmp_path):
    config = _config(main_module, tmp_path)
    Path(config.dataset.datasets_root).mkdir(parents=True)
    stage = _stage(main_module.RunCorticalTilesStage, "run_cortical_tiles", config)
    assert stage.validate() is False


@pytest.mark.unit
def test_stage2_validate_refuses_empty_paths(main_module, tmp_path, work_dir):
    # A subjects folder relative to the working directory must not stand in for an empty datasets_root.
    (work_dir / STAGE1_SUBJECTS).mkdir(parents=True)
    config = _config(main_module, tmp_path)
    config.dataset.datasets_root = ""  # noqa: V101
    stage = _stage(main_module.RunCorticalTilesStage, "run_cortical_tiles", config)
    assert stage.validate() is False


@pytest.mark.unit
def test_stage2_pre_check_folder_defaults_to_stage1_subjects_folder(main_module, tmp_path):
    config = _config(main_module, tmp_path)
    subjects = _write_stage1_tree(config.dataset.datasets_root)
    stage = _stage(main_module.RunCorticalTilesStage, "run_cortical_tiles", config)
    assert stage.subject_input_dir() == str(subjects)


@pytest.mark.unit
def test_pre_check_on_stage1_subjects_folder(main_module, monkeypatch, tmp_path, pipeline_caplog):
    config = _config(main_module, tmp_path)
    _write_stage1_tree(config.dataset.datasets_root)
    monkeypatch.setattr(main_module, "create_default_config", lambda: config)
    run_tiles, seen_before_run = record_logged_before_run(pipeline_caplog, OUTPUT, PRE_CHECK_PREFIX)

    with patch("champollion_pipeline.run_cortical_tiles.RunCorticalTiles.run", side_effect=run_tiles):
        _run_main(main_module, monkeypatch, "--stages", "run_cortical_tiles")

    assert seen_before_run == [True], f"no OUTPUT record starting with {PRE_CHECK_PREFIX!r} before the stage script ran"


# ====================== Stage 6: Morphologist folder ======================


@pytest.mark.unit
def test_snapshots_morphologist_dir_defaults_to_stage1_subjects_folder(main_module, tmp_path):
    config = _config(main_module, tmp_path)
    argv = capture_stage_argv(
        main_module.GenerateSnapshotsStage, SNAPSHOTS_SCRIPT, "generate_snapshots", config, LOGGER_NAME
    )
    expected = str(Path(config.dataset.datasets_root) / STAGE1_SUBJECTS)
    assert option_values(argv, "--morphologist_dir") == [expected]


# ====================== Streaming: subjects folder ======================


def _streaming_settings(main_module, monkeypatch, config) -> StreamingSettings:
    """Run main() in streaming mode with run_parallel_pipeline recorded; return the StreamingSettings it got."""
    config.mode = "streaming"
    config.dataset.embeddings_only = True
    monkeypatch.setattr(main_module, "create_default_config", lambda: config)
    recorder = MagicMock(return_value=0)
    monkeypatch.setattr(streaming_runner, "run_parallel_pipeline", recorder)
    _run_main(main_module, monkeypatch, "--mode", "streaming")
    assert recorder.call_count == 1, f"expected one run_parallel_pipeline call, got {recorder.call_count}"
    args, kwargs = recorder.call_args
    settings = kwargs.get("settings", args[0] if args else None)
    assert isinstance(settings, StreamingSettings), type(settings)
    return settings


@pytest.mark.unit
def test_streaming_subjects_dir_defaults_to_stage1_subjects_folder(main_module, monkeypatch, tmp_path):
    config = _config(main_module, tmp_path)
    settings = _streaming_settings(main_module, monkeypatch, config)
    assert settings.subjects_dir == str(Path(config.dataset.datasets_root) / STAGE1_SUBJECTS)


@pytest.mark.unit
def test_streaming_subjects_dir_is_subjects_folder_override(main_module, monkeypatch, tmp_path):
    config = _config(main_module, tmp_path, str(tmp_path / "explicit" / "subjects"))
    settings = _streaming_settings(main_module, monkeypatch, config)
    assert settings.subjects_dir == config.dataset.morphologist_graphs


# ====================== Regression of DEF-INPUTKIND-BDRABCZUK-9555EBD838DB ======================


@pytest.mark.unit
def test_one_config_chains_stage1_into_stage2(main_module, monkeypatch, tmp_path):
    config = _config(main_module, tmp_path)
    monkeypatch.setattr(main_module, "create_default_config", lambda: config)
    argv: dict[str, list[str]] = {}

    morphologist = fake_script(argv, "morphologist", on_run=lambda args: _write_stage1_tree(args[1]))
    tiles = fake_script(argv, "tiles")
    with patch(MORPHOLOGIST_SCRIPT, morphologist), patch(TILES_SCRIPT, tiles):
        _run_main(main_module, monkeypatch, "--stages", "generate_morphologist_graphs", "run_cortical_tiles")

    assert "morphologist" in argv, "stage 1 did not run"
    assert "tiles" in argv, "stage 2 did not run after stage 1"
    assert argv["tiles"][0] == os.path.join(argv["morphologist"][1], *STAGE1_SUBJECTS.parts)
