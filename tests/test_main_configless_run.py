"""main.py config-less nominal run: the input path alone runs the pipeline (TASK-INPUTKIND-BDRABCZUK-FDBA616D19AD).

With no --config, main.py reads the kind of the input path, infers every path from it with infer_paths, and runs the
stages from the first applicable one to the last: stages 1 to 6 for T1 files, stages 2 to 6 for Morphologist outputs.
The dataset root comes from OUTPUT, --output-path, --datasets-root or inference only (user answer 2026-11-16, Q2), and
the dataset is named after the dataset root folder unless --dataset-name is given (Q1).

Requirements (wording in the ledger):
    test_t1_input_runs_stages_1_to_6: Config-less T1 run runs stages 1 to 6.
    test_morphologist_input_runs_stages_2_to_6: Config-less Morphologist run runs stages 2 to 6.
    test_stages_narrows_configless_run: --stages narrows a config-less run.
    test_t1_run_uses_inferred_config: Config-less T1 run uses the inferred paths.
    test_morphologist_run_uses_inferred_config: Config-less Morphologist run uses the inferred paths.
    test_outputs_path_under_dataset_root: Config-less run outputs_path under the dataset root.
    test_dataset_name_is_root_folder_name: Config-less run names the dataset after the dataset root folder.
    test_input_path_root_is_inferred: Config-less run with an input path takes the inferred dataset root.

main() runs in a temporary working directory with the update check, the process setup and profiling stubbed. The
real PipelineOrchestrator orders the stages; its _run_stage is replaced by a recorder, so no stage executes, and its
logger writes no file. The observables are the PipelineConfig handed to the orchestrator and the stages it runs.
"""

import dataclasses
import logging
import os
from pathlib import Path

import pytest

from champollion_pipeline.config import PipelineConfig
from champollion_pipeline.path_inference import infer_paths
from champollion_pipeline.utils.input_kind import InputKind, detect_input_kind
from tests.main_stage_helpers import enter_work_dir, load_main_module, stub_main_setup

MAIN = load_main_module("_champollion_main_configless_run_under_test")

ALL_STAGES = [
    "generate_morphologist_graphs",
    "run_cortical_tiles",
    "generate_champollion_config",
    "generate_embeddings",
    "put_together_embeddings",
    "generate_snapshots",
]
GRAPH_FOLDER = ("t1mri", "default_acquisition", "default_analysis", "folds", "3.1")
STAGE1_SUBJECTS = ("derivatives", "morphologist-6.0")
CHAMPOLLION_OUTPUTS = ("derivatives", "champollion_V1")
# Every field infer_paths fills: the dataset fields, then outputs_path of the PipelineConfig.
INFERRED_DATASET_FIELDS = (
    "datasets_root",
    "morphologist_graphs",
    "cortical_tiles_output",
    "crops_path",
    "embeddings_path",
    "snapshots_path",
    "path_to_graph",
    "path_sk_with_hull",
)
T1 = "t1"
MORPHOLOGIST = "morphologist"
KIND_CASES = pytest.mark.parametrize("kind", [T1, MORPHOLOGIST])


@dataclasses.dataclass
class RunRecord:
    """What one main() run handed on: the orchestrator's PipelineConfig and the stages it ran, in order."""

    configs: list = dataclasses.field(default_factory=list)
    stages: list = dataclasses.field(default_factory=list)


@pytest.fixture
def work_dir(monkeypatch, tmp_path) -> Path:
    return enter_work_dir(monkeypatch, tmp_path)


@pytest.fixture
def run_main(monkeypatch, work_dir):
    """Run main() with the given arguments; return the RunRecord, failing on a refused command line."""
    stub_main_setup(MAIN, monkeypatch)
    monkeypatch.setattr(MAIN, "start_profiling", lambda *a, **k: None)
    monkeypatch.setattr("champollion_utils.script_builder.start_profiling", lambda *a, **k: None)
    record = RunRecord()
    orchestrator = MAIN.PipelineOrchestrator
    original_init = orchestrator.__init__

    def _init(self, config):
        record.configs.append(config)
        original_init(self, config)

    def _run_stage(self, stage_name, *_lists):
        record.stages.append(stage_name)
        return True

    monkeypatch.setattr(orchestrator, "__init__", _init)
    monkeypatch.setattr(orchestrator, "_setup_logger", lambda self: logging.getLogger("test_main_configless_run"))
    monkeypatch.setattr(orchestrator, "_run_stage", _run_stage)

    def _run(*cli_args: str) -> RunRecord:
        monkeypatch.setattr("sys.argv", ["main.py", *cli_args])
        try:
            exit_code = MAIN.main()
        except SystemExit as exc:
            pytest.fail(f"main.py refused the command line {list(cli_args)} with exit status {exc.code}")
        assert exit_code == 0, f"main() returned {exit_code} for {list(cli_args)}"
        assert len(record.configs) == 1, f"main() ran {len(record.configs)} configurations, expected 1"
        return record

    return _run


def _t1_input(tmp_path: Path) -> Path:
    """Folder t1set holding one top-level T1 file."""
    raw = tmp_path / "t1set"
    raw.mkdir()
    (raw / "sub-01.nii.gz").write_bytes(b"synthetic")
    assert detect_input_kind(str(raw)) is InputKind.T1
    return raw


def _morphologist_input(tmp_path: Path) -> Path:
    """Dataset root morphset holding the left and right graphs of one subject under derivatives/morphologist-6.0."""
    root = tmp_path / "morphset"
    graphs = root.joinpath(*STAGE1_SUBJECTS, "sub-01", *GRAPH_FOLDER)
    graphs.mkdir(parents=True)
    for side in ("L", "R"):
        (graphs / f"{side}sub-01.arg").write_text("synthetic")
    assert detect_input_kind(str(root)) is InputKind.MORPHOLOGIST
    return root


def _input_of(kind: str, tmp_path: Path) -> Path:
    return _t1_input(tmp_path) if kind == T1 else _morphologist_input(tmp_path)


def _expected_inferred(input_path: Path, kind: str) -> PipelineConfig:
    """The PipelineConfig infer_paths returns for input_path and kind from a config whose datasets_root is empty."""
    config = PipelineConfig()
    config.dataset.input_path = str(input_path)  # noqa: V101
    assert config.dataset.datasets_root == ""
    return infer_paths(config, str(input_path), InputKind(kind))


def _assert_inferred_paths(ran: PipelineConfig, expected: PipelineConfig) -> None:
    for name in INFERRED_DATASET_FIELDS:
        assert getattr(ran.dataset, name) == getattr(expected.dataset, name), f"dataset.{name}"
    assert ran.outputs_path == expected.outputs_path, "outputs_path"


def test_t1_input_runs_stages_1_to_6(run_main, tmp_path):
    record = run_main(str(_t1_input(tmp_path)))
    assert record.stages == ALL_STAGES


def test_morphologist_input_runs_stages_2_to_6(run_main, tmp_path):
    record = run_main(str(_morphologist_input(tmp_path)))
    assert record.stages == ALL_STAGES[1:]


@KIND_CASES
def test_stages_narrows_configless_run(run_main, tmp_path, kind):
    picked = ["run_cortical_tiles", "generate_embeddings"]
    record = run_main(str(_input_of(kind, tmp_path)), "--stages", *picked)
    left_out = [stage for stage in record.stages if stage not in picked]
    assert left_out == [], f"stages run though --stages leaves them out: {left_out}"


def test_t1_run_uses_inferred_config(run_main, tmp_path):
    input_path = _t1_input(tmp_path)
    record = run_main(str(input_path))
    _assert_inferred_paths(record.configs[0], _expected_inferred(input_path, T1))


def test_morphologist_run_uses_inferred_config(run_main, tmp_path):
    input_path = _morphologist_input(tmp_path)
    record = run_main(str(input_path))
    _assert_inferred_paths(record.configs[0], _expected_inferred(input_path, MORPHOLOGIST))


OUTPUT_CASES = pytest.mark.parametrize("output_given", [False, True], ids=["root_inferred", "root_from_output"])


@OUTPUT_CASES
@KIND_CASES
def test_outputs_path_under_dataset_root(run_main, tmp_path, kind, output_given):
    args = [str(_input_of(kind, tmp_path))]
    if output_given:
        args.append(str(tmp_path / "outset"))
    record = run_main(*args)
    ran = record.configs[0]
    assert ran.dataset.datasets_root, "the run has no dataset root"
    assert ran.outputs_path == os.path.join(ran.dataset.datasets_root, *CHAMPOLLION_OUTPUTS)


@OUTPUT_CASES
@KIND_CASES
def test_dataset_name_is_root_folder_name(run_main, tmp_path, kind, output_given):
    input_path = _input_of(kind, tmp_path)
    args = [str(input_path)]
    expected_root = Path(_expected_inferred(input_path, kind).dataset.datasets_root)
    if output_given:
        expected_root = tmp_path / "outset"
        args.append(str(expected_root))
    record = run_main(*args)
    ran = record.configs[0]
    assert Path(ran.dataset.datasets_root) == expected_root
    assert ran.dataset.name == expected_root.name


@pytest.mark.parametrize("extra", ["none", "dataset_name", "data_path"])
@KIND_CASES
def test_input_path_root_is_inferred(run_main, tmp_path, kind, extra):
    input_path = _input_of(kind, tmp_path)
    args = [str(input_path)]
    if extra == "dataset_name":
        args += ["--dataset-name", "other_name"]
    elif extra == "data_path":
        args += ["--data-path", str(tmp_path / "elsewhere")]
    record = run_main(*args)
    expected = _expected_inferred(input_path, kind).dataset.datasets_root
    assert record.configs[0].dataset.datasets_root == expected
