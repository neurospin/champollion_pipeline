"""Pre- and post-stage index files of stages 3 to 6 in main.py (TASK-INPUTKIND-BDRABCZUK-34AA9297D58C).

Like run_cortical_tiles, the sequential stages generate_champollion_config, generate_embeddings,
put_together_embeddings and generate_snapshots leave index_pre_<stage>.json (the file index of the folder the
stage reads, written before its script runs) and index_post_<stage>.json (the file index of the folder it wrote)
under PipelineConfig.outputs_path. Their pre-check logs no WARNING on a nominal input folder, which holds region
folders or embedding files, not Morphologist subjects.

Each test builds one dataset tree under tmp_path, enables one stage, replaces the stage's validate() and execute()
(the fake execute writes the stage's output files and notes whether the pre index already exists), runs
PipelineOrchestrator.run() and reads the index files back with BTree.from_dict.
"""

import ast
import importlib.util
import json
import logging
from pathlib import Path

import pytest

from champollion_pipeline._stage_base import StageResult
from champollion_pipeline.file_indexer.btree import BTree

pytestmark = pytest.mark.unit  # noqa: V107

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"

DATASET = "DS"
MASKS = "canonical_25"
CORTICAL_TILES_FOLDER = "cortical_tiles-2026"

CONFIG_STAGE = "generate_champollion_config"
EMBEDDINGS_STAGE = "generate_embeddings"
COMBINE_STAGE = "put_together_embeddings"
SNAPSHOTS_STAGE = "generate_snapshots"

# Region crop folders as run_cortical_tiles writes them under crops/<masks>/2mm.
CROP_FILES = [
    "S.C.-sylv./mask/Lskeleton.npy",
    "S.C.-sylv./mask/Lskeleton_subject.csv",
    "S.Or./mask/Rskeleton.npy",
    "S.Or./mask/Rskeleton_subject.csv",
]
# Stage 3 output under the configs root.
CONFIG_FILES = [f"dataset/{DATASET}/SCsylv_left.yaml", f"dataset/{DATASET}/SOr_right.yaml", "input_qc.tsv"]
# Per-region model folders as generate_embeddings writes them.
REGION_EMBEDDING_FILES = ["SCsylv_left/full_embeddings.csv", "SOr_right/full_embeddings.csv", "output_qc.tsv"]
# Combined embeddings as put_together_embeddings writes them.
COMBINED_FILES = ["SCsylv_left.csv", "SOr_right.csv"]
# Snapshots as generate_snapshots writes them.
SNAPSHOT_FILES = ["sub-01/sulcal_graph.png", "umap.png"]


@pytest.fixture
def main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_stage_indexes_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_files(root: Path, relative_paths: list[str]) -> None:
    for relative in relative_paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def _index_keys(path: Path) -> list[str]:
    assert path.is_file(), f"{path.name} not written under outputs_path"
    tree = BTree.from_dict(json.loads(path.read_text()))
    return sorted(key for key, _ in tree)


class _Layout:
    """Folders of one dataset DS under tmp_path, as the pipeline lays them out by default."""

    def __init__(self, tmp_path: Path, main_module):
        self.tmp_path = tmp_path
        self.main_module = main_module
        self.root = tmp_path / DATASET
        derivatives = self.root / "derivatives"
        self.crops = derivatives / CORTICAL_TILES_FOLDER / "crops" / MASKS / "2mm"
        self.configs = derivatives / "champollion_V1" / "configs"
        self.region_embeddings = derivatives / "champollion_V1" / MASKS / "region_embeddings"
        self.combined = derivatives / "champollion_V1" / MASKS / "embeddings"
        self.snapshots = derivatives / "champollion_V1" / MASKS / "snapshots"
        self.outputs = tmp_path / "outputs"


@pytest.fixture
def layout(tmp_path, main_module) -> _Layout:
    return _Layout(tmp_path, main_module)


def _config(main_module, tmp_path: Path, layout: _Layout, stage_name: str):
    """One enabled sequential stage on dataset DS, with every dataset path at its default under DS."""
    config = main_module.PipelineConfig(
        root_path=str(tmp_path),
        data_path=str(tmp_path / "data"),
        models_path=str(tmp_path / "models"),
        outputs_path=str(layout.outputs),
    )
    config.stages = {stage_name: True}
    config.log_to_file = False  # noqa: V101
    config.log_dir = None  # noqa: V101
    dataset = config.dataset
    dataset.name = DATASET  # noqa: V101
    dataset.masks_version = MASKS  # noqa: V101
    dataset.datasets_root = str(layout.root)  # noqa: V101
    dataset.crops_path = str(layout.crops)  # noqa: V101
    return config


class _FakeExecute:
    """Replaces a stage's execute(): notes whether the pre index exists, writes the output files, succeeds."""

    def __init__(self, stage_name: str, pre_index: Path, output_dir: Path, files: list[str]):
        self.stage_name = stage_name
        self.pre_index = pre_index
        self.output_dir = output_dir
        self.files = files
        self.pre_index_seen_before_run: list[bool] = []

    def __call__(self) -> StageResult:
        self.pre_index_seen_before_run.append(self.pre_index.is_file())
        _write_files(self.output_dir, self.files)
        return StageResult(stage_name=self.stage_name, success=True, message="ok")


def _run_stage(main_module, config, stage_name: str, output_dir: Path, output_files: list[str]) -> _FakeExecute:
    """Run the pipeline with only stage_name enabled; its validate() passes and its script is the fake."""
    orchestrator = main_module.PipelineOrchestrator(config)
    stage = orchestrator.stages[stage_name]
    pre_index = Path(config.outputs_path) / f"index_pre_{stage_name}.json"
    fake = _FakeExecute(stage_name, pre_index, output_dir, output_files)
    stage.validate = lambda: True
    stage.execute = fake
    assert orchestrator.run() == 0
    return fake


def _option_value(args: list[str], option: str) -> str:
    """Value of --option=value in a stage argv."""
    values = [arg.split("=", 1)[1] for arg in args if arg.startswith(f"{option}=")]
    assert len(values) == 1, f"{option} passed {len(values)} times in {args}"
    return values[0]


def _warnings(capture) -> list[str]:
    return [record.getMessage() for record in capture.records if record.levelno >= logging.WARNING]


def _assert_both_index_files(layout: _Layout, stage_name: str) -> None:
    assert (layout.outputs / f"index_pre_{stage_name}.json").is_file()
    assert (layout.outputs / f"index_post_{stage_name}.json").is_file()


def _assert_pre_index(layout: _Layout, stage_name: str, fake: _FakeExecute, expected: list[str]) -> None:
    """index_pre existed before the stage script ran and holds exactly the expected keys."""
    assert fake.pre_index_seen_before_run == [True], "index_pre not written before the stage script ran"
    assert _index_keys(layout.outputs / f"index_pre_{stage_name}.json") == sorted(expected)


def _assert_post_index(layout: _Layout, stage_name: str, expected: list[str]) -> None:
    assert _index_keys(layout.outputs / f"index_post_{stage_name}.json") == sorted(expected)


def _assert_pre_check_logs_no_warning(layout: _Layout, stage_name: str, capture) -> None:
    assert (layout.outputs / f"index_pre_{stage_name}.json").is_file(), "the pre-check did not run"
    assert _warnings(capture) == []


def _stage_args(layout: _Layout, stage_name: str) -> list[str]:
    """build_args() of one enabled stage on dataset DS, before any run."""
    config = _config(layout.main_module, layout.tmp_path, layout, stage_name)
    return layout.main_module.PipelineOrchestrator(config).stages[stage_name].build_args()


def _run_on(layout: _Layout, stage_name, inputs, output_dir, output_files) -> _FakeExecute:
    """Fill the stage's input folders, then run the stage with the fake execute."""
    for folder, files in inputs:
        _write_files(folder, files)
    config = _config(layout.main_module, layout.tmp_path, layout, stage_name)
    return _run_stage(layout.main_module, config, stage_name, output_dir, output_files)


def _check_pre_index(layout: _Layout, run, stage_name, passed, expected) -> None:
    """The stage gets folder as option (passed = (option, folder); None: first positional); pre index = expected."""
    option, folder = passed
    args = _stage_args(layout, stage_name)
    assert Path(args[0] if option is None else _option_value(args, option)) == folder
    _assert_pre_index(layout, stage_name, run(layout), expected)


def _check_post_index(layout: _Layout, run, stage_name, passed, expected) -> None:
    """The stage is passed folder as option (passed = (option, folder)); its post index holds expected."""
    option, folder = passed
    args = _stage_args(layout, stage_name)
    assert Path(_option_value(args, option)) == folder
    run(layout)
    _assert_post_index(layout, stage_name, expected)


def _check_both_files(layout: _Layout, run, stage_name) -> None:
    run(layout)
    _assert_both_index_files(layout, stage_name)


def _check_no_warning(layout: _Layout, run, stage_name, capture) -> None:
    run(layout)
    _assert_pre_check_logs_no_warning(layout, stage_name, capture)


# ---------------------------------------------------------------- generate_champollion_config (stage 3)


def _run_config_stage(layout: _Layout) -> _FakeExecute:
    inputs = [(layout.crops, CROP_FILES)]
    return _run_on(layout, CONFIG_STAGE, inputs, layout.configs, CONFIG_FILES)


def test_config_stage_leaves_both_index_files(layout, pipeline_caplog):
    """REQ-INPUTKIND-BDRABCZUK-A2EF3B0685B1."""
    _check_both_files(layout, _run_config_stage, CONFIG_STAGE)


def test_config_pre_index_holds_the_crops_folder(layout, pipeline_caplog):
    """REQ-INPUTKIND-BDRABCZUK-3BD435D0A56F."""
    fake = _run_config_stage(layout)

    _assert_pre_index(layout, CONFIG_STAGE, fake, CROP_FILES)


def test_config_post_index_holds_the_configs_folder(layout, pipeline_caplog):
    """REQ-INPUTKIND-BDRABCZUK-08F63E76CC7C."""
    _run_config_stage(layout)

    _assert_post_index(layout, CONFIG_STAGE, CONFIG_FILES)


def test_config_pre_check_logs_no_warning(layout, pipeline_caplog):
    """REQ-INPUTKIND-BDRABCZUK-CAAB72275F26."""
    _check_no_warning(layout, _run_config_stage, CONFIG_STAGE, pipeline_caplog)


# ---------------------------------------------------------------- generate_embeddings (stage 4)


def _run_embeddings_stage(layout: _Layout, with_configs: bool = True) -> _FakeExecute:
    inputs = [(layout.crops, CROP_FILES)] + ([(layout.configs, CONFIG_FILES)] if with_configs else [])
    return _run_on(layout, EMBEDDINGS_STAGE, inputs, layout.region_embeddings, REGION_EMBEDDING_FILES)


def test_embeddings_stage_leaves_both_index_files(layout, pipeline_caplog):
    """REQ-INPUTKIND-BDRABCZUK-9C2EC230ECBC."""
    _check_both_files(layout, _run_embeddings_stage, EMBEDDINGS_STAGE)


def test_embeddings_pre_index_holds_the_crops_and_configs_folders(layout, pipeline_caplog):
    """REQ-INPUTKIND-BDRABCZUK-100DFAA9B333: one file, keys relative to <datasets_root>/derivatives."""
    fake = _run_embeddings_stage(layout)

    expected = [f"{CORTICAL_TILES_FOLDER}/crops/{MASKS}/2mm/{name}" for name in CROP_FILES]
    expected += [f"champollion_V1/configs/{name}" for name in CONFIG_FILES]
    _assert_pre_index(layout, EMBEDDINGS_STAGE, fake, expected)


def test_embeddings_pre_index_without_configs_folder_holds_the_crops_only(layout, pipeline_caplog):
    """REQ-INPUTKIND-BDRABCZUK-100DFAA9B333: a missing configs folder adds no key; the keys keep the same root."""
    fake = _run_embeddings_stage(layout, with_configs=False)

    expected = [f"{CORTICAL_TILES_FOLDER}/crops/{MASKS}/2mm/{name}" for name in CROP_FILES]
    _assert_pre_index(layout, EMBEDDINGS_STAGE, fake, expected)


def test_embeddings_post_index_holds_the_region_embeddings_folder(layout, pipeline_caplog):
    """REQ-INPUTKIND-BDRABCZUK-2B90433330BD."""
    _check_post_index(
        layout, _run_embeddings_stage, EMBEDDINGS_STAGE, ("--output", layout.region_embeddings), REGION_EMBEDDING_FILES
    )


def test_embeddings_pre_check_logs_no_warning(layout, pipeline_caplog):
    """REQ-INPUTKIND-BDRABCZUK-DBC99C6C8AB8."""
    _check_no_warning(layout, _run_embeddings_stage, EMBEDDINGS_STAGE, pipeline_caplog)


# ---------------------------------------------------------------- put_together_embeddings (stage 5)


def _run_combine_stage(layout: _Layout) -> _FakeExecute:
    inputs = [(layout.region_embeddings, REGION_EMBEDDING_FILES)]
    return _run_on(layout, COMBINE_STAGE, inputs, layout.combined, COMBINED_FILES)


def test_combine_stage_leaves_both_index_files(layout, pipeline_caplog):
    """REQ-INPUTKIND-BDRABCZUK-C7DF293F788A."""
    _check_both_files(layout, _run_combine_stage, COMBINE_STAGE)


def test_combine_pre_check_logs_no_warning(layout, pipeline_caplog):
    """REQ-INPUTKIND-BDRABCZUK-A560A9B17CDC."""
    _check_no_warning(layout, _run_combine_stage, COMBINE_STAGE, pipeline_caplog)


def test_combine_pre_index_holds_the_region_embeddings_folder(layout, pipeline_caplog):
    """REQ-INPUTKIND-BDRABCZUK-BD11E62B67F5."""
    _check_pre_index(
        layout, _run_combine_stage, COMBINE_STAGE, (None, layout.region_embeddings), REGION_EMBEDDING_FILES
    )


def test_combine_post_index_holds_the_combined_embeddings_folder(layout, pipeline_caplog):
    """REQ-INPUTKIND-BDRABCZUK-B99D925FC29E."""
    _check_post_index(layout, _run_combine_stage, COMBINE_STAGE, ("--output_path", layout.combined), COMBINED_FILES)


# ---------------------------------------------------------------- generate_snapshots (stage 6)


def _run_snapshots_stage(layout: _Layout) -> _FakeExecute:
    inputs = [(layout.combined, COMBINED_FILES)]
    return _run_on(layout, SNAPSHOTS_STAGE, inputs, layout.snapshots, SNAPSHOT_FILES)


def test_snapshots_stage_leaves_both_index_files(layout, pipeline_caplog):
    """REQ-INPUTKIND-BDRABCZUK-8C7D1A47FF6C."""
    _check_both_files(layout, _run_snapshots_stage, SNAPSHOTS_STAGE)


def test_snapshots_post_index_holds_the_snapshots_folder(layout, pipeline_caplog):
    """REQ-INPUTKIND-BDRABCZUK-95ABACD6119B."""
    _check_post_index(layout, _run_snapshots_stage, SNAPSHOTS_STAGE, ("--output_dir", layout.snapshots), SNAPSHOT_FILES)


def test_snapshots_pre_index_holds_the_combined_embeddings_folder(layout, pipeline_caplog):
    """REQ-INPUTKIND-BDRABCZUK-C75463F7A5C8."""
    _check_pre_index(
        layout, _run_snapshots_stage, SNAPSHOTS_STAGE, ("--embeddings_dir", layout.combined), COMBINED_FILES
    )


def test_snapshots_pre_check_logs_no_warning(layout, pipeline_caplog):
    """REQ-INPUTKIND-BDRABCZUK-6429645ABB6A."""
    _check_no_warning(layout, _run_snapshots_stage, SNAPSHOTS_STAGE, pipeline_caplog)


# ---------------------------------------------------------------- the StageIndexer port


class _RecordingIndexer:
    """A fake StageIndexer: records each request and writes no file."""

    def __init__(self):
        self.calls: list[tuple] = []

    def index_pre(self, stage_name, input_dirs, required_patterns):  # noqa: V105 - the StageIndexer port
        self.calls.append(("pre", stage_name, [str(d) for d in input_dirs], list(required_patterns)))

    def index_post(self, stage_name, output_dir):  # noqa: V105 - the StageIndexer port
        self.calls.append(("post", stage_name, str(output_dir)))


def test_orchestrator_routes_index_requests_to_its_stage_indexer(main_module, tmp_path, layout, pipeline_caplog):
    """REQ R1: every pre and post index request goes to the given StageIndexer; the orchestrator writes no file."""
    _write_files(layout.crops, CROP_FILES)
    config = _config(main_module, tmp_path, layout, EMBEDDINGS_STAGE)
    indexer = _RecordingIndexer()
    orchestrator = main_module.PipelineOrchestrator(config, indexer=indexer)
    fake = _FakeExecute(EMBEDDINGS_STAGE, layout.outputs / "unused", layout.region_embeddings, REGION_EMBEDDING_FILES)
    stage = orchestrator.stages[EMBEDDINGS_STAGE]
    stage.validate = lambda: True
    stage.execute = fake

    assert orchestrator.run() == 0

    assert [call[:2] for call in indexer.calls] == [("pre", EMBEDDINGS_STAGE), ("post", EMBEDDINGS_STAGE)]
    assert str(layout.crops) in indexer.calls[0][2]
    assert indexer.calls[1][2] == str(layout.region_embeddings)
    assert list(layout.outputs.glob("index_*.json")) == []


def _imported_modules(path: Path) -> list[tuple[str, list[str]]]:
    """(module, imported names) of every import in the file, relative imports as written."""
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            found.append(("." * node.level + (node.module or ""), [alias.name for alias in node.names]))
        elif isinstance(node, ast.Import):
            found.extend((alias.name, []) for alias in node.names)
    return found


def _from_file_indexer(path: Path) -> list[str]:
    """Names (or module names) the file takes from the champollion_pipeline.file_indexer package."""
    names = []
    for module, imported in _imported_modules(path):
        if "file_indexer" in module.split("."):
            names.extend(imported or [module])
    return names


def test_only_the_default_adapter_is_imported_from_file_indexer():
    """REQ R2: main.py imports the default adapter from file_indexer and nothing else; _stage_base.py nothing."""
    stage_base = MAIN_PY.parent / "src" / "champollion_pipeline" / "_stage_base.py"

    assert _from_file_indexer(MAIN_PY) == ["FileIndexStageIndexer"]
    assert _from_file_indexer(stage_base) == []
