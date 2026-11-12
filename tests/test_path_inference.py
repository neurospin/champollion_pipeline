"""Tests of infer_paths: every unset path field of a config from the one input path and its kind.

Trees copy the two real shapes: raw T1 files in a BIDS folder (stage 1 not run yet) and a Morphologist tree, either
``<root>/derivatives/morphologist-6.0/<sub>`` as stage 1 writes it or an older tree with a ``subjects`` level.
"""

import os
from dataclasses import replace
from pathlib import Path

import pytest

from champollion_pipeline.config import DatasetConfig, PipelineConfig
from champollion_pipeline.derivatives_layout import compute_region_embeddings_dir, compute_snapshots_dir
from champollion_pipeline.path_inference import infer_paths
from champollion_pipeline.utils.input_kind import InputKind, resolve_subjects_dir, stage1_subjects_dir
from champollion_pipeline.utils.lib import CORTICAL_TILES_VERSION

MASKS = "canonical_99"  # not the default: shows that the derived paths read dataset.masks_version
STAGE1_GRAPHS = "t1mri/default_acquisition/default_analysis/folds/3.1"
TREE_GRAPHS = "t1mri/acq_x/my_analysis/folds/3.2"  # names no constant holds: read from the tree
MORPHOLOGIST_6 = "derivatives/morphologist-6.0"
# (input below the dataset root, subjects folder below the dataset root) of the real Morphologist shapes.
MORPHOLOGIST_LAYOUTS = [
    pytest.param(".", MORPHOLOGIST_6, id="dataset_root"),
    pytest.param(MORPHOLOGIST_6, MORPHOLOGIST_6, id="morphologist_folder"),
    pytest.param(".", f"{MORPHOLOGIST_6}/subjects", id="dataset_root_legacy_subjects_level"),
    pytest.param(f"{MORPHOLOGIST_6}/subjects", f"{MORPHOLOGIST_6}/subjects", id="legacy_subjects_folder"),
    pytest.param(".", "derivatives/morphologist-2021/subjects", id="dataset_root_morphologist_2021"),
]
ROOT_SOURCES = [pytest.param(False, id="inferred_root"), pytest.param(True, id="explicit_root")]
# Path fields infer_paths fills, as attribute paths of a PipelineConfig.
PATH_FIELDS = (
    "dataset.datasets_root",
    "dataset.morphologist_graphs",
    "dataset.cortical_tiles_output",
    "dataset.crops_path",
    "dataset.embeddings_path",
    "dataset.snapshots_path",
    "outputs_path",
    "dataset.path_to_graph",
    "dataset.path_sk_with_hull",
)


def _touch(path: Path) -> Path:
    """Create the file ``path`` and its parent folders, return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _t1_input(tmp_path: Path) -> Path:
    """Return a BIDS folder of raw T1 files."""
    root = tmp_path / "bids"
    _touch(root / "sub-01" / "anat" / "sub-01_T1w.nii.gz")
    _touch(root / "sub-02" / "anat" / "sub-02_T1w.nii.gz")
    return root


def _graphs(subjects: Path, folders: dict[str, str]) -> None:
    """Write a left graph per subject of ``folders`` (subject -> graph folder relative to the subject)."""
    for subject, folder in folders.items():
        _touch(subjects / subject / folder / f"L{subject}.arg")


def _morphologist_root(tmp_path: Path, subjects_below: str = MORPHOLOGIST_6, folder: str = TREE_GRAPHS) -> Path:
    """Return a dataset root whose two subjects hold their graphs in ``folder``, one labelled in a session below."""
    root = tmp_path / "dataset"
    _graphs(root / subjects_below, {"s1": folder, "s2": f"{folder}/session_auto"})
    return root


def _config(**dataset_fields: object) -> PipelineConfig:
    """Return a default PipelineConfig with MASKS and the given dataset fields."""
    return PipelineConfig(dataset=DatasetConfig(masks_version=MASKS, **dataset_fields))


def _field(config: PipelineConfig, name: str) -> object:
    """Return the field at attribute path ``name`` of ``config``."""
    owner, _, attribute = name.rpartition(".")
    return getattr(config.dataset if owner else config, attribute)


def _with_field(config: PipelineConfig, name: str, value: str) -> PipelineConfig:
    """Return ``config`` with the field at attribute path ``name`` set to ``value``."""
    owner, _, attribute = name.rpartition(".")
    if owner:
        return replace(config, dataset=replace(config.dataset, **{attribute: value}))
    return replace(config, **{attribute: value})


def _t1_result(tmp_path: Path, explicit_root: bool) -> tuple[PipelineConfig, str]:
    """Return infer_paths of a T1 input, and the dataset root the derived paths must use."""
    explicit = str(tmp_path / "elsewhere")
    config = _config(datasets_root=explicit) if explicit_root else _config()
    result = infer_paths(config, str(_t1_input(tmp_path)), InputKind.T1)
    return result, explicit if explicit_root else str(tmp_path / "bids")


def test_t1_datasets_root_is_absolute_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A T1 input given as a relative path is its own dataset root, as an absolute path."""
    _t1_input(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = infer_paths(_config(), "bids", InputKind.T1)
    assert result.dataset.datasets_root == os.path.abspath("bids")


@pytest.mark.parametrize(("input_below", "subjects_below"), MORPHOLOGIST_LAYOUTS)
def test_morphologist_datasets_root_above_derivatives(tmp_path: Path, input_below: str, subjects_below: str) -> None:
    """The dataset root of a Morphologist input is the folder holding the derivatives folder of its subjects."""
    root = _morphologist_root(tmp_path, subjects_below)
    result = infer_paths(_config(), str(root / input_below), InputKind.MORPHOLOGIST)
    assert result.dataset.datasets_root == str(root)


@pytest.mark.parametrize("explicit_root", ROOT_SOURCES)
def test_t1_morphologist_graphs_is_stage1_subjects_dir(tmp_path: Path, explicit_root: bool) -> None:
    """The subjects folder of a T1 input is where stage 1 will write below the returned dataset root."""
    result, root = _t1_result(tmp_path, explicit_root)
    assert result.dataset.morphologist_graphs == stage1_subjects_dir(root)


@pytest.mark.parametrize(("input_below", "subjects_below"), MORPHOLOGIST_LAYOUTS)
def test_morphologist_graphs_is_resolved_subjects_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, input_below: str, subjects_below: str
) -> None:
    """The subjects folder of a Morphologist input is the resolved one, as an absolute path."""
    _morphologist_root(tmp_path, subjects_below)
    monkeypatch.chdir(tmp_path)
    input_path = os.path.normpath(os.path.join("dataset", input_below))
    result = infer_paths(_config(), input_path, InputKind.MORPHOLOGIST)
    assert result.dataset.morphologist_graphs == os.path.abspath(resolve_subjects_dir(input_path))


@pytest.mark.parametrize("explicit_root", ROOT_SOURCES)
def test_cortical_tiles_output_under_root(tmp_path: Path, explicit_root: bool) -> None:
    """Stage 2 writes into the derivatives folder of the returned dataset root."""
    result, root = _t1_result(tmp_path, explicit_root)
    assert result.dataset.cortical_tiles_output == os.path.join(root, "derivatives")


@pytest.mark.parametrize("explicit_root", ROOT_SOURCES)
def test_crops_path_under_root(tmp_path: Path, explicit_root: bool) -> None:
    """Stage 3 reads the 2mm crops of the configured mask version below the returned dataset root."""
    result, root = _t1_result(tmp_path, explicit_root)
    expected = os.path.join(root, "derivatives", f"cortical_tiles-{CORTICAL_TILES_VERSION}", "crops", MASKS, "2mm")
    assert result.dataset.crops_path == expected


@pytest.mark.parametrize("explicit_root", ROOT_SOURCES)
def test_embeddings_path_under_root(tmp_path: Path, explicit_root: bool) -> None:
    """The region embeddings folder is the versioned one below the returned dataset root."""
    result, root = _t1_result(tmp_path, explicit_root)
    assert result.dataset.embeddings_path == compute_region_embeddings_dir(root, MASKS)


@pytest.mark.parametrize("explicit_root", ROOT_SOURCES)
def test_snapshots_path_under_root(tmp_path: Path, explicit_root: bool) -> None:
    """The snapshots folder is the versioned one below the returned dataset root."""
    result, root = _t1_result(tmp_path, explicit_root)
    assert result.dataset.snapshots_path == compute_snapshots_dir(root, MASKS)


@pytest.mark.parametrize("explicit_root", ROOT_SOURCES)
def test_outputs_path_under_root(tmp_path: Path, explicit_root: bool) -> None:
    """The run outputs (index files, logs) go to the champollion_V1 derivatives folder of the dataset root."""
    result, root = _t1_result(tmp_path, explicit_root)
    assert result.outputs_path == os.path.join(root, "derivatives", "champollion_V1")


def test_t1_path_to_graph_without_bids(tmp_path: Path) -> None:
    """Without --bids, a T1 input reads the graphs where stage 1 writes them."""
    result = infer_paths(_config(), str(_t1_input(tmp_path)), InputKind.T1)
    assert result.dataset.path_to_graph == STAGE1_GRAPHS


def test_t1_path_to_graph_with_bids(tmp_path: Path) -> None:
    """With --bids, the acquisition folder of each scan is matched by a wildcard."""
    result = infer_paths(_config(bids=True), str(_t1_input(tmp_path)), InputKind.T1)
    assert result.dataset.path_to_graph == "t1mri/*/default_analysis/folds/3.1"


def test_morphologist_path_to_graph_from_tree(tmp_path: Path) -> None:
    """Without --bids, the graph folder is read from the tree, a labelled session folder below it included."""
    root = _morphologist_root(tmp_path)
    result = infer_paths(_config(), str(root), InputKind.MORPHOLOGIST)
    assert result.dataset.path_to_graph == TREE_GRAPHS


def test_morphologist_path_to_graph_from_bids_tree(tmp_path: Path) -> None:
    """With --bids, the acquisition folders differ between scans and the inferred path matches them by a wildcard."""
    root = tmp_path / "dataset"
    _graphs(
        root / MORPHOLOGIST_6,
        {
            "sub-01": "t1mri/ses-1_acq-a/my_analysis/folds/3.2",
            "sub-02": "t1mri/ses-2/my_analysis/folds/3.2/session_auto",
        },
    )
    result = infer_paths(_config(bids=True), str(root), InputKind.MORPHOLOGIST)
    assert result.dataset.path_to_graph == "t1mri/*/my_analysis/folds/3.2"


@pytest.mark.parametrize(
    ("kind", "dataset_fields", "expected"),
    [
        pytest.param(InputKind.T1, {}, "t1mri/default_acquisition/default_analysis/segmentation", id="t1"),
        pytest.param(InputKind.T1, {"bids": True}, "t1mri/*/default_analysis/segmentation", id="t1_bids"),
        pytest.param(InputKind.MORPHOLOGIST, {}, "t1mri/acq_x/my_analysis/segmentation", id="morphologist"),
        pytest.param(
            InputKind.MORPHOLOGIST,
            {"path_to_graph": "t1mri/a/b/folds/9.9/base"},
            "t1mri/a/b/segmentation",
            id="explicit_graph_path",
        ),
    ],
)
def test_path_sk_with_hull_beside_folds(
    tmp_path: Path, kind: InputKind, dataset_fields: dict[str, object], expected: str
) -> None:
    """The skeleton folder is the segmentation folder beside the folds folder of the returned graph path."""
    input_path = _t1_input(tmp_path) if kind is InputKind.T1 else _morphologist_root(tmp_path)
    result = infer_paths(_config(**dataset_fields), str(input_path), kind)
    assert result.dataset.path_sk_with_hull == expected


@pytest.mark.parametrize("kind", [InputKind.T1, InputKind.MORPHOLOGIST], ids=lambda kind: kind.value)
@pytest.mark.parametrize("name", PATH_FIELDS)
def test_set_field_is_kept(tmp_path: Path, kind: InputKind, name: str) -> None:
    """A path field holding a value other than its default comes back unchanged, whatever the input."""
    # A subjects folder below a derivatives folder: an explicit morphologist_graphs still gives a root and a graph path.
    explicit_folder = str(tmp_path / "other" / "derivatives" / "explicit")
    _graphs(Path(explicit_folder), {"s9": TREE_GRAPHS})
    explicit = "t1mri/explicit/analysis/folds/0.0" if name == "dataset.path_to_graph" else explicit_folder
    assert _field(PipelineConfig(), name) != explicit
    input_path = _t1_input(tmp_path) if kind is InputKind.T1 else _morphologist_root(tmp_path)
    result = infer_paths(_with_field(_config(), name, explicit), str(input_path), kind)
    assert _field(result, name) == explicit


@pytest.mark.parametrize("kind", [InputKind.NEITHER, InputKind.BOTH], ids=lambda kind: kind.value)
def test_kind_without_inference_raises(tmp_path: Path, kind: InputKind) -> None:
    """An input whose kind is undecided gives no inferred path."""
    with pytest.raises(ValueError, match=kind.value):
        infer_paths(_config(), str(_t1_input(tmp_path)), kind)
