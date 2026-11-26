"""Tests of the user decisions of 2026-11-10 on infer_paths.

Q1 (superseded 2026-11-26 by TASK-ARGDISC-BDRABCZUK-1601F53514C5, user rule "any .arg is a graph"): graphs in
several graph folders are no error and no pick: path inference sets no graph folder path. Q2: a Morphologist input
with no derivatives folder above it, or a single subject, gives no dataset root unless --datasets-root is set.
Q3 and Q4: models_path and sk_qc_path come back as given.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from champollion_pipeline.config import DatasetConfig, PipelineConfig
from champollion_pipeline.path_inference import DatasetRootNotInferable, infer_paths
from champollion_pipeline.utils.input_kind import InputKind

MORPHOLOGIST_6 = "derivatives/morphologist-6.0"
GRAPHS = "t1mri/a/n/folds/3.1"
# (subject -> graph folders below the subject, bids, the folders the error must list in this order)
SEVERAL_FOLDERS = [
    pytest.param(
        {"s2": ["t1mri/a/n/folds/3.1"], "s1": ["t1mri/a/n/folds/3.3"]},
        False,
        "t1mri/a/n/folds/3.1, t1mri/a/n/folds/3.3",
        id="two_versions",
    ),
    pytest.param(
        {"s1": ["t1mri/a1/n/folds/3.1"], "s2": ["t1mri/a2/n/folds/3.1"]},
        False,
        "t1mri/a1/n/folds/3.1, t1mri/a2/n/folds/3.1",
        id="two_acquisitions",
    ),
    pytest.param(
        {"sub-01": ["t1mri/ses-1/n/folds/3.3", "t1mri/ses-2/n/folds/3.1"]},
        True,
        "t1mri/*/n/folds/3.1, t1mri/*/n/folds/3.3",
        id="two_versions_bids",
    ),
]


def _touch(path: Path) -> Path:
    """Create the file ``path`` and its parent folders, return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _graphs(subjects: Path, folders: dict[str, list[str]]) -> None:
    """Write a left graph in each folder of ``folders`` (subject -> graph folders relative to the subject)."""
    for subject, subject_folders in folders.items():
        for folder in subject_folders:
            _touch(subjects / subject / folder / f"L{subject}.arg")


def _morphologist_root(tmp_path: Path, folders: dict[str, list[str]]) -> Path:
    """Return a dataset root whose subjects sit in the folder stage 1 writes, their graphs in ``folders``."""
    root = tmp_path / "dataset"
    _graphs(root / MORPHOLOGIST_6, folders)
    return root


def _t1_input(tmp_path: Path) -> Path:
    """Return a BIDS folder of raw T1 files."""
    root = tmp_path / "bids"
    _touch(root / "sub-01" / "anat" / "sub-01_T1w.nii.gz")
    return root


def _no_derivatives_input(tmp_path: Path) -> Path:
    """Return a subjects folder of two subjects with no folder named derivatives above it."""
    subjects = tmp_path / "subjects"
    _graphs(subjects, {"s1": [GRAPHS], "s2": [GRAPHS]})
    return subjects


def _single_subject_input(tmp_path: Path) -> Path:
    """Return one subject folder (it holds t1mri itself), below a derivatives folder."""
    subjects = tmp_path / "derivatives" / "m"
    _graphs(subjects, {"s1": [GRAPHS]})
    return subjects / "s1"


NO_ROOT_INPUTS = [
    pytest.param(_no_derivatives_input, id="no_derivatives"),
    pytest.param(_single_subject_input, id="single_subject"),
]
UNTOUCHED_VALUES = [pytest.param("", id="empty"), pytest.param("/data/given/value", id="set")]
KINDS = [pytest.param(InputKind.T1, id="t1"), pytest.param(InputKind.MORPHOLOGIST, id="morphologist")]


def _input(tmp_path: Path, kind: InputKind) -> Path:
    """Return an input of ``kind`` from which infer_paths infers every path without error."""
    if kind is InputKind.T1:
        return _t1_input(tmp_path)
    return _morphologist_root(tmp_path, {"s1": [GRAPHS], "s2": [GRAPHS]})


@pytest.mark.parametrize(("folders", "bids", "listed"), SEVERAL_FOLDERS)
def test_several_graph_folders_set_no_graph_folder(
    tmp_path: Path, folders: dict[str, list[str]], bids: bool, listed: str
) -> None:
    """Subjects whose graphs sit in several folds versions or acquisitions, with or without --bids, raise nothing and
    get no graph folder path: stage 2 finds the graphs by graph discovery (REQ-ARGDISC-BDRABCZUK-3CA4DDF7F6C7)."""
    root = _morphologist_root(tmp_path, folders)
    result = infer_paths(PipelineConfig(dataset=DatasetConfig(bids=bids)), str(root), InputKind.MORPHOLOGIST)
    assert result.dataset.path_to_graph is None


def test_root_not_inferable_without_derivatives(tmp_path: Path) -> None:
    """A subjects folder with no derivatives folder above it gives no dataset root."""
    with pytest.raises(DatasetRootNotInferable) as caught:
        infer_paths(PipelineConfig(), str(_no_derivatives_input(tmp_path)), InputKind.MORPHOLOGIST)
    assert isinstance(caught.value, ValueError)


def test_root_not_inferable_from_single_subject(tmp_path: Path) -> None:
    """A single subject has no subjects folder, so no dataset root, even below a derivatives folder."""
    with pytest.raises(DatasetRootNotInferable):
        infer_paths(PipelineConfig(), str(_single_subject_input(tmp_path)), InputKind.MORPHOLOGIST)


@pytest.mark.parametrize("make_input", NO_ROOT_INPUTS)
def test_root_not_inferable_names_datasets_root(tmp_path: Path, make_input: Callable[[Path], Path]) -> None:
    """The error tells the user which option gives the dataset root."""
    with pytest.raises(DatasetRootNotInferable, match="--datasets-root"):
        infer_paths(PipelineConfig(), str(make_input(tmp_path)), InputKind.MORPHOLOGIST)


@pytest.mark.parametrize("make_input", NO_ROOT_INPUTS)
def test_set_datasets_root_raises_no_root_error(tmp_path: Path, make_input: Callable[[Path], Path]) -> None:
    """With --datasets-root set, the same inputs run; the graph folder stays unset."""
    explicit = str(tmp_path / "given_root")
    config = PipelineConfig(dataset=DatasetConfig(datasets_root=explicit))
    result = infer_paths(config, str(make_input(tmp_path)), InputKind.MORPHOLOGIST)
    assert result.dataset.datasets_root == explicit
    assert result.dataset.path_to_graph is None


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", UNTOUCHED_VALUES)
def test_models_path_returned_unchanged(tmp_path: Path, kind: InputKind, value: str) -> None:
    """Path inference neither fills nor checks models_path."""
    config = PipelineConfig(models_path=value)
    result = infer_paths(config, str(_input(tmp_path, kind)), kind)
    assert result.models_path == value


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", UNTOUCHED_VALUES)
def test_sk_qc_path_returned_unchanged(tmp_path: Path, kind: InputKind, value: str) -> None:
    """Path inference neither fills nor checks sk_qc_path."""
    config = PipelineConfig(dataset=DatasetConfig(sk_qc_path=value))
    result = infer_paths(config, str(_input(tmp_path, kind)), kind)
    assert result.dataset.sk_qc_path == value
