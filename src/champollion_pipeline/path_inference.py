"""Path inference: every unset path field of a pipeline config, from the one input path and its kind.

A config-less run gives only the input path. Its kind (T1 files or Morphologist outputs, see input_kind) decides where
the dataset root and the Morphologist subjects folder are; every later path derives from the dataset root through
derivatives_layout. A field set explicitly (command line or YAML) is never overwritten. models_path and sk_qc_path are
never inferred.

Inputs:
    config: PipelineConfig whose empty or default path fields are to be filled
    input_path: the run's input folder (T1 files, or a Morphologist tree)
    kind: InputKind of input_path, from input_kind.detect_input_kind
    utils/input_kind.py: resolve_subjects_dir, stage1_subjects_dir
    derivatives_layout.py and utils/cortical_tiles_config.py: derivatives folder names and path helpers
    main.py: calls infer_paths before running the stages
Outputs:
    a copy of config with datasets_root, morphologist_graphs, cortical_tiles_output, crops_path, embeddings_path,
    snapshots_path, outputs_path and path_sk_with_hull filled; path_to_graph filled for a T1 input, set to None for a
    Morphologist input that gave none (stage 2 then finds the graphs by graph discovery)
    ValueError or DatasetRootNotInferable when a path cannot be inferred
    main.py: runs the stages on the returned config

```mermaid
flowchart LR
    input[(input_path)] --> this[path_inference.py]
    config[(PipelineConfig)] --> this
    kind[input_kind.py] -.kind, subjects folder.-> this
    layout[derivatives_layout.py] -.path helpers.-> this
    this --> filled[(PipelineConfig with inferred paths)]
    this --> error[/error naming the argument to pass/]
    main[main.py] -.calls.-> this
    filled --> main
```
"""

import copy
import os
from pathlib import Path

from champollion_pipeline.config import DatasetConfig, PipelineConfig
from champollion_pipeline.derivatives_layout import (
    CHAMPOLLION_DERIVATIVES_FOLDER,
    DERIVATIVES_DIRNAME,
    compute_region_embeddings_dir,
    compute_snapshots_dir,
)
from champollion_pipeline.run_masks import compute_crops_versions
from champollion_pipeline.utils.cortical_tiles_config import compute_versioned_crops_dir
from champollion_pipeline.utils.input_kind import InputKind, resolve_subjects_dir, stage1_subjects_dir
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER

# Subject-relative graph folder stage 1 writes, without and with --bids (the acquisition matched by a wildcard).
_STAGE1_GRAPH_FOLDER = "t1mri/default_acquisition/default_analysis/folds/3.1"
_STAGE1_BIDS_GRAPH_FOLDER = "t1mri/*/default_analysis/folds/3.1"
_FOLDS_SEGMENT = "/folds/"
_SEGMENTATION = "segmentation"
_DATASETS_ROOT_FLAG = "--datasets-root"

_DATASET_DEFAULTS = DatasetConfig()
_PIPELINE_DEFAULTS = PipelineConfig()


class DatasetRootNotInferable(ValueError):
    """The dataset root of a Morphologist input cannot be read from its path."""

    def __init__(self, input_path: str, reason: str) -> None:
        super().__init__(input_path, reason)
        self.input_path = input_path
        self.reason = reason

    def __str__(self) -> str:
        return f"no dataset root found for {self.input_path}: {self.reason}; give it with {_DATASETS_ROOT_FLAG}"


def infer_paths(config: PipelineConfig, input_path: str, kind: InputKind) -> PipelineConfig:
    """Return a copy of ``config`` with every unset path field inferred from ``input_path`` and ``kind``.

    A field is unset when it equals its DatasetConfig or PipelineConfig default. ``config`` itself is not changed;
    models_path and sk_qc_path come back as given.

    Raises:
        ValueError: ``kind`` is neither t1 nor morphologist.
        AmbiguousSubjectsFolders: from resolve_subjects_dir, a Morphologist input with several subjects folders.
        DatasetRootNotInferable: a Morphologist input with datasets_root unset that is a single subject, or whose
            subjects folder sits below no derivatives folder.
    """
    if kind not in (InputKind.T1, InputKind.MORPHOLOGIST):
        raise ValueError(
            f"no path inference for input kind {kind.value}: the input must hold T1 files or Morphologist outputs"
        )
    result = copy.deepcopy(config)
    if kind is InputKind.T1:
        _infer_t1_paths(result.dataset, input_path)
    else:
        _infer_morphologist_paths(result.dataset, input_path)
    _infer_derived_paths(result)
    return result


def refresh_version_paths(config: PipelineConfig, previous_masks_version: str) -> None:
    """Re-derive, in place, the crops, embeddings and snapshots paths inferred from previous_masks_version.

    Each of crops_path, embeddings_path and snapshots_path equal to the value _infer_derived_paths derives from
    previous_masks_version is derived again from dataset.masks_version (crops_path from the first Mask version whose
    crops the run reads, compute_crops_versions); any other value (given by the user) is left as is. O(56).
    """
    dataset = config.dataset
    root = dataset.datasets_root
    tiles = os.path.join(root, DERIVATIVES_DIRNAME, DERIVATIVES_FOLDER)
    if dataset.crops_path == compute_versioned_crops_dir(tiles, previous_masks_version):
        dataset.crops_path = compute_versioned_crops_dir(tiles, compute_crops_versions(dataset)[0])
    if dataset.embeddings_path == compute_region_embeddings_dir(root, previous_masks_version):
        dataset.embeddings_path = compute_region_embeddings_dir(root, dataset.masks_version)
    if dataset.snapshots_path == compute_snapshots_dir(root, previous_masks_version):
        dataset.snapshots_path = compute_snapshots_dir(root, dataset.masks_version)


def _is_unset(dataset: DatasetConfig, name: str) -> bool:
    """Return whether the field ``name`` of ``dataset`` holds its default value."""
    return getattr(dataset, name) == getattr(_DATASET_DEFAULTS, name)


def _infer_t1_paths(dataset: DatasetConfig, input_path: str) -> None:
    """Fill the root, subjects folder and graph path of a T1 input: stage 1 writes below the input itself."""
    if _is_unset(dataset, "datasets_root"):
        dataset.datasets_root = os.path.abspath(input_path)
    if _is_unset(dataset, "morphologist_graphs"):
        dataset.morphologist_graphs = stage1_subjects_dir(dataset.datasets_root)
    if _is_unset(dataset, "path_to_graph"):
        dataset.path_to_graph = _STAGE1_BIDS_GRAPH_FOLDER if dataset.bids else _STAGE1_GRAPH_FOLDER


def _infer_morphologist_paths(dataset: DatasetConfig, input_path: str) -> None:
    """Fill the subjects folder and root of a Morphologist input from its tree; its graph path stays unset (None)."""
    if _is_unset(dataset, "morphologist_graphs"):
        subjects_dir = resolve_subjects_dir(input_path)
        if subjects_dir is not None:
            dataset.morphologist_graphs = os.path.abspath(subjects_dir)
    if _is_unset(dataset, "datasets_root"):
        dataset.datasets_root = _root_above_derivatives(input_path, dataset.morphologist_graphs)
    if _is_unset(dataset, "path_to_graph"):
        dataset.path_to_graph = None  # stage 2 finds the graphs by graph discovery, whatever the folder names


def _root_above_derivatives(input_path: str, subjects_dir: str) -> str:
    """Return the parent of the nearest folder named derivatives above ``subjects_dir``."""
    if not subjects_dir:
        raise DatasetRootNotInferable(input_path, "it holds no folder of subjects (a single subject?)")
    absolute = Path(os.path.abspath(subjects_dir))
    for parent in absolute.parents:
        if parent.name == DERIVATIVES_DIRNAME:
            return str(parent.parent)
    reason = f"its subjects folder {absolute} sits below no {DERIVATIVES_DIRNAME} folder"
    raise DatasetRootNotInferable(input_path, reason)


def _infer_derived_paths(config: PipelineConfig) -> None:
    """Fill the fields that derive from the dataset root and the graph path, both already returned."""
    dataset = config.dataset
    root = dataset.datasets_root
    derivatives = os.path.join(root, DERIVATIVES_DIRNAME)
    if _is_unset(dataset, "cortical_tiles_output"):
        dataset.cortical_tiles_output = derivatives
    if _is_unset(dataset, "crops_path"):
        tiles = os.path.join(derivatives, DERIVATIVES_FOLDER)
        dataset.crops_path = compute_versioned_crops_dir(tiles, dataset.masks_version)
    if _is_unset(dataset, "embeddings_path"):
        dataset.embeddings_path = compute_region_embeddings_dir(root, dataset.masks_version)
    if _is_unset(dataset, "snapshots_path"):
        dataset.snapshots_path = compute_snapshots_dir(root, dataset.masks_version)
    if config.outputs_path == _PIPELINE_DEFAULTS.outputs_path:
        config.outputs_path = os.path.join(derivatives, CHAMPOLLION_DERIVATIVES_FOLDER)
    if _is_unset(dataset, "path_sk_with_hull") and dataset.path_to_graph and _FOLDS_SEGMENT in dataset.path_to_graph:
        dataset.path_sk_with_hull = dataset.path_to_graph.partition(_FOLDS_SEGMENT)[0] + "/" + _SEGMENTATION
