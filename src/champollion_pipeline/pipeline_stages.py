"""
The six sequential pipeline stages (Strategy implementations of PipelineStage) and their path helpers.

Moved from champollion_pipeline.cli (TASK-CODE-BDRABCZUK-D457E2861B14); cli.py imports them from here,
never the reverse. Each stage names its script module and class as strings and resolves them with
importlib when execute builds the script, so importing this module loads no toolchain package.
"""

import importlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline._stage_base import PipelineStage, StageResult, existing_dir
from champollion_pipeline.config import DatasetConfig, PipelineConfig
from champollion_pipeline.derivatives_layout import (
    DEFAULT_MASKS_VERSION,
    compute_champollion_configs_root,
    compute_combined_embeddings_dir,
    compute_models_cache_dir,
    compute_region_embeddings_dir,
    compute_region_model_names,
    compute_snapshots_dir,
)
from champollion_pipeline.models.fetch import HF_FALLBACK_REPO_ID, resolve_models_source
from champollion_pipeline.run_masks import compute_crops_versions
from champollion_pipeline.utils.cortical_tiles_config import compute_versioned_crops_dir
from champollion_pipeline.utils.input_kind import stage1_subjects_dir
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER, find_dataset_folder
from champollion_pipeline.utils.region_masks import REGION_MASKS_OPTION
from champollion_pipeline.utils.skeleton_qc import SKELETON_SIDES, compute_skeleton_pattern


def _new_script(module_name: str, class_name: str) -> ScriptBuilder:
    """A fresh instance of class_name, looked up in module_name when called (importlib, never at module load)."""
    return getattr(importlib.import_module(module_name), class_name)()


def _compute_masks_args(dataset: DatasetConfig) -> list[str]:
    """Stage 2 and 3 --masks argument: --masks=<masks_version>, none on a map run (dataset.region_masks set)."""
    return [] if dataset.region_masks else [f"--masks={dataset.masks_version}"]


def _compute_region_masks_args(dataset: DatasetConfig) -> list[str]:
    """--region-masks followed by the NAME=VERSION entries on a map run, else nothing; stages 2 to 4 put it last."""
    return [REGION_MASKS_OPTION, *dataset.region_masks] if dataset.region_masks else []


class GenerateMorphologistGraphsStage(PipelineStage):
    """Stage for generating Morphologist graphs."""

    progress_message = "Generating Morphologist graphs..."
    success_message = "Morphologist graphs generated successfully"
    failure_message = "Morphologist failed"

    def validate(self) -> bool:
        """Validate the input data, the output root and the subjects folder override against what stage 1 writes."""
        dataset = self.config.dataset
        input_path = Path(dataset.input_path)
        if not input_path.exists():
            self.logger.error(f"Input path does not exist: {input_path}")
            return False
        if not dataset.datasets_root:
            self.logger.error("dataset.datasets_root, the Morphologist output root, is empty")
            return False
        written = stage1_subjects_dir(dataset.datasets_root)
        if dataset.morphologist_graphs and os.path.abspath(dataset.morphologist_graphs) != os.path.abspath(written):
            self.logger.error(
                f"dataset.morphologist_graphs {dataset.morphologist_graphs} is not where stage 1 writes: {written}"
            )
            return False
        return True

    def build_args(self) -> list[str]:
        args = [
            str(self.config.dataset.input_path),
            str(self.config.dataset.datasets_root),
        ]
        if self.config.dataset.bids:
            args.append("--bids")
        if self.config.parallel:
            args.append("--parallel")
        return args

    def create_script(self) -> ScriptBuilder:
        return _new_script("champollion_pipeline.generate_morphologist_graphs", "GenerateMorphologistGraphs")


def _compute_path_to_graph_args(dataset: DatasetConfig) -> list[str]:
    """--path_to_graph=<value> for stage 2, none without a graph folder path (stage 2 then uses graph discovery)."""
    return [f"--path_to_graph={dataset.path_to_graph}"] if dataset.path_to_graph is not None else []


class RunCorticalTilesStage(PipelineStage):
    """Stage for running cortical_tiles."""

    progress_message = "Running cortical_tiles to generate sulcal regions..."
    success_message = "Cortical tiles completed"
    failure_message = "Failed"

    def validate(self) -> bool:
        """Validate that the Morphologist subjects folder exists."""
        subjects_dir = _compute_morphologist_subjects_dir(self.config.dataset)
        if not subjects_dir or not os.path.isdir(subjects_dir):
            self.logger.error(f"Morphologist subjects folder does not exist: {subjects_dir!r}")
            return False
        return True

    def build_args(self) -> list[str]:
        args = [
            _compute_morphologist_subjects_dir(self.config.dataset),
            str(self.config.dataset.cortical_tiles_output),
            *_compute_path_to_graph_args(self.config.dataset),
            f"--path_sk_with_hull={self.config.dataset.path_sk_with_hull}",
            f"--njobs={self.config.dataset.njobs}",
            *_compute_masks_args(self.config.dataset),
        ]

        if self.config.dataset.skel_seed is not None:
            args.append(f"--skel-seed={self.config.dataset.skel_seed}")
        if self.config.dataset.labelling_session:
            args.append(f"--labelling_session={self.config.dataset.labelling_session}")

        if self.config.dataset.sk_qc_path:
            args.append(f"--sk_qc_path={self.config.dataset.sk_qc_path}")

        if self.config.dataset.bids:
            args.append("--bids")
        if self.config.dataset.allow_prefix_ids:
            args.append("--allow-prefix-ids")

        if self.config.dataset.regions:
            args.extend(["--regions", *self.config.dataset.regions])
        args.extend(_compute_region_masks_args(self.config.dataset))
        return args

    def create_script(self) -> ScriptBuilder:
        return _new_script("champollion_pipeline.run_cortical_tiles", "RunCorticalTiles")

    def subject_input_dir(self) -> str | None:
        return existing_dir(_compute_morphologist_subjects_dir(self.config.dataset))

    def required_file_patterns(self) -> list[str]:
        g = self.config.dataset.path_to_graph
        s = self.config.dataset.path_sk_with_hull
        graph_patterns = (
            [f"{g}/L*.arg", f"{g}/R*.arg"] if g is not None else []
        )  # none: stage 2's input QC finds the graphs
        return [*graph_patterns, *(f"{s}/{compute_skeleton_pattern(side)}" for side in SKELETON_SIDES)]

    def output_dir(self) -> str | None:
        d = self.config.dataset.cortical_tiles_output
        return d if d else None


class GenerateChampollionConfigStage(PipelineStage):
    """Stage for generating Champollion configuration."""

    progress_message = "Generating Champollion configuration..."
    success_message = "Champollion config generated"
    failure_message = "Config generation failed"

    def validate(self) -> bool:
        """Validate that crops exist."""
        crops_path = Path(self.config.dataset.crops_path)
        if not crops_path.exists():
            self.logger.warning(f"Crops path does not exist yet: {crops_path}")
        return True

    def build_args(self) -> list[str]:
        dataset = self.config.dataset
        return [
            str(dataset.crops_path),
            f"--dataset={dataset.name}",
            *_compute_masks_args(dataset),
            *_compute_region_masks_args(dataset),
        ]

    def create_script(self) -> ScriptBuilder:
        return _new_script("champollion_pipeline.generate_champollion_config", "GenerateChampollionConfig")

    def subject_input_dir(self) -> str | None:
        return existing_dir(self.config.dataset.crops_path)

    def output_dir(self) -> str | None:
        return _compute_configs_dir(self.config.dataset) or None


def _compute_configs_dir(dataset: DatasetConfig) -> str:
    """Stage 3's configs root <D>/<name>/derivatives/champollion_V1/configs, D the parent of the name folder in
    crops_path; "" when crops_path or name is empty or crops_path does not hold name. Pure, O(1)."""
    if not dataset.crops_path or not dataset.name:
        return ""
    try:
        parent = find_dataset_folder(dataset.crops_path, dataset.name)
    except ValueError:
        return ""
    return compute_champollion_configs_root(parent, dataset.name)


def _compute_stage4_crops_dir(dataset: DatasetConfig) -> str:
    """The 2mm crops folder generate_embeddings reads (on a map run, that of the map's first Mask version in
    MASK_VERSIONS order); "" without datasets_root. Pure, O(56)."""
    dirs = _compute_stage4_crops_dirs(dataset)
    return dirs[0] if dirs else ""


def _compute_stage4_crops_dirs(dataset: DatasetConfig) -> list[str]:
    """The 2mm crops folder of each Mask version whose crops stage 4 reads; [] without datasets_root. Pure, O(56)."""
    if not dataset.datasets_root:
        return []
    derivatives = os.path.join(dataset.datasets_root, "derivatives", DERIVATIVES_FOLDER)
    return [compute_versioned_crops_dir(derivatives, version) for version in compute_crops_versions(dataset)]


_HF_REPO_ID_REQUIRED = "hf_enabled requires dataset.hf_repo_id"


@contextmanager
def _scoped_env(name: str, value: str | None) -> Iterator[None]:
    """Set env var *name* to *value* for the duration of the block, then restore.

    A no-op when *value* is falsy.
    """
    if not value:
        yield
        return
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _compute_models_cache(dataset: DatasetConfig) -> str:
    """The models cache root stage 4 uses: abspath(dataset.models_cache) when set, else the dataset-tree default."""
    if dataset.models_cache is not None:
        return os.path.abspath(dataset.models_cache)
    return compute_models_cache_dir(dataset.datasets_root)


def _compute_models_source(config: PipelineConfig) -> str:
    """Models source of the embeddings: the Hugging Face repo id when enabled, else the models source cascade.

    The cascade's models cache entry is keyed by masks_version, by DEFAULT_MASKS_VERSION on a map run (the key the map
    was resolved with: the run label of a mixed map names no model set).
    """
    dataset = config.dataset
    if dataset.hf_enabled:
        return dataset.hf_repo_id
    cache_key = DEFAULT_MASKS_VERSION if dataset.region_masks else dataset.masks_version
    return resolve_models_source(str(config.models_path), config.root_path, _compute_models_cache(dataset), cache_key)


class GenerateEmbeddingsStage(PipelineStage):
    """Stage for generating embeddings."""

    progress_message = "Generating embeddings and training classifiers..."
    success_message = "Embeddings generated"
    failure_message = "Embedding generation failed"
    _models_source: str | None = None

    def validate(self) -> bool:
        """Validate that the model source is usable."""
        dataset = self.config.dataset
        if dataset.hf_enabled:
            if not dataset.hf_repo_id:
                self.logger.error(_HF_REPO_ID_REQUIRED)
                return False
            return True
        if not self.config.models_path:
            return True
        models_path = Path(self.config.models_path)
        if not models_path.exists():
            self.logger.error(f"Models path does not exist: {models_path}")
            return False
        return True

    def precheck(self) -> StageResult | None:
        dataset = self.config.dataset
        if dataset.hf_enabled and not dataset.hf_repo_id:
            return StageResult(
                stage_name=self.name,
                success=False,
                message=_HF_REPO_ID_REQUIRED,
                return_code=1,
            )
        return None

    def build_args(self) -> list[str]:
        dataset = self.config.dataset
        self._models_source = _compute_models_source(self.config)
        args = [
            self._models_source,
            dataset.datasets_root,
            *_compute_masks_args(dataset),
            f"--model-version={dataset.masks_version}",
        ]

        region_embeddings_dir = _compute_region_embeddings_source(dataset)
        if region_embeddings_dir:
            args.append(f"--output={region_embeddings_dir}")
        if dataset.overwrite:
            args.append("--overwrite")
        if dataset.cpu:
            args.append("--cpu")
        args.append(f"--checkpoint-epoch={dataset.checkpoint_epoch}")
        args.extend(f"--checkpoint-epoch={entry}" for entry in dataset.checkpoint_epochs)
        if dataset.per_model_folders:
            args.append("--per-model-folders")
        if dataset.models_cache is not None:
            args.append(f"--models-cache={dataset.models_cache}")
        if dataset.allow_version_mismatch:
            args.append("--allow-version-mismatch")
        if dataset.regions:
            args.extend(["--regions", *compute_region_model_names(dataset.regions)])
        args.extend(_compute_region_masks_args(dataset))
        return args

    def create_script(self) -> ScriptBuilder:
        return _new_script("champollion_pipeline.generate_embeddings", "GenerateEmbeddings")

    def run_script(self, script: ScriptBuilder) -> int:
        dataset = self.config.dataset
        hf_token = dataset.hf_token if dataset.hf_enabled else None
        with _scoped_env("HF_TOKEN", hf_token):
            status = script.run()
        if status != 0 and self._used_hf_fallback():
            self.logger.error(
                f"Stage 4 could not obtain the model set from Hugging Face {HF_FALLBACK_REPO_ID} (no "
                f"{os.path.join(self.config.root_path, 'models')}, no entry in the models cache "
                f"{_compute_models_cache(dataset)}); pass --models-path <folder> to use a local model set"
            )
        return status

    def _used_hf_fallback(self) -> bool:
        """True when build_args took the Hugging Face repository as the last step of the models source cascade."""
        dataset = self.config.dataset
        return not dataset.hf_enabled and not self.config.models_path and self._models_source == HF_FALLBACK_REPO_ID

    def subject_input_dir(self) -> str | None:
        return existing_dir(_compute_stage4_crops_dir(self.config.dataset))

    def index_input_dirs(self) -> list[str]:
        dataset = self.config.dataset
        crops_dirs = [existing_dir(d) for d in _compute_stage4_crops_dirs(dataset)]
        return [d for d in (*crops_dirs, _compute_configs_dir(dataset)) if d]

    def output_dir(self) -> str | None:
        return _compute_region_embeddings_source(self.config.dataset) or None


def _compute_region_embeddings_source(dataset: DatasetConfig) -> str:
    """Return the per-region embeddings directory shared by stage 4 (output) and stage 5 (input).

    An explicit embeddings_path wins; otherwise the versioned default under datasets_root;
    otherwise "" (no default can be derived). Pure, no I/O, O(1).
    """
    if dataset.embeddings_path:
        return dataset.embeddings_path
    if dataset.datasets_root:
        return compute_region_embeddings_dir(dataset.datasets_root, dataset.masks_version)
    return ""


def _compute_snapshots_output_dir(dataset: DatasetConfig) -> str:
    """Return the stage 6 output directory.

    An explicit snapshots_path wins; otherwise the versioned default under datasets_root;
    otherwise "". Pure, no I/O, O(1).
    """
    if dataset.snapshots_path:
        return dataset.snapshots_path
    if dataset.datasets_root:
        return compute_snapshots_dir(dataset.datasets_root, dataset.masks_version)
    return ""


def _compute_combined_output_dir(config: PipelineConfig) -> str:
    """Stage 5's --output_path: combined folder under datasets_root, else cortical_tiles_output or outputs_path."""
    dataset = config.dataset
    if dataset.datasets_root:
        return str(Path(compute_combined_embeddings_dir(dataset.datasets_root, dataset.masks_version)))
    return dataset.cortical_tiles_output or config.outputs_path


def _compute_snapshots_embeddings_dir(dataset: DatasetConfig) -> str:
    """Stage 6's --embeddings_dir: the combined folder under datasets_root, else embeddings_path. Pure, O(1)."""
    if dataset.datasets_root:
        return compute_combined_embeddings_dir(dataset.datasets_root, dataset.masks_version)
    return dataset.embeddings_path


def _compute_morphologist_subjects_dir(dataset: DatasetConfig) -> str:
    """Return the Morphologist subjects folder read by stage 2, its pre-check, stage 6 and the streaming mode.

    An explicit morphologist_graphs wins; otherwise the folder stage 1 writes under datasets_root;
    otherwise "". Pure, no I/O, O(1).
    """
    if dataset.morphologist_graphs:
        return dataset.morphologist_graphs
    if dataset.datasets_root:
        return stage1_subjects_dir(dataset.datasets_root)
    return ""


class PutTogetherEmbeddingsStage(PipelineStage):
    """Stage for combining embeddings."""

    progress_message = "Putting together embeddings..."
    success_message = "Embeddings combined successfully"
    failure_message = "Combine failed"

    def validate(self) -> bool:
        """Validate that embeddings exist."""
        embeddings_path = Path(_compute_region_embeddings_source(self.config.dataset))
        if not embeddings_path.exists():
            self.logger.error(f"Embeddings path does not exist: {embeddings_path}")
            return False
        return True

    def build_args(self) -> list[str]:
        dataset = self.config.dataset
        output_path = _compute_combined_output_dir(self.config)
        args = [_compute_region_embeddings_source(dataset), f"--output_path={output_path}"]
        if dataset.regions:
            args.extend(["--regions", *compute_region_model_names(dataset.regions)])
        if dataset.model:
            args.extend(["--model", dataset.model])
        return args

    def create_script(self) -> ScriptBuilder:
        return _new_script("champollion_pipeline.put_together_embeddings", "PutTogetherEmbeddings")

    def subject_input_dir(self) -> str | None:
        return existing_dir(_compute_region_embeddings_source(self.config.dataset))

    def output_dir(self) -> str | None:
        return _compute_combined_output_dir(self.config) or None


class GenerateSnapshotsStage(PipelineStage):
    """Stage for generating visualization snapshots."""

    progress_message = "Generating visualization snapshots..."
    success_message = "Snapshots generated"
    failure_message = "Snapshot generation failed"

    def validate(self) -> bool:
        """Validate that embeddings exist and output path is set."""
        if not self.config.dataset.snapshots_path and not self.config.dataset.datasets_root:
            self.logger.error("snapshots_path or datasets_root must be set in config")
            return False
        embeddings_path = Path(self.config.dataset.embeddings_path)
        if self.config.dataset.embeddings_path and not embeddings_path.exists():
            self.logger.error(f"Embeddings path does not exist: {embeddings_path}")
            return False
        return True

    def build_args(self) -> list[str]:
        dataset = self.config.dataset
        args = [f"--output_dir={_compute_snapshots_output_dir(dataset)}"]
        embeddings_dir = _compute_snapshots_embeddings_dir(dataset)
        if embeddings_dir:
            args.append(f"--embeddings_dir={embeddings_dir}")
        morphologist_dir = _compute_morphologist_subjects_dir(dataset)
        if morphologist_dir:
            args.append(f"--morphologist_dir={morphologist_dir}")
        if self.config.dataset.crops_path:
            args.append(f"--cortical_tiles_dir={self.config.dataset.crops_path}")
        if self.config.dataset.reference_data_path:
            args.append(f"--reference_data_dir={self.config.dataset.reference_data_path}")
        else:
            self.logger.warning(
                "UMAP snapshots skipped: no reference data; set --reference-data-path to render them "
                "(sulcal graph and cortical tiles snapshots still run)"
            )
        if self.config.dataset.champollion_data_root:
            args.append(f"--champollion_data_root={self.config.dataset.champollion_data_root}")
        if dataset.labelling_session:
            args.append(f"--labelling_session={dataset.labelling_session}")
        if dataset.model:
            args.append(f"--model={dataset.model}")
        return args

    def create_script(self) -> ScriptBuilder:
        return _new_script("champollion_pipeline.generate_snapshots", "GenerateSnapshots")

    def subject_input_dir(self) -> str | None:
        return existing_dir(_compute_snapshots_embeddings_dir(self.config.dataset))

    def output_dir(self) -> str | None:
        return _compute_snapshots_output_dir(self.config.dataset) or None
