#!/usr/bin/env python3
"""
Champollion Pipeline Orchestrator

This is the main entry point for the Champollion pipeline.
It orchestrates all pipeline stages using a simple configuration management system.

Usage:
    # Run the default config (enables only generate_embeddings)
    python main.py

    # Run the full pipeline (all stages; needs a config file)
    python main.py --config configs/my_config.yaml --enable-all-stages

    # Run with specific config file
    python main.py --config configs/my_config.yaml

    # Run specific stages only
    python main.py --stages generate_embeddings put_together_embeddings

    # Run with overrides
    python main.py --dataset-name test_data --verbose
"""

import argparse
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from types import ModuleType
from typing import ClassVar

from champollion_utils.process import active_profiling, start_profiling
from champollion_utils.script_builder import ScriptBuilder
from champollion_utils.update_check import check_for_updates

from champollion_pipeline._stage_base import PARTIAL_STATUS, PipelineStage, StageResult
from champollion_pipeline.cli_options import add_config_arguments, apply_config_arguments
from champollion_pipeline.config import ConfigLoader, DatasetConfig, PipelineConfig
from champollion_pipeline.derivatives_layout import (
    compute_combined_embeddings_dir,
    compute_region_embeddings_dir,
    compute_region_model_names,
    compute_snapshots_dir,
)
from champollion_pipeline.file_indexer import SubjectEligibilityChecker
from champollion_pipeline.process_setup import init_pipeline_process
from champollion_pipeline.streaming import StreamingSettings
from champollion_pipeline.utils.input_kind import (
    InputKind,
    detect_input_kind,
    resolve_subjects_dir,
    stage1_subjects_dir,
)
from champollion_pipeline.utils.skeleton_qc import SKELETON_SIDES, compute_skeleton_pattern

logger = logging.getLogger("champollion_pipeline.main")

# Stage 1, skipped when the input already holds Morphologist outputs
_MORPHOLOGIST_STAGE = "generate_morphologist_graphs"

SCRIPT_NAME = "champollion_pipeline"
SCRIPT_DESCRIPTION = "Champollion Pipeline Orchestrator"
_HELP_EPILOG = """
Examples:
  # Run with default config
  python main.py

  # Run with custom config
  python main.py --config configs/my_experiment.yaml

  # Run specific stages
  python main.py --stages generate_embeddings

  # Enable all stages
  python main.py --enable-all-stages

  # Generate template config
  python main.py --generate-config my_config.yaml
        """


# ====================== Pipeline Stage Strategy Pattern ======================


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
        from champollion_pipeline.generate_morphologist_graphs import GenerateMorphologistGraphs

        return GenerateMorphologistGraphs()


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
            f"--path_to_graph={self.config.dataset.path_to_graph}",
            f"--path_sk_with_hull={self.config.dataset.path_sk_with_hull}",
            f"--njobs={self.config.dataset.njobs}",
            f"--masks={self.config.dataset.masks_version}",
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
        return args

    def create_script(self) -> ScriptBuilder:
        from champollion_pipeline.run_cortical_tiles import RunCorticalTiles

        return RunCorticalTiles()

    def subject_input_dir(self) -> str | None:
        d = _compute_morphologist_subjects_dir(self.config.dataset)
        return d if d and os.path.isdir(d) else None

    def required_file_patterns(self) -> list[str]:
        g = self.config.dataset.path_to_graph
        s = self.config.dataset.path_sk_with_hull
        return [f"{g}/L*.arg", f"{g}/R*.arg", *(f"{s}/{compute_skeleton_pattern(side)}" for side in SKELETON_SIDES)]

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
        return [
            str(self.config.dataset.crops_path),
            f"--dataset={self.config.dataset.name}",
            f"--masks={self.config.dataset.masks_version}",
        ]

    def create_script(self) -> ScriptBuilder:
        from champollion_pipeline.generate_champollion_config import GenerateChampollionConfig

        return GenerateChampollionConfig()


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


class GenerateEmbeddingsStage(PipelineStage):
    """Stage for generating embeddings."""

    progress_message = "Generating embeddings and training classifiers..."
    success_message = "Embeddings generated"
    failure_message = "Embedding generation failed"

    def validate(self) -> bool:
        """Validate that the model source is usable."""
        dataset = self.config.dataset
        if dataset.hf_enabled:
            if not dataset.hf_repo_id:
                self.logger.error(_HF_REPO_ID_REQUIRED)
                return False
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
        models_path = _compute_models_source(self.config)
        args = [models_path, dataset.datasets_root, f"--masks={dataset.masks_version}"]
        if dataset.hf_enabled:
            args.append(f"--model-version={dataset.masks_version}")

        region_embeddings_dir = _compute_region_embeddings_source(dataset)
        if region_embeddings_dir:
            args.append(f"--output={region_embeddings_dir}")
        if dataset.overwrite:
            args.append("--overwrite")
        if dataset.cpu:
            args.append("--cpu")
        args.append(f"--checkpoint-epoch={dataset.checkpoint_epoch}")
        if dataset.models_cache is not None:
            args.append(f"--models-cache={dataset.models_cache}")
        if dataset.regions:
            args.extend(["--regions", *compute_region_model_names(dataset.regions)])
        return args

    def create_script(self) -> ScriptBuilder:
        from champollion_pipeline.generate_embeddings import GenerateEmbeddings

        return GenerateEmbeddings()

    def run_script(self, script: ScriptBuilder) -> int:
        dataset = self.config.dataset
        hf_token = dataset.hf_token if dataset.hf_enabled else None
        with _scoped_env("HF_TOKEN", hf_token):
            return script.run()


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
        output_path = (
            Path(compute_combined_embeddings_dir(dataset.datasets_root, dataset.masks_version))
            if dataset.datasets_root
            else (dataset.cortical_tiles_output or self.config.outputs_path)
        )
        args = [_compute_region_embeddings_source(dataset), f"--output_path={output_path}"]
        if dataset.regions:
            args.extend(["--regions", *compute_region_model_names(dataset.regions)])
        return args

    def create_script(self) -> ScriptBuilder:
        from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings

        return PutTogetherEmbeddings()


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
        if dataset.datasets_root:
            combined_dir = compute_combined_embeddings_dir(dataset.datasets_root, dataset.masks_version)
            args.append(f"--embeddings_dir={combined_dir}")
        elif dataset.embeddings_path:
            args.append(f"--embeddings_dir={dataset.embeddings_path}")
        morphologist_dir = _compute_morphologist_subjects_dir(dataset)
        if morphologist_dir:
            args.append(f"--morphologist_dir={morphologist_dir}")
        if self.config.dataset.crops_path:
            args.append(f"--cortical_tiles_dir={self.config.dataset.crops_path}")
        if self.config.dataset.reference_data_path:
            args.append(f"--reference_data_dir={self.config.dataset.reference_data_path}")
        if self.config.dataset.champollion_data_root:
            args.append(f"--champollion_data_root={self.config.dataset.champollion_data_root}")
        return args

    def create_script(self) -> ScriptBuilder:
        from champollion_pipeline.generate_snapshots import GenerateSnapshots

        return GenerateSnapshots()


# ====================== Pipeline Orchestrator ======================


class PipelineOrchestrator:
    """Orchestrates the execution of pipeline stages."""

    # Stage registry mapping names to classes
    STAGE_REGISTRY: ClassVar[dict[str, type[PipelineStage]]] = {
        "generate_morphologist_graphs": GenerateMorphologistGraphsStage,
        "run_cortical_tiles": RunCorticalTilesStage,
        "generate_champollion_config": GenerateChampollionConfigStage,
        "generate_embeddings": GenerateEmbeddingsStage,
        "put_together_embeddings": PutTogetherEmbeddingsStage,
        "generate_snapshots": GenerateSnapshotsStage,
    }

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.stages: dict[str, PipelineStage] = {}
        self._register_stages()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("champollion_pipeline")
        logger.setLevel(getattr(logging, self.config.log_level))
        logger.handlers = []  # Clear existing handlers
        # Console output goes through the root logger's unified handlers (init_pipeline_process)
        logger.propagate = self.config.log_to_console

        # File handler
        if self.config.log_to_file:
            log_dir = Path(self.config.log_dir) if self.config.log_dir else Path(self.config.outputs_path) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "pipeline.log")
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _register_stages(self):
        """Register all available pipeline stages."""
        for stage_name, stage_class in self.STAGE_REGISTRY.items():
            self.stages[stage_name] = stage_class(name=stage_name, config=self.config, logger=self.logger)

    def _get_enabled_stages(self) -> list[str]:
        """Get list of enabled stages in dependency order."""
        enabled = [name for name, enabled in self.config.stages.items() if enabled]

        # Sort by dependencies (topological sort)
        sorted_stages = []
        processed = set()

        def add_stage_with_deps(stage_name: str):
            if stage_name in processed:
                return
            # Add dependencies first
            for dep in self.config.dependencies.get(stage_name, []):
                if dep in enabled:
                    add_stage_with_deps(dep)
            # Then add this stage
            if stage_name not in sorted_stages:
                sorted_stages.append(stage_name)
            processed.add(stage_name)

        for stage_name in enabled:
            add_stage_with_deps(stage_name)

        return sorted_stages

    def _check_dependencies(self, stage_name: str, completed: list[str]) -> bool:
        """Check if stage dependencies are satisfied."""
        depends_on = self.config.dependencies.get(stage_name, [])

        for dep in depends_on:
            if self.config.stages.get(dep, False) and dep not in completed:
                self.logger.error(f"Stage '{stage_name}' depends on '{dep}' which hasn't completed")
                return False

        return True

    def _index_path(self, phase: str, stage_name: str) -> str | None:
        """Path of the pre/post index JSON of a stage, or None without outputs_path."""
        if not self.config.outputs_path:
            return None
        return os.path.join(self.config.outputs_path, f"index_{phase}_{stage_name}.json")

    def _report_pre_stage(self, stage_name: str, stage: PipelineStage) -> None:
        """Index the stage's input subjects and print the eligibility report."""
        input_dir = stage.subject_input_dir()
        if not input_dir:
            return
        patterns = stage.required_file_patterns()
        if not patterns:
            return
        if self.config.outputs_path:
            os.makedirs(self.config.outputs_path, exist_ok=True)
        index_path = self._index_path("pre", stage_name)
        checker = SubjectEligibilityChecker(
            input_dir,
            patterns,
            stage_name=stage_name,
            bids=self.config.dataset.bids,
            path_to_graph=self.config.dataset.path_to_graph,
        )
        checker.check_subject_patterns(save_index_to=index_path).log_summary()

    def _run_stage(self, stage_name: str, completed_stages: list[str], partial_stages: list[str]) -> bool:
        """Check, validate, report and execute one registered stage; True on success."""
        if not self._check_dependencies(stage_name, completed_stages):
            return False
        stage = self.stages[stage_name]
        if not stage.validate():
            self.logger.error(f"Stage '{stage_name}' validation failed")
            return False
        self._report_pre_stage(stage_name, stage)
        with _stage_segment(stage_name):
            result = stage.execute()
        if result.partial:
            partial_stages.append(stage_name)
        if result.success:
            return True
        if self.config.stop_on_error:
            self.logger.error("Stopping pipeline due to stage failure")
        return False

    def _log_summary(self, completed_stages: list[str], failed_stages: list[str], partial_stages: list[str]) -> int:
        """Log the execution summary and return the pipeline exit code."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Pipeline Execution Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"Completed stages: {completed_stages}")
        if failed_stages:
            self.logger.info(f"Failed stages: {failed_stages}")
            self.logger.error("❌ Pipeline completed with errors")
            return 1
        if partial_stages:
            self.logger.warning(f"⚠️ Pipeline completed with partial stages: {partial_stages}")
            return PARTIAL_STATUS
        self.logger.info("✅ Pipeline completed successfully")
        return 0

    def run(self) -> int:
        """Run the pipeline."""
        self.logger.info("🚀 Starting Champollion Pipeline")
        self.logger.info(f"Root path: {self.config.root_path}")
        self.logger.info(f"Dataset: {self.config.dataset.name}")

        enabled_stages = self._get_enabled_stages()
        self.logger.info(f"Enabled stages: {enabled_stages}\n")

        if not enabled_stages:
            self.logger.warning("No stages enabled. Nothing to do.")
            return 0

        completed_stages = []
        failed_stages = []
        partial_stages = []

        for stage_name in enabled_stages:
            if stage_name not in self.stages:
                self.logger.warning(f"Stage '{stage_name}' not registered. Skipping.")
                continue
            if self._run_stage(stage_name, completed_stages, partial_stages):
                completed_stages.append(stage_name)
                continue
            failed_stages.append(stage_name)
            if self.config.stop_on_error:
                break

        return self._log_summary(completed_stages, failed_stages, partial_stages)


# ====================== CLI Interface ======================


def create_default_config() -> PipelineConfig:
    """Create default configuration."""
    root_path = Path(__file__).parent
    config = PipelineConfig(
        root_path=str(root_path),
        data_path=str(root_path / "data"),
        models_path=str(root_path / "models"),
        outputs_path=str(root_path / "outputs"),
        champollion_v1_path=str(root_path / "external" / "champollion_V1"),
    )
    config.dataset.datasets_root = _compute_default_datasets_root(config)
    return config


def _compute_default_datasets_root(config: PipelineConfig) -> str:
    """The datasets_root of a default config: <data_path>/<dataset name>."""
    return str(Path(config.data_path) / config.dataset.name)


def _should_rederive_datasets_root(args: argparse.Namespace) -> bool:
    """True when the command line moves the default dataset folder without naming its root."""
    return args.datasets_root is None and (args.data_path is not None or args.dataset_name is not None)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    return ChampollionPipeline().parse_args()


def _profiling_session() -> ModuleType | None:
    """The profiling module of this process when a session is open, else None; never imports it."""
    return active_profiling()


def _start_config_profiling(config: PipelineConfig, args: argparse.Namespace) -> None:
    """Start profiling from config.profile_dir when neither --profile nor a parent already started it."""
    if args.profile is not None or not config.profile_dir or _profiling_session() is not None:
        return
    start_profiling(config.profile_dir, None, stage=SCRIPT_NAME)


def _stage_segment(stage_name: str) -> AbstractContextManager:
    """Profiling segment named after the stage when a session is open, else a no-op context."""
    session = _profiling_session()
    if session is None:
        return nullcontext()
    return session.profile_segment(stage_name)


def _apply_cli_overrides(config: PipelineConfig, args: argparse.Namespace, *, from_default: bool) -> None:
    """Apply command-line argument overrides to a PipelineConfig in place.

    ``from_default`` must be ``True`` when *config* was produced by
    :func:`create_default_config` (no ``--config`` file was supplied) and
    ``False`` when it was loaded from a YAML file.  The flag gates the
    REQ-DEFROOT dataset-root re-derivation that must only fire for the
    default config (REQ-DEFROOT-03/04).
    """
    apply_config_arguments(config, args)
    if args.enable_all_stages:
        config.stages = dict.fromkeys(config.stages, True)
    if from_default and _should_rederive_datasets_root(args):
        # Default config only: re-derive the root for the new folder (REQ-DEFROOT-03); a --config YAML
        # datasets_root is never rewritten (REQ-DEFROOT-04), and an explicit --datasets-root wins.
        config.dataset.datasets_root = _compute_default_datasets_root(config)


def _save_config_template(args: argparse.Namespace) -> int:
    """Write the default configuration, with the command-line overrides, to --generate-config."""
    config = create_default_config()
    _apply_cli_overrides(config, args, from_default=True)
    ConfigLoader.save_to_yaml(config, args.generate_config)
    logger.output(f"✅ Configuration template generated: {args.generate_config}")
    return 0


def _load_config(args: argparse.Namespace) -> PipelineConfig:
    """Load the --config YAML file, or the default configuration without one."""
    if args.config:
        logger.trace(f"Loading configuration from: {args.config}")
        return ConfigLoader.load_from_yaml(args.config)
    logger.trace("Using default configuration")
    return create_default_config()


def _compute_models_source(config: PipelineConfig) -> str:
    """Models source of the embeddings: the Hugging Face repo id when enabled, else the local models path."""
    dataset = config.dataset
    return dataset.hf_repo_id if dataset.hf_enabled else str(config.models_path)


def _compute_streaming_log_dir(config: PipelineConfig, run_log_path: str | None) -> str:
    """Worker and consumer log folder: beside main.py's run log, under --log DIR, or the default logs folder."""
    if run_log_path is not None:
        return run_log_path.removesuffix(".log")
    if config.log:
        return config.log
    return config.log_dir or str(Path(config.outputs_path) / "logs")


def _build_streaming_settings(
    config: PipelineConfig, args: argparse.Namespace, run_log_path: str | None
) -> StreamingSettings:
    """StreamingSettings of this run from the overridden configuration."""
    return StreamingSettings(
        subjects_dir=_compute_morphologist_subjects_dir(config.dataset),
        output_dir=config.outputs_path,
        log_dir=_compute_streaming_log_dir(config, run_log_path),
        n_workers=config.n_workers,
        worker_timeout=config.worker_timeout,
        dry_run=args.dry_run,
    )


def _run_streaming_mode(config: PipelineConfig, args: argparse.Namespace, *, run_log_path: str | None) -> int:
    """Run the scan-centric streaming mode after its two guards."""
    if not config.dataset.embeddings_only:
        logger.error(
            "--mode streaming requires dataset.embeddings_only: true in the --config YAML "
            "(training aggregates subjects and must remain batch)."
        )
        return 1
    try:
        from champollion_pipeline.streaming.runner import run_parallel_pipeline
    except ImportError as e:
        logger.error(f"could not import champollion_pipeline.streaming.runner: {e}")
        return 1
    return run_parallel_pipeline(
        _build_streaming_settings(config, args, run_log_path), config.dataset, _compute_models_source(config)
    )


def _apply_input_kind(config: PipelineConfig) -> InputKind | None:
    """Fill, in place, the fields the kind of config.dataset.input_path fixes.

    Returns the detected kind, or None when input_path is empty (nothing detected, nothing changed).
    NEITHER and BOTH change nothing here.
    """
    input_path = config.dataset.input_path
    if not input_path:
        return None
    kind = detect_input_kind(input_path)
    if kind is InputKind.T1:
        _apply_t1_input(config.dataset)
    elif kind is InputKind.MORPHOLOGIST:
        _apply_morphologist_input(config)
    return kind


def _apply_t1_input(dataset: DatasetConfig) -> None:
    """Stage 1 output root of a T1 input: the absolute input_path, when datasets_root is not set."""
    if not dataset.datasets_root:
        dataset.datasets_root = os.path.abspath(dataset.input_path)


def _apply_morphologist_input(config: PipelineConfig) -> None:
    """Subjects folder of a Morphologist input, when morphologist_graphs is not set, then the stage 1 skip.

    AmbiguousSubjectsFolders from resolve_subjects_dir propagates unchanged.
    """
    dataset = config.dataset
    if not dataset.morphologist_graphs:
        folder = resolve_subjects_dir(dataset.input_path)
        if folder is not None:
            dataset.morphologist_graphs = os.path.abspath(folder)
    _skip_morphologist_stage(config)


def _skip_morphologist_stage(config: PipelineConfig) -> None:
    """Disable stage 1 and log it once at INFO; a stage 1 already disabled stays silent."""
    if config.stages.get(_MORPHOLOGIST_STAGE):
        config.stages[_MORPHOLOGIST_STAGE] = False
        logger.info(f"Stage {_MORPHOLOGIST_STAGE} skipped: {config.dataset.input_path} holds Morphologist outputs")


class ChampollionPipeline(ScriptBuilder):
    """main.py as a ScriptBuilder script: the shared run log, profiling and process setup."""

    def __init__(self) -> None:
        self._run_log_path: str | None = None
        super().__init__(SCRIPT_NAME, SCRIPT_DESCRIPTION)
        self._add_pipeline_options()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Keep main.py's raw help layout and its examples epilog."""
        return argparse.ArgumentParser(
            prog=self.script_name,
            description=self.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=_HELP_EPILOG,
            allow_abbrev=False,
        )

    def _add_log_option(self) -> None:
        """Add nothing: main.py's --log takes a DIR (or none), and _add_pipeline_options adds it in place."""

    def _add_pipeline_options(self) -> None:
        """Add the orchestrator options, then one option per config field from the registry."""
        self.parser.add_argument("--config", type=str, help="Path to configuration YAML file")
        self.parser.add_argument("--enable-all-stages", action="store_true", help="Enable all pipeline stages")
        self.parser.add_argument(
            "--generate-config", type=str, metavar="OUTPUT_PATH", help="Generate a template configuration file and exit"
        )
        self.parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Enumerate scans and report without processing (streaming mode only).",
        )
        add_config_arguments(self.parser)

    def _resolve_log_path(self) -> str | None:
        """main.py's own run log: only with a bare --log, stored for the streaming log_dir. args.log is left as is."""
        self._run_log_path = self.default_log_path() if self.args.log == "" else None
        return self._run_log_path

    def build(self) -> "ChampollionPipeline":
        """Check for updates through main.py's own import, then parse the arguments."""
        check_for_updates()
        self.parse_args()
        return self

    def run(self) -> int:
        args = self.args
        if args.generate_config:
            return _save_config_template(args)
        config = _load_config(args)
        _apply_cli_overrides(config, args, from_default=not args.config)
        _start_config_profiling(config, args)
        if config.mode == "streaming":
            return _run_streaming_mode(config, args, run_log_path=self._run_log_path)
        _apply_input_kind(config)
        return PipelineOrchestrator(config).run()


def main() -> int:
    """Main entry point for the pipeline."""
    init_pipeline_process()
    return ChampollionPipeline().main()


if __name__ == "__main__":
    sys.exit(main())
