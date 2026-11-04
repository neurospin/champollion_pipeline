#!/usr/bin/env python3
"""
Champollion Pipeline Orchestrator

This is the main entry point for the Champollion pipeline.
It orchestrates all pipeline stages using a simple configuration management system.

Usage:
    # Run full pipeline with default config
    python main.py

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
import re
import sys
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from types import ModuleType
from typing import ClassVar

from champollion_utils.process import active_profiling, start_profiling
from champollion_utils.script_builder import ScriptBuilder
from champollion_utils.update_check import check_for_updates

from champollion_pipeline._stage_base import PipelineStage, StageResult
from champollion_pipeline.config import ConfigLoader, DatasetConfig, PipelineConfig
from champollion_pipeline.derivatives_layout import (
    HEMISPHERES,
    compute_combined_embeddings_dir,
    compute_region_embeddings_dir,
    compute_region_model_name,
    compute_snapshots_dir,
)
from champollion_pipeline.file_indexer import SubjectEligibilityChecker
from champollion_pipeline.process_setup import init_pipeline_process
from champollion_pipeline.streaming import StreamingSettings

logger = logging.getLogger("champollion_pipeline.main")

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
_LOG_HELP = (
    "Write a run log per stage: DIR/<stage>_<YYYYmmdd-HHMMSS>.log; bare --log: each stage's default log path "
    "(its derivatives .log folder) plus this run's own log at ./.log/champollion_pipeline_<YYYYmmdd-HHMMSS>.log. "
    "Overrides the config file's log."
)


# ====================== Pipeline Stage Strategy Pattern ======================


class GenerateMorphologistGraphsStage(PipelineStage):
    """Stage for generating Morphologist graphs."""

    progress_message = "Generating Morphologist graphs..."
    success_message = "Morphologist graphs generated successfully"
    failure_message = "Morphologist failed"

    def validate(self) -> bool:
        """Validate that input data exists."""
        input_path = Path(self.config.dataset.input_path)
        if not input_path.exists():
            self.logger.error(f"Input path does not exist: {input_path}")
            return False
        return True

    def build_args(self) -> list[str]:
        args = [
            str(self.config.dataset.input_path),
            str(self.config.dataset.morphologist_graphs),
        ]
        if self.config.dataset.bids:
            args.append("--bids")
        if getattr(self.config, "parallel", False):
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
        """Validate that Morphologist graphs exist."""
        graphs_path = Path(self.config.dataset.morphologist_graphs)
        if not graphs_path.exists():
            self.logger.error(f"Morphologist graphs path does not exist: {graphs_path}")
            return False
        return True

    def build_args(self) -> list[str]:
        args = [
            str(self.config.dataset.morphologist_graphs),
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
        d = self.config.dataset.morphologist_graphs
        return d if d and os.path.exists(d) else None

    def required_file_patterns(self) -> list[str]:
        side = "R"
        g = self.config.dataset.path_to_graph
        s = self.config.dataset.path_sk_with_hull
        if self.config.dataset.bids:
            # scan_id.path_prefix already includes the session directory;
            # strip the leading ses-XXX/ segment so patterns are session-relative.
            g = re.sub(r"^ses-[^/]*/+", "", g)
            s = re.sub(r"^ses-[^/]*/+", "", s)
        return [
            f"{g}/{side}*.arg",
            f"{s}/{side}*.nii.gz",
        ]

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
            args.extend(["--regions", *_compute_embeddings_region_names(dataset.regions)])
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


def _compute_embeddings_region_names(regions: list[str]) -> list[str]:
    """Return both hemispheres' model names for each distinct region, in input order. O(n)."""
    distinct_regions = dict.fromkeys(regions)
    return [compute_region_model_name(region, hemisphere) for region in distinct_regions for hemisphere in HEMISPHERES]


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
        return [_compute_region_embeddings_source(dataset), f"--output_path={output_path}"]

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
        if self.config.dataset.morphologist_graphs:
            args.append(f"--morphologist_dir={self.config.dataset.morphologist_graphs}")
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
        checker.check(save_index_to=index_path).log_summary()

    def _run_stage(self, stage_name: str, completed_stages: list[str]) -> bool:
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
        if result.success:
            return True
        if self.config.stop_on_error:
            self.logger.error("Stopping pipeline due to stage failure")
        return False

    def _log_summary(self, completed_stages: list[str], failed_stages: list[str]) -> int:
        """Log the execution summary and return the pipeline exit code."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Pipeline Execution Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"Completed stages: {completed_stages}")
        if failed_stages:
            self.logger.info(f"Failed stages: {failed_stages}")
            self.logger.error("❌ Pipeline completed with errors")
            return 1
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

        for stage_name in enabled_stages:
            if stage_name not in self.stages:
                self.logger.warning(f"Stage '{stage_name}' not registered. Skipping.")
                continue
            if self._run_stage(stage_name, completed_stages):
                completed_stages.append(stage_name)
                continue
            failed_stages.append(stage_name)
            if self.config.stop_on_error:
                break

        return self._log_summary(completed_stages, failed_stages)


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
    config.dataset.datasets_root = str(Path(config.data_path) / config.dataset.name)
    return config


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    return ChampollionPipeline().parse_args()


def _apply_stage_overrides(config: PipelineConfig, args: argparse.Namespace) -> None:
    """Apply --stages and --enable-all-stages."""
    if args.stages:
        # Disable all stages, then enable specified ones
        for stage in config.stages:
            config.stages[stage] = False
        for stage in args.stages:
            config.stages[stage] = True

    if args.enable_all_stages:
        for stage in config.stages:
            config.stages[stage] = True


def _apply_dataset_path_overrides(config: PipelineConfig, args: argparse.Namespace, *, from_default: bool) -> None:
    """Apply --dataset-name (re-deriving the default config's datasets_root) and --models-path."""
    if args.dataset_name:
        config.dataset.name = args.dataset_name
        if from_default:
            # Default config only: re-derive the root for the new name (REQ-DEFROOT-03).
            # A --config YAML datasets_root is never rewritten (REQ-DEFROOT-04).
            config.dataset.datasets_root = str(Path(config.data_path) / config.dataset.name)

    if args.models_path:
        config.models_path = args.models_path


def _apply_runtime_overrides(config: PipelineConfig, args: argparse.Namespace) -> None:
    """Apply --verbose, --bids, --allow-prefix-ids, --mode, --n-workers, --worker-timeout and --log."""
    if args.verbose:
        config.verbose = True
        config.log_level = "DEBUG"

    if args.bids:
        config.dataset.bids = True
    if args.allow_prefix_ids:
        config.dataset.allow_prefix_ids = True

    if args.mode:
        config.mode = args.mode
    if args.n_workers is not None:
        config.n_workers = args.n_workers
    if args.worker_timeout is not None:
        config.worker_timeout = args.worker_timeout
    if args.log is not None:
        config.log = args.log
    if args.profile is not None:
        config.profile_dir = args.profile


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
    _apply_stage_overrides(config, args)
    _apply_dataset_path_overrides(config, args, from_default=from_default)
    _apply_runtime_overrides(config, args)


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
        subjects_dir=config.dataset.input_path or config.dataset.morphologist_graphs,
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
        )

    def _add_log_option(self) -> None:
        """Add nothing: main.py's --log takes a DIR (or none), and _add_pipeline_options adds it in place."""

    def _add_pipeline_options(self) -> None:
        """Add the orchestrator options."""
        self.parser.add_argument("--config", type=str, help="Path to configuration YAML file")

        self.parser.add_argument(
            "--stages",
            nargs="+",
            choices=list(PipelineOrchestrator.STAGE_REGISTRY.keys()),
            help="Specific stages to run (overrides config)",
        )

        self.parser.add_argument("--enable-all-stages", action="store_true", help="Enable all pipeline stages")

        self.parser.add_argument("--dataset-name", type=str, help="Dataset name (overrides config)")

        self.parser.add_argument("--models-path", type=str, help="Path to models directory (overrides config)")

        self.parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

        self.parser.add_argument(
            "--generate-config", type=str, metavar="OUTPUT_PATH", help="Generate a template configuration file and exit"
        )

        self.parser.add_argument(
            "--bids",
            action="store_true",
            help=(
                "Input database follows BIDS layout "
                "(subjects/sub-*/ses-*/...). "
                "Activates BIDS-aware scan enumeration in eligibility checks "
                "and passes --bids to cortical_tiles."
            ),
        )

        self.parser.add_argument(
            "--allow-prefix-ids",
            action="store_true",
            help=(
                "TEMPORARY: let cortical_tiles run when a subject ID is a strict prefix "
                "of another (e.g. sub-1 and sub-10); its transforms may then be wrong."
            ),
        )

        self.parser.add_argument(
            "--mode",
            choices=["sequential", "streaming"],
            default=None,
            help=(
                "Execution mode: 'sequential' (default, stage-centric batch) or "
                "'streaming' (scan-centric parallel, one worker per scan). "
                "Streaming mode requires dataset.embeddings_only: true in the --config YAML."
            ),
        )
        self.parser.add_argument(
            "--n-workers",
            type=int,
            default=None,
            metavar="N",
            help="Number of parallel scan workers (streaming mode only). Defaults to os.cpu_count().",
        )
        self.parser.add_argument(
            "--worker-timeout",
            type=int,
            default=None,
            metavar="SECONDS",
            help=(
                "Timeout in seconds of the whole producer-worker run of one scan; "
                "a longer run ends FAILED (default: 7200)."
            ),
        )
        self.parser.add_argument(
            "--log",
            nargs="?",
            const="",
            default=None,
            metavar="DIR",
            help=_LOG_HELP,
        )
        self.parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Enumerate scans and report without processing (streaming mode only).",
        )

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
        return PipelineOrchestrator(config).run()


def main() -> int:
    """Main entry point for the pipeline."""
    init_pipeline_process()
    return ChampollionPipeline().main()


if __name__ == "__main__":
    sys.exit(main())
