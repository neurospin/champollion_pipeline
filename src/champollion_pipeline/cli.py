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

    # Config-less run: infer every path from the input, run stages 1-6 (T1 files) or 2-6 (Morphologist outputs)
    python main.py /data/my_dataset

    # Read HF_TOKEN and the CHAMPOLLION_* variables from ./.env (a bare --env goes after INPUT)
    python main.py /data/my_dataset --env

    # Run with overrides
    python main.py --dataset-name test_data --verbose
"""

import argparse
import logging
import os
import sys
from collections.abc import Iterator as Iterator
from contextlib import AbstractContextManager, nullcontext
from contextlib import contextmanager as contextmanager
from pathlib import Path
from types import ModuleType
from typing import ClassVar

from champollion_utils.process import active_profiling, start_profiling
from champollion_utils.script_builder import ScriptBuilder
from champollion_utils.update_check import check_for_updates

from champollion_pipeline._stage_base import PARTIAL_STATUS, PipelineStage, StageIndexer
from champollion_pipeline._stage_base import StageResult as StageResult
from champollion_pipeline._stage_base import existing_dir as existing_dir
from champollion_pipeline.cli_options import (
    ArgumentConflictError,
    add_config_arguments,
    add_path_arguments,
    apply_config_arguments,
    fold_path_arguments,
)
from champollion_pipeline.config import ConfigLoader, DatasetConfig, PipelineConfig
from champollion_pipeline.derivatives_layout import compute_champollion_configs_root as compute_champollion_configs_root
from champollion_pipeline.derivatives_layout import compute_combined_embeddings_dir as compute_combined_embeddings_dir
from champollion_pipeline.derivatives_layout import compute_region_embeddings_dir as compute_region_embeddings_dir
from champollion_pipeline.derivatives_layout import compute_region_model_names as compute_region_model_names
from champollion_pipeline.derivatives_layout import compute_snapshots_dir as compute_snapshots_dir
from champollion_pipeline.env_file import EnvFileError, add_env_argument, apply_env_values, read_env_values
from champollion_pipeline.file_indexer.pipeline_checks import FileIndexStageIndexer
from champollion_pipeline.input_kind_choice import (
    InputChoiceError,
    check_input_kind_value,
    choose_input_kind,
    choose_subjects_folder,
)
from champollion_pipeline.path_inference import (
    AmbiguousGraphFolders,
    DatasetRootNotInferable,
    infer_paths,
    refresh_version_paths,
)
from champollion_pipeline.pipeline_stages import _HF_REPO_ID_REQUIRED as _HF_REPO_ID_REQUIRED
from champollion_pipeline.pipeline_stages import (
    GenerateChampollionConfigStage,
    GenerateEmbeddingsStage,
    GenerateMorphologistGraphsStage,
    GenerateSnapshotsStage,
    PutTogetherEmbeddingsStage,
    RunCorticalTilesStage,
    _compute_models_source,
    _compute_morphologist_subjects_dir,
)
from champollion_pipeline.pipeline_stages import _compute_combined_output_dir as _compute_combined_output_dir
from champollion_pipeline.pipeline_stages import _compute_configs_dir as _compute_configs_dir
from champollion_pipeline.pipeline_stages import _compute_region_embeddings_source as _compute_region_embeddings_source
from champollion_pipeline.pipeline_stages import _compute_snapshots_embeddings_dir as _compute_snapshots_embeddings_dir
from champollion_pipeline.pipeline_stages import _compute_snapshots_output_dir as _compute_snapshots_output_dir
from champollion_pipeline.pipeline_stages import _compute_stage4_crops_dir as _compute_stage4_crops_dir
from champollion_pipeline.pipeline_stages import _scoped_env as _scoped_env
from champollion_pipeline.process_setup import init_pipeline_process
from champollion_pipeline.replay_config import write_replay_config
from champollion_pipeline.run_masks import RegionMasksError, RunMasks, parse_given_region_masks, resolve_run_masks
from champollion_pipeline.streaming import StreamingSettings
from champollion_pipeline.utils.cortical_tiles_config import compute_versioned_crops_dir as compute_versioned_crops_dir
from champollion_pipeline.utils.input_kind import InputKind
from champollion_pipeline.utils.input_kind import stage1_subjects_dir as stage1_subjects_dir
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER as DERIVATIVES_FOLDER
from champollion_pipeline.utils.lib import find_dataset_folder as find_dataset_folder
from champollion_pipeline.utils.region_masks import REGION_MASKS_OPTION, format_region_masks
from champollion_pipeline.utils.skeleton_qc import SKELETON_SIDES as SKELETON_SIDES
from champollion_pipeline.utils.skeleton_qc import compute_skeleton_pattern as compute_skeleton_pattern

logger = logging.getLogger("champollion_pipeline.main")

# Stage 1, skipped when the input already holds Morphologist outputs
_MORPHOLOGIST_STAGE = "generate_morphologist_graphs"
# Exit status of a command-line usage error, as argparse uses.
_USAGE_ERROR_STATUS = 2

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

    def __init__(self, config: PipelineConfig, indexer: StageIndexer | None = None):
        self.config = config
        self._indexer = indexer
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

    def _stage_indexer(self) -> StageIndexer:
        """The injected StageIndexer, else the default adapter built from the current config."""
        return self._indexer if self._indexer is not None else _default_stage_indexer(self.config)

    def _report_pre_stage(self, stage_name: str, stage: PipelineStage) -> None:
        """Ask the stage indexer to index the stage's input folders; no input folder: nothing."""
        if not stage.subject_input_dir():
            return
        self._stage_indexer().index_pre(stage_name, stage.index_input_dirs(), stage.required_file_patterns())

    def _report_post_stage(self, stage_name: str, stage: PipelineStage) -> None:
        """Index the folder the stage wrote and log its output report; never fails the stage (NTUI)."""
        output_dir = stage.output_dir()
        if not output_dir or not os.path.isdir(output_dir):
            return
        try:
            self._stage_indexer().index_post(stage_name, output_dir)
        except Exception as e:
            self.logger.warning(f"Output report after stage '{stage_name}' failed, stage kept as completed: {e}")

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
            self._report_post_stage(stage_name, stage)
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


def _default_stage_indexer(config: PipelineConfig) -> StageIndexer:
    """The one place the sequential stages' indexer adapter is chosen."""
    return FileIndexStageIndexer(config.outputs_path, config.dataset.bids, config.dataset.path_to_graph)


# ====================== CLI Interface ======================


def create_default_config() -> PipelineConfig:
    """Create default configuration."""
    root_path = Path(PipelineConfig().root_path)
    config = PipelineConfig(
        root_path=str(root_path),
        data_path=str(root_path / "data"),
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
    fold_path_arguments(args)
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
    check_input_kind_value(config.dataset)
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


def _build_streaming_settings(
    config: PipelineConfig, args: argparse.Namespace, run_log_path: str | None
) -> StreamingSettings:
    """StreamingSettings of this run from the overridden configuration."""
    return StreamingSettings(
        subjects_dir=_compute_morphologist_subjects_dir(config.dataset),
        output_dir=config.outputs_path,
        log_dir=StreamingSettings.compute_log_dir(config, run_log_path),
        n_workers=config.n_workers,
        worker_timeout=config.worker_timeout,
        dry_run=args.dry_run,
    )


def _resolve_streaming_input(config: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    """Return config with the paths the input path fixes filled: the streaming subjects folder is the resolved one."""
    if config.dataset.morphologist_graphs or not config.dataset.input_path:
        return config
    configless_input = not args.config
    if configless_input and args.datasets_root is None:
        config.dataset.datasets_root = DatasetConfig().datasets_root
    kind = _apply_input_kind(config)
    if configless_input:
        config = _infer_configless_paths(config, kind, args)
    return config


def _run_streaming_mode(config: PipelineConfig, args: argparse.Namespace, *, run_log_path: str | None) -> int:
    """Run the scan-centric streaming mode after its two guards."""
    if not config.dataset.embeddings_only:
        logger.error(
            "--mode streaming requires dataset.embeddings_only: true in the --config YAML "
            "or the --embeddings-only argument (training aggregates subjects and must remain batch)."
        )
        return 1
    config = _resolve_streaming_input(config, args)
    write_replay_config(config)
    try:
        from champollion_pipeline.streaming.runner import run_parallel_pipeline
    except ImportError as e:
        logger.error(f"could not import champollion_pipeline.streaming.runner: {e}")
        return 1
    return run_parallel_pipeline(
        _build_streaming_settings(config, args, run_log_path), config.dataset, _compute_models_source(config)
    )


def _apply_input_kind(config: PipelineConfig) -> InputKind | None:
    """Fill, in place, the fields the chosen kind of config.dataset.input_path fixes; return that kind.

    choose_input_kind decides the kind (None: input_path and input_kind empty, nothing changed); its
    InputChoiceError propagates.
    """
    kind = choose_input_kind(config.dataset)
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

    choose_subjects_folder asks or stops on several subjects folders; its InputChoiceError propagates.
    """
    dataset = config.dataset
    if not dataset.morphologist_graphs:
        folder = choose_subjects_folder(dataset)
        if folder is not None:
            dataset.morphologist_graphs = os.path.abspath(folder)
    _skip_morphologist_stage(config)


def _skip_morphologist_stage(config: PipelineConfig) -> None:
    """Disable stage 1 and log it once at INFO; a stage 1 already disabled stays silent."""
    if config.stages.get(_MORPHOLOGIST_STAGE):
        config.stages[_MORPHOLOGIST_STAGE] = False
        logger.info(f"Stage {_MORPHOLOGIST_STAGE} skipped: {config.dataset.input_path} holds Morphologist outputs")


def _prepare_configless_input_run(config: PipelineConfig, args: argparse.Namespace) -> None:
    """No --config, an input path: root only from OUTPUT/--output-path/--datasets-root or inference; all stages."""
    if args.datasets_root is None:
        config.dataset.datasets_root = DatasetConfig().datasets_root
    if args.outputs_path is None:
        config.outputs_path = PipelineConfig().outputs_path
    if args.stages is None:
        config.stages = dict.fromkeys(config.stages, True)


def _infer_configless_paths(config: PipelineConfig, kind: InputKind, args: argparse.Namespace) -> PipelineConfig:
    """Config-less run: unset paths inferred, dataset named after its root; no stop here, one WARNING at most."""
    try:
        config = infer_paths(config, config.dataset.input_path, kind)
    except (DatasetRootNotInferable, AmbiguousGraphFolders) as error:
        logger.warning(f"Paths not inferred: {error}")
    root = config.dataset.datasets_root
    if args.dataset_name is None and root:
        config.dataset.name = Path(os.path.abspath(root)).name
    return config


def _check_given_masks(config: PipelineConfig, args: argparse.Namespace) -> None:
    """Refuse --region-masks beside --masks-version, let --masks-version clear a Config file map, check the map.

    Runs in both modes, before any stage and any resolver call. Raises ArgumentConflictError naming both options, or
    RegionMasksError naming the faulty value (one ERROR record, exit 2, in run()).
    """
    if args.region_masks is not None and args.masks_version is not None:
        raise ArgumentConflictError(
            f"{REGION_MASKS_OPTION} cannot be given together with --masks-version: the map names the Mask version "
            "of each region model name; drop one of them"
        )
    if args.masks_version is not None:
        config.dataset.region_masks = []
    parse_given_region_masks(config.dataset.region_masks)


def _resolve_run_masks(config: PipelineConfig, args: argparse.Namespace) -> RunMasks:
    """The run's RunMasks: a Mask version is given by --masks-version or a Config file masks_version without a map.

    A Config file region_masks list that is empty (the replay of a --masks run) is no map. The stage 4 models source
    is computed on every sequential run, a given map run included, so its "Stage 4 models source" line is logged here
    and again by stage 4 (accepted). resolve_run_masks' RegionMasksError (a mixed given map on Hugging Face, a partial
    local model set) reaches run(): one ERROR record, exit 2, before the replay file is written.
    """
    dataset = config.dataset
    keys = ConfigLoader.read_dataset_keys(args.config) if args.config else frozenset()
    is_version_given = args.masks_version is not None or ("masks_version" in keys and not dataset.region_masks)
    models_source = _compute_models_source(config) or None
    return resolve_run_masks(dataset, is_version_given=is_version_given, models_source=models_source)


def _apply_run_masks(config: PipelineConfig, run_masks: RunMasks, *, inferred: bool) -> None:
    """Set the run's map and label on config.dataset; re-key the inferred version paths when the label changed."""
    dataset = config.dataset
    dataset.region_masks = format_region_masks(run_masks.region_masks) if run_masks.region_masks else []
    previous = dataset.masks_version
    dataset.masks_version = run_masks.label
    if inferred and run_masks.label != previous:
        refresh_version_paths(config, previous)


class ChampollionPipeline(ScriptBuilder):
    """main.py as a ScriptBuilder script: the shared run log, profiling and process setup."""

    def __init__(self) -> None:
        self._run_log_path: str | None = None
        super().__init__(SCRIPT_NAME, SCRIPT_DESCRIPTION)
        self._add_pipeline_options()

    def _create_parser(self) -> argparse.ArgumentParser:  # noqa: V105 - ScriptBuilder hook
        """Keep main.py's raw help layout and its examples epilog."""
        return argparse.ArgumentParser(
            prog=self.script_name,
            description=self.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=_HELP_EPILOG,
            allow_abbrev=False,
        )

    def _add_log_option(self) -> None:  # noqa: V105 - ScriptBuilder hook
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
        add_path_arguments(self.parser)
        add_env_argument(self.parser)

    def _resolve_log_path(self) -> str | None:  # noqa: V105 - ScriptBuilder hook
        """main.py's own run log: only with a bare --log, stored for the streaming log_dir. args.log is left as is."""
        self._run_log_path = self.default_log_path() if self.args.log == "" else None
        return self._run_log_path

    def build(self) -> "ChampollionPipeline":
        """Parse the arguments, then check for updates once, never with --generate-config."""
        self.parse_args()
        if not self.args.generate_config:
            check_for_updates()
        return self

    def print_args(self) -> "ChampollionPipeline":
        """Log the parsed arguments at debug level only: the console shows no raw argument dict."""
        shown = {
            key: value
            for key, value in vars(self.args).items()
            if not (key in ("log", "profile", "profile_mode") and value is None)
        }
        logger.debug("%s arguments: %s", self.script_name, shown)
        return self

    def run(self) -> int:
        """Run the pipeline; InputChoiceError (status 1), ArgumentConflictError, EnvFileError, RegionMasksError (2):
        one ERROR record."""
        try:
            return self._run_pipeline(self.args)
        except InputChoiceError as error:
            logger.error(str(error))
            return 1
        except (ArgumentConflictError, EnvFileError, RegionMasksError) as error:
            logger.error(str(error))
            return _USAGE_ERROR_STATUS

    def _run_pipeline(self, args: argparse.Namespace) -> int:
        """The --generate-config template, or the configured run in streaming or batch mode.

        The --env file and the process environment are read first, so their refusals come before any output; their
        values go over the --config YAML (or the default config) of a run and under its command-line arguments, never
        into the template.
        """
        env_values = read_env_values(args.env)
        if args.generate_config:
            return _save_config_template(args)
        config = _load_config(args)
        apply_env_values(config, env_values)
        _apply_cli_overrides(config, args, from_default=not args.config)
        check_input_kind_value(config.dataset)
        _check_given_masks(config, args)
        _start_config_profiling(config, args)
        if config.mode == "streaming":
            return _run_streaming_mode(config, args, run_log_path=self._run_log_path)
        configless_input = not args.config and bool(config.dataset.input_path)
        if configless_input:
            _prepare_configless_input_run(config, args)
        kind = _apply_input_kind(config)
        if configless_input:
            config = _infer_configless_paths(config, kind, args)
        run_masks = _resolve_run_masks(config, args)
        # The mismatch and unresolved checks of TASK-MODELVER-BDRABCZUK-3B2E4833B088 read run_masks here.
        _apply_run_masks(config, run_masks, inferred=configless_input)
        write_replay_config(config)
        return PipelineOrchestrator(config).run()


def main() -> int:
    """Main entry point for the pipeline."""
    init_pipeline_process()
    return ChampollionPipeline().main()


if __name__ == "__main__":
    sys.exit(main())
