#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import yaml
from champollion_utils.update_check import check_for_updates

from champollion_pipeline.derivatives_layout import (
    CHAMPOLLION_DERIVATIVES_FOLDER,
    HEMISPHERES,
    compute_region_model_name,
)
from champollion_pipeline.generate_champollion_config import GenerateChampollionConfig
from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from champollion_pipeline.generate_morphologist_graphs import GenerateMorphologistGraphs
from champollion_pipeline.generate_snapshots import GenerateSnapshots
from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings
from champollion_pipeline.run_cortical_tiles import RunCorticalTiles

# Add src to path for imports – still needed for the optional file_indexer.pipeline_checks
# import below and the lazy parallel_runner import in streaming mode.
pipeline_src = Path(__file__).parent / "src"
sys.path.insert(0, str(pipeline_src))

try:
    from file_indexer.pipeline_checks import SubjectEligibilityChecker, build_output_report
except ImportError:
    SubjectEligibilityChecker = None
    build_output_report = None


# ====================== Configuration Management ======================


# Dataset config keys that no longer exist; old YAML files still carrying them load with a warning.
_REMOVED_DATASET_KEYS = {
    "use_best_model": "it had no effect; best_model_weights.pt is now used by default, "
    "set dataset.use_last_checkpoint: true to use the native Lightning checkpoint instead",
}


@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""

    name: str = "example_dataset"
    dataset_localization: str = "local"
    datasets_root: str = ""
    datasets: List[str] = field(default_factory=lambda: ["example"])
    labels: List[str] = field(default_factory=lambda: ["Sex"])

    # Paths
    input_path: str = ""
    morphologist_graphs: str = ""
    cortical_tiles_output: str = ""
    crops_path: str = ""
    embeddings_path: str = ""

    # Region filter (empty = all 56 regions)
    regions: List[str] = field(default_factory=list)

    # Processing parameters
    njobs: int = 22
    path_to_graph: str = "t1mri/default_acquisition/default_analysis/folds/3.3/base"
    path_sk_with_hull: str = "t1mri/default_acquisition/default_analysis/segmentation/mesh"
    sk_qc_path: str = ""

    # Embeddings parameters
    classifier_name: str = "svm"
    overwrite: bool = False
    embeddings_only: bool = False
    use_last_checkpoint: bool = False
    subsets: List[str] = field(default_factory=lambda: ["full"])
    epochs: List[Optional[int]] = field(default_factory=lambda: [None])
    split: str = "random"
    cv: int = 5
    splits_basedir: Optional[str] = None
    idx_region_evaluation: Optional[int] = None
    short_name: str = "eval"

    # HuggingFace parameters
    hf_enabled: bool = False
    hf_repo_id: Optional[str] = None
    hf_token: Optional[str] = None

    # Unused: no stage reads it. Kept so existing YAML files that set it still load.
    config_path: Optional[str] = None

    # CPU mode (disable CUDA)
    cpu: bool = False

    # BIDS database layout (subjects/sub-*/ses-*/...)
    bids: bool = False

    # Snapshots parameters
    snapshots_path: str = ""
    reference_data_path: str = ""


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""

    # Root paths
    root_path: str = str(Path(__file__).parent)
    data_path: str = ""
    models_path: str = ""
    outputs_path: str = ""
    champollion_v1_path: str = ""

    # Pipeline stages to execute
    stages: Dict[str, bool] = field(
        default_factory=lambda: {
            "generate_morphologist_graphs": False,
            "run_cortical_tiles": False,
            "generate_champollion_config": False,
            "generate_embeddings": True,
            "put_together_embeddings": False,
            "generate_snapshots": False,
        }
    )

    # Stage dependencies
    dependencies: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "generate_morphologist_graphs": [],
            "run_cortical_tiles": ["generate_morphologist_graphs"],
            "generate_champollion_config": ["run_cortical_tiles"],
            "generate_embeddings": ["generate_champollion_config"],
            "put_together_embeddings": ["generate_embeddings"],
            "generate_snapshots": ["put_together_embeddings"],
        }
    )

    # Execution settings
    mode: str = "sequential"
    n_workers: int = 0  # 0 means use os.cpu_count()
    worker_timeout: int = 7200  # seconds per worker
    stop_on_error: bool = True
    verbose: bool = False

    # Logging
    log_level: str = "INFO"
    log_dir: str = ""
    log_to_file: bool = True
    log_to_console: bool = True

    # Dataset configuration
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


class ConfigLoader:
    """Load and manage pipeline configuration."""

    @staticmethod
    def load_from_yaml(config_path: str) -> PipelineConfig:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return ConfigLoader._dict_to_config(config_dict)

    @staticmethod
    def _dict_to_config(config_dict: Dict) -> PipelineConfig:
        """Convert dictionary to PipelineConfig object."""
        # Extract dataset config
        dataset_dict = config_dict.pop("dataset", {})
        for key, hint in _REMOVED_DATASET_KEYS.items():
            if key in dataset_dict:
                dataset_dict.pop(key)
                logging.getLogger("champollion_pipeline").warning(
                    "Ignoring removed config key dataset.%s: %s", key, hint
                )
        dataset_config = DatasetConfig(**dataset_dict)

        # Create pipeline config
        pipeline_config = PipelineConfig(**config_dict, dataset=dataset_config)

        return pipeline_config

    @staticmethod
    def save_to_yaml(config: PipelineConfig, output_path: str):
        """Save configuration to YAML file."""
        # Convert to dictionary
        config_dict = {
            "root_path": config.root_path,
            "data_path": config.data_path,
            "models_path": config.models_path,
            "outputs_path": config.outputs_path,
            "champollion_v1_path": config.champollion_v1_path,
            "stages": config.stages,
            "dependencies": config.dependencies,
            "mode": config.mode,
            "n_workers": config.n_workers,
            "worker_timeout": config.worker_timeout,
            "stop_on_error": config.stop_on_error,
            "verbose": config.verbose,
            "log_level": config.log_level,
            "log_dir": config.log_dir,
            "log_to_file": config.log_to_file,
            "log_to_console": config.log_to_console,
            "dataset": {
                "name": config.dataset.name,
                "dataset_localization": config.dataset.dataset_localization,
                "datasets_root": config.dataset.datasets_root,
                "datasets": config.dataset.datasets,
                "labels": config.dataset.labels,
                "input_path": config.dataset.input_path,
                "morphologist_graphs": config.dataset.morphologist_graphs,
                "cortical_tiles_output": config.dataset.cortical_tiles_output,
                "crops_path": config.dataset.crops_path,
                "embeddings_path": config.dataset.embeddings_path,
                "njobs": config.dataset.njobs,
                "path_to_graph": config.dataset.path_to_graph,
                "path_sk_with_hull": config.dataset.path_sk_with_hull,
                "sk_qc_path": config.dataset.sk_qc_path,
                "regions": config.dataset.regions,
                "classifier_name": config.dataset.classifier_name,
                "overwrite": config.dataset.overwrite,
                "embeddings_only": config.dataset.embeddings_only,
                "use_last_checkpoint": config.dataset.use_last_checkpoint,
                "subsets": config.dataset.subsets,
                "epochs": config.dataset.epochs,
                "split": config.dataset.split,
                "cv": config.dataset.cv,
                "splits_basedir": config.dataset.splits_basedir,
                "idx_region_evaluation": config.dataset.idx_region_evaluation,
                "short_name": config.dataset.short_name,
                "hf_enabled": config.dataset.hf_enabled,
                "hf_repo_id": config.dataset.hf_repo_id,
                # hf_token is a secret: never written (REQ-HFTOKEN-01).
                # Supply it via the HF_TOKEN environment variable.
                "config_path": config.dataset.config_path,
                "cpu": config.dataset.cpu,
                "bids": config.dataset.bids,
                "snapshots_path": config.dataset.snapshots_path,
                "reference_data_path": config.dataset.reference_data_path,
            },
        }

        with open(output_path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
        if config.dataset.hf_token:
            print(
                f"Note: dataset.hf_token not written to {output_path}; set the HF_TOKEN environment variable instead."
            )


# ====================== Pipeline Stage Strategy Pattern ======================


@dataclass
class StageResult:
    """Result of a pipeline stage execution."""

    stage_name: str
    success: bool
    message: str
    return_code: int = 0


class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""

    def __init__(self, name: str, config: PipelineConfig, logger: logging.Logger):
        self.name = name
        self.config = config
        self.logger = logger

    @abstractmethod
    def validate(self) -> bool:
        """Validate stage prerequisites."""
        pass

    @abstractmethod
    def execute(self) -> StageResult:
        """Execute the stage."""
        pass

    def log_start(self):
        """Log stage start."""
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"Starting stage: {self.name}")
        self.logger.info(f"{'=' * 60}")

    def log_end(self, result: StageResult):
        """Log stage end."""
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        self.logger.info(f"{status}: {self.name} - {result.message}")
        self.logger.info(f"{'=' * 60}\n")

    def subject_input_dir(self) -> Optional[str]:
        """Return the subjects directory to index before this stage. None = skip."""
        return None

    def required_file_patterns(self) -> List[str]:
        """Glob patterns (relative to subject dir) required per subject."""
        return []

    def output_dir(self) -> Optional[str]:
        """Return the output directory to index after this stage. None = skip."""
        return None


class GenerateMorphologistGraphsStage(PipelineStage):
    """Stage for generating Morphologist graphs."""

    def validate(self) -> bool:
        """Validate that input data exists."""
        input_path = Path(self.config.dataset.input_path)
        if not input_path.exists():
            self.logger.error(f"Input path does not exist: {input_path}")
            return False
        return True

    def execute(self) -> StageResult:
        """Execute Morphologist graphs generation."""
        self.log_start()
        try:
            self.logger.info("Generating Morphologist graphs...")
            args = [
                str(self.config.dataset.input_path),
                str(self.config.dataset.morphologist_graphs),
            ]
            if self.config.dataset.bids:
                args.append("--bids")
            if getattr(self.config, "parallel", False):
                args.append("--parallel")
            script = GenerateMorphologistGraphs()
            script.parse_args(args)
            return_code = script.run()

            result = StageResult(
                stage_name=self.name,
                success=(return_code == 0),
                message="Morphologist graphs generated successfully" if return_code == 0 else "Morphologist failed",
                return_code=return_code,
            )
        except Exception as e:
            self.logger.exception(f"Exception in {self.name}")
            result = StageResult(stage_name=self.name, success=False, message=f"Failed: {str(e)}", return_code=1)

        self.log_end(result)
        return result


class RunCorticalTilesStage(PipelineStage):
    """Stage for running cortical_tiles."""

    def validate(self) -> bool:
        """Validate that Morphologist graphs exist."""
        graphs_path = Path(self.config.dataset.morphologist_graphs)
        if not graphs_path.exists():
            self.logger.error(f"Morphologist graphs path does not exist: {graphs_path}")
            return False
        return True

    def execute(self) -> StageResult:
        """Execute cortical_tiles."""
        self.log_start()
        try:
            self.logger.info("Running cortical_tiles to generate sulcal regions...")

            # Build arguments from config
            args = [
                str(self.config.dataset.morphologist_graphs),
                str(self.config.dataset.cortical_tiles_output),
                f"--path_to_graph={self.config.dataset.path_to_graph}",
                f"--path_sk_with_hull={self.config.dataset.path_sk_with_hull}",
                f"--njobs={self.config.dataset.njobs}",
            ]

            if self.config.dataset.sk_qc_path:
                args.append(f"--sk_qc_path={self.config.dataset.sk_qc_path}")

            if self.config.dataset.bids:
                args.append("--bids")

            if self.config.dataset.regions:
                args.extend(["--regions"] + self.config.dataset.regions)

            # Parse and run
            script = RunCorticalTiles()
            script.parse_args(args)
            return_code = script.run()

            result = StageResult(
                stage_name=self.name,
                success=(return_code == 0),
                message="Cortical tiles completed" if return_code == 0 else "Failed",
                return_code=return_code,
            )
        except Exception as e:
            self.logger.exception(f"Exception in {self.name}")
            result = StageResult(stage_name=self.name, success=False, message=f"Failed: {str(e)}", return_code=1)

        self.log_end(result)
        return result

    def subject_input_dir(self) -> Optional[str]:
        d = self.config.dataset.morphologist_graphs
        return d if d and os.path.exists(d) else None

    def required_file_patterns(self) -> List[str]:
        import re as _re

        side = "R"
        g = self.config.dataset.path_to_graph
        s = self.config.dataset.path_sk_with_hull
        if self.config.dataset.bids:
            # scan_id.path_prefix already includes the session directory;
            # strip the leading ses-XXX/ segment so patterns are session-relative.
            g = _re.sub(r"^ses-[^/]*/+", "", g)
            s = _re.sub(r"^ses-[^/]*/+", "", s)
        return [
            f"{g}/{side}*.arg",
            f"{s}/{side}*.nii.gz",
        ]

    def output_dir(self) -> Optional[str]:
        d = self.config.dataset.cortical_tiles_output
        return d if d else None


class GenerateChampollionConfigStage(PipelineStage):
    """Stage for generating Champollion configuration."""

    def validate(self) -> bool:
        """Validate that crops exist."""
        crops_path = Path(self.config.dataset.crops_path)
        if not crops_path.exists():
            self.logger.warning(f"Crops path does not exist yet: {crops_path}")
        return True

    def execute(self) -> StageResult:
        """Execute Champollion config generation."""
        self.log_start()
        try:
            self.logger.info("Generating Champollion configuration...")

            args = [str(self.config.dataset.crops_path), f"--dataset={self.config.dataset.name}"]

            script = GenerateChampollionConfig()
            script.parse_args(args)
            return_code = script.run()

            result = StageResult(
                stage_name=self.name,
                success=(return_code == 0),
                message="Champollion config generated" if return_code == 0 else "Config generation failed",
                return_code=return_code,
            )
        except Exception as e:
            self.logger.exception(f"Exception in {self.name}")
            result = StageResult(stage_name=self.name, success=False, message=f"Failed: {str(e)}", return_code=1)

        self.log_end(result)
        return result


_HF_REPO_ID_REQUIRED = "hf_enabled requires dataset.hf_repo_id"


@contextmanager
def _scoped_env(name: str, value: Optional[str]) -> Iterator[None]:
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

    def execute(self) -> StageResult:
        """Execute embeddings generation."""
        dataset = self.config.dataset
        if dataset.hf_enabled and not dataset.hf_repo_id:
            return StageResult(
                stage_name=self.name,
                success=False,
                message=_HF_REPO_ID_REQUIRED,
                return_code=1,
            )
        self.log_start()
        try:
            self.logger.info("Generating embeddings and training classifiers...")

            models_path = dataset.hf_repo_id if dataset.hf_enabled else str(self.config.models_path)
            args = [models_path, dataset.datasets_root]

            if dataset.embeddings_path:
                args.append(f"--output={dataset.embeddings_path}")
            if dataset.overwrite:
                args.append("--overwrite")
            if dataset.cpu:
                args.append("--cpu")
            if dataset.use_last_checkpoint:
                args.append("--use_last_checkpoint")
            if dataset.regions:
                args.extend(["--regions", *_compute_embeddings_region_names(dataset.regions)])

            script = GenerateEmbeddings()
            script.parse_args(args)
            hf_token = dataset.hf_token if dataset.hf_enabled else None
            with _scoped_env("HF_TOKEN", hf_token):
                return_code = script.run()

            result = StageResult(
                stage_name=self.name,
                success=(return_code == 0),
                message="Embeddings generated" if return_code == 0 else "Embedding generation failed",
                return_code=return_code,
            )
        except Exception as e:
            self.logger.exception(f"Exception in {self.name}")
            result = StageResult(stage_name=self.name, success=False, message=f"Failed: {str(e)}", return_code=1)

        self.log_end(result)
        return result


def _compute_combined_embeddings_dir(datasets_root: str) -> Path:
    """Return <datasets_root>/derivatives/champollion_V1/embeddings (pure, no I/O, O(1))."""
    return Path(datasets_root) / "derivatives" / CHAMPOLLION_DERIVATIVES_FOLDER / "embeddings"


def _compute_embeddings_region_names(regions: List[str]) -> List[str]:
    """Return both hemispheres' model names for each distinct region, in input order. O(n)."""
    distinct_regions = dict.fromkeys(regions)
    return [compute_region_model_name(region, hemisphere) for region in distinct_regions for hemisphere in HEMISPHERES]


class PutTogetherEmbeddingsStage(PipelineStage):
    """Stage for combining embeddings."""

    def validate(self) -> bool:
        """Validate that embeddings exist."""
        embeddings_path = Path(self.config.dataset.embeddings_path)
        if not embeddings_path.exists():
            self.logger.error(f"Embeddings path does not exist: {embeddings_path}")
            return False
        return True

    def execute(self) -> StageResult:
        """Execute embeddings combination."""
        self.log_start()
        try:
            self.logger.info("Putting together embeddings...")
            dataset = self.config.dataset
            output_path = (
                _compute_combined_embeddings_dir(dataset.datasets_root)
                if dataset.datasets_root
                else (dataset.cortical_tiles_output or self.config.outputs_path)
            )
            args = [str(dataset.embeddings_path), f"--output_path={output_path}"]
            script = PutTogetherEmbeddings()
            script.parse_args(args)
            return_code = script.run()

            result = StageResult(
                stage_name=self.name,
                success=(return_code == 0),
                message="Embeddings combined successfully" if return_code == 0 else "Combine failed",
                return_code=return_code,
            )
        except Exception as e:
            self.logger.exception(f"Exception in {self.name}")
            result = StageResult(stage_name=self.name, success=False, message=f"Failed: {str(e)}", return_code=1)

        self.log_end(result)
        return result


class GenerateSnapshotsStage(PipelineStage):
    """Stage for generating visualization snapshots."""

    def validate(self) -> bool:
        """Validate that embeddings exist and output path is set."""
        if not self.config.dataset.snapshots_path:
            self.logger.error("snapshots_path must be set in config")
            return False
        embeddings_path = Path(self.config.dataset.embeddings_path)
        if self.config.dataset.embeddings_path and not embeddings_path.exists():
            self.logger.error(f"Embeddings path does not exist: {embeddings_path}")
            return False
        return True

    def execute(self) -> StageResult:
        """Execute snapshot generation."""
        self.log_start()
        try:
            self.logger.info("Generating visualization snapshots...")

            args = [f"--output_dir={self.config.dataset.snapshots_path}"]

            dataset = self.config.dataset
            if dataset.datasets_root:
                args.append(f"--embeddings_dir={_compute_combined_embeddings_dir(dataset.datasets_root)}")
            elif dataset.embeddings_path:
                args.append(f"--embeddings_dir={dataset.embeddings_path}")
            if self.config.dataset.morphologist_graphs:
                args.append(f"--morphologist_dir={self.config.dataset.morphologist_graphs}")
            if self.config.dataset.crops_path:
                args.append(f"--cortical_tiles_dir={self.config.dataset.crops_path}")
            if self.config.dataset.reference_data_path:
                args.append(f"--reference_data_dir={self.config.dataset.reference_data_path}")

            script = GenerateSnapshots()
            script.parse_args(args)
            return_code = script.run()

            result = StageResult(
                stage_name=self.name,
                success=(return_code == 0),
                message=("Snapshots generated" if return_code == 0 else "Snapshot generation failed"),
                return_code=return_code,
            )
        except Exception as e:
            self.logger.exception(f"Exception in {self.name}")
            result = StageResult(
                stage_name=self.name,
                success=False,
                message=f"Failed: {str(e)}",
                return_code=1,
            )

        self.log_end(result)
        return result


# ====================== Pipeline Orchestrator ======================


class PipelineOrchestrator:
    """Orchestrates the execution of pipeline stages."""

    # Stage registry mapping names to classes
    STAGE_REGISTRY = {
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
        self.stages: Dict[str, PipelineStage] = {}
        self._register_stages()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("champollion_pipeline")
        logger.setLevel(getattr(logging, self.config.log_level))
        logger.handlers = []  # Clear existing handlers

        # Console handler
        if self.config.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

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

    def _get_enabled_stages(self) -> List[str]:
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

    def _check_dependencies(self, stage_name: str, completed: List[str]) -> bool:
        """Check if stage dependencies are satisfied."""
        depends_on = self.config.dependencies.get(stage_name, [])

        for dep in depends_on:
            if self.config.stages.get(dep, False) and dep not in completed:
                self.logger.error(f"Stage '{stage_name}' depends on '{dep}' which hasn't completed")
                return False

        return True

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

            # Check dependencies
            if not self._check_dependencies(stage_name, completed_stages):
                failed_stages.append(stage_name)
                if self.config.stop_on_error:
                    break
                continue

            # Get stage
            stage = self.stages[stage_name]

            # Validate
            if not stage.validate():
                self.logger.error(f"Stage '{stage_name}' validation failed")
                failed_stages.append(stage_name)
                if self.config.stop_on_error:
                    break
                continue

            # Pre-stage: index input + eligibility report
            if SubjectEligibilityChecker is not None:
                input_dir = stage.subject_input_dir()
                if input_dir:
                    patterns = stage.required_file_patterns()
                    if patterns:
                        index_path = None
                        if self.config.outputs_path:
                            os.makedirs(self.config.outputs_path, exist_ok=True)
                            index_path = os.path.join(
                                self.config.outputs_path,
                                f"index_pre_{stage_name}.json",
                            )
                        checker = SubjectEligibilityChecker(
                            input_dir,
                            patterns,
                            stage_name=stage_name,
                            bids=self.config.dataset.bids,
                            path_to_graph=self.config.dataset.path_to_graph,
                        )
                        pre_report = checker.check(save_index_to=index_path)
                        pre_report.print()

            # Execute
            result = stage.execute()

            # Post-stage: index output + summary report
            if result.success and build_output_report is not None:
                out_dir = stage.output_dir()
                if out_dir and os.path.exists(out_dir):
                    index_path = None
                    if self.config.outputs_path:
                        index_path = os.path.join(
                            self.config.outputs_path,
                            f"index_post_{stage_name}.json",
                        )
                    out_report = build_output_report(out_dir, stage_name, save_index_to=index_path)
                    out_report.print()

            if result.success:
                completed_stages.append(stage_name)
            else:
                failed_stages.append(stage_name)
                if self.config.stop_on_error:
                    self.logger.error("Stopping pipeline due to stage failure")
                    break

        # Summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Pipeline Execution Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"Completed stages: {completed_stages}")
        if failed_stages:
            self.logger.info(f"Failed stages: {failed_stages}")

        if failed_stages:
            self.logger.error("❌ Pipeline completed with errors")
            return 1
        else:
            self.logger.info("✅ Pipeline completed successfully")
            return 0


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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Champollion Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
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
        """,
    )

    parser.add_argument("--config", type=str, help="Path to configuration YAML file")

    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(PipelineOrchestrator.STAGE_REGISTRY.keys()),
        help="Specific stages to run (overrides config)",
    )

    parser.add_argument("--enable-all-stages", action="store_true", help="Enable all pipeline stages")

    parser.add_argument("--dataset-name", type=str, help="Dataset name (overrides config)")

    parser.add_argument("--models-path", type=str, help="Path to models directory (overrides config)")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument(
        "--generate-config", type=str, metavar="OUTPUT_PATH", help="Generate a template configuration file and exit"
    )

    parser.add_argument(
        "--bids",
        action="store_true",
        help=(
            "Input database follows BIDS layout "
            "(subjects/sub-*/ses-*/...). "
            "Activates BIDS-aware scan enumeration in eligibility checks "
            "and passes --bids to cortical_tiles."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=["sequential", "streaming"],
        default=None,
        help=(
            "Execution mode: 'sequential' (default, stage-centric batch) or "
            "'streaming' (scan-centric parallel, one worker per scan). "
            "Streaming mode requires --embeddings-only."
        ),
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel scan workers (streaming mode only). Defaults to os.cpu_count().",
    )
    parser.add_argument(
        "--worker-timeout",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Per-worker timeout waiting for prerequisite files (default: 7200).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate scans and report without processing (streaming mode only).",
    )

    return parser.parse_args()


def _apply_cli_overrides(config: PipelineConfig, args: argparse.Namespace, *, from_default: bool) -> None:
    """Apply command-line argument overrides to a PipelineConfig in place.

    ``from_default`` must be ``True`` when *config* was produced by
    :func:`create_default_config` (no ``--config`` file was supplied) and
    ``False`` when it was loaded from a YAML file.  The flag gates the
    REQ-DEFROOT dataset-root re-derivation that must only fire for the
    default config (REQ-DEFROOT-03/04).
    """
    if args.stages:
        # Disable all stages, then enable specified ones
        for stage in config.stages:
            config.stages[stage] = False
        for stage in args.stages:
            config.stages[stage] = True

    if args.enable_all_stages:
        for stage in config.stages:
            config.stages[stage] = True

    if args.dataset_name:
        config.dataset.name = args.dataset_name
        if from_default:
            # Default config only: re-derive the root for the new name (REQ-DEFROOT-03).
            # A --config YAML datasets_root is never rewritten (REQ-DEFROOT-04).
            config.dataset.datasets_root = str(Path(config.data_path) / config.dataset.name)

    if args.models_path:
        config.models_path = args.models_path

    if args.verbose:
        config.verbose = True
        config.log_level = "DEBUG"

    if args.bids:
        config.dataset.bids = True

    if args.mode:
        config.mode = args.mode
    if args.n_workers is not None:
        config.n_workers = args.n_workers
    if args.worker_timeout is not None:
        config.worker_timeout = args.worker_timeout


def main() -> int:
    """Main entry point for the pipeline."""
    check_for_updates()
    args = parse_arguments()

    # Generate config template if requested
    if args.generate_config:
        config = create_default_config()
        _apply_cli_overrides(config, args, from_default=True)
        ConfigLoader.save_to_yaml(config, args.generate_config)
        print(f"✅ Configuration template generated: {args.generate_config}")
        return 0

    # Load configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = ConfigLoader.load_from_yaml(args.config)
    else:
        print("Using default configuration")
        config = create_default_config()

    # Apply command-line overrides
    _apply_cli_overrides(config, args, from_default=not args.config)

    # Run pipeline
    if config.mode == "streaming":
        if not config.dataset.embeddings_only:
            print(
                "Error: --mode streaming requires --embeddings-only "
                "(training aggregates subjects and must remain batch).",
                file=sys.stderr,
            )
            return 1
        try:
            from parallel_runner import run_parallel_pipeline
        except ImportError as e:
            print(f"Error: could not import parallel_runner: {e}", file=sys.stderr)
            return 1
        subjects_dir = config.dataset.input_path or config.dataset.morphologist_graphs
        output_dir = config.outputs_path
        log_dir = config.log_dir or str(Path(output_dir) / "logs")
        return run_parallel_pipeline(
            subjects_dir=subjects_dir,
            output_dir=output_dir,
            config=config.dataset,
            n_workers=config.n_workers,
            log_dir=log_dir,
            worker_timeout=config.worker_timeout,
            dry_run=getattr(args, "dry_run", False),
        )

    orchestrator = PipelineOrchestrator(config)
    return orchestrator.run()


if __name__ == "__main__":
    sys.exit(main())
