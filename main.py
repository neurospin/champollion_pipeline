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

import os
import sys
import logging
import argparse
import yaml
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field

# Add src to path for imports
pipeline_src = Path(__file__).parent / "src"
sys.path.insert(0, str(pipeline_src))

# Import pipeline scripts
try:
    from generate_morphologist_graphs import GenerateMorphologistGraphs
    from run_cortical_tiles import RunCorticalTiles
    from generate_champollion_config import GenerateChampollionConfig
    from generate_embeddings import GenerateEmbeddings
    from put_together_embeddings import PutTogetherEmbeddings
except ImportError as e:
    print(f"Warning: Could not import pipeline scripts: {e}")
    print("Some stages may not be available.")


# ====================== Configuration Management ======================

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

    # Processing parameters
    njobs: int = 22
    path_to_graph: str = "t1mri/default_acquisition/default_analysis/folds/3.3/base"
    path_sk_with_hull: str = "t1mri/default_acquisition/default_analysis/segmentation/mesh"
    sk_qc_path: str = ""

    # Embeddings parameters
    classifier_name: str = "svm"
    overwrite: bool = False
    embeddings_only: bool = False
    use_best_model: bool = False
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

    # External dataset config path (for datasets not in champollion_V1/contrastive/configs)
    config_path: Optional[str] = None

    # CPU mode (disable CUDA)
    cpu: bool = False


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
    stages: Dict[str, bool] = field(default_factory=lambda: {
        'generate_morphologist_graphs': False,
        'run_cortical_tiles': False,
        'generate_champollion_config': False,
        'generate_embeddings': True,
        'put_together_embeddings': False,
    })

    # Stage dependencies
    dependencies: Dict[str, List[str]] = field(default_factory=lambda: {
        'generate_morphologist_graphs': [],
        'run_cortical_tiles': ['generate_morphologist_graphs'],
        'generate_champollion_config': ['run_cortical_tiles'],
        'generate_embeddings': ['generate_champollion_config'],
        'put_together_embeddings': ['generate_embeddings'],
    })

    # Execution settings
    mode: str = "sequential"
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
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return ConfigLoader._dict_to_config(config_dict)

    @staticmethod
    def _dict_to_config(config_dict: Dict) -> PipelineConfig:
        """Convert dictionary to PipelineConfig object."""
        # Extract dataset config
        dataset_dict = config_dict.pop('dataset', {})
        dataset_config = DatasetConfig(**dataset_dict)

        # Create pipeline config
        pipeline_config = PipelineConfig(**config_dict, dataset=dataset_config)

        return pipeline_config

    @staticmethod
    def save_to_yaml(config: PipelineConfig, output_path: str):
        """Save configuration to YAML file."""
        # Convert to dictionary
        config_dict = {
            'root_path': config.root_path,
            'data_path': config.data_path,
            'models_path': config.models_path,
            'outputs_path': config.outputs_path,
            'champollion_v1_path': config.champollion_v1_path,
            'stages': config.stages,
            'dependencies': config.dependencies,
            'mode': config.mode,
            'stop_on_error': config.stop_on_error,
            'verbose': config.verbose,
            'log_level': config.log_level,
            'log_dir': config.log_dir,
            'log_to_file': config.log_to_file,
            'log_to_console': config.log_to_console,
            'dataset': {
                'name': config.dataset.name,
                'dataset_localization': config.dataset.dataset_localization,
                'datasets_root': config.dataset.datasets_root,
                'datasets': config.dataset.datasets,
                'labels': config.dataset.labels,
                'input_path': config.dataset.input_path,
                'morphologist_graphs': config.dataset.morphologist_graphs,
                'cortical_tiles_output': config.dataset.cortical_tiles_output,
                'crops_path': config.dataset.crops_path,
                'embeddings_path': config.dataset.embeddings_path,
                'njobs': config.dataset.njobs,
                'path_to_graph': config.dataset.path_to_graph,
                'path_sk_with_hull': config.dataset.path_sk_with_hull,
                'sk_qc_path': config.dataset.sk_qc_path,
                'classifier_name': config.dataset.classifier_name,
                'overwrite': config.dataset.overwrite,
                'embeddings_only': config.dataset.embeddings_only,
                'use_best_model': config.dataset.use_best_model,
                'subsets': config.dataset.subsets,
                'epochs': config.dataset.epochs,
                'split': config.dataset.split,
                'cv': config.dataset.cv,
                'splits_basedir': config.dataset.splits_basedir,
                'idx_region_evaluation': config.dataset.idx_region_evaluation,
                'short_name': config.dataset.short_name,
                'hf_enabled': config.dataset.hf_enabled,
                'hf_repo_id': config.dataset.hf_repo_id,
                'hf_token': config.dataset.hf_token,
                'config_path': config.dataset.config_path,
                'cpu': config.dataset.cpu,
            }
        }

        with open(output_path, 'w') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)


# ====================== Pipeline Stage Strategy Pattern ======================

@dataclass
class StageResult:
    """Result of a pipeline stage execution."""
    stage_name: str
    success: bool
    message: str
    return_code: int = 0
    outputs: Optional[Dict[str, Any]] = None


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
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Starting stage: {self.name}")
        self.logger.info(f"{'='*60}")

    def log_end(self, result: StageResult):
        """Log stage end."""
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        self.logger.info(f"{status}: {self.name} - {result.message}")
        self.logger.info(f"{'='*60}\n")


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
            # Placeholder - implement actual logic
            # script = GenerateMorphologistGraphs()
            # result_code = script.run()

            result = StageResult(
                stage_name=self.name,
                success=True,
                message="Morphologist graphs generated successfully",
                return_code=0
            )
        except Exception as e:
            self.logger.exception(f"Exception in {self.name}")
            result = StageResult(
                stage_name=self.name,
                success=False,
                message=f"Failed: {str(e)}",
                return_code=1
            )

        self.log_end(result)
        return result


class RunCorticalTilesStage(PipelineStage):
    """Stage for running cortical_tiles."""

    def validate(self) -> bool:
        """Validate that Morphologist graphs exist."""
        graphs_path = Path(self.config.dataset.morphologist_graphs)
        if not graphs_path.exists():
            self.logger.error(
                f"Morphologist graphs path does not exist: {graphs_path}"
            )
            return False
        return True

    def execute(self) -> StageResult:
        """Execute cortical_tiles."""
        self.log_start()
        try:
            self.logger.info(
                "Running cortical_tiles to generate sulcal regions..."
            )

            # Build arguments from config
            args = [
                str(self.config.dataset.morphologist_graphs),
                str(self.config.dataset.cortical_tiles_output),
                f"--path_to_graph={self.config.dataset.path_to_graph}",
                f"--path_sk_with_hull={self.config.dataset.path_sk_with_hull}",
                f"--njobs={self.config.dataset.njobs}"
            ]

            if self.config.dataset.sk_qc_path:
                args.append(f"--sk_qc_path={self.config.dataset.sk_qc_path}")

            # Parse and run
            script = RunCorticalTiles()
            script.parse_args(args)
            return_code = script.run()

            result = StageResult(
                stage_name=self.name,
                success=(return_code == 0),
                message="Cortical tiles completed" if return_code == 0 else "Failed",
                return_code=return_code
            )
        except Exception as e:
            self.logger.exception(f"Exception in {self.name}")
            result = StageResult(
                stage_name=self.name,
                success=False,
                message=f"Failed: {str(e)}",
                return_code=1
            )

        self.log_end(result)
        return result


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

            args = [
                str(self.config.dataset.crops_path),
                f"--dataset={self.config.dataset.name}"
            ]

            script = GenerateChampollionConfig()
            script.parse_args(args)
            return_code = script.run()

            result = StageResult(
                stage_name=self.name,
                success=(return_code == 0),
                message="Champollion config generated" if return_code == 0 else "Config generation failed",
                return_code=return_code
            )
        except Exception as e:
            self.logger.exception(f"Exception in {self.name}")
            result = StageResult(
                stage_name=self.name,
                success=False,
                message=f"Failed: {str(e)}",
                return_code=1
            )

        self.log_end(result)
        return result


class GenerateEmbeddingsStage(PipelineStage):
    """Stage for generating embeddings."""

    def validate(self) -> bool:
        """Validate that models and data exist."""
        models_path = Path(self.config.models_path)
        if not models_path.exists():
            self.logger.error(f"Models path does not exist: {models_path}")
            return False
        return True

    def execute(self) -> StageResult:
        """Execute embeddings generation."""
        self.log_start()
        try:
            self.logger.info("Generating embeddings and training classifiers...")

            # Build arguments from config
            args = [
                str(self.config.models_path),
                self.config.dataset.dataset_localization,
                self.config.dataset.datasets_root,
                self.config.dataset.short_name,
            ]

            # Add datasets
            args.append("--datasets")
            args.extend(self.config.dataset.datasets)

            # Add labels
            args.append("--labels")
            args.extend(self.config.dataset.labels)

            # Add optional parameters
            args.append(f"--classifier_name={self.config.dataset.classifier_name}")

            if self.config.dataset.overwrite:
                args.append("--overwrite")
            if self.config.dataset.embeddings_only:
                args.append("--embeddings_only")
            if self.config.dataset.use_best_model:
                args.append("--use_best_model")

            # Add subsets
            args.append("--subsets")
            args.extend(self.config.dataset.subsets)

            # Add epochs
            args.append("--epochs")
            for epoch in self.config.dataset.epochs:
                epoch_str = "None" if epoch is None else str(epoch)
                args.append(epoch_str)

            args.append(f"--split={self.config.dataset.split}")
            args.append(f"--cv={self.config.dataset.cv}")

            if self.config.dataset.splits_basedir:
                args.append(f"--splits_basedir={self.config.dataset.splits_basedir}")

            if self.config.dataset.idx_region_evaluation is not None:
                args.append(f"--idx_region_evaluation={self.config.dataset.idx_region_evaluation}")

            if self.config.verbose:
                args.append("--verbose")

            # HuggingFace parameters
            if self.config.dataset.hf_enabled:
                args.append("--population_source=huggingface")
                args.append(f"--population_source_path={self.config.dataset.hf_repo_id}")
                if self.config.dataset.hf_token:
                    args.append(f"--hf_token={self.config.dataset.hf_token}")

            # External config path for dataset configs
            if self.config.dataset.config_path:
                args.append(f"--config_path={self.config.dataset.config_path}")

            # CPU mode
            if self.config.dataset.cpu:
                args.append("--cpu")

            self.logger.debug(f"Arguments: {args}")

            script = GenerateEmbeddings()
            script.parse_args(args)
            return_code = script.run()

            result = StageResult(
                stage_name=self.name,
                success=(return_code == 0),
                message="Embeddings generated" if return_code == 0 else "Embedding generation failed",
                return_code=return_code
            )
        except Exception as e:
            self.logger.exception(f"Exception in {self.name}")
            result = StageResult(
                stage_name=self.name,
                success=False,
                message=f"Failed: {str(e)}",
                return_code=1
            )

        self.log_end(result)
        return result


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

            # Placeholder - implement actual logic
            # script = PutTogetherEmbeddings()
            # return_code = script.run()

            result = StageResult(
                stage_name=self.name,
                success=True,
                message="Embeddings combined successfully",
                return_code=0
            )
        except Exception as e:
            self.logger.exception(f"Exception in {self.name}")
            result = StageResult(
                stage_name=self.name,
                success=False,
                message=f"Failed: {str(e)}",
                return_code=1
            )

        self.log_end(result)
        return result


# ====================== Pipeline Orchestrator ======================

class PipelineOrchestrator:
    """Orchestrates the execution of pipeline stages."""

    # Stage registry mapping names to classes
    STAGE_REGISTRY = {
        'generate_morphologist_graphs': GenerateMorphologistGraphsStage,
        'run_cortical_tiles': RunCorticalTilesStage,
        'generate_champollion_config': GenerateChampollionConfigStage,
        'generate_embeddings': GenerateEmbeddingsStage,
        'put_together_embeddings': PutTogetherEmbeddingsStage,
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
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File handler
        if self.config.log_to_file:
            log_dir = Path(self.config.log_dir) if self.config.log_dir else Path(self.config.outputs_path) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "pipeline.log")
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _register_stages(self):
        """Register all available pipeline stages."""
        for stage_name, stage_class in self.STAGE_REGISTRY.items():
            self.stages[stage_name] = stage_class(
                name=stage_name,
                config=self.config,
                logger=self.logger
            )

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
                self.logger.error(
                    f"Stage '{stage_name}' depends on '{dep}' which hasn't completed"
                )
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

            # Execute
            result = stage.execute()

            if result.success:
                completed_stages.append(stage_name)
            else:
                failed_stages.append(stage_name)
                if self.config.stop_on_error:
                    self.logger.error("Stopping pipeline due to stage failure")
                    break

        # Summary
        self.logger.info("\n" + "="*60)
        self.logger.info("Pipeline Execution Summary")
        self.logger.info("="*60)
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
    config.dataset.datasets_root = str(Path(config.champollion_v1_path) / "contrastive" / "configs" / "dataset")
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
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file"
    )

    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(PipelineOrchestrator.STAGE_REGISTRY.keys()),
        help="Specific stages to run (overrides config)"
    )

    parser.add_argument(
        "--enable-all-stages",
        action="store_true",
        help="Enable all pipeline stages"
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Dataset name (overrides config)"
    )

    parser.add_argument(
        "--models-path",
        type=str,
        help="Path to models directory (overrides config)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--generate-config",
        type=str,
        metavar="OUTPUT_PATH",
        help="Generate a template configuration file and exit"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the pipeline."""
    args = parse_arguments()

    # Generate config template if requested
    if args.generate_config:
        config = create_default_config()
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

    if args.models_path:
        config.models_path = args.models_path

    if args.verbose:
        config.verbose = True
        config.log_level = "DEBUG"

    # Run pipeline
    orchestrator = PipelineOrchestrator(config)
    return orchestrator.run()


if __name__ == "__main__":
    sys.exit(main())
