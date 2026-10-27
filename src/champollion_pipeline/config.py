#!/usr/bin/env python3
"""
Pipeline configuration: DatasetConfig, PipelineConfig and ConfigLoader (YAML load and save).
"""

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from champollion_pipeline.derivatives_layout import DEFAULT_MASKS_VERSION
from champollion_pipeline.generate_embeddings import DEFAULT_CHECKPOINT_EPOCH

# Dataset config keys that no longer exist; old YAML files still carrying them load with a warning.
_REMOVED_DATASET_KEYS = {
    "use_best_model": "it had no effect; set dataset.checkpoint_epoch to choose the training epoch whose "
    "checkpoint is evaluated",
    "use_last_checkpoint": "replaced by dataset.checkpoint_epoch (training epoch N, 0-based, of the Lightning "
    "checkpoint epoch=N-step=*.ckpt to evaluate; default 80)",
}

# Dataset fields never written by save_to_yaml: secrets, supplied through the environment (REQ-HFTOKEN-01).
_UNSAVED_DATASET_KEYS = ("hf_token",)


@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""

    name: str = "example_dataset"
    dataset_localization: str = "local"  # noqa: V107 - YAML config field
    datasets_root: str = ""
    datasets: list[str] = field(default_factory=lambda: ["example"])  # noqa: V107 - YAML config field
    labels: list[str] = field(default_factory=lambda: ["Sex"])  # noqa: V107 - YAML config field

    # Paths
    input_path: str = ""  # noqa: V107 - YAML config field
    morphologist_graphs: str = ""  # noqa: V107 - YAML config field
    cortical_tiles_output: str = ""  # noqa: V107 - YAML config field
    crops_path: str = ""
    embeddings_path: str = ""

    # Processing parameters
    njobs: int = 22
    path_to_graph: str = "t1mri/default_acquisition/default_analysis/folds/3.3/base"
    path_sk_with_hull: str = "t1mri/default_acquisition/default_analysis/segmentation/mesh"
    sk_qc_path: str = ""
    masks_version: str = DEFAULT_MASKS_VERSION
    skel_seed: int | None = None  # None: the skel_seed in <derivatives>/pipeline_loop_2mm.json
    labelling_session: str = ""  # "": run_cortical_tiles picks the session per subject

    # Region filter (empty = all 56 regions)
    regions: list[str] = field(default_factory=list)

    # Embeddings parameters
    classifier_name: str = "svm"  # noqa: V107 - YAML config field
    overwrite: bool = False
    embeddings_only: bool = False  # noqa: V107 - YAML config field
    checkpoint_epoch: int = DEFAULT_CHECKPOINT_EPOCH
    models_cache: str | None = None  # None: <datasets_root>/derivatives/champollion_V1/models_cache
    subsets: list[str] = field(default_factory=lambda: ["full"])  # noqa: V107 - YAML config field
    epochs: list[int | None] = field(default_factory=lambda: [None])  # noqa: V107 - YAML config field
    split: str = "random"
    cv: int = 5  # noqa: V107 - YAML config field
    splits_basedir: str | None = None  # noqa: V107 - YAML config field
    idx_region_evaluation: int | None = None  # noqa: V107 - YAML config field
    short_name: str = "eval"  # noqa: V107 - YAML config field

    # HuggingFace parameters
    hf_enabled: bool = False
    hf_repo_id: str | None = None  # noqa: V107 - YAML config field
    hf_token: str | None = None

    # Unused: no stage reads it. Kept so existing YAML files that set it still load.
    config_path: str | None = None

    # CPU mode (disable CUDA)
    cpu: bool = False

    # BIDS database layout (subjects/sub-*/ses-*/...)
    bids: bool = False

    # Snapshots parameters
    snapshots_path: str = ""
    reference_data_path: str = ""  # noqa: V107 - YAML config field
    champollion_data_root: str | None = None


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""

    # Root paths
    root_path: str = str(Path(__file__).parents[2])
    data_path: str = ""
    models_path: str = ""
    outputs_path: str = ""  # noqa: V107 - YAML config field
    champollion_v1_path: str = ""

    # Pipeline stages to execute
    stages: dict[str, bool] = field(
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
    dependencies: dict[str, list[str]] = field(
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
    stop_on_error: bool = True  # noqa: V107 - YAML config field
    verbose: bool = False

    # Logging
    log_level: str = "INFO"  # noqa: V107 - YAML config field
    log_dir: str = ""  # noqa: V107 - YAML config field
    log_to_file: bool = True  # noqa: V107 - YAML config field
    log_to_console: bool = True  # noqa: V107 - YAML config field
    # Run-log directory of the in-process stages: "" = each stage's default_log_path(), None = no run log.
    log: str | None = None  # noqa: V107 - YAML config field

    # Dataset configuration
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


class ConfigLoader:
    """Load and manage pipeline configuration."""

    @staticmethod
    def load_from_yaml(config_path: str) -> PipelineConfig:
        """Load configuration from YAML file."""
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        return ConfigLoader._dict_to_config(config_dict)

    @staticmethod
    def _dict_to_config(config_dict: dict) -> PipelineConfig:
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
        config_dict = asdict(config)
        for key in _UNSAVED_DATASET_KEYS:
            config_dict["dataset"].pop(key, None)

        with open(output_path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
        if config.dataset.hf_token:
            print(
                f"Note: dataset.hf_token not written to {output_path}; set the HF_TOKEN environment variable instead."
            )
