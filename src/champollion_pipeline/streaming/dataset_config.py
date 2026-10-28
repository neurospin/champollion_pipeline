"""Champollion config of a streaming run: generated once per dataset, before inference starts."""

from __future__ import annotations

import logging
from os.path import commonpath, isdir, isfile, join, realpath

import champollion_utils.logger  # noqa: F401 - installs Logger.trace/output/fail

from champollion_pipeline.config import DatasetConfig
from champollion_pipeline.derivatives_layout import compute_champollion_configs_root
from champollion_pipeline.generate_champollion_config import GenerateChampollionConfig
from champollion_pipeline.streaming.settings import StreamingSettings
from champollion_pipeline.utils.lib import find_dataset_folder

logger = logging.getLogger(__name__)


def ensure_dataset_config(settings: StreamingSettings, dataset: DatasetConfig) -> str:
    """Generate the dataset's Champollion config unless it already exists; return the configs root.

    The configs root is always outside the champollion_V1 checkout; the dataset_localization YAML the
    config script writes last marks a finished run, so a second call returns without running it again.
    """
    root = compute_champollion_configs_root(find_dataset_folder(dataset.crops_path, dataset.name), dataset.name)
    script = GenerateChampollionConfig()
    script.parse_args(
        [
            dataset.crops_path,
            f"--dataset={dataset.name}",
            f"--masks={dataset.masks_version}",
            f"--localization={dataset.dataset_localization}",
            f"--output={root}",
            f"--external-config={root}",
        ]
    )
    checkout = realpath(script.args.champollion_loc)
    if commonpath([realpath(root), checkout]) == checkout:
        raise ValueError(f"Configs root {root} is inside the champollion_V1 checkout {checkout}")
    if settings.dry_run:
        logger.trace(f"{dataset.name}: dry run, Champollion config would be written to {root}")
        return root
    marker = join(root, "dataset_localization", f"{dataset.dataset_localization}.yaml")
    if isfile(marker):
        logger.trace(f"{dataset.name}: Champollion config already present ({marker})")
        return root
    if not isdir(dataset.crops_path):
        raise FileNotFoundError(f"Crops path {dataset.crops_path} not found for dataset {dataset.name}")
    logger.trace(f"{dataset.name}: generating the Champollion config from {dataset.crops_path} into {root}")
    code = script.run()
    if code != 0:
        raise RuntimeError(f"Champollion config generation failed for dataset {dataset.name} (exit code {code})")
    return root
