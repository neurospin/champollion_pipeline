"""Streaming combine: run PutTogetherEmbeddings once, after the resident consumer has drained.

The path rules restate main.py's precedence (_compute_region_embeddings_source and
PutTogetherEmbeddingsStage.build_args), with config.outputs_path mapped to settings.output_dir.
"""

from __future__ import annotations

import logging
import os
from os.path import isdir

import champollion_utils.logger  # noqa: F401 - installs Logger.trace/output

from champollion_pipeline.config import DatasetConfig
from champollion_pipeline.derivatives_layout import compute_combined_embeddings_dir, compute_region_embeddings_dir
from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings
from champollion_pipeline.streaming.settings import StreamingSettings

logger = logging.getLogger(__name__)

_COMBINED_SUFFIX = "_embeddings.csv"


def run_combine(settings: StreamingSettings, dataset: DatasetConfig) -> str:
    """Collect the per-region embeddings of a streaming run into the combined directory; return that directory."""
    source = _resolve_embeddings_source(dataset)
    combined = _resolve_combined_dir(settings, dataset)
    if settings.dry_run:
        logger.trace(f"Combine dry run: would combine {source} into {combined}")
        return combined
    if not isdir(source):
        raise FileNotFoundError(f"Region embeddings source not found: {source}")
    logger.trace(f"Combine: embeddings source {source}, combined directory {combined}")
    script = PutTogetherEmbeddings()
    script.parse_args([source, f"--output_path={combined}"])
    code = script.run()
    if code != 0:
        raise RuntimeError(f"PutTogetherEmbeddings failed with return code {code} for combined directory {combined}")
    n = sum(1 for name in os.listdir(combined) if name.endswith(_COMBINED_SUFFIX)) if isdir(combined) else 0
    if n == 0:
        logger.warning(f"Combine QC: no *{_COMBINED_SUFFIX} file in combined directory {combined}")
    else:
        logger.output(f"Combined {n} region embeddings files into {combined}")
    return combined


def _resolve_embeddings_source(dataset: DatasetConfig) -> str:
    """Explicit embeddings_path, else the versioned region embeddings directory under datasets_root."""
    if dataset.embeddings_path:
        return dataset.embeddings_path
    if dataset.datasets_root:
        return compute_region_embeddings_dir(dataset.datasets_root, dataset.masks_version)
    raise ValueError(
        "Cannot resolve the region embeddings source: DatasetConfig.embeddings_path "
        "and DatasetConfig.datasets_root are both empty"
    )


def _resolve_combined_dir(settings: StreamingSettings, dataset: DatasetConfig) -> str:
    """Versioned combined directory under datasets_root, else cortical_tiles_output, else the run output_dir."""
    if dataset.datasets_root:
        return compute_combined_embeddings_dir(dataset.datasets_root, dataset.masks_version)
    if dataset.cortical_tiles_output:
        return dataset.cortical_tiles_output
    return settings.output_dir
