#!/usr/bin/env python3
"""
Champollion derivatives path layout and naming.

Shared constants and helpers for where champollion_V1 artefacts live inside a
dataset's own derivatives tree. Pure path computation — no I/O, no imports
from ``external/``. Used by stage 3 (writer) and training (reader) so the two
sides cannot drift apart.
"""

from datetime import datetime
from os.path import abspath, join

from champollion_pipeline.utils.lib import CORTICAL_TILES_VERSION

DERIVATIVES_DIRNAME = "derivatives"
CHAMPOLLION_DERIVATIVES_FOLDER = "champollion_V1"
DEFAULT_MASKS_VERSION = "canonical_25"
CONFIGS_SUBFOLDER = "configs"
REGION_EMBEDDINGS_SUBFOLDER = "region_embeddings"
COMBINED_EMBEDDINGS_SUBFOLDER = "embeddings"
SNAPSHOTS_SUBFOLDER = "snapshots"
MODELS_CACHE_SUBFOLDER = "models_cache"
HEMISPHERES = ("left", "right")
MORPHOLOGIST_DERIVATIVES_FOLDER = "morphologist-6.0"
CORTICAL_TILES_DERIVATIVES_PREFIX = "cortical_tiles-"
RUN_LOG_DIRNAME = ".log"
RUN_LOG_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
# Stages whose run log lives in the per-mask-version folder, in pipeline order.
CHAMPOLLION_RUN_LOG_STAGES: tuple[str, ...] = ("config", "embeddings", "combine", "snapshots", "training")
# Every stage name accepted by compute_run_log_derivatives_dir.
RUN_LOG_STAGES: tuple[str, ...] = ("morphologist", "cortical_tiles", *CHAMPOLLION_RUN_LOG_STAGES)


def compute_champollion_configs_root(dataset_parent: str, dataset: str) -> str:
    """Return the default Hydra configs root for a dataset.

    Args:
        dataset_parent: parent directory of the <dataset> directory
            (what find_dataset_folder returns, or <repo>/data for training).
        dataset: dataset name.

    Returns:
        abspath(<dataset_parent>/<dataset>/derivatives/champollion_V1/configs)

    Complexity: O(1).
    """
    return abspath(
        join(dataset_parent, dataset, DERIVATIVES_DIRNAME, CHAMPOLLION_DERIVATIVES_FOLDER, CONFIGS_SUBFOLDER)
    )


def compute_masks_version_dir(datasets_root: str, masks_version: str) -> str:
    """Return the champollion_V1 directory of one mask version inside a dataset.

    Args:
        datasets_root: dataset root directory (the one holding ``derivatives/``).
        masks_version: cortical_tiles mask version (e.g. "canonical_25").

    Returns:
        <datasets_root>/derivatives/champollion_V1/<masks_version>

    Complexity: O(1).
    """
    return join(datasets_root, DERIVATIVES_DIRNAME, CHAMPOLLION_DERIVATIVES_FOLDER, masks_version)


def compute_region_embeddings_dir(datasets_root: str, masks_version: str) -> str:
    """Return the per-region embeddings directory (stage 4 output, stage 5 input).

    Args:
        datasets_root: dataset root directory.
        masks_version: cortical_tiles mask version.

    Returns:
        <datasets_root>/derivatives/champollion_V1/<masks_version>/region_embeddings

    Complexity: O(1).
    """
    return join(compute_masks_version_dir(datasets_root, masks_version), REGION_EMBEDDINGS_SUBFOLDER)


def compute_combined_embeddings_dir(datasets_root: str, masks_version: str) -> str:  # noqa: V103 - used by main.py
    """Return the combined embeddings directory (stage 5 output, stage 6 input).

    Args:
        datasets_root: dataset root directory.
        masks_version: cortical_tiles mask version.

    Returns:
        <datasets_root>/derivatives/champollion_V1/<masks_version>/embeddings

    Complexity: O(1).
    """
    return join(compute_masks_version_dir(datasets_root, masks_version), COMBINED_EMBEDDINGS_SUBFOLDER)


def compute_snapshots_dir(datasets_root: str, masks_version: str) -> str:  # noqa: V103 - used by main.py
    """Return the default snapshots directory (stage 6 output).

    Args:
        datasets_root: dataset root directory.
        masks_version: cortical_tiles mask version.

    Returns:
        <datasets_root>/derivatives/champollion_V1/<masks_version>/snapshots

    Complexity: O(1).
    """
    return join(compute_masks_version_dir(datasets_root, masks_version), SNAPSHOTS_SUBFOLDER)


def compute_models_cache_dir(datasets_root: str) -> str:
    """Return where downloaded or extracted models are cached for a dataset.

    Args:
        datasets_root: dataset root directory (the one holding ``derivatives/``).

    Returns:
        abspath(<datasets_root>/derivatives/champollion_V1/models_cache)

    Complexity: O(1).
    """
    return abspath(join(datasets_root, DERIVATIVES_DIRNAME, CHAMPOLLION_DERIVATIVES_FOLDER, MODELS_CACHE_SUBFOLDER))


def compute_region_model_name(region: str, hemisphere: str) -> str:
    """Return the champollion_V1 model name for one region and hemisphere.

    The same name is used for the stage-3 dataset YAML and the model directory.

    Args:
        region: sulcal region name as given to cortical_tiles (e.g. "S.C.-sylv.").
        hemisphere: one of HEMISPHERES ("left" or "right").

    Returns:
        region with dots removed, plus "_" and hemisphere (e.g. "SC-sylv_left").

    Raises:
        ValueError: if hemisphere is not in HEMISPHERES.

    Complexity: O(len(region)).
    """
    if hemisphere not in HEMISPHERES:
        raise ValueError(f"hemisphere must be one of {HEMISPHERES}, got {hemisphere!r}")
    return f"{region.replace('.', '')}_{hemisphere}"


def compute_run_log_derivatives_dir(
    stage: str,
    root: str,
    masks_version: str = DEFAULT_MASKS_VERSION,
    cortical_tiles_version: str = CORTICAL_TILES_VERSION,
) -> str:
    """Return the derivatives folder that holds a stage's ``.log`` folder.

    Args:
        stage: one of RUN_LOG_STAGES.
        root: dataset root, the directory holding ``derivatives/``.
        masks_version: mask version; used only for CHAMPOLLION_RUN_LOG_STAGES.
        cortical_tiles_version: used only for "cortical_tiles".

    Raises:
        ValueError: stage not in RUN_LOG_STAGES.

    Pure, no I/O, O(1).
    """
    if stage not in RUN_LOG_STAGES:
        raise ValueError(f"stage must be one of {RUN_LOG_STAGES}, got {stage!r}")
    if stage == "morphologist":
        return join(root, DERIVATIVES_DIRNAME, MORPHOLOGIST_DERIVATIVES_FOLDER)
    if stage == "cortical_tiles":
        return join(root, DERIVATIVES_DIRNAME, CORTICAL_TILES_DERIVATIVES_PREFIX + cortical_tiles_version)
    return compute_masks_version_dir(root, masks_version)


def compute_run_log_path(derivatives_dir: str, stage: str, started_at: datetime) -> str:
    """Return ``<derivatives_dir>/.log/<stage>_<YYYYmmdd-HHMMSS>.log`` for a run started at started_at.

    Pure, no I/O, O(1).
    """
    return join(derivatives_dir, RUN_LOG_DIRNAME, f"{stage}_{started_at.strftime(RUN_LOG_TIMESTAMP_FORMAT)}.log")
