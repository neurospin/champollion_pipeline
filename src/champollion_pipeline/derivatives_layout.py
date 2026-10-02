#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Champollion derivatives path layout and naming.

Shared constants and helpers for where champollion_V1 artefacts live inside a
dataset's own derivatives tree. Pure path computation — no I/O, no imports
from ``external/``. Used by stage 3 (writer) and training (reader) so the two
sides cannot drift apart.
"""

from os.path import abspath, join

CHAMPOLLION_DERIVATIVES_FOLDER = "champollion_V1"
CONFIGS_SUBFOLDER = "configs"
HEMISPHERES = ("left", "right")


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
    return abspath(join(dataset_parent, dataset, "derivatives", CHAMPOLLION_DERIVATIVES_FOLDER, CONFIGS_SUBFOLDER))


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
