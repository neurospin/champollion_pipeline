#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Champollion derivatives path layout.

Shared constants and helpers for where champollion_V1 artefacts live inside a
dataset's own derivatives tree. Pure path computation — no I/O, no imports
from ``external/``. Used by stage 3 (writer) and training (reader) so the two
sides cannot drift apart.
"""

from os.path import abspath, join

CHAMPOLLION_DERIVATIVES_FOLDER = "champollion_V1"
CONFIGS_SUBFOLDER = "configs"


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
