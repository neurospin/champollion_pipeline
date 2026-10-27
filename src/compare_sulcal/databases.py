"""Per-subject sulcal voxel counts and mask statistics of the compare databases mode."""

import glob
import logging
import os
import sys
from os.path import join

import numpy as np

from compare_sulcal import brainvisa

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Database-mode module-level workers (must be picklable for joblib)
# --------------------------------------------------------------------------- #


def _get_subject_voxel_counts(sub, brainvisa_dir):
    """Worker: load one graph and return (subject_name, {sulcus: voxel_count})."""
    if brainvisa_dir not in sys.path:
        sys.path.insert(0, brainvisa_dir)

    matches = sorted(glob.glob(join(sub["dir"], sub["graph_file"])))
    if not matches:
        return sub["subject"], None
    if len(matches) > 1:
        logger.warning(f"{len(matches)} graphs match subject {sub['subject']}; using {matches[0]}")

    graph = brainvisa.aims.read(matches[0])
    counts: dict = {}
    for vertex in graph.vertices():
        name = vertex.get("name")
        if name is None:
            continue
        n = 0
        for bucket_name in ("aims_ss", "aims_bottom", "aims_other"):
            bucket = vertex.get(bucket_name)
            if bucket is not None:
                n += len(list(bucket[0].keys()))
        counts[name] = counts.get(name, 0) + n
    return sub["subject"], counts


def _mask_stats(mask_dir: str, brainvisa_dir: str) -> dict:
    """Return {relative_path: (max, sum, nonzero)} for all masks in mask_dir."""
    if brainvisa_dir not in sys.path:
        sys.path.insert(0, brainvisa_dir)

    stats = {}
    for path in sorted(glob.glob(join(mask_dir, "*/*.nii.gz"))):
        rel = os.path.relpath(path, mask_dir)
        arr = np.asarray(brainvisa.aims.read(path))
        stats[rel] = (int(arr.max()), int(arr.sum()), int(np.count_nonzero(arr)))
    return stats
