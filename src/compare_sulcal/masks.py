"""Mask loading and per-mask metrics (voxel diff, Wasserstein distance, buckets) of the compare modes."""

import math
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance as _wasserstein_1d

from compare_sulcal import brainvisa

ONE_SIDE_EMPTY_BUCKET = "one_side_empty"


# --------------------------------------------------------------------------- #
# Shared NIfTI mask helpers
# --------------------------------------------------------------------------- #


def load_mask_vol(path: str) -> np.ndarray:
    """Read a NIfTI mask with PyAIMS and return a 3-D float64 array (x, y, z).

    Raw integer counts are preserved as-is for use as Wasserstein weights.
    """
    vol = brainvisa.aims.read(path)
    return np.asarray(vol, dtype=np.float64).squeeze()


def voxel_diff(a: np.ndarray, b: np.ndarray) -> dict:
    """Count voxels that changed between two mask volumes.

    Returns:
      changed        – total voxels where a[i] != b[i]
      added          – voxels that went from 0 in a to nonzero in b
      removed        – voxels that went from nonzero in a to 0 in b
      value_changed  – voxels nonzero in both a and b whose values differ
    Invariant: changed == added + removed + value_changed.
    """
    a_nz = a != 0
    b_nz = b != 0
    return {
        "changed": int(np.sum(a != b)),
        "added": int(np.sum(~a_nz & b_nz)),
        "removed": int(np.sum(a_nz & ~b_nz)),
        "value_changed": int(np.sum(a_nz & b_nz & (a != b))),
    }


def wasserstein_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Approximate 3-D Wasserstein distance (in voxels) via axis-marginals.

    Projects each 3-D map onto X, Y, Z, computes 1-D Wasserstein distances,
    then combines as Euclidean norm. Returns 0.0 when both maps are empty and
    math.inf when exactly one is empty (no finite cost moves mass onto nothing).
    """
    is_a_empty = a.sum() == 0.0
    is_b_empty = b.sum() == 0.0
    if is_a_empty and is_b_empty:
        return 0.0
    if is_a_empty or is_b_empty:
        return math.inf
    d_sq = 0.0
    for axis in range(3):
        other = tuple(i for i in range(3) if i != axis)
        proj_a = a.sum(axis=other)
        proj_b = b.sum(axis=other)
        s_a, s_b = proj_a.sum(), proj_b.sum()
        if s_a == 0.0 or s_b == 0.0:
            continue
        positions = np.arange(len(proj_a), dtype=np.float64)
        d = float(_wasserstein_1d(positions, positions, proj_a / s_a, proj_b / s_b))
        d_sq += d * d
    return float(np.sqrt(d_sq))


def bucket_label(value: float, step: float) -> str:
    low = (value // step) * step
    high = low + step
    if step == int(step):
        return f"{int(low)}-{int(high)}vox"
    return f"{low:.2f}-{high:.2f}vox"


def sort_buckets(b: dict) -> dict:
    return dict(sorted(b.items(), key=_compute_bucket_sort_key))


def _compute_bucket_sort_key(item) -> tuple[bool, float]:
    """Order numeric buckets by lower bound and put the one-side-empty bucket last."""
    label = item[0]
    if label == ONE_SIDE_EMPTY_BUCKET:
        return (True, 0.0)
    return (False, float(label.split("-")[0]))


def _normalize_distance_for_json(distance: float) -> float | None:
    """Map an infinite distance to None so the report stays strict JSON."""
    return None if math.isinf(distance) else distance


def find_masks(directory: Path, glob_pattern: str) -> dict[str, str]:
    """Return {path relative to directory: resolved absolute path} for files matching glob_pattern, sorted."""
    return {str(p.relative_to(directory)): str(p.resolve()) for p in sorted(directory.glob(glob_pattern))}


def save_xor_vol(ref_path: str, a: np.ndarray, b: np.ndarray, out_path: str) -> None:
    """Write a NIfTI XOR image (1 = voxel differs, 0 = same) to out_path.

    Voxel size and transformation are copied from ref_path so the output
    sits in the same space as the input masks.
    """
    ref_vol = brainvisa.aims.read(ref_path)
    xor = (a != b).astype(np.int16)
    if xor.ndim == 3:
        xor = xor[..., np.newaxis]
    out_vol = brainvisa.aims.Volume(xor)
    out_vol.copyHeaderFrom(ref_vol.header())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    brainvisa.aims.write(out_vol, out_path)
