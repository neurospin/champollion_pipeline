"""Crop alignment and per-subject comparison of the compare crops mode."""

import csv
from pathlib import Path

import numpy as np

from compare_sulcal import brainvisa, masks

# --------------------------------------------------------------------------- #
# Crop-set comparison (crops mode)
# --------------------------------------------------------------------------- #

_CROPS_CSV_COLUMNS = ["region", "side", "subject", "n_a", "n_b", "kept", "lost", "gained", "pct_lost", "dice", "shift"]


_CROP_INFO_ABSENT = frozenset({"no_mask_cropped", "no_transformation"})


_CROPS_INPUT_TYPES = ("skeleton", "label", "extremities", "distmap", "distbottom")


def _load_crop_set(set_dir: Path, region: str, side: str, input_type: str) -> "tuple[np.ndarray, list[str]] | None":
    """Load .npy crop array and subject list for one region/side.

    Returns (array, subjects) where array has shape (n_subjects, x, y, z, 1),
    or None if either file is missing.
    """
    mask_dir = set_dir / region / "mask"
    npy_path = mask_dir / f"{side}{input_type}.npy"
    csv_path = mask_dir / f"{side}{input_type}_subject.csv"
    if not npy_path.is_file() or not csv_path.is_file():
        return None
    arr = np.load(str(npy_path), mmap_mode="r")
    subjects: list[str] = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row:
                subjects.append(str(row[0]))
    return arr, subjects


def _read_crop_header(set_dir: Path, region: str, side: str) -> "dict | None":
    """Read the PyAIMS header of mask/{side}mask_cropped.nii.gz.

    Returns a dict with voxel_size (3 floats), referentials (list[str]) and
    transformations (list of np.ndarray 4x4, row-major); transformations key
    absent when the header has none.  Returns None when the file is missing or
    Finder.check() returns False.
    """
    path = set_dir / region / "mask" / f"{side}mask_cropped.nii.gz"
    if not path.is_file():
        return None
    finder = brainvisa.aims.Finder()
    if not finder.check(str(path)):
        return None
    raw = finder.header()
    result: dict = {
        "voxel_size": raw["voxel_size"][:3],
        "referentials": list(raw["referentials"]),
    }
    if "transformations" in raw:
        result["transformations"] = [np.asarray(list(t), dtype=float).reshape(4, 4) for t in raw["transformations"]]
    return result


def _compute_crop_offset(
    header_a: "dict | None", header_b: "dict | None"
) -> "tuple[tuple[int, int, int] | None, str | None]":
    """Compute the integer voxel offset between two crop sets from AIMS headers.

    Returns (offset, None) on success or (None, reason) on failure.
    Checks proceed in the order defined by the contract.
    """
    if header_a is None or header_b is None:
        return None, "no_mask_cropped"
    if "transformations" not in header_a or "transformations" not in header_b:
        return None, "no_transformation"
    refs_a = header_a["referentials"]
    refs_b_set = set(header_b["referentials"])
    common_ref = next((r for r in refs_a if r in refs_b_set), None)
    if common_ref is None:
        return None, "referential_differs"
    idx_a = refs_a.index(common_ref)
    idx_b = header_b["referentials"].index(common_ref)
    T_a = header_a["transformations"][idx_a]
    T_b = header_b["transformations"][idx_b]
    vs_a = np.array(header_a["voxel_size"][:3], dtype=float)
    vs_b = np.array(header_b["voxel_size"][:3], dtype=float)
    if not np.allclose(vs_a, vs_b):
        return None, "voxel_size_differs"
    R_a = T_a[:3, :3]
    diag_r = np.diag(R_a)
    if not (np.all(diag_r != 0) and np.allclose(R_a, np.diag(np.sign(diag_r)))):
        return None, "non_axis_aligned"
    R_b = T_b[:3, :3]
    if not np.allclose(R_a, R_b):
        return None, "transformations_differ"
    t_a = T_a[:3, 3]
    t_b = T_b[:3, 3]
    off = diag_r * (t_b - t_a) / vs_a
    if not np.allclose(off, np.round(off)):
        return None, "non_integer_offset"
    return tuple(int(x) for x in np.round(off)), None


def _compute_union_grid(
    shape_a, shape_b, offset
) -> "tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]":
    """Compute union grid dimensions and positions for two crops with a voxel offset.

    Returns (union_shape, pos_a, pos_b).  Identity when offset is (0,0,0) and
    shapes match.
    """
    lo = tuple(min(0, offset[i]) for i in range(3))
    hi = tuple(max(shape_a[i], offset[i] + shape_b[i]) for i in range(3))
    union_shape = tuple(hi[i] - lo[i] for i in range(3))
    pos_a = tuple(-lo[i] for i in range(3))
    pos_b = tuple(offset[i] - lo[i] for i in range(3))
    return union_shape, pos_a, pos_b


def _embed_in_union_grid(
    vol: np.ndarray, position: "tuple[int, int, int]", union_shape: "tuple[int, int, int]"
) -> np.ndarray:
    """Place a bool volume at position inside a zero-filled union_shape array."""
    result = np.zeros(union_shape, dtype=bool)
    sx, sy, sz = vol.shape[0], vol.shape[1], vol.shape[2]
    px, py, pz = position
    result[px : px + sx, py : py + sy, pz : pz + sz] = vol.astype(bool)
    return result


_XOR_DTYPE = np.int16


def _compute_aligned_pair(
    row_a: np.ndarray, row_b: np.ndarray, alignment: "dict | None"
) -> "tuple[np.ndarray, np.ndarray]":
    """Reshape mmap rows and optionally embed both in the union grid.

    row_a / row_b have shape (x, y, z, 1) (mmap rows from the crop arrays).
    alignment is None for direct comparison (equal shapes) or a dict with
    keys pos_a, pos_b and union_shape for header-aligned comparison.
    """
    vol_a = np.asarray(row_a).reshape(row_a.shape[:3]) != 0
    vol_b = np.asarray(row_b).reshape(row_b.shape[:3]) != 0
    if alignment is not None:
        vol_a = _embed_in_union_grid(vol_a, alignment["pos_a"], alignment["union_shape"])
        vol_b = _embed_in_union_grid(vol_b, alignment["pos_b"], alignment["union_shape"])
    return vol_a, vol_b


def _compute_crop_xor(vol_a: np.ndarray, vol_b: np.ndarray) -> np.ndarray:
    """Return (vol_a != vol_b) as int16; 3-D, shape of the comparison grid."""
    return (vol_a != vol_b).astype(_XOR_DTYPE)


def _make_crop_xor_header(header_a: dict, lo: "tuple[int, int, int]") -> dict:
    """Build the XOR volume header shifted to the union-grid origin.

    header_a is _read_crop_header's dict (voxel_size 3 floats, referentials
    list[str], transformations list of 4x4 np arrays).  lo = -pos_a, i.e. the
    minimum corner of the union grid in set A voxel coordinates (negative or
    zero per axis).  Each transformation T is updated so that
    T_u[:3, 3] = T[:3, 3] + T[:3, :3] @ (lo * vs); rotation and last row
    are unchanged.
    """
    vs = np.asarray(header_a["voxel_size"][:3], dtype=float)
    shift_mm = np.asarray(lo, dtype=float) * vs
    transformations_out = []
    for T in header_a["transformations"]:
        T_u = T.copy()
        T_u[:3, 3] = T[:3, 3] + T[:3, :3] @ shift_mm
        transformations_out.append([float(x) for x in T_u.ravel()])
    return {
        "voxel_size": [float(v) for v in vs] + [1.0],
        "referentials": list(header_a["referentials"]),
        "transformations": transformations_out,
    }


def _save_crop_xor(xor: np.ndarray, side_dir: Path, subject: str, xor_header: "dict | None") -> Path:
    """Write a 3-D XOR array to side_dir (caller must ensure it exists).

    xor_header None  -> numpy .npy (no geometry available).
    xor_header given -> aims.Volume 4-D int16 written as <subject>_xor.nii.gz.
    Returns the written path.
    """
    if xor_header is None:
        out = side_dir / f"{subject}_xor.npy"
        np.save(out, xor)
        return out
    out = side_dir / f"{subject}_xor.nii.gz"
    vol = brainvisa.aims.Volume(xor[..., np.newaxis])
    hdr = vol.header()
    hdr["voxel_size"] = xor_header["voxel_size"]
    hdr["referentials"] = xor_header["referentials"]
    hdr["transformations"] = xor_header["transformations"]
    brainvisa.aims.write(vol, str(out))
    return out


def _compare_subject_crops(crop_a: np.ndarray, crop_b: np.ndarray) -> dict:
    """Compute per-subject voxel metrics from two aligned bool volumes.

    Returns n_a, n_b, kept, lost, gained, pct_lost, dice, shift.
    """
    n_a = int(crop_a.sum())
    n_b = int(crop_b.sum())
    kept = int((crop_a & crop_b).sum())
    lost = int((crop_a & ~crop_b).sum())
    gained = int((~crop_a & crop_b).sum())
    pct_lost = 100.0 * lost / n_a if n_a else 0.0
    dice = 2.0 * kept / (n_a + n_b) if (n_a + n_b) else 1.0
    shift = masks.wasserstein_distance(crop_a.astype(np.float64), crop_b.astype(np.float64))
    return {
        "n_a": n_a,
        "n_b": n_b,
        "kept": kept,
        "lost": lost,
        "gained": gained,
        "pct_lost": pct_lost,
        "dice": dice,
        "shift": shift,
    }


def _summarise_region(
    rows: list, shape_a, shape_b, offset: "tuple[int, int, int] | None", top_k: int, only_in_a: int, only_in_b: int
) -> dict:
    """Build the summary.json regions entry for one compared region/side."""
    pct_list = [r["pct_lost"] for r in rows]
    dice_list = [r["dice"] for r in rows]
    subjects_changed = sum(1 for r in rows if r["lost"] > 0 or r["gained"] > 0)
    subjects_emptied = sum(1 for r in rows if r["n_a"] > 0 and r["n_b"] == 0)
    top_lost = sorted(rows, key=lambda r: (-r["pct_lost"], -r["lost"], r["subject"]))[:top_k]
    return {
        "n_subjects_compared": len(rows),
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "subjects_changed": subjects_changed,
        "pct_lost_mean": round(float(np.mean(pct_list)), 4),
        "pct_lost_p95": round(float(np.percentile(pct_list, 95)), 4),
        "pct_lost_max": round(float(np.max(pct_list)), 4),
        "dice_mean": round(float(np.mean(dice_list)), 4),
        "dice_min": round(float(np.min(dice_list)), 4),
        "subjects_emptied": subjects_emptied,
        "top_lost": [
            {"subject": r["subject"], "pct_lost": round(r["pct_lost"], 6), "lost": r["lost"]} for r in top_lost
        ],
        "crop_shape_a": [int(x) for x in shape_a],
        "crop_shape_b": [int(x) for x in shape_b],
        "alignment_offset_vox": list(offset) if offset is not None else None,
    }


def _unaligned_skip_reason(
    key: str, reason: str, shape_a: "tuple[int, ...]", shape_b: "tuple[int, ...]"
) -> "str | None":
    """Return the skip reason for a region/side with no crop offset, or None to compare directly.

    Prints the comparing-directly WARNING when the crop info is absent and the shapes are equal.
    """
    if reason not in _CROP_INFO_ABSENT or shape_a != shape_b:
        return reason
    print(f"WARNING: {key}: {reason}, comparing directly (equal shapes {shape_a}).")
    return None


def _build_crop_alignment(
    shape_a: "tuple[int, ...]", shape_b: "tuple[int, ...]", offset: "tuple[int, int, int] | None"
) -> "tuple[dict | None, tuple[int, int, int] | None]":
    """Return (alignment, lo): the union-grid alignment and set A's origin in it, both None without an offset."""
    if offset is None:
        return None, None
    union_shape, pos_a, pos_b = _compute_union_grid(shape_a, shape_b, offset)
    lo: tuple[int, int, int] = tuple(-p for p in pos_a)
    return {"pos_a": pos_a, "pos_b": pos_b, "union_shape": union_shape}, lo


def _pair_subjects(subjects_a: "list[str]", subjects_b: "list[str]") -> "list[tuple[str, int, int]]":
    """Pair subjects by ID in set A order: (subject, index in A, index in B) per A entry also in B."""
    index_b = {s: j for j, s in enumerate(subjects_b)}
    return [(subj, i, index_b[subj]) for i, subj in enumerate(subjects_a) if subj in index_b]


def _save_top_lost_xors(
    side_dir: Path,
    top_lost: "list[dict]",
    pair_index: dict,
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    alignment: "dict | None",
    xor_header: "dict | None",
) -> None:
    """Write one XOR volume per top_lost subject under side_dir (created if missing)."""
    side_dir.mkdir(parents=True, exist_ok=True)
    for entry in top_lost:
        i, j = pair_index[entry["subject"]]
        vol_a, vol_b = _compute_aligned_pair(arr_a[i], arr_b[j], alignment)
        _save_crop_xor(_compute_crop_xor(vol_a, vol_b), side_dir, entry["subject"], xor_header)


def _compare_region_side(
    set_a: Path,
    set_b: Path,
    region: str,
    side: str,
    input_type: str,
    top_k: int,
    explicit: bool,
    xor_dir: "Path | None" = None,
) -> dict:
    """Compare one region/side across two crop sets.

    Returns {"key", "region", "side", "rows", "summary", "skipped"}.
    """
    key = f"{region}/{side}"

    result_a = _load_crop_set(set_a, region, side, input_type)
    result_b = _load_crop_set(set_b, region, side, input_type)

    def _skip(reason, arr_a=None, arr_b=None):
        entry: dict = {"region": region, "side": side, "reason": reason}
        if arr_a is not None and arr_b is not None:
            entry["shape_a"] = [int(x) for x in arr_a.shape[1:4]]
            entry["shape_b"] = [int(x) for x in arr_b.shape[1:4]]
        return {"key": key, "region": region, "side": side, "rows": [], "summary": None, "skipped": entry}

    if result_a is None and result_b is None:
        if explicit:
            return _skip("missing_in_a_and_b")
        return {"key": key, "region": region, "side": side, "rows": [], "summary": None, "skipped": None}
    if result_a is None:
        return _skip("missing_in_a")
    if result_b is None:
        return _skip("missing_in_b")

    arr_a, subjects_a = result_a
    arr_b, subjects_b = result_b

    if arr_a.shape[0] != len(subjects_a) or arr_b.shape[0] != len(subjects_b):
        return _skip("subject_csv_mismatch", arr_a, arr_b)

    shape_a = arr_a.shape[1:4]
    shape_b = arr_b.shape[1:4]

    header_a = _read_crop_header(set_a, region, side)
    header_b = _read_crop_header(set_b, region, side)
    offset, reason = _compute_crop_offset(header_a, header_b)

    if offset is None:
        skip_reason = _unaligned_skip_reason(key, reason, shape_a, shape_b)
        if skip_reason is not None:
            return _skip(skip_reason, arr_a, arr_b)

    alignment, lo = _build_crop_alignment(shape_a, shape_b, offset)

    pairs = _pair_subjects(subjects_a, subjects_b)
    n_common = len(pairs)
    only_in_a = len(subjects_a) - n_common
    only_in_b = len(subjects_b) - n_common

    if n_common == 0:
        return _skip("no_common_subjects", arr_a, arr_b)

    pair_index: dict = {}
    rows: list[dict] = []
    for subj, i, j in pairs:
        pair_index[subj] = (i, j)
        vol_a, vol_b = _compute_aligned_pair(arr_a[i], arr_b[j], alignment)
        metrics = _compare_subject_crops(vol_a, vol_b)
        metrics["subject"] = subj
        rows.append(metrics)

    summary = _summarise_region(rows, shape_a, shape_b, offset, top_k, only_in_a, only_in_b)

    if xor_dir is not None and summary["top_lost"]:
        xor_header = None if offset is None else _make_crop_xor_header(header_a, lo)
        _save_top_lost_xors(
            xor_dir / region / side, summary["top_lost"], pair_index, arr_a, arr_b, alignment, xor_header
        )

    return {"key": key, "region": region, "side": side, "rows": rows, "summary": summary, "skipped": None}


# --------------------------------------------------------------------------- #
# Crops comparison helpers
# --------------------------------------------------------------------------- #


def _list_crop_regions(set_a: Path, set_b: Path, regions: "list[str] | None") -> list:
    """Return the region list to compare.

    When regions is None, returns the sorted intersection of region directories
    present in both sets.  Otherwise returns the given list unchanged.
    """
    if regions is not None:
        return list(regions)
    dirs_a = {p.name for p in set_a.iterdir() if p.is_dir()}
    dirs_b = {p.name for p in set_b.iterdir() if p.is_dir()}
    return sorted(dirs_a & dirs_b)
