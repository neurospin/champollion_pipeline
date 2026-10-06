#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified comparison script for sulcal data.

Subcommands
-----------
  masks           Compare two sets of sulcal NIfTI masks (side/sulcus.nii.gz).
  cortical_tiles  Compare two cortical_tiles outputs (region/mask/<pattern>).
  databases       Compare sulcal labeling between two graph annotation campaigns.
  crops           Compare two cortical_tiles crops/2mm .npy crop sets per subject (see below).

Mask metrics (masks / cortical_tiles)
--------------------------------------
  wasserstein  – approximate 3-D Earth Mover's Distance via axis-marginal
                 projections (in voxels; for 2 mm masks, 1 vox ≈ 2 mm).
  diff         – voxel-level change counts: changed / added / removed.
  both         – compute and report both metrics.

Usage
-----
    python compare.py masks --set_a /path/to/mask/canonical_25/2.0 \\
        --set_b /path/to/mask/corrected/2.0 --metric both

    python compare.py cortical_tiles \\
        --set_a /path/to/run_a/crops/2mm \\
        --set_b /path/to/run_b/crops/2mm \\
        --pattern "*mask_skeleton.nii.gz" --metric diff

    python compare.py databases \\
        --labeled_subjects_dir /neurospin/.../manually_labeled/pclean/all \\
        --path_to_graph_a t1mri/t1/default_analysis/folds/3.3/base2018_manual \\
        --path_to_graph_b t1mri/t1/default_analysis/folds/3.3/base2018b_manual \\
        --label_a base2018 --label_b base2018b

Crops comparison (crops)
------------------------
  Compares two cortical_tiles crops/2mm directories subject by subject, per
  region and side, on the .npy crop sets (mask/{side}{input_type}.npy plus
  mask/{side}{input_type}_subject.csv); subjects are paired by ID.

  Prerequisite: PyAIMS (soma.aims). For each region and side, crops reads only
  the header of mask/{side}mask_cropped.nii.gz (no voxel data) to align the two
  crop grids. The header transformation maps AIMS storage order (identical to
  .npy order, so no axis flip) to a referential; with a common referential,
  equal voxel size vs and equal diagonal +/-1 rotations R, set B is placed at
  offset = R⁻¹(t_b − t_a)/vs voxels from set A, and both crops are compared on
  their union grid.

  Options:
    --set_a, --set_b  crops/2mm directories to compare (required).
    --output          output directory (default: crops_comparison).
    --input_type      crop file stem: skeleton, label, extremities, distmap or
                      distbottom (default: skeleton).
    --regions         region names (default: region dirs present in both sets).
    --side            L, R or both (default: both).
    --top_k           length of each top_lost list (default: 5).
    --njobs           joblib workers over region/side pairs (default: 1).
    --xor_dir         optional directory for per-subject XOR files (default: none).

  Outputs, written into --output:
    per_subject.csv  one row per region, side and common subject. Columns:
                     region, side, subject; n_a, n_b (voxels in A, in B);
                     kept, lost, gained (in both, A only, B only);
                     pct_lost (100 * lost / n_a); dice (Dice of A and B);
                     shift (Wasserstein distance in voxels, inf when one
                     side is empty).
    summary.json     top-level keys: set_a, set_b, input_type, regions, skipped.
                     regions maps "<region>/<side>" to: n_subjects_compared,
                     only_in_a, only_in_b, subjects_changed, pct_lost_mean,
                     pct_lost_p95, pct_lost_max, dice_mean, dice_min,
                     subjects_emptied, top_lost, crop_shape_a, crop_shape_b,
                     alignment_offset_vox (null = compared without header
                     check).
                     skipped is a list of entries with keys region, side,
                     reason and, when both crops loaded, shape_a, shape_b.

  XOR files, written into --xor_dir when given:
    one file per top_lost subject of each compared region/side, at
    <xor_dir>/<region>/<side>/<subject>_xor.nii.gz (PyAIMS, when
    alignment_offset_vox is a list: set A voxel size and referentials,
    transformation moved to the union-grid origin) or <subject>_xor.npy
    (numpy, when alignment_offset_vox is null); 1 = voxel differs, 0 = same,
    on the grid the subject was compared on.

  Skip reasons (summary.json skipped entries):
    missing_in_a            .npy or _subject.csv absent in set A only.
    missing_in_b            .npy or _subject.csv absent in set B only.
    missing_in_a_and_b      absent in both sets, for an explicit --regions name.
    subject_csv_mismatch    _subject.csv row count differs from the .npy length.
    no_mask_cropped         mask_cropped.nii.gz missing or unreadable, shapes differ.
    no_transformation       header has no transformation, shapes differ.
    referential_differs     the two headers share no referential.
    voxel_size_differs      voxel sizes differ.
    non_axis_aligned        rotation is not diagonal +/-1.
    transformations_differ  rotations differ between A and B.
    non_integer_offset      offset is not a whole number of voxels.
    no_common_subjects      no subject ID present in both sets.
  no_mask_cropped and no_transformation with equal shapes are compared
  directly (WARNING on stdout, alignment_offset_vox null) instead of skipped.

  Example:
    python compare.py crops \\
        --set_a /path/to/run_a/crops/2mm \\
        --set_b /path/to/run_b/crops/2mm \\
        --output crops_comparison --side both --njobs 8
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from os.path import abspath, dirname, join
from pathlib import Path

import numpy as np
from champollion_utils.script_builder import ScriptBuilder
from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance as _wasserstein_1d

try:
    from soma import aims
except ImportError as _aims_import_error:
    aims = None
    _AIMS_IMPORT_ERROR: "ImportError | None" = _aims_import_error
else:
    _AIMS_IMPORT_ERROR = None

ONE_SIDE_EMPTY_BUCKET = "one_side_empty"
VIEWER_TOP_K = 5

# Anatomist viewer, run in a fresh Python subprocess (clean QApplication).
# Reads the viewer entries (name, path_a, path_b, path_xor) as JSON from sys.argv[1].
_VIEWER_SCRIPT = """\
import json, sys
import anatomist.direct.api as ana
from soma.qt_gui.qt_backend import Qt

VIEWS = ("Axial", "Sagittal", "Coronal")
entries = json.loads(sys.argv[1])
a = ana.Anatomist()
alive = []
for i, e in enumerate(entries, 1):
    print(f"[{i}/{len(entries)}] {e['name']}: set_a grey, set_b violet, XOR red")
    va, vb, vx = (a.loadObject(e[k]) for k in ("path_a", "path_b", "path_xor"))
    va.setPalette("B-W LINEAR")
    vb.setPalette("VIOLET-lfusion")
    vx.setPalette("RED TEMPERATURE")
    fusion = a.fusionObjects([va, vb, vx], method="Fusion2DMethod")
    block = a.createWindowsBlock(3)
    windows = [a.createWindow(view, block=block) for view in VIEWS]
    for w in windows:
        w.addObjects([fusion])
    a.linkWindows(windows)
    alive.extend([va, vb, vx, fusion, block, *windows])
qt_app = Qt.QApplication.instance()
if qt_app is None:
    sys.exit("ERROR: no Qt application after creating Anatomist; cannot run the viewer.")
print("Anatomist ready. Close all windows to exit.")
qt_app.exec_()
"""

_AIMS_UNAVAILABLE_MESSAGE = (
    "ERROR: PyAIMS (soma.aims) is not available in this environment; "
    "the masks, cortical_tiles, databases and crops subcommands require it."
)


# --------------------------------------------------------------------------- #
# Shared NIfTI mask helpers
# --------------------------------------------------------------------------- #


def load_mask_vol(path: str) -> np.ndarray:
    """Read a NIfTI mask with PyAIMS and return a 3-D float64 array (x, y, z).

    Raw integer counts are preserved as-is for use as Wasserstein weights.
    """
    vol = aims.read(path)
    return np.asarray(vol, dtype=np.float64).squeeze()


def voxel_diff(a: np.ndarray, b: np.ndarray) -> dict:
    """Count voxels that changed between two mask volumes.

    Returns:
      changed  – total voxels where a[i] != b[i]
      added    – voxels that went from 0 in a to nonzero in b
      removed  – voxels that went from nonzero in a to 0 in b
    """
    a_nz = a != 0
    b_nz = b != 0
    return {
        "changed": int(np.sum(a != b)),
        "added": int(np.sum(~a_nz & b_nz)),
        "removed": int(np.sum(a_nz & ~b_nz)),
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


def _compute_top_changed_masks(scores: dict[str, float], top_k: int) -> list[str]:
    """Return up to top_k mask names with a positive score, highest first (inf first)."""
    changed = [(name, score) for name, score in scores.items() if score > 0]
    return [name for name, _score in sorted(changed, key=lambda kv: -kv[1])[:top_k]]


def _has_usable_display() -> bool:
    """True when a display is set and Qt is not forced to the offscreen platform."""
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return False
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _make_viewer_entries(names: list, masks_a: dict, masks_b: dict, xor_root: Path) -> list[dict]:
    """Build the viewer entries, in rank order, with XOR files under xor_root."""
    return [
        {"name": name, "path_a": masks_a[name], "path_b": masks_b[name], "path_xor": str(xor_root / name)}
        for name in names
    ]


def _save_temp_xor_vols(names: list, masks_a: dict, masks_b: dict, xor_root: Path) -> None:
    """Write one XOR volume per mask under xor_root, mirroring the mask names."""
    for name in names:
        arr_a = load_mask_vol(masks_a[name])
        arr_b = load_mask_vol(masks_b[name])
        save_xor_vol(masks_a[name], arr_a, arr_b, str(xor_root / name))


def _launch_viewer(entries: list[dict]) -> int:
    """Run the Anatomist viewer subprocess; return 1 (stderr note) if it exits non-zero."""
    rc = subprocess.run([sys.executable, "-c", _VIEWER_SCRIPT, json.dumps(entries)], check=False).returncode
    if rc != 0:
        print(f"ERROR: Anatomist viewer exited with return code {rc}.", file=sys.stderr)
        return 1
    return 0


def visualise_mask_diffs(
    scores: dict[str, float],
    masks_a: dict,
    masks_b: dict,
    xor_dir: "str | Path | None" = None,
    top_k: int = VIEWER_TOP_K,
) -> int:
    """Open an interactive Anatomist viewer on the most-changed mask pairs.

    Shows the 5 most-changed masks by default (top_k), ranked by score:
    changed voxel count, or Wasserstein distance (inf, one side empty, first).
    Each mask is shown as a fusion of set_a (grey), set_b (violet) and their
    XOR (red) in linked Axial, Sagittal and Coronal windows. XOR files come
    from xor_dir when given, otherwise from save_xor_vol outputs in a
    temporary directory removed afterwards.

    Prerequisites: Anatomist and a usable display; without a display the
    viewer is skipped with a note on stderr. The viewer runs in a fresh Python
    subprocess so Anatomist gets a clean QApplication.

    Args:
        scores: mask name -> ranking score (0 = unchanged, not shown).
        masks_a: mask name -> set_a NIfTI path.
        masks_b: mask name -> set_b NIfTI path.
        xor_dir: directory holding already-written XOR files, or None.
        top_k: maximum number of masks to show.

    Returns:
        0 when the viewer ran, nothing changed or it was skipped; 1 when the
        viewer exited non-zero. Ranking costs O(n log n) in len(scores).
    """
    names = _compute_top_changed_masks(scores, top_k)
    if not names:
        print("No changed masks to visualise.")
        return 0
    if not _has_usable_display():
        print(
            "Skipping --visualisation: no usable display (DISPLAY/WAYLAND_DISPLAY unset or QT_QPA_PLATFORM=offscreen).",
            file=sys.stderr,
        )
        return 0

    print(f"\nOpening Anatomist for the {len(names)} most-changed mask(s)...")
    if xor_dir is not None:
        return _launch_viewer(_make_viewer_entries(names, masks_a, masks_b, Path(xor_dir)))
    with tempfile.TemporaryDirectory(prefix="compare_xor_") as tmp:
        xor_root = Path(tmp)
        _save_temp_xor_vols(names, masks_a, masks_b, xor_root)
        return _launch_viewer(_make_viewer_entries(names, masks_a, masks_b, xor_root))


def save_xor_vol(ref_path: str, a: np.ndarray, b: np.ndarray, out_path: str) -> None:
    """Write a NIfTI XOR image (1 = voxel differs, 0 = same) to out_path.

    Voxel size and transformation are copied from ref_path so the output
    sits in the same space as the input masks.
    """
    ref_vol = aims.read(ref_path)
    xor = (a != b).astype(np.int16)
    if xor.ndim == 3:
        xor = xor[..., np.newaxis]
    out_vol = aims.Volume(xor)
    out_vol.copyHeaderFrom(ref_vol.header())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    aims.write(out_vol, out_path)


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
    finder = aims.Finder()
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
    vol = aims.Volume(xor[..., np.newaxis])
    hdr = vol.header()
    hdr["voxel_size"] = xor_header["voxel_size"]
    hdr["referentials"] = xor_header["referentials"]
    hdr["transformations"] = xor_header["transformations"]
    aims.write(vol, str(out))
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
    shift = wasserstein_distance(crop_a.astype(np.float64), crop_b.astype(np.float64))
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
        if reason in _CROP_INFO_ABSENT:
            if shape_a != shape_b:
                return _skip(reason, arr_a, arr_b)
            print(f"WARNING: {key}: {reason}, comparing directly (equal shapes {shape_a}).")
        else:
            return _skip(reason, arr_a, arr_b)

    # Build alignment
    if offset is not None:
        union_shape, pos_a, pos_b = _compute_union_grid(shape_a, shape_b, offset)
        alignment: "dict | None" = {"pos_a": pos_a, "pos_b": pos_b, "union_shape": union_shape}
        lo: "tuple[int, int, int]" = tuple(-p for p in pos_a)
    else:
        union_shape = shape_a
        pos_a = pos_b = None
        alignment = None

    # Pair subjects by ID, preserving set A order
    index_b = {s: j for j, s in enumerate(subjects_b)}
    n_common = 0
    for s in subjects_a:
        if s in index_b:
            n_common += 1

    only_in_a = len(subjects_a) - n_common
    only_in_b = len(subjects_b) - n_common

    if n_common == 0:
        return _skip("no_common_subjects", arr_a, arr_b)

    pair_index: dict = {}
    rows: list[dict] = []
    for i, subj in enumerate(subjects_a):
        if subj not in index_b:
            continue
        j = index_b[subj]
        pair_index[subj] = (i, j)
        vol_a, vol_b = _compute_aligned_pair(arr_a[i], arr_b[j], alignment)
        metrics = _compare_subject_crops(vol_a, vol_b)
        metrics["subject"] = subj
        rows.append(metrics)

    summary = _summarise_region(rows, shape_a, shape_b, offset, top_k, only_in_a, only_in_b)

    if xor_dir is not None and summary["top_lost"]:
        side_dir = xor_dir / region / side
        side_dir.mkdir(parents=True, exist_ok=True)
        xor_header = None if offset is None else _make_crop_xor_header(header_a, lo)
        for entry in summary["top_lost"]:
            i, j = pair_index[entry["subject"]]
            vol_a, vol_b = _compute_aligned_pair(arr_a[i], arr_b[j], alignment)
            _save_crop_xor(_compute_crop_xor(vol_a, vol_b), side_dir, entry["subject"], xor_header)

    return {"key": key, "region": region, "side": side, "rows": rows, "summary": summary, "skipped": None}


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


# --------------------------------------------------------------------------- #
# Database-mode module-level workers (must be picklable for joblib)
# --------------------------------------------------------------------------- #


def _get_subject_voxel_counts(sub, brainvisa_dir):
    """Worker: load one graph and return (subject_name, {sulcus: voxel_count})."""
    import glob as _glob
    import sys as _sys

    if brainvisa_dir not in _sys.path:
        _sys.path.insert(0, brainvisa_dir)
    from soma import aims  # noqa: PLC0415

    matches = _glob.glob(join(sub["dir"], sub["graph_file"]))
    if not matches:
        return sub["subject"], None

    graph = aims.read(matches[0])
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
    import glob as _g
    import sys as _s

    if brainvisa_dir not in _s.path:
        _s.path.insert(0, brainvisa_dir)
    from soma import aims  # noqa: PLC0415

    stats = {}
    for path in sorted(_g.glob(join(mask_dir, "*/*.nii.gz"))):
        rel = os.path.relpath(path, mask_dir)
        arr = np.asarray(aims.read(path))
        stats[rel] = (int(arr.max()), int(arr.sum()), int(np.count_nonzero(arr)))
    return stats


# --------------------------------------------------------------------------- #
# Unified script class
# --------------------------------------------------------------------------- #


class Compare(ScriptBuilder):
    """Unified comparison tool for sulcal masks, cortical tiles, and databases."""

    def __init__(self):
        super().__init__(
            script_name="compare",
            description="Compare sulcal masks, cortical_tiles outputs, or graph databases.",
        )
        subparsers = self.parser.add_subparsers(
            dest="mode", required=True, metavar="MODE", description="Choose a comparison mode."
        )

        # ── Shared parent for mask-comparison arguments ────────────────────
        mask_parent = argparse.ArgumentParser(add_help=False)
        mask_parent.add_argument("--set_a", required=True, help="Path to the first set directory.")
        mask_parent.add_argument("--set_b", required=True, help="Path to the second set directory.")
        mask_parent.add_argument(
            "--output", default="comparison_report.json", help="Output JSON file path. Default: comparison_report.json."
        )
        mask_parent.add_argument(
            "--metric",
            choices=["wasserstein", "diff", "both"],
            default="diff",
            help="Comparison metric. Default: diff.",
        )
        mask_parent.add_argument(
            "--bucket_step", type=float, default=1.0, help="Bucket width for the summary table (in voxels). Default: 1."
        )
        mask_parent.add_argument(
            "--xor_dir",
            default=None,
            help="Optional output directory for per-mask XOR NIfTI images "
            "(1 = voxel differs, 0 = same). Files are written with the "
            "same relative path as the input masks.",
        )
        mask_parent.add_argument(
            "--visualisation",
            action="store_true",
            default=False,
            help="Open an interactive Anatomist viewer on the 5 most changed masks, ranked by "
            "changed voxels (--metric diff/both) or by Wasserstein distance (--metric "
            "wasserstein). Each mask is shown as a fusion of set_a (grey), set_b (violet) "
            "and their XOR (red) in linked Axial, Sagittal and Coronal windows. XOR files "
            "come from --xor_dir when given, otherwise from a temporary directory removed "
            "afterwards. Requires Anatomist and a usable display (DISPLAY or "
            "WAYLAND_DISPLAY set, QT_QPA_PLATFORM not offscreen); skipped otherwise. Exits 1 "
            "if the viewer fails.",
        )

        # ── masks subcommand ───────────────────────────────────────────────
        subparsers.add_parser(
            "masks",
            parents=[mask_parent],
            help="Compare two sets of sulcal NIfTI masks (side/sulcus.nii.gz).",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # ── cortical_tiles subcommand ──────────────────────────────────────
        tiles_p = subparsers.add_parser(
            "cortical_tiles",
            parents=[mask_parent],
            help="Compare two cortical_tiles outputs (crops/2mm directory).",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        tiles_p.add_argument(
            "--pattern",
            default="*mask_skeleton.nii.gz",
            help="Glob pattern for mask files inside each region's mask/ folder.",
        )

        # ── crops subcommand ──────────────────────────────────────────────
        crops_p = subparsers.add_parser(
            "crops",
            help="Compare two cortical_tiles crops/2mm directories (.npy crop sets).",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        crops_p.add_argument("--set_a", required=True, help="Path to the first crops/2mm directory.")
        crops_p.add_argument("--set_b", required=True, help="Path to the second crops/2mm directory.")
        crops_p.add_argument(
            "--output", default="crops_comparison", help="Output directory. Default: crops_comparison."
        )
        crops_p.add_argument(
            "--input_type",
            choices=_CROPS_INPUT_TYPES,
            default="skeleton",
            help="Reads mask/{side}{input_type}.npy and _subject.csv. Default: skeleton.",
        )
        crops_p.add_argument(
            "--xor_dir",
            default=None,
            help="Optional directory for per-subject XOR files of the top_lost subjects.",
        )
        crops_p.add_argument(
            "--regions",
            nargs="+",
            default=None,
            help="Region names to compare. Default: intersection of region dirs in both sets.",
        )
        crops_p.add_argument(
            "--side",
            choices=["L", "R", "both"],
            default="both",
            help="Hemisphere side(s) to compare. Default: both.",
        )
        crops_p.add_argument("--top_k", type=int, default=5, help="Length of top_lost list. Default: 5.")
        crops_p.add_argument("--njobs", type=int, default=1, help="Joblib workers over region/side pairs. Default: 1.")

        # ── databases subcommand ───────────────────────────────────────────
        db_p = subparsers.add_parser(
            "databases",
            help="Compare sulcal labeling between two graph annotation campaigns.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        db_p.add_argument(
            "--labeled_subjects_dir", required=True, help="Root directory containing subject subdirectories."
        )
        db_p.add_argument(
            "--path_to_graph_a",
            required=True,
            help="Relative sub-path for campaign A (e.g. t1mri/t1/default_analysis/folds/3.3/base2018_manual).",
        )
        db_p.add_argument("--path_to_graph_b", required=True, help="Relative sub-path for campaign B.")
        db_p.add_argument("--label_a", default="A", help="Name for campaign A.")
        db_p.add_argument("--label_b", default="B", help="Name for campaign B.")
        db_p.add_argument("--side", default="both", help="Hemisphere side: L, R, or both.")
        db_p.add_argument("--masks_a", default=None, help="Mask directory for campaign A. Optional.")
        db_p.add_argument("--masks_b", default=None, help="Mask directory for campaign B. Optional.")
        db_p.add_argument("--output", default="db_comparison.csv", help="Output CSV file path.")
        db_p.add_argument("--njobs", type=int, default=None, help="Parallel workers. Default: cpu_count - 2 (max 22).")

    # ---------------------------------------------------------------------- #
    # Dispatch
    # ---------------------------------------------------------------------- #

    def run(self) -> int:
        if aims is None:
            print(f"{_AIMS_UNAVAILABLE_MESSAGE} ({_AIMS_IMPORT_ERROR})", file=sys.stderr)
            return 1
        if self.args.mode == "masks":
            return self._run_masks()
        if self.args.mode == "cortical_tiles":
            return self._run_cortical_tiles()
        if self.args.mode == "databases":
            return self._run_databases()
        if self.args.mode == "crops":
            return self._run_crops()
        print(f"ERROR: unknown mode '{self.args.mode}'")
        return 1

    # ---------------------------------------------------------------------- #
    # Crops comparison mode
    # ---------------------------------------------------------------------- #

    def _run_crops(self) -> int:
        """Run the crops comparison subcommand."""
        set_a = Path(self.args.set_a)
        set_b = Path(self.args.set_b)
        if not set_a.is_dir():
            print(f"ERROR: --set_a is not a directory: {set_a}", file=sys.stderr)
            return 1
        if not set_b.is_dir():
            print(f"ERROR: --set_b is not a directory: {set_b}", file=sys.stderr)
            return 1

        out_dir = Path(self.args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        sides = ["L", "R"] if self.args.side == "both" else [self.args.side]
        regions = _list_crop_regions(set_a, set_b, self.args.regions)
        explicit = self.args.regions is not None
        xor_dir = Path(self.args.xor_dir) if self.args.xor_dir else None

        pairs = [(region, side) for region in regions for side in sides]

        results = Parallel(n_jobs=self.args.njobs)(
            delayed(_compare_region_side)(
                set_a, set_b, region, side, self.args.input_type, self.args.top_k, explicit, xor_dir=xor_dir
            )
            for region, side in pairs
        )

        all_rows: list[dict] = []
        regions_summary: dict = {}
        skipped_list: list[dict] = []

        for result in results:
            key = result["key"]
            region_name = result["region"]
            side_name = result["side"]
            if result["skipped"] is not None:
                skipped_list.append(result["skipped"])
                print(f"SKIP  {key}: {result['skipped']['reason']}")
            if result["summary"] is not None:
                regions_summary[key] = result["summary"]
                s = result["summary"]
                print(
                    f"  {key}: n={s['n_subjects_compared']} "
                    f"changed={s['subjects_changed']} "
                    f"pct_lost_mean={s['pct_lost_mean']:.4f} "
                    f"dice_min={s['dice_min']:.4f}"
                )
            for row in result["rows"]:
                all_rows.append(
                    {
                        "region": region_name,
                        "side": side_name,
                        "subject": row["subject"],
                        "n_a": row["n_a"],
                        "n_b": row["n_b"],
                        "kept": row["kept"],
                        "lost": row["lost"],
                        "gained": row["gained"],
                        "pct_lost": round(row["pct_lost"], 6),
                        "dice": round(row["dice"], 6),
                        "shift": "inf" if math.isinf(row["shift"]) else round(row["shift"], 6),
                    }
                )

        csv_path = out_dir / "per_subject.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CROPS_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(all_rows)

        summary = {
            "set_a": str(set_a.resolve()),
            "set_b": str(set_b.resolve()),
            "input_type": self.args.input_type,
            "regions": regions_summary,
            "skipped": skipped_list,
        }
        json_path = out_dir / "summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, allow_nan=False)

        print(f"Wrote {csv_path} and {json_path}")
        if xor_dir is not None:
            print(f"XOR files written under: {xor_dir}")
        return 0

    # ---------------------------------------------------------------------- #
    # Mask comparison modes
    # ---------------------------------------------------------------------- #

    def _find_masks(self, directory: Path, glob_pattern: str) -> dict:
        return {str(p.relative_to(directory)): str(p.resolve()) for p in sorted(directory.glob(glob_pattern))}

    def _run_masks(self) -> int:
        return self._compare_nifti_masks("*/*.nii.gz")

    def _run_cortical_tiles(self) -> int:
        return self._compare_nifti_masks(f"*/mask/{self.args.pattern}")

    def _compare_nifti_masks(self, glob_pattern: str) -> int:
        dir_a = Path(self.args.set_a)
        dir_b = Path(self.args.set_b)

        if not self.validate_paths([str(dir_a), str(dir_b)]):
            return 1

        masks_a = self._find_masks(dir_a, glob_pattern)
        masks_b = self._find_masks(dir_b, glob_pattern)

        common = sorted(set(masks_a) & set(masks_b))
        only_a = sorted(set(masks_a) - set(masks_b))
        only_b = sorted(set(masks_b) - set(masks_a))

        print(f"Masks in set_a:      {len(masks_a)}")
        print(f"Masks in set_b:      {len(masks_b)}")
        print(f"Common:              {len(common)}")
        print(f"Only in set_a:       {len(only_a)}")
        print(f"Only in set_b:       {len(only_b)}")

        use_wass = self.args.metric in ("wasserstein", "both")
        use_diff = self.args.metric in ("diff", "both")
        step = self.args.bucket_step

        wass_buckets: dict[str, list] = {}
        diff_buckets: dict[str, list] = {}
        distances: dict[str, float] = {}
        diffs: dict[str, dict] = {}
        skipped_shape_mismatch: dict[str, dict[str, list[int]]] = {}

        for name in common:
            arr_a = load_mask_vol(masks_a[name])
            arr_b = load_mask_vol(masks_b[name])

            if arr_a.shape != arr_b.shape:
                print(f"WARNING: shape mismatch for {name}: {arr_a.shape} vs {arr_b.shape}, skipping.")
                skipped_shape_mismatch[name] = {
                    "shape_a": [int(n) for n in arr_a.shape],
                    "shape_b": [int(n) for n in arr_b.shape],
                }
                continue

            if use_wass:
                dist = wasserstein_distance(arr_a, arr_b)
                is_one_side_empty = math.isinf(dist)
                distances[name] = dist if is_one_side_empty else round(dist, 3)
                bucket = ONE_SIDE_EMPTY_BUCKET if is_one_side_empty else bucket_label(dist, step)
                wass_buckets.setdefault(bucket, []).append(name)

            if use_diff:
                d = voxel_diff(arr_a, arr_b)
                diffs[name] = d
                diff_buckets.setdefault(bucket_label(d["changed"], step), []).append(name)

            if self.args.xor_dir:
                out_xor = str(Path(self.args.xor_dir) / name)
                save_xor_vol(masks_a[name], arr_a, arr_b, out_xor)

        total_compared = len(common) - len(skipped_shape_mismatch)
        print(f"Compared:            {total_compared}")
        print(f"Skipped (shape mismatch): {len(skipped_shape_mismatch)}")

        report = {
            "mode": self.args.mode,
            "set_a": str(dir_a),
            "set_b": str(dir_b),
            "metric": self.args.metric,
            "summary": {
                "total_common": len(common),
                "total_compared": total_compared,
                "skipped_shape_mismatch": skipped_shape_mismatch,
                "only_in_set_a": only_a,
                "only_in_set_b": only_b,
            },
        }
        if self.args.mode == "cortical_tiles":
            report["mask_pattern"] = self.args.pattern

        if use_wass:
            report["wasserstein_by_bucket"] = sort_buckets(wass_buckets)
            report["wasserstein_per_mask"] = {
                name: _normalize_distance_for_json(d) for name, d in sorted(distances.items(), key=lambda kv: -kv[1])
            }

        if use_diff:
            report["diff_by_bucket"] = sort_buckets(diff_buckets)
            report["diffs_per_mask"] = dict(sorted(diffs.items(), key=lambda kv: -kv[1]["changed"]))

        if self.args.xor_dir:
            report["xor_dir"] = str(self.args.xor_dir)

        out_path = Path(self.args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, allow_nan=False)
        print(f"\nReport written to: {out_path}")
        if self.args.xor_dir:
            print(f"XOR images written to: {self.args.xor_dir}")

        if use_wass and wass_buckets:
            print(f"\nWasserstein buckets (step={step}):")
            for lbl, names in sort_buckets(wass_buckets).items():
                print(f"  {lbl:15s}  {len(names):3d} mask(s)")

        if use_diff and diff_buckets:
            print(f"\nDiff buckets (step={step}):")
            for lbl, names in sort_buckets(diff_buckets).items():
                print(f"  {lbl:15s}  {len(names):3d} mask(s)")
            unchanged = sum(1 for d in diffs.values() if d["changed"] == 0)
            print(f"Unchanged masks: {unchanged}/{len(diffs)}")

        if not self.args.visualisation:
            return 0
        scores = {name: d["changed"] for name, d in diffs.items()} if use_diff else distances
        return visualise_mask_diffs(scores, masks_a, masks_b, xor_dir=self.args.xor_dir)

    # ---------------------------------------------------------------------- #
    # Database comparison mode
    # ---------------------------------------------------------------------- #

    def _load_database(self, path_to_graph, sides, subjects_dir, njobs, brainvisa_dir) -> dict:
        from cortical_tiles.brainvisa.utils.subjects import get_all_subjects_as_dictionary
        from joblib import Parallel, delayed

        all_data: dict = {}
        for side in sides:
            pattern = "%(subject)s/" + path_to_graph + "/%(side)s%(subject)s*.arg"
            subjects = get_all_subjects_as_dictionary([subjects_dir], [pattern], side)
            print(f"    [{side}] {len(subjects)} subjects found, loading with {njobs} worker(s)…")

            results = Parallel(n_jobs=njobs, prefer="processes")(
                delayed(_get_subject_voxel_counts)(sub, brainvisa_dir) for sub in subjects
            )
            n_ok = 0
            for sub_name, counts in results:
                if counts is None:
                    print(f"    [{side}] WARNING: no graph for {sub_name}")
                    continue
                all_data.setdefault(sub_name, {}).update(counts)
                n_ok += 1
            print(f"    [{side}] Done ({n_ok}/{len(subjects)}).")
        return all_data

    def _run_databases(self) -> int:
        from joblib import cpu_count

        brainvisa_dir = abspath(
            join(dirname(__file__), "..", "external", "cortical_tiles", "cortical_tiles", "brainvisa")
        )
        if brainvisa_dir not in sys.path:
            sys.path.insert(0, brainvisa_dir)

        sides = ["L", "R"] if self.args.side == "both" else [self.args.side]
        njobs = self.args.njobs or max(1, min(22, cpu_count() - 2))
        la, lb = self.args.label_a, self.args.label_b

        print(f"\nLoading campaign A ({la}): {self.args.path_to_graph_a}")
        data_a = self._load_database(
            self.args.path_to_graph_a, sides, self.args.labeled_subjects_dir, njobs, brainvisa_dir
        )

        print(f"\nLoading campaign B ({lb}): {self.args.path_to_graph_b}")
        data_b = self._load_database(
            self.args.path_to_graph_b, sides, self.args.labeled_subjects_dir, njobs, brainvisa_dir
        )

        subs_a, subs_b = set(data_a), set(data_b)
        print(f"\nSubjects in {la} only:  {len(subs_a - subs_b)}")
        print(f"Subjects in {lb} only:  {len(subs_b - subs_a)}")
        print(f"Subjects in both:        {len(subs_a & subs_b)}")

        all_sulci = sorted({s for d in data_a.values() for s in d} | {s for d in data_b.values() for s in d})

        rows = []
        for sulcus in all_sulci:
            counts_a = [data_a[s][sulcus] for s in sorted(subs_a) if sulcus in data_a[s]]
            counts_b = [data_b[s][sulcus] for s in sorted(subs_b) if sulcus in data_b[s]]
            n_a, n_b = len(counts_a), len(counts_b)
            pct_a = 100.0 * n_a / len(subs_a) if subs_a else 0
            pct_b = 100.0 * n_b / len(subs_b) if subs_b else 0
            vpsa = np.mean(counts_a) if counts_a else 0.0
            vpsb = np.mean(counts_b) if counts_b else 0.0
            rows.append(
                {
                    "sulcus": sulcus,
                    f"N_{la}": n_a,
                    f"N_{lb}": n_b,
                    f"pct_{la}": round(pct_a, 1),
                    f"pct_{lb}": round(pct_b, 1),
                    f"vox_per_subject_{la}": round(vpsa, 1),
                    f"vox_per_subject_{lb}": round(vpsb, 1),
                    "vox_ratio_B_over_A": round(vpsb / vpsa, 3) if vpsa > 0 else None,
                }
            )

        # Optional mask stats
        mask_a_stats, mask_b_stats = {}, {}
        if self.args.masks_a and os.path.isdir(self.args.masks_a):
            print(f"\nLoading mask stats from {la}: {self.args.masks_a}")
            mask_a_stats = _mask_stats(abspath(self.args.masks_a), brainvisa_dir)
        if self.args.masks_b and os.path.isdir(self.args.masks_b):
            print(f"Loading mask stats from {lb}: {self.args.masks_b}")
            mask_b_stats = _mask_stats(abspath(self.args.masks_b), brainvisa_dir)

        if mask_a_stats or mask_b_stats:
            all_mask_keys = sorted(set(mask_a_stats) | set(mask_b_stats))
            mask_lookup: dict = {}
            for key in all_mask_keys:
                sulcus_name = os.path.basename(key).replace(".nii.gz", "")
                sa, sb = mask_a_stats.get(key), mask_b_stats.get(key)
                n_sa = len(subs_a) or 1
                n_sb = len(subs_b) or 1
                mask_lookup.setdefault(sulcus_name, {}).update(
                    {
                        f"mask_max_{la}": sa[0] if sa else None,
                        f"mask_max_{lb}": sb[0] if sb else None,
                        f"mask_sum_per_sub_{la}": round(sa[1] / n_sa, 1) if sa else None,
                        f"mask_sum_per_sub_{lb}": round(sb[1] / n_sb, 1) if sb else None,
                    }
                )
            for row in rows:
                row.update(mask_lookup.get(row["sulcus"], {}))

        output_path = abspath(self.args.output)
        os.makedirs(dirname(output_path) or ".", exist_ok=True)
        if rows:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        print(f"\nSulci with biggest voxel-density difference ({lb}/{la}):\n")
        print(
            f"  {'Sulcus':<45}  {f'N({la})':>7}  {f'N({lb})':>7}  "
            f"{f'vox/sub({la})':>12}  {f'vox/sub({lb})':>12}  {'ratio':>6}"
        )
        print("  " + "-" * 100)
        sortable = [r for r in rows if r.get("vox_ratio_B_over_A") is not None]
        for row in sorted(sortable, key=lambda r: abs(r["vox_ratio_B_over_A"] - 1.0), reverse=True)[:30]:
            print(
                f"  {row['sulcus']:<45}  "
                f"{row[f'N_{la}']:>7}  {row[f'N_{lb}']:>7}  "
                f"{row[f'vox_per_subject_{la}']:>12.1f}  "
                f"{row[f'vox_per_subject_{lb}']:>12.1f}  "
                f"{row['vox_ratio_B_over_A']:>6.3f}"
            )

        print(f"\nCSV written to: {output_path}")
        return 0


def main():
    script = Compare()
    return script.main()


if __name__ == "__main__":
    sys.exit(main())
