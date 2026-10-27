"""Anatomist viewer of the most changed masks (compare masks/cortical_tiles --visualisation)."""

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import champollion_utils.logger  # noqa: F401 - installs Logger.trace/output/fail

from compare_sulcal import masks

logger = logging.getLogger(__name__)

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
        arr_a = masks.load_mask_vol(masks_a[name])
        arr_b = masks.load_mask_vol(masks_b[name])
        masks.save_xor_vol(masks_a[name], arr_a, arr_b, str(xor_root / name))


def _launch_viewer(entries: list[dict]) -> int:
    """Run the Anatomist viewer subprocess; return 1 (stderr note) if it exits non-zero."""
    rc = subprocess.run([sys.executable, "-c", _VIEWER_SCRIPT, json.dumps(entries)], check=False).returncode
    if rc != 0:
        logger.error(f"Anatomist viewer exited with return code {rc}.")
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
        logger.trace("No changed masks to visualise.")
        return 0
    if not _has_usable_display():
        logger.warning(
            "Skipping --visualisation: no usable display (DISPLAY/WAYLAND_DISPLAY unset or QT_QPA_PLATFORM=offscreen)."
        )
        return 0

    logger.trace(f"Opening Anatomist for the {len(names)} most-changed mask(s)...")
    if xor_dir is not None:
        return _launch_viewer(_make_viewer_entries(names, masks_a, masks_b, Path(xor_dir)))
    with tempfile.TemporaryDirectory(prefix="compare_xor_") as tmp:
        xor_root = Path(tmp)
        _save_temp_xor_vols(names, masks_a, masks_b, xor_root)
        return _launch_viewer(_make_viewer_entries(names, masks_a, masks_b, xor_root))
