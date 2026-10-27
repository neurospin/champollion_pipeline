"""Sulcal graph snapshots: subject, graph and white-mesh discovery and the Anatomist render."""

import glob
import os
import os.path as osp

from champollion_pipeline.snapshot_types import brainvisa_imports

HEMISPHERES = ("left", "right")


def list_acquisitions(subject_dir):
    """Return acquisition folder names that contain at least one sulcal graph.

    Scans ``<subject_dir>/t1mri/`` and returns all subdirectory names whose
    subtree contains a ``.arg`` sulcal/folds graph file.  These are the valid
    choices for ``--acquisition``.

    Args:
        subject_dir: Path to a single subject folder.

    Returns:
        Sorted list of acquisition folder names.
    """
    t1mri_dir = osp.join(subject_dir, "t1mri")
    if not osp.isdir(t1mri_dir):
        return []
    candidates = []
    for name in os.listdir(t1mri_dir):
        acq_dir = osp.join(t1mri_dir, name)
        if not osp.isdir(acq_dir):
            continue
        found = any(
            ("sulci" in g.lower() or "folds" in g.lower())
            for g in glob.glob(osp.join(acq_dir, "**", "*.arg"), recursive=True)
        )
        if found:
            candidates.append(name)
    return sorted(candidates)


def find_sulcal_graphs(morphologist_dir, subject=None, acquisition=None):
    """Find sulcal graph files (.arg) in the Morphologist output directory.

    Args:
        morphologist_dir: Path to derivatives/morphologist-6.0/ directory
        subject: Optional subject folder name to restrict the search (e.g.
            ``"sub_0001"``).  When omitted the entire directory is searched.
        acquisition: Optional acquisition folder name to restrict the search
            (e.g. ``"wk30"``).  When omitted all acquisitions are returned.

    Returns:
        List of paths to .arg files
    """
    root = osp.join(morphologist_dir, subject) if subject else morphologist_dir
    if subject and not osp.isdir(root):
        candidates = [m for m in glob.glob(osp.join(morphologist_dir, "**", subject), recursive=True) if osp.isdir(m)]
        if candidates:
            root = candidates[0]
            print(f"  Subject found at: {root}")
        else:
            print(f"  Subject directory not found for '{subject}' under {morphologist_dir}")
            return []
    pattern = osp.join(root, "**", "*.arg")
    graphs = glob.glob(pattern, recursive=True)
    graphs = [g for g in graphs if "sulci" in g.lower() or "folds" in g.lower()]
    if acquisition:
        graphs = [g for g in graphs if f"/{acquisition}/" in g.replace("\\", "/")]
    return graphs


def find_white_mesh(graph_path):
    """Find the white matter mesh corresponding to a sulcal graph.

    Navigates from the .arg graph path up to the analysis root
    (``default_analysis`` or acquisition index ``0``), then into
    ``segmentation/mesh/`` to locate the matching
    ``*_Lwhite.gii`` or ``*_Rwhite.gii`` file.

    Args:
        graph_path: Path to .arg sulcal graph file

    Returns:
        Path to the white mesh file, or None if not found.
    """
    parts = graph_path.replace("\\", "/").split("/")
    idx = None
    for anchor in ("default_analysis", "0"):
        try:
            candidate = parts.index(anchor)
        except ValueError:
            continue
        if anchor == "0" and (candidate < 1 or parts[candidate - 1] != "default_acquisition"):
            continue
        idx = candidate
        break

    if idx is None:
        return None

    analysis_dir = "/".join(parts[: idx + 1])
    mesh_dir = osp.join(analysis_dir, "segmentation", "mesh")
    if not osp.isdir(mesh_dir):
        return None

    fname = osp.basename(graph_path).lower()
    is_right = fname.startswith("r") or "_r" in fname or "right" in fname
    pattern = osp.join(mesh_dir, "*Rwhite.gii" if is_right else "*Lwhite.gii")

    matches = glob.glob(pattern)
    return matches[0] if matches else None


def generate_sulcal_graph_snapshot(  # noqa: PLR0913 - snapshot API; keyword defaults used by generate_snapshots
    graph_path, output_path, size=(800, 600), view_quaternion=None, mesh_path=None, a=None
):
    """Generate a snapshot of a sulcal graph.

    Args:
        graph_path: Path to .arg sulcal graph file
        output_path: Path to save the snapshot image
        size: Tuple of (width, height)
        view_quaternion: Camera orientation as (x, y, z, w). Defaults to
            left side view.
        mesh_path: Optional path to a white matter mesh (.gii) to render
            as a semi-transparent background surface.
        a: Existing Anatomist instance to reuse. If None a new one is created.
    """
    if view_quaternion is None:
        view_quaternion = (0.5, 0.5, 0.5, 0.5)

    if a is None:
        a = brainvisa_imports.require("anatomist.headless").Anatomist()

    objects = []
    graph = a.loadObject(graph_path)
    objects.append(graph)

    win = a.createWindow("3D")
    win.addObjects(graph)

    if mesh_path and osp.exists(mesh_path):
        mesh = a.loadObject(mesh_path)
        mesh.setMaterial(diffuse=[0.8, 0.8, 0.8, 0.37])
        win.addObjects(mesh)
        objects.append(mesh)

    a.execute("WindowConfig", windows=[win], cursor_visibility=0)
    win.camera(view_quaternion=view_quaternion)
    win.focusView()
    a.sync()

    image = win.snapshotImage(size[0], size[1])
    image.save(output_path)
    print(f"  Saved: {output_path}")

    a.removeObjects(objects, [win])
    a.closeWindows([win])
    a.deleteObjects(objects)

    return output_path


def _detect_hemi(graph_path):
    """Return 'right' or 'left' based on graph filename."""
    fname = osp.basename(graph_path).lower()
    if fname.startswith("r") or "_r" in fname or "right" in fname:
        return "right"
    return "left"


def list_sulcal_subjects(morphologist_dir, subject=None):
    """Return ``[subject]`` when given, else the sorted subdirectory names of ``morphologist_dir``."""
    if subject:
        return [subject]
    return sorted(name for name in os.listdir(morphologist_dir) if osp.isdir(osp.join(morphologist_dir, name)))


def find_sulcal_graphs_by_hemi(morphologist_dir, subject, acquisition=None):
    """Return one subject's sorted sulcal graphs grouped as ``{"left": [...], "right": [...]}``."""
    by_hemi = {hemi: [] for hemi in HEMISPHERES}
    for graph in sorted(find_sulcal_graphs(morphologist_dir, subject=subject, acquisition=acquisition)):
        by_hemi[_detect_hemi(graph)].append(graph)
    return by_hemi


def find_first_kept_subject_graphs(morphologist_dir, subject=None, acquisition=None, excluded=frozenset()):
    """Return the graphs by hemisphere of the first non-excluded subject that has a sulcal graph.

    Subjects are taken in ``list_sulcal_subjects`` order; when none qualifies,
    both hemisphere lists are empty.
    """
    for name in list_sulcal_subjects(morphologist_dir, subject):
        if name in excluded:
            continue
        by_hemi = find_sulcal_graphs_by_hemi(morphologist_dir, name, acquisition)
        if any(by_hemi.values()):
            return by_hemi
    return {hemi: [] for hemi in HEMISPHERES}
