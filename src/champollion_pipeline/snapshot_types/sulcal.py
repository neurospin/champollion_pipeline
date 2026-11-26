"""Sulcal graph snapshots: subject, graph and white-mesh discovery and the Anatomist render."""

import glob
import logging
import os
import os.path as osp
from collections.abc import Mapping

import champollion_utils.logger  # noqa: F401 - installs Logger.trace/output/fail (LOG 22, OUTPUT 25, FAIL 45)

from champollion_pipeline.file_indexer.pipeline_checks import FileIndexStageIndexer, natural_key
from champollion_pipeline.snapshot_types import brainvisa_imports
from champollion_pipeline.utils.graph_item import GraphItem
from champollion_pipeline.utils.input_kind import resolve_subjects_dir
from champollion_pipeline.utils.subject_folders import list_morphologist_subjects

logger = logging.getLogger(__name__)

HEMISPHERES = ("left", "right")
# Maps the hemisphere attribute of a discovered graph (L or R) to the hemisphere key; not a side detection.
SIDE_NAME = {"L": "left", "R": "right"}
# A white mesh of side X is a .gii file whose name ends with one of these, {side} being L or R.
WHITE_MESH_SUFFIXES = ("{side}white.gii", "hemi-{side}_white.gii")


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
        # The subject name is a literal folder name, never a glob pattern.
        pattern = osp.join(glob.escape(morphologist_dir), "**", glob.escape(subject))
        candidates = sorted(m for m in glob.glob(pattern, recursive=True) if osp.isdir(m))
        if not candidates:
            logger.warning(f"Subject directory not found for '{subject}' under {morphologist_dir}")
            return []
        if len(candidates) > 1:
            logger.warning(f"Subject '{subject}' is ambiguous under {morphologist_dir}: {', '.join(candidates)}")
            return []
        root = candidates[0]
        logger.trace(f"Subject found at: {root}")
    graphs = glob.glob(osp.join(glob.escape(root), "**", "*.arg"), recursive=True)
    # Name tests apply below the search root only: a root named like "folds" must not admit every graph.
    relative = {g: osp.relpath(g, root).replace("\\", "/") for g in graphs}
    graphs = [g for g in graphs if "sulci" in relative[g].lower() or "folds" in relative[g].lower()]
    if acquisition:
        graphs = [g for g in graphs if f"/{acquisition}/" in f"/{relative[g]}"]
    graphs.sort()
    return graphs


def compute_session_graphs(graphs, sessions):
    """Return the graphs lying directly in the first of ``sessions`` that holds one, else ``graphs`` unchanged.

    A graph lies in a session when its parent folder is named after it
    (``folds/3.1/<session>/<side><subject>_<session>.arg``).

    Args:
        graphs: Sulcal graph paths of one hemisphere.
        sessions: Labelling session folder names, in priority order.

    Returns:
        A new list of graph paths, in the order of ``graphs``.

    Complexity: O(len(sessions) * len(graphs)).
    """
    for session in sessions:
        in_session = [graph for graph in graphs if osp.basename(osp.dirname(graph)) == session]
        if in_session:
            return in_session
    return list(graphs)


def find_white_mesh(graph_path, attributes: Mapping[str, str | None]):
    """Find the white mesh of a discovered sulcal graph, from the nearest folder above it that holds one.

    The search starts in the graph's own folder and goes up one folder at a time. At each folder, the folder and
    everything below it (links not followed) are searched for a white mesh of the graph's hemisphere. The search ends
    after the nearest folder above the graph named after the graph's subject attribute.

    Args:
        graph_path: Path to the .arg sulcal graph file.
        attributes: The ``attributes`` of the graph's discover_graphs item (hemisphere, subject).

    Returns:
        Path to the white mesh file that comes first in sorted() order in the nearest folder holding one, or None
        when none is found or when the hemisphere attribute is not L or R.

    Complexity: O(depth x entries below the subject folder) worst case.
    """
    side = attributes.get("hemisphere")
    if side not in SIDE_NAME:
        return None
    suffixes = tuple(suffix.format(side=side) for suffix in WHITE_MESH_SUFFIXES)
    own = osp.dirname(graph_path)
    folders = [own]
    while osp.basename(folders[-1]) != attributes.get("subject") and osp.dirname(folders[-1]) != folders[-1]:
        folders.append(osp.dirname(folders[-1]))
    if osp.basename(folders[-1]) != attributes.get("subject"):
        folders = [own]
    for folder in folders:
        matches = sorted(
            osp.join(root, name) for root, _, names in os.walk(folder) for name in names if name.endswith(suffixes)
        )
        if matches:
            return matches[0]
    return None


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
    logger.output(f"Saved: {output_path}")

    a.removeObjects(objects, [win])
    a.closeWindows([win])
    a.deleteObjects(objects)

    return output_path


def discover_sulcal_graphs(morphologist_dir):
    """Return the GraphItems of the StageIndexer discover_graphs operation for ``morphologist_dir``.

    Raises:
        LooseGraphError: a .arg file sits directly in a folder whose name does not start with sub-.
    """
    return FileIndexStageIndexer("").discover_graphs(morphologist_dir)


def graph_path(morphologist_dir, item: GraphItem) -> str:
    """Return the path of ``item``'s graph file: ``morphologist_dir`` joined with the item's POSIX relative path."""
    return osp.join(morphologist_dir, *item.path.split("/"))


def list_sulcal_subjects(morphologist_dir, subject=None, items=None):
    """Return ``[subject]`` when given, else the subject folder names and the subject attributes of the graphs.

    Without ``subject`` the names are those of the subject folders (the children of the folder
    ``resolve_subjects_dir`` returns, else the Morphologist folder itself) united with the subject attributes of
    ``items``, once each, in natural order. No graph at all gives an empty list.

    Args:
        morphologist_dir: Path of the Morphologist folder.
        subject: Optional subject name; returned alone with no I/O.
        items: The discover_graphs result when the caller holds it, else it is discovered now.

    Raises:
        AmbiguousSubjectsFolders: the graphs sit in several subjects folders.
        LooseGraphError: a .arg file sits directly in a folder whose name does not start with sub-.
    """
    if subject:
        return [subject]
    if items is None:
        items = discover_sulcal_graphs(morphologist_dir)
    if not items:
        return []
    folder = resolve_subjects_dir(morphologist_dir)
    folders = [osp.basename(osp.abspath(morphologist_dir))] if folder is None else list_morphologist_subjects(folder)
    names = {*folders, *(item.attributes["subject"] for item in items)}
    result = sorted(names, key=natural_key)
    assert result == sorted(set(result), key=natural_key), "list_sulcal_subjects post: unique, natural order"
    return result


def find_sulcal_graph_items_by_hemi(morphologist_dir, subject, acquisition=None, items=None):
    """Return one subject's discovered graph items as ``{"left": [...], "right": [...]}``, in discovery order.

    The subject's graphs are the items whose subject attribute equals ``subject``, whatever the folder names on their
    paths. ``acquisition`` keeps the items with that folder name on their path. An item whose hemisphere attribute is
    None goes in neither list and is reported by one WARNING naming its path.
    """
    if items is None:
        items = discover_sulcal_graphs(morphologist_dir)
    by_hemi = {hemi: [] for hemi in HEMISPHERES}
    for item in items:
        if item.attributes["subject"] != subject:
            continue
        if acquisition and f"/{acquisition}/" not in f"/{item.path}":
            continue
        side = SIDE_NAME.get(item.attributes["hemisphere"])
        if side is None:
            logger.warning(
                f"Sulcal graph {graph_path(morphologist_dir, item)} has no hemisphere attribute "
                "(hemi-L/hemi-R entity or leading L or R): left out of both hemisphere lists"
            )
            continue
        by_hemi[side].append(item)
    return by_hemi


def find_sulcal_graphs_by_hemi(morphologist_dir, subject, acquisition=None, items=None):
    """Return the lists of ``find_sulcal_graph_items_by_hemi`` as graph path strings."""
    by_hemi = find_sulcal_graph_items_by_hemi(morphologist_dir, subject, acquisition, items)
    return {hemi: [graph_path(morphologist_dir, item) for item in found] for hemi, found in by_hemi.items()}


def find_first_kept_subject_graphs(morphologist_dir, subject=None, acquisition=None, excluded=frozenset(), items=None):
    """Return the graph items by hemisphere of the first non-excluded subject that has a sulcal graph.

    Subjects are taken in ``list_sulcal_subjects`` order; when none qualifies, both hemisphere lists are empty.
    """
    if items is None:
        items = discover_sulcal_graphs(morphologist_dir)
    for name in list_sulcal_subjects(morphologist_dir, subject, items):
        if name in excluded:
            continue
        by_hemi = find_sulcal_graph_items_by_hemi(morphologist_dir, name, acquisition, items)
        if any(by_hemi.values()):
            return by_hemi
    return {hemi: [] for hemi in HEMISPHERES}


def choose_hemisphere_graphs(morphologist_dir, items, sessions, explicit):
    """Return the items of one hemisphere to render from: a labelling session's, else the labelled ones.

    A graph lies in a session when its parent folder is named like a session of ``sessions`` (first match wins). With
    no such graph and no explicit ``--labelling_session``, the items whose labelling_set attribute is not None are
    kept when there is at least one. Else every item is kept. The order of ``items`` is kept.
    """
    paths = [graph_path(morphologist_dir, item) for item in items]
    if any(osp.basename(osp.dirname(path)) in sessions for path in paths):
        kept = set(compute_session_graphs(paths, sessions))
        return [item for item, path in zip(items, paths, strict=True) if path in kept]
    if not explicit:
        labelled = [item for item in items if item.attributes["labelling_set"] is not None]
        if labelled:
            return labelled
    return list(items)
