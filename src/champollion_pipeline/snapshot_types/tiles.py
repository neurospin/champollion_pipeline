"""Cortical tiles snapshots: completed-region discovery and the Anatomist render of the region graphs."""

import logging
import os
import os.path as osp

import champollion_utils.logger  # noqa: F401 - installs Logger.trace/output/fail (LOG 22, OUTPUT 25, FAIL 45)

from champollion_pipeline.snapshot_types import brainvisa_imports

logger = logging.getLogger(__name__)

# Hint appended to the missing region graph warning: where the region graphs are looked up
REGION_GRAPH_ROOT_HINT = (
    "Pass --champollion_data_root <dir> (or set CHAMPOLLION_DATA_ROOT before launch) "
    "to a directory containing mask/2mm/regions/meshes/."
)


# Fallback path for ICBM152 meshes (used when BrainVISA resource lookup fails)
ICBM_MESH_DIR_FALLBACK = (
    "/neurospin/dico/data/bv_databases/templates/"
    "morphologist_templates/icbm152/"
    "mni_icbm152_nlin_asym_09c/t1mri/default_acquisition/"
    "default_analysis/segmentation/mesh"
)


def compute_region_graph_path(champollion_data_root, side, level):
    """Return the region graph path of one hemisphere side ("L" or "R") at a threshold level."""
    return osp.join(f"{champollion_data_root}/mask/2mm/regions/meshes", f"{side}regions_model_{level}.arg")


def find_completed_regions(crops_dir):
    """Scan the crops directory to find which regions were processed.

    Args:
        crops_dir: Path to crops/2mm/ directory

    Returns:
        dict with keys 'left' and 'right', each a list of region
        directory names that contain mask_skeleton files.
    """
    result = {"left": [], "right": []}
    if not osp.isdir(crops_dir):
        return result
    for region_name in sorted(os.listdir(crops_dir)):
        region_dir = osp.join(crops_dir, region_name)
        if not osp.isdir(region_dir):
            continue
        mask_dir = osp.join(region_dir, "mask")
        if not osp.isdir(mask_dir):
            continue
        for prefix, hemi in [("L", "left"), ("R", "right")]:
            if osp.exists(osp.join(mask_dir, f"{prefix}mask_skeleton.nii.gz")):
                result[hemi].append(region_name)
    return result


def generate_tiles_snapshot(crops_dir, output_path, size=(800, 600), level=1, champollion_data_root=None):  # noqa: PLR0915 - linear Anatomist rendering sequence; split deferred (user decision 2026-10-23)
    """Generate snapshots of cortical tiles regions using Anatomist.

    Loads region graphs from the Champollion model data, overlays
    them on ICBM152 hemisphere meshes, selects only regions that
    were actually processed (found in crops_dir), and takes one
    snapshot per hemisphere.

    Based on display_champo_regions.py from cortical_tiles.

    Args:
        crops_dir: Path to crops/2mm/ directory (used to determine
            which regions to highlight)
        output_path: Base path for snapshot images (suffixed with
            _left.png / _right.png)
        size: Tuple of (width, height)
        level: Region threshold level (0-3, default 1)
        champollion_data_root: Override path to Champollion data
            directory (containing mask/2mm/regions/meshes/).
            If None, uses cortical_tiles config default.

    Returns:
        List of generated snapshot file paths
    """
    ana = brainvisa_imports.require("anatomist.headless")
    config = brainvisa_imports.require("cortical_tiles.config")
    aims = brainvisa_imports.require("soma.aims")

    a = ana.Anatomist()

    completed = find_completed_regions(crops_dir)
    total = sum(len(v) for v in completed.values())
    if total == 0:
        logger.warning("No completed regions found in crops directory")
        return []

    root = champollion_data_root if champollion_data_root else config.config().get_champollion_data_root_dir()

    nom = aims.read(aims.carto.Paths.findResourceFile("nomenclature/hierarchy/champollion_v1.hie"))
    anom = a.toAObject(nom)

    icbm_mesh_dir = aims.carto.Paths.findResourceFile(
        "disco_templates_hbp_morpho/icbm152/mni_icbm152_nlin_asym_09c/"
        "t1mri/default_acquisition/default_analysis/segmentation/mesh",
        "disco",
    )
    if icbm_mesh_dir is None:
        icbm_mesh_dir = ICBM_MESH_DIR_FALLBACK

    basename = osp.splitext(output_path)[0]
    ext = osp.splitext(output_path)[1] or ".png"

    hemispheres = [
        ("left", "L", (0.5, 0.5, 0.5, 0.5), "mni_icbm152_nlin_asym_09c_Lhemi.gii"),
        ("right", "R", (0.5, -0.5, -0.5, 0.5), "mni_icbm152_nlin_asym_09c_Rhemi.gii"),
    ]

    snapshots = []
    group = 0
    for hemi_name, side, quat, mesh_file in hemispheres:
        region_names = completed[hemi_name]
        if not region_names:
            continue

        logger.trace(f"{hemi_name}: {len(region_names)} region(s)")

        graph_path = compute_region_graph_path(root, side, level)
        if not osp.exists(graph_path):
            logger.warning(f"Region graph not found: {graph_path}; {REGION_GRAPH_ROOT_HINT}")
            continue

        reg_graph = a.loadObject(graph_path)
        reg_graph.applyBuiltinReferential()

        mesh_path = osp.join(icbm_mesh_dir, mesh_file)
        if not osp.exists(mesh_path):
            logger.warning(f"ICBM mesh not found: {mesh_path}")
            continue

        mesh = a.loadObject(mesh_path)
        mesh.setMaterial(diffuse=[0.8, 0.8, 0.8, 0.37])
        mesh.applyBuiltinReferential()

        win = a.createWindow("3D")
        a.execute("WindowConfig", windows=[win], cursor_visibility=0)
        win.addObjects([reg_graph, mesh], add_graph_nodes=False)
        win.setReferential(reg_graph.referential)

        sel_names = " ".join(f"{r}_{hemi_name}" for r in region_names)
        a.execute("LinkWindows", windows=[win], group=group)
        a.execute("SelectByNomenclature", nomenclature=anom, names=sel_names, group=group)
        a.execute("SelectByNomenclature", nomenclature=anom, names=sel_names, modifiers="toggle", group=group)

        win.camera(view_quaternion=quat)
        win.focusView()
        a.sync()

        snap_path = f"{basename}_{hemi_name}{ext}"
        image = win.snapshotImage(size[0], size[1])
        image.save(snap_path)
        snapshots.append(snap_path)
        logger.output(f"Saved: {snap_path}")

        a.removeObjects([reg_graph, mesh], [win])
        a.closeWindows([win])
        a.deleteObjects([reg_graph, mesh])

        group += 1

    a.deleteObjects([anom])
    return snapshots
