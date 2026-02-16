#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate visualization snapshots for the Champollion pipeline.

Creates three types of visualizations:
1. Sulcal graph mesh from Morphologist output
2. Cortical tiles regions on ICBM152 template (one snapshot per hemisphere)
3. UMAP scatter plot — new subject projected onto UKB40 reference

Requires the BrainVISA/Anatomist environment (runs in headless mode).
"""

import glob
import json
import os
import os.path as osp
import sys

import numpy as np

from champollion_utils.script_builder import ScriptBuilder


# Fallback path for ICBM152 meshes (used when BrainVISA resource lookup fails)
ICBM_MESH_DIR_FALLBACK = (
    '/neurospin/dico/data/bv_databases/templates/'
    'morphologist_templates/icbm152/'
    'mni_icbm152_nlin_asym_09c/t1mri/default_acquisition/'
    'default_analysis/segmentation/mesh'
)


def find_sulcal_graphs(morphologist_dir):
    """Find sulcal graph files (.arg) in the Morphologist output directory.

    Args:
        morphologist_dir: Path to derivatives/morphologist-6.0/ directory

    Returns:
        List of paths to .arg files
    """
    pattern = osp.join(morphologist_dir, "**", "*.arg")
    graphs = glob.glob(pattern, recursive=True)
    sulcal_graphs = [g for g in graphs if "sulci" in g.lower() or "folds" in g.lower()]
    return sulcal_graphs


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
        if (anchor == "0"
                and (candidate < 1
                     or parts[candidate - 1]
                     != "default_acquisition")):
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
    if fname.startswith("r") or "_r" in fname or "right" in fname:
        pattern = osp.join(mesh_dir, "*Rwhite.gii")
    else:
        pattern = osp.join(mesh_dir, "*Lwhite.gii")

    matches = glob.glob(pattern)
    return matches[0] if matches else None


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


def generate_sulcal_graph_snapshot(graph_path, output_path, size=(800, 600),
                                   view_quaternion=None, mesh_path=None):
    """Generate a snapshot of a sulcal graph.

    Args:
        graph_path: Path to .arg sulcal graph file
        output_path: Path to save the snapshot image
        size: Tuple of (width, height)
        view_quaternion: Camera orientation as (x, y, z, w). Defaults to
            left side view.
        mesh_path: Optional path to a white matter mesh (.gii) to render
            as a semi-transparent background surface.
    """
    import anatomist.headless as ana

    if view_quaternion is None:
        view_quaternion = (0.5, 0.5, 0.5, 0.5)

    a = ana.Anatomist()

    graph = a.loadObject(graph_path)

    win = a.createWindow("3D")
    win.addObjects(graph)

    if mesh_path and osp.exists(mesh_path):
        mesh = a.loadObject(mesh_path)
        mesh.setMaterial(diffuse=[0.8, 0.8, 0.8, 0.37])
        win.addObjects(mesh)

    a.execute("WindowConfig", windows=[win], cursor_visibility=0)
    win.camera(view_quaternion=view_quaternion)
    win.focusView()

    image = win.snapshotImage(size[0], size[1])
    image.save(output_path)
    print(f"  Saved: {output_path}")

    return output_path


def generate_tiles_snapshot(crops_dir, output_path, size=(800, 600), level=1,
                            champollion_data_root=None):
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
            If None, uses deep_folding config default.

    Returns:
        List of generated snapshot file paths
    """
    import anatomist.headless as ana
    from soma import aims
    from deep_folding import config

    a = ana.Anatomist()

    completed = find_completed_regions(crops_dir)
    total = sum(len(v) for v in completed.values())
    if total == 0:
        print("  No completed regions found in crops directory")
        return []

    if champollion_data_root:
        root = champollion_data_root
    else:
        root = config.config().get_champollion_data_root_dir()
    regions_graph_dir = f"{root}/mask/2mm/regions/meshes"

    nom = aims.read(aims.carto.Paths.findResourceFile(
        'nomenclature/hierarchy/champollion_v1.hie'))
    anom = a.toAObject(nom)

    icbm_mesh_dir = aims.carto.Paths.findResourceFile(
        'disco_templates_hbp_morpho/icbm152/mni_icbm152_nlin_asym_09c/'
        't1mri/default_acquisition/default_analysis/segmentation/mesh',
        'disco')
    if icbm_mesh_dir is None:
        icbm_mesh_dir = ICBM_MESH_DIR_FALLBACK

    basename = osp.splitext(output_path)[0]
    ext = osp.splitext(output_path)[1] or ".png"

    hemispheres = [
        ("left", "L", (0.5, 0.5, 0.5, 0.5),
         "mni_icbm152_nlin_asym_09c_Lhemi.gii"),
        ("right", "R", (0.5, -0.5, -0.5, 0.5),
         "mni_icbm152_nlin_asym_09c_Rhemi.gii"),
    ]

    snapshots = []
    group = 0
    for hemi_name, side, quat, mesh_file in hemispheres:
        region_names = completed[hemi_name]
        if not region_names:
            continue

        print(f"  {hemi_name}: {len(region_names)} region(s)")

        graph_path = osp.join(
            regions_graph_dir, f"{side}regions_model_{level}.arg"
        )
        if not osp.exists(graph_path):
            print(f"  Region graph not found: {graph_path}")
            continue

        reg_graph = a.loadObject(graph_path)
        reg_graph.applyBuiltinReferential()

        mesh_path = osp.join(icbm_mesh_dir, mesh_file)
        if not osp.exists(mesh_path):
            print(f"  ICBM mesh not found: {mesh_path}")
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
        a.execute("SelectByNomenclature", nomenclature=anom,
                  names=sel_names, group=group)
        a.execute("SelectByNomenclature", nomenclature=anom,
                  names=sel_names, modifiers="toggle", group=group)

        win.camera(view_quaternion=quat)
        win.focusView()

        snap_path = f"{basename}_{hemi_name}{ext}"
        image = win.snapshotImage(size[0], size[1])
        image.save(snap_path)
        snapshots.append(snap_path)
        print(f"  Saved: {snap_path}")

        group += 1

    return snapshots


COLLATERAL_FILES = {
    "left": "FColl-SRh_left_name06-43-43--210_embeddings.csv",
    "right": "FColl-SRh_right_name06-56-15--113_embeddings.csv",
}


def generate_umap_snapshot(embeddings_dir, reference_data_dir, output_path,
                           size=(800, 600)):
    """Generate UMAP scatter plots for the collateral sulcus region.

    Projects the pipeline's new subject(s) onto a pre-trained UMAP fitted
    on UKBioBank40 reference embeddings. Produces one plot per hemisphere.

    Args:
        embeddings_dir: Path to pipeline embeddings (stage 5 output)
        reference_data_dir: Path to pre-trained UMAP models and coords
        output_path: Base path for output images (suffixed with _left/_right)
        size: Tuple of (width, height)

    Returns:
        List of generated snapshot file paths
    """
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    basename = osp.splitext(output_path)[0]
    ext = osp.splitext(output_path)[1] or ".png"
    snapshots = []

    for hemi, csv_name in COLLATERAL_FILES.items():
        region = csv_name.split("_")[0]
        model_path = osp.join(
            reference_data_dir, f"umap_{region}_{hemi}.pkl"
        )
        coords_path = osp.join(
            reference_data_dir, f"umap_{region}_{hemi}_coords.npy"
        )
        if not osp.exists(model_path) or not osp.exists(coords_path):
            print(f"  UMAP artifacts not found for {hemi}, skipping")
            continue

        model = joblib.load(model_path)
        ref_coords = np.load(coords_path)
        print(f"  [{hemi}] Loaded {ref_coords.shape[0]} reference points")

        new_csv = osp.join(embeddings_dir, csv_name)
        if not osp.exists(new_csv):
            print(f"  [{hemi}] No embedding found at {new_csv}, skipping")
            continue

        df = pd.read_csv(new_csv)
        X_new = df.drop(columns=["ID"]).values.astype(np.float32)
        new_coords = model.transform(X_new)
        print(f"  [{hemi}] Projected {X_new.shape[0]} new subject(s)")

        fig, ax = plt.subplots(
            figsize=(size[0] / 100, size[1] / 100)
        )
        ax.scatter(
            ref_coords[:, 0], ref_coords[:, 1],
            s=1, c="#4a90d9", alpha=0.08,
            label=f"UKB40 (n={ref_coords.shape[0]:,})", rasterized=True,
        )
        ax.scatter(
            new_coords[:, 0], new_coords[:, 1],
            s=80, c="#e74c3c", edgecolors="white", linewidths=0.8,
            zorder=5, label="Your subject",
        )
        ax.set_title(
            f"Collateral sulcus \u2014 {hemi}", fontsize=12
        )
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        ax.tick_params(labelsize=8)
        plt.tight_layout()

        snap = f"{basename}_{hemi}{ext}"
        plt.savefig(snap, dpi=150)
        plt.close(fig)
        snapshots.append(snap)
        print(f"  Saved: {snap}")

    return snapshots


class GenerateSnapshots(ScriptBuilder):
    """Script for generating visualization snapshots."""

    def __init__(self):
        super().__init__(
            script_name="generate_snapshots",
            description="Generate visualization snapshots for Champollion pipeline output",
        )
        (self
         .add_optional_argument("--morphologist_dir", "Path to Morphologist output directory")
         .add_optional_argument("--embeddings_dir", "Path to embeddings output directory")
         .add_optional_argument("--cortical_tiles_dir", "Path to cortical tiles crops directory")
         .add_argument("--output_dir", type=str, required=True, help="Directory to save snapshot images")
         .add_optional_argument("--width", "Snapshot width", default=800, type_=int)
         .add_optional_argument("--height", "Snapshot height", default=600, type_=int)
         .add_flag("--sulcal-only", "Only generate sulcal graph snapshots")
         .add_flag("--tiles-only", "Only generate cortical tiles snapshots")
         .add_flag("--umap-only", "Only generate UMAP scatter plots")
         .add_optional_argument(
             "--reference_data_dir",
             "Path to pre-trained UMAP models and reference coords")
         .add_optional_argument(
             "--tiles_level",
             "Region threshold level (0-3)",
             default=1, type_=int)
         .add_optional_argument(
             "--champollion_data_root",
             "Override path to Champollion data directory"))

    def run(self) -> int:
        """Run all requested snapshot generation steps."""
        os.makedirs(self.args.output_dir, exist_ok=True)

        only_flags = (
            self.args.sulcal_only,
            self.args.tiles_only,
            self.args.umap_only,
        )
        run_sulcal = not any(only_flags) or self.args.sulcal_only
        run_tiles = not any(only_flags) or self.args.tiles_only
        run_umap = not any(only_flags) or self.args.umap_only

        size = (self.args.width, self.args.height)
        all_snapshots = []

        if run_sulcal:
            all_snapshots.extend(self._run_sulcal(size))

        if run_tiles:
            all_snapshots.extend(self._run_tiles(size))

        if run_umap:
            all_snapshots.extend(self._run_umap(size))

        print(f"\nGenerated {len(all_snapshots)} snapshot(s)")

        manifest_path = osp.join(self.args.output_dir, "snapshots_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump({"snapshots": all_snapshots}, f, indent=2)
        print(f"Manifest saved to: {manifest_path}")

        return 0

    def _run_sulcal(self, size):
        """Generate sulcal graph snapshots."""
        snapshots = []
        if self.args.morphologist_dir and osp.exists(self.args.morphologist_dir):
            print("\nGenerating sulcal graph snapshots...")
            graphs = find_sulcal_graphs(self.args.morphologist_dir)
            print(f"  Found {len(graphs)} sulcal graph(s)")

            QUAT_LEFT = (0.5, 0.5, 0.5, 0.5)
            QUAT_RIGHT = (0.5, -0.5, -0.5, 0.5)

            for graph_path in graphs[:2]:
                fname = osp.basename(graph_path).lower()
                if fname.startswith("r") or "_r" in fname or "right" in fname:
                    hemi = "right"
                    quat = QUAT_RIGHT
                else:
                    hemi = "left"
                    quat = QUAT_LEFT

                white_mesh = find_white_mesh(graph_path)
                if white_mesh:
                    print(f"  White mesh: {white_mesh}")

                out = osp.join(self.args.output_dir, f"sulcal_graph_{hemi}.png")
                try:
                    snap = generate_sulcal_graph_snapshot(
                        graph_path, out, size,
                        view_quaternion=quat,
                        mesh_path=white_mesh,
                    )
                    snapshots.append(snap)
                except Exception as e:
                    print(f"  Error processing {graph_path}: {e}")
        elif self.args.morphologist_dir:
            print(f"Morphologist directory not found: {self.args.morphologist_dir}")
        return snapshots

    def _run_tiles(self, size):
        """Generate cortical tiles snapshots."""
        snapshots = []
        if self.args.cortical_tiles_dir and osp.exists(self.args.cortical_tiles_dir):
            print("\nGenerating cortical tiles snapshots...")
            out = osp.join(self.args.output_dir, "tiles_masks.png")
            try:
                snaps = generate_tiles_snapshot(
                    self.args.cortical_tiles_dir, out, size,
                    level=self.args.tiles_level,
                    champollion_data_root=self.args.champollion_data_root,
                )
                snapshots.extend(snaps)
            except ImportError as e:
                print(f"  Anatomist not available for tiles snapshot: {e}")
            except Exception as e:
                print(f"  Error generating tiles snapshot: {e}")
        elif self.args.cortical_tiles_dir:
            print(f"Cortical tiles directory not found: {self.args.cortical_tiles_dir}")
        return snapshots

    def _run_umap(self, size):
        """Generate UMAP scatter plots."""
        snapshots = []
        if (self.args.embeddings_dir
                and osp.exists(self.args.embeddings_dir)
                and self.args.reference_data_dir
                and osp.exists(self.args.reference_data_dir)):
            print("\nGenerating UMAP scatter plots...")
            out = osp.join(self.args.output_dir, "umap_collateral.png")
            try:
                snaps = generate_umap_snapshot(
                    self.args.embeddings_dir,
                    self.args.reference_data_dir,
                    out, size,
                )
                snapshots.extend(snaps)
            except Exception as e:
                print(f"  Error generating UMAP snapshot: {e}")
        elif self.args.reference_data_dir and not osp.exists(self.args.reference_data_dir):
            print(f"Reference data directory not found: {self.args.reference_data_dir}")
        return snapshots


def main():
    script = GenerateSnapshots()
    return script.build().print_args().run()


if __name__ == "__main__":
    sys.exit(main())
