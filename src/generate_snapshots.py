#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate visualization snapshots for the Champollion pipeline.

Creates four types of visualizations:
1. Sulcal graph mesh from Morphologist output
2. Region coverage map showing which regions have embeddings
3. Cortical tiles mask overlay (one snapshot per hemisphere)
4. UMAP scatter plot — new subject projected onto UKB40 reference

Requires the BrainVISA/Anatomist environment (runs in headless mode).
"""

import argparse
import glob
import json
import os
import os.path as osp
import sys

import numpy as np


def find_sulcal_graphs(morphologist_dir):
    """Find sulcal graph files (.arg) in the Morphologist output directory.

    Args:
        morphologist_dir: Path to derivatives/morphologist-6.0/ directory

    Returns:
        List of paths to .arg files
    """
    pattern = osp.join(morphologist_dir, "**", "*.arg")
    graphs = glob.glob(pattern, recursive=True)
    # Filter to get only the main sulcal graphs (L and R hemispheres)
    sulcal_graphs = [g for g in graphs if "sulci" in g.lower() or "folds" in g.lower()]
    return sulcal_graphs


def find_embeddings(embeddings_dir):
    """Find embedding CSV files and extract region names.

    Args:
        embeddings_dir: Path to embeddings output directory

    Returns:
        Set of region names that have embeddings
    """
    csv_files = glob.glob(osp.join(embeddings_dir, "*.csv"))
    regions = set()
    for f in csv_files:
        # Extract region from filename like "F.C.M._left--model_embeddings.csv"
        basename = osp.basename(f)
        if "_embeddings.csv" in basename:
            region_part = basename.replace("_embeddings.csv", "")
            # Region name is before the hemisphere indicator
            regions.add(region_part)
    return regions


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
    # Walk up from the graph to find the analysis root.
    # Morphologist < 6 uses "default_analysis";
    # Morphologist 6.0 uses acquisition index "0".
    parts = graph_path.replace("\\", "/").split("/")
    idx = None
    for anchor in ("default_analysis", "0"):
        try:
            candidate = parts.index(anchor)
        except ValueError:
            continue
        # Guard: "0" must sit under "default_acquisition"
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

    # Determine hemisphere from the graph filename
    fname = osp.basename(graph_path).lower()
    if fname.startswith("r") or "_r" in fname or "right" in fname:
        pattern = osp.join(mesh_dir, "*Rwhite.gii")
    else:
        pattern = osp.join(mesh_dir, "*Lwhite.gii")

    matches = glob.glob(pattern)
    return matches[0] if matches else None


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
        view_quaternion = (0.5, 0.5, 0.5, 0.5)  # left side view

    a = ana.Anatomist()

    # Load the graph
    graph = a.loadObject(graph_path)

    # Create window and add graph
    win = a.createWindow("3D")
    win.addObjects(graph)

    # Optionally overlay a semi-transparent white mesh
    if mesh_path and osp.exists(mesh_path):
        mesh = a.loadObject(mesh_path)
        mesh.setMaterial(diffuse=[0.8, 0.8, 0.8, 0.37])
        win.addObjects(mesh)

    # Set up the view
    a.execute("WindowConfig", windows=[win], cursor_visibility=0)
    win.camera(view_quaternion=view_quaternion)
    win.focusView()

    image = win.snapshotImage(size[0], size[1])
    image.save(output_path)
    print(f"  Saved: {output_path}")

    return output_path


def _parse_embedding_regions(embeddings_dir):
    """Parse region names from embedding CSV filenames.

    Filenames follow the pattern: {region}_{model}_embeddings.csv
    e.g. FCLp-subsc-FCLa-INSULA_left_name17--43--58--232_embeddings.csv
    The region is everything before the last _model_ part before _embeddings.csv.

    Args:
        embeddings_dir: Path to embeddings output directory

    Returns:
        Dict mapping region names to 1.0, or empty dict if none found.
    """
    csv_files = glob.glob(osp.join(embeddings_dir, "*_embeddings.csv"))
    if not csv_files:
        # Also try recursive search
        csv_files = glob.glob(
            osp.join(embeddings_dir, "**", "*_embeddings.csv"),
            recursive=True
        )
    if not csv_files:
        print("  No embedding files found")
        return {}

    regions = set()
    for f in csv_files:
        basename = osp.basename(f)
        # Remove _embeddings.csv suffix
        name = basename.replace("_embeddings.csv", "")
        # The model part is the last _-separated token (e.g. name17--43--58--232)
        # The region is everything before it
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            regions.add(parts[0])
        else:
            regions.add(name)

    print(f"  Found {len(regions)} region(s) from {len(csv_files)} CSV file(s)")
    return {r: 1.0 for r in regions}


def generate_coverage_map(embeddings_dir, output_path, size=(800, 600)):
    """Generate a coverage map showing which regions have embeddings.

    Tries Anatomist glassbrain first, falls back to matplotlib bar chart.

    Args:
        embeddings_dir: Path to embeddings output directory
        output_path: Path to save the coverage map image
        size: Tuple of (width, height)
    """
    coverage_data = _parse_embedding_regions(embeddings_dir)
    if not coverage_data:
        print("  No regions found, skipping coverage map")
        return None

    # Try glassbrain visualization first
    try:
        import anatomist.headless as ana
        from deep_folding.visualization.champo_glassbrain import glassbrain

        a = ana.Anatomist()

        glassbrain(
            regions_csv=coverage_data,
            filenames=output_path,
            sizes=[size],
            palette="Blue-Green-Red-Yellow",
            bounds=[0, 1],
        )
        print(f"  Saved glassbrain coverage map: {output_path}")
        return output_path

    except Exception as e:
        print(f"  Glassbrain unavailable ({e}), using matplotlib fallback")

    # Matplotlib fallback: produce a .png bar chart of covered regions
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        regions = sorted(coverage_data.keys())
        fig, ax = plt.subplots(
            figsize=(size[0] / 100, max(size[1] / 100, len(regions) * 0.35))
        )
        ax.barh(range(len(regions)), [1] * len(regions), color="#4CAF50")
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels(regions, fontsize=8)
        ax.set_xlabel("Coverage")
        ax.set_title(
            f"Embedding Coverage: {len(regions)} region(s)",
            fontsize=12
        )
        ax.set_xlim(0, 1.2)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"  Saved matplotlib coverage map: {output_path}")
        return output_path

    except Exception as e2:
        print(f"  Matplotlib also failed ({e2}), skipping coverage map")
        return None


def generate_tiles_snapshot(crops_dir, output_path, size=(800, 600)):
    """Generate snapshots of cortical tiles masks using Anatomist.

    Loads all {L|R}mask_skeleton.nii.gz from each region in crops_dir,
    overlays them in an Anatomist 3D window, and takes one snapshot per
    hemisphere (left and right).

    Falls back to nibabel + matplotlib if Anatomist is unavailable.

    Args:
        crops_dir: Path to crops/2mm/ directory containing region folders
        output_path: Base path for snapshot images (suffixed with
            _left.png / _right.png)
        size: Tuple of (width, height)

    Returns:
        List of generated snapshot file paths
    """
    basename = osp.splitext(output_path)[0]
    ext = osp.splitext(output_path)[1] or ".png"

    hemispheres = [
        ("left", "L", (0.5, 0.5, 0.5, 0.5)),
        ("right", "R", (0.5, -0.5, -0.5, 0.5)),
    ]

    # Collect mask files per hemisphere
    masks_by_hemi = {}
    for hemi_name, prefix, _ in hemispheres:
        pattern = osp.join(
            crops_dir, "*", "mask",
            f"{prefix}mask_skeleton.nii.gz"
        )
        masks_by_hemi[hemi_name] = glob.glob(pattern)

    total = sum(len(v) for v in masks_by_hemi.values())
    if total == 0:
        print("  No mask files found in crops directory")
        return []

    for hemi_name, count in masks_by_hemi.items():
        print(f"  Found {len(count)} {hemi_name} mask(s)")

    # Use nibabel + matplotlib to render orthogonal slice overlays.
    # (Anatomist 3D window produces blank images for volume masks.)
    try:
        import nibabel as nib
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        snapshots = []
        for hemi_name, prefix, _ in hemispheres:
            mask_files = masks_by_hemi[hemi_name]
            if not mask_files:
                continue

            # Load and overlay masks
            ref = nib.load(mask_files[0])
            overlay = np.zeros(ref.shape, dtype=np.float32)
            for i, mf in enumerate(mask_files, 1):
                data = nib.load(mf).get_fdata()
                overlay[data > 0] = i

            # Take middle slice along each axis
            mid = [s // 2 for s in overlay.shape]
            fig, axes = plt.subplots(1, 3, figsize=(
                size[0] / 100, size[1] / 100
            ))
            slices = [
                overlay[mid[0], :, :],
                overlay[:, mid[1], :],
                overlay[:, :, mid[2]],
            ]
            titles = ["Sagittal", "Coronal", "Axial"]
            for ax, sl, title in zip(axes, slices, titles):
                ax.imshow(
                    sl.T, origin="lower",
                    cmap="tab20", interpolation="nearest"
                )
                ax.set_title(title, fontsize=9)
                ax.axis("off")
            fig.suptitle(
                f"Cortical tiles masks — {hemi_name} "
                f"({len(mask_files)} regions)",
                fontsize=11,
            )
            plt.tight_layout()
            snap = f"{basename}_{hemi_name}{ext}"
            plt.savefig(snap, dpi=150)
            plt.close(fig)
            snapshots.append(snap)
            print(f"  Saved: {snap}")

        return snapshots

    except Exception as e:
        print(
            f"  Matplotlib rendering failed ({e}), "
            f"skipping tiles snapshot"
        )
        return []


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
        # Load pre-trained model and reference coords
        region = csv_name.split("_")[0]  # "FColl-SRh"
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

        # Load new subject embedding
        new_csv = osp.join(embeddings_dir, csv_name)
        if not osp.exists(new_csv):
            print(f"  [{hemi}] No embedding found at {new_csv}, skipping")
            continue

        df = pd.read_csv(new_csv)
        X_new = df.drop(columns=["ID"]).values.astype(np.float32)
        new_coords = model.transform(X_new)
        print(f"  [{hemi}] Projected {X_new.shape[0]} new subject(s)")

        # Plot
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization snapshots "
        "for Champollion pipeline output"
    )
    parser.add_argument(
        "--morphologist_dir",
        type=str,
        help="Path to Morphologist output directory "
        "(derivatives/morphologist-6.0/)"
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        help="Path to embeddings output directory"
    )
    parser.add_argument(
        "--cortical_tiles_dir",
        type=str,
        help="Path to cortical tiles crops directory "
        "(e.g. crops/2mm/)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save snapshot images"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=800,
        help="Snapshot width (default: 800)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=600,
        help="Snapshot height (default: 600)"
    )
    parser.add_argument(
        "--sulcal-only",
        action="store_true",
        help="Only generate sulcal graph snapshots"
    )
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help="Only generate coverage map"
    )
    parser.add_argument(
        "--tiles-only",
        action="store_true",
        help="Only generate cortical tiles snapshots"
    )
    parser.add_argument(
        "--umap-only",
        action="store_true",
        help="Only generate UMAP scatter plots"
    )
    parser.add_argument(
        "--reference_data_dir",
        type=str,
        help="Path to pre-trained UMAP models and reference coords"
    )

    args = parser.parse_args()
    print(f"Arguments: {args}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which snapshot types to run.
    # --*-only flags restrict to a single type.
    only_flags = (
        args.sulcal_only,
        args.coverage_only,
        args.tiles_only,
        args.umap_only,
    )
    run_sulcal = not any(only_flags) or args.sulcal_only
    run_coverage = not any(only_flags) or args.coverage_only
    run_tiles = not any(only_flags) or args.tiles_only
    run_umap = not any(only_flags) or args.umap_only

    size = (args.width, args.height)
    all_snapshots = []

    # Generate sulcal graph snapshots
    if run_sulcal:
        if (args.morphologist_dir
                and osp.exists(args.morphologist_dir)):
            print("\nGenerating sulcal graph snapshots...")
            graphs = find_sulcal_graphs(
                args.morphologist_dir
            )
            print(f"  Found {len(graphs)} sulcal graph(s)")

            # Quaternions for each hemisphere
            QUAT_LEFT = (0.5, 0.5, 0.5, 0.5)
            QUAT_RIGHT = (0.5, -0.5, -0.5, 0.5)

            for graph_path in graphs[:2]:
                # Detect hemisphere from filename
                # Morphologist names files L{subject}.arg / R{subject}.arg
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

                out = osp.join(
                    args.output_dir,
                    f"sulcal_graph_{hemi}.png"
                )
                try:
                    snap = generate_sulcal_graph_snapshot(
                        graph_path, out, size,
                        view_quaternion=quat,
                        mesh_path=white_mesh,
                    )
                    all_snapshots.append(snap)
                except Exception as e:
                    print(
                        f"  Error processing "
                        f"{graph_path}: {e}"
                    )
        elif args.morphologist_dir:
            print(
                f"Morphologist directory not found: "
                f"{args.morphologist_dir}"
            )

    # Generate cortical tiles mask snapshots
    if run_tiles:
        if (args.cortical_tiles_dir
                and osp.exists(args.cortical_tiles_dir)):
            print("\nGenerating cortical tiles snapshots...")
            out = osp.join(
                args.output_dir, "tiles_masks.png"
            )
            try:
                snaps = generate_tiles_snapshot(
                    args.cortical_tiles_dir, out, size
                )
                all_snapshots.extend(snaps)
            except Exception as e:
                print(
                    f"  Error generating tiles snapshot: "
                    f"{e}"
                )
        elif args.cortical_tiles_dir:
            print(
                f"Cortical tiles directory not found: "
                f"{args.cortical_tiles_dir}"
            )

    # Generate coverage map
    if run_coverage:
        if (args.embeddings_dir
                and osp.exists(args.embeddings_dir)):
            print("\nGenerating region coverage map...")
            out = osp.join(
                args.output_dir, "coverage_map.png"
            )
            try:
                result = generate_coverage_map(
                    args.embeddings_dir, out, size
                )
                if result:
                    all_snapshots.append(result)
            except Exception as e:
                print(
                    f"  Error generating coverage map: "
                    f"{e}"
                )
        elif args.embeddings_dir:
            print(
                f"Embeddings directory not found: "
                f"{args.embeddings_dir}"
            )

    # Generate UMAP scatter plots
    if run_umap:
        if (args.embeddings_dir
                and osp.exists(args.embeddings_dir)
                and args.reference_data_dir
                and osp.exists(args.reference_data_dir)):
            print("\nGenerating UMAP scatter plots...")
            out = osp.join(
                args.output_dir, "umap_collateral.png"
            )
            try:
                snaps = generate_umap_snapshot(
                    args.embeddings_dir,
                    args.reference_data_dir,
                    out, size,
                )
                all_snapshots.extend(snaps)
            except Exception as e:
                print(
                    f"  Error generating UMAP snapshot: "
                    f"{e}"
                )
        elif args.reference_data_dir and not osp.exists(
            args.reference_data_dir
        ):
            print(
                f"Reference data directory not found: "
                f"{args.reference_data_dir}"
            )

    print(f"\nGenerated {len(all_snapshots)} snapshot(s)")

    # Write manifest of generated files
    manifest_path = osp.join(args.output_dir, "snapshots_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"snapshots": all_snapshots}, f, indent=2)
    print(f"Manifest saved to: {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
