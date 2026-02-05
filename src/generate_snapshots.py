#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate visualization snapshots for the Champollion pipeline.

Creates two types of visualizations:
1. Sulcal graph mesh from Morphologist output
2. Region coverage map showing which regions have embeddings

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


def generate_sulcal_graph_snapshot(graph_path, output_path, size=(800, 600)):
    """Generate a snapshot of a sulcal graph.

    Args:
        graph_path: Path to .arg sulcal graph file
        output_path: Path to save the snapshot image
        size: Tuple of (width, height)
    """
    import anatomist.headless as ana

    a = ana.Anatomist()

    # Load the graph
    graph = a.loadObject(graph_path)

    # Create window and add graph
    win = a.createWindow("3D")
    win.addObjects(graph)

    # Set up the view
    a.execute("WindowConfig", windows=[win], cursor_visibility=0)

    # Take snapshots from different angles
    quaternions = [
        (0.5, -0.5, -0.5, 0.5),  # right side view
        (0.5, 0.5, 0.5, 0.5),    # left side view
    ]

    basename = osp.splitext(output_path)[0]
    ext = osp.splitext(output_path)[1] or ".png"

    snapshots = []
    for i, quat in enumerate(quaternions):
        win.camera(view_quaternion=quat)
        win.focusView()

        snapshot_path = f"{basename}_{i:02d}{ext}"
        image = win.snapshotImage(size[0], size[1])
        image.save(snapshot_path)
        snapshots.append(snapshot_path)
        print(f"  Saved: {snapshot_path}")

    return snapshots


def generate_coverage_map(embeddings_dir, output_path, size=(800, 600)):
    """Generate a glassbrain-style coverage map showing which regions have data.

    Args:
        embeddings_dir: Path to embeddings output directory
        output_path: Path to save the coverage map image
        size: Tuple of (width, height)
    """
    # Find which regions have embeddings
    csv_files = glob.glob(osp.join(embeddings_dir, "*.csv"))
    if not csv_files:
        print("  No embedding files found, skipping coverage map")
        return None

    # Create a simple coverage data dict (1 for present, 0 for absent)
    # Map region names to values
    coverage_data = {}
    for f in csv_files:
        basename = osp.basename(f)
        if "_embeddings.csv" in basename:
            # Parse the region name from filename
            parts = basename.replace("_embeddings.csv", "").split("_")
            if len(parts) >= 2:
                region = parts[0]
                coverage_data[region] = 1.0

    if not coverage_data:
        print("  Could not parse region names from files")
        return None

    try:
        # Try to use the glassbrain visualization
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
        print(f"  Saved coverage map: {output_path}")
        return output_path

    except ImportError as e:
        print(f"  Warning: Could not import glassbrain module: {e}")
        print("  Generating simple coverage summary instead")

        # Fallback: create a text-based summary
        summary_path = osp.splitext(output_path)[0] + "_summary.txt"
        with open(summary_path, "w") as f:
            f.write("Region Coverage Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total regions with embeddings: {len(coverage_data)}\n\n")
            f.write("Regions:\n")
            for region in sorted(coverage_data.keys()):
                f.write(f"  - {region}\n")
        print(f"  Saved summary: {summary_path}")
        return summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization snapshots for Champollion pipeline output"
    )
    parser.add_argument(
        "--morphologist_dir",
        type=str,
        help="Path to Morphologist output directory (derivatives/morphologist-6.0/)"
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        help="Path to embeddings output directory"
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
        help="Only generate sulcal graph snapshots (skip coverage map)"
    )
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help="Only generate coverage map (skip sulcal graphs)"
    )

    args = parser.parse_args()
    print(f"Arguments: {args}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    size = (args.width, args.height)
    all_snapshots = []

    # Generate sulcal graph snapshots (unless --coverage-only)
    if not args.coverage_only:
        if args.morphologist_dir and osp.exists(args.morphologist_dir):
            print("\nGenerating sulcal graph snapshots...")
            graphs = find_sulcal_graphs(args.morphologist_dir)
            print(f"  Found {len(graphs)} sulcal graph(s)")

            for i, graph_path in enumerate(graphs[:2]):  # Limit to first 2 graphs
                output_path = osp.join(args.output_dir, f"sulcal_graph_{i:02d}.png")
                try:
                    snapshots = generate_sulcal_graph_snapshot(graph_path, output_path, size)
                    all_snapshots.extend(snapshots)
                except Exception as e:
                    print(f"  Error processing {graph_path}: {e}")
        elif args.morphologist_dir:
            print(f"Morphologist directory not found: {args.morphologist_dir}")

    # Generate coverage map (unless --sulcal-only)
    if not args.sulcal_only:
        if args.embeddings_dir and osp.exists(args.embeddings_dir):
            print("\nGenerating region coverage map...")
            output_path = osp.join(args.output_dir, "coverage_map.png")
            try:
                result = generate_coverage_map(args.embeddings_dir, output_path, size)
                if result:
                    all_snapshots.append(result)
            except Exception as e:
                print(f"  Error generating coverage map: {e}")
        elif args.embeddings_dir:
            print(f"Embeddings directory not found: {args.embeddings_dir}")

    print(f"\nGenerated {len(all_snapshots)} snapshot(s)")

    # Write manifest of generated files
    manifest_path = osp.join(args.output_dir, "snapshots_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"snapshots": all_snapshots}, f, indent=2)
    print(f"Manifest saved to: {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
