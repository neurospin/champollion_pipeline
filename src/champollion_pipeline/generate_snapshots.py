#!/usr/bin/env python3
"""
Generate visualization snapshots for the Champollion pipeline.

Creates three types of visualizations:
1. Sulcal graph mesh from Morphologist output
2. Cortical tiles regions on ICBM152 template (one snapshot per hemisphere)
3. UMAP scatter plot — new subject projected onto UKB40 reference

Requires the BrainVISA/Anatomist environment (runs in headless mode).
"""

import json
import os
import os.path as osp
import sys
from datetime import datetime

from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.derivatives_layout import compute_run_log_path
from champollion_pipeline.process_setup import init_pipeline_process
from champollion_pipeline.snapshot_types.sulcal import (
    _detect_hemi,
    find_sulcal_graphs,
    find_white_mesh,
    generate_sulcal_graph_snapshot,
    list_acquisitions,
)
from champollion_pipeline.snapshot_types.tiles import (
    ICBM_MESH_DIR_FALLBACK,
    REGION_GRAPH_ROOT_HINT,
    find_completed_regions,
    generate_tiles_snapshot,
)
from champollion_pipeline.snapshot_types.umap import discover_umap_pairs, generate_umap_snapshot

__all__ = [
    "ICBM_MESH_DIR_FALLBACK",
    "REGION_GRAPH_ROOT_HINT",
    "GenerateSnapshots",
    "discover_umap_pairs",
    "find_completed_regions",
    "find_sulcal_graphs",
    "find_white_mesh",
    "generate_sulcal_graph_snapshot",
    "generate_tiles_snapshot",
    "generate_umap_snapshot",
    "list_acquisitions",
    "main",
]


class GenerateSnapshots(ScriptBuilder):
    """Script for generating visualization snapshots."""

    def __init__(self):
        super().__init__(
            script_name="generate_snapshots",
            description="Generate visualization snapshots for Champollion pipeline output",
        )
        (
            self.add_optional_argument("--morphologist_dir", "Path to Morphologist output directory")
            .add_optional_argument(
                "--subject",
                "Subject folder name to visualize (e.g. sub_0001). When omitted the first subject found is used.",
            )
            .add_optional_argument(
                "--acquisition",
                "Acquisition folder name to use (e.g. wk30, wk40). Required when a subject has multiple segmentations.",
            )
            .add_optional_argument("--embeddings_dir", "Path to embeddings output directory")
            .add_optional_argument("--cortical_tiles_dir", "Path to cortical tiles crops directory")
            .add_argument("--output_dir", type=str, required=True, help="Directory to save snapshot images")
            .add_optional_argument("--width", "Snapshot width", default=800, type_=int)
            .add_optional_argument("--height", "Snapshot height", default=600, type_=int)
            .add_flag("--sulcal-only", "Only generate sulcal graph snapshots")
            .add_flag("--tiles-only", "Only generate cortical tiles snapshots")
            .add_flag("--umap-only", "Only generate UMAP scatter plots")
            .add_optional_argument("--reference_data_dir", "Path to pre-trained UMAP models and reference coords")
            .add_optional_argument(
                "--umap_region",
                "Comma-separated list of region names to generate UMAP plots for "
                "(e.g. FColl-SRh,S.Or.). Defaults to all regions with available models.",
            )
            .add_optional_argument("--tiles_level", "Region threshold level (0-3)", default=1, type_=int)
            .add_optional_argument("--champollion_data_root", "Override path to Champollion data directory")
        )

    def default_log_path(self) -> str:
        """Run-log path of a bare --log: <output_dir>/.log/snapshots_<stamp>.log."""
        return compute_run_log_path(self.args.output_dir, "snapshots", datetime.now().astimezone())

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
        morphologist_dir = self.args.morphologist_dir
        if not morphologist_dir or not osp.exists(morphologist_dir):
            if morphologist_dir:
                print(f"Morphologist directory not found: {morphologist_dir}")
            return snapshots

        print("\nGenerating sulcal graph snapshots...")
        subject = getattr(self.args, "subject", None)
        acquisition = getattr(self.args, "acquisition", None)

        graphs = find_sulcal_graphs(morphologist_dir, subject=subject, acquisition=acquisition)

        # Group by hemisphere — keep one graph per side
        by_hemi = {"left": [], "right": []}
        for g in graphs:
            by_hemi[_detect_hemi(g)].append(g)

        # Warn if multiple acquisitions exist and user hasn't disambiguated
        if not acquisition:
            for hemi, hemi_graphs in by_hemi.items():
                if len(hemi_graphs) > 1:
                    subject_dir = osp.join(morphologist_dir, subject) if subject else morphologist_dir
                    acqs = list_acquisitions(subject_dir)
                    acq_list = ", ".join(acqs) if acqs else "unknown"
                    print(
                        f"  Warning: {len(hemi_graphs)} {hemi} graphs found "
                        f"(acquisitions: {acq_list}). "
                        f"Using the first one. Re-run with --acquisition <name> "
                        f"to select a specific one."
                    )
                    break

        QUAT = {"left": (0.5, 0.5, 0.5, 0.5), "right": (0.5, -0.5, -0.5, 0.5)}

        total = sum(len(v) for v in by_hemi.values())
        print(f"  Found {total} sulcal graph(s)")

        for hemi, hemi_graphs in by_hemi.items():
            if not hemi_graphs:
                continue
            graph_path = hemi_graphs[0]
            white_mesh = find_white_mesh(graph_path)
            if white_mesh:
                print(f"  White mesh: {white_mesh}")
            acq_tag = f"_{acquisition}" if acquisition else ""
            out = osp.join(self.args.output_dir, f"sulcal_graph{acq_tag}_{hemi}.png")
            try:
                snap = generate_sulcal_graph_snapshot(
                    graph_path,
                    out,
                    size,
                    view_quaternion=QUAT[hemi],
                    mesh_path=white_mesh,
                )
                snapshots.append(snap)
            except Exception as e:
                print(f"  Error processing {graph_path}: {e}")

        return snapshots

    def _run_tiles(self, size):
        """Generate cortical tiles snapshots."""
        snapshots = []
        if self.args.cortical_tiles_dir and osp.exists(self.args.cortical_tiles_dir):
            # Accept either the crops/2mm/ path directly or the cortical_tiles root;
            # auto-detect crops/2mm/ subdirectory when the root is given.
            crops_dir = self.args.cortical_tiles_dir
            candidate = osp.join(crops_dir, "crops", "2mm")
            if osp.isdir(candidate):
                crops_dir = candidate
            print("\nGenerating cortical tiles snapshots...")
            out = osp.join(self.args.output_dir, "tiles_masks.png")
            try:
                snaps = generate_tiles_snapshot(
                    crops_dir,
                    out,
                    size,
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

        if not self.args.embeddings_dir:
            return snapshots
        if not osp.exists(self.args.embeddings_dir):
            print(f"Embeddings directory not found: {self.args.embeddings_dir}")
            return snapshots
        if not self.args.reference_data_dir:
            return snapshots
        if not osp.exists(self.args.reference_data_dir):
            print(f"Reference data directory not found: {self.args.reference_data_dir}")
            return snapshots

        regions = None
        umap_region = getattr(self.args, "umap_region", None)
        if umap_region:
            regions = [r.strip() for r in umap_region.split(",")]
            print(f"\nUMAP region filter: {', '.join(regions)}")

        print("\nGenerating UMAP scatter plots...")
        out = osp.join(self.args.output_dir, "umap.png")
        try:
            snaps = generate_umap_snapshot(
                self.args.embeddings_dir,
                self.args.reference_data_dir,
                out,
                size,
                regions=regions,
            )
            snapshots.extend(snaps)
        except Exception as e:
            print(f"  Error generating UMAP snapshot: {e}")
        return snapshots


def main():
    init_pipeline_process()
    script = GenerateSnapshots()
    return script.main()


if __name__ == "__main__":
    sys.exit(main())
