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
import logging
import os
import os.path as osp
import sys
from datetime import datetime

from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.derivatives_layout import compute_run_log_path
from champollion_pipeline.process_setup import init_pipeline_process
from champollion_pipeline.snapshot_types.input_qc import (
    SnapshotsInputQC,
    check_embedding_inputs,
    check_sulcal_inputs,
    find_crop_regions_without_mask,
    find_missing_region_graphs,
    find_missing_umap_references,
    log_warning,
)
from champollion_pipeline.snapshot_types.sulcal import (
    _detect_hemi,
    find_first_kept_subject_graphs,
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
from champollion_pipeline.snapshot_types.umap import discover_umap_pairs, generate_umap_snapshot, list_embedding_csvs

logger = logging.getLogger(__name__)

__all__ = [
    "ICBM_MESH_DIR_FALLBACK",
    "REGION_GRAPH_ROOT_HINT",
    "GenerateSnapshots",
    "_detect_hemi",
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

# Camera orientation (x, y, z, w) of each hemisphere's sulcal graph snapshot
SULCAL_VIEW_QUATERNIONS = {"left": (0.5, 0.5, 0.5, 0.5), "right": (0.5, -0.5, -0.5, 0.5)}


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
        """Run the input QC, then all requested snapshot generation steps; always returns 0."""
        os.makedirs(self.args.output_dir, exist_ok=True)

        run_sulcal, run_tiles, run_umap = self._selected_steps()
        excluded = self._run_input_qc(run_sulcal, run_tiles, run_umap)

        size = (self.args.width, self.args.height)
        all_snapshots = []

        if run_sulcal:
            all_snapshots.extend(self._run_sulcal(size, excluded=excluded))

        if run_tiles:
            all_snapshots.extend(self._run_tiles(size))

        if run_umap:
            all_snapshots.extend(self._run_umap(size, excluded=excluded))

        logger.output(f"Generated {len(all_snapshots)} snapshot(s)")

        manifest_path = osp.join(self.args.output_dir, "snapshots_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump({"snapshots": all_snapshots}, f, indent=2)
        logger.output(f"Manifest saved to: {manifest_path}")

        return 0

    def _selected_steps(self) -> tuple[bool, bool, bool]:
        """Return (run_sulcal, run_tiles, run_umap) from the --*-only flags (none set runs all)."""
        only_flags = (self.args.sulcal_only, self.args.tiles_only, self.args.umap_only)
        run_all = not any(only_flags)
        return run_all or self.args.sulcal_only, run_all or self.args.tiles_only, run_all or self.args.umap_only

    def _morphologist_dir(self) -> str | None:
        """Return --morphologist_dir when given and existing, else None."""
        morphologist_dir = self.args.morphologist_dir
        return morphologist_dir if morphologist_dir and osp.exists(morphologist_dir) else None

    def _umap_dirs(self) -> tuple[str, str] | None:
        """Return (embeddings_dir, reference_data_dir) when both are given and exist, else None."""
        dirs = (self.args.embeddings_dir, self.args.reference_data_dir)
        return dirs if all(d and osp.exists(d) for d in dirs) else None

    def _umap_regions(self) -> list[str] | None:
        """Return the --umap_region names, or None when no filter is given."""
        umap_region = getattr(self.args, "umap_region", None)
        return [r.strip() for r in umap_region.split(",")] if umap_region else None

    def _crops_dir(self) -> str | None:
        """Return the crops directory (<dir>/crops/2mm when present) of an existing --cortical_tiles_dir."""
        tiles_dir = self.args.cortical_tiles_dir
        if not tiles_dir or not osp.exists(tiles_dir):
            return None
        candidate = osp.join(tiles_dir, "crops", "2mm")
        return candidate if osp.isdir(candidate) else tiles_dir

    def _run_input_qc(self, run_sulcal, run_tiles, run_umap) -> frozenset[str]:
        """Check the selected steps' inputs, write the QC file, warn, and return the excluded subjects."""
        report = SnapshotsInputQC()
        morphologist_dir = self._morphologist_dir()
        if run_sulcal and morphologist_dir:
            check_sulcal_inputs(
                report,
                morphologist_dir,
                getattr(self.args, "subject", None),
                getattr(self.args, "acquisition", None),
            )
        umap_dirs = self._umap_dirs() if run_umap else None
        csvs = list_embedding_csvs(umap_dirs[0], self._umap_regions()) if umap_dirs else []
        check_embedding_inputs(report, csvs)

        qc_path = report.write(self.args.output_dir)
        excluded = report.excluded()
        log_warning(f"snapshots input QC excluded {len(excluded)} subject(s), see {qc_path}", excluded)
        if umap_dirs:
            reference_data_dir = umap_dirs[1]
            log_warning(
                f"missing UMAP reference file(s) in {reference_data_dir}",
                find_missing_umap_references(reference_data_dir, csvs),
            )
        crops_dir = self._crops_dir() if run_tiles else None
        if crops_dir:
            log_warning(
                f"crop region folder(s) of {crops_dir} with no Lmask_skeleton.nii.gz or "
                "Rmask_skeleton.nii.gz under mask/",
                find_crop_regions_without_mask(crops_dir),
            )
            if self.args.champollion_data_root:
                log_warning(
                    "missing region graph file(s)",
                    find_missing_region_graphs(self.args.champollion_data_root, crops_dir, self.args.tiles_level),
                )
        return frozenset(excluded)

    def _run_sulcal(self, size, excluded=frozenset()):
        """Generate sulcal graph snapshots of the first subject not excluded by the input QC."""
        snapshots = []
        morphologist_dir = self.args.morphologist_dir
        if not morphologist_dir or not osp.exists(morphologist_dir):
            if morphologist_dir:
                logger.warning(f"Morphologist directory not found: {morphologist_dir}")
            return snapshots

        logger.trace("Generating sulcal graph snapshots...")
        subject = getattr(self.args, "subject", None)
        acquisition = getattr(self.args, "acquisition", None)

        by_hemi = find_first_kept_subject_graphs(morphologist_dir, subject, acquisition, excluded)
        if not acquisition:
            self._warn_multiple_acquisitions(by_hemi, morphologist_dir, subject)

        total = sum(len(v) for v in by_hemi.values())
        logger.trace(f"Found {total} sulcal graph(s)")

        for hemi, hemi_graphs in by_hemi.items():
            if not hemi_graphs:
                continue
            graph_path = hemi_graphs[0]
            white_mesh = find_white_mesh(graph_path)
            if white_mesh:
                logger.trace(f"White mesh: {white_mesh}")
            acq_tag = f"_{acquisition}" if acquisition else ""
            out = osp.join(self.args.output_dir, f"sulcal_graph{acq_tag}_{hemi}.png")
            try:
                snap = generate_sulcal_graph_snapshot(
                    graph_path,
                    out,
                    size,
                    view_quaternion=SULCAL_VIEW_QUATERNIONS[hemi],
                    mesh_path=white_mesh,
                )
                snapshots.append(snap)
            except Exception as e:
                logger.error(f"Error processing {graph_path}: {e}")

        return snapshots

    def _warn_multiple_acquisitions(self, by_hemi, morphologist_dir, subject):
        """Warn once when a hemisphere has several graphs and no --acquisition was given."""
        for hemi, hemi_graphs in by_hemi.items():
            if len(hemi_graphs) > 1:
                subject_dir = osp.join(morphologist_dir, subject) if subject else morphologist_dir
                acqs = list_acquisitions(subject_dir)
                acq_list = ", ".join(acqs) if acqs else "unknown"
                logger.warning(
                    f"{len(hemi_graphs)} {hemi} graphs found "
                    f"(acquisitions: {acq_list}). "
                    f"Using the first one. Re-run with --acquisition <name> "
                    f"to select a specific one."
                )
                break

    def _run_tiles(self, size):
        """Generate cortical tiles snapshots."""
        snapshots = []
        # Accept either the crops/2mm/ path directly or the cortical_tiles root
        crops_dir = self._crops_dir()
        if crops_dir:
            logger.trace("Generating cortical tiles snapshots...")
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
                logger.error(f"Anatomist not available for tiles snapshot: {e}")
            except Exception as e:
                logger.error(f"Error generating tiles snapshot: {e}")
        elif self.args.cortical_tiles_dir:
            logger.warning(f"Cortical tiles directory not found: {self.args.cortical_tiles_dir}")
        return snapshots

    def _run_umap(self, size, excluded=frozenset()):
        """Generate UMAP scatter plots."""
        snapshots = []

        if not self.args.embeddings_dir:
            return snapshots
        if not osp.exists(self.args.embeddings_dir):
            logger.warning(f"Embeddings directory not found: {self.args.embeddings_dir}")
            return snapshots
        if not self.args.reference_data_dir:
            return snapshots
        if not osp.exists(self.args.reference_data_dir):
            logger.warning(f"Reference data directory not found: {self.args.reference_data_dir}")
            return snapshots

        regions = self._umap_regions()
        if regions:
            logger.trace(f"UMAP region filter: {', '.join(regions)}")

        logger.trace("Generating UMAP scatter plots...")
        out = osp.join(self.args.output_dir, "umap.png")
        try:
            snaps = generate_umap_snapshot(
                self.args.embeddings_dir,
                self.args.reference_data_dir,
                out,
                size,
                regions=regions,
                exclude_ids=excluded,
            )
            snapshots.extend(snaps)
        except Exception as e:
            logger.error(f"Error generating UMAP snapshot: {e}")
        return snapshots


def main():
    init_pipeline_process()
    script = GenerateSnapshots()
    return script.main()


if __name__ == "__main__":
    sys.exit(main())
