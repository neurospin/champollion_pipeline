#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate sulcal mask files from manually labelled graph data.
"""

import glob
import json
import os
import sys
from os.path import abspath, dirname, exists, join

import numpy as np

from champollion_utils.script_builder import ScriptBuilder

# Default regions — same 28 as generate_sulcal_regions.py
_REGIONS_DEFAULT = [
    "S.C.-sylv.", "S.C.-S.Pe.C.", "S.C.-S.Po.C.",
    "S.Pe.C.", "S.Po.C.", "S.F.int.-F.C.M.ant.",
    "S.F.inf.-BROCA-S.Pe.C.inf.", "S.T.s.", "Sc.Cal.-S.Li.",
    "F.C.M.post.-S.p.C.", "S.T.i.-S.O.T.lat.",
    "OCCIPITAL", "F.I.P.-F.I.P.Po.C.inf.", "S.F.inter.-S.F.sup.",
    "S.F.median-S.F.pol.tr.-S.F.sup.", "S.Or.",
    "S.Or.-S.Olf.", "F.P.O.-S.Cu.-Sc.Cal.",
    "S.s.P.-S.Pa.int.", "S.T.s.br.",
    "Lobule_parietal_sup.", "S.F.marginal-S.F.inf.ant.",
    "F.Coll.-S.Rh.", "S.T.i.-S.T.s.-S.T.pol.",
    "F.C.L.p.-subsc.-F.C.L.a.-INSULA.", "S.F.int.-S.R.",
    "S.Call.", "S.Call.-S.s.P.-S.intraCing.",
]


def get_sulci_for_regions(regions, sides, json_path):
    """Return the set of unique sulcus names covered by the given regions and sides.

    Reads sulci_regions_champollion_V1.json, looks up each region+side key,
    collects all sulcus sub-keys, strips the side suffix, and returns a
    deduplicated set of bare sulcus names.
    """
    with open(json_path, 'r') as f:
        brain_regions = json.load(f)["brain"]

    sulci = set()
    for region in regions:
        for side in sides:
            side_suffix = "_left" if side == "L" else "_right"
            key = f"{region}{side_suffix}"
            if key in brain_regions:
                for sulcus_with_side in brain_regions[key]:
                    sulci.add(sulcus_with_side.replace(side_suffix, ""))
    return sulci


def _compute_one_sulcus(sulcus_full, per_subject_voxels, voxel_size_tuple,
                        mask_dir, side, brainvisa_dir, public_use=False):
    """Worker: build one sulcal mask from pre-extracted voxel coords.

    Receives only pure-numpy data so it is safe to call from joblib threads
    or processes.  Each invocation creates its own aims.Volume objects
    independently — no PyAIMS state is shared with other workers.

    When public_use=True, per-subject NIfTI files are not written; only the
    aggregated count mask is produced.
    """
    import os as _os  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import numpy as _np  # noqa: PLC0415
    from os.path import join as _join  # noqa: PLC0415
    if brainvisa_dir not in sys.path:
        sys.path.insert(0, brainvisa_dir)
    from soma import aims  # noqa: PLC0415
    from compute_mask import initialize_mask, write_mask  # noqa: PLC0415

    try:
        sample_dir = _join(mask_dir, side, sulcus_full)
        _os.makedirs(_join(mask_dir, side), exist_ok=True)
        if not public_use:
            _os.makedirs(sample_dir, exist_ok=True)

        mask = initialize_mask(voxel_size_tuple)
        arr = _np.asarray(mask)

        for sub_name, sub_voxels in per_subject_voxels.items():
            voxels = sub_voxels.get(sulcus_full, _np.empty((0, 3), dtype=_np.int32))
            arr_one = _np.zeros(arr.shape, dtype=_np.int16)
            if len(voxels):
                # vectorised: no Python loop over individual voxel coords
                arr_one[voxels[:, 0], voxels[:, 1], voxels[:, 2], 0] = 1
            if not public_use:
                vol_one = aims.Volume(arr_one)
                vol_one.copyHeaderFrom(mask.header())
                vol_one.header()['voxel_size'] = mask.header()['voxel_size']
                aims.write(vol_one, _join(sample_dir, f"{sub_name}.nii.gz"))
            arr += arr_one

        write_mask(mask, _join(mask_dir, side, sulcus_full + '.nii.gz'))
    except Exception as e:
        return f"failed: {e}"
    return "ok"


def _load_and_extract_subject(sub, sulci_full_set, voxel_size_tuple,
                              brainvisa_dir):
    """Worker: load one subject graph and extract voxel coords for all sulci.

    Returns (sub_name, sub_data) where sub_data is
    {sulcus_full: np.ndarray (K,3) int32}, or (sub_name, None) if no graph
    file was found.
    """
    import glob as _glob  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import numpy as _np  # noqa: PLC0415
    from os.path import join as _join  # noqa: PLC0415
    if brainvisa_dir not in sys.path:
        sys.path.insert(0, brainvisa_dir)
    from soma import aims  # noqa: PLC0415

    matches = _glob.glob(_join(sub['dir'], sub['graph_file'] % sub))
    if not matches:
        return sub['subject'], None

    graph = aims.read(matches[0])
    g_to_icbm = aims.GraphManip.getICBM2009cTemplateTransform(graph)
    voxel_size_in = graph['voxel_size'][:3]

    sub_data: dict = {}
    for vertex in graph.vertices():
        vname = vertex.get('name')
        if vname not in sulci_full_set:
            continue
        parts = []
        for bucket_name in ('aims_ss', 'aims_bottom', 'aims_other'):
            bucket = vertex.get(bucket_name)
            if bucket is None:
                continue
            vr = _np.asarray([
                g_to_icbm.transform(_np.array(v) * voxel_size_in)
                for v in bucket[0].keys()
            ])
            if vr.shape[0] == 0:
                continue
            parts.append(
                _np.round(vr / voxel_size_tuple).astype(_np.int32))
        sub_data[vname] = (
            _np.vstack(parts) if parts
            else _np.empty((0, 3), dtype=_np.int32))
    return sub['subject'], sub_data


def _run_buffered(sulci, sides, mask_dir, labeled_subjects_dir,
                  path_to_graph_supervised, nb_subjects, voxel_size, force,
                  njobs, brainvisa_dir, public_use=False):
    """Load and extract subject graphs in parallel, then compute masks.

    Yields (side, results_dict) where results_dict maps sulcus → status string
    ('ok', 'skipped', or 'failed: <msg>').

    Two-phase approach:
      Phase 1 (parallel) — M subject workers each load one graph and extract
                           voxel coords for all sulci; returns pure-numpy data.
      Phase 2 (parallel) — N sulcus workers each build one mask from the
                           pre-extracted voxels.

    Peak memory: one graph at a time per thread (no full graph_buffer).
    Vertex iterations: M (one pass per subject, not M×N).
    """
    from joblib import Parallel, delayed  # noqa: PLC0415
    from deep_folding.brainvisa.utils.subjects import (  # noqa: PLC0415
        get_all_subjects_as_dictionary,
        select_subjects_int_if_list_of_dict,
    )
    from deep_folding.brainvisa.utils.sulcus import (  # noqa: PLC0415
        complete_sulci_name,
    )

    voxel_size_tuple = (voxel_size, voxel_size, voxel_size)

    for side in sides:
        graph_file_pattern = (
            '%(subject)s/'
            + path_to_graph_supervised
            + '/%(side)s%(subject)s*.arg'
        )

        subjects = get_all_subjects_as_dictionary(
            [labeled_subjects_dir], [graph_file_pattern], side)
        subjects = select_subjects_int_if_list_of_dict(
            subjects, subjects, nb_subjects)

        # ── Phase 1b: determine which sulci need work ──────────────────────
        sulci_to_run = {
            complete_sulci_name(s, side): s
            for s in sorted(sulci)
            if force or not exists(
                join(mask_dir, side, complete_sulci_name(s, side) + '.nii.gz'))
        }
        sulci_full_set = set(sulci_to_run)

        # ── Phase 1: parallel load + extract per subject ───────────────────
        print(f"  [{side}] Loading and extracting {len(subjects)} subjects "
              f"({len(sulci_to_run)} sulci) with {njobs} worker(s)…")
        raw_results = Parallel(n_jobs=njobs, prefer='processes')(
            delayed(_load_and_extract_subject)(
                sub, sulci_full_set, voxel_size_tuple, brainvisa_dir,
            )
            for sub in subjects
        )
        per_subject_voxels: dict = {}
        for sub_name, sub_data in raw_results:
            if sub_data is None:
                print(f"  [{side}] WARNING: no graph for {sub_name}, skipped")
            else:
                per_subject_voxels[sub_name] = sub_data
        print(f"  [{side}] Load+extract done ({len(per_subject_voxels)} subjects).")

        # ── Phase 2: parallel sulcus compute ──────────────────────────────
        print(f"  [{side}] Computing {len(sulci_to_run)} sulci "
              f"with {njobs} worker(s)…")
        sorted_sulci_full = sorted(sulci_to_run)
        try:
            job_outputs = Parallel(n_jobs=njobs, prefer='threads')(
                delayed(_compute_one_sulcus)(
                    sf, per_subject_voxels, voxel_size_tuple,
                    mask_dir, side, brainvisa_dir, public_use,
                )
                for sf in sorted_sulci_full
            )
            run_results = {
                sulci_to_run[sf]: r
                for sf, r in zip(sorted_sulci_full, job_outputs)
            }
        except Exception as e:
            run_results = {sulci_to_run[sf]: f"failed: {e}"
                          for sf in sorted_sulci_full}

        # merge skipped entries
        results = {}
        for s in sorted(sulci):
            sf = complete_sulci_name(s, side)
            results[s] = run_results.get(s, "skipped") if sf in sulci_full_set \
                else "skipped"

        # completeness check: warn about any expected mask files missing on disk
        missing = [
            sf for sf in sulci_to_run
            if not exists(join(mask_dir, side, sf + '.nii.gz'))
        ]
        if missing:
            print(f"  [{side}] WARNING: {len(missing)} mask file(s) missing "
                  f"after run:")
            for sf in missing:
                print(f"    MISSING: {sf}.nii.gz")
        else:
            print(f"  [{side}] All {len(sulci_to_run)} mask files present.")

        yield side, results


class GenerateMasks(ScriptBuilder):
    """Script for generating sulcal mask files from manually labelled graphs."""

    def __init__(self):
        super().__init__(
            script_name="generate_masks",
            description=(
                "Generate binary sulcal mask files (.nii.gz) in MNI space "
                "from manually labelled graph data."
            )
        )
        (self.add_required_argument(
             "--labeled_subjects_dir",
             "Directory containing manually labelled subjects "
             "(e.g. /neurospin/dico/data/bv_databases/human/manually_labeled/pclean/all).")
         .add_required_argument(
             "--path_to_graph_supervised",
             "Relative sub-path from each subject dir to the graph files "
             "(e.g. t1mri/t1/default_analysis/folds/3.3/base2026_manual).")
         .add_required_argument(
             "--output_dir",
             "Directory where mask files will be written. "
             "Output structure: {output_dir}/{voxel_size}/{side}/{sulcus}.nii.gz")
         .add_argument(
             "--regions", nargs="+", default=None,
             help="Sulcal regions to process. Default: all 28 regions.")
         .add_argument(
             "--sides", nargs="+", default=["L", "R"],
             help="Hemisphere sides to process. Default: L R.")
         .add_optional_argument(
             "--voxel_size", "Output voxel size in mm.", default=2.0, type_=float)
         .add_optional_argument(
             "--nb_subjects",
             "Number of subjects to use (-1 = all).", default=-1, type_=int)
         .add_optional_argument(
             "--masks",
             "Mask version tag appended before the voxel-size level "
             "(e.g. 'canonical_25'). If omitted, output_dir is used directly.",
             default=None)
         .add_argument(
             "--buffered", action="store_true", default=False,
             help="Load all subject graphs into RAM once before processing "
                  "sulci. Faster when generating many sulci (reduces I/O "
                  "from N×M to M reads). Requires more memory.")
         .add_optional_argument(
             "--njobs",
             "Parallel workers used in --buffered mode. "
             "Default: cpu_count - 2 (max 22).",
             default=None, type_=int)
         .add_argument(
             "--force", action="store_true", default=False,
             help="Force recompute even if the mask file already exists.")
         .add_flag(
             "--public_use",
             "Skip per-subject NIfTI files; write only the aggregated mask."))

    def run(self):
        """Execute mask generation for all (sulcus × side) combinations."""
        print(f"generate_masks.py/labeled_subjects_dir: {self.args.labeled_subjects_dir}")
        print(f"generate_masks.py/path_to_graph_supervised: {self.args.path_to_graph_supervised}")
        print(f"generate_masks.py/output_dir: {self.args.output_dir}")

        regions = self.args.regions if self.args.regions else _REGIONS_DEFAULT
        sides = self.args.sides

        brainvisa_dir = abspath(join(
            dirname(__file__),
            '..', 'external', 'cortical_tiles', 'deep_folding', 'brainvisa'
        ))
        if brainvisa_dir not in sys.path:
            sys.path.insert(0, brainvisa_dir)

        from compute_mask import compute_mask  # noqa: PLC0415

        json_path = abspath(join(
            dirname(__file__), '..', 'sulci_regions_champollion_V1.json'
        ))
        sulci = get_sulci_for_regions(regions, sides, json_path)
        print(f"generate_masks.py: {len(sulci)} unique sulci from "
              f"{len(regions)} region(s) × {len(sides)} side(s)")

        vox_str = f"{int(self.args.voxel_size)}mm"
        mask_dir = (
            join(abspath(self.args.output_dir), self.args.masks, vox_str)
            if self.args.masks
            else join(abspath(self.args.output_dir), vox_str)
        )

        n_ok = 0
        n_skipped = 0
        n_failed = 0

        use_buffered = self.args.buffered or (self.args.njobs is not None)
        if self.args.njobs is not None and not self.args.buffered:
            print("generate_masks.py: --njobs implies --buffered, "
                  "enabling buffered mode automatically.")

        if use_buffered:
            from deep_folding.brainvisa.utils.parallel import define_njobs  # noqa: PLC0415
            njobs = define_njobs(self.args.njobs)
            print(f"generate_masks.py: buffered mode — "
                  f"loading graphs once per side, {njobs} worker(s)")
            for side, results in _run_buffered(
                    sulci, sides, mask_dir,
                    self.args.labeled_subjects_dir,
                    self.args.path_to_graph_supervised,
                    self.args.nb_subjects,
                    self.args.voxel_size,
                    self.args.force,
                    njobs,
                    brainvisa_dir,
                    public_use=self.args.public_use):
                for sulcus, result in results.items():
                    if result == "skipped":
                        print(f"  ✓ [{side}] {sulcus} (already exists, skipping)")
                        n_skipped += 1
                    elif result == "ok":
                        print(f"  → [{side}] {sulcus} ok")
                        n_ok += 1
                    else:
                        print(f"  ✗ [{side}] {sulcus}  {result}")
                        n_failed += 1
        else:
            for side in sides:
                side_suffix = "_left" if side == "L" else "_right"
                for sulcus in sorted(sulci):
                    mask_file = join(
                        mask_dir, side, sulcus + side_suffix + ".nii.gz")
                    if not self.args.force and exists(mask_file):
                        print(f"  ✓ [{side}] {sulcus} (already exists, skipping)")
                        n_skipped += 1
                        continue
                    print(f"  → [{side}] {sulcus}")
                    try:
                        compute_mask(
                            src_dir=self.args.labeled_subjects_dir,
                            path_to_graph=self.args.path_to_graph_supervised,
                            mask_dir=mask_dir,
                            sulcus=sulcus,
                            new_sulcus=None,
                            side=side,
                            number_subjects=self.args.nb_subjects,
                            out_voxel_size=self.args.voxel_size,
                        )
                        n_ok += 1
                    except Exception as e:
                        print(f"     ERROR: {e}")
                        n_failed += 1

        sep = "=" * 60
        print(f"\n{sep}")
        print(f"MASK GENERATION SUMMARY")
        print(f"  Total:    {len(sulci) * len(sides)}")
        print(f"  Skipped:  {n_skipped} (already exist)")
        print(f"  Success:  {n_ok}")
        print(f"  Failed:   {n_failed}")
        print(f"\n  Output:   {mask_dir}/{{side}}/{{sulcus}}.nii.gz")
        print(f"{sep}\n")

        return 0 if n_failed == 0 else 1


def main():
    """Main entry point."""
    script = GenerateMasks()
    return script.build().print_args().run()


if __name__ == "__main__":
    exit(main())
