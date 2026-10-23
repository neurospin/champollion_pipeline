#!/usr/bin/env python3
"""
Unified comparison script for sulcal data.

Subcommands
-----------
  masks           Compare two sets of sulcal NIfTI masks (side/sulcus.nii.gz).
  cortical_tiles  Compare two cortical_tiles outputs (region/mask/<pattern>).
  databases       Compare sulcal labeling between two graph annotation campaigns.
  crops           Compare two cortical_tiles crops/2mm .npy crop sets per subject (see below).

Mask metrics (masks / cortical_tiles)
--------------------------------------
  wasserstein  – approximate 3-D Earth Mover's Distance via axis-marginal
                 projections (in voxels; for 2 mm masks, 1 vox ≈ 2 mm).
  diff         – voxel-level change counts: changed / added / removed.
  both         – compute and report both metrics.

Usage
-----
    python compare.py masks --set_a /path/to/mask/canonical_25/2.0 \\
        --set_b /path/to/mask/corrected/2.0 --metric both

    python compare.py cortical_tiles \\
        --set_a /path/to/run_a/crops/2mm \\
        --set_b /path/to/run_b/crops/2mm \\
        --pattern "*mask_skeleton.nii.gz" --metric diff

    python compare.py databases \\
        --labeled_subjects_dir /neurospin/.../manually_labeled/pclean/all \\
        --path_to_graph_a t1mri/t1/default_analysis/folds/3.3/base2018_manual \\
        --path_to_graph_b t1mri/t1/default_analysis/folds/3.3/base2018b_manual \\
        --label_a base2018 --label_b base2018b

Crops comparison (crops)
------------------------
  Compares two cortical_tiles crops/2mm directories subject by subject, per
  region and side, on the .npy crop sets (mask/{side}{input_type}.npy plus
  mask/{side}{input_type}_subject.csv); subjects are paired by ID.

  Prerequisite: PyAIMS (soma.aims). For each region and side, crops reads only
  the header of mask/{side}mask_cropped.nii.gz (no voxel data) to align the two
  crop grids. The header transformation maps AIMS storage order (identical to
  .npy order, so no axis flip) to a referential; with a common referential,
  equal voxel size vs and equal diagonal +/-1 rotations R, set B is placed at
  offset = R⁻¹(t_b − t_a)/vs voxels from set A, and both crops are compared on
  their union grid.

  Options:
    --set_a, --set_b  crops/2mm directories to compare (required).
    --output          output directory (default: crops_comparison).
    --input_type      crop file stem: skeleton, label, extremities, distmap or
                      distbottom (default: skeleton).
    --regions         region names (default: region dirs present in both sets).
    --side            L, R or both (default: both).
    --top_k           length of each top_lost list (default: 5).
    --njobs           joblib workers over region/side pairs (default: 1).
    --xor_dir         optional directory for per-subject XOR files (default: none).

  Outputs, written into --output:
    per_subject.csv  one row per region, side and common subject. Columns:
                     region, side, subject; n_a, n_b (voxels in A, in B);
                     kept, lost, gained (in both, A only, B only);
                     pct_lost (100 * lost / n_a); dice (Dice of A and B);
                     shift (Wasserstein distance in voxels, inf when one
                     side is empty).
    summary.json     top-level keys: set_a, set_b, input_type, regions, skipped.
                     regions maps "<region>/<side>" to: n_subjects_compared,
                     only_in_a, only_in_b, subjects_changed, pct_lost_mean,
                     pct_lost_p95, pct_lost_max, dice_mean, dice_min,
                     subjects_emptied, top_lost, crop_shape_a, crop_shape_b,
                     alignment_offset_vox (null = compared without header
                     check).
                     skipped is a list of entries with keys region, side,
                     reason and, when both crops loaded, shape_a, shape_b.

  XOR files, written into --xor_dir when given:
    one file per top_lost subject of each compared region/side, at
    <xor_dir>/<region>/<side>/<subject>_xor.nii.gz (PyAIMS, when
    alignment_offset_vox is a list: set A voxel size and referentials,
    transformation moved to the union-grid origin) or <subject>_xor.npy
    (numpy, when alignment_offset_vox is null); 1 = voxel differs, 0 = same,
    on the grid the subject was compared on.

  Skip reasons (summary.json skipped entries):
    missing_in_a            .npy or _subject.csv absent in set A only.
    missing_in_b            .npy or _subject.csv absent in set B only.
    missing_in_a_and_b      absent in both sets, for an explicit --regions name.
    subject_csv_mismatch    _subject.csv row count differs from the .npy length.
    no_mask_cropped         mask_cropped.nii.gz missing or unreadable, shapes differ.
    no_transformation       header has no transformation, shapes differ.
    referential_differs     the two headers share no referential.
    voxel_size_differs      voxel sizes differ.
    non_axis_aligned        rotation is not diagonal +/-1.
    transformations_differ  rotations differ between A and B.
    non_integer_offset      offset is not a whole number of voxels.
    no_common_subjects      no subject ID present in both sets.
  no_mask_cropped and no_transformation with equal shapes are compared
  directly (WARNING on stdout, alignment_offset_vox null) instead of skipped.

  Example:
    python compare.py crops \\
        --set_a /path/to/run_a/crops/2mm \\
        --set_b /path/to/run_b/crops/2mm \\
        --output crops_comparison --side both --njobs 8

Embeddings comparison (embeddings)
-----------------------------------
  Compares two champollion_V1 embedding directories (--set_a, --set_b),
  subject by subject, per region and side. Subjects are paired by ID.

  Raw UMAP coordinates of two separate UMAP fits are not comparable
  without alignment; --umap_a and --umap_b projection references are
  aligned via Procrustes (translation, rotation/reflection, isotropic scale).

  Options:
    --set_a, --set_b         Embedding directories to compare (required).
    --output                 Output directory (default: embeddings_comparison).
    --regions                Region names to compare (default: all in both sets).
    --side                   L, R or both (default: both).
    --k                      k-nearest-neighbour overlap k (default: 15).
    --njobs                  Joblib workers over region/side pairs (default: 1).
    --umap_a                 UMAP reference directory for set A (.pkl files).
    --umap_b                 UMAP reference directory for set B (both or neither).
    --covariate              CSV file with per-subject covariate values.
    --covariate_column       Column name of the covariate in --covariate CSV.

  Covariate CSV contract: plain table with a ``subject`` column and the
  ``--covariate_column`` column, one row per subject; no region/side
  filtering is done. A crops per_subject.csv must first be filtered to one
  region/side; a repeated subject is rejected with exit 1.

  Outputs, written into --output:
    per_subject.csv  one row per region, side and common subject.
    summary.json     regions["<region>/<side>"] with n_subjects, only_in_a,
                     only_in_b, cka, knn_overlap_mean, knn_overlap_median
                     and, with a covariate, spearman_covariate_knn_change /
                     spearman_covariate_displacement; top-level umap_skipped
                     list of {region, side, reason} (always present).
    umap_<region>_<side>.png  per compared region/side when UMAP refs given.
"""

import sys

from compare_sulcal.cli import Compare, main, run_subcommand
from compare_sulcal.databases import _get_subject_voxel_counts, _mask_stats
from compare_sulcal.embeddings import knn_overlap, linear_cka, procrustes_align
from compare_sulcal.masks import (
    bucket_label,
    find_masks,
    load_mask_vol,
    save_xor_vol,
    sort_buckets,
    voxel_diff,
    wasserstein_distance,
)
from compare_sulcal.viewer import _VIEWER_SCRIPT, _has_usable_display, visualise_mask_diffs

__all__ = [
    "Compare",
    "_VIEWER_SCRIPT",
    "_get_subject_voxel_counts",
    "_has_usable_display",
    "_mask_stats",
    "bucket_label",
    "find_masks",
    "knn_overlap",
    "linear_cka",
    "load_mask_vol",
    "main",
    "procrustes_align",
    "run_subcommand",
    "save_xor_vol",
    "sort_buckets",
    "visualise_mask_diffs",
    "voxel_diff",
    "wasserstein_distance",
]

if __name__ == "__main__":
    sys.exit(main())
