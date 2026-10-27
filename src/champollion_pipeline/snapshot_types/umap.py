"""UMAP snapshots: embedding CSV discovery and the matplotlib scatter plots."""

import glob
import logging
import os.path as osp
import re

import champollion_utils.logger  # noqa: F401 - installs Logger.trace/output/fail (LOG 22, OUTPUT 25, FAIL 45)
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Embeddings CSV names accepted by UMAP discovery, tried in order (first match wins).
_EMBEDDING_CSV_PATTERNS = (
    # _{region}--{hemi}--{model}_embeddings.csv
    re.compile(r"^_(?P<region>.+?)--(?P<hemi>left|right)--(?P<suffix>.+)_embeddings\.csv$"),
    # {region}_{hemi}_embeddings.csv (combine stage) and {region}_{hemi}_{suffix}_embeddings.csv
    re.compile(r"^(?P<region>.+?)_(?P<hemi>left|right)(?:_(?P<suffix>.+))?_embeddings\.csv$"),
)


def _parse_embedding_csv_name(name):
    """Parse (region, hemi) from an embeddings CSV filename.

    Accepted naming conventions:
      - Combine stage: {region}_{hemi}_embeddings.csv
      - Suffixed: {region}_{hemi}_{suffix}_embeddings.csv
      - Model: _{region}--{hemi}--{model}_embeddings.csv

    The region is the text before the first ``_left``/``_right`` hemisphere token
    and is kept unchanged (hyphens, underscores and dots), except that ``--`` is
    rewritten to ``_`` in the model naming. Region names must therefore not
    contain ``_left_``/``_right_`` or end with ``_left``/``_right``.

    Returns (region, hemi) or (None, None) if not parseable.
    """
    for index, pattern in enumerate(_EMBEDDING_CSV_PATTERNS):
        match = pattern.match(name)
        if match:
            region = match.group("region")
            if index == 0:
                region = region.replace("--", "_")
            return region, match.group("hemi")
    return None, None


def compute_umap_reference_paths(reference_data_dir, region, hemi):
    """Return (UMAP model path, reference coordinates path) of one region and hemisphere."""
    return (
        osp.join(reference_data_dir, f"umap_{region}_{hemi}.pkl"),
        osp.join(reference_data_dir, f"umap_{region}_{hemi}_coords.npy"),
    )


def _iter_embedding_csvs(embeddings_dir):
    """Yield (path, region, hemi) per ``*_embeddings.csv`` in sorted order; (None, None) when unparseable."""
    for csv_path in sorted(glob.glob(osp.join(embeddings_dir, "*_embeddings.csv"))):
        region, hemi = _parse_embedding_csv_name(osp.basename(csv_path))
        yield csv_path, region, hemi


def list_embedding_csvs(embeddings_dir, regions=None):
    """Return the parseable embeddings CSVs as (path, region, hemi), sorted, kept by the region filter."""
    return [
        (csv_path, region, hemi)
        for csv_path, region, hemi in _iter_embedding_csvs(embeddings_dir)
        if region is not None and (not regions or region in regions)
    ]


def discover_umap_pairs(embeddings_dir, reference_data_dir, regions=None):
    csv_files = list(_iter_embedding_csvs(embeddings_dir))
    if not csv_files:
        logger.warning(f"No *_embeddings.csv files found in: {embeddings_dir}")
        return []
    logger.trace(f"Found {len(csv_files)} embedding CSV file(s)")

    pairs = []
    for csv_path, region, hemi in csv_files:
        if region is None:
            logger.warning(f"Unrecognized filename format: {osp.basename(csv_path)}")
            continue
        if regions and region not in regions:
            continue
        model_path, coords_path = compute_umap_reference_paths(reference_data_dir, region, hemi)
        if osp.exists(model_path) and osp.exists(coords_path):
            pairs.append((csv_path, model_path, coords_path, region, hemi))
        else:
            logger.trace(
                f"UMAP model not found for {region} {hemi} — skipping "
                f"(expected {osp.basename(model_path)} in reference_data_dir)"
            )
    return pairs


def generate_umap_snapshot(  # noqa: PLR0913 - snapshot API; keyword-only QC exclusion set
    embeddings_dir, reference_data_dir, output_path, size=(800, 600), regions=None, *, exclude_ids=frozenset()
):
    """Generate UMAP scatter plots for embedding regions.

    Discovers embedding CSV files in ``embeddings_dir`` and generates a UMAP
    projection plot for every (region, hemisphere) pair that has both a CSV and
    pre-trained model artefacts in ``reference_data_dir``.

    Args:
        embeddings_dir: Path to pipeline embeddings directory.
        reference_data_dir: Path to pre-trained UMAP models and reference
            coordinate arrays.
        output_path: Base path for output images.  Each plot is saved as
            ``{basename}_{region}_{hemi}{ext}``.
        size: Tuple of (width, height) in pixels.
        regions: Optional list of region names to include.  ``None`` means
            generate plots for all available regions.
        exclude_ids: Subject IDs whose rows are dropped before projection
            (subjects excluded by the stage input QC).

    Returns:
        List of generated snapshot file paths.
    """
    matplotlib.use("Agg")

    basename = osp.splitext(output_path)[0]
    ext = osp.splitext(output_path)[1] or ".png"
    snapshots = []

    pairs = discover_umap_pairs(embeddings_dir, reference_data_dir, regions=regions)
    if not pairs:
        logger.warning("No matching (embedding CSV, UMAP model) pairs found")
        return snapshots

    for pair in pairs:
        region, hemi = pair[3], pair[4]
        snap = _plot_umap_pair(pair, size, f"{basename}_{region}_{hemi}{ext}", exclude_ids)
        if snap:
            snapshots.append(snap)

    return snapshots


def _plot_umap_pair(pair, size, snap, exclude_ids):
    """Project one CSV's kept rows onto its UMAP reference and save the plot; None when no row is kept."""
    csv_path, model_path, coords_path, region, hemi = pair
    model = joblib.load(model_path)
    ref_coords = np.load(coords_path)
    logger.trace(f"[{region} {hemi}] Loaded {ref_coords.shape[0]} reference points")

    df = pd.read_csv(csv_path, dtype={"ID": str})
    df = df[~df["ID"].isin(exclude_ids)]
    if df.empty:
        logger.trace(f"[{region} {hemi}] No kept subject to project, skipping")
        return None
    X_new = df.drop(columns=["ID"]).values.astype(np.float32)
    new_coords = model.transform(X_new)
    logger.trace(f"[{region} {hemi}] Projected {X_new.shape[0]} new subject(s)")

    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100))
    ax.scatter(
        ref_coords[:, 0],
        ref_coords[:, 1],
        s=1,
        c="#4a90d9",
        alpha=0.08,
        label=f"UKB40 (n={ref_coords.shape[0]:,})",
        rasterized=True,
    )
    ax.scatter(
        new_coords[:, 0],
        new_coords[:, 1],
        s=80,
        c="#e74c3c",
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
        label=f"Your subject(s) (n={X_new.shape[0]})",
    )
    ax.set_title(f"{region} \u2014 {hemi}", fontsize=12)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout()

    plt.savefig(snap, dpi=150)
    plt.close(fig)
    logger.output(f"Saved: {snap}")
    return snap
