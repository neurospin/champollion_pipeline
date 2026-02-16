#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train UMAP models on UKBioBank reference embeddings.

Produces anonymized artifacts (no subject IDs) for runtime visualization:
  - Fitted UMAP model (.pkl) for projecting new subjects
  - Pre-computed 2D coordinates (.npy) for plotting the reference cloud

Usage:
    python train_umap_reference.py /path/to/ukb40 --output reference_data/
"""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import umap

# Collateral sulcus region files (both hemispheres)
COLLATERAL_FILES = {
    "left": "FColl-SRh_left_name06-43-43--210_embeddings.csv",
    "right": "FColl-SRh_right_name06-56-15--113_embeddings.csv",
}


def train_and_save(embeddings_dir, output_dir, n_neighbors=15, min_dist=0.1,
                   random_state=42):
    """Train UMAP on reference embeddings and save anonymized artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for hemi, filename in COLLATERAL_FILES.items():
        csv_path = os.path.join(embeddings_dir, filename)
        print(f"\n[{hemi}] Loading {csv_path}")

        df = pd.read_csv(csv_path)
        # Drop subject ID column — only keep embedding dimensions
        X = df.drop(columns=["ID"]).values.astype(np.float32)
        print(f"  Shape: {X.shape} ({X.shape[0]} subjects, {X.shape[1]} dims)")

        print(f"  Fitting UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
        model = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
        coords_2d = model.fit_transform(X)

        # Save fitted model (contains transform parameters, no subject data)
        region = filename.split("_")[0]  # "FColl-SRh"
        model_path = output_dir / f"umap_{region}_{hemi}.pkl"
        joblib.dump(model, model_path)
        print(f"  Saved model: {model_path}")

        # Save anonymous 2D coordinates (no IDs, just float pairs)
        coords_path = output_dir / f"umap_{region}_{hemi}_coords.npy"
        np.save(coords_path, coords_2d.astype(np.float32))
        print(f"  Saved coords: {coords_path} {coords_2d.shape}")

    print("\nDone. Artifacts contain no subject identifiers.")


def main():
    parser = argparse.ArgumentParser(
        description="Train UMAP on UKBioBank reference embeddings."
    )
    parser.add_argument(
        "embeddings_dir",
        help="Path to directory with UKB40 embedding CSVs.",
    )
    parser.add_argument(
        "--output", default="reference_data",
        help="Output directory for UMAP artifacts (default: reference_data/).",
    )
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train_and_save(
        args.embeddings_dir, args.output,
        n_neighbors=args.n_neighbors, min_dist=args.min_dist,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
