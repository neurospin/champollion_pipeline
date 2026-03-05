#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate UMAP reference model artefacts for all ROIs.

For each (region, hemisphere) pair found in the Champollion_V1 pre-trained
models directory, this script:

  1. Loads the reference-population embeddings CSV from each model directory
     (e.g. ``ukb40_random_embeddings/full_embeddings.csv``).
  2. Concatenates embeddings across model runs for the same ROI.
  3. Fits a UMAP model on the concatenated reference embeddings.
  4. Saves the fitted model and the projected reference coordinates:
       ``umap_{region}_{hemi}.pkl``
       ``umap_{region}_{hemi}_coords.npy``

The output files are consumed by ``generate_snapshots.py`` via
``discover_umap_pairs()`` during UMAP visualisation.

Typical usage::

    pixi run generate-umap-reference \\
        --models_dir /neurospin/dico/.../Champollion_V1_after_ablation \\
        --reference_subpath ukb40_random_embeddings/full_embeddings.csv \\
        --output_dir reference_data/
"""

import os
import os.path as osp
import sys

import joblib
import numpy as np

from champollion_utils.script_builder import ScriptBuilder

# Re-use the model-discovery helpers from champollion_V1 utils
sys.path.insert(
    0,
    osp.abspath(
        osp.join(osp.dirname(__file__), "..", "external",
                 "champollion_V1", "contrastive", "utils")
    ),
)
from put_together_embeddings_files import get_model_paths  # noqa: E402


def _parse_region_hemi(dir_name):
    """Split a directory name like ``FColl-SRh_left`` into (region, hemi).

    Matches the ``_left`` / ``_right`` suffix so that region names containing
    underscores (e.g. ``S.T.s.ter.pf.or._left``) are handled correctly.

    Returns ``(None, None)`` if the name does not end with a recognised suffix.
    """
    for sfx in ("_left", "_right"):
        if dir_name.endswith(sfx):
            return dir_name[: -len(sfx)], sfx[1:]
    return None, None


def generate_umap_reference(
    models_dir,
    reference_subpath,
    output_dir,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
    overwrite=False,
):
    """Train and save UMAP reference artefacts for all ROIs.

    Args:
        models_dir: Path to the Champollion_V1 pre-trained models directory
            (e.g. ``Champollion_V1_after_ablation/``).
        reference_subpath: Subpath within each model directory to the
            reference-population embeddings CSV
            (e.g. ``ukb40_random_embeddings/full_embeddings.csv``).
        output_dir: Directory where ``.pkl`` and ``_coords.npy`` files are
            written.
        n_neighbors: UMAP ``n_neighbors`` parameter.
        min_dist: UMAP ``min_dist`` parameter.
        random_state: Random seed for reproducibility.
        overwrite: Re-generate files that already exist.

    Returns:
        List of (region, hemi) pairs for which artefacts were successfully
        generated.
    """
    import pandas as pd
    import umap as umap_lib

    os.makedirs(output_dir, exist_ok=True)

    models_dir = osp.abspath(models_dir)
    model_paths = get_model_paths(models_dir)
    if not model_paths:
        print(f"No model directories found in: {models_dir}")
        return []

    # Group embedding arrays by (region, hemi)
    embeddings_by_roi = {}  # (region, hemi) -> list of np.ndarray

    for model_path in sorted(model_paths):
        relative = model_path.replace(models_dir.rstrip("/") + "/", "")
        region_hemi = relative.split("/")[0]

        region, hemi = _parse_region_hemi(region_hemi)
        if region is None:
            print(f"  Skipping unrecognised dir name: {region_hemi}")
            continue

        csv_path = osp.join(model_path, reference_subpath)
        if not osp.exists(csv_path):
            print(f"  [{region} {hemi}] Reference CSV not found: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
            X = df.drop(columns=["ID"], errors="ignore").values.astype(np.float32)
        except Exception as e:
            print(f"  [{region} {hemi}] Could not load {csv_path}: {e}")
            continue

        key = (region, hemi)
        embeddings_by_roi.setdefault(key, []).append(X)

    if not embeddings_by_roi:
        print("No reference embeddings found. Check --models_dir and --reference_subpath.")
        return []

    generated = []
    total = len(embeddings_by_roi)
    for i, ((region, hemi), arrays) in enumerate(sorted(embeddings_by_roi.items()), 1):
        model_out = osp.join(output_dir, f"umap_{region}_{hemi}.pkl")
        coords_out = osp.join(output_dir, f"umap_{region}_{hemi}_coords.npy")

        if not overwrite and osp.exists(model_out) and osp.exists(coords_out):
            print(f"  [{i}/{total}] {region} {hemi} — already exists, skipping "
                  f"(use --overwrite to regenerate)")
            generated.append((region, hemi))
            continue

        X_ref = np.vstack(arrays)
        print(f"  [{i}/{total}] {region} {hemi} — fitting UMAP on "
              f"{X_ref.shape[0]:,} points ({X_ref.shape[1]} dims)...")

        reducer = umap_lib.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
        ref_coords = reducer.fit_transform(X_ref)

        joblib.dump(reducer, model_out)
        np.save(coords_out, ref_coords)
        print(f"    Saved: {osp.basename(model_out)}, {osp.basename(coords_out)}")
        generated.append((region, hemi))

    return generated


class GenerateUmapReference(ScriptBuilder):
    """Script to generate UMAP reference artefacts for all ROIs."""

    def __init__(self):
        super().__init__(
            script_name="generate_umap_reference",
            description=(
                "Train and save UMAP reference models (.pkl + _coords.npy) "
                "for all ROIs in the Champollion_V1 models directory."
            ),
        )
        (self
         .add_required_argument(
             "--models_dir",
             "Path to Champollion_V1 pre-trained models directory "
             "(e.g. Champollion_V1_after_ablation/).")
         .add_required_argument(
             "--reference_subpath",
             "Subpath within each model directory to the reference-population "
             "embeddings CSV (e.g. ukb40_random_embeddings/full_embeddings.csv).")
         .add_optional_argument(
             "--output_dir",
             "Directory where .pkl and _coords.npy files are written.",
             default="reference_data/")
         .add_optional_argument(
             "--n_neighbors",
             "UMAP n_neighbors parameter.",
             default=15, type_=int)
         .add_optional_argument(
             "--min_dist",
             "UMAP min_dist parameter.",
             default=0.1, type_=float)
         .add_optional_argument(
             "--random_state",
             "Random seed for reproducibility.",
             default=42, type_=int)
         .add_flag(
             "--overwrite",
             "Re-generate artefacts even if output files already exist."))

    def run(self) -> int:
        print(f"Models directory : {self.args.models_dir}")
        print(f"Reference subpath: {self.args.reference_subpath}")
        print(f"Output directory : {self.args.output_dir}")
        print(f"UMAP params      : n_neighbors={self.args.n_neighbors}, "
              f"min_dist={self.args.min_dist}, random_state={self.args.random_state}")
        if self.args.overwrite:
            print("Overwrite mode   : ON")
        print()

        generated = generate_umap_reference(
            models_dir=self.args.models_dir,
            reference_subpath=self.args.reference_subpath,
            output_dir=self.args.output_dir,
            n_neighbors=self.args.n_neighbors,
            min_dist=self.args.min_dist,
            random_state=self.args.random_state,
            overwrite=self.args.overwrite,
        )

        print(f"\nDone: {len(generated)} UMAP reference model(s) available.")
        return 0


def main():
    script = GenerateUmapReference()
    return script.build().print_args().run()


if __name__ == "__main__":
    sys.exit(main())
