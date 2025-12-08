#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper script to generate embeddings and train classifiers.
This script manages user inputs and calls embeddings_pipeline.py.
"""

import pprint
from argparse import ArgumentParser
from os import chdir
from os import getcwd
from os.path import dirname
from os.path import join
from subprocess import run
from typing import List, Optional

def run_embeddings_pipeline(
    config_path: str,
    dir_path: str,
    datasets_root: str,
    dataset_localization: str = "neurospin",
    datasets: List[str] = ["toto"],
    labels: List[str] = ["Sex"],
    short_name: Optional[str] = None,
    classifier_name: str = "svm",
    overwrite: bool = False,
    embeddings_only: bool = False,
    use_best_model: bool = False,
    subsets: List[str] = ["full"],
    epochs: List[Optional[int]] = [None],
    split: str = "random",
    cv: int = 5,
    splits_basedir: str = "",
    idx_region_evaluation: Optional[int] = None,
    verbose: bool = False,
    njobs: Optional[int] = None
) -> None:
    """Run the embeddings pipeline with the given arguments."""
    print(f"generate_embedding.py/config_path: {config_path}")
    print(f"generate_embedding.py/dir_path: {dir_path}")
    print(f"generate_embedding.py/datasets_root: {datasets_root}")

    local_dir: str = getcwd()

    # Build the command to call embeddings_pipeline.py
    cmd = [
        "python3",
        join(dirname(__file__), "embeddings_pipeline.py"),
        f"--config_path={config_path}",
        f"--dir_path={dir_path}",
        f"--datasets_root={datasets_root}",
        f"--dataset_localization={dataset_localization}"
    ]

    # Add optional arguments
    if datasets != ["toto"]:
        for dataset in datasets:
            cmd.append(f"--datasets={dataset}")

    if labels != ["Sex"]:
        for label in labels:
            cmd.append(f"--labels={label}")

    if short_name is not None:
        cmd.append(f"--short_name={short_name}")

    if classifier_name != "svm":
        cmd.append(f"--classifier_name={classifier_name}")

    if overwrite:
        cmd.append("--overwrite")

    if embeddings_only:
        cmd.append("--embeddings_only")

    if use_best_model:
        cmd.append("--use_best_model")

    if subsets != ["full"]:
        for subset in subsets:
            cmd.append(f"--subsets={subset}")

    # Handle epochs
    for epoch in epochs:
        if epoch is None:
            cmd.append("--epochs=None")
        else:
            cmd.append(f"--epochs={epoch}")

    if split != "random":
        cmd.append(f"--split={split}")

    if cv != 5:
        cmd.append(f"--cv={cv}")

    if splits_basedir != "":
        cmd.append(f"--splits_basedir={splits_basedir}")

    if idx_region_evaluation is not None:
        cmd.append(f"--idx_region_evaluation={idx_region_evaluation}")

    if verbose:
        cmd.append("--verbose")

    # Print the command for debugging
    print("Running command:")
    print(" ".join(cmd))

    # Execute the command
    run(" ".join(cmd), shell=True, executable="/bin/bash")

    # Return to the original directory
    chdir(local_dir)

def main() -> None:
    """Main function to handle user inputs and call the pipeline script."""
    parser = ArgumentParser(
        prog="generate_embedding",
        description="Generate embeddings and train classifiers for deep learning models."
    )

    # Required arguments
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the directory containing classifier configs (e.g., svm.yaml, logistic.yaml)."
    )
    parser.add_argument(
        "--dir_path",
        type=str,
        required=True,
        help="Path to the directory containing model folders."
    )
    parser.add_argument(
        "--datasets_root",
        type=str,
        required=True,
        help="Root path to the dataset YAML configs (e.g., '/path/to/champollion_config_data')."
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--dataset_localization",
        type=str,
        default="neurospin",
        help="Key for dataset localization (default: 'neurospin')."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["toto"],
        help="List of dataset names (default: ['toto'])."
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=["Sex"],
        help="List of labels (default: ['Sex'])."
    )
    parser.add_argument(
        "--short_name",
        type=str,
        default=None,
        help="Custom output folder name (default: derived from datasets)."
    )
    parser.add_argument(
        "--classifier_name",
        type=str,
        default="svm",
        help="Classifier name (default: 'svm')."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embeddings (default: False)."
    )
    parser.add_argument(
        "--embeddings_only",
        action="store_true",
        help="Only compute embeddings (skip classifiers, default: False)."
    )
    parser.add_argument(
        "--use_best_model",
        action="store_true",
        help="Use the best model saved during training (default: False)."
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=["full"],
        help="Subsets of data to train on (default: ['full'])."
    )
    parser.add_argument(
        "--epochs",
        type=str,
        nargs="+",
        default=["None"],
        help="List of epochs to evaluate (default: [None], uses last epoch). Use 'None' for last epoch."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="random",
        help="Splitting strategy ('random' or 'custom', default: 'random')."
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)."
    )
    parser.add_argument(
        "--splits_basedir",
        type=str,
        default="",
        help="Directory for custom splits (default: None)."
    )
    parser.add_argument(
        "--idx_region_evaluation",
        type=int,
        default=None,
        help="Index of the region to evaluate (for multi-head models, default: None)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (default: False)."
    )

    args = parser.parse_args()

    # Convert epochs to proper format
    epochs = []
    for epoch in args.epochs:
        if epoch.lower() == "none":
            epochs.append(None)
        else:
            epochs.append(int(epoch))

    # Call the pipeline function
    run_embeddings_pipeline(
        config_path=args.config_path,
        dir_path=args.dir_path,
        datasets_root=args.datasets_root,
        dataset_localization=args.dataset_localization,
        datasets=args.datasets,
        labels=args.labels,
        short_name=args.short_name,
        classifier_name=args.classifier_name,
        overwrite=args.overwrite,
        embeddings_only=args.embeddings_only,
        use_best_model=args.use_best_model,
        subsets=args.subsets,
        epochs=epochs,
        split=args.split,
        cv=args.cv,
        splits_basedir=args.splits_basedir,
        idx_region_evaluation=args.idx_region_evaluation,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
