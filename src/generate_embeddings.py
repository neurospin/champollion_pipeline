#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper script to generate embeddings and train classifiers.
This script manages user inputs and calls embeddings_pipeline.py using subprocess.
"""

import os
import sys
import argparse
from subprocess import run, check_call
from pathlib import Path

def main() -> None:
    """Main function to handle user inputs and call the pipeline script."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings and train classifiers for deep learning models."
    )

    # Required arguments
    parser.add_argument(
        "models_path",
        type=str,
        help="Path to the directory containing model folders."
    )
    parser.add_argument(
        "dataset_localization",
        type=str,
        help="Key for dataset localization."
    )
    parser.add_argument(
        "datasets_root",
        type=str,
        help="Root path to the dataset YAML configs."
    )
    parser.add_argument(
        "short_name",
        type=str,
        help="Name of the directory where to store both embeddings and aucs."
    )

    # Optional arguments with defaults
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

    local_dir: str = os.getcwd()

    champollion_dir: str = "../../champollion_V1/constrastive/"

    os.chdir(champollion_dir)

    # Build the command to call embeddings_pipeline.py
    cmd = [
        sys.executable,
        "evaluation/embeddings_pipeline.py",
        args.models_path,
        args.dataset_localization,
        args.datasets_root,
        args.short_name
    ]

    # Add optional arguments if they differ from defaults
    if args.datasets != ["toto"]:
        for dataset in args.datasets:
            cmd.extend(["--datasets", dataset])

    if args.labels != ["Sex"]:
        for label in args.labels:
            cmd.extend(["--labels", label])

    if args.classifier_name != "svm":
        cmd.extend(["--classifier_name", args.classifier_name])

    if args.overwrite:
        cmd.append("--overwrite")

    if args.embeddings_only:
        cmd.append("--embeddings_only")

    if args.use_best_model:
        cmd.append("--use_best_model")

    if args.subsets != ["full"]:
        for subset in args.subsets:
            cmd.extend(["--subsets", subset])

    # Handle epochs
    for epoch in args.epochs:
        if epoch.lower() == "none":
            cmd.extend(["--epochs", "None"])
        else:
            cmd.extend(["--epochs", epoch])

    if args.split != "random":
        cmd.extend(["--split", args.split])

    if args.cv != 5:
        cmd.extend(["--cv", str(args.cv)])

    if args.splits_basedir != "":
        cmd.extend(["--splits_basedir", args.splits_basedir])

    if args.idx_region_evaluation is not None:
        cmd.extend(["--idx_region_evaluation", str(args.idx_region_evaluation)])

    if args.verbose:
        cmd.append("--verbose")

    # Print the command for debugging
    print("Running command:")
    print(" ".join(cmd))

    # Execute the command
    try:
        check_call(cmd)
    except Exception as e:
        print(f"Error running command: {e}")
        sys.exit(1)

    os.chdir(local_dir)

if __name__ == "__main__":
    main()
