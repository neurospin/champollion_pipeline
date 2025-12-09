# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Wrapper script to generate embeddings and train classifiers.
# This script manages user inputs and calls embeddings_pipeline.py.
# """

# import os
# import sys
# import pprint
# import argparse
# from os.path import dirname, join
# from subprocess import run, check_call

# def run_embeddings_pipeline(
#     dir_path: str,
#     datasets_root: str,
#     dataset_localization: str = "neurospin",
#     datasets: list = ["toto"],
#     labels: list = ["Sex"],
#     short_name: str = None,
#     classifier_name: str = "svm",
#     overwrite: bool = False,
#     embeddings_only: bool = False,
#     use_best_model: bool = False,
#     subsets: list = ["full"],
#     epochs: list = [None],
#     split: str = "random",
#     cv: int = 5,
#     splits_basedir: str = "",
#     idx_region_evaluation: int = None,
#     verbose: bool = False
# ) -> int:
#     """Run the embeddings pipeline with the given arguments."""
#     print(f"generate_embedding.py/dir_path: {dir_path}")
#     print(f"generate_embedding.py/datasets_root: {datasets_root}")

#     # Build the command to call embeddings_pipeline.py
#     cmd = [
#         sys.executable,
#         join(dirname(__file__), "embeddings_pipeline.py"),
#         f"--dir_path={dir_path}",
#         f"--datasets_root={datasets_root}",
#         f"--dataset_localization={dataset_localization}"
#     ]

#     # Add optional arguments if they differ from defaults
#     if datasets != ["toto"]:
#         for dataset in datasets:
#             cmd.append(f"--datasets={dataset}")

#     if labels != ["Sex"]:
#         for label in labels:
#             cmd.append(f"--labels={label}")

#     if short_name is not None:
#         cmd.append(f"--short_name={short_name}")

#     if classifier_name != "svm":
#         cmd.append(f"--classifier_name={classifier_name}")

#     if overwrite:
#         cmd.append("--overwrite")

#     if embeddings_only:
#         cmd.append("--embeddings_only")

#     if use_best_model:
#         cmd.append("--use_best_model")

#     if subsets != ["full"]:
#         for subset in subsets:
#             cmd.append(f"--subsets={subset}")

#     # Handle epochs
#     for epoch in epochs:
#         if epoch is None:
#             cmd.append("--epochs=None")
#         else:
#             cmd.append(f"--epochs={epoch}")

#     if split != "random":
#         cmd.append(f"--split={split}")

#     if cv != 5:
#         cmd.append(f"--cv={cv}")

#     if splits_basedir != "":
#         cmd.append(f"--splits_basedir={splits_basedir}")

#     if idx_region_evaluation is not None:
#         cmd.append(f"--idx_region_evaluation={idx_region_evaluation}")

#     if verbose:
#         cmd.append("--verbose")

#     # Print the command for debugging
#     print("Running command:")
#     print(" ".join(cmd))

#     # Execute the command
#     try:
#         return check_call(cmd)
#     except Exception as e:
#         print(f"Error running command: {e}")
#         return 1

# def main() -> int:
#     """Main function to handle user inputs and call the pipeline script."""
#     parser = argparse.ArgumentParser(
#         description="Generate embeddings and train classifiers for deep learning models."
#     )

#     # Required arguments
#     parser.add_argument(
#         "dir_path",
#         type=str,
#         help="Path to the directory containing model folders."
#     )
#     parser.add_argument(
#         "datasets_root",
#         type=str,
#         help="Root path to the dataset YAML configs."
#     )

#     # Optional arguments with defaults
#     parser.add_argument(
#         "--dataset_localization",
#         type=str,
#         default="neurospin",
#         help="Key for dataset localization (default: 'neurospin')."
#     )
#     parser.add_argument(
#         "--datasets",
#         type=str,
#         nargs="+",
#         default=["toto"],
#         help="List of dataset names (default: ['toto'])."
#     )
#     parser.add_argument(
#         "--labels",
#         type=str,
#         nargs="+",
#         default=["Sex"],
#         help="List of labels (default: ['Sex'])."
#     )
#     parser.add_argument(
#         "--short_name",
#         type=str,
#         default=None,
#         help="Custom output folder name (default: derived from datasets)."
#     )
#     parser.add_argument(
#         "--classifier_name",
#         type=str,
#         default="svm",
#         help="Classifier name (default: 'svm')."
#     )
#     parser.add_argument(
#         "--overwrite",
#         action="store_true",
#         help="Overwrite existing embeddings (default: False)."
#     )
#     parser.add_argument(
#         "--embeddings_only",
#         action="store_true",
#         help="Only compute embeddings (skip classifiers, default: False)."
#     )
#     parser.add_argument(
#         "--use_best_model",
#         action="store_true",
#         help="Use the best model saved during training (default: False)."
#     )
#     parser.add_argument(
#         "--subsets",
#         type=str,
#         nargs="+",
#         default=["full"],
#         help="Subsets of data to train on (default: ['full'])."
#     )
#     parser.add_argument(
#         "--epochs",
#         type=str,
#         nargs="+",
#         default=["None"],
#         help="List of epochs to evaluate (default: [None], uses last epoch). Use 'None' for last epoch."
#     )
#     parser.add_argument(
#         "--split",
#         type=str,
#         default="random",
#         help="Splitting strategy ('random' or 'custom', default: 'random')."
#     )
#     parser.add_argument(
#         "--cv",
#         type=int,
#         default=5,
#         help="Number of cross-validation folds (default: 5)."
#     )
#     parser.add_argument(
#         "--splits_basedir",
#         type=str,
#         default="",
#         help="Directory for custom splits (default: None)."
#     )
#     parser.add_argument(
#         "--idx_region_evaluation",
#         type=int,
#         default=None,
#         help="Index of the region to evaluate (for multi-head models, default: None)."
#     )
#     parser.add_argument(
#         "--verbose",
#         action="store_true",
#         help="Enable verbose output (default: False)."
#     )

#     args = parser.parse_args()

#     # Convert epochs to proper format
#     epochs = []
#     for epoch in args.epochs:
#         if epoch.lower() == "none":
#             epochs.append(None)
#         else:
#             epochs.append(int(epoch))

#     # Call the pipeline function
#     return run_embeddings_pipeline(
#         dir_path=args.dir_path,
#         datasets_root=args.datasets_root,
#         dataset_localization=args.dataset_localization,
#         datasets=args.datasets,
#         labels=args.labels,
#         short_name=args.short_name,
#         classifier_name=args.classifier_name,
#         overwrite=args.overwrite,
#         embeddings_only=args.embeddings_only,
#         use_best_model=args.use_best_model,
#         subsets=args.subsets,
#         epochs=epochs,
#         split=args.split,
#         cv=args.cv,
#         splits_basedir=args.splits_basedir,
#         idx_region_evaluation=args.idx_region_evaluation,
#         verbose=args.verbose
#     )

# if __name__ == "__main__":
#     sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper script to generate embeddings and train classifiers.
This script manages user inputs and calls embeddings_pipeline.py.
"""

import os
import sys
import argparse

# Add the directory containing embeddings_pipeline.py to the Python path
sys.path.append(os.path.dirname(os.path.abspath("../../champollion_V1/contrastive/utils/")))

# Import the embeddings_pipeline function
from embeddings_pipeline import embeddings_pipeline

def main() -> None:
    """Main function to handle user inputs and call the pipeline script."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings and train classifiers for deep learning models."
    )

    # Required arguments
    parser.add_argument(
        "dir_path",
        type=str,
        help="Path to the directory containing model folders."
    )
    parser.add_argument(
        "datasets_root",
        type=str,
        help="Root path to the dataset YAML configs."
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

    # Call the embeddings_pipeline function directly
    embeddings_pipeline(
        dir_path=args.dir_path,
        dataset_localization=args.dataset_localization,
        datasets_root=args.datasets_root,
        datasets=args.datasets,
        idx_region_evaluation=args.idx_region_evaluation,
        labels=args.labels,
        short_name=args.short_name,
        classifier_name=args.classifier_name,
        overwrite=args.overwrite,
        embeddings=True,  # Hardcoded as in the original function call
        embeddings_only=args.embeddings_only,
        use_best_model=args.use_best_model,
        subsets=args.subsets,
        epochs=epochs,
        split=args.split,
        cv=args.cv,
        splits_basedir=args.splits_basedir if args.splits_basedir else None,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
