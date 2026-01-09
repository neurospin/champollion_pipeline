#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper script to generate embeddings and train classifiers.
This script manages user inputs and calls embeddings_pipeline.py using subprocess.
"""

import os

from champollion_utils.src.champollion_utils.script_builder import ScriptBuilder


class GenerateEmbeddings(ScriptBuilder):
    """Script for generating embeddings and training classifiers."""

    def __init__(self):
        super().__init__(
            script_name="generate_embeddings",
            description="Generate embeddings and train classifiers for deep learning models."
        )
        # Configure arguments using method chaining
        (self.add_argument("models_path", type=str,
                           help="Path to the directory containing model folders.")
         .add_argument("dataset_localization", type=str,
                       help="Key for dataset localization.")
         .add_argument("datasets_root", type=str,
                       help="Root path to the dataset YAML configs.")
         .add_argument("short_name", type=str,
                       help="Name of the directory where to store both embeddings and aucs.")
         .add_argument("--datasets", type=str, nargs="+",
                       default=["toto"], help="List of dataset names (default: ['toto']).")
         .add_argument("--labels", type=str, nargs="+",
                       default=["Sex"], help="List of labels (default: ['Sex']).")
         .add_optional_argument("--classifier_name", "Classifier name.",
                                default="svm")
         .add_flag("--overwrite", "Overwrite existing embeddings.")
         .add_flag("--embeddings_only",
                   "Only compute embeddings (skip classifiers).")
         .add_flag("--use_best_model",
                   "Use the best model saved during training.")
         .add_argument("--subsets", type=str, nargs="+",
                       default=["full"], help="Subsets of data to train on (default: ['full']).")
         .add_argument("--epochs", type=str, nargs="+",
                       default=["None"], help="List of epochs to evaluate (default: [None]).")
         .add_optional_argument("--split",
                                "Splitting strategy ('random' or 'custom').", default="random")
         .add_optional_argument("--cv",
                                "Number of cross-validation folds.",
                                default=5,
                                type_=int)
         .add_optional_argument("--splits_basedir",
                                "Directory for custom splits.", default="")
         .add_optional_argument("--idx_region_evaluation",
                                "Index of region to evaluate (multi-head models).",
                                default=None,
                                type_=int)
         .add_flag("--verbose", "Enable verbose output."))

    def run(self):
        """Execute the embeddings pipeline script."""
        local_dir = os.getcwd()
        champollion_dir = "../../champollion_V1/contrastive/"

        os.chdir(champollion_dir)

        # Use build_command to construct the command
        defaults = {
            "datasets": ["toto"],
            "labels": ["Sex"],
            "classifier_name": "svm",
            "overwrite": False,
            "embeddings_only": False,
            "use_best_model": False,
            "subsets": ["full"],
            "epochs": ["None"],
            "split": "random",
            "cv": 5,
            "splits_basedir": "",
            "idx_region_evaluation": None,
            "verbose": False
        }

        cmd = self.build_command(
            script_path="evaluation/embeddings_pipeline.py",
            required_args=[
                "models_path",
                "dataset_localization",
                "datasets_root",
                "short_name"
                ],
            defaults=defaults
        )

        result = self.execute_command(cmd, shell=False)

        os.chdir(local_dir)

        return result


def main():
    """Main entry point."""
    script = GenerateEmbeddings()
    return script.build().print_args().run()


if __name__ == "__main__":
    exit(main())
