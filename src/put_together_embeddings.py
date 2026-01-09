#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to put together embeddings of Champollion_V1 in a single folder.
"""

from os import chdir, getcwd, makedirs
from os.path import dirname, join

from champollion_utils.src.champollion_utils.script_builder import ScriptBuilder


class PutTogetherEmbeddings(ScriptBuilder):
    """Script for combining embeddings from multiple models."""

    def __init__(self):
        super().__init__(
            script_name="put_together_embeddings",
            description="Put together embeddings of Champollion_V1 in a single folder."
        )
        # Configure arguments using method chaining
        (self.add_required_argument("--embeddings_subpath", "Sub-path to embeddings inside model folder.")
         .add_required_argument("--output_path", "Folder where to put all embeddings.")
         .add_optional_argument("--path_models", "Path where all models lie.",
                                default="/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation"))

    def run(self):
        """Execute the put_together_embeddings script."""
        print(f"put_together_embeddings.py/embeddings_subpath: {self.args.embeddings_subpath}")
        print(f"put_together_embeddings.py/path_models: {self.args.path_models}")
        print(f"put_together_embeddings.py/output_path: {self.args.output_path}")

        # Create output directory
        makedirs(self.args.output_path, exist_ok=True)

        # Validate paths
        if not self.validate_paths([self.args.output_path, self.args.path_models]):
            raise ValueError(
                f"put_together_embeddings.py: Please input valid paths. "
                f"Given paths: {self.args.output_path}, {self.args.path_models}"
            )

        local_dir = getcwd()

        # Move to champollion's script location
        champollion_path = join(
            dirname(dirname(local_dir)),
            'champollion_V1/contrastive/utils'
        )
        chdir(champollion_path)

        # Use build_command to construct the command
        defaults = {
            "path_models": "/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation"
        }

        cmd = self.build_command(
            script_path="put_together_embeddings_files.py",
            required_args=["embeddings_subpath", "output_path"],
            defaults=defaults
        )

        result = self.execute_command(cmd, shell=False)

        chdir(local_dir)

        return result


def main():
    """Main entry point."""
    script = PutTogetherEmbeddings()
    return script.build().print_args().run()


if __name__ == "__main__":
    exit(main())
