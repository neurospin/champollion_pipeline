#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to collect per-region full_embeddings.csv files into a single output folder.
"""

import os
import shutil
from os.path import exists, join

from champollion_utils.script_builder import ScriptBuilder


class PutTogetherEmbeddings(ScriptBuilder):
    """Collect per-region embeddings from {dataset}embeddings/ into a single folder."""

    def __init__(self):
        super().__init__(
            script_name="put_together_embeddings",
            description="Collect per-region embeddings into a single output folder.",
        )
        (
            self.add_argument(
                "embeddings_source",
                type=str,
                help="Path to the {dataset}embeddings/ directory produced by generate_embeddings.",
            ).add_required_argument("--output_path", "Folder where collected embeddings will be written.")
        )

    def run(self):
        """Copy {region}/full_embeddings.csv → {output_path}/{region}_embeddings.csv for all regions."""
        source = self.args.embeddings_source
        output_path = self.args.output_path

        print(f"put_together_embeddings/embeddings_source: {source}")
        print(f"put_together_embeddings/output_path: {output_path}")

        os.makedirs(output_path, exist_ok=True)

        copied = 0
        for region in sorted(os.listdir(source)):
            region_dir = join(source, region)
            if not os.path.isdir(region_dir):
                continue
            src_csv = join(region_dir, "full_embeddings.csv")
            if not exists(src_csv):
                print(f"  [skip] {region}: no full_embeddings.csv")
                continue
            dst_csv = join(output_path, f"{region}_embeddings.csv")
            shutil.copyfile(src_csv, dst_csv)
            copied += 1

        print(f"{copied} embeddings copied to {output_path}")
        if copied == 0:
            print(f"WARNING: no embeddings found in {source}")
        return 0


def main():
    """Main entry point."""
    script = PutTogetherEmbeddings()
    return script.build().print_args().run()


if __name__ == "__main__":
    exit(main())
