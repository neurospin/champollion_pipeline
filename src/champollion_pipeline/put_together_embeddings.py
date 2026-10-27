#!/usr/bin/env python3
"""
Script to collect per-region full_embeddings.csv files into a single output folder.
"""

import logging
import os
import shutil
from datetime import datetime
from os.path import exists, join

from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.derivatives_layout import compute_run_log_path
from champollion_pipeline.process_setup import init_pipeline_process

logger = logging.getLogger(__name__)


class PutTogetherEmbeddings(ScriptBuilder):
    """Collect per-region embeddings into a single folder.

    Reads the per-region directory written by generate_embeddings (stage 4),
    <datasets_root>/derivatives/champollion_V1/<masks>/region_embeddings by default.
    """

    def __init__(self):
        super().__init__(
            script_name="put_together_embeddings",
            description="Collect per-region embeddings into a single output folder.",
        )
        (
            self.add_argument(
                "embeddings_source",
                type=str,
                help=(
                    "Per-region embeddings directory produced by generate_embeddings, by default "
                    "<datasets_root>/derivatives/champollion_V1/<masks>/region_embeddings."
                ),
            ).add_required_argument("--output_path", "Folder where collected embeddings will be written.")
        )

    def default_log_path(self) -> str:
        """Run-log path of a bare --log: <output_path>/.log/combine_<stamp>.log."""
        return compute_run_log_path(
            self.args.output_path,
            "combine",
            datetime.now().astimezone(),
        )

    def run(self):
        """Copy {region}/full_embeddings.csv → {output_path}/{region}_embeddings.csv for all regions."""
        source = self.args.embeddings_source
        output_path = self.args.output_path

        logger.trace(f"put_together_embeddings/embeddings_source: {source}")
        logger.trace(f"put_together_embeddings/output_path: {output_path}")

        os.makedirs(output_path, exist_ok=True)

        copied = 0
        for region in sorted(os.listdir(source)):
            region_dir = join(source, region)
            if not os.path.isdir(region_dir):
                continue
            src_csv = join(region_dir, "full_embeddings.csv")
            if not exists(src_csv):
                logger.warning(f"[skip] {region}: no full_embeddings.csv")
                continue
            dst_csv = join(output_path, f"{region}_embeddings.csv")
            shutil.copyfile(src_csv, dst_csv)
            copied += 1

        logger.output(f"{copied} embeddings copied to {output_path}")
        if copied == 0:
            logger.warning(f"no embeddings found in {source}")
        return 0


def main():
    """Main entry point."""
    init_pipeline_process()
    script = PutTogetherEmbeddings()
    return script.main()


if __name__ == "__main__":
    exit(main())
