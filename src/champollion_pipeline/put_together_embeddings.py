#!/usr/bin/env python3
"""
Script to check per-region full_embeddings.csv files and collect them into a single output folder.
"""

import logging
import os
import shutil
from datetime import datetime
from os.path import exists, join

from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.derivatives_layout import DEFAULT_REGIONS, compute_region_model_names, compute_run_log_path
from champollion_pipeline.models.embeddings_qc import (
    EMBEDDINGS_FILENAME,
    OUTPUT_QC_FILENAME,
    WIDTH_SIDECAR_FILENAME,
    compute_excluded_subjects,
    compute_region_copy_path,
    fetch_embeddings_ids,
    fetch_region_copies_qc_rows,
    fetch_region_files_qc_rows,
    read_header_width,
    read_region_width,
    remove_subject_rows,
)
from champollion_pipeline.process_setup import init_pipeline_process
from champollion_pipeline.utils.qc_report import ID_COLUMN, QC_COLUMN, log_qc_exclusions, read_qc_tsv, write_qc_tsv

INPUT_QC_FILENAME = "input_qc.tsv"
FAILED_STATUS = 1
PARTIAL_STATUS = 3  # value of _stage_base.PARTIAL_STATUS, not imported: that module chain imports torch

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
            )
            .add_required_argument("--output_path", "Folder where collected embeddings will be written.")
            .add_argument(
                "--regions",
                type=str,
                nargs="+",
                default=None,
                help=(
                    "Region model names to check and collect (e.g. SC-sylv_left SC-sylv_right). "
                    "By default the 28 default regions in both hemispheres are checked."
                ),
            )
        )
        self._excluded_subjects: frozenset[str] = frozenset()
        self._region_widths: list[tuple[str, int | None]] = []

    def default_log_path(self) -> str:
        """Run-log path of a bare --log: <output_path>/.log/combine_<stamp>.log."""
        return compute_run_log_path(
            self.args.output_path,
            "combine",
            datetime.now().astimezone(),
        )

    def run(self) -> int:
        """Check the region files, copy them to {output_path}/{region}_embeddings.csv, then check the copies."""
        source = self.args.embeddings_source
        output_path = self.args.output_path

        logger.trace(f"put_together_embeddings/embeddings_source: {source}")
        logger.trace(f"put_together_embeddings/output_path: {output_path}")

        os.makedirs(output_path, exist_ok=True)

        input_status = self._run_input_qc(source, output_path)
        if input_status == FAILED_STATUS:
            return input_status
        copied = self._copy_embeddings(source, output_path)

        logger.output(f"{copied} embeddings copied to {output_path}")
        if copied == 0:
            logger.warning(f"no embeddings found in {source}")
        return self._run_output_qc(output_path) or input_status

    def _run_input_qc(self, source: str, output_path: str) -> int:
        """Write <output_path>/input_qc.tsv, keep its qc 0 subjects; return the status of _report_qc_rows."""
        regions = list(dict.fromkeys(self.args.regions or compute_region_model_names(DEFAULT_REGIONS)))
        self._region_widths = [(region, self._fetch_region_width(source, region)) for region in regions]
        rows = fetch_region_files_qc_rows(source, self._region_widths, self._fetch_expected_subjects(source, regions))
        self._excluded_subjects = compute_excluded_subjects(rows)
        return self._report_qc_rows(rows, join(output_path, INPUT_QC_FILENAME), "input")

    def _run_output_qc(self, output_path: str) -> int:
        """Write <output_path>/output_qc.tsv over the copies of the checked regions; return that of _report_qc_rows."""
        input_rows = read_qc_tsv(join(output_path, INPUT_QC_FILENAME))
        rows = fetch_region_copies_qc_rows(output_path, self._region_widths, input_rows)
        return self._report_qc_rows(rows, join(output_path, OUTPUT_QC_FILENAME), "output")

    def _fetch_region_width(self, source: str, region: str) -> int | None:
        """The width of the sidecar of ``region``, else one WARNING and the width of its header."""
        width = read_region_width(join(source, region))
        if width is not None:
            return width
        logger.warning(f"no usable {WIDTH_SIDECAR_FILENAME} for {region}: width taken from its header")
        return read_header_width(join(source, region, EMBEDDINGS_FILENAME))

    def _fetch_expected_subjects(self, source: str, regions: list[str]) -> set[str]:
        """The qc 1 subjects of the output_qc.tsv of stage 4, else the IDs of the region files."""
        report = join(source, OUTPUT_QC_FILENAME)
        if exists(report):
            return {row[ID_COLUMN] for row in read_qc_tsv(report) if row[QC_COLUMN] == 1}
        return fetch_embeddings_ids(source, regions)

    def _report_qc_rows(self, rows: list[dict[str, object]], qc_path: str, step: str) -> int:
        """Write and log ``rows``; return 0, PARTIAL_STATUS with a qc 0 row, or FAILED_STATUS with no qc 1 row."""
        write_qc_tsv(qc_path, rows)
        n_excluded = log_qc_exclusions(rows, logger, step)
        if n_excluded == len(rows):
            logger.fail(f"no subject passed the embeddings {step} QC, see {qc_path}")
            return FAILED_STATUS
        return PARTIAL_STATUS if n_excluded else 0

    def _copy_embeddings(self, source: str, output_path: str) -> int:
        """Copy the region files of ``source``, the --regions ones when given, less the qc 0 rows; return the count."""
        named = self.args.regions
        copied = 0
        for region in sorted(os.listdir(source)):
            src_csv = join(source, region, EMBEDDINGS_FILENAME)
            if not exists(src_csv) or (named and region not in named):
                continue
            dst_csv = compute_region_copy_path(output_path, region)
            shutil.copyfile(src_csv, dst_csv)
            if self._excluded_subjects:
                remove_subject_rows(dst_csv, self._excluded_subjects)
            copied += 1
        return copied


def main():
    """Main entry point."""
    init_pipeline_process()
    script = PutTogetherEmbeddings()
    return script.main()


if __name__ == "__main__":
    exit(main())
