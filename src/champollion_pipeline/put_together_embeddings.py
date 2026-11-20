#!/usr/bin/env python3
"""
Script to check per-region full_embeddings.csv files and collect them into a single output folder.
"""

import logging
import os
import shutil
from datetime import datetime
from os.path import exists, isdir, join

from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.derivatives_layout import DEFAULT_REGIONS, compute_region_model_names, compute_run_log_path
from champollion_pipeline.models.embeddings_qc import (
    EMBEDDINGS_FILENAME,
    OUTPUT_QC_FILENAME,
    WIDTH_SIDECAR_FILENAME,
    compute_excluded_subjects,
    compute_region_copy_path,
    fetch_embeddings_ids,
    fetch_mixed_regions,
    fetch_model_files_qc_rows,
    fetch_model_ids,
    fetch_region_copies_qc_rows,
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


def _compute_worst_status(first: int, second: int) -> int:
    """The graver of two statuses: FAILED_STATUS, then PARTIAL_STATUS, then 0."""
    for status in (FAILED_STATUS, PARTIAL_STATUS):
        if status in (first, second):
            return status
    return 0


def _compute_region_folder(source: str, region: str, model_id: str) -> str:
    """The folder of ``region`` below ``source`` that serves ``model_id``: the flat region folder, else <region>/<id>.

    An empty ``model_id`` is a source with no model id: the region folder.
    """
    if not model_id or exists(join(source, region, EMBEDDINGS_FILENAME)):
        return region
    return f"{region}/{model_id}"


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
                "--model",
                type=str,
                default="",
                help=(
                    "Model id to combine: only <output_path>/<model id>/ is written. By default every model id "
                    "found below the region folders gets its own folder; a source with no model id keeps the flat "
                    "<output_path>/<region>_embeddings.csv."
                ),
            )
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
        """Check the region files, copy them to {output_path}/[{model id}/]{region}_embeddings.csv, check the copies."""
        source = self.args.embeddings_source
        output_path = self.args.output_path

        logger.trace(f"put_together_embeddings/embeddings_source: {source}")
        logger.trace(f"put_together_embeddings/output_path: {output_path}")

        regions = self._fetch_source_regions(source)
        refused = self._refuse_layout(source, regions)
        if refused:
            return refused
        os.makedirs(output_path, exist_ok=True)
        model_ids = fetch_model_ids(source, regions)
        if not model_ids:
            return self._run_model(source, output_path, "")
        status = 0
        for model_id in [self.args.model] if self.args.model else model_ids:
            status = _compute_worst_status(status, self._run_model(source, join(output_path, model_id), model_id))
        return status

    def _fetch_source_regions(self, source: str) -> list[str]:
        """The sorted region folders of ``source``, the --regions ones when given."""
        named = self.args.regions
        if not isdir(source):
            return []
        return [region for region in sorted(os.listdir(source)) if not named or region in named]

    def _refuse_layout(self, source: str, regions: list[str]) -> int:
        """FAILED_STATUS, after one ERROR, for a region holding both forms or a --model matching no id; else 0."""
        mixed = fetch_mixed_regions(source, regions)
        if mixed:
            folders = ", ".join(join(source, region) for region in mixed)
            logger.error(f"a region folder holds {EMBEDDINGS_FILENAME} beside model id folders: {folders}")
            return FAILED_STATUS
        model_ids = fetch_model_ids(source, regions)
        if self.args.model and self.args.model not in model_ids:
            listed = ", ".join(model_ids) or "none"
            logger.error(f"--model {self.args.model} matches no model id of {source}; model ids: {listed}")
            return FAILED_STATUS
        return 0

    def _run_model(self, source: str, folder: str, model_id: str) -> int:
        """Check, copy and check again the files of ``model_id`` into ``folder``; return the graver QC status."""
        os.makedirs(folder, exist_ok=True)
        self._excluded_subjects = frozenset()
        input_status = self._run_input_qc(source, folder, model_id)
        if input_status == FAILED_STATUS:
            return input_status
        copied = self._copy_embeddings(source, folder, model_id)

        logger.output(f"{copied} embeddings copied to {folder}")
        if copied == 0:
            logger.warning(f"no embeddings found in {source}")
        return self._run_output_qc(folder) or input_status

    def _run_input_qc(self, source: str, output_path: str, model_id: str = "") -> int:
        """Write <output_path>/input_qc.tsv, keep its qc 0 subjects; return the status of _report_qc_rows."""
        regions = list(dict.fromkeys(self.args.regions or compute_region_model_names(DEFAULT_REGIONS)))
        folders = [(region, _compute_region_folder(source, region, model_id)) for region in regions]
        self._region_widths = [(region, self._fetch_region_width(source, region, folder)) for region, folder in folders]
        expected = self._fetch_expected_subjects(source, [folder for _, folder in folders])
        rows = fetch_model_files_qc_rows(
            source,
            [
                (region, folder, width)
                for (region, folder), (_, width) in zip(folders, self._region_widths, strict=True)
            ],
            expected,
        )
        self._excluded_subjects = compute_excluded_subjects(rows)
        return self._report_qc_rows(rows, join(output_path, INPUT_QC_FILENAME), "input")

    def _run_output_qc(self, output_path: str) -> int:
        """Write <output_path>/output_qc.tsv over the copies of the checked regions; return that of _report_qc_rows."""
        input_rows = read_qc_tsv(join(output_path, INPUT_QC_FILENAME))
        rows = fetch_region_copies_qc_rows(output_path, self._region_widths, input_rows)
        return self._report_qc_rows(rows, join(output_path, OUTPUT_QC_FILENAME), "output")

    def _fetch_region_width(self, source: str, region: str, folder: str) -> int | None:
        """The width of the sidecar of ``folder``, else one WARNING and the width of the header of its file."""
        width = read_region_width(join(source, folder))
        if width is not None:
            return width
        logger.warning(f"no usable {WIDTH_SIDECAR_FILENAME} for {region}: width taken from its header")
        return read_header_width(join(source, folder, EMBEDDINGS_FILENAME))

    def _fetch_expected_subjects(self, source: str, folders: list[str]) -> set[str]:
        """The qc 1 subjects of the output_qc.tsv of stage 4, else the IDs of the files of ``folders``."""
        report = join(source, OUTPUT_QC_FILENAME)
        if exists(report):
            return {row[ID_COLUMN] for row in read_qc_tsv(report) if row[QC_COLUMN] == 1}
        return fetch_embeddings_ids(source, folders)

    def _report_qc_rows(self, rows: list[dict[str, object]], qc_path: str, step: str) -> int:
        """Write and log ``rows``; return 0, PARTIAL_STATUS with a qc 0 row, or FAILED_STATUS with no qc 1 row."""
        write_qc_tsv(qc_path, rows)
        n_excluded = log_qc_exclusions(rows, logger, step)
        if n_excluded == len(rows):
            logger.fail(f"no subject passed the embeddings {step} QC, see {qc_path}")
            return FAILED_STATUS
        return PARTIAL_STATUS if n_excluded else 0

    def _copy_embeddings(self, source: str, output_path: str, model_id: str = "") -> int:
        """Copy the files of ``model_id`` of the --regions ones when given, less the qc 0 rows; return the count."""
        copied = 0
        for region in self._fetch_source_regions(source):
            src_csv = join(source, _compute_region_folder(source, region, model_id), EMBEDDINGS_FILENAME)
            if not exists(src_csv):
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
