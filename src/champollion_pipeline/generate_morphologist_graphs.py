#!/usr/bin/env python3
"""
Script to generate graphs with morphologist from the user's raw data.
"""

import logging
from datetime import datetime
from os import chdir, getcwd, listdir
from os.path import basename, isfile, join, splitext

from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.derivatives_layout import (
    DERIVATIVES_DIRNAME,
    MORPHOLOGIST_DERIVATIVES_FOLDER,
    compute_run_log_derivatives_dir,
    compute_run_log_path,
)
from champollion_pipeline.process_setup import init_pipeline_process
from champollion_pipeline.utils.nifti_check import check_nifti_input
from champollion_pipeline.utils.qc_report import write_qc_tsv

# Raw data extensions accepted as stage 1 input; also the suffixes stripped to get a participant_id
INPUT_EXTENSIONS = (".nii.gz", ".nii", ".gz")
INPUT_QC_FILENAME = "input_qc.tsv"
DUPLICATE_SUBJECT = "duplicate_subject"
NO_BIDS_SUBJECT = "no_bids_subject"

logger = logging.getLogger(__name__)


def _participant_id(name: str) -> str:
    """Return ``name`` without its first matching INPUT_EXTENSIONS suffix; the whole name if that leaves nothing."""
    for suffix in INPUT_EXTENSIONS:
        if name.endswith(suffix):
            return name[: -len(suffix)] or name
    return name


def _bids_entity(entity: str) -> tuple[str, str] | None:
    """Return (key, label) of a ``key-label`` BIDS entity with an ASCII alphanumeric label, else None."""
    key, sep, label = entity.partition("-")
    if sep != "-" or not label or not (label.isascii() and label.isalnum()):
        return None
    return key, label


def _bids_participant_id(name: str) -> str | None:
    """Return ``sub-<label>`` (plus ``_ses-<label>`` when that is the next entity) of a BIDS file name, else None."""
    entities = name.partition(".")[0].split("_")
    subject = _bids_entity(entities[0])
    if subject is None or subject[0] != "sub":
        return None
    participant_id = f"sub-{subject[1]}"
    session = _bids_entity(entities[1]) if len(entities) > 1 else None
    if session is not None and session[0] == "ses":
        participant_id += f"_ses-{session[1]}"
    return participant_id


class GenerateMorphologistGraphs(ScriptBuilder):
    """Script for generating graphs with morphologist."""

    def __init__(self):
        super().__init__(
            script_name="morphologist_graphs_generator",
            description="Generating graphs with morphologist from the user raw data.",
        )
        # Configure arguments using method chaining
        (
            self.add_argument("input", help="Absolute path to the user's raw data.")
            .add_argument(
                "output",
                help="Absolute path to the generated graphs from morphologist. "
                "Morphologist will create a $output/derivatives/morphologist-6.0/ "
                "directory for output generations.",
            )
            .add_flag("--parallel", "Enable parallel processing using Soma-Workflow (--swf).")
            .add_flag(
                "--enable-sulcal-recognition",
                "Enable sulcal recognition (adds 10-20 min/subject, disabled by default for embeddings pipeline).",
            )
            .add_flag(
                "--bids",
                "Input is BIDS-named; run morphologist-cli with --if/--of "
                "morphologist-bids-2.0 instead of morphologist-auto-nonoverlap-1.0.",
            )
        )

    def _get_input_files(self):
        """Get list of valid input files."""
        input_files = [
            f.strip('"').strip("'")
            for f in listdir(self.args.input)
            if isfile(join(self.args.input, f)) and splitext(basename(f))[1] in INPUT_EXTENSIONS
        ]

        return input_files

    def _input_key(self, name: str) -> tuple[str, bool]:
        """Return (participant_id, is_no_bids) of one input file name."""
        if not self.args.bids:
            return _participant_id(name), False
        participant_id = _bids_participant_id(name)
        if participant_id is None:
            return _participant_id(name), True
        return participant_id, False

    def _run_input_qc(self, input_files):
        """Check each input file, write input_qc.tsv, log one warning, and return the kept file names."""
        by_subject = {}
        no_bids = set()
        for name in input_files:
            subject, is_no_bids = self._input_key(name)
            by_subject.setdefault(subject, []).append(name)
            if is_no_bids:
                no_bids.add(subject)

        rows = []
        kept = set()
        for subject, names in sorted(by_subject.items()):
            if len(names) > 1:
                reasons = (DUPLICATE_SUBJECT,)
            elif subject in no_bids:
                reasons = (NO_BIDS_SUBJECT,)
            else:
                reasons = check_nifti_input(join(self.args.input, names[0]))
            rows.append({"participant_id": subject, "qc": 0 if reasons else 1, "reason": ";".join(reasons)})
            if not reasons:
                kept.add(names[0])

        qc_path = join(self.args.output, DERIVATIVES_DIRNAME, MORPHOLOGIST_DERIVATIVES_FOLDER, INPUT_QC_FILENAME)
        write_qc_tsv(qc_path, rows)
        excluded = [row["participant_id"] for row in rows if row["qc"] == 0]
        if excluded:
            logger.warning("input QC excluded %d subject(s), see %s: %s", len(excluded), qc_path, ", ".join(excluded))
        return [name for name in input_files if name in kept]

    def default_log_path(self) -> str:
        """Run-log path of a bare --log: <output>/derivatives/morphologist-6.0/.log/morphologist_<stamp>.log."""
        return compute_run_log_path(
            compute_run_log_derivatives_dir("morphologist", self.args.output),
            "morphologist",
            datetime.now().astimezone(),
        )

    def run(self):
        """Execute the morphologist script."""
        print(f"Current working directory: {getcwd()}")

        # Validate paths
        if not self.validate_paths([self.args.input, self.args.output]):
            raise ValueError("generate_morphologist_graphs.py: Please input valid paths.")

        input_files = self._get_input_files()
        kept = self._run_input_qc(input_files)
        if not kept:
            return 0

        local_dir = getcwd()
        chdir(self.args.input)

        # Build command
        # Force nonoverlap format by default to prevent morphologist-cli from auto-detecting
        # BIDS format when filenames contain BIDS entities (_acq-, _run-, etc.) and silently
        # overriding the input format to morphologist-bids-2.0. --bids opts in explicitly.
        io_format = "morphologist-bids-2.0" if self.args.bids else "morphologist-auto-nonoverlap-1.0"
        cmd = [
            "morphologist-cli",
            *kept,
            self.args.output,
            "--",
            "--if",
            io_format,
            "--of",
            io_format,
        ]

        # Add parallel processing flag (capsul options come first)
        if self.args.parallel:
            cmd.append("--swf")

        # Skip sulcal recognition by default (process parameters come last)
        # Must disable BOTH sulci_labelling and sulcal_morphometry (which depends on it)
        # CAPSUL pipeline_steps do not cascade dependencies automatically
        if not self.args.enable_sulcal_recognition:
            cmd.append('pipeline_steps="sulci_labelling=False, sulcal_morphometry=False"')

        result = self.execute_command(cmd, shell=True)

        chdir(local_dir)

        return result


def main():
    """Main entry point."""
    init_pipeline_process()
    script = GenerateMorphologistGraphs()
    return script.main()


if __name__ == "__main__":
    exit(main())
