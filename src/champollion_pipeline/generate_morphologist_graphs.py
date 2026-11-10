#!/usr/bin/env python3
"""
Script to generate graphs with morphologist from the user's raw data.
"""

import logging
from datetime import datetime
from os import getcwd
from os.path import basename, join
from typing import NamedTuple

from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.derivatives_layout import (
    DERIVATIVES_DIRNAME,
    MORPHOLOGIST_DERIVATIVES_FOLDER,
    compute_run_log_derivatives_dir,
    compute_run_log_path,
)
from champollion_pipeline.process_setup import init_pipeline_process
from champollion_pipeline.utils.input_kind import T1_SUFFIXES, find_t1_files
from champollion_pipeline.utils.nifti_check import check_nifti_input
from champollion_pipeline.utils.qc_report import ACQUISITION_COLUMN, write_qc_tsv

INPUT_QC_FILENAME = "input_qc.tsv"
DUPLICATE_SUBJECT = "duplicate_subject"
NO_BIDS_SUBJECT = "no_bids_subject"
NO_BIDS_ENTITY = "no_bids_entity"
DEFAULT_ACQUISITION = "default_acquisition"
T1W_ENDING = "_T1w"  # input_kind's _BIDS_T1_ENDING is private
ENTITY_KEY_PARTS = ("ses", "acq", "run")  # cortical_tiles skip_list.SESSION_PARTS: stage 1 imports no cortical_tiles

logger = logging.getLogger(__name__)


def _participant_id(name: str) -> str:
    """Return ``name`` without its first matching T1_SUFFIXES suffix; the whole name if that leaves nothing."""
    for suffix in T1_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)] or name
    return name


def _is_ascii_alnum(text: str) -> bool:
    """True when ``text`` is non-empty and only ASCII letters or digits."""
    return text.isascii() and text.isalnum()


def _bids_entity(entity: str) -> tuple[str, str] | None:
    """Return (key, label) of a ``key-label`` BIDS entity, both ASCII letters or digits, else None."""
    key, sep, label = entity.partition("-")
    if sep != "-" or not _is_ascii_alnum(key) or not _is_ascii_alnum(label):
        return None
    return key, label


class _InputKey(NamedTuple):
    """QC key of one input file: participant_id, acquisition folder ("" when none) and refusal reason ("" when none)."""

    participant_id: str
    acquisition: str
    refusal: str


def _bids_name_text(name: str, participant_id: str) -> str:
    """Text between sub-<label>_ and the _T1w ending or the file extension; "" when there is none.

    A lone ``_`` (``sub-01__T1w``) stays as text, so that name is refused rather than read as having none.
    """
    rest = _participant_id(name).removeprefix(participant_id).removesuffix(T1W_ENDING)
    return rest.removeprefix("_") or rest


def _compute_bids_key(name: str) -> _InputKey:
    """QC key of a BIDS-named file: sub-<label>, its acquisition folder, and its refusal.

    The acquisition folder is the file name text after sub-<label>_, verbatim, or default_acquisition when there is
    none. A name not starting with sub-<label> is refused as no_bids_subject (empty folder); a text holding a part
    other than a ``key-value`` entity is refused as no_bids_entity (the text kept as folder).
    """
    participant_id = _bids_participant_id(name)
    if participant_id is None:
        return _InputKey(_participant_id(name), "", NO_BIDS_SUBJECT)
    text = _bids_name_text(name, participant_id)
    if not text:
        return _InputKey(participant_id, DEFAULT_ACQUISITION, "")
    if all(_bids_entity(part) for part in text.split("_")):
        return _InputKey(participant_id, text, "")
    return _InputKey(participant_id, text, NO_BIDS_ENTITY)


def _compute_entity_key(acquisition: str) -> tuple[str | None, ...]:
    """Values of the ses, acq and run entities of an acquisition folder, None for an absent one.

    The first value of each key counts, so an absent entity equals only an absent one and the order of the
    entities in the folder text does not count.
    """
    values: dict[str, str] = {}
    for part in acquisition.split("_"):
        entity = _bids_entity(part)
        if entity is not None:
            values.setdefault(*entity)
    return tuple(values.get(key) for key in ENTITY_KEY_PARTS)


def _bids_participant_id(name: str) -> str | None:
    """Return ``sub-<label>`` of a BIDS file name, the label ended by ``_`` or ``.``, else None."""
    subject = _bids_entity(name.partition(".")[0].split("_")[0])
    if subject is None or subject[0] != "sub":
        return None
    return f"sub-{subject[1]}"


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
                "Input is BIDS-named: morphologist-cli gets subject sub-<label> and, as acquisition folder, the file "
                "name text between sub-<label> and _T1w (default_acquisition when none).",
            )
        )

    def _get_input_files(self):
        """Return the T1 files of the input directory, as paths relative to it."""
        return find_t1_files(self.args.input)

    def _input_key(self, name: str) -> _InputKey:
        """QC key of one input file name, folders excluded."""
        if self.args.bids:
            return _compute_bids_key(name)
        return _InputKey(_participant_id(name), "", "")

    def _attribute_args(self, kept: list[str]) -> list[str]:
        """capsul -a arguments naming the subject and acquisition folder of the kept files under --bids, else none.

        One file gets plain values, several get lists in ``kept`` order (capsul iterates over them). Only kept files
        reach here, so each value holds only ASCII letters, digits, ``-`` and ``_``.
        """
        if not self.args.bids:
            return []
        keys = [self._input_key(basename(name)) for name in kept]
        if len(keys) == 1:
            return ["-a", f"subject={keys[0].participant_id}", "-a", f"acquisition={keys[0].acquisition}"]
        subjects = [key.participant_id for key in keys]
        acquisitions = [key.acquisition for key in keys]
        return ["-a", f"subject={subjects!r}", "-a", f"acquisition={acquisitions!r}"]

    @staticmethod
    def _find_colliding_scans(groups: dict[tuple[str, str], list[tuple[str, str]]]) -> set[tuple[str, str]]:
        """Scans of one participant whose different acquisition folders hold equal ses, acq and run entities.

        Only scans none of whose files is refused count (a refused one goes out on its own reason).
        """
        index: dict[tuple[str, tuple[str | None, ...]], list[tuple[str, str]]] = {}
        for scan, files in groups.items():
            if not any(refusal for _, refusal in files):
                index.setdefault((scan[0], _compute_entity_key(scan[1])), []).append(scan)
        return {scan for scans in index.values() if len(scans) > 1 for scan in scans}

    def _compute_scan_reasons(self, files: list[tuple[str, str]], collides: bool) -> tuple[str, ...]:
        """QC reasons of one scan from its (relative path, refusal) files; empty when it is kept."""
        if len(files) > 1 or collides:
            return (DUPLICATE_SUBJECT,)
        name, refusal = files[0]
        if refusal:
            return (refusal,)
        return tuple(check_nifti_input(join(self.args.input, name)))

    def _run_input_qc(self, input_files):
        """Check each input file, write input_qc.tsv, log each exclusion, and return the kept relative paths."""
        groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for name in input_files:
            key = self._input_key(basename(name))
            groups.setdefault((key.participant_id, key.acquisition), []).append((name, key.refusal))
        colliding = self._find_colliding_scans(groups) if self.args.bids else set()

        rows = []
        kept = set()
        for scan, files in sorted(groups.items()):
            reasons = self._compute_scan_reasons(files, scan in colliding)
            row = {"participant_id": scan[0], "qc": 0 if reasons else 1, "reason": ";".join(reasons)}
            if self.args.bids:
                row[ACQUISITION_COLUMN] = scan[1]
            rows.append(row)
            if not reasons:
                kept.add(files[0][0])

        qc_path = join(self.args.output, DERIVATIVES_DIRNAME, MORPHOLOGIST_DERIVATIVES_FOLDER, INPUT_QC_FILENAME)
        write_qc_tsv(qc_path, rows, extra_columns=(ACQUISITION_COLUMN,) if self.args.bids else ())
        for row in rows:
            if row["qc"] == 0:
                logger.error("input QC excluded %s: %s", row["participant_id"], row["reason"])
        if not kept:
            logger.fail("no input file passed the input QC, see %s", qc_path)
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
        logger.trace(f"Current working directory: {getcwd()}")

        # Validate paths
        if not self.validate_paths([self.args.input, self.args.output]):
            raise ValueError("generate_morphologist_graphs.py: Please input valid paths.")

        input_files = self._get_input_files()
        kept = self._run_input_qc(input_files)
        if not kept:
            return 1

        # Build command
        # Nonoverlap is the format with a t1mri folder, with or without --bids: left alone, morphologist-cli
        # picks morphologist-bids-2.0, which has none, so stages 2+ would find no scan. With --bids the
        # subject and acquisition attributes place each file at sub-<label>/t1mri/<acquisition folder>.
        io_format = "morphologist-auto-nonoverlap-1.0"
        cmd = [
            "morphologist-cli",
            *kept,
            self.args.output,
            "--",
            "--if",
            io_format,
            "--of",
            io_format,
            *self._attribute_args(kept),
        ]

        # Add parallel processing flag (capsul options come first)
        if self.args.parallel:
            cmd.append("--swf")

        # Skip sulcal recognition by default (process parameters come last)
        # Must disable BOTH sulci_labelling and sulcal_morphometry (which depends on it)
        # CAPSUL pipeline_steps do not cascade dependencies automatically
        # One argv element, no quotes: without a shell nothing strips them (NTUI: no shell sees user names).
        if not self.args.enable_sulcal_recognition:
            cmd.append("pipeline_steps=sulci_labelling=False, sulcal_morphometry=False")

        return self.execute_command(cmd, shell=False, cwd=self.args.input)


def main():
    """Main entry point."""
    init_pipeline_process()
    script = GenerateMorphologistGraphs()
    return script.main()


if __name__ == "__main__":
    exit(main())
