#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk a Morphologist subjects directory and write a labelling QC TSV.

Reports, per subject, whether the labelled graphs that cortical_tiles'
remove_ventricle needs (both hemispheres) exist.

TASK-199 / COMP-LABELLING-QC.
"""

import csv
import glob
import os
import os.path
from collections import Counter

from champollion_utils.script_builder import ScriptBuilder
from joblib import Parallel, delayed

from champollion_pipeline.process_setup import init_pipeline_process

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABELLING_SESSION_DEFAULT = "deepcnn_session_auto"
SIDES = ("L", "R")
QC_COLUMNS = ("participant_id", "qc", "labelled_L", "labelled_R", "reason")
GRAPH_WILDCARD = "*"

REASON_AMBIGUOUS_BIDS_PATH = "ambiguous_bids_path"
REASON_NO_SESSION_DIR = "no_session_dir"
REASON_EMPTY_SESSION_DIR = "empty_session_dir"
REASON_MISSING_BOTH = "missing_both"
REASON_MISSING_L = "missing_L"
REASON_MISSING_R = "missing_R"
REASON_OK = ""

# Sentinel returned by find_graph_dir when a BIDS glob yields more than one match.
GRAPH_DIR_AMBIGUOUS = object()


# ---------------------------------------------------------------------------
# Module-level functions (picklable for joblib worker processes)
# ---------------------------------------------------------------------------


def fetch_subject_names(subjects_dir: str) -> list:
    """Return sorted names of every subdirectory of subjects_dir (files ignored)."""
    return sorted(name for name in os.listdir(subjects_dir) if os.path.isdir(os.path.join(subjects_dir, name)))


def find_graph_dir(subject_dir: str, path_to_graph: str, is_bids: bool):
    """Return the resolved graph directory for a subject.

    Classic layout (or BIDS without '*'): returns os.path.join(subject_dir, path_to_graph)
    without checking existence.

    BIDS layout with '*': globs for matching directories under subject_dir.
    Returns the single match, None (0 matches), or GRAPH_DIR_AMBIGUOUS (>1 match).
    """
    if is_bids and GRAPH_WILDCARD in path_to_graph:
        pattern = os.path.join(glob.escape(subject_dir), path_to_graph)
        matches = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 0:
            return None
        return GRAPH_DIR_AMBIGUOUS
    return os.path.join(subject_dir, path_to_graph)


def make_graph_filename(side: str, subject: str, labelling_session: str) -> str:
    """Return the expected labelled graph filename for one hemisphere.

    Pattern: <S><subject>_<session>.arg
    """
    return f"{side}{subject}_{labelling_session}.arg"


def compute_reason(has_session_dir: bool, is_session_empty: bool, labelled: dict) -> str:
    """Return the first applicable reason code (pure function).

    Precedence:
      no_session_dir → empty_session_dir → missing_both → missing_L → missing_R → "".
    """
    if not has_session_dir:
        return REASON_NO_SESSION_DIR
    if is_session_empty:
        return REASON_EMPTY_SESSION_DIR
    has_l = labelled.get("L", False)
    has_r = labelled.get("R", False)
    if not has_l and not has_r:
        return REASON_MISSING_BOTH
    if not has_l:
        return REASON_MISSING_L
    if not has_r:
        return REASON_MISSING_R
    return REASON_OK


def fetch_subject_row(
    subjects_dir: str, subject: str, path_to_graph: str, labelling_session: str, is_bids: bool
) -> dict:
    """Build one TSV row for a subject; all dict values are strings."""
    subject_dir = os.path.join(subjects_dir, subject)
    graph_dir = find_graph_dir(subject_dir, path_to_graph, is_bids)

    # BIDS glob returned more than one match — ambiguous path.
    if graph_dir is GRAPH_DIR_AMBIGUOUS:
        return {
            "participant_id": subject,
            "qc": "0",
            "labelled_L": "False",
            "labelled_R": "False",
            "reason": REASON_AMBIGUOUS_BIDS_PATH,
        }

    # No graph dir resolved (0 BIDS glob matches).
    if graph_dir is None:
        return {
            "participant_id": subject,
            "qc": "0",
            "labelled_L": "False",
            "labelled_R": "False",
            "reason": REASON_NO_SESSION_DIR,
        }

    session_dir = os.path.join(graph_dir, labelling_session)
    has_session_dir = os.path.isdir(session_dir)

    if not has_session_dir:
        return {
            "participant_id": subject,
            "qc": "0",
            "labelled_L": "False",
            "labelled_R": "False",
            "reason": REASON_NO_SESSION_DIR,
        }

    entries = os.listdir(session_dir)
    is_session_empty = len(entries) == 0

    labelled = {
        side: os.path.isfile(os.path.join(session_dir, make_graph_filename(side, subject, labelling_session)))
        for side in SIDES
    }

    reason = compute_reason(has_session_dir=True, is_session_empty=is_session_empty, labelled=labelled)

    # Architecture: labelled_S is False for both sides when reason is one of the first three.
    if reason in (REASON_AMBIGUOUS_BIDS_PATH, REASON_NO_SESSION_DIR, REASON_EMPTY_SESSION_DIR):
        labelled = {"L": False, "R": False}

    qc = "1" if (labelled.get("L", False) and labelled.get("R", False)) else "0"

    return {
        "participant_id": subject,
        "qc": qc,
        "labelled_L": str(labelled.get("L", False)),
        "labelled_R": str(labelled.get("R", False)),
        "reason": reason,
    }


def save_qc_tsv(rows: list, output_path: str) -> None:
    """Write rows as a tab-separated file at output_path.

    Parent directories are created as needed. The file uses a newline terminator
    of '\\n' so the output is byte-identical across platforms.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=QC_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI class
# ---------------------------------------------------------------------------


class GenerateLabellingQC(ScriptBuilder):
    """Walk a Morphologist subjects directory and write a labelling QC TSV."""

    def __init__(self):
        super().__init__(
            script_name="generate_labelling_qc",
            description=(
                "Walk a Morphologist subjects directory and write a TSV reporting, "
                "per subject, whether the labelled graphs for both hemispheres exist."
            ),
        )
        (
            self.add_argument(
                "input",
                help="Path to the Morphologist subjects directory (one subdirectory per subject).",
            )
            .add_argument(
                "output",
                help="Path to the output QC TSV file.",
            )
            .add_required_argument(
                "--path_to_graph",
                "Relative path from each subject directory to the graph folder "
                "(same as --path_to_graph in run_cortical_tiles.py). "
                "With --bids, may contain a '*' wildcard.",
            )
            .add_optional_argument(
                "--labelling_session",
                f"Name of the labelling session subdirectory (default: {LABELLING_SESSION_DEFAULT}).",
                default=LABELLING_SESSION_DEFAULT,
            )
            .add_flag(
                "--bids",
                "Interpret '*' in --path_to_graph as a glob wildcard (BIDS layout).",
            )
            .add_optional_argument(
                "--njobs",
                "Number of parallel worker processes used to check subjects (default: 1).",
                type_=int,
                default=1,
            )
        )

    def run(self) -> int:
        """Walk subjects directory, check labelling graphs, write QC TSV."""
        subjects_dir = self.args.input
        if not os.path.isdir(subjects_dir):
            raise ValueError(f"Input is not a directory: {subjects_dir}")

        subject_names = fetch_subject_names(subjects_dir)
        rows = Parallel(n_jobs=self.args.njobs)(
            delayed(fetch_subject_row)(
                subjects_dir,
                subject,
                self.args.path_to_graph,
                self.args.labelling_session,
                self.args.bids,
            )
            for subject in subject_names
        )
        rows = sorted(rows, key=lambda r: r["participant_id"])
        save_qc_tsv(rows, self.args.output)

        total = len(rows)
        n_pass = sum(1 for r in rows if r["qc"] == "1")
        print(f"Total: {total}  qc=1: {n_pass}  qc=0: {total - n_pass}")

        reason_counts = Counter(r["reason"] for r in rows if r["reason"])
        for reason, count in sorted(reason_counts.items()):
            print(f"  {reason}: {count}")

        return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the labelling QC script."""
    init_pipeline_process()
    return GenerateLabellingQC().build().print_args().run()


if __name__ == "__main__":
    exit(main())
