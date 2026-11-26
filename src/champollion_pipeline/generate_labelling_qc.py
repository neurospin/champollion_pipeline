#!/usr/bin/env python3
"""
Walk a Morphologist subjects directory and write a labelling QC TSV.

Reports, per subject, whether the labelled graphs that cortical_tiles'
remove_ventricle needs (both hemispheres) exist.

TASK-199 / COMP-LABELLING-QC.
"""

import csv
import glob
import logging
import os
import os.path
import posixpath
from collections import Counter
from datetime import datetime

from champollion_utils.script_builder import ScriptBuilder
from joblib import Parallel, delayed

from champollion_pipeline.derivatives_layout import compute_run_log_path
from champollion_pipeline.file_indexer.pipeline_checks import FileIndexStageIndexer
from champollion_pipeline.process_setup import init_pipeline_process
from champollion_pipeline.utils.graph_item import LooseGraphError
from champollion_pipeline.utils.lib import (
    LABELLING_SESSION_CANDIDATES,
    ExplicitLabellingSession,
    labelling_session_candidates,
)
from champollion_pipeline.utils.subject_folders import list_morphologist_subjects

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABELLING_SESSION_DEFAULT = LABELLING_SESSION_CANDIDATES[0]
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
    """Return the subject folders of subjects_dir in ascending order, by the shared subject-folder rule."""
    return list_morphologist_subjects(subjects_dir)


def discover_graph_folder(subject_dir: str) -> tuple[str | None, dict[str, set[str]]]:
    """Return the graph folder of a subject found by graph discovery, and the sides each labelling set holds there.

    The folder is the one holding the first labelled graph in natural path order, else the first graph; the sides
    map labelling-set name to the hemispheres ("L", "R") graphs exist for in that folder. (None, {}) when the
    subject has no graph. First graph folder only, for now: a data selector will replace it in a later version.

    Raises:
        LooseGraphError: a graph sits directly in subject_dir and its name does not start with sub-.
    """
    items = FileIndexStageIndexer("").discover_graphs(subject_dir)
    if not items:
        return None, {}

    def folder_of(item) -> str:
        folder = posixpath.dirname(item.path)
        return posixpath.dirname(folder) if item.attributes["labelling_set"] else folder

    labelled = [item for item in items if item.attributes["labelling_set"]]
    graph_folder = folder_of((labelled or items)[0])
    sides: dict[str, set[str]] = {}
    for item in labelled:
        if folder_of(item) == graph_folder and item.attributes["hemisphere"]:
            sides.setdefault(item.attributes["labelling_set"], set()).add(item.attributes["hemisphere"])
    return os.path.join(subject_dir, *graph_folder.split("/")) if graph_folder else subject_dir, sides


def find_graph_dir(subject_dir: str, path_to_graph: str | None, is_bids: bool):
    """Return the resolved graph directory for a subject.

    path_to_graph None: the graph folder found by graph discovery below subject_dir (None when no graph).

    Given, classic layout (or BIDS without '*'): returns os.path.join(subject_dir, path_to_graph)
    without checking existence.

    Given, BIDS layout with '*': globs for matching directories under subject_dir.
    Returns the single match, None (0 matches), or GRAPH_DIR_AMBIGUOUS (>1 match).
    """
    if path_to_graph is None:
        return discover_graph_folder(subject_dir)[0]
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


def find_session(graph_dir: str, sessions: tuple[str, ...]) -> str | None:
    """First of sessions whose folder exists under graph_dir (an empty folder counts), or None."""
    for session in sessions:
        if os.path.isdir(os.path.join(graph_dir, session)):
            return session
    return None


def fetch_subject_row(
    subjects_dir: str, subject: str, path_to_graph: str | None, sessions: tuple[str, ...], is_bids: bool
) -> dict:
    """Build one TSV row for a subject; all dict values are strings.

    path_to_graph None: the graph folder and the labelled hemispheres come from graph discovery.
    """
    subject_dir = os.path.join(subjects_dir, subject)
    discovered_sides = None
    if path_to_graph is None:
        graph_dir, discovered_sides = discover_graph_folder(subject_dir)
    else:
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

    labelling_session = find_session(graph_dir, sessions)

    if labelling_session is None:
        return {
            "participant_id": subject,
            "qc": "0",
            "labelled_L": "False",
            "labelled_R": "False",
            "reason": REASON_NO_SESSION_DIR,
        }

    session_dir = os.path.join(graph_dir, labelling_session)
    entries = os.listdir(session_dir)
    is_session_empty = len(entries) == 0

    if discovered_sides is None:
        labelled = {
            side: os.path.isfile(os.path.join(session_dir, make_graph_filename(side, subject, labelling_session)))
            for side in SIDES
        }
    else:
        labelled = {side: side in discovered_sides.get(labelling_session, ()) for side in SIDES}

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
            .add_optional_argument(
                "--path_to_graph",
                "Optional override: relative path from each subject directory to the graph folder "
                "(same as --path_to_graph in run_cortical_tiles.py). "
                "With --bids, may contain a '*' wildcard. Default: each subject's graph folder is found by "
                "graph discovery (first graph folder only, for now).",
            )
            .add_argument(
                "--labelling_session",
                action=ExplicitLabellingSession,
                default=LABELLING_SESSION_DEFAULT,
                help=(
                    "Name of the labelling session subdirectory. When given, only that session is checked; "
                    "when omitted, each subject uses the first existing folder of "
                    f"{', '.join(LABELLING_SESSION_CANDIDATES)}."
                ),
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

    def default_log_path(self) -> str:
        """Run-log path of a bare --log: .log/generate_labelling_qc_<stamp>.log next to the output TSV."""
        return compute_run_log_path(os.path.dirname(self.args.output), self.script_name, datetime.now().astimezone())

    def run(self) -> int:
        """Walk subjects directory, check labelling graphs, write QC TSV; return 1 when no subject has qc 1."""
        subjects_dir = self.args.input
        if not os.path.isdir(subjects_dir):
            raise ValueError(f"Input is not a directory: {subjects_dir}")

        subject_names = fetch_subject_names(subjects_dir)
        sessions = labelling_session_candidates(self.args)
        try:
            rows = Parallel(n_jobs=self.args.njobs)(
                delayed(fetch_subject_row)(
                    subjects_dir,
                    subject,
                    self.args.path_to_graph,
                    sessions,
                    self.args.bids,
                )
                for subject in subject_names
            )
        except LooseGraphError as error:
            logger.error(str(error))
            return 2
        rows = sorted(rows, key=lambda r: r["participant_id"])
        save_qc_tsv(rows, self.args.output)

        total = len(rows)
        n_pass = sum(1 for r in rows if r["qc"] == "1")
        logger.output(f"Total: {total}  qc=1: {n_pass}  qc=0: {total - n_pass}")

        reason_counts = Counter(r["reason"] for r in rows if r["reason"])
        for reason, count in sorted(reason_counts.items()):
            logger.output(f"{reason}: {count}")

        for row in rows:
            if row["qc"] == "0":
                logger.error(f"labelling QC excluded {row['participant_id']}: {row['reason']}")
        if n_pass == 0:
            logger.fail(f"no subject passed the labelling QC, see {self.args.output}")
            return 1
        return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the labelling QC script."""
    init_pipeline_process()
    return GenerateLabellingQC().main()


if __name__ == "__main__":
    exit(main())
