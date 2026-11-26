"""
Stage 2 input listing and input QC: the subject folders of the input, the folders --path_to_graph
matches (``*`` its only wildcard, BIDS volume ids through cortical_tiles' session_key), the --sk_qc_path
file, the input_qc.tsv rows, the temporary subject-ID finders and the end status of the stage. Moved
from run_cortical_tiles.py, which no longer re-exports them.
"""

import glob
import os
from os.path import isdir, join

from cortical_tiles.brainvisa.utils.skip_list import session_key
from soma import aims

from champollion_pipeline.run_cortical_tiles_graphs import GraphListing
from champollion_pipeline.utils.qc_report import REASON_SEPARATOR, read_qc_tsv
from champollion_pipeline.utils.skeleton_qc import compute_skeleton_reasons
from champollion_pipeline.utils.subject_folders import list_morphologist_subjects
from champollion_pipeline.utils.subject_name import subject_name_problem

GRAPH_WILDCARD = "*"  # the only wildcard read in --path_to_graph
GRAPH_SIDES = ("L", "R")
SK_QC_TSV_MARK = ".tsv"  # as cortical_tiles: tab-separated when the path contains it, else comma-separated
REASON_MISSING_GRAPH_STEM = "missing_graph_"  # + side
REASON_NOT_IN_SK_QC = "not_in_sk_qc_file"
REASON_EXCLUDED_BY_SK_QC = "excluded_by_sk_qc"
REASON_NO_SUBJECT_FOLDER = "no_subject_folder"
REASON_INVALID_SK_QC = "invalid_sk_qc_file"
CROP_STATUS_PARTIAL = 3  # generate_sulcal_regions.py: some subjects skipped, at least one left


def resolve_graph_paths(morpho_dir: str, path_to_graph: str, subject: str) -> list[str]:
    """Folders under ``<morpho_dir>/<subject>`` that path_to_graph matches.

    ``*`` is the only wildcard (other glob characters are literal); each match is
    returned relative to the subject folder. A subject id that is not a plain
    folder name, or a path leaving the subject folder, matches nothing.
    """
    if subject_name_problem(subject):
        return []
    subject_dir = join(morpho_dir, subject)
    pattern = GRAPH_WILDCARD.join(glob.escape(part) for part in path_to_graph.split(GRAPH_WILDCARD))
    rels = set()
    for match in glob.glob(join(glob.escape(subject_dir), pattern)):
        if not isdir(match):
            continue
        rel = os.path.relpath(match, subject_dir)
        if rel in (os.curdir, os.pardir) or rel.startswith(os.pardir + os.sep):
            continue
        rels.add(rel)
    return sorted(rels)


def resolve_bids_graph_paths(morpho_dir: str, path_to_graph: str, volume_id: str) -> list[str]:
    """Folders of resolve_graph_paths for the subject of a BIDS volume id whose session_key is that id.

    The subject is the id text before the first ``_``; cortical_tiles' session_key reads each folder
    path relative to the subject folder, so a root path holding ses/acq/run text does not count.
    """
    subject = volume_id.partition("_")[0]
    rels = resolve_graph_paths(morpho_dir, path_to_graph, subject)
    return [rel for rel in rels if session_key(subject, rel, True) == volume_id]


def list_input_subjects(input_dir: str) -> list[str]:
    """Sorted subject folder names of ``input_dir``, by the shared rule; none for a non-directory."""
    return list_morphologist_subjects(input_dir)


def fetch_sk_qc_ids(path: str) -> dict[str, int]:
    """Map each participant_id of a --sk_qc_path file to its qc.

    The file is read as cortical_tiles reads it: tab-separated when the path contains .tsv, else
    comma-separated. Raises OSError, ValueError or csv.Error on a file to refuse.
    """
    records = read_qc_tsv(path, delimiter="\t" if SK_QC_TSV_MARK in path else ",")
    sk_qc = {str(record["participant_id"]): int(record["qc"]) for record in records}
    if "" in sk_qc:
        raise ValueError(f"{path}: empty participant_id")
    return sk_qc


def has_hemisphere_graph(input_abs: str, path_to_graph: str, subject: str, side: str) -> bool:
    """True when a folder path_to_graph matches for ``subject`` holds a <side>*.arg file."""
    for graph_path in resolve_graph_paths(input_abs, path_to_graph, subject):
        pattern = join(glob.escape(join(input_abs, subject, graph_path)), f"{side}*.arg")
        if any(os.path.isfile(match) for match in glob.glob(pattern)):
            return True
    return False


def _has_graph(input_abs: str, path_to_graph: str | None, subject: str, side: str, graphs: GraphListing | None) -> bool:
    """True when ``graphs`` lists a graph of ``side`` for ``subject``; else when path_to_graph finds one."""
    if graphs is not None:
        return graphs.has_side(subject, side)
    return has_hemisphere_graph(input_abs, path_to_graph or "", subject, side)


def compute_sk_qc_reason(subject: str, sk_qc: dict[str, int] | None, is_refused: bool) -> str | None:
    """Reason the --sk_qc_path file gives against ``subject``, None when it gives none."""
    if is_refused:
        return REASON_INVALID_SK_QC
    if sk_qc is None:
        return None
    if subject not in sk_qc:
        return REASON_NOT_IN_SK_QC
    return REASON_EXCLUDED_BY_SK_QC if sk_qc[subject] == 0 else None


def compute_end_status(status: int, crop_qc_abs: str) -> int:
    """``status``, or CROP_STATUS_PARTIAL when a skeleton reason excluded a subject from a run that would return 0."""
    return CROP_STATUS_PARTIAL if crop_qc_abs and status == 0 else status


def fetch_input_qc_rows(  # noqa: PLR0913 - path_sk_with_hull and graphs optional, five-argument calls kept
    input_abs: str,
    path_to_graph: str | None,
    sk_qc: dict[str, int] | None,
    is_refused: bool,
    is_input_only: bool,
    path_sk_with_hull: str | None = None,
    graphs: GraphListing | None = None,
) -> list[dict[str, object]]:
    """input_qc.tsv rows: one per input subject folder, then, unless ``is_input_only``, the file ids with no folder.

    Reasons in order: missing_graph_L, missing_graph_R, L then R skeleton (with path_sk_with_hull), the file reason.
    A hemisphere has its graph when ``graphs`` (graph discovery, no path_to_graph) lists one for the subject folder,
    else when a folder ``path_to_graph`` matches holds a <side>*.arg file.
    """
    subjects = list_input_subjects(input_abs)
    rows: list[dict[str, object]] = []
    for subject in subjects:
        reasons = [
            f"{REASON_MISSING_GRAPH_STEM}{side}"
            for side in GRAPH_SIDES
            if not _has_graph(input_abs, path_to_graph, subject, side, graphs)
        ]
        if path_sk_with_hull is not None:
            folders = resolve_graph_paths(input_abs, path_sk_with_hull, subject)
            reasons += compute_skeleton_reasons(join(input_abs, subject), folders, aims.read)
        sk_qc_reason = compute_sk_qc_reason(subject, sk_qc, is_refused)
        if sk_qc_reason:
            reasons.append(sk_qc_reason)
        rows.append({"participant_id": subject, "qc": 0 if reasons else 1, "reason": REASON_SEPARATOR.join(reasons)})
    if sk_qc is not None and not is_input_only:
        rows.extend(
            {"participant_id": subject, "qc": 0, "reason": REASON_NO_SUBJECT_FOLDER}
            for subject in sorted(set(sk_qc) - set(subjects))
        )
    return rows


# TEMPORARY (DEF-CORTILES-BDRABCZUK-9C7D2B9265BC)
def find_dotted_ids(input_dir: str) -> list[str]:
    """Sorted input subject IDs (the list_input_subjects names) that contain a dot; none for a non-directory."""
    return [name for name in list_input_subjects(input_dir) if "." in name]


# TEMPORARY (DEF-CORTILES-BDRABCZUK-8A44DAD7F312)
def find_prefix_id_pairs(input_dir: str) -> list[tuple[str, str]]:
    """Every (shorter, longer) pair of input subject IDs where one strictly prefixes the other, sorted.

    Subject IDs are the list_input_subjects names of ``input_dir``, as cortical_tiles lists them
    (with or without --bids). An input that is not a directory gives no pair.
    """
    names = list_input_subjects(input_dir)
    pairs = []
    for i, shorter in enumerate(names):
        for longer in names[i + 1 :]:
            if not longer.startswith(shorter):
                break
            pairs.append((shorter, longer))
    return sorted(pairs)
