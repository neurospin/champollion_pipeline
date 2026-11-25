"""
Stage 2 whole-brain volume: fusion of the L and R skeletons, labelling pre-check (labelling_qc.tsv) and
ventricle removal. Fusion comes first, since remove_ventricle(F) reads the tree add_left_and_right_volumes
wrote; src_dir is the crop subprocess's skeletons/<vox>/{L,R}, never input_abs (the Morphologist graphs
directory, read-only); morpho_dir stays input_abs, where remove_ventricle locates the labelled graphs.
The cortical_tiles steps come in as callables from run_cortical_tiles, which re-exports every moved name.
"""

import argparse
import logging
import os
import re
import tempfile
from collections.abc import Callable
from os.path import abspath, isdir, join

from champollion_pipeline.generate_labelling_qc import REASON_NO_SESSION_DIR, fetch_subject_row
from champollion_pipeline.run_cortical_tiles_inputs import (
    CROP_STATUS_PARTIAL,
    GRAPH_SIDES,
    resolve_bids_graph_paths,
    resolve_graph_paths,
)
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER, LABELLING_SESSION_CANDIDATES
from champollion_pipeline.utils.qc_report import write_qc_tsv

# The stage 2 logger the Requirements name (REQ-CTFAIL-BDRABCZUK-547E62BB41B1, -C6A79DB942F0, -F8AFF7D8C015,
# -F8EA34944489, REQ-LOGGING-BDRABCZUK-12F726C73E22, REQ-WHOLEBRAIN-BDRABCZUK-0918A9D01062), not __name__.
logger = logging.getLogger("champollion_pipeline.run_cortical_tiles")

WHOLE_BRAIN_SRC_FILENAME = "resampled_skeleton"
WHOLE_BRAIN_OUT_FILENAME = "skeleton_generated_without_ventricle_"  # upstream default, now passed explicitly
# Whole-brain exclusion reasons; tuple order is the clause order in the warning.
EXCLUDED_NO_GRAPH_FOLDER = "no_graph_folder"  # non-BIDS: --path_to_graph matches no folder
EXCLUDED_SEVERAL_GRAPH_FOLDERS = "several_graph_folders"  # non-BIDS: it matches two or more folders
EXCLUDED_NO_LABELLED_GRAPH = "no_labelled_graph"  # graph folder found, no candidate session dir in it
EXCLUSION_REASONS = (EXCLUDED_NO_GRAPH_FOLDER, EXCLUDED_SEVERAL_GRAPH_FOLDERS, EXCLUDED_NO_LABELLED_GRAPH)
LABELLING_QC_FILENAME = "labelling_qc.tsv"
LABELLING_QC_EXTRA_COLUMNS = ("labelled_L", "labelled_R")  # with write_qc_tsv: generate_labelling_qc.QC_COLUMNS


def list_side_volumes(volumes_dir: str, filename: str, side: str = "F") -> dict[str, str]:
    """Map each subject with a volume in <volumes_dir>/<side>/ to its absolute path.

    The subject id is what follows ``<side><filename>`` in the file name, with one
    leading ``_`` stripped (cortical_tiles' own subject-id rule). A missing folder gives {}.
    """
    side_dir = join(volumes_dir, side)
    if not isdir(side_dir):
        return {}
    pattern = re.compile(rf"^{re.escape(side + filename)}(.*)\.nii\.gz$")
    volumes = {}
    for name in sorted(os.listdir(side_dir)):
        match = pattern.match(name)
        if match:
            subject = match.group(1)
            volumes[subject[1:] if subject.startswith("_") else subject] = abspath(join(side_dir, name))
    return volumes


def call_whole_brain_step(name: str, step: Callable[..., None], **kwargs: object) -> bool:
    """Call ``step``; a raise gives one ERROR record with its traceback and False, a return True."""
    try:
        step(**kwargs)
    except Exception:
        logger.exception(f"whole-brain {name} raised")
        return False
    return True


def labelled_session_dir(morpho_dir: str, path_to_graph: str, subject: str, session: str, bids: bool) -> str:
    """Directory holding a subject's labelled graphs for one session.

    ``path_to_graph`` is a resolved graph path (no ``*``); with BIDS names the
    folder is the subject part of the id, as in cortical_tiles' get_labelled_graph.
    """
    folder = subject.partition("_")[0] if bids else subject
    return join(morpho_dir, folder, path_to_graph, session)


def resolve_labelling_session(
    morpho_dir: str, path_to_graph: str, subject: str, candidates: tuple[str, ...], bids: bool
) -> str | None:
    """First candidate session whose labelled-graph directory exists for the subject, else None."""
    for session in candidates:
        if isdir(labelled_session_dir(morpho_dir, path_to_graph, subject, session, bids)):
            return session
    return None


def fetch_labelling_qc_row(
    morpho_dir: str, path_to_graph: str, subject: str, candidates: tuple[str, ...], bids: bool
) -> dict[str, str]:
    """labelling_qc.tsv row of a fused subject, at the graphs whole-brain ventricle removal reads.

    ``*`` must match exactly one folder: with BIDS names, the one folder of the subject part of
    the id whose session_key is the fused id. None or several gives qc 0 no_session_dir.
    """
    folder = subject.partition("_")[0] if bids else subject
    resolve = resolve_bids_graph_paths if bids else resolve_graph_paths
    graph_paths = resolve(morpho_dir, path_to_graph, subject)
    if len(graph_paths) == 1:
        return {**fetch_subject_row(morpho_dir, folder, graph_paths[0], candidates, False), "participant_id": subject}
    return {
        "participant_id": subject,
        "qc": "0",
        "labelled_L": "False",
        "labelled_R": "False",
        "reason": REASON_NO_SESSION_DIR,
    }


def exclusion_warning(excluded: dict[str, list[str]], path_to_graph: str, candidates: tuple[str, ...]) -> str:
    """One-line warning naming the subjects left out of whole-brain removal, by reason."""
    clause_for = {
        EXCLUDED_NO_GRAPH_FOLDER: f"--path_to_graph {path_to_graph!r} matches no folder for {{ids}}",
        EXCLUDED_SEVERAL_GRAPH_FOLDERS: (
            f"--path_to_graph {path_to_graph!r} matches several folders for {{ids}} (ambiguous, not guessed)"
        ),
        EXCLUDED_NO_LABELLED_GRAPH: (
            f"no labelled graph under {' or '.join(candidates)} for {{ids}} (run Morphologist with sulcal recognition)"
        ),
    }
    clauses = [
        clause_for[reason].format(ids=", ".join(sorted(excluded[reason])))
        for reason in EXCLUSION_REASONS
        if excluded.get(reason)
    ]
    n = sum(len(ids) for ids in excluded.values())
    return (
        f"{n} subject(s) excluded from whole-brain ventricle removal, which "
        "requires sulcal recognition (one labelled graph per subject, found through --path_to_graph): "
        f"{'; '.join(clauses)}."
    )


def write_labelling_qc(
    args: argparse.Namespace, subjects: list[str], input_abs: str, output_abs: str, candidates: tuple[str, ...]
) -> tuple[str, list[str], bool]:
    """Write labelling_qc.tsv, one ERROR per failing subject; return its path, the subjects with qc 1 and
    whether no subject has a labelling session folder (every row no_session_dir: no ERROR is logged then,
    the caller warns once)."""
    rows = [
        fetch_labelling_qc_row(input_abs, args.path_to_graph, subject, candidates, args.bids)
        for subject in sorted(subjects)
    ]
    qc_path = join(output_abs, DERIVATIVES_FOLDER, LABELLING_QC_FILENAME)
    write_qc_tsv(qc_path, rows, extra_columns=LABELLING_QC_EXTRA_COLUMNS)
    has_no_labels = all(row["reason"] == REASON_NO_SESSION_DIR for row in rows)
    for row in rows:
        if row["qc"] == "0" and not has_no_labels:
            logger.error(f"labelling QC excluded {row['participant_id']}: {row['reason']}")
    return qc_path, [row["participant_id"] for row in rows if row["qc"] == "1"], has_no_labels


def group_whole_brain_subjects(
    args: argparse.Namespace, subjects: list[str], input_abs: str, candidates: tuple[str, ...]
) -> tuple[dict[tuple[str, str], list[str]], dict[str, list[str]]]:
    """Group subjects by (labelling session, resolved graph path); sort the rest by exclusion reason."""
    groups: dict[tuple[str, str], list[str]] = {}
    excluded: dict[str, list[str]] = {}
    for subject in subjects:
        resolve = resolve_bids_graph_paths if args.bids else resolve_graph_paths
        graph_paths = resolve(input_abs, args.path_to_graph, subject)
        if not graph_paths:
            excluded.setdefault(EXCLUDED_NO_GRAPH_FOLDER, []).append(subject)
            continue
        if len(graph_paths) > 1:
            excluded.setdefault(EXCLUDED_SEVERAL_GRAPH_FOLDERS, []).append(subject)
            continue
        session = resolve_labelling_session(input_abs, graph_paths[0], subject, candidates, args.bids)
        if session is None:
            excluded.setdefault(EXCLUDED_NO_LABELLED_GRAPH, []).append(subject)
        else:
            groups.setdefault((session, graph_paths[0]), []).append(subject)
    return groups, excluded


def fuse_whole_brain(skeleton_src_dir: str, fuse: Callable[..., None]) -> bool:
    """Fuse the L and R skeletons; False when the call raises or a subject with both sides has no fused volume."""
    is_done = call_whole_brain_step(
        "fusion",
        fuse,
        src_dir=skeleton_src_dir,
        parallel=True,
        # Its own -1 default breaks on the string-only revalidation in
        # get_number_subjects in the cortical_tiles brainvisa subjects utils,
        # so pass "all" explicitly to route around that upstream quirk,
        # which lives under external/ and is not ours to fix.
        number_subjects="all",
    )
    left, right = (list_side_volumes(skeleton_src_dir, WHOLE_BRAIN_SRC_FILENAME, side) for side in GRAPH_SIDES)
    fused = list_side_volumes(skeleton_src_dir, WHOLE_BRAIN_SRC_FILENAME)
    lost = sorted((left.keys() & right.keys()) - fused.keys())
    for subject in lost:
        logger.error(f"whole-brain fusion left no volume for {subject}")
    return is_done and not lost


def compute_fusion_failure_status(output_abs: str) -> int:
    """Status of a whole-brain failure: CROP_STATUS_PARTIAL when whole_brain/F holds a volume, else 1."""
    whole_brain_dir = join(output_abs, DERIVATIVES_FOLDER, "whole_brain")
    return CROP_STATUS_PARTIAL if list_side_volumes(whole_brain_dir, WHOLE_BRAIN_OUT_FILENAME) else 1


def remove_ventricle_whole_brain(
    args: argparse.Namespace, skeleton_src_dir: str, input_abs: str, output_abs: str, remove: Callable[..., None]
) -> bool:
    """Strip the ventricle from each fused volume, grouping subjects by labelling session.

    remove_ventricle takes one labelling session per call and enumerates
    subjects from <src_dir>/F/, so when subjects resolve to different
    sessions (or some have none) each group gets a staged F/ of symlinks.
    Without --bids, a ``*`` in path_to_graph is resolved per subject, so
    groups are keyed by (session, graph path). Subjects that cannot be
    resolved are left out with one warning. labelling_qc.tsv is written first;
    with no qc 1 row, one warning names it and no volume is processed.
    False when a call raises (later groups still run) or a qc 1 subject has
    no whole_brain/F volume.
    """
    candidates = (args.labelling_session,) if args.labelling_session_explicit else LABELLING_SESSION_CANDIDATES

    fused = list_side_volumes(skeleton_src_dir, WHOLE_BRAIN_SRC_FILENAME)
    if not fused:
        logger.warning(f"No fused volume under {skeleton_src_dir}/F; whole-brain ventricle removal skipped.")
        return True

    qc_path, passed, has_no_labels = write_labelling_qc(args, list(fused), input_abs, output_abs, candidates)
    groups, excluded = group_whole_brain_subjects(args, list(fused), input_abs, candidates)
    exclusions = exclusion_warning(excluded, args.path_to_graph, candidates) if excluded else ""
    if has_no_labels:
        no_labels = (
            f"Whole-brain volume left out: the Morphologist graphs carry no sulcal labels (no labelling session "
            f"folder for any subject), see {qc_path}. Run stage 1 with --enable-sulcal-recognition to produce it."
        )
        logger.warning(f"{no_labels} {exclusions}".rstrip())
        return True
    if not passed:
        left_out = f"Whole-brain volume left out: no subject passed the labelling pre-check, see {qc_path}."
        logger.warning(f"{left_out} {exclusions}".rstrip())
        return True
    if exclusions:
        logger.warning(exclusions)

    whole_brain_dir = join(output_abs, DERIVATIVES_FOLDER, "whole_brain")

    def call(src_dir: str, session: str, graph_path: str) -> bool:
        return call_whole_brain_step(
            "ventricle removal",
            remove,
            side="F",
            src_dir=src_dir,
            morpho_dir=input_abs,
            path_to_graph=graph_path,
            # add_left_and_right_volumes' default output_filename
            # ("resampled_skeleton") does not match remove_ventricle's
            # default src_filename ("skeleton_generated_") -- pass it
            # explicitly so remove_ventricle reads what was just fused.
            src_filename=WHOLE_BRAIN_SRC_FILENAME,
            output_filename=WHOLE_BRAIN_OUT_FILENAME,
            parallel=True,
            output_dir=whole_brain_dir,
            transform_dir=join(output_abs, DERIVATIVES_FOLDER, "transforms"),
            labelling_session=session,
            bids=args.bids,
        )

    is_done = True
    if len(groups) == 1 and not excluded:
        is_done = call(skeleton_src_dir, *next(iter(groups)))
    else:
        for session, graph_path in sorted(groups, key=lambda key: (candidates.index(key[0]), key[1])):
            with tempfile.TemporaryDirectory(prefix="whole_brain_") as staged:
                os.makedirs(join(staged, "F"))
                for subject in groups[(session, graph_path)]:
                    os.symlink(fused[subject], join(staged, "F", os.path.basename(fused[subject])))
                is_done = call(staged, session, graph_path) and is_done

    lost = sorted(set(passed) - list_side_volumes(whole_brain_dir, WHOLE_BRAIN_OUT_FILENAME).keys())
    for subject in lost:
        logger.error(f"whole-brain ventricle removal left no volume for {subject}")
    return is_done and not lost
