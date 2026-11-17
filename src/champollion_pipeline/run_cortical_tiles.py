#!/usr/bin/env python3
"""
Script to generate sulcal regions with cortical_tiles from Morphologist's graphs.

Output structure
----------------
Sulcal region crops are written to::

    {output}/cortical_tiles-{VERSION}/crops/{voxel_size}mm/{region}/

where ``{voxel_size}mm`` is the voxel resolution (e.g. ``2mm``) and
``{VERSION}`` is the cortical_tiles release year (e.g. ``2026``).

When a mask version tag is provided via ``--masks`` (e.g. ``canonical_25``),
it overrides the ``masks_version`` field in the pipeline JSON config so that
the correct labelled masks are used during region extraction.
"""

import argparse
import csv
import glob
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime
from os.path import abspath, dirname, exists, isdir, join

import numpy as np
from champollion_utils.script_builder import ScriptBuilder
from cortical_tiles.brainvisa import add_left_and_right_volumes, remove_ventricle
from cortical_tiles.brainvisa.utils.skip_list import (
    SKIP_LIST_FILENAME,
    SKIP_SIDES,
    SkipEntry,
    load_skip_entries,
    session_key,
)
from joblib import cpu_count
from soma import aims

from champollion_pipeline.derivatives_layout import compute_run_log_path
from champollion_pipeline.generate_labelling_qc import REASON_NO_SESSION_DIR, fetch_subject_row
from champollion_pipeline.generate_masks import _REGIONS_DEFAULT
from champollion_pipeline.process_setup import init_pipeline_process
from champollion_pipeline.utils.cortical_tiles_config import (
    INPUT_TYPES_DEFAULT,
    CorticalTilesConfigFactory,
    compute_versioned_crops_dir,
    versioned_crops_exist,
)
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER, LABELLING_SESSION_CANDIDATES
from champollion_pipeline.utils.qc_report import (
    CROP_KIND_MASK,
    REASON_SEPARATOR,
    fetch_crop_qc_rows,
    read_qc_tsv,
    replace_missing_crop_reasons,
    write_qc_tsv,
)
from champollion_pipeline.utils.resource_copy import grants_no_write, replace_with_writable_copy
from champollion_pipeline.utils.skeleton_qc import compute_skeleton_reasons, save_config_skel_qc, save_crop_qc
from champollion_pipeline.utils.subject_folders import list_morphologist_subjects
from champollion_pipeline.utils.subject_name import subject_name_problem
from champollion_pipeline.utils.transform_qc import transform_qc_rows

logger = logging.getLogger(__name__)

LABELLING_SESSION_DEFAULT = LABELLING_SESSION_CANDIDATES[0]
WHOLE_BRAIN_SRC_FILENAME = "resampled_skeleton"
WHOLE_BRAIN_OUT_FILENAME = "skeleton_generated_without_ventricle_"  # upstream default, now passed explicitly
GRAPH_WILDCARD = "*"  # the only wildcard read in --path_to_graph
# Whole-brain exclusion reasons; tuple order is the clause order in the warning.
EXCLUDED_NO_GRAPH_FOLDER = "no_graph_folder"  # non-BIDS: --path_to_graph matches no folder
EXCLUDED_SEVERAL_GRAPH_FOLDERS = "several_graph_folders"  # non-BIDS: it matches two or more folders
EXCLUDED_NO_LABELLED_GRAPH = "no_labelled_graph"  # graph folder found, no candidate session dir in it
EXCLUSION_REASONS = (EXCLUDED_NO_GRAPH_FOLDER, EXCLUDED_SEVERAL_GRAPH_FOLDERS, EXCLUDED_NO_LABELLED_GRAPH)
LABELLING_QC_FILENAME = "labelling_qc.tsv"
LABELLING_QC_EXTRA_COLUMNS = ("labelled_L", "labelled_R")  # with write_qc_tsv: generate_labelling_qc.QC_COLUMNS
# TEMPORARY (DEF-CORTILES-BDRABCZUK-8A44DAD7F312): the prefix-ID guard below (and its flag, check and test file)
# goes once cortical_tiles stops globbing {subject}*/ in generate_ICBM2009c_transforms.py.
PREFIX_PAIRS_SHOWN = 5  # prefix pairs named in the refusal or warning
# TEMPORARY (DEF-CORTILES-BDRABCZUK-8A44DAD7F312): transform QC, a report of subjects whose ICBM2009c
# transform is byte-identical to another subject's; goes with the prefix-ID guard.
TRANSFORM_QC_FILENAME = "transform_qc.tsv"
INPUT_QC_FILENAME = "input_qc.tsv"
GRAPH_SIDES = ("L", "R")
SK_QC_TSV_MARK = ".tsv"  # as cortical_tiles: tab-separated when the path contains it, else comma-separated
REASON_MISSING_GRAPH_STEM = "missing_graph_"  # + side
REASON_NOT_IN_SK_QC = "not_in_sk_qc_file"
REASON_EXCLUDED_BY_SK_QC = "excluded_by_sk_qc"
REASON_NO_SUBJECT_FOLDER = "no_subject_folder"
REASON_INVALID_SK_QC = "invalid_sk_qc_file"
OUTPUT_QC_FILENAME = "output_qc.tsv"
CROP_MARKER_FILENAME = "crops_complete.marker"
CROP_STATUS_PARTIAL = 3  # generate_sulcal_regions.py: some subjects skipped, at least one left


class _ExplicitLabellingSession(argparse.Action):
    """Store --labelling_session and record that the user gave it explicitly."""

    def __call__(self, parser, namespace, values, _option_string=None):
        namespace.labelling_session = values
        namespace.labelling_session_explicit = True


ROOT_FORBIDDEN_SUBSTRINGS = ("ses-", "acq-", "run-", "_cropped")


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


def fetch_input_qc_rows(  # noqa: PLR0913 - path_sk_with_hull optional, five-argument calls kept
    input_abs: str,
    path_to_graph: str,
    sk_qc: dict[str, int] | None,
    is_refused: bool,
    is_input_only: bool,
    path_sk_with_hull: str | None = None,
) -> list[dict[str, object]]:
    """input_qc.tsv rows: one per input subject folder, then, unless ``is_input_only``, the file ids with no folder.

    Reasons in order: missing_graph_L, missing_graph_R, L then R skeleton (with path_sk_with_hull), the file reason.
    """
    subjects = list_input_subjects(input_abs)
    rows: list[dict[str, object]] = []
    for subject in subjects:
        reasons = [
            f"{REASON_MISSING_GRAPH_STEM}{side}"
            for side in GRAPH_SIDES
            if not has_hemisphere_graph(input_abs, path_to_graph, subject, side)
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


class RunCorticalTiles(ScriptBuilder):
    """Script for running cortical_tiles to generate sulcal regions."""

    def __init__(self):
        super().__init__(
            script_name="run_cortical_tiles",
            description="Generating sulcal regions with cortical_tiles from Morphologist's graphs.",
        )
        # Configure arguments using method chaining
        (
            self.add_argument(
                "input",
                help="Absolute path to the directory containing subject folders "
                "(e.g., morphologist's output subjects directory).",
            )
            .add_argument("output", help="Absolute path to the generated sulcal regions from cortical_tiles.")
            .add_optional_argument("--region-file", "Absolute path to the user's sulcal region's configuration file.")
            .add_required_argument(
                "--path_to_graph", "Contains the sub-path that, for each subject, permits getting the sulcal graphs."
            )
            .add_required_argument("--path_sk_with_hull", "Contains the sub-path where to get the skeleton with hull.")
            .add_optional_argument("--sk_qc_path", "The path to the QC file if it exists.", default="")
            .add_flag(
                "--sk_qc_input_only",
                "Limit the input QC to the subjects of the input: an id of the --sk_qc_path file with no "
                "input subject folder gets no input_qc.tsv row. For a run on part of the cohort the file covers.",
            )
            .add_optional_argument("--njobs", "Number of CPU cores allowed to use.", default=None, type_=int)
            .add_argument(
                "--input-types",
                nargs="+",
                default=None,
                choices=INPUT_TYPES_DEFAULT,
                help="Input types to generate, among skeleton and foldlabel. Default: skeleton foldlabel.",
            )
            .add_argument(
                "--regions",
                nargs="+",
                default=None,
                help="Restrict processing to these sulcal regions (space-separated). Default: all 28 regions.",
            )
            .add_optional_argument(
                "--masks",
                "Mask version tag (e.g. 'canonical_25'). Overrides masks_version in the pipeline JSON config.",
                default=None,
            )
            .add_argument(
                "--labelling_session",
                action=_ExplicitLabellingSession,
                default=LABELLING_SESSION_DEFAULT,
                help="Morphologist labelling session whose labelled graphs locate the ventricle "
                "for whole-brain removal (e.g. '0_auto'). Default: per subject, the first of "
                f"{' and '.join(LABELLING_SESSION_CANDIDATES)} that holds a labelled graph.",
            )
            .add_optional_argument(
                "--skel-seed",
                "Skeleton seed forwarded to generate_sulcal_regions.py as --skel_seed; "
                "overrides skel_seed in the derivatives pipeline_loop_2mm.json.",
                default=None,
                type_=int,
            )
            .add_flag("--overwrite", "Re-generate crops even if they already exist for this mask version.")
            .add_flag(
                "--bids",
                "Input is BIDS-named (sub-/ses-/run- prefixed files). Threaded into the "
                "pipeline JSON config and the whole-brain ventricle-removal call.",
            )
            .add_flag(  # TEMPORARY (DEF-CORTILES-BDRABCZUK-8A44DAD7F312)
                "--allow-prefix-ids",
                "Run even when one input subject ID strictly prefixes another (e.g. sub-1 and sub-10); "
                "logs a warning instead of refusing. cortical_tiles may then give a subject the ICBM2009c "
                "transform of the other. Temporary: removed once cortical_tiles fixes its {subject}*/ glob.",
            )
        )
        self.parser.set_defaults(labelling_session_explicit=False)

    # TEMPORARY (DEF-CORTILES-BDRABCZUK-8A44DAD7F312, -9C7D2B9265BC)
    def _check_subject_ids(self, input_abs: str, output_abs: str) -> bool:
        """Run the temporary subject-ID and root guards; False when one refuses the input."""
        return (
            self._check_root_substrings(input_abs, output_abs)
            and self._check_dotted_ids(input_abs)
            and self._check_prefix_ids(input_abs)
        )

    # TEMPORARY (DEF-CORTILES-BDRABCZUK-9C7D2B9265BC)
    def _check_root_substrings(self, input_abs: str, output_abs: str) -> bool:
        """Refuse (False, error) an input or output root whose path holds a substring cortical_tiles misparses."""
        for root in (input_abs, output_abs):
            bad = next((s for s in ROOT_FORBIDDEN_SUBSTRINGS if s in root), None)
            if bad:
                logger.error(
                    f"Root path {root} contains '{bad}'. cortical_tiles matches ses-/acq-/run- unanchored on the "
                    "full path and splits on _cropped, which corrupts subject parsing. Use a root without it."
                )
                return False
        return True

    # TEMPORARY (DEF-CORTILES-BDRABCZUK-9C7D2B9265BC)
    def _check_dotted_ids(self, input_abs: str) -> bool:
        """Refuse (False, error) a non-BIDS input with a dotted subject ID."""
        if self.args.bids:
            return True
        dotted = find_dotted_ids(input_abs)
        if not dotted:
            return True
        logger.error(
            f"Subject ID(s) containing a dot: {', '.join(dotted)}. cortical_tiles splits IDs on the first dot "
            "and would merge them (A.1 and A.2 become A). Rename the subject folders, or pass --bids."
        )
        return False

    # TEMPORARY (DEF-CORTILES-BDRABCZUK-8A44DAD7F312)
    def _check_prefix_ids(self, input_abs: str) -> bool:
        """Refuse (False, error) or allow with a warning (True) an input with prefix subject IDs."""
        pairs = find_prefix_id_pairs(input_abs)
        if not pairs:
            return True
        listed = ", ".join(f"({a}, {b})" for a, b in pairs[:PREFIX_PAIRS_SHOWN])
        more = f" and {len(pairs) - PREFIX_PAIRS_SHOWN} more" if len(pairs) > PREFIX_PAIRS_SHOWN else ""
        head = f"{len(pairs)} input subject ID pair(s) where one ID strictly prefixes another: {listed}{more}."
        if not self.args.allow_prefix_ids:
            logger.error(
                head + " cortical_tiles globs '{subject}*/' and may give a subject the ICBM2009c transform of "
                "another one. Rename the subject folders, or pass --allow-prefix-ids to run anyway."
            )
            return False
        logger.warning(
            head + " --allow-prefix-ids given: running anyway; cortical_tiles may give a subject the ICBM2009c "
            "transform of another one (check these subjects' transforms)."
        )
        return True

    def _preflight_check(self, output_abs: str, config_path: str) -> bool:
        """Check for existing crops and migrate legacy flat structure if needed.

        Returns False (caller should skip the crop step), without --overwrite, when the crops of the requested
        mask version hold the completion marker or legacy crops conflict with them. Crops with no marker give True.
        """
        existing_cfg = CorticalTilesConfigFactory.from_pipeline_json(config_path)
        requested_cfg = CorticalTilesConfigFactory.from_args(self.args)

        vox_str = f"{int(requested_cfg.out_voxel_size)}mm"
        derivatives_abs = join(output_abs, DERIVATIVES_FOLDER)

        # Migrate legacy crops written before versioned path was introduced.
        # Old structure: crops/2mm/  →  New structure: crops/{masks_version}/2mm/
        legacy_crops = join(derivatives_abs, "crops", vox_str)
        if isdir(legacy_crops) and existing_cfg is not None:
            old_version = existing_cfg.masks_version
            target = join(derivatives_abs, "crops", old_version, vox_str)
            if not isdir(target):
                logger.trace(f"Migrating legacy crops to versioned path: crops/{old_version}/...")
                shutil.move(legacy_crops, target)
            else:
                if not self.args.overwrite:
                    logger.error(
                        f"Legacy crops found at crops/{vox_str}/ but versioned crops already exist "
                        f"at crops/{old_version}/{vox_str}/. Please resolve manually or use --overwrite "
                        "to remove both and regenerate."
                    )
                    return False
                logger.warning(
                    f"--overwrite: removing legacy crops/{vox_str}/ and existing crops/{old_version}/{vox_str}/..."
                )
                shutil.rmtree(legacy_crops)
                shutil.rmtree(target)

        if versioned_crops_exist(derivatives_abs, requested_cfg.masks_version, requested_cfg.out_voxel_size):
            if self.args.overwrite:
                logger.warning(f"Overwriting existing crops for '{requested_cfg.masks_version}'.")
            elif exists(self._compute_crop_marker_path(output_abs)):
                logger.warning(
                    f"Crops for mask version '{requested_cfg.masks_version}' already exist. "
                    "Use --overwrite to regenerate."
                )
                return False
            else:
                logger.warning(
                    f"Crops for mask version '{requested_cfg.masks_version}' hold no {CROP_MARKER_FILENAME}: "
                    "not complete, regenerating them."
                )
        return True

    def _compute_requested_crops_dir(self, output_abs: str) -> str:
        """Crops folder of the requested masks version and voxel size."""
        requested_cfg = CorticalTilesConfigFactory.from_args(self.args)
        derivatives = join(output_abs, DERIVATIVES_FOLDER)
        return compute_versioned_crops_dir(derivatives, requested_cfg.masks_version, requested_cfg.out_voxel_size)

    def _compute_crop_marker_path(self, output_abs: str) -> str:
        """Path of the completion marker in the requested crops folder."""
        return join(self._compute_requested_crops_dir(output_abs), CROP_MARKER_FILENAME)

    def _write_crop_marker(self, output_abs: str) -> None:
        """Lay the empty completion marker in the requested crops folder."""
        marker_path = self._compute_crop_marker_path(output_abs)
        os.makedirs(dirname(marker_path), exist_ok=True)
        with open(marker_path, "w"):
            pass

    def _write_transform_qc(self, output_abs: str) -> None:
        """Write <derivatives>/transform_qc.tsv and log one summary record (warning when a subject is flagged)."""
        derivatives = join(output_abs, DERIVATIVES_FOLDER)
        if not isdir(derivatives):
            logger.info(f"transform QC skipped: no derivatives directory {derivatives}.")
            return
        rows = transform_qc_rows(join(derivatives, "transforms"))
        qc_path = join(derivatives, TRANSFORM_QC_FILENAME)
        write_qc_tsv(qc_path, rows)
        flagged = sum(1 for row in rows if row["qc"] == 0)
        if flagged > 0:
            logger.warning(
                f"transform QC: {flagged} subject(s) with an ICBM2009c transform byte-identical to another "
                f"subject's, see {qc_path}; their crops may use the wrong transform."
            )
        else:
            logger.output(f"transform QC: {len(rows)} subject(s), no byte-identical transform, see {qc_path}.")

    def _write_labelling_qc(
        self, subjects: list[str], input_abs: str, output_abs: str, candidates: tuple[str, ...]
    ) -> tuple[str, list[str]]:
        """Write labelling_qc.tsv, one ERROR per failing subject; return its path and the subjects with qc 1."""
        rows = [
            fetch_labelling_qc_row(input_abs, self.args.path_to_graph, subject, candidates, self.args.bids)
            for subject in sorted(subjects)
        ]
        qc_path = join(output_abs, DERIVATIVES_FOLDER, LABELLING_QC_FILENAME)
        write_qc_tsv(qc_path, rows, extra_columns=LABELLING_QC_EXTRA_COLUMNS)
        for row in rows:
            if row["qc"] == "0":
                logger.error(f"labelling QC excluded {row['participant_id']}: {row['reason']}")
        return qc_path, [row["participant_id"] for row in rows if row["qc"] == "1"]

    def _write_input_qc(self, input_abs: str, output_abs: str, sk_qc_abs: str) -> tuple[str, bool, str]:
        """Write input_qc.tsv, one record per qc 0 row, and crop_qc.tsv; return path, a subject passed, crop_qc path."""
        sk_qc = None
        is_refused = False
        if sk_qc_abs:
            try:
                sk_qc = fetch_sk_qc_ids(sk_qc_abs)
            except (OSError, ValueError, csv.Error) as error:
                logger.error(f"--sk_qc_path file {self.args.sk_qc_path} refused: {error}")
                is_refused = True
        args = self.args
        rows = fetch_input_qc_rows(
            input_abs, args.path_to_graph, sk_qc, is_refused, args.sk_qc_input_only, args.path_sk_with_hull
        )
        qc_path = join(output_abs, DERIVATIVES_FOLDER, INPUT_QC_FILENAME)
        write_qc_tsv(qc_path, rows)
        for row in rows:
            if row["qc"] == 0:
                log = logger.warning if row["reason"] == REASON_EXCLUDED_BY_SK_QC else logger.error
                log(f"input QC excluded {row['participant_id']}: {row['reason']}")
        return qc_path, any(row["qc"] == 1 for row in rows), save_crop_qc(join(output_abs, DERIVATIVES_FOLDER), rows)

    def _list_input_types(self) -> list[str]:
        """Input types of this run: --input-types, else INPUT_TYPES_DEFAULT."""
        return list(self.args.input_types or INPUT_TYPES_DEFAULT)

    def _list_crop_kinds(self) -> list[str]:
        """Crop kinds the output QC checks: the input types, then mask."""
        return [*self._list_input_types(), CROP_KIND_MASK]

    def _read_skip_lists(self, output_abs: str) -> dict[str, list[SkipEntry]]:
        """Entries of the skip list of each side; a list its reader refuses gives one ERROR and no entry."""
        derivatives = join(output_abs, DERIVATIVES_FOLDER)
        skips: dict[str, list[SkipEntry]] = {}
        for side in SKIP_SIDES:
            try:
                skips[side] = load_skip_entries(derivatives, side)
            except ValueError as error:
                path = join(derivatives, SKIP_LIST_FILENAME.format(side=side))
                logger.error(f"skip list of side {side} not read, {path}: {error}")
                skips[side] = []
        return skips

    def _log_skip_summary(self, skips: dict[str, list[SkipEntry]]) -> None:
        """Log one ERROR record per side and step holding an entry, naming its sorted skipped subjects once each."""
        for side, entries in skips.items():
            subjects_per_step: dict[str, set[str]] = {}
            for entry in entries:
                subjects_per_step.setdefault(entry.step, set()).add(entry.subject)
            for step, subjects in subjects_per_step.items():
                logger.error(f"side {side} step {step} skipped subjects: {', '.join(sorted(subjects))}")

    def _write_output_qc(self, output_abs: str, skips: dict[str, list[SkipEntry]]) -> tuple[str, bool]:
        """Write output_qc.tsv, one record per qc 0 row; return its path and whether a subject passed."""
        derivatives = join(output_abs, DERIVATIVES_FOLDER)
        crops_dir = self._compute_requested_crops_dir(output_abs)
        input_records = read_qc_tsv(join(derivatives, INPUT_QC_FILENAME))
        subjects = [str(record["participant_id"]) for record in input_records if record["qc"] == 1]
        regions = self.args.regions or _REGIONS_DEFAULT
        crop_rows = fetch_crop_qc_rows(crops_dir, regions, self._list_crop_kinds(), subjects, self.args.bids)
        rows = replace_missing_crop_reasons(crop_rows, skips, self.args.bids)
        qc_path = join(derivatives, OUTPUT_QC_FILENAME)
        write_qc_tsv(qc_path, rows)
        for row in rows:
            if row["qc"] == 0:
                logger.error(f"output QC excluded {row['participant_id']}: {row['reason']}")
        return qc_path, any(row["qc"] == 1 for row in rows)

    def _apply_output_qc(
        self, output_abs: str, status: int, skips: dict[str, list[SkipEntry]], has_crop_step: bool = False
    ) -> int:
        """Run the output QC: 1 with a FAIL record when no subject passed it, else ``status``.

        After a crop step (``has_crop_step``), a subject passing it lays the marker, whatever ``status`` is.
        """
        qc_path, has_complete_subject = self._write_output_qc(output_abs, skips)
        if not has_complete_subject:
            logger.fail(f"no subject passed the output QC, see {qc_path}")
            return 1
        if has_crop_step:
            self._write_crop_marker(output_abs)
        return status

    def _generate_mask_npys(self, output_abs: str) -> None:
        """Convert {side}mask_cropped.nii.gz → {side}mask.npy for every region.

        Scans the full crops tree so it works for partial runs (--regions) and
        any mask version.  Skips files that already exist.
        """
        pattern = join(output_abs, DERIVATIVES_FOLDER, "crops", "**", "mask", "*mask_cropped.nii.gz")
        nii_files = sorted(glob.glob(pattern, recursive=True))
        if not nii_files:
            logger.warning("no *mask_cropped.nii.gz files found — skipping.")
            return
        generated = 0
        for nii_path in nii_files:
            npy_path = nii_path.replace("mask_cropped.nii.gz", "mask.npy")
            if exists(npy_path):
                continue
            vol = aims.read(nii_path)
            np.save(npy_path, vol.np.astype(bool))
            logger.output(f"Generated {npy_path}")
            generated += 1
        logger.output(f"{generated} mask .npy file(s) written.")

    def default_log_path(self) -> str:
        """Run-log path of a bare --log: .log/cortical_tiles_<stamp>.log in the cortical tiles derivatives folder."""
        return compute_run_log_path(
            join(abspath(self.args.output), DERIVATIVES_FOLDER),
            "cortical_tiles",
            datetime.now().astimezone(),
        )

    def _build_crop_command(self, script_path: str, sk_qc_abs: str) -> list[str]:
        """Command line of generate_sulcal_regions.py for this run."""
        cmd = [
            sys.executable,
            script_path,
            "-d",
            abspath(self.args.output),
            "--path_to_graph",
            self.args.path_to_graph,
            "--path_sk_with_hull",
            self.args.path_sk_with_hull,
            "--njobs",
            str(self.args.njobs),
        ]

        if self.args.regions:
            cmd.extend(["-r", *self.args.regions])

        if sk_qc_abs:
            cmd.extend(["--sk_qc_path", sk_qc_abs])

        if self.args.skel_seed is not None:
            cmd.extend(["--skel_seed", str(self.args.skel_seed)])

        cmd.extend(["-y", *self._list_input_types()])
        return cmd

    def _write_crop_config(self, config_file_path: str, input_abs: str, sk_qc_abs: str) -> None:
        """Set the run's paths and options in the pipeline JSON config, when it exists."""
        # Set graphs_dir and output_dir in the pipeline JSON config.
        # generate_sulcal_regions.py derives both from path_dataset (-d arg)
        # when they are "$local". Since -d is now the subjects dir (not the
        # dataset root), we set them explicitly so outputs land in the right place.
        if not exists(config_file_path):
            return
        with open(config_file_path) as f:
            config = json.load(f)
        config["graphs_dir"] = input_abs
        config["output_dir"] = join(abspath(self.args.output), DERIVATIVES_FOLDER)
        config["brain_regions_json"] = abspath(
            join(dirname(__file__), "resources", "sulci_regions_champollion_V1.json")
        )
        config["path_to_graph"] = self.args.path_to_graph
        config["path_to_skeleton_with_hull"] = self.args.path_sk_with_hull
        config["masks_version"] = self.args.masks if self.args.masks else "canonical_25"
        config["skel_qc_path"] = sk_qc_abs
        if self.args.bids:
            config["bids"] = True
        config.pop("skip_distbottom", None)
        with open(config_file_path, "w") as f:
            json.dump(config, f, indent=3)

    def run(self):
        """Execute the cortical_tiles script."""
        logger.trace(f"input: {self.args.input}")
        logger.trace(f"output: {self.args.output}")

        # Validate paths
        if not self.validate_paths([self.args.input, self.args.output]):
            raise ValueError("run_cortical_tiles.py: Please input valid paths.")

        # Convert input to absolute path
        input_abs = abspath(self.args.input)
        output_abs = abspath(self.args.output)
        sk_qc_abs = abspath(self.args.sk_qc_path) if self.args.sk_qc_path else ""
        # TEMPORARY (DEF-CORTILES-BDRABCZUK-8A44DAD7F312, -9C7D2B9265BC)
        if not self._check_subject_ids(input_abs, output_abs):
            return 1

        qc_path, has_usable_subject, crop_qc_abs = self._write_input_qc(input_abs, output_abs, sk_qc_abs)
        if not has_usable_subject:
            logger.fail(f"no subject passed the input QC, see {qc_path}")
            return 1

        config_file_path: str = join(output_abs, "pipeline_loop_2mm.json")

        if not self._preflight_check(output_abs, config_file_path):
            save_config_skel_qc(config_file_path, crop_qc_abs)
            status = self._apply_output_qc(output_abs, 0, self._read_skip_lists(output_abs))
            return compute_end_status(status, crop_qc_abs)

        # Copy pipeline config template into the output (derivatives) directory.
        # generate_sulcal_regions.py reads it from {path_dataset}/pipeline_loop_2mm.json,
        # and we pass output as -d so the config is never inside the subjects directory
        # (which would cause generate_skeletons.py to list it as a subject).
        # Existence test, not validate_paths: a missing config is nominal on a first run and must not log ERROR.
        if not exists(config_file_path) or grants_no_write(config_file_path):
            source_config = abspath(join(dirname(__file__), "resources", "pipeline_loop_2mm.json"))
            replace_with_writable_copy(source_config, config_file_path)

        crop_sk_qc = crop_qc_abs or sk_qc_abs
        self._write_crop_config(config_file_path, input_abs, crop_sk_qc)

        # Prepare njobs
        if self.args.njobs is None:
            self.args.njobs = max(1, min(22, cpu_count() - 2))

        if self.args.njobs >= cpu_count():
            logger.warning(f"{self.args.njobs} jobs requested but only {cpu_count()} cores available.")

        # Build command to run cortical_tiles script directly
        # Get absolute paths
        script_path = abspath(
            join(
                dirname(__file__),
                "..",
                "..",
                "external",
                "cortical_tiles",
                "cortical_tiles",
                "brainvisa",
                "generate_sulcal_regions.py",
            )
        )

        # The script runs from its own directory (subprocess cwd)
        script_dir = dirname(script_path)

        cmd = self._build_crop_command(script_path, crop_sk_qc)

        with suppress(FileNotFoundError):  # not exists(): tests mock it
            os.remove(self._compute_crop_marker_path(output_abs))

        result = self.execute_command(cmd, shell=False, cwd=script_dir)

        if result not in (0, CROP_STATUS_PARTIAL):
            return result

        skips = self._read_skip_lists(output_abs)
        if result == CROP_STATUS_PARTIAL:
            self._log_skip_summary(skips)

        self._write_transform_qc(output_abs)  # TEMPORARY (DEF-CORTILES-BDRABCZUK-8A44DAD7F312)
        self._generate_mask_npys(output_abs)

        status = self._run_whole_brain(input_abs, output_abs, result)
        return compute_end_status(self._apply_output_qc(output_abs, status, skips, has_crop_step=True), crop_qc_abs)

    def _run_whole_brain(self, input_abs: str, output_abs: str, crop_status: int) -> int:
        """Fuse the L+R skeletons, then strip the ventricle from the fused volume; return the stage status so far.

        Runs after a complete or partial crop, alongside the per-region crops. Fusion comes first, since
        remove_ventricle(F) reads from the exact <skeleton_src_dir>/F/ tree add_left_and_right_volumes just
        wrote. Both steps always run; ``crop_status`` stands unless one of them failed.

        src_dir must point at the actual L/R resampled skeletons that generate_sulcal_regions.py (the crop
        subprocess) writes to {output}/cortical_tiles-{YEAR}/skeletons/{voxel_size}/{L,R}/ -- NOT input_abs
        (the Morphologist graphs directory, which has no L/ or R/ subdirs and is read-only). morpho_dir stays
        input_abs: remove_ventricle needs it separately to locate the labelled graphs used to strip the ventricle.
        """
        requested_cfg = CorticalTilesConfigFactory.from_args(self.args)
        vox_str = f"{int(requested_cfg.out_voxel_size)}mm"
        skeleton_src_dir = join(output_abs, DERIVATIVES_FOLDER, "skeletons", vox_str)

        is_fused = self._fuse_whole_brain(skeleton_src_dir)
        is_stripped = self._remove_ventricle_whole_brain(skeleton_src_dir, input_abs, output_abs)
        return crop_status if is_fused and is_stripped else self._compute_fusion_failure_status(output_abs)

    def _fuse_whole_brain(self, skeleton_src_dir: str) -> bool:
        """Fuse the L and R skeletons; False when the call raises or a subject with both sides has no fused volume."""
        is_done = call_whole_brain_step(
            "fusion",
            add_left_and_right_volumes.add_left_and_right_volumes,
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

    def _compute_fusion_failure_status(self, output_abs: str) -> int:
        """Status of a whole-brain failure: CROP_STATUS_PARTIAL when whole_brain/F holds a volume, else 1."""
        whole_brain_dir = join(output_abs, DERIVATIVES_FOLDER, "whole_brain")
        return CROP_STATUS_PARTIAL if list_side_volumes(whole_brain_dir, WHOLE_BRAIN_OUT_FILENAME) else 1

    def _remove_ventricle_whole_brain(self, skeleton_src_dir: str, input_abs: str, output_abs: str) -> bool:
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
        if self.args.labelling_session_explicit:
            candidates = (self.args.labelling_session,)
        else:
            candidates = LABELLING_SESSION_CANDIDATES

        fused = list_side_volumes(skeleton_src_dir, WHOLE_BRAIN_SRC_FILENAME)
        if not fused:
            logger.warning(f"No fused volume under {skeleton_src_dir}/F; whole-brain ventricle removal skipped.")
            return True

        qc_path, passed = self._write_labelling_qc(list(fused), input_abs, output_abs, candidates)
        groups, excluded = self._group_whole_brain_subjects(list(fused), input_abs, candidates)
        exclusions = exclusion_warning(excluded, self.args.path_to_graph, candidates) if excluded else ""
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
                remove_ventricle.remove_ventricle,
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
                bids=self.args.bids,
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

    def _group_whole_brain_subjects(
        self, subjects: list[str], input_abs: str, candidates: tuple[str, ...]
    ) -> tuple[dict[tuple[str, str], list[str]], dict[str, list[str]]]:
        """Group subjects by (labelling session, resolved graph path); sort the rest by exclusion reason."""
        groups: dict[tuple[str, str], list[str]] = {}
        excluded: dict[str, list[str]] = {}
        for subject in subjects:
            resolve = resolve_bids_graph_paths if self.args.bids else resolve_graph_paths
            graph_paths = resolve(input_abs, self.args.path_to_graph, subject)
            if not graph_paths:
                excluded.setdefault(EXCLUDED_NO_GRAPH_FOLDER, []).append(subject)
                continue
            if len(graph_paths) > 1:
                excluded.setdefault(EXCLUDED_SEVERAL_GRAPH_FOLDERS, []).append(subject)
                continue
            session = resolve_labelling_session(input_abs, graph_paths[0], subject, candidates, self.args.bids)
            if session is None:
                excluded.setdefault(EXCLUDED_NO_LABELLED_GRAPH, []).append(subject)
            else:
                groups.setdefault((session, graph_paths[0]), []).append(subject)
        return groups, excluded


def main():
    """Main entry point."""
    init_pipeline_process()
    script = RunCorticalTiles()
    return script.main()


if __name__ == "__main__":
    exit(main())
