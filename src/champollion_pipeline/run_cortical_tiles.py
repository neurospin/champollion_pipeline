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
import glob
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime
from os.path import abspath, dirname, exists, isdir, join

import numpy as np
from champollion_utils.script_builder import ScriptBuilder
from cortical_tiles.brainvisa import add_left_and_right_volumes, remove_ventricle
from joblib import cpu_count
from soma import aims

from champollion_pipeline.derivatives_layout import compute_run_log_path
from champollion_pipeline.generate_labelling_qc import REASON_NO_SESSION_DIR, fetch_subject_row
from champollion_pipeline.process_setup import init_pipeline_process
from champollion_pipeline.utils.cortical_tiles_config import CorticalTilesConfigFactory, versioned_crops_exist
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER, LABELLING_SESSION_CANDIDATES
from champollion_pipeline.utils.qc_report import write_qc_tsv
from champollion_pipeline.utils.subject_name import subject_name_problem

logger = logging.getLogger(__name__)

LABELLING_SESSION_DEFAULT = LABELLING_SESSION_CANDIDATES[0]
WHOLE_BRAIN_SRC_FILENAME = "resampled_skeleton"
GRAPH_WILDCARD = "*"  # the only wildcard read in --path_to_graph
# Whole-brain exclusion reasons; tuple order is the clause order in the warning.
EXCLUDED_NO_GRAPH_FOLDER = "no_graph_folder"  # non-BIDS: --path_to_graph matches no folder
EXCLUDED_SEVERAL_GRAPH_FOLDERS = "several_graph_folders"  # non-BIDS: it matches two or more folders
EXCLUDED_NO_LABELLED_GRAPH = "no_labelled_graph"  # graph folder found, no candidate session dir in it
EXCLUSION_REASONS = (EXCLUDED_NO_GRAPH_FOLDER, EXCLUDED_SEVERAL_GRAPH_FOLDERS, EXCLUDED_NO_LABELLED_GRAPH)
LABELLING_QC_FILENAME = "labelling_qc.tsv"
LABELLING_QC_EXTRA_COLUMNS = ("labelled_L", "labelled_R")  # with write_qc_tsv: generate_labelling_qc.QC_COLUMNS
INPUT_TYPES_DEFAULT = ("skeleton", "foldlabel")
# TEMPORARY (DEF-CORTILES-BDRABCZUK-8A44DAD7F312): the prefix-ID guard below (and its flag, check and test file)
# goes once cortical_tiles stops globbing {subject}*/ in generate_ICBM2009c_transforms.py.
PREFIX_PAIRS_SHOWN = 5  # prefix pairs named in the refusal or warning
# TEMPORARY (DEF-CORTILES-BDRABCZUK-8A44DAD7F312): transform QC, a report of subjects whose ICBM2009c
# transform is byte-identical to another subject's; goes with the prefix-ID guard.
TRANSFORM_QC_FILENAME = "transform_qc.tsv"
TRANSFORM_SIDES = ("L", "R")
TRANSFORM_STEM = "transform_to_ICBM2009c_"  # files are <side><TRANSFORM_STEM><ID>.trm


class _ExplicitLabellingSession(argparse.Action):
    """Store --labelling_session and record that the user gave it explicitly."""

    def __call__(self, parser, namespace, values, _option_string=None):
        namespace.labelling_session = values
        namespace.labelling_session_explicit = True


ROOT_FORBIDDEN_SUBSTRINGS = ("ses-", "acq-", "run-", "_cropped")


def transform_qc_rows(transforms_dir: str) -> list[dict[str, object]]:
    """QC rows of the ICBM2009c transforms in <transforms_dir>/<side>/, one per subject, sorted by ID.

    A subject fails (qc 0) when its transform file on a side has exactly the same bytes as another
    subject's on that side; L and R are never compared with each other. Pure: no logging, no write.
    """
    flagged: dict[str, list[str]] = {}
    subjects: set[str] = set()
    for side in TRANSFORM_SIDES:
        side_dir = join(transforms_dir, side)
        if not isdir(side_dir):
            continue
        prefix = side + TRANSFORM_STEM
        groups: dict[bytes, list[str]] = {}
        with os.scandir(side_dir) as entries:
            for entry in entries:
                name = entry.name
                if not (entry.is_file() and name.startswith(prefix) and name.endswith(".trm")):
                    continue
                subject = name[len(prefix) : -len(".trm")]
                if not subject:
                    continue
                with open(entry.path, "rb") as handle:
                    groups.setdefault(handle.read(), []).append(subject)
                subjects.add(subject)
        for group in groups.values():
            for subject in group:
                others = sorted(other for other in group if other != subject)
                if others:
                    flagged.setdefault(subject, []).append(f"{side} transform byte-identical to {', '.join(others)}")
    return [
        {"participant_id": subject, "qc": 0 if subject in flagged else 1, "reason": "; ".join(flagged.get(subject, []))}
        for subject in sorted(subjects)
    ]


def list_fused_subjects(skeleton_src_dir: str, src_filename: str) -> dict[str, str]:
    """Map each subject with a fused volume in <skeleton_src_dir>/F/ to its absolute path.

    The subject id is what follows ``F<src_filename>`` in the file name, with one
    leading ``_`` stripped (cortical_tiles' own subject-id rule). Missing F/ gives {}.
    """
    f_dir = join(skeleton_src_dir, "F")
    if not isdir(f_dir):
        return {}
    pattern = re.compile(rf"^F{re.escape(src_filename)}(.*)\.nii\.gz$")
    fused = {}
    for name in sorted(os.listdir(f_dir)):
        match = pattern.match(name)
        if match:
            subject = match.group(1)
            fused[subject[1:] if subject.startswith("_") else subject] = abspath(join(f_dir, name))
    return fused


def labelled_session_dir(morpho_dir: str, path_to_graph: str, subject: str, session: str, bids: bool) -> str:
    """Directory holding a subject's labelled graphs for one session.

    Mirrors cortical_tiles' RemoveVentricleFromVolume.get_labelled_graph: with BIDS
    names, the subject id splits into the subject folder and the keys that replace
    ``*`` in path_to_graph.
    """
    if bids:
        parts = subject.split("_")
        return join(morpho_dir, parts[0], path_to_graph.replace("*", "_".join(parts[1:])), session)
    return join(morpho_dir, subject, path_to_graph, session)


def resolve_labelling_session(
    morpho_dir: str, path_to_graph: str, subject: str, candidates: tuple[str, ...], bids: bool
) -> str | None:
    """First candidate session whose labelled-graph directory exists for the subject, else None."""
    for session in candidates:
        if isdir(labelled_session_dir(morpho_dir, path_to_graph, subject, session, bids)):
            return session
    return None


def resolve_graph_paths(morpho_dir: str, path_to_graph: str, subject: str) -> list[str]:
    """Folders under ``<morpho_dir>/<subject>`` that path_to_graph matches (non-BIDS).

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


def fetch_labelling_qc_row(
    morpho_dir: str, path_to_graph: str, subject: str, candidates: tuple[str, ...], bids: bool
) -> dict[str, str]:
    """labelling_qc.tsv row of a fused subject, at the graphs whole-brain ventricle removal reads.

    With BIDS names the fused id splits into the subject folder and the keys that replace ``*``,
    as in cortical_tiles' get_labelled_graph. Without them ``*`` must match exactly one folder;
    none or several gives what generate_labelling_qc computes there, which never expands it.
    """
    if bids:
        folder, _, keys = subject.partition("_")
        row = fetch_subject_row(morpho_dir, folder, path_to_graph.replace(GRAPH_WILDCARD, keys), candidates, False)
        return {**row, "participant_id": subject}
    graph_paths = resolve_graph_paths(morpho_dir, path_to_graph, subject)
    if len(graph_paths) == 1:
        return fetch_subject_row(morpho_dir, subject, graph_paths[0], candidates, False)
    return {
        "participant_id": subject,
        "qc": "0",
        "labelled_L": "False",
        "labelled_R": "False",
        "reason": REASON_NO_SESSION_DIR,
    }


# TEMPORARY (DEF-CORTILES-BDRABCZUK-9C7D2B9265BC)
def find_dotted_ids(input_dir: str) -> list[str]:
    """Sorted input subject IDs (non-dot subdirectory names) that contain a dot; none for a non-directory."""
    if not isdir(input_dir):
        return []
    return sorted(e.name for e in os.scandir(input_dir) if e.is_dir() and not e.name.startswith(".") and "." in e.name)


# TEMPORARY (DEF-CORTILES-BDRABCZUK-8A44DAD7F312)
def find_prefix_id_pairs(input_dir: str) -> list[tuple[str, str]]:
    """Every (shorter, longer) pair of input subject IDs where one strictly prefixes the other, sorted.

    Subject IDs are the non-dot subdirectory names of ``input_dir``, as cortical_tiles lists them
    (with or without --bids). An input that is not a directory gives no pair.
    """
    if not isdir(input_dir):
        return []
    names = sorted(e.name for e in os.scandir(input_dir) if e.is_dir() and not e.name.startswith("."))
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
            .add_optional_argument("--njobs", "Number of CPU cores allowed to use.", default=None, type_=int)
            .add_argument(
                "--input-types",
                nargs="+",
                default=None,
                help=(
                    "Input types to generate (e.g. skeleton foldlabel extremities). "
                    "Default: skeleton foldlabel (extremities only on request)."
                ),
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
        _distbottom_group = self.parser.add_mutually_exclusive_group()
        _distbottom_group.add_argument(
            "--with-distbottom",
            action="store_true",
            help="Generate distbottom crops (off by default; unused by champollion_V1).",
        )
        _distbottom_group.add_argument(
            "--skip-distbottom",
            action="store_true",
            help="Deprecated: distbottom is skipped by default. Kept for backward compatibility.",
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

        Returns False (caller should abort) when crops for the requested mask
        version already exist and --overwrite was not passed.
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
            if not self.args.overwrite:
                logger.warning(
                    f"Crops for mask version '{requested_cfg.masks_version}' already exist. "
                    "Use --overwrite to regenerate."
                )
                return False
            logger.warning(f"Overwriting existing crops for '{requested_cfg.masks_version}'.")
        return True

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
    ) -> tuple[str, bool]:
        """Write labelling_qc.tsv, one ERROR per failing subject; return its path and whether a subject passed."""
        rows = [
            fetch_labelling_qc_row(input_abs, self.args.path_to_graph, subject, candidates, self.args.bids)
            for subject in sorted(subjects)
        ]
        qc_path = join(output_abs, DERIVATIVES_FOLDER, LABELLING_QC_FILENAME)
        write_qc_tsv(qc_path, rows, extra_columns=LABELLING_QC_EXTRA_COLUMNS)
        for row in rows:
            if row["qc"] == "0":
                logger.error(f"labelling QC excluded {row['participant_id']}: {row['reason']}")
        return qc_path, any(row["qc"] == "1" for row in rows)

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

    def run(self):  # noqa: PLR0915 - linear stage sequence; split deferred to the guard/QC removal task
        """Execute the cortical_tiles script."""
        logger.trace(f"input: {self.args.input}")
        logger.trace(f"output: {self.args.output}")

        # Validate paths
        if not self.validate_paths([self.args.input, self.args.output]):
            raise ValueError("run_cortical_tiles.py: Please input valid paths.")

        # Convert input to absolute path
        input_abs = abspath(self.args.input)
        output_abs = abspath(self.args.output)
        # TEMPORARY (DEF-CORTILES-BDRABCZUK-8A44DAD7F312, -9C7D2B9265BC)
        if not self._check_subject_ids(input_abs, output_abs):
            return 1

        config_file_path: str = join(output_abs, "pipeline_loop_2mm.json")

        if not self._preflight_check(output_abs, config_file_path):
            return 0

        # Copy pipeline config template into the output (derivatives) directory.
        # generate_sulcal_regions.py reads it from {path_dataset}/pipeline_loop_2mm.json,
        # and we pass output as -d so the config is never inside the subjects directory
        # (which would cause generate_skeletons.py to list it as a subject).
        if not self.validate_paths([config_file_path]):
            source_config = abspath(join(dirname(__file__), "..", "..", "pipeline_loop_2mm.json"))
            self.execute_command(["cp", source_config, config_file_path], shell=False)

        # Set graphs_dir and output_dir in the pipeline JSON config.
        # generate_sulcal_regions.py derives both from path_dataset (-d arg)
        # when they are "$local". Since -d is now the subjects dir (not the
        # dataset root), we set them explicitly so outputs land in the right place.
        if exists(config_file_path):
            with open(config_file_path) as f:
                config = json.load(f)
            config["graphs_dir"] = input_abs
            config["output_dir"] = join(abspath(self.args.output), DERIVATIVES_FOLDER)
            config["path_to_graph"] = self.args.path_to_graph
            config["path_to_skeleton_with_hull"] = self.args.path_sk_with_hull
            config["masks_version"] = self.args.masks if self.args.masks else "canonical_25"
            config["skel_qc_path"] = self.args.sk_qc_path if self.args.sk_qc_path else ""
            if self.args.bids:
                config["bids"] = True
            config["skip_distbottom"] = not self.args.with_distbottom
            with open(config_file_path, "w") as f:
                json.dump(config, f, indent=3)

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

        if self.args.sk_qc_path:
            cmd.extend(["--sk_qc_path", self.args.sk_qc_path])

        if self.args.skel_seed is not None:
            cmd.extend(["--skel_seed", str(self.args.skel_seed)])

        input_types = self.args.input_types or list(INPUT_TYPES_DEFAULT)
        cmd.extend(["-y", *input_types])

        result = self.execute_command(cmd, shell=False, cwd=script_dir)

        if result != 0:
            return result

        self._write_transform_qc(output_abs)  # TEMPORARY (DEF-CORTILES-BDRABCZUK-8A44DAD7F312)
        self._generate_mask_npys(output_abs)

        # Whole-brain volume: fuse L+R skeletons, then strip the ventricle
        # from the fused volume. Runs only after a successful crop (see the
        # guard above), alongside the per-region crops above -- fuse first,
        # since remove_ventricle(F)
        # reads from the exact <skeleton_src_dir>/F/ tree
        # add_left_and_right_volumes just wrote.
        #
        # src_dir must point at the actual L/R resampled skeletons that
        # generate_sulcal_regions.py (the crop subprocess above) writes to
        # {output}/cortical_tiles-{YEAR}/skeletons/{voxel_size}/{L,R}/ --
        # NOT input_abs (the Morphologist graphs directory, which has no
        # L/ or R/ subdirs and is read-only). morpho_dir stays input_abs:
        # remove_ventricle needs it separately to locate the labelled
        # graphs used to strip the ventricle.
        requested_cfg = CorticalTilesConfigFactory.from_args(self.args)
        vox_str = f"{int(requested_cfg.out_voxel_size)}mm"
        skeleton_src_dir = join(output_abs, DERIVATIVES_FOLDER, "skeletons", vox_str)

        add_left_and_right_volumes.add_left_and_right_volumes(
            src_dir=skeleton_src_dir,
            parallel=True,
            # Its own -1 default breaks on the string-only revalidation in
            # get_number_subjects in the cortical_tiles brainvisa subjects utils,
            # so pass "all" explicitly to route around that upstream quirk,
            # which lives under external/ and is not ours to fix.
            number_subjects="all",
        )
        self._remove_ventricle_whole_brain(skeleton_src_dir, input_abs, output_abs)

        return result

    def _remove_ventricle_whole_brain(self, skeleton_src_dir: str, input_abs: str, output_abs: str) -> None:
        """Strip the ventricle from each fused volume, grouping subjects by labelling session.

        remove_ventricle takes one labelling session per call and enumerates
        subjects from <src_dir>/F/, so when subjects resolve to different
        sessions (or some have none) each group gets a staged F/ of symlinks.
        Without --bids, a ``*`` in path_to_graph is resolved per subject, so
        groups are keyed by (session, graph path). Subjects that cannot be
        resolved are left out with one warning; the crops are unaffected and
        run() still returns the crop result. labelling_qc.tsv is written first;
        with no qc 1 row, one warning names it and no volume is processed.
        """
        if self.args.labelling_session_explicit:
            candidates = (self.args.labelling_session,)
        else:
            candidates = LABELLING_SESSION_CANDIDATES

        fused = list_fused_subjects(skeleton_src_dir, WHOLE_BRAIN_SRC_FILENAME)
        if not fused:
            logger.warning(f"No fused volume under {skeleton_src_dir}/F; whole-brain ventricle removal skipped.")
            return

        qc_path, has_usable_subject = self._write_labelling_qc(list(fused), input_abs, output_abs, candidates)
        groups, excluded = self._group_whole_brain_subjects(list(fused), input_abs, candidates)
        exclusions = exclusion_warning(excluded, self.args.path_to_graph, candidates) if excluded else ""
        if not has_usable_subject:
            left_out = f"Whole-brain volume left out: no subject passed the labelling pre-check, see {qc_path}."
            logger.warning(f"{left_out} {exclusions}".rstrip())
            return
        if exclusions:
            logger.warning(exclusions)

        def call(src_dir: str, session: str, graph_path: str) -> None:
            remove_ventricle.remove_ventricle(
                side="F",
                src_dir=src_dir,
                morpho_dir=input_abs,
                path_to_graph=graph_path,
                # add_left_and_right_volumes' default output_filename
                # ("resampled_skeleton") does not match remove_ventricle's
                # default src_filename ("skeleton_generated_") -- pass it
                # explicitly so remove_ventricle reads what was just fused.
                src_filename=WHOLE_BRAIN_SRC_FILENAME,
                parallel=True,
                output_dir=join(output_abs, DERIVATIVES_FOLDER, "whole_brain"),
                transform_dir=join(output_abs, DERIVATIVES_FOLDER, "transforms"),
                labelling_session=session,
                bids=self.args.bids,
            )

        if len(groups) == 1 and not excluded:
            call(skeleton_src_dir, *next(iter(groups)))
            return

        for session, graph_path in sorted(groups, key=lambda key: (candidates.index(key[0]), key[1])):
            with tempfile.TemporaryDirectory(prefix="whole_brain_") as staged:
                os.makedirs(join(staged, "F"))
                for subject in groups[(session, graph_path)]:
                    os.symlink(fused[subject], join(staged, "F", os.path.basename(fused[subject])))
                call(staged, session, graph_path)

    def _group_whole_brain_subjects(
        self, subjects: list[str], input_abs: str, candidates: tuple[str, ...]
    ) -> tuple[dict[tuple[str, str], list[str]], dict[str, list[str]]]:
        """Group subjects by (labelling session, resolved graph path); sort the rest by exclusion reason."""
        groups: dict[tuple[str, str], list[str]] = {}
        excluded: dict[str, list[str]] = {}
        for subject in subjects:
            if self.args.bids:
                graph_paths = [self.args.path_to_graph]
            else:
                graph_paths = resolve_graph_paths(input_abs, self.args.path_to_graph, subject)
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
