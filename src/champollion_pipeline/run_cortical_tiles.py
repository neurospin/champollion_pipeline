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
import logging
import sys
from datetime import datetime
from os.path import abspath, dirname, exists, isdir, join

import numpy as np
from champollion_utils.script_builder import ScriptBuilder
from cortical_tiles.brainvisa import add_left_and_right_volumes, remove_ventricle
from cortical_tiles.brainvisa.utils.crop_subjects import EmptyCropError, exclude_empty_crops
from cortical_tiles.brainvisa.utils.skip_list import SKIP_LIST_FILENAME, SKIP_SIDES, SkipEntry, load_skip_entries
from joblib import cpu_count
from soma import aims

from champollion_pipeline.derivatives_layout import compute_run_log_path
from champollion_pipeline.generate_masks import _REGIONS_DEFAULT
from champollion_pipeline.process_setup import init_pipeline_process
from champollion_pipeline.run_cortical_tiles_inputs import (
    CROP_STATUS_PARTIAL,
    REASON_EXCLUDED_BY_SK_QC,
    compute_end_status,
    fetch_input_qc_rows,
    fetch_sk_qc_ids,
    find_dotted_ids,
    find_prefix_id_pairs,
)
from champollion_pipeline.run_cortical_tiles_whole_brain import (
    compute_fusion_failure_status,
    fuse_whole_brain,
    remove_ventricle_whole_brain,
)
from champollion_pipeline.utils.cortical_tiles_config import (
    INPUT_TYPES_DEFAULT,
    CorticalTilesConfig,
    CorticalTilesConfigFactory,
    CropPass,
    check_version_crops,
    compute_crop_targets,
    compute_missing_hemispheres,
    compute_versioned_crops_dir,
    merge_skip_lists,
    migrate_legacy_crops,
    plan_crop_passes,
    plan_unmapped_crop,
    read_marker_names,
    remove_marker,
    write_crop_config,
    write_marker_names,
)
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER, LABELLING_SESSION_CANDIDATES
from champollion_pipeline.utils.qc_report import (
    CROP_KIND_MASK,
    CROP_MASK_FOLDER,
    fetch_crop_qc_rows,
    fetch_grouped_crop_qc_rows,
    read_qc_tsv,
    replace_missing_crop_reasons,
    write_qc_tsv,
)
from champollion_pipeline.utils.region_masks import (
    REGION_MASKS_OPTION,
    add_region_masks_option,
    find_unmapped_region_models,
    refuse_masks_with_region_masks,
)
from champollion_pipeline.utils.resource_copy import grants_no_write, replace_with_writable_copy
from champollion_pipeline.utils.skeleton_qc import save_config_skel_qc, save_crop_qc
from champollion_pipeline.utils.transform_qc import transform_qc_rows

logger = logging.getLogger(__name__)

LABELLING_SESSION_DEFAULT = LABELLING_SESSION_CANDIDATES[0]
# TEMPORARY (DEF-CORTILES-BDRABCZUK-8A44DAD7F312): the prefix-ID guard below (and its flag, check and test file)
# goes once cortical_tiles stops globbing {subject}*/ in generate_ICBM2009c_transforms.py.
PREFIX_PAIRS_SHOWN = 5  # prefix pairs named in the refusal or warning
# TEMPORARY (DEF-CORTILES-BDRABCZUK-8A44DAD7F312): transform QC, a report of subjects whose ICBM2009c
# transform is byte-identical to another subject's; goes with the prefix-ID guard.
TRANSFORM_QC_FILENAME = "transform_qc.tsv"
INPUT_QC_FILENAME = "input_qc.tsv"
OUTPUT_QC_FILENAME = "output_qc.tsv"
SkipLists = dict[str, list[SkipEntry]]  # side -> skip-list entries


class _ExplicitLabellingSession(argparse.Action):
    """Store --labelling_session and record that the user gave it explicitly."""

    def __call__(self, parser, namespace, values, _option_string=None):
        namespace.labelling_session = values
        namespace.labelling_session_explicit = True


ROOT_FORBIDDEN_SUBSTRINGS = ("ses-", "acq-", "run-", "_cropped")


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
        add_region_masks_option(self)
        self.parser.set_defaults(labelling_session_explicit=False)

    def parse_args(self, args: list[str] | None = None) -> argparse.Namespace:
        """Parse; exit 2 on --masks beside --region-masks, or on a hemisphere of the run the map gives no version."""
        parsed = super().parse_args(args)
        refuse_masks_with_region_masks(self.parser, self.args)
        missing = find_unmapped_region_models(self.args.region_mask_versions or {}, self._list_run_regions())
        if self.args.region_mask_versions is not None and missing:
            self.parser.error(f"{REGION_MASKS_OPTION} gives no Mask version to {', '.join(missing)}")
        return parsed

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
        """Migrate legacy crops, then check the requested version's crops: False (no crop step) on a legacy
        conflict or, without --overwrite, on a marker listing each hemisphere of the run (path without a map)."""
        ((version, wanted),) = self._compute_crop_targets().items()
        return self._migrate_legacy_crops(output_abs, config_path) and bool(
            self._check_version_crops(output_abs, version, wanted)
        )

    def _check_version_crops(self, output_abs: str, version: str, wanted: list[str]) -> list[str]:
        """Names of ``wanted`` to crop with ``version`` (check_version_crops in the run's derivatives folder)."""
        crops = CorticalTilesConfig(version, CorticalTilesConfigFactory.from_args(self.args).out_voxel_size)
        derivatives = join(output_abs, DERIVATIVES_FOLDER)
        return check_version_crops(derivatives, crops, wanted, self.args.overwrite, logger)

    def _migrate_legacy_crops(self, output_abs: str, config_path: str) -> bool:
        """Move crops/<vox>/, written before the versioned path, under its masks version; False on a conflict."""
        vox_str = f"{int(CorticalTilesConfigFactory.from_args(self.args).out_voxel_size)}mm"
        derivatives_abs = join(output_abs, DERIVATIVES_FOLDER)
        return migrate_legacy_crops(config_path, derivatives_abs, vox_str, self.args.overwrite, logger)

    def _list_run_regions(self) -> list[str]:
        """Sulcal regions of the run: --regions, else the 28."""
        return list(self.args.regions or _REGIONS_DEFAULT)

    def _compute_crop_targets(self) -> dict[str, list[str]]:
        """Mask version -> region model names of the run cropped with it (the requested version without a map)."""
        masks = CorticalTilesConfigFactory.from_args(self.args).masks_version
        return compute_crop_targets(self.args.region_mask_versions, masks, self._list_run_regions())

    def _compute_crops_dir(self, output_abs: str, version: str) -> str:
        """Crops folder of ``version`` at the run's voxel size."""
        voxel_size = CorticalTilesConfigFactory.from_args(self.args).out_voxel_size
        return compute_versioned_crops_dir(join(output_abs, DERIVATIVES_FOLDER), version, voxel_size)

    def _plan_crops(self, output_abs: str, config_path: str) -> dict[str, list[str]] | None:
        """Mask version -> region model names this run crops with it; None when the crop step is skipped."""
        targets = self._compute_crop_targets()
        if self.args.region_mask_versions is None:
            if not self._preflight_check(output_abs, config_path):
                return None
            ((version, wanted),) = targets.items()
            prior = read_marker_names(self._compute_crops_dir(output_abs, version))
            return {version: compute_missing_hemispheres(wanted, prior, self.args.overwrite) or wanted}
        if not self._migrate_legacy_crops(output_abs, config_path):
            return None
        to_crop = {version: self._check_version_crops(output_abs, version, names) for version, names in targets.items()}
        return to_crop if any(to_crop.values()) else None

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

    def _read_skip_lists(self, output_abs: str) -> SkipLists:
        """Entries of the skip list of each side; a list its reader refuses gives one ERROR and no entry."""
        derivatives = join(output_abs, DERIVATIVES_FOLDER)
        skips: SkipLists = {}
        for side in SKIP_SIDES:
            try:
                skips[side] = load_skip_entries(derivatives, side)
            except ValueError as error:
                path = join(derivatives, SKIP_LIST_FILENAME.format(side=side))
                logger.error(f"skip list of side {side} not read, {path}: {error}")
                skips[side] = []
        return skips

    def _log_skip_summary(self, skips: SkipLists) -> None:
        """Log one ERROR record per side and step holding an entry, naming its sorted skipped subjects once each."""
        for side, entries in skips.items():
            subjects_per_step: dict[str, set[str]] = {}
            for entry in entries:
                subjects_per_step.setdefault(entry.step, set()).add(entry.subject)
            for step, subjects in subjects_per_step.items():
                logger.error(f"side {side} step {step} skipped subjects: {', '.join(sorted(subjects))}")

    def _write_output_qc(self, output_abs: str, skips: SkipLists) -> tuple[str, bool]:
        """Write output_qc.tsv (a hemisphere read in its Mask version's folder), one record per qc 0 row; return
        its path and whether a subject passed."""
        derivatives = join(output_abs, DERIVATIVES_FOLDER)
        input_records = read_qc_tsv(join(derivatives, INPUT_QC_FILENAME))
        subjects = [str(record["participant_id"]) for record in input_records if record["qc"] == 1]
        regions, kinds = self._list_run_regions(), self._list_crop_kinds()
        if self.args.region_mask_versions is None:
            (version,) = self._compute_crop_targets()
            crops_dir = self._compute_crops_dir(output_abs, version)
            crop_rows = fetch_crop_qc_rows(crops_dir, regions, kinds, subjects, self.args.bids)
        else:
            groups = [
                (self._compute_crops_dir(output_abs, crop.version), crop.side, crop.regions)
                for crop in plan_crop_passes(self._compute_crop_targets(), regions)
            ]
            crop_rows = fetch_grouped_crop_qc_rows(groups, kinds, subjects, self.args.bids)
        rows = replace_missing_crop_reasons(crop_rows, skips, self.args.bids)
        qc_path = join(derivatives, OUTPUT_QC_FILENAME)
        write_qc_tsv(qc_path, rows)
        for row in rows:
            if row["qc"] == 0:
                logger.error(f"output QC excluded {row['participant_id']}: {row['reason']}")
        return qc_path, any(row["qc"] == 1 for row in rows)

    def _apply_output_qc(
        self, output_abs: str, status: int, skips: SkipLists, markers: dict[str, frozenset[str]] | None = None
    ) -> int:
        """Run the output QC: 1 with a FAIL record when no subject passed it, else ``status``; a subject passing
        it writes ``markers`` (Mask version -> names listed), whatever ``status`` is."""
        qc_path, has_complete_subject = self._write_output_qc(output_abs, skips)
        if not has_complete_subject:
            logger.fail(f"no subject passed the output QC, see {qc_path}")
            return 1
        for version, names in (markers or {}).items():
            write_marker_names(self._compute_crops_dir(output_abs, version), names)
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

    def _build_crop_command(
        self, script_path: str, sk_qc_abs: str, crop_pass: CropPass | None = None, regions: list[str] | None = None
    ) -> list[str]:
        """Command line of generate_sulcal_regions.py: the run's (-r ``regions`` if given), or a crop pass's."""
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

        regions = list(crop_pass.regions) if crop_pass else regions or self.args.regions
        if regions:
            cmd.extend(["-r", *regions])
        if crop_pass:
            cmd.extend(["-i", crop_pass.side, "--masks", crop_pass.version])

        if sk_qc_abs:
            cmd.extend(["--sk_qc_path", sk_qc_abs])

        if self.args.skel_seed is not None:
            cmd.extend(["--skel_seed", str(self.args.skel_seed)])

        cmd.extend(["-y", *self._list_input_types()])
        return cmd

    def _run_crop_passes(
        self, script_path: str, sk_qc_abs: str, output_abs: str, to_crop: dict[str, list[str]]
    ) -> tuple[int, SkipLists, dict[str, frozenset[str]]]:
        """Delete the marker of each version to crop, run the crop subprocess(es) (one without a map, one per crop
        pass with one, up to a status outside (0, 3)), then drop empty-crop subjects; return the crop status, merged
        skip lists and, per version cropped, the names its marker lists after a passed output QC."""
        # Before the first crop subprocess: a run stopping in any pass leaves no marker listing a name it crops.
        priors = {v: remove_marker(self._compute_crops_dir(output_abs, v)) for v, names in to_crop.items() if names}
        script_dir = dirname(script_path)  # the script runs from its own directory
        if self.args.region_mask_versions is None:
            ((version, names),) = to_crop.items()
            regions, cropped = plan_unmapped_crop(names, self._list_run_regions())
            cmd = self._build_crop_command(script_path, sk_qc_abs, regions=regions)
            result = self.execute_command(cmd, shell=False, cwd=script_dir)
            return result, self._read_skip_lists(output_abs), {version: priors[version].union(cropped)}
        merged, done, status = {}, [], 0  # skip lists, crop passes run, crop status
        for crop_pass in plan_crop_passes(to_crop, self._list_run_regions()):
            cmd = self._build_crop_command(script_path, sk_qc_abs, crop_pass)
            result = self.execute_command(cmd, shell=False, cwd=script_dir)
            done.append(crop_pass)
            merge_skip_lists(merged, self._read_skip_lists(output_abs))
            status = max(status, result) if result in (0, CROP_STATUS_PARTIAL) else result
            if status not in (0, CROP_STATUS_PARTIAL):
                break
        self._exclude_empty_crops(output_abs, done, merged)
        return status, merged, {version: prior.union(to_crop[version]) for version, prior in priors.items()}

    def _exclude_empty_crops(self, output_abs: str, passes: list[CropPass], skips: SkipLists) -> None:
        """Drop each subject a skip list holds for an empty skeleton crop from the crops of every pass run."""
        empty = EmptyCropError.__name__
        dropped = sorted({entry.subject for entries in skips.values() for entry in entries if entry.exc_type == empty})
        if not dropped:
            return
        for crop_pass in passes:
            crops_dir = self._compute_crops_dir(output_abs, crop_pass.version)
            region_dirs = [join(crops_dir, region, CROP_MASK_FOLDER) for region in crop_pass.regions]
            exclude_empty_crops(region_dirs, dropped, [crop_pass.side])

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

        to_crop = self._plan_crops(output_abs, config_file_path)
        if to_crop is None:
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
        derivatives_abs = join(output_abs, DERIVATIVES_FOLDER)
        write_crop_config(config_file_path, input_abs, derivatives_abs, self.args, crop_sk_qc)

        if self.args.njobs is None:
            self.args.njobs = max(1, min(22, cpu_count() - 2))

        if self.args.njobs >= cpu_count():
            logger.warning(f"{self.args.njobs} jobs requested but only {cpu_count()} cores available.")

        brainvisa_dir = join(dirname(__file__), "..", "..", "external", "cortical_tiles", "cortical_tiles", "brainvisa")
        script_path = abspath(join(brainvisa_dir, "generate_sulcal_regions.py"))

        result, skips, markers = self._run_crop_passes(script_path, crop_sk_qc, output_abs, to_crop)
        if result not in (0, CROP_STATUS_PARTIAL):
            return result

        if result == CROP_STATUS_PARTIAL:
            self._log_skip_summary(skips)

        self._write_transform_qc(output_abs)  # TEMPORARY (DEF-CORTILES-BDRABCZUK-8A44DAD7F312)
        self._generate_mask_npys(output_abs)

        status = self._run_whole_brain(input_abs, output_abs, result)
        return compute_end_status(self._apply_output_qc(output_abs, status, skips, markers), crop_qc_abs)

    def _run_whole_brain(self, input_abs: str, output_abs: str, crop_status: int) -> int:
        """Fuse the L+R skeletons, then strip the ventricle (run_cortical_tiles_whole_brain); return the status."""
        requested_cfg = CorticalTilesConfigFactory.from_args(self.args)
        vox_str = f"{int(requested_cfg.out_voxel_size)}mm"
        skeleton_src_dir = join(output_abs, DERIVATIVES_FOLDER, "skeletons", vox_str)

        is_fused = self._fuse_whole_brain(skeleton_src_dir)
        is_stripped = self._remove_ventricle_whole_brain(skeleton_src_dir, input_abs, output_abs)
        return crop_status if is_fused and is_stripped else compute_fusion_failure_status(output_abs)

    def _fuse_whole_brain(self, skeleton_src_dir: str) -> bool:
        """Fuse the L and R skeletons (run_cortical_tiles_whole_brain.fuse_whole_brain)."""
        return fuse_whole_brain(skeleton_src_dir, add_left_and_right_volumes.add_left_and_right_volumes)

    def _remove_ventricle_whole_brain(self, skeleton_src_dir: str, input_abs: str, output_abs: str) -> bool:
        """Strip the ventricle from each fused volume (run_cortical_tiles_whole_brain.remove_ventricle_whole_brain)."""
        return remove_ventricle_whole_brain(
            self.args, skeleton_src_dir, input_abs, output_abs, remove_ventricle.remove_ventricle
        )


def main():
    """Main entry point."""
    init_pipeline_process()
    script = RunCorticalTiles()
    return script.main()


if __name__ == "__main__":
    exit(main())
