#!/usr/bin/env python3
"""
Stage 2 output QC tests of run_cortical_tiles.py: output_qc.tsv and the exit policy (user decision
2026-11-04, "Fail only if nothing usable"). Synthetic fixtures only.

The crop files are laid by the mocked crop step, the mask arrays by the mocked mask step. A subject
excluded by the input QC gets no row whether or not it stays in the crop files. The skip lists are
written by the record_skip of the cortical_tiles version pinned by the gitlink, so their format is
the real one. TestCropChild alone runs a real child process, a stand-in for generate_sulcal_regions.py.
"""

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cortical_tiles.brainvisa.utils.skip_list import SKIP_LIST_FILENAME, record_skip

from champollion_pipeline.generate_masks import _REGIONS_DEFAULT
from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from champollion_pipeline.utils.cortical_tiles_config import compute_versioned_crops_dir
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER, LABELLING_SESSION_CANDIDATES
from champollion_pipeline.utils.qc_report import read_qc_tsv
from tests.log_assertions import FAIL
from tests.stage2_layout import GRAPH_DIR, TEMPLATE_CONFIG, add_crops, add_graphs, crop_csv

_MODULE = "champollion_pipeline.run_cortical_tiles"
_LOGGER_NAME = _MODULE
_QC_FILE = "output_qc.tsv"
_CROP_SCRIPT = "generate_sulcal_regions.py"
_TRANSFORM_QC_FILE = "transform_qc.tsv"
_PARTIAL_STATUS = 3
_STOPPING_STATUSES = (1, 2, 4)
_MASKS_DEFAULT = "canonical_25"
_MASKS_OTHER = "other_masks"
_REGIONS = ("S.C.-sylv.", "S.T.s.")
_DEFAULT_KINDS = ("skeleton", "foldlabel")
_SUBJECT = "sub-01"
_OTHER_SUBJECT = "sub-02"
_COHORT = {_SUBJECT: "LR", _OTHER_SUBJECT: "LR"}
_COHORT_OF_THREE = {**_COHORT, "sub-03": "LR"}
_OUTPUT_QC_RECORD_START = "output QC excluded"
_STEP = "generate_foldlabels"
_OTHER_STEP = "generate_crops"
_CHILD_LINE = "stand-in crop script output"
_CHILD_SOURCE = """\
import shutil
import sys

shutil.copytree(sys.argv[1], sys.argv[2], dirs_exist_ok=True)
print(sys.argv[3], flush=True)
sys.exit(int(sys.argv[4]))
"""


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage2_output_qc():
    """Run the real output QC: overrides the conftest fixture of the same name."""


@dataclass
class _StageRun:
    """What run() did with the crop subprocess, the mask step and the whole-brain steps mocked."""

    result: int
    qc_path: Path
    fuse: MagicMock
    crop_runs: list[str]
    mask_step: MagicMock
    remove: MagicMock
    report_at_whole_brain: list[bool]

    def rows(self):
        """Rows of output_qc.tsv, keyed by participant_id."""
        assert self.qc_path.is_file(), f"{self.qc_path} was not written"
        return {row["participant_id"]: row for row in read_qc_tsv(self.qc_path)}


@dataclass(frozen=True)
class _Skip:
    """One skip-list entry: the crop step records it and leaves ``subject`` out of the subject CSVs of ``side``."""

    subject: str = _OTHER_SUBJECT
    side: str = "L"
    step: str = _STEP
    message: str = "no fold graph"

    def reason(self):
        """The output_qc.tsv reason of this entry, the recorded exception being a ValueError."""
        return f"skipped:{self.side}:{self.step}:ValueError:{self.message}"


@dataclass(frozen=True)
class _CropStep:
    """What the mocked crop and mask steps do: ``crop_subjects`` None takes the subjects of the run."""

    crop_subjects: list[str] | None = None
    crop_regions: tuple[str, ...] = _REGIONS
    kinds: tuple[str, ...] = _DEFAULT_KINDS
    crop_status: int = 0
    has_masks: bool = True
    masks_version: str = _MASKS_DEFAULT
    has_existing_crops: bool = False
    has_fused_subject: bool = False
    legacy_version: str | None = None
    skips: tuple[_Skip, ...] = ()
    malformed_sides: str = ""


def _add_legacy_conflict(output_dir, derivatives, version):
    """Lay legacy crops/2mm/ beside crops/<version>/2mm/, with a pipeline_loop_2mm.json naming ``version``."""
    for folder in (Path(derivatives) / "crops" / "2mm", Path(derivatives) / "crops" / version / "2mm"):
        folder.mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / TEMPLATE_CONFIG.name).write_text(json.dumps({"masks_version": version}))


def _add_fused_subject(input_dir, derivatives, subject):
    """Give ``subject`` labelled graphs and the fused volume that whole-brain ventricle removal reads."""
    session = LABELLING_SESSION_CANDIDATES[0]
    session_dir = Path(input_dir) / subject / GRAPH_DIR / session
    session_dir.mkdir()
    for side in "LR":
        (session_dir / f"{side}{subject}_{session}.arg").touch()
    fused_dir = Path(derivatives).resolve() / "skeletons" / "2mm" / "F"
    fused_dir.mkdir(parents=True, exist_ok=True)
    (fused_dir / f"Fresampled_skeleton_{subject}.nii.gz").touch()


def _add_skips(derivatives, crops_dir, step):
    """Write the skip lists of ``step`` under ``derivatives``, dropping each entry's subject from its side's CSVs."""
    for side in step.malformed_sides:
        (derivatives / SKIP_LIST_FILENAME.format(side=side)).write_text("not json\n")
    for skip in step.skips:
        record_skip(derivatives, skip.side, skip.step, skip.subject, ValueError(skip.message))
        for region in step.crop_regions:
            for kind in step.kinds:
                path = crop_csv(crops_dir, region, skip.side, kind)
                kept = [line for line in path.read_text().splitlines() if line != skip.subject]
                path.write_text("\n".join(kept) + "\n")


def _run_stage(temp_dir, subjects, extra_args=None, **crop_step):
    """Run run() on ``subjects`` (``{subject: hemispheres holding a graph}``); ``crop_step``: fields of _CropStep.

    The mocked crop step returns the crop status and lays the crops under the crops folder of the masks version;
    the mocked mask step lays the mask arrays. With ``has_existing_crops`` both lie on disk before run() starts,
    with ``has_fused_subject`` the crop step also leaves each subject a volume for whole-brain ventricle removal,
    with ``legacy_version`` legacy crops conflict with the crops of that version, with ``skips`` it writes
    those skip-list entries (before run() with ``has_existing_crops``), with ``malformed_sides`` a skip list refused
    by its reader on these sides.
    ``extra_args`` defaults to --regions with ``_REGIONS``.
    """
    step = _CropStep(**crop_step)
    root = Path(temp_dir)
    input_dir, output_dir = root / "input", root / "output"
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    for subject, sides in subjects.items():
        add_graphs(input_dir, subject, GRAPH_DIR, sides)
    derivatives = output_dir / DERIVATIVES_FOLDER
    crops_dir = compute_versioned_crops_dir(str(derivatives), step.masks_version)
    listed = list(subjects) if step.crop_subjects is None else step.crop_subjects

    crop_runs, report_at_whole_brain = [], []
    if step.legacy_version:
        _add_legacy_conflict(output_dir, derivatives, step.legacy_version)
    if step.has_existing_crops:
        add_crops(crops_dir, listed, step.crop_regions, (*step.kinds, *(("mask",) if step.has_masks else ())))
        _add_skips(derivatives, crops_dir, step)

    def execute(command, *_args, **_kwargs):
        if any(_CROP_SCRIPT in str(part) for part in command):
            crop_runs.append(_CROP_SCRIPT)
            add_crops(crops_dir, listed, step.crop_regions, step.kinds)
            _add_skips(derivatives, crops_dir, step)
            for subject in subjects if step.has_fused_subject else ():
                _add_fused_subject(input_dir, derivatives, subject)
        return step.crop_status

    def note_report(*_args, **_kwargs):
        report_at_whole_brain.append((derivatives / _QC_FILE).exists())

    def write_masks(_output_abs):
        if step.has_masks:
            add_crops(crops_dir, listed, step.crop_regions, ("mask",))

    script = RunCorticalTiles()
    script.parse_args(
        [
            str(input_dir),
            str(output_dir),
            "--path_to_graph",
            GRAPH_DIR,
            "--path_sk_with_hull",
            "skeleton/path",
            *(["--regions", *_REGIONS] if extra_args is None else extra_args),
        ]
    )
    with (
        patch.object(script, "validate_paths", return_value=True),
        patch.object(script, "execute_command", side_effect=execute),
        patch.object(script, "_generate_mask_npys", side_effect=write_masks) as mask_step,
        patch(
            "champollion_pipeline.run_cortical_tiles.add_left_and_right_volumes.add_left_and_right_volumes",
            side_effect=note_report,
        ) as fuse,
        patch(
            "champollion_pipeline.run_cortical_tiles.remove_ventricle.remove_ventricle",
            side_effect=note_report,
        ) as remove,
    ):
        result = script.run()
    return _StageRun(result, derivatives / _QC_FILE, fuse, crop_runs, mask_step, remove, report_at_whole_brain)


def _messages_at(level, pipeline_caplog):
    """Messages of the run_cortical_tiles logger records at exactly ``level``."""
    return [r.getMessage() for r in pipeline_caplog.records if r.name == _LOGGER_NAME and r.levelno == level]


def _summary_records(pipeline_caplog):
    """ERROR records of the logger, the one record per qc 0 row of output_qc.tsv left out."""
    errors = _messages_at(logging.ERROR, pipeline_caplog)
    return [message for message in errors if not message.startswith(_OUTPUT_QC_RECORD_START)]


def _record_of(records, side, step):
    """The only record naming ``side`` as a word and ``step``."""
    found = [message for message in records if re.search(rf"\b{side}\b", message) and step in message]
    assert len(found) == 1, records
    return found[0]


class TestReport:
    def test_one_row_per_qc1_input_row(self, temp_dir):
        assert sorted(_run_stage(temp_dir, _COHORT).rows()) == sorted(_COHORT)

    @pytest.mark.parametrize(
        "crop_subjects",
        [[_SUBJECT, _OTHER_SUBJECT], [_SUBJECT]],
        ids=["left_in_the_crops", "left_out_of_the_crops"],
    )
    def test_subject_excluded_by_input_qc_gets_no_row(self, temp_dir, crop_subjects):
        run = _run_stage(temp_dir, {_SUBJECT: "LR", _OTHER_SUBJECT: "L"}, crop_subjects=crop_subjects)

        assert list(run.rows()) == [_SUBJECT]
        assert run.rows()[_SUBJECT]["qc"] == 1

    def test_masks_written_by_the_mask_step_count(self, temp_dir):
        assert _run_stage(temp_dir, {_SUBJECT: "LR"}).rows()[_SUBJECT]["qc"] == 1

    def test_no_report_when_the_crop_step_fails(self, temp_dir):
        run = _run_stage(temp_dir, _COHORT, crop_status=2)

        assert run.result == 2
        assert not run.qc_path.exists()

    def test_no_report_without_a_qc1_input_row(self, temp_dir):
        run = _run_stage(temp_dir, {_SUBJECT: "L"})

        assert run.result == 1
        assert not run.qc_path.exists()


class TestRows:
    def test_complete_subject_has_qc_1(self, temp_dir):
        assert _run_stage(temp_dir, _COHORT, crop_subjects=[_SUBJECT]).rows()[_SUBJECT]["qc"] == 1

    def test_incomplete_subject_has_qc_0(self, temp_dir):
        assert _run_stage(temp_dir, _COHORT, crop_subjects=[_SUBJECT]).rows()[_OTHER_SUBJECT]["qc"] == 0

    def test_complete_subject_has_an_empty_reason(self, temp_dir):
        assert _run_stage(temp_dir, _COHORT, crop_subjects=[_SUBJECT]).rows()[_SUBJECT]["reason"] == ""

    def test_incomplete_subject_has_the_missing_crop_reason(self, temp_dir):
        rows = _run_stage(temp_dir, _COHORT, crop_subjects=[_SUBJECT]).rows()

        expected = [
            f"missing_crop:{kind}:{side}:{region}" for region in _REGIONS for side in "LR" for kind in _DEFAULT_KINDS
        ]
        assert rows[_OTHER_SUBJECT]["reason"] == ";".join(expected)


class TestCheckedKinds:
    def test_default_input_types_need_the_label_files(self, temp_dir):
        row = _run_stage(temp_dir, {_SUBJECT: "LR"}, kinds=("skeleton",)).rows()[_SUBJECT]

        assert row["qc"] == 0
        assert f"missing_crop:foldlabel:L:{_REGIONS[0]}" in row["reason"]

    def test_skeleton_only_run_needs_no_label_files(self, temp_dir):
        args = ["--regions", *_REGIONS, "--input-types", "skeleton"]

        assert _run_stage(temp_dir, {_SUBJECT: "LR"}, args, kinds=("skeleton",)).rows()[_SUBJECT]["qc"] == 1

    def test_extremities_on_request_need_their_files(self, temp_dir):
        args = ["--regions", *_REGIONS, "--input-types", "skeleton", "extremities"]

        row = _run_stage(temp_dir, {_SUBJECT: "LR"}, args, kinds=("skeleton",)).rows()[_SUBJECT]

        assert f"missing_crop:extremities:R:{_REGIONS[1]}" in row["reason"]

    def test_absent_mask_npy_fails_the_subject(self, temp_dir):
        row = _run_stage(temp_dir, {_SUBJECT: "LR"}, has_masks=False).rows()[_SUBJECT]

        assert row["qc"] == 0
        assert f"missing_crop:mask:L:{_REGIONS[0]}" in row["reason"]

    def test_with_distbottom_needs_the_distbottom_files(self, temp_dir):
        args = ["--regions", *_REGIONS, "--with-distbottom"]

        row = _run_stage(temp_dir, {_SUBJECT: "LR"}, args).rows()[_SUBJECT]

        assert f"missing_crop:distbottom:L:{_REGIONS[0]}" in row["reason"]

    def test_default_run_needs_no_distbottom_files(self, temp_dir):
        assert _run_stage(temp_dir, {_SUBJECT: "LR"}).rows()[_SUBJECT]["reason"] == ""

    def test_with_distbottom_without_skeleton_needs_no_distbottom_files(self, temp_dir):
        args = ["--regions", *_REGIONS, "--input-types", "foldlabel", "--with-distbottom"]

        assert _run_stage(temp_dir, {_SUBJECT: "LR"}, args, kinds=("foldlabel",)).rows()[_SUBJECT]["qc"] == 1


class TestCheckedRegions:
    def test_region_outside_regions_is_not_checked(self, temp_dir):
        args = ["--regions", _REGIONS[0]]

        run = _run_stage(temp_dir, {_SUBJECT: "LR"}, args, crop_regions=_REGIONS[:1])

        assert run.rows()[_SUBJECT]["qc"] == 1

    def test_default_regions_are_the_28_regions(self, temp_dir):
        assert len(_REGIONS_DEFAULT) == 28

        run = _run_stage(temp_dir, {_SUBJECT: "LR"}, [], crop_regions=_REGIONS_DEFAULT)

        assert run.rows()[_SUBJECT]["qc"] == 1

    def test_default_run_reports_a_lost_region(self, temp_dir):
        lost = _REGIONS_DEFAULT[-1]

        row = _run_stage(temp_dir, {_SUBJECT: "LR"}, [], crop_regions=_REGIONS_DEFAULT[:-1]).rows()[_SUBJECT]

        assert f"missing_crop:skeleton:L:{lost}" in row["reason"]

    def test_masks_option_selects_the_crops_folder(self, temp_dir):
        args = ["--regions", *_REGIONS, "--masks", _MASKS_OTHER]

        run = _run_stage(temp_dir, {_SUBJECT: "LR"}, args, masks_version=_MASKS_OTHER)

        assert run.rows()[_SUBJECT]["qc"] == 1

    def test_crops_of_another_mask_version_do_not_count(self, temp_dir):
        args = ["--regions", *_REGIONS, "--masks", _MASKS_OTHER]

        assert _run_stage(temp_dir, {_SUBJECT: "LR"}, args).rows()[_SUBJECT]["qc"] == 0


class TestExistingCrops:
    """A rerun without --overwrite keeps the crops on disk: run() stops before the crop step."""

    def test_rerun_without_overwrite_writes_the_report(self, temp_dir):
        run = _run_stage(temp_dir, _COHORT, has_existing_crops=True)

        assert sorted(run.rows()) == sorted(_COHORT)

    def test_rerun_runs_no_crop_step(self, temp_dir):
        run = _run_stage(temp_dir, _COHORT, has_existing_crops=True)

        assert run.qc_path.is_file()
        assert run.crop_runs == []

    def test_rerun_with_a_complete_subject_returns_0(self, temp_dir):
        run = _run_stage(temp_dir, _COHORT, crop_subjects=[_SUBJECT], has_existing_crops=True)

        assert run.rows()[_OTHER_SUBJECT]["qc"] == 0
        assert run.result == 0

    def test_rerun_reports_an_input_type_absent_from_the_crops(self, temp_dir):
        row = _run_stage(temp_dir, {_SUBJECT: "LR"}, kinds=("skeleton",), has_existing_crops=True).rows()[_SUBJECT]

        assert f"missing_crop:foldlabel:L:{_REGIONS[0]}" in row["reason"]

    def test_rerun_without_a_complete_subject_returns_1(self, temp_dir):
        assert _run_stage(temp_dir, _COHORT, has_masks=False, has_existing_crops=True).result == 1


class TestLegacyConflict:
    """Legacy and versioned crops without --overwrite: run() stops before the crop step, as on any rerun."""

    def test_conflict_writes_the_report(self, temp_dir):
        run = _run_stage(temp_dir, _COHORT, has_existing_crops=True, legacy_version=_MASKS_DEFAULT)

        assert sorted(run.rows()) == sorted(_COHORT)

    def test_conflict_runs_no_crop_step(self, temp_dir):
        run = _run_stage(temp_dir, _COHORT, has_existing_crops=True, legacy_version=_MASKS_DEFAULT)

        assert run.qc_path.is_file()
        assert run.crop_runs == []

    def test_conflict_with_a_complete_subject_returns_0(self, temp_dir):
        run = _run_stage(
            temp_dir, _COHORT, crop_subjects=[_SUBJECT], has_existing_crops=True, legacy_version=_MASKS_DEFAULT
        )

        assert run.rows()[_OTHER_SUBJECT]["qc"] == 0
        assert run.result == 0

    def test_conflict_without_the_requested_crops_returns_1(self, temp_dir):
        assert _run_stage(temp_dir, _COHORT, legacy_version=_MASKS_OTHER).result == 1

    def test_conflict_without_a_complete_subject_returns_1(self, temp_dir):
        run = _run_stage(temp_dir, _COHORT, has_masks=False, has_existing_crops=True, legacy_version=_MASKS_DEFAULT)

        assert run.result == 1


class TestStepOrder:
    @pytest.mark.parametrize("crop_status", [0, _PARTIAL_STATUS])
    def test_report_follows_fusion_and_ventricle_removal(self, temp_dir, crop_status):
        run = _run_stage(temp_dir, _COHORT, crop_status=crop_status, has_fused_subject=True)

        assert run.qc_path.is_file()
        assert run.report_at_whole_brain == [False, False]


class TestCropStatus3:
    """Crop status 3 is a partial crop: run() goes on with the subjects that were processed."""

    def test_status_3_runs_the_mask_step(self, temp_dir):
        run = _run_stage(temp_dir, _COHORT, crop_status=_PARTIAL_STATUS)

        run.mask_step.assert_called_once()

    def test_status_3_fuses_the_hemispheres(self, temp_dir):
        run = _run_stage(temp_dir, _COHORT, crop_status=_PARTIAL_STATUS)

        run.fuse.assert_called_once()

    def test_status_3_removes_the_ventricle(self, temp_dir):
        run = _run_stage(temp_dir, _COHORT, crop_status=_PARTIAL_STATUS, has_fused_subject=True)

        run.remove.assert_called_once()

    def test_status_3_writes_the_transform_qc(self, temp_dir):
        run = _run_stage(temp_dir, _COHORT, crop_status=_PARTIAL_STATUS)

        assert (run.qc_path.parent / _TRANSFORM_QC_FILE).is_file()

    def test_status_3_writes_the_output_report(self, temp_dir):
        run = _run_stage(temp_dir, _COHORT, crop_subjects=[_SUBJECT], crop_status=_PARTIAL_STATUS)

        assert {subject: row["qc"] for subject, row in run.rows().items()} == {_SUBJECT: 1, _OTHER_SUBJECT: 0}

    def test_status_3_with_a_qc1_row_returns_3(self, temp_dir):
        run = _run_stage(temp_dir, _COHORT, crop_subjects=[_SUBJECT], crop_status=_PARTIAL_STATUS)

        assert run.qc_path.is_file()
        assert run.result == _PARTIAL_STATUS

    def test_status_3_without_a_qc1_row_returns_1(self, temp_dir):
        assert _run_stage(temp_dir, _COHORT, crop_subjects=[], crop_status=_PARTIAL_STATUS).result == 1


@pytest.mark.parametrize("crop_status", _STOPPING_STATUSES)
class TestOtherCropStatus:
    """A crop status other than 0 and 3 stops run() at the crop step."""

    def test_other_status_is_returned(self, temp_dir, crop_status):
        assert _run_stage(temp_dir, _COHORT, crop_status=crop_status).result == crop_status

    def test_other_status_skips_the_mask_step(self, temp_dir, crop_status):
        _run_stage(temp_dir, _COHORT, crop_status=crop_status).mask_step.assert_not_called()

    def test_other_status_skips_the_fusion(self, temp_dir, crop_status):
        _run_stage(temp_dir, _COHORT, crop_status=crop_status).fuse.assert_not_called()

    def test_other_status_skips_ventricle_removal(self, temp_dir, crop_status):
        run = _run_stage(temp_dir, _COHORT, crop_status=crop_status, has_fused_subject=True)

        run.remove.assert_not_called()


class TestBids:
    def test_session_keyed_crops_keep_the_subject(self, temp_dir):
        args = ["--regions", *_REGIONS, "--bids"]

        run = _run_stage(temp_dir, {_SUBJECT: "LR"}, args, crop_subjects=[f"{_SUBJECT}_ses-1_run-2"])

        assert run.rows()[_SUBJECT]["qc"] == 1


class TestExitPolicy:
    def test_one_error_record_per_qc0_row(self, temp_dir, pipeline_caplog):
        run = _run_stage(temp_dir, _COHORT, crop_subjects=[_SUBJECT])

        errors = _messages_at(logging.ERROR, pipeline_caplog)
        assert len(errors) == 1
        assert _OTHER_SUBJECT in errors[0]
        assert run.rows()[_OTHER_SUBJECT]["reason"] in errors[0]

    def test_no_output_qc_error_record_on_a_complete_run(self, temp_dir, pipeline_caplog):
        run = _run_stage(temp_dir, _COHORT)

        assert run.qc_path.is_file()
        assert _messages_at(logging.ERROR, pipeline_caplog) == []

    @pytest.mark.parametrize(
        ("crop_step", "crop_runs"),
        [
            ({"crop_subjects": []}, [_CROP_SCRIPT]),
            ({"has_masks": False, "has_existing_crops": True, "legacy_version": _MASKS_DEFAULT}, []),
        ],
        ids=["crop_run", "legacy_conflict"],
    )
    def test_no_qc1_row_emits_one_fail_record(self, temp_dir, pipeline_caplog, crop_step, crop_runs):
        run = _run_stage(temp_dir, _COHORT, **crop_step)

        assert run.crop_runs == crop_runs
        fails = _messages_at(FAIL, pipeline_caplog)
        assert len(fails) == 1
        assert str(run.qc_path) in fails[0]

    def test_qc1_row_emits_no_fail_record(self, temp_dir, pipeline_caplog):
        run = _run_stage(temp_dir, _COHORT, crop_subjects=[_SUBJECT])

        assert run.qc_path.is_file()
        assert _messages_at(FAIL, pipeline_caplog) == []

    def test_no_qc1_row_returns_1(self, temp_dir):
        assert _run_stage(temp_dir, _COHORT, crop_subjects=[]).result == 1

    def test_incomplete_subject_keeps_the_crop_status(self, temp_dir):
        run = _run_stage(temp_dir, _COHORT, crop_subjects=[_SUBJECT])

        assert run.rows()[_OTHER_SUBJECT]["qc"] == 0
        assert run.result == 0

    def test_complete_run_keeps_the_crop_status(self, temp_dir):
        run = _run_stage(temp_dir, _COHORT)

        assert run.rows()[_SUBJECT]["qc"] == 1
        assert run.result == 0

    def test_no_qc1_row_still_fuses_the_hemispheres(self, temp_dir):
        run = _run_stage(temp_dir, _COHORT, crop_subjects=[])

        assert {row["qc"] for row in run.rows().values()} == {0}
        run.fuse.assert_called_once()


class TestSkipSummary:
    """Crop status 3: one ERROR record per side and step read from the skip lists."""

    _SKIPS = (_Skip(_OTHER_SUBJECT), _Skip(_SUBJECT), _Skip(_OTHER_SUBJECT, "R", _OTHER_STEP))

    def test_one_record_per_side_and_step(self, temp_dir, pipeline_caplog):
        _run_stage(temp_dir, _COHORT_OF_THREE, crop_status=_PARTIAL_STATUS, skips=self._SKIPS)

        assert len(_summary_records(pipeline_caplog)) == 2

    def test_record_names_side_step_and_sorted_subjects(self, temp_dir, pipeline_caplog):
        _run_stage(temp_dir, _COHORT_OF_THREE, crop_status=_PARTIAL_STATUS, skips=self._SKIPS)

        records = _summary_records(pipeline_caplog)
        left = _record_of(records, "L", _STEP)
        right = _record_of(records, "R", _OTHER_STEP)
        assert left.index(_SUBJECT) < left.index(_OTHER_SUBJECT)
        assert _OTHER_SUBJECT in right
        assert _SUBJECT not in right

    def test_record_names_a_subject_once(self, temp_dir, pipeline_caplog):
        _run_stage(temp_dir, _COHORT, crop_status=_PARTIAL_STATUS, skips=(_Skip(), _Skip(message="other region")))

        records = _summary_records(pipeline_caplog)
        assert len(records) == 1
        assert records[0].count(_OTHER_SUBJECT) == 1


class TestSkipReasons:
    """A subject skipped on a side shows the skip-list cause of that side in place of missing_crop."""

    @staticmethod
    def _reasons(run, subject=_OTHER_SUBJECT):
        """Reasons of the row of ``subject``."""
        return run.rows()[subject]["reason"].split(";")

    def test_row_holds_the_skip_reason(self, temp_dir):
        run = _run_stage(temp_dir, _COHORT, skips=(_Skip(),))

        assert run.rows()[_OTHER_SUBJECT]["qc"] == 0
        assert _Skip().reason() in self._reasons(run)

    def test_row_holds_a_reason_per_step(self, temp_dir):
        skips = (_Skip(), _Skip(step=_OTHER_STEP), _Skip(side="R"))

        run = _run_stage(temp_dir, _COHORT, skips=skips)

        assert sorted(self._reasons(run)) == sorted(skip.reason() for skip in skips)

    def test_error_record_shows_the_skip_reason(self, temp_dir, pipeline_caplog):
        _run_stage(temp_dir, _COHORT, skips=(_Skip(),))

        errors = _messages_at(logging.ERROR, pipeline_caplog)
        assert any(message.startswith(_OUTPUT_QC_RECORD_START) and _Skip().reason() in message for message in errors)

    def test_rerun_reads_the_skip_list(self, temp_dir):
        run = _run_stage(temp_dir, _COHORT, skips=(_Skip(),), has_existing_crops=True)

        assert run.crop_runs == []
        assert _Skip().reason() in self._reasons(run)

    def test_skip_reason_replaces_the_missing_crop_reasons_of_the_side(self, temp_dir):
        run = _run_stage(temp_dir, _COHORT, skips=(_Skip(),))

        assert self._reasons(run) == [_Skip().reason()]

    def test_side_without_an_entry_keeps_its_missing_crop_reasons(self, temp_dir):
        run = _run_stage(temp_dir, _COHORT, crop_subjects=[_SUBJECT], skips=(_Skip(),))

        reasons = self._reasons(run)
        assert f"missing_crop:skeleton:R:{_REGIONS[0]}" in reasons
        assert not [reason for reason in reasons if reason.startswith("missing_crop") and ":L:" in reason]

    def test_session_skipped_beside_a_complete_session_keeps_qc_1(self, temp_dir):
        args = ["--regions", *_REGIONS, "--bids"]
        skip = _Skip(f"{_SUBJECT}_ses-1")

        row = _run_stage(temp_dir, {_SUBJECT: "LR"}, args, crop_subjects=[f"{_SUBJECT}_ses-2"], skips=(skip,)).rows()

        assert (row[_SUBJECT]["qc"], row[_SUBJECT]["reason"]) == (1, "")

    def test_bids_session_key_reaches_its_participant(self, temp_dir):
        args = ["--regions", *_REGIONS, "--bids"]
        skip = _Skip(f"{_OTHER_SUBJECT}_ses-1")

        run = _run_stage(temp_dir, _COHORT, args, crop_subjects=[f"{_SUBJECT}_ses-1"], skips=(skip,))

        assert skip.reason() in self._reasons(run)

    def test_non_bids_entry_matches_the_whole_subject(self, temp_dir):
        skip = _Skip(f"{_OTHER_SUBJECT}_ses-1")

        run = _run_stage(temp_dir, _COHORT, crop_subjects=[_SUBJECT], skips=(skip,))

        assert run.rows()[_OTHER_SUBJECT]["qc"] == 0
        assert skip.reason() not in self._reasons(run)
        assert f"missing_crop:skeleton:L:{_REGIONS[0]}" in self._reasons(run)

    def test_message_separators_become_spaces(self, temp_dir):
        run = _run_stage(temp_dir, _COHORT, skips=(_Skip(message="a\nb\rc\td;e"),))

        assert self._reasons(run) == [_Skip(message="a b c d e").reason()]

    def test_repeated_entry_gives_one_reason(self, temp_dir):
        assert self._reasons(_run_stage(temp_dir, _COHORT, skips=(_Skip(), _Skip()))) == [_Skip().reason()]


class TestMalformedSkipList:
    """A skip list its reader refuses costs the skip reasons of that side, not the report."""

    def test_error_record_names_the_file(self, temp_dir, pipeline_caplog):
        _run_stage(temp_dir, _COHORT, crop_status=_PARTIAL_STATUS, malformed_sides="L")

        assert any(SKIP_LIST_FILENAME.format(side="L") in message for message in _summary_records(pipeline_caplog))

    def test_other_side_keeps_its_skip_reason(self, temp_dir):
        skip = _Skip(side="R")

        run = _run_stage(temp_dir, _COHORT, crop_subjects=[_SUBJECT], skips=(skip,), malformed_sides="L")

        reasons = run.rows()[_OTHER_SUBJECT]["reason"].split(";")
        assert skip.reason() in reasons
        assert f"missing_crop:skeleton:L:{_REGIONS[0]}" in reasons


class TestCropChild:
    """Crop status 3 from a real child process, through ScriptBuilder.main(), --log and execute_command unmocked."""

    @staticmethod
    def _run_main(temp_dir, monkeypatch):
        """Run main() on a stand-in crop script exiting 3; return its exit code, output_qc.tsv and run-log paths.

        The stand-in copies a tree holding the crops of _SUBJECT and a skip list naming _OTHER_SUBJECT on side L.
        """
        root = Path(temp_dir)
        input_dir, output_dir, staged = root / "input", root / "output", root / "staged"
        output_dir.mkdir()
        for subject in _COHORT:
            add_graphs(input_dir, subject)
        derivatives = output_dir / DERIVATIVES_FOLDER
        crops_dir = Path(compute_versioned_crops_dir(str(derivatives), _MASKS_DEFAULT))
        staged_crops = staged / crops_dir.relative_to(derivatives)
        add_crops(staged_crops, list(_COHORT), _REGIONS, (*_DEFAULT_KINDS, "mask"))
        _add_skips(staged, staged_crops, _CropStep(skips=(_Skip(),)))
        child = root / "crop_stand_in.py"
        child.write_text(_CHILD_SOURCE)
        log_path = root / "run.log"
        command = [sys.executable, str(child), str(staged), str(derivatives), _CHILD_LINE, str(_PARTIAL_STATUS)]
        argv = [str(input_dir), str(output_dir), "--path_to_graph", GRAPH_DIR, "--path_sk_with_hull", "skeleton/path"]
        monkeypatch.setattr(sys, "argv", ["run_cortical_tiles", *argv, "--regions", *_REGIONS, f"--log={log_path}"])
        monkeypatch.setattr("champollion_utils.script_builder.check_for_updates", lambda *_a, **_k: None)
        monkeypatch.setattr("champollion_utils.script_builder.init_process", lambda *_a, **_k: None)
        script = RunCorticalTiles()
        with (
            patch.object(script, "_build_crop_command", return_value=command),
            patch(f"{_MODULE}.add_left_and_right_volumes.add_left_and_right_volumes"),
            patch(f"{_MODULE}.remove_ventricle.remove_ventricle"),
        ):
            result = script.main()
        return result, derivatives / _QC_FILE, log_path

    def test_child_status_3_runs_the_stage_to_its_end(self, temp_dir, monkeypatch, pipeline_caplog):
        result, qc_path, _ = self._run_main(temp_dir, monkeypatch)

        rows = {row["participant_id"]: row for row in read_qc_tsv(qc_path)}
        assert result == _PARTIAL_STATUS
        assert rows[_SUBJECT]["qc"] == 1
        assert rows[_OTHER_SUBJECT]["reason"] == _Skip().reason()
        assert _OTHER_SUBJECT in _record_of(_summary_records(pipeline_caplog), "L", _STEP)

    def test_child_stdout_reaches_the_run_log(self, temp_dir, monkeypatch):
        _, _, log_path = self._run_main(temp_dir, monkeypatch)

        assert _CHILD_LINE in log_path.read_text()
