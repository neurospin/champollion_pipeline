#!/usr/bin/env python3
"""
Crop completion marker of run_cortical_tiles.py (user decision 2026-11-03): written after a success or a
partial (status 3) run, absent after a crash. Synthetic fixtures only: the crop step and the mask step are
mocked and lay their files, the output QC is the real one.

A test of a run that must leave no marker starts from an earlier marker and passes --overwrite, so that the
marker of the previous run has to go. Crops holding no marker count as not complete, whatever made them, and are
regenerated with a WARNING (user decisions at red).

A whole-brain raise stage 2 catches ends no run: the crops stay complete, so a run returning 3, or 1 with a qc 1 row
in the output QC, gets the marker (user answers at the third red pass). The cases of _CAUGHT_STEPS need the catch of
TASK-CTFAIL-BDRABCZUK-2ADCB1007301: until it merges the raise leaves run() and result stays None.
"""

import json
import logging
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from champollion_pipeline.utils.cortical_tiles_config import compute_versioned_crops_dir
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER, LABELLING_SESSION_CANDIDATES
from tests.stage2_layout import CROP_MARKER, GRAPH_DIR, TEMPLATE_CONFIG, add_crop_marker, add_crops, add_graphs

_MODULE = "champollion_pipeline.run_cortical_tiles"
_CROP_SCRIPT = "generate_sulcal_regions.py"
_PARTIAL_STATUS = 3
_STOPPING_STATUSES = (1, 2, 4)
_MASKS_DEFAULT = "canonical_25"
_MASKS_OTHER = "other_masks"
_REGIONS = ("S.C.-sylv.", "S.T.s.")
_KINDS = ("skeleton", "foldlabel")
_SUBJECT = "sub-01"
_NO_RAISE = "no raise"
_STEP_MASKS = "mask step"
_STEP_TRANSFORM_QC = "transform QC"
_STEP_OUTPUT_QC = "output QC"
_STEP_FUSION = "fusion"
_STEP_VENTRICLE = "ventricle removal"
_UNCAUGHT_STEPS = (_STEP_MASKS, _STEP_TRANSFORM_QC, _STEP_OUTPUT_QC)
_CAUGHT_STEPS = (_STEP_FUSION, _STEP_VENTRICLE)
_FUSED_VOLUME = f"Fresampled_skeleton_{_SUBJECT}.nii.gz"
_WHOLE_BRAIN_VOLUME = f"Fskeleton_generated_without_ventricle_{_SUBJECT}.nii.gz"


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage2_output_qc():
    """Run the real output QC: overrides the conftest fixture of the same name."""


class _StepFailure(Exception):
    """Raised by the mocked step of a test on a crash after the crop step."""


@dataclass
class _CropStep:
    """How the mocked stage 2 behaves in ``_run_stage``."""

    status: int = 0
    masks_version: str = _MASKS_DEFAULT
    has_masks: bool = True
    has_earlier_marker: bool = False
    has_overwrite: bool = False
    has_whole_brain_volume: bool = False
    raising_step: str = _NO_RAISE


@dataclass
class _StageRun:
    """What ``_run_stage`` observed: the return value, the crop runs, each marker state at a crop start."""

    result: int | None
    crop_runs: int
    markers_at_crop_start: list[bool]
    marker: Path


def _crops_dir(root, masks_version):
    derivatives = Path(root) / "output" / DERIVATIVES_FOLDER
    return Path(compute_versioned_crops_dir(str(derivatives), masks_version))


def _add_earlier_crops(root, masks_version=_MASKS_DEFAULT):
    """Lay crops holding no marker, as a crashed run or a run older than the marker leaves them."""
    add_crops(_crops_dir(root, masks_version), [_SUBJECT], _REGIONS, _KINDS)


def _add_legacy_crops(root):
    """Lay crops under the legacy crops/2mm/, with a pipeline_loop_2mm.json naming the default masks version."""
    output_dir = Path(root) / "output"
    add_crops(output_dir / DERIVATIVES_FOLDER / "crops" / "2mm", [_SUBJECT], _REGIONS, _KINDS)
    (output_dir / TEMPLATE_CONFIG.name).write_text(json.dumps({"masks_version": _MASKS_DEFAULT}))


def _add_fused_subject(root):
    """Lay what takes run() to remove_ventricle: a fused volume, and a labelled graph per hemisphere."""
    session = LABELLING_SESSION_CANDIDATES[0]
    session_dir = Path(root) / "input" / _SUBJECT / GRAPH_DIR / session
    session_dir.mkdir()
    for side in "LR":
        (session_dir / f"{side}{_SUBJECT}_{session}.arg").touch()
    fused_dir = Path(root) / "output" / DERIVATIVES_FOLDER / "skeletons" / "2mm" / "F"
    fused_dir.mkdir(parents=True)
    (fused_dir / _FUSED_VOLUME).touch()


def _add_whole_brain_volume(root):
    """Lay the whole_brain/F volume a subject keeps from the part of the run done before the raise."""
    volume_dir = Path(root) / "output" / DERIVATIVES_FOLDER / "whole_brain" / "F"
    volume_dir.mkdir(parents=True)
    (volume_dir / _WHOLE_BRAIN_VOLUME).touch()


def _raise_at(step, name):
    return _StepFailure(name) if step.raising_step == name else None


def _run_stage(root, **crop_step):
    """Run run() on one subject; ``crop_step``: fields of _CropStep. A step made to raise propagates _StepFailure."""
    step = _CropStep(**crop_step)
    input_dir, output_dir = Path(root) / "input", Path(root) / "output"
    add_graphs(input_dir, _SUBJECT, GRAPH_DIR, "LR")
    output_dir.mkdir(exist_ok=True)
    if step.raising_step == _STEP_VENTRICLE:
        _add_fused_subject(root)
    if step.has_whole_brain_volume:
        _add_whole_brain_volume(root)
    crops_dir = _crops_dir(root, step.masks_version)
    marker = crops_dir / CROP_MARKER
    if step.has_earlier_marker:
        add_crops(crops_dir, [_SUBJECT], _REGIONS, _KINDS)
        add_crop_marker(crops_dir)
    run = _StageRun(None, 0, [], marker)

    def execute(command, *_args, **_kwargs):
        if any(_CROP_SCRIPT in str(part) for part in command):
            run.crop_runs += 1
            run.markers_at_crop_start.append(marker.exists())
            add_crops(crops_dir, [_SUBJECT], _REGIONS, _KINDS)
        return step.status

    def write_masks(_output_abs):
        if step.has_masks:
            add_crops(crops_dir, [_SUBJECT], _REGIONS, ("mask",))

    script = RunCorticalTiles()
    script.parse_args(
        [
            str(input_dir),
            str(output_dir),
            "--path_to_graph",
            GRAPH_DIR,
            "--path_sk_with_hull",
            "skeleton/path",
            "--masks",
            step.masks_version,
            "--regions",
            *_REGIONS,
            *(["--overwrite"] if step.has_overwrite else []),
        ]
    )
    with (
        patch.object(script, "validate_paths", return_value=True),
        patch.object(script, "execute_command", side_effect=execute),
        patch.object(
            script,
            "_generate_mask_npys",
            side_effect=_raise_at(step, _STEP_MASKS) or write_masks,
        ),
        patch(
            f"{_MODULE}.add_left_and_right_volumes.add_left_and_right_volumes",
            side_effect=_raise_at(step, _STEP_FUSION),
        ),
        patch(
            f"{_MODULE}.remove_ventricle.remove_ventricle",
            side_effect=_raise_at(step, _STEP_VENTRICLE),
        ),
        patch.object(
            script,
            "_remove_ventricle_whole_brain",
            wraps=script._remove_ventricle_whole_brain if step.raising_step == _STEP_VENTRICLE else None,
        ),
        patch.object(
            script,
            "_write_transform_qc",
            wraps=script._write_transform_qc,
            side_effect=_raise_at(step, _STEP_TRANSFORM_QC),
        ),
        patch.object(
            script,
            "_write_output_qc",
            wraps=script._write_output_qc,
            side_effect=_raise_at(step, _STEP_OUTPUT_QC),
        ),
        suppress(_StepFailure),
    ):
        run.result = script.run()
    return run


@pytest.mark.parametrize("masks_version", (_MASKS_DEFAULT, _MASKS_OTHER))
def test_status_0_run_writes_the_marker(temp_dir, masks_version):
    run = _run_stage(temp_dir, masks_version=masks_version)

    assert run.result == 0
    assert run.marker.is_file()


@pytest.mark.parametrize("raising_step", (_NO_RAISE, *_CAUGHT_STEPS))
def test_status_3_run_writes_the_marker(temp_dir, raising_step):
    run = _run_stage(temp_dir, status=_PARTIAL_STATUS, raising_step=raising_step, has_whole_brain_volume=True)

    assert run.result == _PARTIAL_STATUS
    assert run.marker.is_file()


@pytest.mark.parametrize("raising_step", _CAUGHT_STEPS)
def test_status_0_run_returning_3_writes_the_marker(temp_dir, raising_step):
    run = _run_stage(temp_dir, raising_step=raising_step, has_whole_brain_volume=True)

    assert run.result == _PARTIAL_STATUS
    assert run.marker.is_file()


@pytest.mark.parametrize("status", (0, _PARTIAL_STATUS))
@pytest.mark.parametrize("raising_step", _CAUGHT_STEPS)
def test_run_returning_1_with_a_passing_subject_writes_the_marker(temp_dir, status, raising_step):
    run = _run_stage(temp_dir, status=status, raising_step=raising_step)

    assert run.result == 1
    assert run.marker.is_file()


def test_marker_is_gone_when_the_crop_step_starts(temp_dir):
    run = _run_stage(temp_dir, has_earlier_marker=True, has_overwrite=True)

    assert run.markers_at_crop_start == [False]


@pytest.mark.parametrize("status", _STOPPING_STATUSES)
def test_stopping_crop_status_leaves_no_marker(temp_dir, status):
    run = _run_stage(temp_dir, status=status, has_earlier_marker=True, has_overwrite=True)

    assert run.result == status
    assert not run.marker.exists()


@pytest.mark.parametrize("status", (0, _PARTIAL_STATUS))
@pytest.mark.parametrize("raising_step", (_NO_RAISE, *_CAUGHT_STEPS))
def test_run_without_a_passing_subject_leaves_no_marker(temp_dir, status, raising_step):
    run = _run_stage(
        temp_dir,
        status=status,
        raising_step=raising_step,
        has_masks=False,
        has_earlier_marker=True,
        has_overwrite=True,
    )

    assert run.result == 1
    assert not run.marker.exists()


@pytest.mark.parametrize("raising_step", _UNCAUGHT_STEPS)
def test_step_raising_out_of_run_leaves_no_marker(temp_dir, raising_step):
    run = _run_stage(temp_dir, raising_step=raising_step, has_earlier_marker=True, has_overwrite=True)

    assert run.crop_runs == 1
    assert run.result is None
    assert not run.marker.exists()


def test_marked_crops_run_no_crop_step(temp_dir):
    run = _run_stage(temp_dir, has_earlier_marker=True)

    assert run.crop_runs == 0
    assert run.marker.is_file()


def test_marked_rerun_returning_1_keeps_the_marker(temp_dir):
    run = _run_stage(temp_dir, has_earlier_marker=True, has_masks=False)

    assert run.crop_runs == 0
    assert run.result == 1
    assert run.marker.is_file()


def test_unmarked_crops_start_the_crop_step(temp_dir):
    _add_earlier_crops(temp_dir)

    assert _run_stage(temp_dir).crop_runs == 1


def test_unmarked_crops_get_the_marker_from_a_status_0_run(temp_dir):
    _add_earlier_crops(temp_dir)

    assert _run_stage(temp_dir).marker.is_file()


def test_moved_legacy_crops_start_the_crop_step(temp_dir):
    _add_legacy_crops(temp_dir)

    assert _run_stage(temp_dir).crop_runs == 1


def test_marker_of_another_masks_version_does_not_count(temp_dir):
    add_crop_marker(_crops_dir(temp_dir, _MASKS_DEFAULT))
    _add_earlier_crops(temp_dir, _MASKS_OTHER)

    assert _run_stage(temp_dir, masks_version=_MASKS_OTHER).crop_runs == 1


def test_unmarked_crops_log_a_warning_naming_the_marker(temp_dir, pipeline_caplog):
    _add_earlier_crops(temp_dir)

    _run_stage(temp_dir)

    assert pipeline_caplog.assert_logged(logging.WARNING, CROP_MARKER).name == _MODULE
