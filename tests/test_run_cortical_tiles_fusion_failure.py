#!/usr/bin/env python3
"""
Whole-brain fusion failures of run_cortical_tiles.py (user decision 2026-11-03): a failure of
add_left_and_right_volumes or remove_ventricle logs at ERROR and makes the stage status non-zero.
Synthetic fixtures only, the two cortical_tiles functions faked.

Two kinds of failure: the function raises, or it returns having left a subject without its volume.
remove_ventricle swallows a per-subject error (REQ-CTILESTEST-BDRABCZUK-E3CD2FBA3E2F), so a volume missing
after the call is the only trace of that failure, and no traceback reaches the wrapper.

A failure gives 3 when a subject has a whole_brain/F volume, 1 when none has, and a raise stops neither
ventricle removal nor the later session groups (user decisions 2026-11-05). No qc 1 row stays 1.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline.derivatives_layout import DEFAULT_MASKS_VERSION
from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from champollion_pipeline.utils.cortical_tiles_config import compute_versioned_crops_dir
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER, LABELLING_SESSION_CANDIDATES
from champollion_pipeline.utils.qc_report import read_qc_tsv
from tests.stage2_layout import GRAPH_DIR, add_crops, add_graphs, write_whole_brain_volumes

_MODULE = "champollion_pipeline.run_cortical_tiles"
_LOGGER_NAME = _MODULE
_MASKS = DEFAULT_MASKS_VERSION
_REGIONS = ("S.C.-sylv.", "S.T.s.")
_KINDS = ("skeleton", "foldlabel")
_SUBJECT = "sub-01"
_OTHER_SUBJECT = "sub-02"
_COHORT = (_SUBJECT, _OTHER_SUBJECT)
_LOST_SUBJECTS = ((_SUBJECT,), _COHORT)  # some of the cohort, then all of it
_CROP_STATUSES = (0, 3)  # the crop statuses that reach the whole-brain step
_PARTIAL_STATUS = 3
_FAILED_STATUS = 1


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage2_output_qc():
    """Run the real output QC: overrides the conftest fixture of the same name."""


@dataclass
class _StageRun:
    """What run() gave with the crop subprocess, the mask step and the two whole-brain functions faked."""

    result: int
    qc_path: Path
    remove: MagicMock


def _skeleton_dir(derivatives):
    """Where generate_sulcal_regions.py writes the L and R skeletons, and the fusion writes F."""
    return Path(derivatives).resolve() / "skeletons" / "2mm"


def _add_skeletons(derivatives, subject, sides):
    """Lay the resampled skeleton of ``subject`` on each of ``sides``."""
    for side in sides:
        folder = _skeleton_dir(derivatives) / side
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{side}resampled_skeleton_{subject}.nii.gz").touch()


def _add_labelled_graphs(input_dir, subject, sides, session):
    """Lay the labelled graph of ``subject`` on each of ``sides``, in the labelling session ``session``."""
    session_dir = Path(input_dir) / subject / GRAPH_DIR / session
    session_dir.mkdir()
    for side in sides:
        (session_dir / f"{side}{subject}_{session}.arg").touch()


def _drop_volumes(folder, subjects):
    """Delete the volume of each of ``subjects`` in ``folder``."""
    for subject in subjects:
        for volume in Path(folder).glob(f"*{subject}.nii.gz"):
            volume.unlink()


def _fuse(*, src_dir, **_kwargs):
    """Fake add_left_and_right_volumes: a fused volume per subject holding both sides, none for a one-sided subject."""
    root = Path(src_dir)
    left, right = ({path.name[1:] for path in (root / side).glob("*.nii.gz")} for side in "LR")
    (root / "F").mkdir(exist_ok=True)
    for name in left & right:
        (root / "F" / f"F{name}").touch()


def _fuse_losing(*lost):
    """Fake add_left_and_right_volumes returning with no fused volume for the subjects ``lost``."""

    def fuse(*, src_dir, **_kwargs):
        _fuse(src_dir=src_dir)
        _drop_volumes(Path(src_dir) / "F", lost)

    return fuse


def _remove_losing(*lost):
    """Fake remove_ventricle returning with no whole-brain volume for the subjects ``lost``."""

    def remove(*args, **kwargs):
        write_whole_brain_volumes(*args, **kwargs)
        _drop_volumes(Path(kwargs["output_dir"]) / "F", lost)

    return remove


def _then_raise(fake, error):
    """``fake`` followed by ``error``: the fusion functions raise once part of the cohort is done."""

    def call(*args, **kwargs):
        fake(*args, **kwargs)
        raise error

    return call


def _raise_on_first_call(error):
    """Fake remove_ventricle raising ``error`` on its first call, writing its volumes on the later ones."""
    calls = []

    def remove(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise error
        write_whole_brain_volumes(*args, **kwargs)

    return remove


@dataclass(frozen=True)
class _WholeBrainStep:
    """One of the two faked cortical_tiles functions: its _run_stage keyword, its losing fake, the error it raises."""

    keyword: str
    losing: Callable[..., Callable]
    error: Callable[[], Exception]

    def returning(self, *lost):
        """_run_stage keywords making the step return with no volume for the subjects ``lost``."""
        return {self.keyword: self.losing(*lost)}

    def raising(self, *lost, error=None):
        """_run_stage keywords making the step raise ``error`` once it lost the subjects ``lost``."""
        return {self.keyword: _then_raise(self.losing(*lost), error or self.error())}


_FUSION = _WholeBrainStep("fuse", _fuse_losing, partial(RuntimeError, "fusion failed"))
_VENTRICLE = _WholeBrainStep("remove", _remove_losing, partial(IndexError, "ventricle voxel outside the volume grid"))
_EACH_STEP = pytest.mark.parametrize("step", (_FUSION, _VENTRICLE), ids=("fusion", "ventricle"))


def _run_stage(  # noqa: PLR0913 - keyword-only variations of the one fixture
    temp_dir,
    *,
    fuse=_fuse,
    remove=write_whole_brain_volumes,
    crop_status=0,
    has_crops=True,
    one_sided=(),
    right_graph_missing=(),
    second_session=(),
):
    """Run run() on ``_COHORT``; return the status, the output QC path and the remove_ventricle mock.

    ``fuse`` and ``remove`` stand for the two cortical_tiles functions, ``crop_status`` is what the crop step
    returns, ``has_crops`` False leaves no crop. A subject of ``one_sided`` holds a resampled skeleton on the
    left alone, of ``right_graph_missing`` a labelled graph on the left alone, of ``second_session`` its labelled
    graphs in the second candidate session, so it falls in a later session group.
    """
    root = Path(temp_dir)
    input_dir, output_dir = root / "input", root / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    derivatives = output_dir / DERIVATIVES_FOLDER
    crops_dir = compute_versioned_crops_dir(str(derivatives), _MASKS)
    for subject in _COHORT:
        add_graphs(input_dir, subject, GRAPH_DIR)
        session = LABELLING_SESSION_CANDIDATES[1 if subject in second_session else 0]
        _add_labelled_graphs(input_dir, subject, "L" if subject in right_graph_missing else "LR", session)

    def crop_step(*_args, **_kwargs):
        if has_crops:
            add_crops(crops_dir, _COHORT, _REGIONS, _KINDS)
        for subject in _COHORT:
            _add_skeletons(derivatives, subject, "L" if subject in one_sided else "LR")
        return crop_status

    def mask_step(_output_abs):
        if has_crops:
            add_crops(crops_dir, _COHORT, _REGIONS, ("mask",))

    script = RunCorticalTiles()
    script.parse_args(
        [
            str(input_dir),
            str(output_dir),
            "--path_to_graph",
            GRAPH_DIR,
            "--path_sk_with_hull",
            "skeleton/path",
            "--regions",
            *_REGIONS,
        ]
    )
    with (
        patch.object(script, "validate_paths", return_value=True),
        patch.object(script, "execute_command", side_effect=crop_step),
        patch.object(script, "_generate_mask_npys", side_effect=mask_step),
        patch(f"{_MODULE}.add_left_and_right_volumes.add_left_and_right_volumes", side_effect=fuse),
        patch(f"{_MODULE}.remove_ventricle.remove_ventricle", side_effect=remove) as removal,
    ):
        result = script.run()
    return _StageRun(result, derivatives / "output_qc.tsv", removal)


def _error_records(pipeline_caplog):
    """ERROR records of the run_cortical_tiles logger."""
    return [r for r in pipeline_caplog.records if r.name == _LOGGER_NAME and r.levelno == logging.ERROR]


def _assert_one_traceback_record(pipeline_caplog, error):
    """Exactly one ERROR record carries ``error`` with its traceback."""
    carrying = [r for r in _error_records(pipeline_caplog) if r.exc_info and r.exc_info[1] is error]
    assert len(carrying) == 1, [r.getMessage() for r in _error_records(pipeline_caplog)]
    assert carrying[0].exc_info[2] is not None, "exc_info holds no traceback object"


def _assert_subjects_named(pipeline_caplog, named):
    """The ERROR records name each subject of ``named`` and no other subject of the cohort."""
    messages = [r.getMessage() for r in _error_records(pipeline_caplog)]
    for subject in _COHORT:
        is_named = any(re.search(rf"\b{subject}\b", message) for message in messages)
        assert is_named == (subject in named), (subject, messages)


@_EACH_STEP
class TestWholeBrainStepRaises:
    @pytest.mark.parametrize("lost", _LOST_SUBJECTS)
    def test_one_error_record_carries_the_traceback(self, temp_dir, pipeline_caplog, step, lost):
        error = step.error()
        _run_stage(temp_dir, **step.raising(*lost, error=error))
        _assert_one_traceback_record(pipeline_caplog, error)

    @pytest.mark.parametrize("crop_status", _CROP_STATUSES)
    def test_status_is_3_with_a_whole_brain_volume(self, temp_dir, step, crop_status):
        assert _run_stage(temp_dir, crop_status=crop_status, **step.raising(_SUBJECT)).result == _PARTIAL_STATUS

    @pytest.mark.parametrize("crop_status", _CROP_STATUSES)
    def test_status_is_1_without_a_whole_brain_volume(self, temp_dir, step, crop_status):
        assert _run_stage(temp_dir, crop_status=crop_status, **step.raising(*_COHORT)).result == _FAILED_STATUS

    def test_output_qc_is_written(self, temp_dir, step):
        run = _run_stage(temp_dir, **step.raising(_SUBJECT))
        assert sorted(row["participant_id"] for row in read_qc_tsv(run.qc_path)) == sorted(_COHORT)


class TestFusionRaises:
    def test_ventricle_removal_still_runs(self, temp_dir):
        run = _run_stage(temp_dir, **_FUSION.raising(_SUBJECT))
        run.remove.assert_called_once()


class TestVentricleRemovalRaises:
    def test_later_session_group_still_runs(self, temp_dir):
        run = _run_stage(temp_dir, remove=_raise_on_first_call(_VENTRICLE.error()), second_session=(_OTHER_SUBJECT,))
        assert run.remove.call_count == len(_COHORT)


@_EACH_STEP
class TestWholeBrainStepLosesAVolume:
    @pytest.mark.parametrize("lost", _LOST_SUBJECTS)
    def test_error_record_names_the_subject(self, temp_dir, pipeline_caplog, step, lost):
        _run_stage(temp_dir, **step.returning(*lost))
        _assert_subjects_named(pipeline_caplog, lost)

    def test_status_is_3_with_a_whole_brain_volume(self, temp_dir, step):
        assert _run_stage(temp_dir, **step.returning(_SUBJECT)).result == _PARTIAL_STATUS

    @pytest.mark.parametrize("crop_status", _CROP_STATUSES)
    def test_status_is_1_without_a_whole_brain_volume(self, temp_dir, step, crop_status):
        assert _run_stage(temp_dir, crop_status=crop_status, **step.returning(*_COHORT)).result == _FAILED_STATUS


class TestOutputQcWins:
    def test_no_qc_1_row_gives_1_with_a_whole_brain_volume(self, temp_dir):
        assert _run_stage(temp_dir, has_crops=False, **_FUSION.raising(_SUBJECT)).result == _FAILED_STATUS


class TestNoFusionFailure:
    """Pass before green: these keep the failure check from catching subjects it must leave alone."""

    def test_one_sided_subject_keeps_the_crop_status(self, temp_dir):
        assert _run_stage(temp_dir, one_sided=(_SUBJECT,)).result == 0

    def test_labelling_qc_0_subject_keeps_the_crop_status(self, temp_dir):
        run = _run_stage(temp_dir, right_graph_missing=(_SUBJECT,), remove=_remove_losing(_SUBJECT))
        assert run.result == 0
