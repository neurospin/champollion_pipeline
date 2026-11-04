#!/usr/bin/env python3
"""
Input QC tests of train_champollion.py: input_qc.tsv, the augmentation setting it depends on and the
exit policy (user decisions 2026-11-04: fail only if nothing usable; a partial cohort launches train.py
unchanged and keeps its status, unlike stages 3 and 4, which return 3).

Synthetic fixtures only: the region config comes from the shipped reference.yaml, a fake configs tree
stands in for the champollion_V1 checkout and train.py is mocked.
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from champollion_pipeline import train_champollion
from champollion_pipeline.utils.qc_report import read_qc_tsv
from tests.log_assertions import FAIL
from tests.stage2_layout import add_crops, crop_array, crop_csv
from tests.test_train_champollion import _build_model_dir, make_script

_LOGGER_NAME = "champollion_pipeline.train_champollion"
_REFERENCE_CONFIG = Path(__file__).resolve().parent.parent / "reference.yaml"
_QC_FILE = "input_qc.tsv"
_DATASET = "TEST01"
_CROP = "S.C.-sylv."
_REGION_BY_SIDE = {"L": "SC-sylv_left", "R": "SC-sylv_right"}
_SIDE = "R"
_OTHER_SIDE = "L"
_KINDS = ("skeleton", "foldlabel", "mask")
_CHECKED_KINDS = ("foldlabel", "mask")
_SUBJECT = "sub-01"
_OTHER_SUBJECT = "sub-02"
_COHORT = (_SUBJECT, _OTHER_SUBJECT)
_AUGMENTED_GROUP = "mixed"
_PLAIN_GROUP = "no_augmentation"


def _gap(kind, side=_SIDE):
    """Reason of a crop gap."""
    return f"missing_crop:{kind}:{side}:{_CROP}"


def _set_group(configs, group, is_augmented):
    """Write augmentations/<group>.yaml in the ``configs`` tree."""
    path = configs / "augmentations" / f"{group}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# @package _global_\napply_augmentations: {is_augmented}\nfill_value: 0\n")


def _set_default_group(configs, group):
    """Write a config.yaml in the ``configs`` tree whose defaults list names ``group``."""
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "config.yaml").write_text(f"defaults:\n  - _self_\n  - augmentations: {group}\n  - mode: encoder\n")


def _write_region_config(config_dir, side):
    """Write the dataset YAML generate_champollion_config.py would leave for the crop on ``side``."""
    region = _REGION_BY_SIDE[side]
    text = (
        _REFERENCE_CONFIG.read_text()
        .replace("TESTXX/crops/2mm", f"{_DATASET}/crops/2mm")
        .replace("REPLACE_CROP_NAME", _CROP)
        .replace("REPLACE_DATASET", region)
        .replace("REPLACE_SIDE", side)
    )
    path = config_dir / "dataset" / _DATASET / f"{region}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@dataclass(frozen=True)
class _Layout:
    """Folders of one synthetic training run."""

    checkout_configs: Path
    config_dir: Path
    crops_dir: Path
    output_dir: Path

    @property
    def report(self):
        """Path of input_qc.tsv."""
        return self.output_dir / _QC_FILE

    def rows(self):
        """Rows of input_qc.tsv, keyed by participant_id."""
        assert self.report.is_file(), f"{self.report} was not written"
        return {row["participant_id"]: row for row in read_qc_tsv(self.report)}

    def lose_label_array(self, side=_SIDE):
        """Remove the label array of ``side``: each subject has a gap there."""
        crop_array(self.crops_dir, _CROP, side, "foldlabel").unlink()

    def lose_mask_array(self, side=_SIDE):
        """Remove the mask array of ``side``: each subject has a gap there."""
        crop_array(self.crops_dir, _CROP, side, "mask").unlink()

    def empty_cohort(self):
        """Leave the subjects_all file with its header alone."""
        crop_csv(self.crops_dir, _CROP, _SIDE, "skeleton").write_text("Subject\n")

    def leave_out_of_label_csv(self):
        """Leave ``_OTHER_SUBJECT`` out of the label subject CSV of the trained side: a partial cohort."""
        add_crops(self.crops_dir, [_SUBJECT], [_CROP], ("foldlabel",), sides=_SIDE)


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """A complete cohort, a region config and a fake checkout whose config.yaml turns augmentation on."""
    checkout = tmp_path / "champollion"
    monkeypatch.setattr(train_champollion, "_CONTRASTIVE_DIR", str(checkout))
    checkout_configs = checkout / "configs"
    _set_default_group(checkout_configs, _AUGMENTED_GROUP)
    _set_group(checkout_configs, _AUGMENTED_GROUP, True)
    _set_group(checkout_configs, _PLAIN_GROUP, False)

    data_dir = tmp_path / "data"
    config_dir = tmp_path / "configs"
    for side in _REGION_BY_SIDE:
        _write_region_config(config_dir, side)
    localization = config_dir / "dataset_localization" / "local.yaml"
    localization.parent.mkdir(parents=True)
    localization.write_text(f"# @package _global_\ndataset_folder: {data_dir}\n")

    crops_dir = data_dir / _DATASET / "crops" / "2mm"
    add_crops(crops_dir, _COHORT, [_CROP], _KINDS)
    return _Layout(checkout_configs, config_dir, crops_dir, tmp_path / "model")


@dataclass(frozen=True)
class _Run:
    """What TrainChampollion.run() did with train.py mocked."""

    result: int
    commands: list[list[str]]
    has_report_at_launch: list[bool]


def _train(layout, *extra_args, side=_SIDE, status=0, leaves_model=True):
    """Run TrainChampollion on ``layout``; the mocked train.py returns ``status``."""
    script = make_script(
        [
            "--dataset",
            _DATASET,
            "--region",
            _REGION_BY_SIDE[side],
            "--config-dir",
            str(layout.config_dir),
            "--output_dir",
            str(layout.output_dir),
            *extra_args,
        ]
    )
    commands = []
    has_report_at_launch = []

    def fake_execute(cmd, shell=False):
        commands.append(cmd)
        has_report_at_launch.append(layout.report.is_file())
        if leaves_model:
            _build_model_dir(layout.output_dir)
        return status

    script.execute_command = fake_execute
    return _Run(script.run(), commands, has_report_at_launch)


def _messages_at(level, pipeline_caplog):
    """Messages of the train_champollion logger at ``level``."""
    return [r.getMessage() for r in pipeline_caplog.records if r.name == _LOGGER_NAME and r.levelno == level]


def _assert_partial(layout):
    """input_qc.tsv keeps ``_SUBJECT`` and excludes ``_OTHER_SUBJECT``."""
    rows = layout.rows()
    assert (rows[_SUBJECT]["qc"], rows[_OTHER_SUBJECT]["qc"]) == (1, 0)


_LOST_COHORTS = pytest.mark.parametrize(
    "lose", [_Layout.lose_label_array, _Layout.lose_mask_array, _Layout.empty_cohort]
)


class TestReport:
    def test_one_row_per_subject_of_subjects_all(self, layout):
        _train(layout)

        rows = layout.rows()

        assert sorted(rows) == sorted(_COHORT)
        assert {(row["qc"], row["reason"]) for row in rows.values()} == {(1, "")}

    def test_report_holds_the_shared_columns(self, layout):
        _train(layout)

        assert layout.report.read_text().splitlines()[0].split("\t") == ["participant_id", "qc", "reason"]

    def test_report_is_written_before_train_py_launches(self, layout):
        assert _train(layout).has_report_at_launch == [True]


class TestCropGaps:
    def test_subject_left_out_of_the_label_csv(self, layout):
        layout.leave_out_of_label_csv()

        _train(layout)

        rows = layout.rows()
        assert (rows[_OTHER_SUBJECT]["qc"], rows[_OTHER_SUBJECT]["reason"]) == (0, _gap("foldlabel"))
        assert (rows[_SUBJECT]["qc"], rows[_SUBJECT]["reason"]) == (1, "")

    def test_absent_label_array_fails_each_subject(self, layout):
        layout.lose_label_array()

        _train(layout)

        assert {row["reason"] for row in layout.rows().values()} == {_gap("foldlabel")}

    def test_absent_mask_array_fails_each_subject(self, layout):
        layout.lose_mask_array()

        _train(layout)

        assert {row["reason"] for row in layout.rows().values()} == {_gap("mask")}

    def test_skeleton_array_is_not_a_checked_kind(self, layout):
        crop_array(layout.crops_dir, _CROP, _SIDE, "skeleton").unlink()

        _train(layout)

        assert {row["qc"] for row in layout.rows().values()} == {1}


class TestTrainedSide:
    @pytest.mark.parametrize("kind", _CHECKED_KINDS)
    def test_gap_on_the_other_side_keeps_the_subject(self, layout, kind):
        crop_array(layout.crops_dir, _CROP, _OTHER_SIDE, kind).unlink()

        _train(layout)

        assert {(row["qc"], row["reason"]) for row in layout.rows().values()} == {(1, "")}

    def test_side_follows_the_region_config(self, layout):
        layout.lose_label_array(_OTHER_SIDE)

        _train(layout, side=_OTHER_SIDE)

        assert {row["reason"] for row in layout.rows().values()} == {_gap("foldlabel", _OTHER_SIDE)}


class TestAugmentationSetting:
    def test_group_of_the_checkout_turns_the_check_on(self, layout):
        _train(layout)

        assert layout.report.is_file()

    def test_config_dir_group_file_turns_the_check_on(self, layout):
        _set_group(layout.checkout_configs, _AUGMENTED_GROUP, False)
        _set_group(layout.config_dir, _AUGMENTED_GROUP, True)

        _train(layout)

        assert layout.report.is_file()

    def test_config_dir_group_file_turns_the_check_off(self, layout):
        _set_group(layout.config_dir, _AUGMENTED_GROUP, False)

        _train(layout)

        assert not layout.report.exists()

    def test_config_dir_config_yaml_names_the_augmented_group(self, layout):
        _set_default_group(layout.checkout_configs, _PLAIN_GROUP)
        _set_default_group(layout.config_dir, _AUGMENTED_GROUP)

        _train(layout)

        assert layout.report.is_file()

    def test_config_dir_config_yaml_names_the_plain_group(self, layout):
        _set_default_group(layout.config_dir, _PLAIN_GROUP)

        _train(layout)

        assert not layout.report.exists()


class TestNoAugmentation:
    def test_train_py_launches_without_a_report(self, layout):
        _set_default_group(layout.checkout_configs, _PLAIN_GROUP)
        layout.lose_label_array()

        run = _train(layout)

        assert len(run.commands) == 1
        assert not layout.report.exists()


class TestErrorRecords:
    def test_one_error_record_per_qc0_row(self, layout, pipeline_caplog):
        layout.leave_out_of_label_csv()

        _train(layout)

        errors = _messages_at(logging.ERROR, pipeline_caplog)
        assert len(errors) == 1
        assert _OTHER_SUBJECT in errors[0]
        assert _gap("foldlabel") in errors[0]

    def test_no_error_record_on_a_complete_cohort(self, layout, pipeline_caplog):
        _train(layout)

        assert _messages_at(logging.ERROR, pipeline_caplog) == []


class TestNoValidSubject:
    @_LOST_COHORTS
    def test_returns_1(self, layout, lose):
        lose(layout)

        assert _train(layout).result == 1

    @_LOST_COHORTS
    def test_runs_no_train_command(self, layout, lose):
        lose(layout)

        assert _train(layout).commands == []


class TestFailRecord:
    @_LOST_COHORTS
    def test_no_qc1_row_emits_one_fail_record(self, layout, pipeline_caplog, lose):
        lose(layout)

        _train(layout)

        fails = _messages_at(FAIL, pipeline_caplog)
        assert len(fails) == 1
        assert str(layout.report) in fails[0]

    def test_qc1_row_emits_no_fail_record(self, layout, pipeline_caplog):
        layout.leave_out_of_label_csv()

        _train(layout)

        assert _messages_at(FAIL, pipeline_caplog) == []


class TestUnbuiltReport:
    @pytest.fixture
    def blocked_report(self, layout):
        """A directory at the input_qc.tsv path: writing the report raises IsADirectoryError."""
        layout.report.mkdir(parents=True)
        return layout

    def test_training_launches(self, blocked_report):
        assert len(_train(blocked_report, "--overwrite").commands) == 1

    def test_warning_names_the_exception(self, blocked_report, pipeline_caplog):
        _train(blocked_report, "--overwrite")

        assert any("IsADirectoryError" in message for message in _messages_at(logging.WARNING, pipeline_caplog))


class TestRetryAfterRefusal:
    def test_retry_without_overwrite_trains(self, layout):
        layout.lose_label_array()
        refused = _train(layout)
        add_crops(layout.crops_dir, _COHORT, [_CROP], _KINDS)

        retry = _train(layout)

        assert (refused.result, len(refused.commands)) == (1, 0)
        assert (retry.result, len(retry.commands)) == (0, 1)

    def test_retry_on_lost_crops_is_refused_again(self, layout):
        layout.lose_label_array()
        _train(layout)

        retry = _train(layout)

        assert (retry.result, retry.commands) == (1, [])


class TestPartialCohort:
    def test_launches_on_the_unchanged_subject_set(self, layout):
        complete = _train(layout)
        shutil.rmtree(layout.output_dir)
        layout.leave_out_of_label_csv()

        partial = _train(layout)

        _assert_partial(layout)
        assert partial.commands == complete.commands

    @pytest.mark.parametrize(
        ("status", "leaves_model"),
        [(0, True), (2, True), (0, False)],
        ids=["train_py_succeeds", "train_py_fails", "output_qc_fails"],
    )
    def test_status_matches_a_complete_cohort(self, layout, status, leaves_model):
        complete = _train(layout, status=status, leaves_model=leaves_model)
        shutil.rmtree(layout.output_dir)
        layout.leave_out_of_label_csv()

        partial = _train(layout, status=status, leaves_model=leaves_model)

        _assert_partial(layout)
        assert partial.result == complete.result
