#!/usr/bin/env python3
"""
Stage 3 input QC tests of generate_champollion_config.py: input_qc.tsv and the exit policy (user decision
2026-11-04, "Fail only if nothing usable"; a partial exclusion returns 3). Synthetic fixtures only.

Each test passes an explicit --champollion_loc under the test folder, so nothing reaches the real checkout.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.generate_champollion_config import GenerateChampollionConfig
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from champollion_pipeline.utils.qc_report import fetch_crop_qc_rows, read_qc_tsv
from tests.log_assertions import FAIL, assert_one_subject_error
from tests.stage2_layout import add_crops, crop_array, crop_csv
from tests.test_generate_champollion_config_internals import fake_execute_command, make_crop

_MODULE = "champollion_pipeline.generate_champollion_config"
_QC_FILE = "input_qc.tsv"
_DATASET = "TEST01"
_REGIONS = ("S.C.-sylv.", "S.T.s.")
_KINDS = ("skeleton", "foldlabel", "mask")
_SUBJECT = "sub-01"
_OTHER_SUBJECT = "sub-02"
_COHORT = (_SUBJECT, _OTHER_SUBJECT)
_PARTIAL_STATUS = 3


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage3_input_qc():
    """Run the real input QC: overrides the conftest fixture of the same name."""


@pytest.fixture
def crops(temp_dir):
    """Crops folder of the dataset, holding the complete crops of the cohort in ``_REGIONS``."""
    return _add_cohort(_crops_dir(temp_dir))


@pytest.fixture
def partial_crops(crops):
    """Crops folder where ``_OTHER_SUBJECT`` is left out of the skeleton CSV of the left side of ``_REGIONS[0]``."""
    add_crops(crops, [_SUBJECT], _REGIONS[:1], sides="L")
    make_crop(crops, _REGIONS[0], side="L")
    return crops


@pytest.fixture
def lost_crops(crops):
    """Crops folder where no subject passes: the left skeleton array of ``_REGIONS[0]`` is gone."""
    crop_array(crops, _REGIONS[0], "L", "skeleton").unlink()
    return crops


@dataclass
class _StageRun:
    """What run() did: its result, the default configs root, and whether the report preceded the first config."""

    result: int
    configs_root: Path
    champollion_loc: Path
    report_at_first_config: list[bool]

    @property
    def qc_path(self):
        """Path of input_qc.tsv in the configs root."""
        return self.configs_root / _QC_FILE

    def rows(self):
        """Rows of input_qc.tsv, keyed by participant_id."""
        assert self.qc_path.is_file(), f"{self.qc_path} was not written"
        return {row["participant_id"]: row for row in read_qc_tsv(self.qc_path)}

    def yaml_files(self):
        """YAML files under the configs root."""
        return sorted(self.configs_root.rglob("*.yaml"))


def _dataset_parent(crops_dir):
    """<D>: the parent of the <dataset> directory in ``crops_dir``."""
    return next(parent.parent for parent in Path(crops_dir).parents if parent.name == _DATASET)


def _crops_dir(root):
    """Crops folder of the dataset under ``root``."""
    return Path(root) / "data" / _DATASET / "derivatives" / DERIVATIVES_FOLDER / "crops" / "canonical_25" / "2mm"


def _add_cohort(crops_dir, subjects=_COHORT, regions=_REGIONS):
    """Lay complete crops of ``subjects`` in ``regions``, with skeleton arrays stage 3 reads a size from."""
    add_crops(crops_dir, subjects, regions, _KINDS)
    for region in regions:
        for side in "LR":
            make_crop(crops_dir, region, side=side)
    return Path(crops_dir)


def _run_stage(crops_dir, *extra_args, output=None):
    """Run run() on ``crops_dir``; ``output``, when given, is --output and the configs root of the returned run."""
    crops_dir = Path(crops_dir)
    dataset_parent = _dataset_parent(crops_dir)
    default_root = dataset_parent / _DATASET / "derivatives" / "champollion_V1" / "configs"
    configs_root = Path(output) if output else default_root
    champollion_loc = dataset_parent / "champollion_V1_checkout"
    report_at_first_config = []

    def execute(command, shell=False):
        if command[0] == "cp":
            report_at_first_config.append((configs_root / _QC_FILE).is_file())
        return fake_execute_command(command, shell)

    script = GenerateChampollionConfig()
    script.parse_args(
        [
            str(crops_dir),
            "--dataset",
            _DATASET,
            "--champollion_loc",
            str(champollion_loc),
            *(["--output", str(output)] if output else []),
            *extra_args,
        ]
    )
    script.execute_command = execute
    return _StageRun(script.run(), configs_root, champollion_loc, report_at_first_config)


def _gap(kind, side, region=_REGIONS[0]):
    """Reason of a crop gap."""
    return f"missing_crop:{kind}:{side}:{region}"


def _messages_at(level, pipeline_caplog):
    """Messages of the generate_champollion_config logger records at exactly ``level``."""
    return [r.getMessage() for r in pipeline_caplog.records if r.name == _MODULE and r.levelno == level]


class TestReport:
    def test_report_lands_in_the_output_folder(self, crops, temp_dir):
        output = Path(temp_dir) / "chosen_configs"

        _run_stage(crops, output=output)

        assert (output / _QC_FILE).is_file()

    def test_report_lands_in_the_derivatives_configs_by_default(self, crops):
        assert sorted(_run_stage(crops).rows()) == sorted(_COHORT)

    def test_external_config_leaves_the_report_in_the_configs_root(self, crops, temp_dir):
        external = Path(temp_dir) / "elsewhere" / "local.yaml"

        run = _run_stage(crops, "--external-config", str(external))

        assert run.qc_path.is_file()
        assert not (external.parent / _QC_FILE).exists()

    def test_report_precedes_the_first_config_write(self, crops):
        assert _run_stage(crops).report_at_first_config == [True]

    def test_default_run_writes_nothing_under_champollion_loc(self, crops):
        run = _run_stage(crops)

        assert run.qc_path.is_file()
        assert not run.champollion_loc.exists()

    def test_run_without_a_passed_subject_writes_the_report(self, lost_crops):
        assert {row["qc"] for row in _run_stage(lost_crops).rows().values()} == {0}


class TestRows:
    def test_rows_are_those_of_the_shared_check(self, partial_crops):
        expected = fetch_crop_qc_rows(partial_crops, sorted(_REGIONS), _KINDS)

        assert list(_run_stage(partial_crops).rows().values()) == expected

    def test_report_holds_what_the_shared_check_returns(self, crops):
        marked = [{"participant_id": "sub-marked", "qc": 1, "reason": ""}]

        with patch(f"{_MODULE}.fetch_crop_qc_rows", return_value=marked, create=True):
            run = _run_stage(crops)

        assert list(run.rows().values()) == marked

    def test_complete_subject_has_qc_1(self, partial_crops):
        row = _run_stage(partial_crops).rows()[_SUBJECT]

        assert (row["qc"], row["reason"]) == (1, "")

    def test_incomplete_subject_has_the_missing_crop_reason(self, partial_crops):
        row = _run_stage(partial_crops).rows()[_OTHER_SUBJECT]

        assert (row["qc"], row["reason"]) == (0, _gap("skeleton", "L"))


class TestCheckedRegions:
    def test_each_region_folder_is_checked(self, crops):
        add_crops(crops, [_SUBJECT], _REGIONS[1:], sides="R")

        row = _run_stage(crops).rows()[_OTHER_SUBJECT]

        assert row["reason"] == _gap("skeleton", "R", _REGIONS[1])

    def test_file_under_crop_path_is_not_a_region(self, crops):
        (crops / "notes.txt").write_text("not a region\n")

        assert _run_stage(crops).rows()[_SUBJECT]["reason"] == ""

    def test_region_absent_from_crop_path_is_not_required(self, temp_dir):
        crops = _add_cohort(_crops_dir(temp_dir), regions=_REGIONS[:1])

        assert _run_stage(crops).rows()[_SUBJECT]["qc"] == 1

    def test_external_crops_are_read_in_crop_path(self, temp_dir):
        crops = _add_cohort(Path(temp_dir) / "shared" / _DATASET / "some" / "crops" / "2mm")

        run = _run_stage(crops, "--external_crops")

        assert {row["qc"] for row in run.rows().values()} == {1}


class TestCheckedKinds:
    @pytest.mark.parametrize("kind", ["skeleton", "foldlabel"])
    def test_absent_default_input_type_fails_the_subject(self, crops, kind):
        crop_array(crops, _REGIONS[0], "R", kind).unlink()

        row = _run_stage(crops).rows()[_SUBJECT]

        assert (row["qc"], row["reason"]) == (0, _gap(kind, "R"))

    def test_distbottom_and_extremities_are_not_required(self, crops):
        assert not crop_array(crops, _REGIONS[0], "L", "distbottom").exists()
        assert not crop_array(crops, _REGIONS[0], "L", "extremities").exists()

        assert _run_stage(crops).rows()[_SUBJECT]["qc"] == 1

    def test_absent_mask_npy_fails_the_subject(self, crops):
        crop_array(crops, _REGIONS[1], "L", "mask").unlink()

        row = _run_stage(crops).rows()[_SUBJECT]

        assert (row["qc"], row["reason"]) == (0, _gap("mask", "L", _REGIONS[1]))


class TestSubjectSet:
    def test_one_row_per_subject_of_the_csvs(self, crops):
        assert sorted(_run_stage(crops).rows()) == sorted(_COHORT)

    def test_subject_of_one_csv_gets_a_qc_0_row(self, crops):
        csv_path = crop_csv(crops, _REGIONS[1], "R", "foldlabel")
        csv_path.write_text(csv_path.read_text() + "sub-03\n")

        assert _run_stage(crops).rows()["sub-03"]["qc"] == 0

    def test_crop_path_without_a_subject_gives_no_row(self, temp_dir):
        crops = _crops_dir(temp_dir)
        crops.mkdir(parents=True)

        assert _run_stage(crops).rows() == {}


class TestExitPolicy:
    def test_one_error_record_per_qc0_row(self, partial_crops, pipeline_caplog):
        run = _run_stage(partial_crops)

        reason = run.rows()[_OTHER_SUBJECT]["reason"]
        assert_one_subject_error(_messages_at(logging.ERROR, pipeline_caplog), _OTHER_SUBJECT, reason)

    def test_no_error_record_on_a_complete_run(self, crops, pipeline_caplog):
        run = _run_stage(crops)

        assert run.qc_path.is_file()
        assert _messages_at(logging.ERROR, pipeline_caplog) == []

    def test_no_qc1_row_emits_one_fail_record(self, lost_crops, pipeline_caplog):
        run = _run_stage(lost_crops)

        fails = _messages_at(FAIL, pipeline_caplog)
        assert len(fails) == 1
        assert str(run.qc_path) in fails[0]

    def test_qc1_row_emits_no_fail_record(self, partial_crops, pipeline_caplog):
        run = _run_stage(partial_crops)

        assert run.qc_path.is_file()
        assert _messages_at(FAIL, pipeline_caplog) == []

    def test_no_qc1_row_returns_1(self, lost_crops):
        assert _run_stage(lost_crops).result == 1

    def test_empty_report_returns_1(self, temp_dir):
        crops = _crops_dir(temp_dir)
        crops.mkdir(parents=True)

        assert _run_stage(crops).result == 1

    def test_no_qc1_row_writes_no_yaml_in_the_configs_root(self, lost_crops):
        run = _run_stage(lost_crops)

        assert run.qc_path.is_file()
        assert run.yaml_files() == []

    def test_no_qc1_row_writes_no_yaml_at_the_external_config(self, lost_crops, temp_dir):
        external = Path(temp_dir) / "elsewhere" / "local.yaml"

        run = _run_stage(lost_crops, "--external-config", str(external))

        assert run.qc_path.is_file()
        assert not external.exists()

    def test_partial_exclusion_returns_3(self, partial_crops):
        assert _run_stage(partial_crops).result == _PARTIAL_STATUS

    def test_complete_run_returns_0(self, crops):
        run = _run_stage(crops)

        assert {row["qc"] for row in run.rows().values()} == {1}
        assert run.result == 0

    def test_partial_exclusion_writes_the_region_yamls(self, partial_crops):
        run = _run_stage(partial_crops)

        assert run.qc_path.is_file()
        assert (run.configs_root / "dataset" / _DATASET / "SC-sylv_left.yaml").is_file()

    def test_complete_run_writes_the_region_yamls(self, crops):
        run = _run_stage(crops)

        assert run.qc_path.is_file()
        assert (run.configs_root / "dataset" / _DATASET / "SC-sylv_right.yaml").is_file()
