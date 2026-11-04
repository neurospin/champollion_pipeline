#!/usr/bin/env python3
"""
Stage 4 input QC tests of generate_embeddings.py: input_qc.tsv over the crops of the run's region models,
and the exit policy (user decisions 2026-11-04). Synthetic fixtures only; evaluate.py is faked.

The crops come from tests/stage2_layout.py. Only the skeleton files count: evaluate.py reads nothing else.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.derivatives_layout import DEFAULT_MASKS_VERSION, compute_region_embeddings_dir
from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from champollion_pipeline.utils.lib import CORTICAL_TILES_VERSION
from champollion_pipeline.utils.qc_report import fetch_crop_qc_rows, read_qc_tsv
from tests.log_assertions import FAIL
from tests.stage2_layout import add_crops, crop_array
from tests.test_generate_embeddings_any_failure import FakeEvaluate, make_models, make_script

_LOGGER_NAME = "champollion_pipeline.generate_embeddings"
_QC_FILE = "input_qc.tsv"
_KIND = "skeleton"
_PARTIAL_STATUS = 3
_EVALUATE_FAILURE = 7
_MASKS_OTHER = "other_masks"
_CROPS = ("S.C.-sylv.", "S.T.s.")
_MODELS = ("SC-sylv_left", "SC-sylv_right", "STs_left", "STs_right")
_FIRST_CROP_MODELS = _MODELS[:2]
_SUBJECT = "sub-01"
_OTHER_SUBJECT = "sub-02"
_COHORT = (_SUBJECT, _OTHER_SUBJECT)
_KEPT_EMBEDDINGS = f"ID,dim1\n{_SUBJECT},0.123456789012345678\n"
_EMBEDDINGS = f"{_KEPT_EMBEDDINGS}{_OTHER_SUBJECT},0.5\n"


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage4_input_qc():
    """Run the real input QC: overrides the conftest fixture of the same name."""


def _default_crops_dir(dataset, masks=DEFAULT_MASKS_VERSION):
    """Crops folder of a default run on ``dataset``."""
    return Path(dataset) / "derivatives" / f"cortical_tiles-{CORTICAL_TILES_VERSION}" / "crops" / masks / "2mm"


def _lay_complete(crops_dir):
    add_crops(crops_dir, _COHORT, _CROPS)


def _lay_partial(crops_dir):
    """Crops where _OTHER_SUBJECT misses the last region."""
    add_crops(crops_dir, _COHORT, _CROPS[:1])
    add_crops(crops_dir, [_SUBJECT], _CROPS[1:])


def _lay_disjoint(crops_dir):
    """Crops where each subject misses a region."""
    add_crops(crops_dir, [_SUBJECT], _CROPS[:1])
    add_crops(crops_dir, [_OTHER_SUBJECT], _CROPS[1:])


def _lay_empty_array(crops_dir):
    """Complete crops but for a zero-byte skeleton array."""
    _lay_complete(crops_dir)
    crop_array(crops_dir, _CROPS[0], "L", _KIND).write_bytes(b"")


def _lay_first_crop(crops_dir):
    """Crops of the first region alone."""
    add_crops(crops_dir, _COHORT, _CROPS[:1])


def _lay_left_of_first_crop(crops_dir):
    """Left crops of the first region alone."""
    add_crops(crops_dir, _COHORT, _CROPS[:1], sides="L")


def _lay_left_gap(crops_dir):
    """Left crops where _OTHER_SUBJECT misses the last region."""
    add_crops(crops_dir, _COHORT, _CROPS[:1], sides="L")
    add_crops(crops_dir, [_SUBJECT], _CROPS[1:], sides="L")


def _lay_left_then_right(crops_dir):
    """Left crops of the first region, right crops of the last one."""
    add_crops(crops_dir, _COHORT, _CROPS[:1], sides="L")
    add_crops(crops_dir, _COHORT, _CROPS[1:], sides="R")


def _lay_nothing(_crops_dir):
    """No crop at all."""


@dataclass
class _StageRun:
    """What run() did with fetch_models and evaluate.py mocked."""

    result: int
    qc_path: Path
    crops_dir: Path
    output_base: Path
    evaluated: list[str]
    report_at_evaluate: list[bool]

    def rows(self):
        """Rows of input_qc.tsv, keyed by participant_id."""
        assert self.qc_path.is_file(), f"{self.qc_path} was not written"
        return {row["participant_id"]: row for row in read_qc_tsv(self.qc_path)}

    def qc_values(self):
        """The qc value of each participant_id of input_qc.tsv."""
        return {subject: row["qc"] for subject, row in self.rows().items()}

    def embeddings_file(self, region):
        """The full_embeddings.csv of ``region``."""
        return self.output_base / region / "full_embeddings.csv"


@dataclass(frozen=True)
class _Stage:
    """A run of the stage: ``lay`` writes the crops in ``crops_dir``, default the one of a default run."""

    lay: Callable[[Path], None] = _lay_complete
    extra_args: tuple[str, ...] = ()
    codes: tuple[tuple[str, int], ...] = ()
    crops_dir: Path | None = None
    has_output_option: bool = True
    reused_regions: tuple[str, ...] = ()


_DEFAULT_STAGE = _Stage()


def _run_stage(root, stage=_DEFAULT_STAGE):
    """Run GenerateEmbeddings.run() on _MODELS under ``root`` as ``stage`` says."""
    root = Path(root)
    dataset, output = root / "dataset", root / "out"
    crops_dir = _default_crops_dir(dataset) if stage.crops_dir is None else root / stage.crops_dir
    stage.lay(crops_dir)
    models = make_models(root, list(_MODELS))
    output_args = ["--output", str(output)] if stage.has_output_option else []
    script = make_script([str(models), str(dataset), *output_args, *stage.extra_args])
    output_base = (
        output if stage.has_output_option else Path(compute_region_embeddings_dir(str(dataset), script.args.masks))
    )
    for region in stage.reused_regions:
        (output_base / region).mkdir(parents=True)
        (output_base / region / "full_embeddings.csv").write_text(_EMBEDDINGS)
    qc_path = output_base / _QC_FILE

    evaluate = FakeEvaluate(dict(stage.codes), _EMBEDDINGS)
    report_at_evaluate = []

    def execute(command, **kwargs):
        report_at_evaluate.append(qc_path.is_file())
        return evaluate(command, **kwargs)

    script.execute_command = execute
    with patch.object(GenerateEmbeddings, "fetch_models", return_value=str(models)):
        result = script.run()
    return _StageRun(result, qc_path, crops_dir, output_base, evaluate.calls, report_at_evaluate)


def _messages_at(level, pipeline_caplog):
    """Messages of the generate_embeddings logger records at exactly ``level``."""
    return [r.getMessage() for r in pipeline_caplog.records if r.name == _LOGGER_NAME and r.levelno == level]


class TestReport:
    def test_complete_crops_give_qc_1_rows(self, tmp_path):
        assert _run_stage(tmp_path).qc_values() == {_SUBJECT: 1, _OTHER_SUBJECT: 1}

    def test_one_row_per_subject_of_a_subject_csv(self, tmp_path):
        assert sorted(_run_stage(tmp_path, _Stage(_lay_disjoint)).rows()) == sorted(_COHORT)

    def test_other_crop_kinds_are_not_required(self, tmp_path):
        run = _run_stage(tmp_path)

        assert not crop_array(run.crops_dir, _CROPS[0], "L", "mask").exists()
        assert not crop_array(run.crops_dir, _CROPS[0], "L", "foldlabel").exists()
        assert run.qc_values() == {_SUBJECT: 1, _OTHER_SUBJECT: 1}

    def test_complete_subject_has_an_empty_reason(self, tmp_path):
        assert _run_stage(tmp_path).rows()[_SUBJECT]["reason"] == ""

    def test_subject_missing_from_a_region_has_qc_0(self, tmp_path):
        run = _run_stage(tmp_path, _Stage(_lay_partial))

        assert run.qc_values() == {_SUBJECT: 1, _OTHER_SUBJECT: 0}

    def test_reason_names_the_lost_side_and_region(self, tmp_path):
        reason = _run_stage(tmp_path, _Stage(_lay_partial)).rows()[_OTHER_SUBJECT]["reason"]

        assert set(reason.split(";")) == {f"missing_crop:{_KIND}:{side}:{_CROPS[1]}" for side in "LR"}

    def test_rows_are_those_of_the_shared_check(self, tmp_path):
        run = _run_stage(tmp_path, _Stage(_lay_disjoint))

        expected = fetch_crop_qc_rows(run.crops_dir, _CROPS, [_KIND])
        assert {subject: set(row["reason"].split(";")) for subject, row in run.rows().items()} == {
            row["participant_id"]: set(row["reason"].split(";")) for row in expected
        }

    def test_report_precedes_the_first_evaluate_command(self, tmp_path):
        run = _run_stage(tmp_path)

        assert run.report_at_evaluate
        assert all(run.report_at_evaluate)

    def test_default_output_folder_holds_the_report(self, tmp_path):
        run = _run_stage(tmp_path, _Stage(has_output_option=False))

        assert sorted(run.rows()) == sorted(_COHORT)

    def test_rerun_without_overwrite_checks_the_reused_regions(self, tmp_path):
        run = _run_stage(tmp_path, _Stage(_lay_partial, reused_regions=_MODELS[2:]))

        assert run.qc_values()[_OTHER_SUBJECT] == 0

    def test_masks_option_selects_the_crops_folder(self, tmp_path):
        crops_dir = _default_crops_dir("dataset", _MASKS_OTHER)

        run = _run_stage(tmp_path, _Stage(extra_args=("--masks", _MASKS_OTHER), crops_dir=crops_dir))

        assert run.qc_values() == {_SUBJECT: 1, _OTHER_SUBJECT: 1}

    def test_legacy_option_selects_the_legacy_crops_folder(self, tmp_path):
        crops_dir = Path("dataset") / "derivatives" / "deep_folding-2025" / "crops" / "2mm"

        run = _run_stage(tmp_path, _Stage(extra_args=("--legacy",), crops_dir=crops_dir))

        assert run.qc_values() == {_SUBJECT: 1, _OTHER_SUBJECT: 1}

    def test_region_outside_regions_is_not_checked(self, tmp_path):
        run = _run_stage(tmp_path, _Stage(_lay_first_crop, extra_args=("--regions", *_FIRST_CROP_MODELS)))

        assert run.qc_values() == {_SUBJECT: 1, _OTHER_SUBJECT: 1}

    def test_region_model_without_a_crop_folder_excludes_the_subjects(self, tmp_path):
        rows = _run_stage(tmp_path, _Stage(_lay_first_crop)).rows()

        assert f"missing_crop:{_KIND}:L:STs" in rows[_SUBJECT]["reason"]
        assert rows[_SUBJECT]["qc"] == 0


class TestSides:
    def test_left_run_passes_on_left_crops(self, tmp_path):
        run = _run_stage(tmp_path, _Stage(_lay_left_of_first_crop, extra_args=("--regions", _MODELS[0])))

        assert run.qc_values() == {_SUBJECT: 1, _OTHER_SUBJECT: 1}

    def test_each_region_is_checked_on_its_own_side(self, tmp_path):
        run = _run_stage(tmp_path, _Stage(_lay_left_then_right, extra_args=("--regions", _MODELS[0], _MODELS[3])))

        assert run.qc_values() == {_SUBJECT: 1, _OTHER_SUBJECT: 1}

    def test_gap_on_a_run_side_is_reported(self, tmp_path):
        run = _run_stage(tmp_path, _Stage(_lay_left_gap, extra_args=("--regions", _MODELS[0], _MODELS[2])))

        assert run.rows()[_OTHER_SUBJECT]["reason"] == f"missing_crop:{_KIND}:L:{_CROPS[1]}"

    def test_region_run_on_both_sides_needs_both(self, tmp_path):
        run = _run_stage(tmp_path, _Stage(_lay_left_of_first_crop, extra_args=("--regions", *_FIRST_CROP_MODELS)))

        assert run.qc_values() == {_SUBJECT: 0, _OTHER_SUBJECT: 0}


class TestErrorRecords:
    def test_one_error_record_per_qc0_row(self, tmp_path, pipeline_caplog):
        run = _run_stage(tmp_path, _Stage(_lay_partial))

        errors = _messages_at(logging.ERROR, pipeline_caplog)
        assert len(errors) == 1
        assert _OTHER_SUBJECT in errors[0]
        assert run.rows()[_OTHER_SUBJECT]["reason"] in errors[0]

    def test_no_error_record_on_complete_crops(self, tmp_path, pipeline_caplog):
        run = _run_stage(tmp_path)

        assert run.qc_path.is_file()
        assert _messages_at(logging.ERROR, pipeline_caplog) == []


@pytest.mark.parametrize(
    "lay",
    [_lay_disjoint, _lay_empty_array, _lay_first_crop, _lay_nothing],
    ids=["each_subject_lost_a_region", "empty_array", "region_absent", "no_crop"],
)
class TestNoUsableSubject:
    def test_no_qc1_row_returns_1(self, tmp_path, lay):
        assert _run_stage(tmp_path, _Stage(lay)).result == 1

    def test_no_qc1_row_emits_one_fail_record(self, tmp_path, pipeline_caplog, lay):
        run = _run_stage(tmp_path, _Stage(lay))

        fails = _messages_at(FAIL, pipeline_caplog)
        assert len(fails) == 1
        assert str(run.qc_path) in fails[0]

    def test_no_qc1_row_runs_no_inference(self, tmp_path, lay):
        assert _run_stage(tmp_path, _Stage(lay)).evaluated == []


class TestPartialExclusion:
    def test_partial_exclusion_returns_3(self, tmp_path):
        assert _run_stage(tmp_path, _Stage(_lay_partial)).result == _PARTIAL_STATUS

    def test_failed_region_wins_over_partial_exclusion(self, tmp_path):
        run = _run_stage(tmp_path, _Stage(_lay_partial, codes=((_MODELS[0], _EVALUATE_FAILURE),)))

        assert run.qc_values() == {_SUBJECT: 1, _OTHER_SUBJECT: 0}
        assert run.result == _EVALUATE_FAILURE

    def test_qc1_row_emits_no_fail_record(self, tmp_path, pipeline_caplog):
        run = _run_stage(tmp_path, _Stage(_lay_partial))

        assert run.qc_path.is_file()
        assert _messages_at(FAIL, pipeline_caplog) == []


class TestExcludedRows:
    def test_qc0_subject_leaves_each_embeddings_file(self, tmp_path):
        run = _run_stage(tmp_path, _Stage(_lay_partial))

        assert run.qc_values()[_OTHER_SUBJECT] == 0
        assert {run.embeddings_file(region).read_text().count(_OTHER_SUBJECT) for region in _MODELS} == {0}

    def test_reused_embeddings_file_loses_the_qc0_subject(self, tmp_path):
        run = _run_stage(tmp_path, _Stage(_lay_partial, reused_regions=_MODELS[:1]))

        assert run.qc_values()[_OTHER_SUBJECT] == 0
        assert _OTHER_SUBJECT not in run.embeddings_file(_MODELS[0]).read_text()

    def test_failed_region_leaves_no_embeddings_file(self, tmp_path):
        run = _run_stage(tmp_path, _Stage(_lay_partial, codes=((_MODELS[0], _EVALUATE_FAILURE),)))

        assert run.qc_values()[_OTHER_SUBJECT] == 0
        assert not run.embeddings_file(_MODELS[0]).exists()
        assert _OTHER_SUBJECT not in run.embeddings_file(_MODELS[1]).read_text()

    def test_kept_rows_stay_byte_identical(self, tmp_path):
        run = _run_stage(tmp_path, _Stage(_lay_partial))

        assert run.qc_values()[_SUBJECT] == 1
        assert run.embeddings_file(_MODELS[0]).read_text() == _KEPT_EMBEDDINGS

    def test_complete_cohort_leaves_the_files_unchanged(self, tmp_path):
        run = _run_stage(tmp_path)

        assert run.qc_path.is_file()
        assert run.embeddings_file(_MODELS[0]).read_text() == _EMBEDDINGS


class TestInference:
    @pytest.mark.parametrize("lay", [_lay_complete, _lay_partial], ids=["complete", "partial"])
    def test_qc1_row_evaluates_each_region_model(self, tmp_path, lay):
        run = _run_stage(tmp_path, _Stage(lay))

        assert run.qc_path.is_file()
        assert sorted(run.evaluated) == sorted(_MODELS)

    def test_region_with_existing_embeddings_is_not_evaluated(self, tmp_path):
        run = _run_stage(tmp_path, _Stage(reused_regions=_MODELS[:1]))

        assert run.qc_path.is_file()
        assert sorted(run.evaluated) == sorted(_MODELS[1:])


class TestCompleteCohort:
    def test_complete_crops_return_0(self, tmp_path):
        run = _run_stage(tmp_path)

        assert run.qc_path.is_file()
        assert run.result == 0

    def test_complete_crops_keep_the_evaluate_status(self, tmp_path):
        run = _run_stage(tmp_path, _Stage(codes=((_MODELS[0], _EVALUATE_FAILURE),)))

        assert run.qc_path.is_file()
        assert run.result == _EVALUATE_FAILURE
