"""LOGGING convention for purge_subject.py (TASK-LOGGING-BDRABCZUK-1761ECDBE832).

PurgeSubject reports through the champollion_pipeline.purge_subject logger instead of print().
The deletion listings are results (OUTPUT, on stdout), so the champollion_sulcal_mcp job log
keeps them.

REQ-LOGGING-BDRABCZUK-061FFAA732E9: the purge start is logged at LOG with the subject and derivatives path.
REQ-LOGGING-BDRABCZUK-B32699DFEAA6: a per-subject directory deletion is logged at OUTPUT after Delete dir:.
REQ-LOGGING-BDRABCZUK-CB340A45D968: a per-subject crop file deletion is logged at OUTPUT after Delete file:.
REQ-LOGGING-BDRABCZUK-72A20D868576: an aggregated array row filter is logged at OUTPUT with the array and region.
REQ-LOGGING-BDRABCZUK-8C8D59E9888B: with --dry-run, each deletion record message starts with [dry-run].
REQ-LOGGING-BDRABCZUK-FC255AB5D93C: the purge completion is logged at OUTPUT with Done. and the subject.
(REQ-LOGGING-BDRABCZUK-B32699DFEAA6 under --dry-run, -C1CBF454947E and -D1792CAE3479:
tests/test_purge_subject.py, TestPurgePerSubjectDirs::test_dry_run_prints_delete_dir,
TestPurgeAggregatedArrays::test_handles_subject_not_in_any_array and
TestRunValidation::test_skips_crop_purge_when_crops_2mm_missing.
REQ-LOGGING-BDRABCZUK-C96C8DAAF703 and -22ABEF2F2CBD: tests/test_logging_convention.py, through
LOG_CONVERTED_MODULES.)

Synthetic subject IDs and a synthetic derivatives tree only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from champollion_pipeline.purge_subject import PurgeSubject
from tests.log_assertions import LOG, OUTPUT

pytestmark = pytest.mark.unit  # noqa: V107

LOGGER_NAME = "champollion_pipeline.purge_subject"
SUBJECT = "sub-01"
DRY_RUN_TAG = "[dry-run]"


def _script(derivatives: Path, dry_run: bool = False) -> PurgeSubject:
    script = PurgeSubject()
    script.args = script.parse_args([str(derivatives), "--subject", SUBJECT] + (["--dry-run"] if dry_run else []))
    return script


def _skeleton_dir(derivatives: Path) -> Path:
    skeleton_dir = derivatives / "skeletons" / SUBJECT
    skeleton_dir.mkdir(parents=True)
    return skeleton_dir


def _crop_file(derivatives: Path) -> Path:
    mask_dir = derivatives / "crops" / "2mm" / "region_a" / "mask" / "Rcrops"
    mask_dir.mkdir(parents=True)
    crop_file = mask_dir / f"{SUBJECT}.nii.gz"
    crop_file.touch()
    return crop_file


def _aggregated_array(derivatives: Path) -> Path:
    """region_a/mask/Lskeleton.npy with its subject CSV listing SUBJECT and sub-02."""
    mask_dir = derivatives / "crops" / "2mm" / "region_a" / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    np.save(mask_dir / "Lskeleton.npy", np.zeros((2, 1, 4, 4, 4), dtype=np.float32))
    pd.DataFrame({"participant_id": [SUBJECT, "sub-02"]}).to_csv(mask_dir / "Lskeleton_subject.csv", index=False)
    return mask_dir / "Lskeleton.npy"


def test_purge_start_logged_at_log(tmp_path, pipeline_caplog):
    assert _script(tmp_path).run() == 0

    records = [
        record
        for record in pipeline_caplog.records
        if record.levelno == LOG and SUBJECT in record.getMessage() and str(tmp_path) in record.getMessage()
    ]
    assert records, [(record.levelname, record.getMessage()) for record in pipeline_caplog.records]
    assert records[0].name == LOGGER_NAME


def test_deleted_dir_logged_at_output(tmp_path, pipeline_caplog):
    skeleton_dir = _skeleton_dir(tmp_path)

    _script(tmp_path)._purge_per_subject_dirs(str(tmp_path))

    assert not skeleton_dir.exists()
    record = pipeline_caplog.assert_logged(OUTPUT, "Delete dir:")
    assert str(skeleton_dir) in record.getMessage().split("Delete dir:", 1)[1]
    assert record.name == LOGGER_NAME


def test_deleted_crop_file_logged_at_output(tmp_path, pipeline_caplog):
    crop_file = _crop_file(tmp_path)

    _script(tmp_path)._purge_per_subject_crop_files(str(tmp_path / "crops" / "2mm"))

    assert not crop_file.exists()
    record = pipeline_caplog.assert_logged(OUTPUT, "Delete file:")
    assert str(crop_file) in record.getMessage().split("Delete file:", 1)[1]
    assert record.name == LOGGER_NAME


def test_array_row_filter_logged_at_output(tmp_path, pipeline_caplog):
    npy_path = _aggregated_array(tmp_path)

    _script(tmp_path)._purge_aggregated_arrays(str(tmp_path / "crops" / "2mm"))

    assert np.load(npy_path).shape[0] == 1
    records = [
        record
        for record in pipeline_caplog.records
        if record.levelno == OUTPUT and npy_path.name in record.getMessage() and "region_a" in record.getMessage()
    ]
    assert records, [(record.levelname, record.getMessage()) for record in pipeline_caplog.records]
    assert records[0].name == LOGGER_NAME


@pytest.mark.parametrize("kind", ["Delete dir:", "Delete file:", "Filter row(s)"])
def test_dry_run_deletion_record_starts_with_dry_run_tag(tmp_path, pipeline_caplog, kind):
    _skeleton_dir(tmp_path)
    _crop_file(tmp_path)
    _aggregated_array(tmp_path)

    assert _script(tmp_path, dry_run=True).run() == 0

    messages = [record.getMessage() for record in pipeline_caplog.records if kind in record.getMessage()]
    assert messages, [(record.levelname, record.getMessage()) for record in pipeline_caplog.records]
    assert [message for message in messages if not message.startswith(DRY_RUN_TAG)] == []


def test_purge_completion_logged_at_output(tmp_path, pipeline_caplog):
    assert _script(tmp_path).run() == 0

    record = pipeline_caplog.assert_logged(OUTPUT, "Done.")
    assert SUBJECT in record.getMessage().split("Done.", 1)[1]
    assert record.name == LOGGER_NAME
