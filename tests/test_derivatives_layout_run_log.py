#!/usr/bin/env python3
"""
Run-log location and naming in derivatives_layout (RUNLOG epic, TASK-RUNLOG-BDRABCZUK-75A665A1B97C).

Each stage writes its run log to <derivatives folder>/.log/<stage>_<YYYYmmdd-HHMMSS>.log, where
the derivatives folder is:

- morphologist: <root>/derivatives/morphologist-6.0;
- cortical_tiles: <root>/derivatives/cortical_tiles-<cortical tiles version>;
- config, embeddings, combine, snapshots, training: <root>/derivatives/champollion_V1/<masks_version>
  (per mask version, next to that version's outputs).

Pure path computation: nothing is created on disk; the start time is passed in.
"""

from datetime import datetime
from os.path import join

import pytest

from champollion_pipeline.derivatives_layout import (
    RUN_LOG_STAGES,
    compute_masks_version_dir,
    compute_run_log_derivatives_dir,
    compute_run_log_path,
)
from champollion_pipeline.utils.lib import CORTICAL_TILES_VERSION

CHAMPOLLION_STAGES = ("config", "embeddings", "combine", "snapshots", "training")
EXPECTED_STAGES = ("morphologist", "cortical_tiles", *CHAMPOLLION_STAGES)


# REQ-RUNLOG-BDRABCZUK-4F308618DDD6
def test_run_log_path_is_under_dot_log_with_timestamped_name():
    started_at = datetime(2026, 10, 21, 14, 7, 33)
    path = compute_run_log_path("/data/ds/derivatives/morphologist-6.0", "morphologist", started_at)
    assert path == "/data/ds/derivatives/morphologist-6.0/.log/morphologist_20261021-140733.log"


# REQ-RUNLOG-BDRABCZUK-4F308618DDD6
def test_run_log_path_zero_pads_timestamp_fields():
    started_at = datetime(2027, 1, 2, 3, 4, 5)
    path = compute_run_log_path("/d", "embeddings", started_at)
    assert path == join("/d", ".log", "embeddings_20270102-030405.log")


# REQ-RUNLOG-BDRABCZUK-4F308618DDD6
def test_run_log_path_creates_nothing_on_disk(tmp_path):
    derivatives_dir = tmp_path / "derivatives" / "champollion_V1" / "canonical_25"
    path = compute_run_log_path(str(derivatives_dir), "combine", datetime(2026, 10, 21, 0, 0, 0))
    assert path == str(derivatives_dir / ".log" / "combine_20261021-000000.log")
    assert list(tmp_path.iterdir()) == []


# REQ-RUNLOG-BDRABCZUK-D90EF1126A05
def test_morphologist_run_log_folder():
    assert compute_run_log_derivatives_dir("morphologist", "/data/ds") == "/data/ds/derivatives/morphologist-6.0"


# REQ-RUNLOG-BDRABCZUK-A91C0E954E3B
def test_cortical_tiles_run_log_folder_default_version():
    expected = f"/data/ds/derivatives/cortical_tiles-{CORTICAL_TILES_VERSION}"
    assert compute_run_log_derivatives_dir("cortical_tiles", "/data/ds") == expected


# REQ-RUNLOG-BDRABCZUK-A91C0E954E3B
def test_cortical_tiles_run_log_folder_explicit_version():
    folder = compute_run_log_derivatives_dir("cortical_tiles", "/data/ds", cortical_tiles_version="2027")
    assert folder == "/data/ds/derivatives/cortical_tiles-2027"


# REQ-RUNLOG-BDRABCZUK-33A8DB747677
@pytest.mark.parametrize("stage", CHAMPOLLION_STAGES)
def test_champollion_stages_run_log_folder_is_mask_version_dir(stage):
    folder = compute_run_log_derivatives_dir(stage, "/data/ds", masks_version="canonical_30")
    assert folder == "/data/ds/derivatives/champollion_V1/canonical_30"
    assert folder == compute_masks_version_dir("/data/ds", "canonical_30")


# REQ-RUNLOG-BDRABCZUK-FDB89CB9A4BC
def test_unknown_stage_raises_value_error():
    assert set(RUN_LOG_STAGES) == set(EXPECTED_STAGES)
    with pytest.raises(ValueError) as excinfo:
        compute_run_log_derivatives_dir("embedings", "/data/ds")
    message = str(excinfo.value)
    for stage in EXPECTED_STAGES:
        assert stage in message
