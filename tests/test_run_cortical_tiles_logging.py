"""LOGGING convention for run_cortical_tiles.py (TASK-LOGGING-BDRABCZUK-940DB0BFBA89).

RunCorticalTiles reports through the champollion_pipeline.run_cortical_tiles logger
instead of print().

REQ-LOGGING-BDRABCZUK-72B7182FAE4D: run() logs the input argument value at LOG.
REQ-LOGGING-BDRABCZUK-3CD073B72CB8: run() logs the output argument value at LOG.
REQ-LOGGING-BDRABCZUK-96189D0D4932: a legacy crops migration is logged at LOG with crops/<old masks version>.
REQ-LOGGING-BDRABCZUK-AE8BA5B863AC: legacy and versioned crops without --overwrite: ERROR, Legacy crops found.
REQ-LOGGING-BDRABCZUK-BE793CD2FCD3: legacy and versioned crops with --overwrite: WARNING, removing legacy.
REQ-LOGGING-BDRABCZUK-E3A392EBDC6B: requested crops exist without --overwrite: WARNING, <version> ... already exist.
REQ-LOGGING-BDRABCZUK-BA817B9F4607: requested crops exist with --overwrite: WARNING, Overwriting ... <version>.
REQ-LOGGING-BDRABCZUK-E8ADAAC230D6: no *mask_cropped.nii.gz file: WARNING.
REQ-LOGGING-BDRABCZUK-499A5AA26075: each written mask .npy path is logged at OUTPUT.
REQ-LOGGING-BDRABCZUK-B8918C62EB6A: the written mask .npy count is logged at OUTPUT.
REQ-LOGGING-BDRABCZUK-12F726C73E22: no fused volume under F: WARNING with the skeleton source directory.
(REQ-LOGGING-BDRABCZUK-AA4776030940, --njobs at or above the core count, and
REQ-LABELSESSION-BDRABCZUK-56365C1DEE1F, the exclusion warning: tests/test_run_cortical_tiles.py.
REQ-LOGGING-BDRABCZUK-C96C8DAAF703 and -22ABEF2F2CBD: tests/test_logging_convention.py, through
LOG_CONVERTED_MODULES.)
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from tests.log_assertions import LOG, OUTPUT

pytestmark = pytest.mark.unit  # noqa: V107

LOGGER_NAME = "champollion_pipeline.run_cortical_tiles"
OLD_VERSION = "old_version_v1"
NEW_VERSION = "new_version_v2"
MASK_SUBPATH = Path("crops") / NEW_VERSION / "2mm" / "SC-sylv" / "mask"


def _script(input_dir: Path, output_dir: Path, *extra: str) -> RunCorticalTiles:
    script = RunCorticalTiles()
    script.parse_args(
        [str(input_dir), str(output_dir), "--path_to_graph", "graphs", "--path_sk_with_hull", "skeleton", *extra]
    )
    return script


def _crops(output_dir: Path) -> Path:
    return output_dir / DERIVATIVES_FOLDER / "crops"


def _legacy_layout(output_dir: Path, *, with_versioned: bool) -> Path:
    """Legacy crops/2mm/ plus a pipeline_loop_2mm.json naming OLD_VERSION; returns the config path."""
    legacy = _crops(output_dir) / "2mm"
    legacy.mkdir(parents=True)
    (legacy / "region.nii.gz").write_text("x")
    if with_versioned:
        versioned = _crops(output_dir) / OLD_VERSION / "2mm"
        versioned.mkdir(parents=True)
        (versioned / "region.nii.gz").write_text("x")
    config = output_dir / "pipeline_loop_2mm.json"
    config.write_text(json.dumps({"masks_version": OLD_VERSION}))
    return config


def _existing_requested_crops(output_dir: Path) -> Path:
    """Non-empty crops/NEW_VERSION/2mm/ and no pipeline_loop_2mm.json; returns the config path."""
    versioned = _crops(output_dir) / NEW_VERSION / "2mm"
    versioned.mkdir(parents=True)
    (versioned / "region.nii.gz").write_text("x")
    return output_dir / "pipeline_loop_2mm.json"


def _assert_record(capture, level: int, text: str) -> logging.LogRecord:
    record = capture.assert_logged(level, text)
    assert record.name == LOGGER_NAME
    return record


# ---------------------------------------------------------------------------
# run(): input and output values
# ---------------------------------------------------------------------------


def _run_mocked(script: RunCorticalTiles) -> None:
    with (
        patch.object(script, "validate_paths", return_value=True),
        patch.object(script, "execute_command", return_value=0),
        patch.object(script, "_preflight_check", return_value=True),
        patch.object(script, "_generate_mask_npys"),
        patch.object(script, "_remove_ventricle_whole_brain"),
        patch("champollion_pipeline.run_cortical_tiles.add_left_and_right_volumes.add_left_and_right_volumes"),
    ):
        assert script.run() == 0


def test_input_logged_at_log(tmp_path, pipeline_caplog):
    input_dir = tmp_path / "graphs_input"
    _run_mocked(_script(input_dir, tmp_path / "tiles_output"))

    _assert_record(pipeline_caplog, LOG, str(input_dir))


def test_output_logged_at_log(tmp_path, pipeline_caplog):
    output_dir = tmp_path / "tiles_output"
    _run_mocked(_script(tmp_path / "graphs_input", output_dir))

    _assert_record(pipeline_caplog, LOG, str(output_dir))


# ---------------------------------------------------------------------------
# _preflight_check: legacy crops and existing crops
# ---------------------------------------------------------------------------


def test_legacy_crops_migration_logged_at_log(tmp_path, pipeline_caplog):
    config = _legacy_layout(tmp_path, with_versioned=False)

    assert _script(tmp_path, tmp_path, "--masks", NEW_VERSION)._preflight_check(str(tmp_path), str(config))

    _assert_record(pipeline_caplog, LOG, f"crops/{OLD_VERSION}")


def test_legacy_crops_conflict_logged_at_error(tmp_path, pipeline_caplog):
    config = _legacy_layout(tmp_path, with_versioned=True)

    assert not _script(tmp_path, tmp_path, "--masks", NEW_VERSION)._preflight_check(str(tmp_path), str(config))

    _assert_record(pipeline_caplog, logging.ERROR, "Legacy crops found")


def test_legacy_crops_overwrite_logged_at_warning(tmp_path, pipeline_caplog):
    config = _legacy_layout(tmp_path, with_versioned=True)
    script = _script(tmp_path, tmp_path, "--masks", NEW_VERSION, "--overwrite")

    assert script._preflight_check(str(tmp_path), str(config))

    _assert_record(pipeline_caplog, logging.WARNING, "removing legacy")


def test_existing_crops_without_overwrite_logged_at_warning(tmp_path, pipeline_caplog):
    config = _existing_requested_crops(tmp_path)

    assert not _script(tmp_path, tmp_path, "--masks", NEW_VERSION)._preflight_check(str(tmp_path), str(config))

    message = _assert_record(pipeline_caplog, logging.WARNING, "already exist").getMessage()
    assert NEW_VERSION in message[: message.index("already exist")], message


def test_existing_crops_overwrite_logged_at_warning(tmp_path, pipeline_caplog):
    config = _existing_requested_crops(tmp_path)
    script = _script(tmp_path, tmp_path, "--masks", NEW_VERSION, "--overwrite")

    assert script._preflight_check(str(tmp_path), str(config))

    message = _assert_record(pipeline_caplog, logging.WARNING, "Overwriting existing crops").getMessage()
    assert NEW_VERSION in message[message.index("Overwriting existing crops") :], message


# ---------------------------------------------------------------------------
# _generate_mask_npys
# ---------------------------------------------------------------------------


def _mask_dir(output_dir: Path) -> Path:
    mask_dir = output_dir / DERIVATIVES_FOLDER / MASK_SUBPATH
    mask_dir.mkdir(parents=True)
    return mask_dir


def _generate_npys(output_dir: Path) -> None:
    volume = MagicMock()
    volume.np = np.ones((2, 2, 2, 1))
    with (
        patch("champollion_pipeline.run_cortical_tiles.aims.read", return_value=volume),
        patch("champollion_pipeline.run_cortical_tiles.np.save"),
    ):
        RunCorticalTiles()._generate_mask_npys(str(output_dir))


def test_no_mask_cropped_files_logged_at_warning(tmp_path, pipeline_caplog):
    _generate_npys(tmp_path)

    _assert_record(pipeline_caplog, logging.WARNING, "no *mask_cropped.nii.gz files found")


def test_written_mask_npy_logged_at_output(tmp_path, pipeline_caplog):
    mask_dir = _mask_dir(tmp_path)
    (mask_dir / "Lmask_cropped.nii.gz").write_bytes(b"fake")

    _generate_npys(tmp_path)

    _assert_record(pipeline_caplog, OUTPUT, str(mask_dir / "Lmask.npy"))


def test_written_mask_npy_count_logged_at_output(tmp_path, pipeline_caplog):
    mask_dir = _mask_dir(tmp_path)
    (mask_dir / "Lmask_cropped.nii.gz").write_bytes(b"fake")
    (mask_dir / "Rmask_cropped.nii.gz").write_bytes(b"fake")
    (mask_dir / "Smask_cropped.nii.gz").write_bytes(b"fake")
    (mask_dir / "Smask.npy").write_bytes(b"existing")

    _generate_npys(tmp_path)

    _assert_record(pipeline_caplog, OUTPUT, "2 mask .npy file(s) written")


# ---------------------------------------------------------------------------
# _remove_ventricle_whole_brain
# ---------------------------------------------------------------------------


def test_no_fused_volume_logged_at_warning(tmp_path, pipeline_caplog):
    skeleton_src_dir = tmp_path / "skeletons" / "2mm"
    skeleton_src_dir.mkdir(parents=True)
    script = _script(tmp_path, tmp_path)

    with patch("champollion_pipeline.run_cortical_tiles.remove_ventricle.remove_ventricle") as remove:
        script._remove_ventricle_whole_brain(str(skeleton_src_dir), str(tmp_path), str(tmp_path))

    remove.assert_not_called()
    _assert_record(pipeline_caplog, logging.WARNING, str(skeleton_src_dir))
