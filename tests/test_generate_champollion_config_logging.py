"""LOGGING convention for generate_champollion_config.py (TASK-LOGGING-BDRABCZUK-1761ECDBE832).

GenerateChampollionConfig reports through the champollion_pipeline.generate_champollion_config
logger instead of print().

REQ-LOGGING-BDRABCZUK-789B9C22A483: a written region dataset YAML is logged at OUTPUT with its path.
REQ-LOGGING-BDRABCZUK-20450AC31E4C: a written dataset_localization YAML is logged at OUTPUT with its path.
REQ-LOGGING-BDRABCZUK-0858C03FFF7D: run() logs the absolute champollion_loc path at LOG.
REQ-LOGGING-BDRABCZUK-1148208E20C7: stale in-checkout dataset configs are logged at WARNING with their path.
(REQ-LOGGING-BDRABCZUK-D596BEC255BD: tests/test_generate_champollion_config_internals.py,
TestCreateDatasetConfigs::test_sulci_without_size_information_are_reported_as_skipped.
REQ-LOGGING-BDRABCZUK-C96C8DAAF703 and -22ABEF2F2CBD: tests/test_logging_convention.py, through
LOG_CONVERTED_MODULES.)

Synthetic crops only; the ``mkdir``/``cp`` shell-outs of run() are replaced in process.
"""

import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from champollion_pipeline.generate_champollion_config import GenerateChampollionConfig
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from tests.log_assertions import LOG, OUTPUT

pytestmark = pytest.mark.unit  # noqa: V107

LOGGER_NAME = "champollion_pipeline.generate_champollion_config"
DATASET = "TEST01"
REF = "dataset_name: REPLACE_DATASET\ncrop: REPLACE_CROP_NAME\nside: REPLACE_SIDE\n"


def _fake_execute_command(cmd, shell=False):
    """In-process stand-in for the ``mkdir -p`` / ``cp`` shell-outs of run()."""
    if cmd[0] == "mkdir":
        os.makedirs(cmd[-1], exist_ok=True)
    elif cmd[0] == "cp":
        shutil.copy(cmd[1], cmd[2])
    return 0


def _make_crop(crop_root: Path, crop_name: str, side: str) -> None:
    """Create <crop_root>/<crop_name>/mask/<side>skeleton.npy holding a (1, 4, 5, 6) array."""
    mask_dir = crop_root / crop_name / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    np.save(mask_dir / f"{side}skeleton.npy", np.zeros((1, 4, 5, 6), dtype=np.int16))


def _crop_tree(tmp_path: Path) -> Path:
    """Pipeline-shaped crops tree holding one region with both sides."""
    crops = tmp_path / "data" / DATASET / "derivatives" / DERIVATIVES_FOLDER / "crops" / "canonical_25" / "2mm"
    _make_crop(crops, "S.C.-sylv.", "L")
    _make_crop(crops, "S.C.-sylv.", "R")
    return crops


def _run(tmp_path: Path, champollion_loc: Path) -> Path:
    """Run GenerateChampollionConfig on a synthetic tree; return the configs root (--output)."""
    out = tmp_path / "configs"
    script = GenerateChampollionConfig()
    script.parse_args(
        [
            str(_crop_tree(tmp_path)),
            "--dataset",
            DATASET,
            "--champollion_loc",
            str(champollion_loc),
            "--output",
            str(out),
        ]
    )
    script.execute_command = _fake_execute_command
    assert script.run() == 0
    return out


def test_region_dataset_yaml_logged_at_output(tmp_path, pipeline_caplog):
    crops = tmp_path / "crops"
    _make_crop(crops, "S.C.-sylv.", "L")
    dataset_loc = tmp_path / "dataset"
    dataset_loc.mkdir()
    script = GenerateChampollionConfig()
    script.parse_args([str(crops), "--dataset", DATASET])

    script._create_dataset_configs(str(crops), str(dataset_loc), REF)

    written = dataset_loc / "SC-sylv_left.yaml"
    assert written.is_file()
    record = pipeline_caplog.assert_logged(OUTPUT, str(written))
    assert record.name == LOGGER_NAME


def test_localization_yaml_logged_at_output(tmp_path, pipeline_caplog):
    dest = tmp_path / "dataset_localization" / "local.yaml"
    script = GenerateChampollionConfig()
    script.parse_args([str(tmp_path), "--dataset", DATASET])

    script._write_localization_yaml(str(dest), "/synthetic/data")

    assert dest.is_file()
    record = pipeline_caplog.assert_logged(OUTPUT, str(dest))
    assert record.name == LOGGER_NAME


def test_champollion_loc_logged_at_log(tmp_path, pipeline_caplog):
    champollion_loc = tmp_path / "champollion_V1"

    _run(tmp_path, champollion_loc)

    record = pipeline_caplog.assert_logged(LOG, os.path.abspath(champollion_loc))
    assert record.name == LOGGER_NAME


def test_stale_checkout_dataset_configs_logged_at_warning(tmp_path, pipeline_caplog):
    champollion_loc = tmp_path / "champollion_V1"
    stale = champollion_loc / "champollion" / "configs" / "dataset" / DATASET
    stale.mkdir(parents=True)

    _run(tmp_path, champollion_loc)

    record = pipeline_caplog.assert_logged(logging.WARNING, str(stale))
    assert record.name == LOGGER_NAME
