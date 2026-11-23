#!/usr/bin/env python3
"""
Stage 3 input and output QC of generate_champollion_config.py given a per-region Mask version map.

With --region-masks, the QC follows the map (user decision Q6, 2026-11-23): input_qc.tsv checks the crops of each
region model name in its mapped crops/<version>/2mm folder, and output_qc.tsv holds one row per region model name
whose region that folder holds. A mapped name whose region has no folder there is ignored, even when crop_path
(the --masks folder) holds that region. Both reports run for real here (the conftest stubs are overridden).
Synthetic crops only; each run passes an explicit --output and --champollion_loc under the test folder.
"""

import csv

import pytest

from champollion_pipeline.derivatives_layout import DEFAULT_MASKS_VERSION
from champollion_pipeline.utils.qc_report import read_qc_tsv
from tests.stage2_layout import crop_csv
from tests.test_generate_champollion_config_region_versions import (
    _COHORT,
    _OTHER_VERSION,
    _STS,
    _SYLV,
    _configs_root,
    _crops_dir,
    _full_map,
    _lay_crop,
    _model_name,
    _refusal,
    _run_stage,
    build_unmapped_argv,
)

_INPUT_QC = "input_qc.tsv"
_OUTPUT_QC = "output_qc.tsv"
_MISSING_CROP = "missing_crop"


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage3_input_qc():
    """Run the real input QC: overrides the conftest fixture of the same name."""


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage3_output_qc():
    """Run the real output QC: overrides the conftest fixture of the same name."""


def _input_rows(root):
    """Rows of input_qc.tsv of the run under ``root``, keyed by participant_id."""
    path = _configs_root(root) / _INPUT_QC
    assert path.is_file(), f"{path} was not written"
    return {row["participant_id"]: row for row in read_qc_tsv(path)}


def _output_checks(root):
    """The check cells of output_qc.tsv of the run under ``root``, in file order."""
    path = _configs_root(root) / _OUTPUT_QC
    assert path.is_file(), f"{path} was not written"
    with path.open(newline="", encoding="utf-8") as report:
        return [row["check"] for row in csv.DictReader(report, delimiter="\t")]


def _drop_second_subject(root, version, region):
    """Leave the second subject of the cohort out of the left skeleton CSV of ``region`` in the ``version`` folder."""
    crop_csv(_crops_dir(root, version), region, "L", "skeleton").write_text(f"Subject\n{_COHORT[0]}\n")


def _yaml_names(region):
    """Region dataset YAML file names of both hemispheres of ``region``."""
    return [f"{_model_name(region, side)}.yaml" for side in "LR"]


class TestInputQcFollowsTheMap:
    """input_qc.tsv over the crops of each region model name in its mapped crops/<version>/2mm folder."""

    def test_gap_in_the_mapped_folder_fails_the_subject(self, temp_dir):
        """REQ-MODELVER-BDRABCZUK-A8C71095194F: a crop of M missing from crops/V/2mm gives that subject qc 0."""
        _lay_crop(temp_dir, _OTHER_VERSION, _SYLV)
        _lay_crop(temp_dir, DEFAULT_MASKS_VERSION, _STS)
        _drop_second_subject(temp_dir, _OTHER_VERSION, _SYLV)

        _run_stage(temp_dir, DEFAULT_MASKS_VERSION, _full_map({_SYLV: _OTHER_VERSION, _STS: DEFAULT_MASKS_VERSION}))

        rows = _input_rows(temp_dir)
        assert rows[_COHORT[1]]["qc"] == 0
        assert f"{_MISSING_CROP}:skeleton:L:{_SYLV}" in str(rows[_COHORT[1]]["reason"])
        assert rows[_COHORT[0]]["qc"] == 1

    def test_mapped_name_without_a_crop_folder_adds_no_reason(self, temp_dir):
        """REQ-MODELVER-BDRABCZUK-A8023A03A3F7: no missing_crop reason for M without its region folder."""
        _lay_crop(temp_dir, _OTHER_VERSION, _SYLV)
        _lay_crop(temp_dir, DEFAULT_MASKS_VERSION, _STS)
        _drop_second_subject(temp_dir, DEFAULT_MASKS_VERSION, _STS)

        _run_stage(temp_dir, DEFAULT_MASKS_VERSION, _full_map({_SYLV: _OTHER_VERSION, _STS: _OTHER_VERSION}))

        rows = _input_rows(temp_dir)
        assert sorted(rows) == sorted(_COHORT)
        assert {subject: row["reason"] for subject, row in rows.items() if _STS in str(row["reason"])} == {}


class TestOutputQcFollowsTheMap:
    """output_qc.tsv rows follow the region model names of the map whose region the mapped folder holds."""

    def test_one_row_per_configured_region_model_name(self, temp_dir):
        """REQ-MODELVER-BDRABCZUK-B50C5D9C4F3B: one row per M whose region crops/V/2mm holds, check = M YAML."""
        _lay_crop(temp_dir, _OTHER_VERSION, _SYLV)
        _lay_crop(temp_dir, DEFAULT_MASKS_VERSION, _STS)

        _run_stage(temp_dir, DEFAULT_MASKS_VERSION, _full_map({_SYLV: _OTHER_VERSION, _STS: DEFAULT_MASKS_VERSION}))

        checks = _output_checks(temp_dir)
        assert {name: checks.count(name) for name in (*_yaml_names(_SYLV), *_yaml_names(_STS))} == dict.fromkeys(
            (*_yaml_names(_SYLV), *_yaml_names(_STS)), 1
        )

    def test_mapped_name_without_a_crop_folder_gets_no_row(self, temp_dir):
        """REQ-MODELVER-BDRABCZUK-FAF88B88F448: no row for M when crops/V/2mm holds no folder of its region."""
        _lay_crop(temp_dir, _OTHER_VERSION, _SYLV)
        _lay_crop(temp_dir, DEFAULT_MASKS_VERSION, _STS)

        _run_stage(temp_dir, DEFAULT_MASKS_VERSION, _full_map({_SYLV: _OTHER_VERSION, _STS: _OTHER_VERSION}))

        checks = _output_checks(temp_dir)
        assert set(_yaml_names(_SYLV)) <= set(checks)
        assert [check for check in checks if check in _yaml_names(_STS)] == []


class TestUnmappedRegionWritesNothing:
    """REQ-MODELVER-BDRABCZUK-203810F56823: an unmapped crop_path region stops the run before any write (Q2)."""

    def test_unmapped_region_model_name_writes_no_file(self, temp_dir, capsys):
        _refusal(capsys, build_unmapped_argv(temp_dir))

        configs_root = _configs_root(temp_dir)
        assert not configs_root.exists() or list(configs_root.rglob("*")) == []
