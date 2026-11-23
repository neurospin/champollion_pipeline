#!/usr/bin/env python3
"""
Stage 3 tests of generate_champollion_config.py given a per-region Mask version map.

A model set may mix Mask versions: each region model name (region and hemisphere, e.g. SC-sylv_left) carries the
Mask version its model was trained on, and stage 2 crops each version into its own crops/<version>/2mm folder. The
region dataset YAML of a region model name must then name, and take its input_size from, the crops of its version.

The map (region model name -> Mask version) reaches stage 3 as --region-masks NAME=VERSION pairs (user decision
2026-11-23), written by ``_build_argv`` only. Stage 3 refuses --masks and --external_crops beside the map, and a
region folder of crop_path whose region model name the map does not name (exit 2). Synthetic crops only, complete
for ``_COHORT`` so the input QC over the mapped folders passes; each run passes an explicit --output and
--champollion_loc under the test folder.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from champollion_pipeline.derivatives_layout import DEFAULT_MASKS_VERSION, MASK_VERSIONS, compute_region_model_name
from champollion_pipeline.generate_champollion_config import CONFIG_PATH_KEYS, GenerateChampollionConfig
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from tests.stage2_layout import add_crops
from tests.test_generate_champollion_config_internals import fake_execute_command

_DATASET = "TEST01"
_SYLV = "S.C.-sylv."
_STS = "S.T.s."
_OTHER_VERSION = next(version for version in MASK_VERSIONS if version != DEFAULT_MASKS_VERSION)
_MAP_OPTION = "--region-masks"
_KINDS = ("skeleton", "foldlabel", "mask")
_COHORT = ("sub-01", "sub-02")


def _crops_dir(root, version):
    """crops/<version>/2mm folder of the dataset derivatives under ``root``."""
    return Path(root) / "data" / _DATASET / "derivatives" / DERIVATIVES_FOLDER / "crops" / version / "2mm"


def _lay_crop(root, version, region, sides="LR", shape=(1, 4, 5, 6)):
    """Lay complete crops of ``_COHORT`` in ``region`` for each of ``sides`` in the ``version`` crops folder.

    The skeleton array, which stage 3 reads the input_size from, has ``shape``.
    """
    add_crops(_crops_dir(root, version), _COHORT, [region], _KINDS, sides)
    mask_dir = _crops_dir(root, version) / region / "mask"
    for side in sides:
        np.save(mask_dir / f"{side}skeleton.npy", np.zeros(shape, dtype=np.int16))


def _model_name(region, side):
    """Region model name of ``region`` for side "L" or "R"."""
    return compute_region_model_name(region, "left" if side == "L" else "right")


def _configs_root(root):
    """The --output configs root of every run under ``root``."""
    return Path(root) / "configs"


def _build_argv(root, crop_version, region_masks=None, masks=None, extra=()):
    """Command line of a run on the ``crop_version`` crops folder.

    ``region_masks``, when given, is the per-region Mask version map (region model name -> Mask version), passed
    as --region-masks NAME=VERSION pairs.
    """
    root = Path(root)
    argv = [
        str(_crops_dir(root, crop_version)),
        "--dataset",
        _DATASET,
        "--champollion_loc",
        str(root / "champollion_V1_checkout"),
        "--output",
        str(_configs_root(root)),
        *extra,
    ]
    if masks is not None:
        argv += ["--masks", masks]
    if region_masks is not None:
        argv += [_MAP_OPTION, *(f"{name}={version}" for name, version in region_masks.items())]
    return argv


def _start_stage(argv):
    """Parse ``argv`` and run generate_champollion_config; return its status."""
    script = GenerateChampollionConfig()
    script.parse_args(argv)
    script.execute_command = fake_execute_command
    return script.run()


def _run_stage(root, crop_version, region_masks=None, masks=None):
    """Run generate_champollion_config on the ``crop_version`` crops folder; return the region YAMLs folder."""
    _start_stage(_build_argv(root, crop_version, region_masks, masks))
    return _configs_root(root) / "dataset" / _DATASET


def _refusal(capsys, argv):
    """The argparse error message of a run of ``argv`` that exits with status 2.

    Fails when the run does not exit, exits with another status, or refuses --region-masks as an unknown option.
    """
    with pytest.raises(SystemExit) as stop:
        _start_stage(argv)
    assert stop.value.code == 2
    message = capsys.readouterr().err.partition("error: ")[2]
    assert "unrecognized arguments" not in message, message
    return message


def build_unmapped_argv(root):
    """Run on crop_path holding _SYLV and _STS, with a map leaving out the right hemisphere of _STS."""
    for region in (_SYLV, _STS):
        _lay_crop(root, DEFAULT_MASKS_VERSION, region)
    region_masks = _full_map({_SYLV: _OTHER_VERSION})
    region_masks[_model_name(_STS, "L")] = DEFAULT_MASKS_VERSION
    return _build_argv(root, DEFAULT_MASKS_VERSION, region_masks)


def _full_map(versions):
    """Map giving both hemispheres of each region of ``versions`` (region -> Mask version) its version."""
    return {_model_name(region, side): version for region, version in versions.items() for side in "LR"}


def _read_config(dataset_loc, model_name):
    """Mapping of the region dataset YAML of ``model_name``."""
    path = Path(dataset_loc) / f"{model_name}.yaml"
    assert path.is_file(), f"{path.name} was not written"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _crops_prefix(version, region):
    """Start of every crop file path of a region dataset YAML of ``region`` cropped under ``version``."""
    return f"${{dataset_folder}}/{_DATASET}/derivatives/{DERIVATIVES_FOLDER}/crops/{version}/2mm/{region}/mask/"


def _wrong_paths(config, version, region):
    """The crop file path keys of ``config`` whose value does not lie under the ``version`` crops of ``region``."""
    prefix = _crops_prefix(version, region)
    return {key: config.get(key) for key in CONFIG_PATH_KEYS if not str(config.get(key)).startswith(prefix)}


class TestCropPathsFollowTheMap:
    """REQ-MODELVER-BDRABCZUK-1143DFC67130: YAML crop paths lie under crops/<mapped version>/2mm."""

    def test_each_region_names_the_crops_of_its_mapped_version(self, temp_dir):
        for version in (DEFAULT_MASKS_VERSION, _OTHER_VERSION):
            _lay_crop(temp_dir, version, _SYLV)
            _lay_crop(temp_dir, version, _STS)
        versions = {_SYLV: _OTHER_VERSION, _STS: DEFAULT_MASKS_VERSION}
        region_masks = _full_map(versions)

        dataset_loc = _run_stage(temp_dir, DEFAULT_MASKS_VERSION, region_masks)

        for region, version in versions.items():
            for side in "LR":
                config = _read_config(dataset_loc, _model_name(region, side))
                assert _wrong_paths(config, version, region) == {}

    def test_each_hemisphere_names_the_crops_of_its_mapped_version(self, temp_dir):
        for version in (DEFAULT_MASKS_VERSION, _OTHER_VERSION):
            _lay_crop(temp_dir, version, _SYLV)
        sides = {"L": _OTHER_VERSION, "R": DEFAULT_MASKS_VERSION}
        region_masks = {_model_name(_SYLV, side): version for side, version in sides.items()}

        dataset_loc = _run_stage(temp_dir, DEFAULT_MASKS_VERSION, region_masks)

        for side, version in sides.items():
            config = _read_config(dataset_loc, _model_name(_SYLV, side))
            assert _wrong_paths(config, version, _SYLV) == {}


class TestCropSizeFollowsTheMap:
    """REQ-MODELVER-BDRABCZUK-9C55E6CF1CCA: input_size comes from the crop arrays in the mapped version folder."""

    def test_input_size_comes_from_the_mapped_version_folder(self, temp_dir):
        _lay_crop(temp_dir, _OTHER_VERSION, _SYLV, shape=(1, 4, 5, 6))
        _lay_crop(temp_dir, DEFAULT_MASKS_VERSION, _SYLV, shape=(1, 7, 8, 9))
        region_masks = {_model_name(_SYLV, side): _OTHER_VERSION for side in "LR"}

        dataset_loc = _run_stage(temp_dir, DEFAULT_MASKS_VERSION, region_masks)

        for side in "LR":
            assert _read_config(dataset_loc, _model_name(_SYLV, side))["input_size"] == "(1, 4, 5, 6)"


class TestRegionsOutsideCropPath:
    """REQ-MODELVER-BDRABCZUK-6BBA8C990F11: a region cropped only in its mapped version folder gets its YAML."""

    def test_region_cropped_only_in_its_mapped_version_gets_a_yaml(self, temp_dir):
        _lay_crop(temp_dir, _OTHER_VERSION, _SYLV)
        _lay_crop(temp_dir, DEFAULT_MASKS_VERSION, _STS)
        versions = {_SYLV: _OTHER_VERSION, _STS: DEFAULT_MASKS_VERSION}
        region_masks = _full_map(versions)

        dataset_loc = _run_stage(temp_dir, DEFAULT_MASKS_VERSION, region_masks)

        written = {path.stem for path in Path(dataset_loc).glob("*.yaml")}
        assert {_model_name(_SYLV, side) for side in "LR"} <= written


class TestWithoutAMap:
    """REQ-MODELVER-BDRABCZUK-5D77FEE3D0E3: with no map, region YAMLs name the --masks crops folder."""

    @pytest.mark.parametrize("masks", [DEFAULT_MASKS_VERSION, _OTHER_VERSION])
    def test_region_yamls_name_the_masks_version_crops(self, temp_dir, masks):
        for region in (_SYLV, _STS):
            _lay_crop(temp_dir, masks, region)

        dataset_loc = _run_stage(temp_dir, masks, masks=masks)

        for region in (_SYLV, _STS):
            for side in "LR":
                config = _read_config(dataset_loc, _model_name(region, side))
                assert _wrong_paths(config, masks, region) == {}


class TestOptionRefusals:
    """--masks and --external_crops beside the map exit 2 (user decisions Q1 and Q4, 2026-11-23)."""

    def test_masks_beside_the_map_exits_2_naming_masks(self, temp_dir, capsys):
        """REQ-MODELVER-BDRABCZUK-70531C5E1CF0: --masks together with --region-masks exits 2, naming --masks."""
        _lay_crop(temp_dir, DEFAULT_MASKS_VERSION, _SYLV)
        argv = _build_argv(temp_dir, DEFAULT_MASKS_VERSION, _full_map({_SYLV: _OTHER_VERSION}), masks=_OTHER_VERSION)

        assert "--masks" in _refusal(capsys, argv)

    def test_external_crops_beside_the_map_exits_2_naming_both(self, temp_dir, capsys):
        """REQ-MODELVER-BDRABCZUK-021478784940: --external_crops beside --region-masks exits 2, naming both."""
        _lay_crop(temp_dir, DEFAULT_MASKS_VERSION, _SYLV)
        region_masks = _full_map({_SYLV: DEFAULT_MASKS_VERSION})
        argv = _build_argv(temp_dir, DEFAULT_MASKS_VERSION, region_masks, extra=["--external_crops"])

        message = _refusal(capsys, argv)

        assert "--external_crops" in message
        assert _MAP_OPTION in message


class TestUnmappedRegion:
    """REQ-MODELVER-BDRABCZUK-203810F56823: a crop_path region hemisphere the map does not name exits 2, naming it (Q2).

    The no-write half runs with the real QC in test_generate_champollion_config_region_versions_qc.py.
    """

    def test_unmapped_region_model_name_exits_2_naming_it(self, temp_dir, capsys):
        assert _model_name(_STS, "R") in _refusal(capsys, build_unmapped_argv(temp_dir))
