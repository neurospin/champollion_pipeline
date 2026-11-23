#!/usr/bin/env python3
"""
The shared --region-masks option of utils/region_masks.py (TASK-MODELVER-BDRABCZUK-037E4DCBA29C), through a minimal
ScriptBuilder stage that declares --masks (default None, as stages 2 and 4) and adds the option. The stage Tasks
test the option through stages 2, 3 and 4 themselves.
"""

import pytest
from champollion_utils.script_builder import ScriptBuilder
from hypothesis import given, settings
from hypothesis import strategies as st

from champollion_pipeline import derivatives_layout
from champollion_pipeline.derivatives_layout import DEFAULT_REGIONS, compute_region_model_names
from champollion_pipeline.models import version_options
from champollion_pipeline.utils.region_masks import (
    REGION_MASKS_OPTION,
    add_region_masks_option,
    format_region_masks,
    refuse_masks_with_region_masks,
)

_KNOWN_VERSIONS = ("canonical_25", "canonical_corrected_26_1")
_REGION_MODEL_NAMES = tuple(compute_region_model_names(DEFAULT_REGIONS))
_LEFT_SYLVIAN = "SC-sylv_left"
_RIGHT_STS = "STs_right"


class _Stage(ScriptBuilder):
    """A pipeline stage reduced to --masks and --region-masks."""

    def __init__(self):
        super().__init__("stage", "Stage taking --region-masks.")
        self.add_optional_argument("--masks", "Mask version of the whole run.", default=None)
        add_region_masks_option(self)

    def run(self) -> int:
        return 0


def _parse(*argv):
    """Parsed options of the stage."""
    return _Stage().parse_args(list(argv))


def _parse_error(capsys, *argv):
    """stderr of a parse that must exit with status 2."""
    with pytest.raises(SystemExit) as exit_info:
        _parse(*argv)
    assert exit_info.value.code == 2
    return capsys.readouterr().err


# REQ-MODELVER-BDRABCZUK-AE780ED9A9ED
def test_valid_pairs_give_a_map_in_input_order():
    args = _parse(REGION_MASKS_OPTION, f"{_RIGHT_STS}=canonical_25", f"{_LEFT_SYLVIAN}=canonical_corrected_26_1")

    assert list(args.region_mask_versions.items()) == [
        (_RIGHT_STS, "canonical_25"),
        (_LEFT_SYLVIAN, "canonical_corrected_26_1"),
    ]


def test_no_option_gives_no_map():
    assert _parse().region_mask_versions is None


# REQ-MODELVER-BDRABCZUK-616DC9777771
def test_value_without_equals_sign_exits_2_naming_it(capsys):
    assert "SC-sylv_left:canonical_25" in _parse_error(capsys, REGION_MASKS_OPTION, "SC-sylv_left:canonical_25")


# REQ-MODELVER-BDRABCZUK-52CDD37E1131
@pytest.mark.parametrize("pair", [f"{_LEFT_SYLVIAN}=", "=canonical_25"])
def test_pair_with_an_empty_side_exits_2_naming_it(capsys, pair):
    assert pair in _parse_error(capsys, REGION_MASKS_OPTION, pair)


# REQ-MODELVER-BDRABCZUK-00501C84AB3A
def test_unknown_region_model_name_exits_2_naming_it(capsys):
    error = _parse_error(capsys, REGION_MASKS_OPTION, f"{_RIGHT_STS}=canonical_25", "SC-sylv_middle=canonical_25")

    assert "SC-sylv_middle" in error


# REQ-MODELVER-BDRABCZUK-EE86FC536D00
def test_repeated_region_model_name_exits_2_naming_it(capsys):
    error = _parse_error(
        capsys, REGION_MASKS_OPTION, f"{_LEFT_SYLVIAN}=canonical_25", f"{_LEFT_SYLVIAN}=canonical_corrected_26_1"
    )

    assert _LEFT_SYLVIAN in error


# REQ-MODELVER-BDRABCZUK-3946A8EFF65F
def test_unknown_mask_version_exits_2_naming_it(capsys):
    assert "canonical_99" in _parse_error(capsys, REGION_MASKS_OPTION, f"{_LEFT_SYLVIAN}=canonical_99")


# REQ-MODELVER-BDRABCZUK-60B9708697EF
def test_masks_beside_region_masks_exits_2_naming_masks(capsys):
    stage = _Stage()
    args = stage.parse_args(["--masks", "canonical_25", REGION_MASKS_OPTION, f"{_LEFT_SYLVIAN}=canonical_25"])

    with pytest.raises(SystemExit) as exit_info:
        refuse_masks_with_region_masks(stage.parser, args)

    assert exit_info.value.code == 2
    assert "--masks" in capsys.readouterr().err


def test_masks_alone_is_accepted():
    stage = _Stage()
    args = stage.parse_args(["--masks", "canonical_25"])

    assert refuse_masks_with_region_masks(stage.parser, args) is None


def test_region_masks_alone_is_accepted():
    stage = _Stage()
    args = stage.parse_args([REGION_MASKS_OPTION, f"{_LEFT_SYLVIAN}=canonical_25"])

    assert refuse_masks_with_region_masks(stage.parser, args) is None


# REQ-MODELVER-BDRABCZUK-09D8FF636467: invariant parse(format(m)) == m
@settings(deadline=None)
@given(st.dictionaries(st.sampled_from(_REGION_MODEL_NAMES), st.sampled_from(_KNOWN_VERSIONS), min_size=1))
def test_format_then_parse_gives_the_map_back(versions):
    assert _parse(REGION_MASKS_OPTION, *format_region_masks(versions)).region_mask_versions == versions


def test_format_gives_one_pair_per_entry():
    pairs = format_region_masks({_RIGHT_STS: "canonical_25", _LEFT_SYLVIAN: "canonical_corrected_26_1"})

    assert sorted(pairs) == [f"{_LEFT_SYLVIAN}=canonical_corrected_26_1", f"{_RIGHT_STS}=canonical_25"]


# REQ-MODELVER-BDRABCZUK-9C1214A274C1
def test_known_mask_versions_live_in_derivatives_layout():
    assert derivatives_layout.MASK_VERSIONS == _KNOWN_VERSIONS
    assert version_options.MASK_VERSIONS is derivatives_layout.MASK_VERSIONS
