#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
--bucket_step validation of the src/compare.py masks and cortical_tiles subcommands.

Requirements: REQ-COMPARE-76 (a --bucket_step value that is not a finite
number greater than zero is an argparse usage error, exit 2, naming
--bucket_step and containing the word "positive"), REQ-COMPARE-77 (each
finite positive value is still parsed unchanged).

Only argument parsing is exercised; PyAIMS is stubbed in conftest.py.
"""

import pytest

from compare import Compare

_MODES = ["masks", "cortical_tiles"]
_INVALID_STEPS = ["0", "-1", "-0.5", "abc", "nan", "inf"]
_VALID_STEPS = [("1", 1.0), ("0.5", 0.5), ("2.5", 2.5), ("0.001", 0.001)]


def _parse_mask_mode(mode, *extra):
    return Compare().parse_args([mode, "--set_a", "set_a", "--set_b", "set_b", *extra])


@pytest.mark.unit
class TestBucketStepRejectsNonPositive:
    """REQ-COMPARE-76: a --bucket_step that is not a finite number > 0 exits 2 with a usage error."""

    @pytest.mark.parametrize("value", _INVALID_STEPS)
    @pytest.mark.parametrize("mode", _MODES)
    def test_invalid_bucket_step_exits_with_status_2(self, mode, value):
        with pytest.raises(SystemExit) as excinfo:
            _parse_mask_mode(mode, f"--bucket_step={value}")

        assert excinfo.value.code == 2

    @pytest.mark.parametrize("value", _INVALID_STEPS)
    @pytest.mark.parametrize("mode", _MODES)
    def test_invalid_bucket_step_error_names_option_and_positive(self, mode, value, capsys):
        with pytest.raises(SystemExit):
            _parse_mask_mode(mode, f"--bucket_step={value}")

        err = capsys.readouterr().err
        assert "--bucket_step" in err
        assert "positive" in err.lower()


@pytest.mark.unit
class TestBucketStepAcceptsPositive:
    """REQ-COMPARE-77: each finite --bucket_step > 0 is stored unchanged as args.bucket_step."""

    @pytest.mark.parametrize(("value", "expected"), _VALID_STEPS)
    @pytest.mark.parametrize("mode", _MODES)
    def test_positive_bucket_step_is_stored_unchanged(self, mode, value, expected):
        args = _parse_mask_mode(mode, "--bucket_step", value)

        assert args.bucket_step == expected
