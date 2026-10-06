#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
--side validation of the src/compare.py databases subcommand.

Requirements: REQ-COMPARE-74 (a --side value outside L/R/both is an argparse
usage error, exit 2, naming the valid choices), REQ-COMPARE-75 (each of L, R
and both is still parsed unchanged).

Only argument parsing is exercised; PyAIMS is stubbed in conftest.py.
"""

import pytest

from compare import Compare

_VALID_SIDES = ["L", "R", "both"]


def _parse_databases(*extra):
    return Compare().parse_args(
        [
            "databases",
            "--labeled_subjects_dir",
            "subjects",
            "--path_to_graph_a",
            "graph_a",
            "--path_to_graph_b",
            "graph_b",
            *extra,
        ]
    )


@pytest.mark.unit
class TestDatabasesSideRejectsInvalid:
    """REQ-COMPARE-74: --side outside L/R/both exits 2 with a usage error naming the valid choices."""

    @pytest.mark.parametrize("value", ["left", "l", "Both", "LR"])
    def test_invalid_side_exits_with_status_2(self, value):
        with pytest.raises(SystemExit) as excinfo:
            _parse_databases("--side", value)

        assert excinfo.value.code == 2

    def test_invalid_side_error_names_valid_choices(self, capsys):
        with pytest.raises(SystemExit):
            _parse_databases("--side", "left")

        err = capsys.readouterr().err
        assert "--side" in err
        for side in _VALID_SIDES:
            assert f"'{side}'" in err


@pytest.mark.unit
class TestDatabasesSideAcceptsValid:
    """REQ-COMPARE-75: each of L, R and both is stored unchanged as args.side."""

    @pytest.mark.parametrize("value", _VALID_SIDES)
    def test_valid_side_is_stored_unchanged(self, value):
        assert _parse_databases("--side", value).side == value
