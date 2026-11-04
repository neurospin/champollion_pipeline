#!/usr/bin/env python3
"""
Argument parsing, --njobs and path validation tests of run_cortical_tiles.py
(moved out of test_run_cortical_tiles.py).
"""

import logging
import re
from unittest.mock import patch

import pytest

from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from tests.stage2_layout import parsed_script, run_mocked, stage_dirs
from tests.test_run_cortical_tiles import (  # noqa: F401
    _LOGGER_NAME,
    _stub_whole_brain_functions,  # noqa: V103 - autouse fixture, as in the source module
)


def _script_on_graphs(temp_dir, *extra_args):
    """A parsed RunCorticalTiles on an input holding one subject with both graphs."""
    return parsed_script(*stage_dirs(temp_dir), *extra_args)


def _njobs_after_run(temp_dir):
    """The --njobs value run() settles on when none is given."""
    script = _script_on_graphs(temp_dir)
    run_mocked(script)
    return script.args.njobs


class TestRunCorticalTilesArguments:
    """Test argument parsing."""

    def test_parse_required_arguments(self):
        """Test parsing required arguments."""
        script = RunCorticalTiles()
        args = script.parse_args(
            ["/input", "/output", "--path_to_graph", "graphs/path", "--path_sk_with_hull", "skeleton/path"]
        )
        assert args.input == "/input"
        assert args.output == "/output"
        assert args.path_to_graph == "graphs/path"
        assert args.path_sk_with_hull == "skeleton/path"

    def test_optional_region_file(self):
        """Test optional region-file argument."""
        script = RunCorticalTiles()
        args = script.parse_args(
            [
                "/input",
                "/output",
                "--path_to_graph",
                "graphs/path",
                "--path_sk_with_hull",
                "skeleton/path",
                "--region-file",
                "/path/to/regions.json",
            ]
        )
        assert args.region_file == "/path/to/regions.json"

    def test_optional_sk_qc_path_default(self):
        """Test that sk_qc_path defaults to empty string."""
        script = RunCorticalTiles()
        args = script.parse_args(
            ["/input", "/output", "--path_to_graph", "graphs/path", "--path_sk_with_hull", "skeleton/path"]
        )
        assert args.sk_qc_path == ""

    def test_optional_njobs_default(self):
        """Test that njobs defaults to None."""
        script = RunCorticalTiles()
        args = script.parse_args(
            ["/input", "/output", "--path_to_graph", "graphs/path", "--path_sk_with_hull", "skeleton/path"]
        )
        assert args.njobs is None

    def test_njobs_custom_value(self):
        """Test setting custom njobs value."""
        script = RunCorticalTiles()
        args = script.parse_args(
            [
                "/input",
                "/output",
                "--path_to_graph",
                "graphs/path",
                "--path_sk_with_hull",
                "skeleton/path",
                "--njobs",
                "10",
            ]
        )
        assert args.njobs == 10

    def test_missing_required_arguments(self):
        """Test that missing required arguments raise error."""
        script = RunCorticalTiles()
        with pytest.raises(SystemExit):
            script.parse_args(["/input", "/output"])  # Missing path_to_graph and path_sk_with_hull

    def test_input_types_default_none(self):
        """Test that input_types defaults to None."""
        script = RunCorticalTiles()
        args = script.parse_args(["/input", "/output", "--path_to_graph", "graphs", "--path_sk_with_hull", "skeleton"])
        assert args.input_types is None

    def test_input_types_can_be_set(self):
        """Test that --input-types accepts multiple values."""
        script = RunCorticalTiles()
        args = script.parse_args(
            [
                "/input",
                "/output",
                "--path_to_graph",
                "graphs",
                "--path_sk_with_hull",
                "skeleton",
                "--input-types",
                "skeleton",
                "foldlabel",
                "extremities",
            ]
        )
        assert args.input_types == ["skeleton", "foldlabel", "extremities"]

    def test_skip_distbottom_default_false(self):
        """Test that --skip-distbottom defaults to False."""
        args = parsed_script("/input", "/output").args
        assert args.skip_distbottom is False

    def test_skip_distbottom_can_be_set(self):
        """Test that --skip-distbottom can be set to True."""
        args = parsed_script("/input", "/output", "--skip-distbottom").args
        assert args.skip_distbottom is True

    def test_bids_default_false(self):
        """--bids is an opt-in flag on RunCorticalTiles's parser, defaulting to False (REQ-BIDS-01/-02)."""
        args = parsed_script("/input", "/output").args
        assert args.bids is False

    def test_bids_can_be_set(self):
        """--bids is recognized by RunCorticalTiles's parser and sets args.bids to True (REQ-BIDS-01/-02).

        main.py's RunCorticalTilesStage already appends --bids when
        config.dataset.bids is set; without this flag the receiving parser
        rejects it.
        """
        args = parsed_script("/input", "/output", "--bids").args
        assert args.bids is True


class TestNjobsHandling:
    """Test njobs calculation and validation."""

    @patch("champollion_pipeline.run_cortical_tiles.cpu_count", return_value=24)
    def test_njobs_none_uses_default_calculation(self, _mock_cpu, temp_dir):
        """Test that njobs=None calculates min(22, cpu_count-2)."""
        # njobs should be min(22, 24-2) = 22
        assert _njobs_after_run(temp_dir) == 22

    @patch("champollion_pipeline.run_cortical_tiles.cpu_count", return_value=8)
    def test_njobs_none_with_low_cpu_count(self, _mock_cpu, temp_dir):
        """Test njobs calculation with low CPU count."""
        # njobs should be min(22, 8-2) = 6
        assert _njobs_after_run(temp_dir) == 6

    @patch("champollion_pipeline.run_cortical_tiles.cpu_count", return_value=8)
    def test_njobs_exceeds_cpu_count_logs_warning(self, _mock_cpu, temp_dir, pipeline_caplog):
        """REQ-LOGGING-BDRABCZUK-AA4776030940: njobs >= cpu_count is logged at WARNING with both counts."""
        script = _script_on_graphs(temp_dir, "--njobs", "10")  # More than cpu_count

        run_mocked(script)

        record = pipeline_caplog.assert_logged(logging.WARNING, "10")
        assert record.name == _LOGGER_NAME
        assert re.search(r"\b10\b.*\b8\b", record.getMessage()), record.getMessage()

    @patch("champollion_pipeline.run_cortical_tiles.cpu_count", return_value=1)
    def test_njobs_minimum_one_with_cpu_count_one(self, _mock_cpu, temp_dir):
        """njobs is at least 1 even on a single-core machine (cpu_count=1)."""
        # max(1, min(22, 1 - 2)) = max(1, -1) = 1
        assert _njobs_after_run(temp_dir) == 1

    @patch("champollion_pipeline.run_cortical_tiles.cpu_count", return_value=2)
    def test_njobs_minimum_one_with_cpu_count_two(self, _mock_cpu, temp_dir):
        """njobs is at least 1 on a dual-core machine (cpu_count=2)."""
        # max(1, min(22, 2 - 2)) = max(1, 0) = 1
        assert _njobs_after_run(temp_dir) == 1


class TestValidatePaths:
    """Test path validation."""

    def test_validate_paths_success(self, temp_dir):
        """Test successful path validation."""
        input_dir, output_dir = stage_dirs(temp_dir)
        script = parsed_script(input_dir, output_dir)

        assert script.validate_paths([str(input_dir), str(output_dir)]) is True

    def test_validate_paths_failure_raises_error(self):
        """Test that invalid paths raise ValueError."""
        script = parsed_script("/nonexistent/input", "/nonexistent/output")

        with (
            patch.object(script, "validate_paths", return_value=False),
            pytest.raises(ValueError, match="Please input valid paths"),
        ):
            script.run()
