"""LOGGING convention for generate_morphologist_graphs.py (TASK-LOGGING-BDRABCZUK-940DB0BFBA89).

REQ-LOGGING-BDRABCZUK-205E7A01F3F1: run() logs the current working directory at LOG on the
champollion_pipeline.generate_morphologist_graphs logger instead of printing it.
(REQ-LOGGING-BDRABCZUK-C96C8DAAF703 and -22ABEF2F2CBD: tests/test_logging_convention.py, through
LOG_CONVERTED_MODULES.)
"""

from unittest.mock import patch

import pytest

from champollion_pipeline.generate_morphologist_graphs import GenerateMorphologistGraphs
from tests.log_assertions import LOG

pytestmark = pytest.mark.unit  # noqa: V107

LOGGER_NAME = "champollion_pipeline.generate_morphologist_graphs"
WORKING_DIRECTORY = "/launch/working/directory"


def test_working_directory_logged_at_log(tmp_path, pipeline_caplog):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    script = GenerateMorphologistGraphs()
    script.parse_args([str(input_dir), str(output_dir)])

    with patch("champollion_pipeline.generate_morphologist_graphs.getcwd", return_value=WORKING_DIRECTORY):
        assert script.run() == 1

    record = pipeline_caplog.assert_logged(LOG, WORKING_DIRECTORY)
    assert record.name == LOGGER_NAME
