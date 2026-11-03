"""Dotted subject-ID guard of run_cortical_tiles.py (TASK-CORTILES-BDRABCZUK-5B27B8A23300).

cortical_tiles' quality_checks splits a subject ID on its first dot, so without --bids
A.1 and A.2 merge into one subject. BIDS IDs are alphanumeric and are not checked.

REQ-CORTILES-BDRABCZUK-7AB50623DA6C: dotted ID without --bids: non-zero code,
    generate_sulcal_regions.py not run.
REQ-CORTILES-BDRABCZUK-CA042778DB6F: refusal logs one ERROR record naming each dotted ID.
"""

import logging

import pytest

from tests.test_run_cortical_tiles_prefix_ids import _messages, _ran_cortical_tiles, _run

pytestmark = pytest.mark.unit  # noqa: V107


class TestDottedIdsRefused:
    def test_refuses_without_running_cortical_tiles(self, temp_dir):
        """REQ-CORTILES-BDRABCZUK-7AB50623DA6C."""
        result, execute = _run(temp_dir, ["A.1", "A.2", "B"])

        assert result != 0
        assert not _ran_cortical_tiles(execute)

    def test_bids_run_is_unaffected(self, temp_dir):
        """REQ-CORTILES-BDRABCZUK-7AB50623DA6C: with --bids the dotted name is not checked."""
        result, execute = _run(temp_dir, ["A.1", "B"], "--bids")

        assert result == 0
        assert _ran_cortical_tiles(execute)

    def test_undotted_ids_run(self, temp_dir):
        """REQ-CORTILES-BDRABCZUK-7AB50623DA6C: files and dot directories do not count."""
        result, execute = _run(temp_dir, ["A", "B", ".cache.d"], files=["C.1", "x.minf"])

        assert result == 0
        assert _ran_cortical_tiles(execute)

    def test_error_names_each_dotted_id(self, temp_dir, pipeline_caplog):
        """REQ-CORTILES-BDRABCZUK-CA042778DB6F."""
        _run(temp_dir, ["A.1", "A.2", "B"])

        errors = _messages(pipeline_caplog, logging.ERROR, "A.1")
        assert len(errors) == 1, errors
        assert "A.2" in errors[0]
        assert "B" not in errors[0].replace("--bids", "")
