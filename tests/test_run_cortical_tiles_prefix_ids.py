"""Prefix subject IDs reach stage 2 (TASK-CORTILES-BDRABCZUK-96603E82E1F7).

cortical_tiles 7b80c89 matches subject folders exactly (tests/test_cortical_tiles_exact_subject_folders.py), so the
temporary prefix guard of TASK-CORTILES-BDRABCZUK-E9CFBC50E516 is gone (DEF-CORTILES-BDRABCZUK-8A44DAD7F312).
--allow-prefix-ids stays accepted, for config replay, as a no-op that logs a deprecation warning.

REQ-CORTILES-BDRABCZUK-CBC851B76315: prefix pair without --allow-prefix-ids: generate_sulcal_regions.py runs.
REQ-CORTILES-BDRABCZUK-ABF279058F0A: prefix pair: no ERROR record naming the pair.
REQ-CORTILES-BDRABCZUK-AD5F7A32399E: --allow-prefix-ids: one deprecation WARNING record.
REQ-INPUTKIND-BDRABCZUK-DED80F6406DA: prefix pair with --allow-prefix-ids: generate_sulcal_regions.py runs.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from tests.stage2_layout import add_graphs

pytestmark = pytest.mark.unit  # noqa: V107

LOGGER_NAME = "champollion_pipeline.run_cortical_tiles"
ALLOW_FLAG = "--allow-prefix-ids"
CORTICAL_TILES_SCRIPT = "generate_sulcal_regions.py"


def _layout(input_dir: Path, folders, files=()) -> None:
    for name in folders:
        add_graphs(input_dir, name)
    for name in files:
        (input_dir / name).write_text("x")


def _run(temp_dir: str, folders, *extra: str, files=()):
    """Run RunCorticalTiles.run on an input directory holding ``folders`` (and ``files``),
    with every external step mocked. Returns run()'s code and the execute_command mock."""
    input_dir = Path(temp_dir) / "input"
    output_dir = Path(temp_dir) / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    _layout(input_dir, folders, files)

    script = RunCorticalTiles()
    script.parse_args(
        [str(input_dir), str(output_dir), "--path_to_graph", "graphs", "--path_sk_with_hull", "skeleton", *extra]
    )
    with (
        patch.object(script, "execute_command", return_value=0) as execute,
        patch.object(script, "_generate_mask_npys"),
        patch(
            "champollion_pipeline.run_cortical_tiles.add_left_and_right_volumes.add_left_and_right_volumes",
            MagicMock(),
        ),
        patch("champollion_pipeline.run_cortical_tiles.remove_ventricle.remove_ventricle", MagicMock()),
    ):
        result = script.run()
    return result, execute


def _ran_cortical_tiles(execute: MagicMock) -> bool:
    """True when one execute_command call launched generate_sulcal_regions.py."""
    for call in execute.call_args_list:
        command = call.args[0] if call.args else call.kwargs.get("cmd", [])
        if any(str(part).endswith(CORTICAL_TILES_SCRIPT) for part in command):
            return True
    return False


def _messages(pipeline_caplog, level: int, text: str) -> list[str]:
    return [
        record.getMessage()
        for record in pipeline_caplog.records
        if record.name == LOGGER_NAME and record.levelno == level and text in record.getMessage()
    ]


class TestPrefixIdsAccepted:
    """A prefix pair no longer stops the run."""

    @pytest.mark.parametrize("extra", [[], ["--bids"]], ids=["plain", "bids"])
    def test_runs_cortical_tiles(self, temp_dir, extra):
        """REQ-CORTILES-BDRABCZUK-CBC851B76315."""
        result, execute = _run(temp_dir, ["sub-1", "sub-10", "sub-2"], *extra)

        assert result == 0
        assert _ran_cortical_tiles(execute)

    def test_logs_no_error_for_the_pairs(self, temp_dir, pipeline_caplog):
        """REQ-CORTILES-BDRABCZUK-ABF279058F0A: sub-1, sub-10, sub-100 make pairs and no error."""
        _run(temp_dir, ["sub-1", "sub-10", "sub-100"])

        assert not _messages(pipeline_caplog, logging.ERROR, "sub-10")

    def test_files_and_dot_directories_do_not_matter(self, temp_dir):
        """REQ-CORTILES-BDRABCZUK-CBC851B76315: only non-dot subdirectories are subjects; the run goes on."""
        result, execute = _run(
            temp_dir,
            ["sub-1", "sub-2", ".cache", ".cache2"],
            files=["sub-10", "sub-1.minf", "database.sqlite"],
        )

        assert result == 0
        assert _ran_cortical_tiles(execute)


class TestAllowFlagDeprecated:
    """--allow-prefix-ids is accepted and does nothing but warn."""

    @pytest.mark.parametrize("extra", [[], ["--bids"]], ids=["plain", "bids"])
    def test_runs_cortical_tiles(self, temp_dir, extra):
        """REQ-INPUTKIND-BDRABCZUK-DED80F6406DA."""
        result, execute = _run(temp_dir, ["sub-1", "sub-10", "sub-2"], ALLOW_FLAG, *extra)

        assert result == 0
        assert _ran_cortical_tiles(execute)

    @pytest.mark.parametrize("folders", [["sub-1", "sub-10"], ["sub-1", "sub-2"]], ids=["pair", "no_pair"])
    def test_warns_once_that_the_flag_is_deprecated(self, temp_dir, pipeline_caplog, folders):
        """REQ-CORTILES-BDRABCZUK-AD5F7A32399E: one WARNING, with or without a prefix pair."""
        _run(temp_dir, folders, ALLOW_FLAG)

        warnings_ = _messages(pipeline_caplog, logging.WARNING, ALLOW_FLAG)
        assert len(warnings_) == 1, warnings_
        assert "deprecated" in warnings_[0] and "no effect" in warnings_[0], warnings_[0]

    def test_no_warning_without_the_flag(self, temp_dir, pipeline_caplog):
        """REQ-CORTILES-BDRABCZUK-AD5F7A32399E: the warning belongs to the flag."""
        _run(temp_dir, ["sub-1", "sub-10"])

        assert not _messages(pipeline_caplog, logging.WARNING, ALLOW_FLAG)
