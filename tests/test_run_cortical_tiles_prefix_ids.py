"""Prefix subject-ID guard of run_cortical_tiles.py (TASK-CORTILES-BDRABCZUK-E9CFBC50E516).

cortical_tiles' generate_ICBM2009c_transforms.py globs ``{subject}*/`` and keeps the
first match, so a subject whose folder name prefixes another one (sub-1 / sub-10) can
receive the other subject's ICBM2009c transform (DEF-CORTILES-BDRABCZUK-8A44DAD7F312).
cortical_tiles lists the entries of the input directory as subjects, with or without
--bids; the glob only matches directories, and never a name starting with a dot.

REQ-CORTILES-BDRABCZUK-6C72CEECC13B: prefix pair without --allow-prefix-ids: non-zero
    code, generate_sulcal_regions.py not run.
REQ-CORTILES-BDRABCZUK-3BC325A938C2: prefix pair without --allow-prefix-ids: ERROR record
    with the pair count and the first five pairs in lexicographic order.
REQ-CORTILES-BDRABCZUK-3F455F6B133E: prefix pair with --allow-prefix-ids: WARNING record
    with the pair count and the first five pairs in lexicographic order.
REQ-CORTILES-BDRABCZUK-DE1299433603: prefix pair with --allow-prefix-ids:
    generate_sulcal_regions.py runs.
"""

import logging
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline.run_cortical_tiles import RunCorticalTiles

pytestmark = pytest.mark.unit  # noqa: V107

LOGGER_NAME = "champollion_pipeline.run_cortical_tiles"
ALLOW_FLAG = "--allow-prefix-ids"
CORTICAL_TILES_SCRIPT = "generate_sulcal_regions.py"

# Seven disjoint prefix pairs: (sub-a, sub-a0) ... (sub-g, sub-g0), in lexicographic order.
SEVEN_PAIRS = [(f"sub-{letter}", f"sub-{letter}0") for letter in "abcdefg"]


def _layout(input_dir: Path, folders, files=()) -> None:
    for name in folders:
        (input_dir / name).mkdir()
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


def _assert_count_and_first_five(message: str) -> None:
    assert re.search(r"\b7\b", message), message
    for shorter, longer in SEVEN_PAIRS[:5]:
        assert shorter in message and longer in message, (shorter, longer, message)
    for _, longer in SEVEN_PAIRS[5:]:
        assert longer not in message, (longer, message)


class TestPrefixIdsRefused:
    """Without --allow-prefix-ids, a prefix pair stops the run (NTUI: check the input first)."""

    @pytest.mark.parametrize("extra", [[], ["--bids"]], ids=["plain", "bids"])
    def test_refuses_without_running_cortical_tiles(self, temp_dir, extra):
        """REQ-CORTILES-BDRABCZUK-6C72CEECC13B."""
        result, execute = _run(temp_dir, ["sub-1", "sub-10", "sub-2"], *extra)

        assert result != 0
        assert not _ran_cortical_tiles(execute)

    def test_files_and_dot_directories_do_not_count(self, temp_dir, pipeline_caplog):
        """REQ-CORTILES-BDRABCZUK-6C72CEECC13B: only non-dot subdirectory names are compared."""
        result, execute = _run(
            temp_dir,
            ["sub-1", "sub-2", ".cache", ".cache2"],
            files=["sub-10", "sub-1.minf", "database.sqlite"],
        )

        assert result == 0
        assert _ran_cortical_tiles(execute)
        assert not _messages(pipeline_caplog, logging.ERROR, "sub-1")

    def test_error_names_count_and_first_five_pairs(self, temp_dir, pipeline_caplog):
        """REQ-CORTILES-BDRABCZUK-3BC325A938C2."""
        folders = [name for pair in SEVEN_PAIRS for name in pair]

        _run(temp_dir, folders)

        errors = _messages(pipeline_caplog, logging.ERROR, SEVEN_PAIRS[0][1])
        assert len(errors) == 1, errors
        _assert_count_and_first_five(errors[0])

    def test_error_counts_each_prefix_pair(self, temp_dir, pipeline_caplog):
        """REQ-CORTILES-BDRABCZUK-3BC325A938C2: sub-1, sub-10, sub-100 make three pairs."""
        _run(temp_dir, ["sub-1", "sub-10", "sub-100"])

        errors = _messages(pipeline_caplog, logging.ERROR, "sub-10")
        assert len(errors) == 1, errors
        assert re.search(r"\b3\b", errors[0]), errors[0]


class TestPrefixIdsAllowed:
    """With --allow-prefix-ids, a prefix pair is a warning and the run goes on."""

    @pytest.mark.parametrize("extra", [[], ["--bids"]], ids=["plain", "bids"])
    def test_runs_cortical_tiles(self, temp_dir, extra):
        """REQ-CORTILES-BDRABCZUK-DE1299433603."""
        result, execute = _run(temp_dir, ["sub-1", "sub-10", "sub-2"], ALLOW_FLAG, *extra)

        assert result == 0
        assert _ran_cortical_tiles(execute)

    def test_warning_names_count_and_first_five_pairs(self, temp_dir, pipeline_caplog):
        """REQ-CORTILES-BDRABCZUK-3F455F6B133E."""
        folders = [name for pair in SEVEN_PAIRS for name in pair]

        _run(temp_dir, folders, ALLOW_FLAG)

        warnings_ = _messages(pipeline_caplog, logging.WARNING, SEVEN_PAIRS[0][1])
        assert len(warnings_) == 1, warnings_
        _assert_count_and_first_five(warnings_[0])
        assert not _messages(pipeline_caplog, logging.ERROR, SEVEN_PAIRS[0][1])
