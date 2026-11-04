"""Root-path substring guard of run_cortical_tiles.py (TASK-CORTILES-BDRABCZUK-75B35A29E44A).

cortical_tiles applies unanchored BIDS regexes (ses-, acq-, run-) to the full path and
splits on ``_cropped``, so an input or output root containing one of these substrings
corrupts subject/session parsing (DEF-CORTILES-BDRABCZUK-9C7D2B9265BC).
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from tests.stage2_layout import add_graphs

pytestmark = pytest.mark.unit  # noqa: V107

LOGGER_NAME = "champollion_pipeline.run_cortical_tiles"
CORTICAL_TILES_SCRIPT = "generate_sulcal_regions.py"
SUBSTRINGS = ["ses-", "acq-", "run-", "_cropped"]


def _run(base: Path, input_name: str = "input", output_name: str = "output"):
    input_dir = base / input_name
    output_dir = base / output_name
    input_dir.mkdir()
    output_dir.mkdir()
    add_graphs(input_dir, "sub-a")

    script = RunCorticalTiles()
    script.parse_args([str(input_dir), str(output_dir), "--path_to_graph", "graphs", "--path_sk_with_hull", "skeleton"])
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
    return result, execute, input_dir, output_dir


def _ran_cortical_tiles(execute) -> bool:
    return any(CORTICAL_TILES_SCRIPT in " ".join(map(str, call.args[0])) for call in execute.call_args_list)


class TestRootSubstringRefused:
    @pytest.mark.parametrize("substring", SUBSTRINGS)
    @pytest.mark.parametrize("which", ["input", "output"])
    def test_refuses_without_running_cortical_tiles(self, tmp_path, substring, which):
        names = {"input": "input", "output": "output"}
        names[which] = f"x{substring}y"
        result, execute, _, _ = _run(tmp_path, names["input"], names["output"])
        assert result != 0
        assert not _ran_cortical_tiles(execute)

    @pytest.mark.parametrize("substring", SUBSTRINGS)
    @pytest.mark.parametrize("which", ["input", "output"])
    def test_error_names_substring_and_root(self, tmp_path, caplog, substring, which):
        names = {"input": "input", "output": "output"}
        names[which] = f"x{substring}y"
        with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
            _, _, input_dir, output_dir = _run(tmp_path, names["input"], names["output"])
        root = input_dir if which == "input" else output_dir
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR and r.name == LOGGER_NAME]
        assert any(f"'{substring}'" in m and str(root) in m for m in errors)

    def test_clean_roots_are_not_refused(self, tmp_path):
        result, execute, _, _ = _run(tmp_path)
        assert result in (0, None)
        assert _ran_cortical_tiles(execute)
