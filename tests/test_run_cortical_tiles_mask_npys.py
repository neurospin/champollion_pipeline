#!/usr/bin/env python3
"""
run() generates the mask npys of stage 2.

Moved out of test_run_cortical_tiles.py, which is above the 1000-line limit.
The other _generate_mask_npys tests stay there.
"""

from unittest.mock import MagicMock, patch

from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from tests.stage2_layout import (  # noqa: F401
    _stub_whole_brain_functions,  # noqa: V103 - autouse fixture
    stage_dirs,
)

_GRAPH_SUBPATH = "graphs/path"


class TestGenerateMaskNpys:
    """Tests for _generate_mask_npys post-processing step."""

    def test_called_after_cortical_tiles_run(self, tmp_path):
        """_generate_mask_npys is invoked at the end of run()."""
        input_dir, output_dir = stage_dirs(tmp_path, _GRAPH_SUBPATH)
        (output_dir / "pipeline_loop_2mm.json").write_text("{}")

        script = RunCorticalTiles()
        script.parse_args(
            [
                str(input_dir),
                str(output_dir),
                "--path_to_graph",
                "graphs/path",
                "--path_sk_with_hull",
                "skel/path",
            ]
        )

        with (
            patch.object(script, "execute_command", return_value=0),
            patch.object(script, "_generate_mask_npys") as mock_gen,
            patch("champollion_pipeline.run_cortical_tiles.exists", return_value=True),
            patch("champollion_pipeline.run_cortical_tiles.load_skip_entries", return_value=[]),
            patch(
                "builtins.open",
                MagicMock(
                    return_value=MagicMock(
                        __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value="{}"))),
                        __exit__=MagicMock(return_value=False),
                    )
                ),
            ),
        ):
            script.run()

        mock_gen.assert_called_once()
