"""Stage 2 fixtures of tests that take run_cortical_tiles.py through its input QC and its mocked crop step.

run() needs a subject holding one graph per hemisphere at <input>/<subject>/<path_to_graph>/<side>*.arg.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline.run_cortical_tiles import RunCorticalTiles

GRAPH_DIR = "graphs"
SUBJECT = "subj01"
TEMPLATE_CONFIG = Path(__file__).resolve().parent.parent / "pipeline_loop_2mm.json"


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_whole_brain_functions(monkeypatch):
    """Prevent the real whole-brain submodule calls from running in tests that
    don't care about them (REQ-WHOLEBRAIN-01 runs unconditionally in run()).
    TestWholeBrainGeneration patches these itself, taking precedence within
    its own `with patch(...)` blocks.
    """
    monkeypatch.setattr(
        "champollion_pipeline.run_cortical_tiles.add_left_and_right_volumes.add_left_and_right_volumes",
        MagicMock(),
    )
    monkeypatch.setattr(
        "champollion_pipeline.run_cortical_tiles.remove_ventricle.remove_ventricle",
        MagicMock(),
    )


def add_graphs(input_dir, subject=SUBJECT, graph_dir=GRAPH_DIR, sides="LR"):
    """Lay out ``subject`` under ``input_dir``: ``graph_dir`` holding a graph per hemisphere of ``sides``."""
    folder = Path(input_dir) / subject / graph_dir
    folder.mkdir(parents=True, exist_ok=True)
    for side in sides:
        (folder / f"{side}{subject}.arg").touch()


def stage_dirs(root, graph_dir=GRAPH_DIR):
    """Create <root>/input, holding one subject with both graphs in ``graph_dir``, and <root>/output."""
    input_dir, output_dir = Path(root) / "input", Path(root) / "output"
    add_graphs(input_dir, graph_dir=graph_dir)
    output_dir.mkdir()
    return input_dir, output_dir


def stage_dirs_on_template(root, config_overrides=None):
    """stage_dirs, plus <root>/output/pipeline_loop_2mm.json seeded from the shipped template; return all three."""
    input_dir, output_dir = stage_dirs(root)
    config = json.loads(TEMPLATE_CONFIG.read_text())
    config.update(config_overrides or {})
    config_path = output_dir / TEMPLATE_CONFIG.name
    config_path.write_text(json.dumps(config))
    return input_dir, output_dir, config_path


def parsed_script(input_dir, output_dir, *extra_args):
    """A RunCorticalTiles parsed on the two paths, the graph and skeleton options and ``extra_args``."""
    script = RunCorticalTiles()
    script.parse_args(
        [str(input_dir), str(output_dir), "--path_to_graph", GRAPH_DIR, "--path_sk_with_hull", "skeleton", *extra_args]
    )
    return script


def run_mocked(script):
    """Run ``script`` with its paths taken as valid and its commands mocked; return the execute_command mock."""
    with (
        patch.object(script, "validate_paths", return_value=True),
        patch.object(script, "execute_command", return_value=0) as mock_exec,
    ):
        script.run()
    return mock_exec


def run_on_config(input_dir, output_dir, config_data, *extra_args):
    """run_mocked on a pipeline_loop_2mm.json holding ``config_data``; return that file's content after the run."""
    config_path = Path(output_dir) / TEMPLATE_CONFIG.name
    config_path.write_text(json.dumps(config_data))
    run_mocked(parsed_script(input_dir, output_dir, *extra_args))
    return json.loads(config_path.read_text())
