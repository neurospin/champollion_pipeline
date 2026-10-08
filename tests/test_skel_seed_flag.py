#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic skeleton seed for stage 2 (TASK-SKELSEED 5/7).

The pipeline JSON template ships a fixed skeleton seed; run_cortical_tiles
exposes ``--skel-seed`` to override it and forwards it to cortical_tiles'
generate_sulcal_regions.py as ``--skel_seed``. Without the flag nothing is
forwarded, so the value in the derivatives pipeline_loop_2mm.json applies.

- REQ-SKELSEED-BDRABCZUK-AEF499A5BC4F: template sets skel_seed to the integer 42.
- REQ-SKELSEED-BDRABCZUK-A128EA98D5B5: --skel-seed N is forwarded as --skel_seed N.
- REQ-SKELSEED-BDRABCZUK-1AE6804B7DAF: without --skel-seed, no --skel_seed is forwarded.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline.run_cortical_tiles import RunCorticalTiles

pytestmark = pytest.mark.unit  # noqa: V107

TEMPLATE_CONFIG = Path(__file__).resolve().parent.parent / "pipeline_loop_2mm.json"


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_whole_brain_functions(monkeypatch):
    """Keep the unconditional whole-brain submodule calls out of these tests."""
    monkeypatch.setattr(
        "champollion_pipeline.run_cortical_tiles.add_left_and_right_volumes.add_left_and_right_volumes",
        MagicMock(),
    )
    monkeypatch.setattr(
        "champollion_pipeline.run_cortical_tiles.remove_ventricle.remove_ventricle",
        MagicMock(),
    )


def _make_dirs(tmp_path, config_overrides=None):
    """Create input/output dirs and seed output with the shipped pipeline JSON template."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    config = json.loads(TEMPLATE_CONFIG.read_text())
    config.update(config_overrides or {})
    config_path = output_dir / "pipeline_loop_2mm.json"
    config_path.write_text(json.dumps(config))
    return input_dir, output_dir, config_path


def _run(input_dir, output_dir, extra_args=()):
    """Run RunCorticalTiles.run() with the subprocess mocked.

    Returns the parsed script and the command handed to generate_sulcal_regions.py.
    """
    script = RunCorticalTiles()
    script.parse_args(
        [
            str(input_dir),
            str(output_dir),
            "--path_to_graph",
            "graphs",
            "--path_sk_with_hull",
            "skeleton",
            "--njobs",
            "1",
            *extra_args,
        ]
    )
    with patch.object(script, "validate_paths", return_value=True):
        with patch.object(script, "execute_command", return_value=0) as mock_exec:
            with patch("champollion_pipeline.run_cortical_tiles.chdir"):
                with patch("champollion_pipeline.run_cortical_tiles.getcwd", return_value="/original"):
                    script.run()
    cmd = mock_exec.call_args[0][0]
    assert any(str(token).endswith("generate_sulcal_regions.py") for token in cmd), cmd
    return script, cmd


def test_template_sets_skel_seed_42():
    """REQ-SKELSEED-BDRABCZUK-AEF499A5BC4F: the shipped template pins skel_seed = 42 (an int, not a bool/str)."""
    config = json.loads(TEMPLATE_CONFIG.read_text())
    assert "skel_seed" in config, f"skel_seed missing from {TEMPLATE_CONFIG.name}"
    assert type(config["skel_seed"]) is int
    assert config["skel_seed"] == 42


def test_skel_seed_flag_forwarded_to_generate_sulcal_regions(tmp_path):
    """REQ-SKELSEED-BDRABCZUK-A128EA98D5B5: --skel-seed 7 reaches generate_sulcal_regions.py as --skel_seed 7."""
    input_dir, output_dir, _ = _make_dirs(tmp_path)
    try:
        _, cmd = _run(input_dir, output_dir, ["--skel-seed", "7"])
    except SystemExit as exc:  # argparse rejects an unknown option
        pytest.fail(f"run_cortical_tiles does not accept --skel-seed (argparse exit {exc.code})")
    assert "--skel_seed" in cmd, f"--skel_seed not forwarded: {cmd}"
    assert cmd.count("--skel_seed") == 1, cmd
    assert cmd[cmd.index("--skel_seed") + 1] == "7", cmd


def test_no_skel_seed_flag_forwards_no_skel_seed_option(tmp_path):
    """REQ-SKELSEED-BDRABCZUK-1AE6804B7DAF: without --skel-seed no --skel_seed is forwarded.

    The flag must exist and default to None (not to the template's 42), so a
    user-edited skel_seed in the derivatives pipeline_loop_2mm.json applies and
    stays untouched.
    """
    input_dir, output_dir, config_path = _make_dirs(tmp_path, {"skel_seed": 7})
    script, cmd = _run(input_dir, output_dir)
    assert vars(script.args).get("skel_seed", "<no --skel-seed option>") is None
    assert "--skel_seed" not in cmd, f"--skel_seed forwarded without the flag: {cmd}"
    assert json.loads(config_path.read_text())["skel_seed"] == 7
