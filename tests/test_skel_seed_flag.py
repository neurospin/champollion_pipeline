#!/usr/bin/env python3
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

import pytest

from tests.stage2_layout import (  # noqa: F401
    TEMPLATE_CONFIG,
    _stub_whole_brain_functions,  # noqa: V103 - autouse fixture
    parsed_script,
    run_mocked,
    stage_dirs_on_template,
)

pytestmark = pytest.mark.unit  # noqa: V107


def _run(input_dir, output_dir, extra_args=()):
    """Run RunCorticalTiles.run() with the subprocess mocked.

    Returns the parsed script and the command handed to generate_sulcal_regions.py.
    """
    script = parsed_script(input_dir, output_dir, "--njobs", "1", *extra_args)
    cmd = run_mocked(script).call_args[0][0]
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
    input_dir, output_dir, _ = stage_dirs_on_template(tmp_path)
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
    input_dir, output_dir, config_path = stage_dirs_on_template(tmp_path, {"skel_seed": 7})
    script, cmd = _run(input_dir, output_dir)
    assert vars(script.args).get("skel_seed", "<no --skel-seed option>") is None
    assert "--skel_seed" not in cmd, f"--skel_seed forwarded without the flag: {cmd}"
    assert json.loads(config_path.read_text())["skel_seed"] == 7
