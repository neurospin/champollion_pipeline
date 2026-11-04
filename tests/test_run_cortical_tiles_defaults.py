#!/usr/bin/env python3
"""
Stage-2 defaults for run_cortical_tiles.py (TASK-129).

champollion_V1 (commit 770a5b74) no longer reads extremities or distbottom
crops, so stage 2 must not generate them unless explicitly asked to:

- REQ-TILESDEF-01: without --input-types, forward exactly skeleton and foldlabel.
- REQ-TILESDEF-02: without --with-distbottom, write skip_distbottom = true.
- REQ-TILESDEF-03: with --with-distbottom, write skip_distbottom = false.
"""

import json

import pytest

from tests.stage2_layout import (  # noqa: F401
    _stub_whole_brain_functions,  # noqa: V103 - autouse fixture
    parsed_script,
    run_mocked,
    stage_dirs_on_template,
)

pytestmark = pytest.mark.unit  # noqa: V107


def _run(input_dir, output_dir, extra_args=()):
    """Run RunCorticalTiles.run() with the subprocess mocked; return the captured command."""
    script = parsed_script(input_dir, output_dir, "--njobs", "1", *extra_args)
    return run_mocked(script).call_args[0][0]


def _values_after_option(cmd, option):
    """Return the tokens following `option` up to the next option token (or the end)."""
    assert option in cmd, f"{option} not in command: {cmd}"
    start = cmd.index(option) + 1
    values = []
    for token in cmd[start:]:
        if token.startswith("-"):
            break
        values.append(token)
    return values


class TestDefaultInputTypes:
    """REQ-TILESDEF-01."""

    def test_default_forwards_exactly_skeleton_and_foldlabel(self, tmp_path):
        """Without --input-types, -y carries exactly skeleton and foldlabel (no extremities)."""
        input_dir, output_dir, _ = stage_dirs_on_template(tmp_path)
        cmd = _run(input_dir, output_dir)
        values = _values_after_option(cmd, "-y")
        assert sorted(values) == ["foldlabel", "skeleton"]

    def test_explicit_input_types_forwarded_verbatim(self, tmp_path):
        """Regression guard: extremities can still be requested explicitly."""
        input_dir, output_dir, _ = stage_dirs_on_template(tmp_path)
        cmd = _run(
            input_dir,
            output_dir,
            ["--input-types", "skeleton", "foldlabel", "extremities"],
        )
        values = _values_after_option(cmd, "-y")
        assert values == ["skeleton", "foldlabel", "extremities"]


class TestDistbottomDefault:
    """REQ-TILESDEF-02."""

    def test_default_sets_skip_distbottom_true(self, tmp_path):
        """Without --with-distbottom, the written pipeline JSON has skip_distbottom = true."""
        input_dir, output_dir, config_path = stage_dirs_on_template(tmp_path)
        assert json.loads(config_path.read_text())["skip_distbottom"] is False  # template default
        _run(input_dir, output_dir)
        assert json.loads(config_path.read_text())["skip_distbottom"] is True


class TestDistbottomOptIn:
    """REQ-TILESDEF-03."""

    def test_with_distbottom_sets_skip_distbottom_false(self, tmp_path):
        """--with-distbottom writes skip_distbottom = false, even over a pre-existing true."""
        input_dir, output_dir, config_path = stage_dirs_on_template(tmp_path, {"skip_distbottom": True})
        _run(input_dir, output_dir, ["--with-distbottom"])
        assert json.loads(config_path.read_text())["skip_distbottom"] is False
