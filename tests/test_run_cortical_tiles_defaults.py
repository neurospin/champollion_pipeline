#!/usr/bin/env python3
"""
Stage-2 defaults for run_cortical_tiles.py (TASK-129).

champollion_V1 (commit 770a5b74) no longer reads extremities or distbottom
crops, so stage 2 does not generate them (distbottom options:
tests/test_run_cortical_tiles_no_distbottom.py):

- REQ-TILESDEF-01: without --input-types, forward exactly skeleton and foldlabel.
- REQ-NODISTB-BDRABCZUK-0E5E6CB2FAF0: the shipped pipeline_loop_2mm.json sets input_type to skeleton
  (extremities is no longer an input type: tests/test_run_cortical_tiles_arguments.py::TestInputTypesRestricted).
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
        """Regression guard: an explicit --input-types is forwarded as given, in its order."""
        input_dir, output_dir, _ = stage_dirs_on_template(tmp_path)
        cmd = _run(
            input_dir,
            output_dir,
            ["--input-types", "foldlabel", "skeleton"],
        )
        values = _values_after_option(cmd, "-y")
        assert values == ["foldlabel", "skeleton"]


class TestShippedTemplate:
    """REQ-NODISTB-BDRABCZUK-0E5E6CB2FAF0: the shipped template names no removed input type."""

    def test_template_input_type_is_skeleton(self):
        assert json.loads(TEMPLATE_CONFIG.read_text())["input_type"] == "skeleton"
