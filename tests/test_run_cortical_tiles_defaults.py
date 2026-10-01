#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-2 defaults for run_cortical_tiles.py (TASK-129).

champollion_V1 (commit 770a5b74) no longer reads extremities or distbottom
crops, so stage 2 must not generate them unless explicitly asked to:

- REQ-TILESDEF-01: without --input-types, forward exactly skeleton and foldlabel.
- REQ-TILESDEF-02: without --with-distbottom, write skip_distbottom = true.
- REQ-TILESDEF-03: with --with-distbottom, write skip_distbottom = false.
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
    """Run RunCorticalTiles.run() with the subprocess mocked; return the captured command."""
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
    return mock_exec.call_args[0][0]


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
        input_dir, output_dir, _ = _make_dirs(tmp_path)
        cmd = _run(input_dir, output_dir)
        values = _values_after_option(cmd, "-y")
        assert sorted(values) == ["foldlabel", "skeleton"]

    def test_explicit_input_types_forwarded_verbatim(self, tmp_path):
        """Regression guard: extremities can still be requested explicitly."""
        input_dir, output_dir, _ = _make_dirs(tmp_path)
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
        input_dir, output_dir, config_path = _make_dirs(tmp_path)
        assert json.loads(config_path.read_text())["skip_distbottom"] is False  # template default
        _run(input_dir, output_dir)
        assert json.loads(config_path.read_text())["skip_distbottom"] is True


class TestDistbottomOptIn:
    """REQ-TILESDEF-03."""

    def test_with_distbottom_sets_skip_distbottom_false(self, tmp_path):
        """--with-distbottom writes skip_distbottom = false, even over a pre-existing true."""
        input_dir, output_dir, config_path = _make_dirs(tmp_path, {"skip_distbottom": True})
        _run(input_dir, output_dir, ["--with-distbottom"])
        assert json.loads(config_path.read_text())["skip_distbottom"] is False
