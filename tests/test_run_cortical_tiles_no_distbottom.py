#!/usr/bin/env python3
"""
Stage 2 carries no distbottom option or key (epic TASK-NODISTB-BDRABCZUK-5DDC4636377C).

cortical_tiles no longer generates distbottom crops, so run_cortical_tiles.py refuses
both distbottom flags and writes no skip_distbottom key:

- REQ-NODISTB-BDRABCZUK-48941167DE7E: --with-distbottom is an argparse usage error (exit 2).
- REQ-NODISTB-BDRABCZUK-AA7C788C2305: --skip-distbottom is an argparse usage error (exit 2).
- REQ-NODISTB-BDRABCZUK-E562D25CD3A3: the written pipeline_loop_2mm.json has no top-level skip_distbottom key.
- REQ-NODISTB-BDRABCZUK-A117AE28A324: the shipped template has no top-level skip_distbottom key.
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


def _exit_code_of_parse(tmp_path, flag):
    """Parse a run_cortical_tiles command line carrying ``flag``; return the SystemExit code."""
    with pytest.raises(SystemExit) as excinfo:
        parsed_script(tmp_path / "input", tmp_path / "output", flag)
    return excinfo.value.code


class TestDistbottomFlagsRefused:
    """REQ-NODISTB-BDRABCZUK-48941167DE7E, REQ-NODISTB-BDRABCZUK-AA7C788C2305."""

    def test_with_distbottom_is_a_usage_error(self, tmp_path):
        assert _exit_code_of_parse(tmp_path, "--with-distbottom") == 2

    def test_skip_distbottom_is_a_usage_error(self, tmp_path):
        assert _exit_code_of_parse(tmp_path, "--skip-distbottom") == 2


class TestWrittenConfig:
    """REQ-NODISTB-BDRABCZUK-E562D25CD3A3."""

    @staticmethod
    def _written_config(tmp_path, config_overrides=None):
        input_dir, output_dir, config_path = stage_dirs_on_template(tmp_path, config_overrides)
        run_mocked(parsed_script(input_dir, output_dir, "--njobs", "1"))
        return json.loads(config_path.read_text())

    def test_template_seeded_config_has_no_skip_distbottom_key(self, tmp_path):
        assert "skip_distbottom" not in self._written_config(tmp_path)

    def test_legacy_skip_distbottom_key_is_dropped(self, tmp_path):
        assert "skip_distbottom" not in self._written_config(tmp_path, {"skip_distbottom": True})


class TestShippedTemplate:
    """REQ-NODISTB-BDRABCZUK-A117AE28A324."""

    def test_template_has_no_skip_distbottom_key(self):
        assert "skip_distbottom" not in json.loads(TEMPLATE_CONFIG.read_text())
