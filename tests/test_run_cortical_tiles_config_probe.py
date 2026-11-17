#!/usr/bin/env python3
"""
Stage 2 existence probe of pipeline_loop_2mm.json (DEF-CORTILES-BDRABCZUK-C1A5989DEEA4): on a nominal first run the
file is absent from the output directory, and the probe must not log the ERROR that ScriptBuilder.validate_paths
emits by specification for a missing path. The real validate_paths runs; only the crop command is mocked.
"""

import logging
from unittest.mock import patch

from tests.stage2_layout import TEMPLATE_CONFIG, parsed_script, stage_dirs


def _run_without_config(root):
    input_dir, output_dir = stage_dirs(root)
    script = parsed_script(input_dir, output_dir)
    with patch.object(script, "execute_command", return_value=0):
        script.run()
    return output_dir


def test_absent_config_logs_no_error_record(temp_dir, caplog):
    with caplog.at_level(logging.ERROR):
        _run_without_config(temp_dir)

    # validate_paths logs through the champollion_utils logger, outside the pipeline_caplog filter: read caplog.
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert not [m for m in errors if TEMPLATE_CONFIG.name in m]


def test_absent_config_is_still_seeded_from_the_template(temp_dir):
    output_dir = _run_without_config(temp_dir)

    assert (output_dir / TEMPLATE_CONFIG.name).is_file()
