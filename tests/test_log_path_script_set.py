#!/usr/bin/env python3
"""
--log=PATH over one set of scripts: each writes the run log to PATH.

One rule over a set: every console script declared in pyproject.toml's [project.scripts] other than
champollion-pipeline, plus the four scripts of src/ that are not console scripts (compare_databases,
generate_umap_reference, run_cka, upload_models_to_hf). champollion-pipeline is outside the set: its
--log=DIR keeps the folder meaning of python main.py --log=DIR (user decision 2026-11-20).

The console-script members come from pyproject.toml, so a console script added later joins the set
and needs a command line in tests/test_stage_main_log.py (checked there). Each member's entry point
runs in-process with sys.argv patched, run() replaced by a stub returning 0, and the update check and
the process-wide logging setup replaced by no-ops, as in tests/test_stage_main_log.py and
tests/test_extra_scripts_log.py, whose command lines and class lookups this file reuses.

The set is finite and each member is cheap to run, so the property is checked on every member by
parametrisation rather than on a Hypothesis sample of it.

Requirement: the invariant generalising REQ-RUNLOG-BDRABCZUK-209D8D39D925,
REQ-RUNLOG-BDRABCZUK-B24D2B887167, REQ-RUNLOG-BDRABCZUK-757852C3E384,
REQ-RUNLOG-BDRABCZUK-658AF9FCBEB2 and REQ-RUNLOG-BDRABCZUK-2B8669E8821D (candidate INV-7F238545CFF7).
"""

import pytest

from tests.main_stage_helpers import stub_script_entry_point
from tests.test_extra_scripts_log import SCRIPT_ARGV, SCRIPTS, _module_and_script_class
from tests.test_stage_main_log import ENTRY_POINT_ARGV, LOG_PATH_SCRIPTS, _entry_point_and_script_class

# Console scripts (hyphenated names) and src/ scripts (module names): the two name sets do not meet.
LOG_PATH_SET = sorted([*LOG_PATH_SCRIPTS, *SCRIPTS])


def _member(name: str):
    """Return the member's module, entry point, ScriptBuilder subclass and smallest command line."""
    if name in ENTRY_POINT_ARGV:
        module, entry_point, script_class = _entry_point_and_script_class(name)
        return module, entry_point, script_class, ENTRY_POINT_ARGV[name]
    module, script_class = _module_and_script_class(name)
    return module, module.main, script_class, SCRIPT_ARGV[name]


def test_set_leaves_out_champollion_pipeline():
    """champollion-pipeline keeps --log=DIR, so the --log=PATH rule does not cover it."""
    assert "champollion-pipeline" not in LOG_PATH_SET
    assert len(LOG_PATH_SET) == len(set(LOG_PATH_SET))


@pytest.mark.parametrize("name", LOG_PATH_SET)
def test_script_log_path_writes_run_log(tmp_path, monkeypatch, name):
    """Running the member with --log=PATH writes a run log file at PATH."""
    module, entry_point, script_class, argv = _member(name)
    log_path = tmp_path / "explicit" / f"{name}.log"
    calls = stub_script_entry_point(monkeypatch, module, script_class, [name, *argv, f"--log={log_path}"])

    assert entry_point() == 0
    assert len(calls) == 1
    assert log_path.is_file(), f"{name} --log={log_path} wrote no run log"
