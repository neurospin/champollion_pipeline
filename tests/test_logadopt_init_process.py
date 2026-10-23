"""Unified-logger adoption in the pipeline entry points (TASK-214).

REQ-LOGADOPT-01: main.py run as a script logs one CRASH line for an uncaught Exception.
REQ-LOGADOPT-02: each stage script run as __main__ logs one CRASH line for an uncaught Exception.
REQ-LOGADOPT-03: main.py prints each orchestrator record once, in the unified format.
REQ-LOGADOPT-05: a stage exception caught by main.py is logged at FAIL with the stage name and traceback.

Process-wide state (root logger, sys.excepthook) is involved, so the entry
points run in subprocesses. A small driver stubs the update check (no network)
and then runs the target with runpy as ``__main__``.
"""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_PY = REPO_ROOT / "main.py"
STAGE_DIR = REPO_ROOT / "src" / "champollion_pipeline"

STAGE_SCRIPTS = [
    "generate_champollion_config",
    "generate_embeddings",
    "generate_labelling_qc",
    "generate_masks",
    "generate_morphologist_graphs",
    "generate_snapshots",
    "prune_failed_subjects",
    "purge_subject",
    "put_together_embeddings",
    "run_cortical_tiles",
    "train_champollion",
]

# HH:MM:SS LEVEL    module::qualname | message
UNIFIED_LINE = re.compile(r"^\d\d:\d\d:\d\d (?P<level>[A-Z]+)\s+(?P<provenance>\S+::\S+) \| (?P<message>.*)$")
CRASH_WORD = re.compile(r"\bCRASH\b")

DRIVER_TEMPLATE = """\
import runpy
import sys

import champollion_utils.script_builder
import champollion_utils.update_check


def _no_update_check(*args, **kwargs):
    return None


champollion_utils.update_check.check_for_updates = _no_update_check
champollion_utils.script_builder.check_for_updates = _no_update_check


def _raise_from_build(self, *args, **kwargs):
    raise RuntimeError("raised by the test driver")


if {raise_in!r}:
    setattr(champollion_utils.script_builder.ScriptBuilder, {raise_in!r}, _raise_from_build)

target = {target!r}
sys.argv = [target] + {argv!r}
runpy.run_path(target, run_name="__main__")
"""


def _run_driver(tmp_path: Path, target: Path, argv, *, raise_in: str = "") -> subprocess.CompletedProcess:
    """Run target as __main__; when raise_in names a ScriptBuilder method, that method raises."""
    driver = tmp_path / "crash_driver.py"
    driver.write_text(
        DRIVER_TEMPLATE.format(target=str(target), argv=list(argv), raise_in=raise_in),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["NO_COLOR"] = "1"
    return subprocess.run(
        [sys.executable, str(driver)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _crash_lines(stderr: str):
    return [line for line in stderr.splitlines() if CRASH_WORD.search(line)]


def _assert_one_unified_crash_line(result: subprocess.CompletedProcess, provenance: str) -> None:
    crash_lines = _crash_lines(result.stderr)
    assert len(crash_lines) == 1, f"expected exactly one CRASH line on stderr, got {crash_lines!r}\n{result.stderr}"
    match = UNIFIED_LINE.match(crash_lines[0])
    assert match is not None, f"CRASH line is not in the unified format: {crash_lines[0]!r}"
    assert match["level"] == "CRASH"
    assert match["provenance"] == provenance
    # Exit status is unchanged (the MCP server derives job status from it).
    assert result.returncode == 1, result.stderr


def test_main_py_uncaught_exception_logs_one_crash_line(tmp_path):
    """REQ-LOGADOPT-01: a missing --config file escapes main() as FileNotFoundError."""
    missing = tmp_path / "missing_config.yaml"
    result = _run_driver(tmp_path, MAIN_PY, ["--config", str(missing)])
    _assert_one_unified_crash_line(result, "champollion_pipeline.config::ConfigLoader.load_from_yaml")


@pytest.mark.parametrize("script", STAGE_SCRIPTS)
def test_stage_script_uncaught_exception_logs_one_crash_line(tmp_path, script):
    """REQ-LOGADOPT-02: an Exception raised from ScriptBuilder.build escapes the stage script."""
    result = _run_driver(tmp_path, STAGE_DIR / f"{script}.py", [], raise_in="build")
    _assert_one_unified_crash_line(result, "crash_driver::_raise_from_build")


def test_main_py_orchestrator_record_printed_once_in_unified_format(tmp_path):
    """REQ-LOGADOPT-03: an orchestrator record appears on the console once, in the unified format."""
    config = tmp_path / "no_stages.yaml"
    config.write_text(
        textwrap.dedent(
            """\
            log_to_file: false
            log_to_console: true
            stages:
              generate_morphologist_graphs: false
              run_cortical_tiles: false
              generate_champollion_config: false
              generate_embeddings: false
              put_together_embeddings: false
              generate_snapshots: false
            """
        ),
        encoding="utf-8",
    )
    result = _run_driver(tmp_path, MAIN_PY, ["--config", str(config)])
    assert result.returncode == 0, result.stderr

    console = result.stdout.splitlines() + result.stderr.splitlines()
    lines = [line for line in console if "Starting Champollion Pipeline" in line]
    assert len(lines) == 1, f"expected the record once on the console, got {lines!r}"
    match = UNIFIED_LINE.match(lines[0])
    assert match is not None, f"orchestrator line is not in the unified format: {lines[0]!r}"
    assert match["level"] == "INFO"


def test_main_py_stage_exception_logged_at_fail_with_traceback(tmp_path):
    """REQ-LOGADOPT-05: an Exception raised inside a stage is logged at FAIL, naming the stage, with its traceback."""
    input_dir = tmp_path / "subjects"
    input_dir.mkdir()
    config = tmp_path / "one_stage.yaml"
    config.write_text(
        textwrap.dedent(
            f"""\
            log_to_file: false
            log_to_console: true
            dataset:
              input_path: {input_dir}
              morphologist_graphs: {tmp_path / "graphs"}
            stages:
              generate_morphologist_graphs: true
              run_cortical_tiles: false
              generate_champollion_config: false
              generate_embeddings: false
              put_together_embeddings: false
              generate_snapshots: false
            """
        ),
        encoding="utf-8",
    )
    # The stage calls script.parse_args(args) on its ScriptBuilder subclass; make that raise.
    result = _run_driver(tmp_path, MAIN_PY, ["--config", str(config)], raise_in="parse_args")

    stderr_lines = result.stderr.splitlines()
    fail_indexes = [
        index
        for index, line in enumerate(stderr_lines)
        if (match := UNIFIED_LINE.match(line)) is not None and match["level"] == "FAIL"
    ]
    assert len(fail_indexes) == 1, f"expected exactly one FAIL line on stderr\n{result.stderr}"
    fail_line = stderr_lines[fail_indexes[0]]
    assert "generate_morphologist_graphs" in fail_line, fail_line
    following = stderr_lines[fail_indexes[0] + 1 :]
    assert following and following[0] == "Traceback (most recent call last):", result.stderr
    assert any("RuntimeError: raised by the test driver" in line for line in following), result.stderr
    # Recover-and-continue behaviour unchanged: the stage is marked failed and the run exits 1.
    assert result.returncode == 1, result.stderr
