#!/usr/bin/env python3
"""
--log from the console entry points of the stage scripts (RUNLOG epic, TASK-RUNLOG-BDRABCZUK-3FC46A6918C4).

champollion_utils ScriptBuilder (>= 0.3.0) owns --log [PATH]: ScriptBuilder.main() wraps print_args,
before_run and run in a RunLogCapture. The module-level main() of a stage script is the console entry
point declared in pyproject.toml's [project.scripts] (and what ``python -m`` runs); it must go through
ScriptBuilder.main() for --log to write anything.

Each test calls the entry point in-process with sys.argv patched. The script class's run() is replaced
by a stub returning 0, so no stage work runs; the update check and the process-wide logging setup
(init_pipeline_process, init_process) are replaced by no-ops so the root logger and sys.excepthook of
the test session stay untouched. The bare --log case replaces default_log_path() by a temporary path:
the stage-specific paths themselves are covered by test_stage_default_log_path.py and
test_maintenance_default_log_path.py.
"""

import importlib
import inspect
import sys
from os.path import join
from pathlib import Path

import pytest
import tomllib
from champollion_utils.script_builder import ScriptBuilder

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

ROOT = "/data/myproject"
DERIVATIVES = join(ROOT, "derivatives")
MASK_VERSION_DIR = join(DERIVATIVES, "champollion_V1", "canonical_26")
HF_REPO = "neurospin/Champollion_V1"

# Smallest command line each console script parses (without --log); nothing is read from these paths.
ENTRY_POINT_ARGV = {
    "champollion-morphologist": ["/raw/t1", ROOT],
    "champollion-cortical-tiles": [
        join(DERIVATIVES, "morphologist-6.0", "subjects"),
        DERIVATIVES,
        "--path_to_graph=t1mri/default_acquisition/default_analysis/folds/3.3/base",
        "--path_sk_with_hull=t1mri/default_acquisition/default_analysis/segmentation/mesh",
    ],
    "champollion-config": [
        join(DERIVATIVES, "cortical_tiles-2026", "crops", "canonical_26", "2mm"),
        "--dataset=myproject",
        "--masks=canonical_26",
    ],
    "champollion-embeddings": [HF_REPO, ROOT, "--masks=canonical_26"],
    "champollion-combine": [
        join(MASK_VERSION_DIR, "region_embeddings"),
        f"--output_path={join(MASK_VERSION_DIR, 'combined')}",
    ],
    "champollion-snapshots": [
        f"--output_dir={join(MASK_VERSION_DIR, 'snapshots')}",
        f"--embeddings_dir={join(MASK_VERSION_DIR, 'embeddings')}",
    ],
    "champollion-train": ["--dataset=myproject", "--region=cingulate_left"],
    "champollion-prune": [DERIVATIVES, "--qc=/data/myproject/qc.tsv"],
    "champollion-purge": [DERIVATIVES, "--subject=sub-123456"],
    "champollion-labelling-qc": [
        join(DERIVATIVES, "morphologist-6.0", "subjects"),
        join(ROOT, "qc", "labelling_qc.tsv"),
        "--path_to_graph=t1mri/default_acquisition/default_analysis/folds/3.3",
    ],
}


def _console_scripts() -> dict[str, str]:
    """Return the [project.scripts] table of pyproject.toml: entry-point name -> "module:function"."""
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]["scripts"]


CONSOLE_SCRIPTS = sorted(_console_scripts())


def test_every_console_script_has_a_command_line():
    """The parametrized tests below cover each declared console script."""
    assert set(CONSOLE_SCRIPTS) == set(ENTRY_POINT_ARGV)


def _entry_point_and_script_class(name: str):
    """Import the console script's module; return its entry-point function and its ScriptBuilder subclass."""
    module_name, function_name = _console_scripts()[name].split(":")
    module = importlib.import_module(module_name)
    classes = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, ScriptBuilder) and obj is not ScriptBuilder and obj.__module__ == module_name
    ]
    assert len(classes) == 1, f"{module_name} defines {classes!r}, expected one ScriptBuilder subclass"
    return module, getattr(module, function_name), classes[0]


@pytest.fixture
def stubbed_entry_point(monkeypatch):
    """Return a function (name, log_arg) -> (exit code, run calls) running the entry point with run() stubbed."""

    def _run(name: str, log_arg: str):
        module, entry_point, script_class = _entry_point_and_script_class(name)
        calls = []

        def _fake_run(self):
            calls.append(self)
            return 0

        monkeypatch.setattr(script_class, "run", _fake_run)
        monkeypatch.setattr("champollion_utils.script_builder.check_for_updates", lambda *a, **k: None)
        monkeypatch.setattr("champollion_utils.script_builder.init_process", lambda *a, **k: None)
        for setup_name in ("init_pipeline_process", "detach_frozen_root_handlers"):
            if hasattr(module, setup_name):
                monkeypatch.setattr(module, setup_name, lambda *a, **k: None)
        monkeypatch.setattr(sys, "argv", [name, *ENTRY_POINT_ARGV[name], log_arg])
        return entry_point(), calls

    return _run


# REQ-RUNLOG-BDRABCZUK-79BCF456CFC5
@pytest.mark.parametrize("name", CONSOLE_SCRIPTS)
def test_console_script_log_path_writes_run_log(tmp_path, stubbed_entry_point, name):
    log_path = tmp_path / "explicit" / f"{name}.log"
    exit_code, calls = stubbed_entry_point(name, f"--log={log_path}")
    assert exit_code == 0
    assert len(calls) == 1
    assert log_path.is_file(), f"{name} --log={log_path} wrote no run log"


# REQ-RUNLOG-BDRABCZUK-51FEBBE2CBB2
@pytest.mark.parametrize("name", CONSOLE_SCRIPTS)
def test_console_script_bare_log_writes_at_default_log_path(tmp_path, monkeypatch, stubbed_entry_point, name):
    log_path = tmp_path / "default" / f"{name}.log"
    _, _, script_class = _entry_point_and_script_class(name)
    monkeypatch.setattr(script_class, "default_log_path", lambda self: str(log_path))
    exit_code, calls = stubbed_entry_point(name, "--log")
    assert exit_code == 0
    assert len(calls) == 1
    assert log_path.is_file(), f"{name} --log wrote no run log at default_log_path() {log_path}"
