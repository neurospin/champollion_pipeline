#!/usr/bin/env python3
"""
--log of the four extra scripts in src/ (RUNLOG epic follow-up, user decisions 2026-10-22).

compare_databases, generate_umap_reference, run_cka and upload_models_to_hf are not declared in
pyproject.toml's [project.scripts], so test_stage_main_log.py does not cover them. Their module-level
main() must go through ScriptBuilder.main() (champollion_utils >= 0.3.0) for --log [PATH] to write a
run log, and each overrides default_log_path() so a bare --log lands next to what the script writes:

- compare_databases: <parent of --output>/.log/compare_databases_<YYYYmmdd-HHMMSS>.log
- generate_umap_reference: <--output_dir>/.log/generate_umap_reference_<YYYYmmdd-HHMMSS>.log
- run_cka: <--output_dir>/.log/run_cka_<YYYYmmdd-HHMMSS>.log
- upload_models_to_hf: <parent of models_dir>/.log/upload_models_to_hf_<YYYYmmdd-HHMMSS>.log
  (outside the folder uploaded to HuggingFace)

The main() tests call the entry point in-process with sys.argv patched. The script class's run() is
replaced by a stub returning 0, so no work runs; the update check and the process-wide logging setup
(init_pipeline_process, init_process) are replaced by no-ops so the root logger and sys.excepthook of
the test session stay untouched. The default_log_path() tests parse a command line and call
default_log_path(); nothing runs and nothing is written. The start time is bracketed by two clock
reads, so the expected file name is one of two stamps.
"""

import importlib
import inspect
import sys
from datetime import datetime
from os.path import join

import pytest
from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.derivatives_layout import compute_run_log_path

DATA = "/data/myproject"

# Smallest command line each script parses (without --log); nothing is read from these paths.
SCRIPT_ARGV = {
    "compare_databases": [
        f"--labeled_subjects_dir={join(DATA, 'subjects')}",
        "--path_to_graph_a=t1mri/t1/default_analysis/folds/3.3/base2018_manual",
        "--path_to_graph_b=t1mri/t1/default_analysis/folds/3.3/base2018b_manual",
        f"--output={join(DATA, 'qc', 'db_comparison.csv')}",
    ],
    "generate_umap_reference": [
        f"--models_dir={join(DATA, 'models')}",
        "--reference_subpath=ukb40_random_embeddings/full_embeddings.csv",
        f"--output_dir={join(DATA, 'reference_data')}",
    ],
    "run_cka": [
        f"--path_a={join(DATA, 'embeddings_a')}",
        f"--path_b={join(DATA, 'embeddings_b')}",
        f"--output_dir={join(DATA, 'cka')}",
    ],
    "upload_models_to_hf": [
        join(DATA, "models", "Champollion_V1"),
        "neurospin/champollion",
        "--masks-version=canonical_25",
    ],
}
SCRIPTS = sorted(SCRIPT_ARGV)


def _module_and_script_class(name: str):
    """Import the script's module from src/; return it and its single ScriptBuilder subclass."""
    module = importlib.import_module(name)
    classes = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, ScriptBuilder) and obj is not ScriptBuilder and obj.__module__ == name
    ]
    assert len(classes) == 1, f"{name} defines {classes!r}, expected one ScriptBuilder subclass"
    return module, classes[0]


@pytest.fixture
def stubbed_main(monkeypatch):
    """Return a function (name, log_arg) -> (exit code, run calls) running the module's main() with run() stubbed."""

    def _run(name: str, log_arg: str):
        module, script_class = _module_and_script_class(name)
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
        monkeypatch.setattr(sys, "argv", [name, *SCRIPT_ARGV[name], log_arg])
        return module.main(), calls

    return _run


def _default_log_path_and_candidates(name, argv, derivatives_dir):
    """Parse argv, call default_log_path(); return it and the paths allowed for a start time around the call."""
    _, script_class = _module_and_script_class(name)
    script = script_class()
    script.parse_args(argv)
    before = datetime.now().astimezone()
    path = script.default_log_path()
    after = datetime.now().astimezone()
    candidates = {compute_run_log_path(derivatives_dir, name, moment) for moment in (before, after)}
    return path, candidates


# ---------------------------------------------------------------------------
# main() goes through ScriptBuilder.main(): --log=PATH and bare --log write a run log
# ---------------------------------------------------------------------------


# REQ-RUNLOG-BDRABCZUK-B24D2B887167 (compare_databases), REQ-RUNLOG-BDRABCZUK-757852C3E384
# (generate_umap_reference), REQ-RUNLOG-BDRABCZUK-658AF9FCBEB2 (run_cka),
# REQ-RUNLOG-BDRABCZUK-2B8669E8821D (upload_models_to_hf)
@pytest.mark.parametrize("name", SCRIPTS)
def test_log_path_writes_run_log(tmp_path, stubbed_main, name):
    log_path = tmp_path / "explicit" / f"{name}.log"
    exit_code, calls = stubbed_main(name, f"--log={log_path}")
    assert exit_code == 0
    assert len(calls) == 1
    assert log_path.is_file(), f"{name} --log={log_path} wrote no run log"


# REQ-RUNLOG-BDRABCZUK-4A2D44CE340E (compare_databases), REQ-RUNLOG-BDRABCZUK-CF3411AE01C3
# (generate_umap_reference), REQ-RUNLOG-BDRABCZUK-A022B3C10A19 (run_cka),
# REQ-RUNLOG-BDRABCZUK-AE0603700EE5 (upload_models_to_hf)
@pytest.mark.parametrize("name", SCRIPTS)
def test_bare_log_writes_run_log_at_default_log_path(tmp_path, monkeypatch, stubbed_main, name):
    log_path = tmp_path / "default" / f"{name}.log"
    _, script_class = _module_and_script_class(name)
    monkeypatch.setattr(script_class, "default_log_path", lambda self: str(log_path))
    exit_code, calls = stubbed_main(name, "--log")
    assert exit_code == 0
    assert len(calls) == 1
    assert log_path.is_file(), f"{name} --log wrote no run log at default_log_path() {log_path}"


# ---------------------------------------------------------------------------
# default_log_path(): where a bare --log writes, per script
# ---------------------------------------------------------------------------


# REQ-RUNLOG-BDRABCZUK-4A2D44CE340E
def test_compare_databases_bare_log_writes_next_to_output_csv():
    argv = [*SCRIPT_ARGV["compare_databases"], "--log"]
    path, candidates = _default_log_path_and_candidates("compare_databases", argv, join(DATA, "qc"))
    assert path in candidates


# REQ-RUNLOG-BDRABCZUK-CF3411AE01C3
def test_umap_reference_bare_log_writes_under_output_dir():
    argv = [*SCRIPT_ARGV["generate_umap_reference"], "--log"]
    path, candidates = _default_log_path_and_candidates("generate_umap_reference", argv, join(DATA, "reference_data"))
    assert path in candidates


# REQ-RUNLOG-BDRABCZUK-A022B3C10A19
def test_run_cka_bare_log_writes_under_output_dir():
    argv = [*SCRIPT_ARGV["run_cka"], "--log"]
    path, candidates = _default_log_path_and_candidates("run_cka", argv, join(DATA, "cka"))
    assert path in candidates


# REQ-RUNLOG-BDRABCZUK-AE0603700EE5
@pytest.mark.parametrize("trailing_slash", ["", "/"], ids=["plain", "trailing_slash"])
def test_upload_models_bare_log_writes_beside_models_dir(trailing_slash):
    models_dir = join(DATA, "models", "Champollion_V1") + trailing_slash
    argv = [models_dir, "neurospin/champollion", "--masks-version=canonical_25", "--log"]
    path, candidates = _default_log_path_and_candidates("upload_models_to_hf", argv, join(DATA, "models"))
    assert path in candidates
