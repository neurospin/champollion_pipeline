"""main.py without a terminal on an input holding neither kind, both kinds, sourcedata T1s or two Morphologist
outputs: a stop before any stage (exit status 2 for neither kind, TASK-ARGDISC-BDRABCZUK-7F0CEB924F51; 1 otherwise)
and a clear ERROR record.

Split from tests/test_main_input_autodetect_ambiguous.py (TASK-INPUTKIND-BDRABCZUK-830F5595E185);
behaviour TASK-INPUTKIND-BDRABCZUK-5ECD06E2023D.
Requirements: one per test below (alm links on the Task root).
Harness: tests/main_input_autodetect_harness.py.
"""

from pathlib import Path

import pytest

from champollion_pipeline.utils.input_kind import scan_input, sourcedata_t1_files
from tests.main_input_autodetect_harness import (
    BATCH_CASES,
    INPUT_KIND_OPTION,
    STAGE1,
    _assert_error_contains,
    _assert_stopped,
    _batch,
    _both_dataset,
    _candidates,
    _config,
    _empty_dataset,
    _main_module_under_test,
    _run_stages,
    _sourcedata_dataset,
    _stop_config,
    _streams,
    _two_outputs_dataset,
)


@pytest.fixture
def main_module(monkeypatch, tmp_path):
    return _main_module_under_test(monkeypatch, tmp_path)


# ====================== Neither kind ======================


@pytest.mark.unit
def test_neither_kind_exits_2_before_any_stage(main_module, monkeypatch, tmp_path):
    _batch(monkeypatch)
    config = _stop_config(main_module, tmp_path, _empty_dataset(tmp_path))
    _assert_stopped(*_run_stages(main_module, monkeypatch, config), expected=2)


@pytest.mark.unit
def test_neither_kind_error_names_input_path_and_option(main_module, monkeypatch, tmp_path, pipeline_caplog):
    _batch(monkeypatch)
    empty = _empty_dataset(tmp_path)
    _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, empty))
    _assert_error_contains(pipeline_caplog, str(empty), INPUT_KIND_OPTION)


# ====================== Both kinds and sourcedata without a terminal ======================


def _assert_batch_stops(main_module, monkeypatch, tmp_path, ttys, dataset: Path) -> None:
    """Without a terminal on one stream, main() on the dataset exits 1 before any stage."""
    _streams(monkeypatch, *ttys)
    config = _stop_config(main_module, tmp_path, dataset)
    _assert_stopped(*_run_stages(main_module, monkeypatch, config))


@pytest.mark.unit
@BATCH_CASES
def test_both_kinds_in_batch_exits_1_before_any_stage(main_module, monkeypatch, tmp_path, ttys):
    _assert_batch_stops(main_module, monkeypatch, tmp_path, ttys, _both_dataset(tmp_path))


@pytest.mark.unit
def test_both_kinds_batch_error_names_examples(main_module, monkeypatch, tmp_path, pipeline_caplog):
    _batch(monkeypatch)
    root = _both_dataset(tmp_path)
    scan = scan_input(str(root))
    _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root))
    _assert_error_contains(pipeline_caplog, scan.first_t1, scan.first_graph, INPUT_KIND_OPTION)


@pytest.mark.unit
@BATCH_CASES
def test_sourcedata_in_batch_exits_1_before_any_stage(main_module, monkeypatch, tmp_path, ttys):
    _assert_batch_stops(main_module, monkeypatch, tmp_path, ttys, _sourcedata_dataset(tmp_path))


@pytest.mark.unit
def test_sourcedata_batch_error_names_first_sourcedata_t1(main_module, monkeypatch, tmp_path, pipeline_caplog):
    _batch(monkeypatch)
    root = _sourcedata_dataset(tmp_path)
    _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root))
    _assert_error_contains(pipeline_caplog, sourcedata_t1_files(str(root))[0], INPUT_KIND_OPTION)


@pytest.mark.unit
def test_input_inside_sourcedata_runs_stage1(main_module, monkeypatch, tmp_path):
    _batch(monkeypatch)
    sourcedata = _sourcedata_dataset(tmp_path) / "sourcedata"
    assert sourcedata_t1_files(str(sourcedata)) == []
    status, argv = _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, sourcedata))
    assert STAGE1 in argv, f"stage 1 did not run on an input inside sourcedata (exit status {status})"
    assert argv[STAGE1][1] == str(sourcedata), f"stage 1 output root {argv[STAGE1][1]!r}"


# ====================== Two Morphologist outputs ======================


@pytest.mark.unit
@BATCH_CASES
@pytest.mark.parametrize("kind", [None, "morphologist"], ids=["kind_unset", "kind_morphologist"])
def test_two_subjects_folders_in_batch_exit_1(main_module, monkeypatch, tmp_path, ttys, kind):
    _streams(monkeypatch, *ttys)
    config = _stop_config(main_module, tmp_path, _two_outputs_dataset(tmp_path))
    args = (INPUT_KIND_OPTION, kind) if kind else ()
    _assert_stopped(*_run_stages(main_module, monkeypatch, config, *args))


@pytest.mark.unit
def test_two_subjects_folders_batch_error_names_folders(main_module, monkeypatch, tmp_path, pipeline_caplog):
    _batch(monkeypatch)
    root = _two_outputs_dataset(tmp_path)
    candidates = _candidates(root)
    _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root))
    _assert_error_contains(pipeline_caplog, *candidates)
