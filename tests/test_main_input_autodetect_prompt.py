"""main.py on a terminal: the input kind prompt (both kinds, sourcedata T1s), the subjects folder prompt (two
Morphologist outputs), their answers, and the stops on an unlisted answer or the end of input.

Split from tests/test_main_input_autodetect_ambiguous.py (TASK-INPUTKIND-BDRABCZUK-830F5595E185);
behaviour TASK-INPUTKIND-BDRABCZUK-5ECD06E2023D.
Requirements: one per test below (alm links on the Task root).
Harness: tests/main_input_autodetect_harness.py.
"""

import os

import pytest

from champollion_pipeline.utils.input_kind import scan_input, sourcedata_t1_files
from tests.main_input_autodetect_harness import (
    DATASETS,
    KINDS,
    MORPHOLOGIST_6,
    STAGE1,
    STAGE2,
    UNKNOWN_KIND,
    UNLISTED_FOLDER,
    _assert_stopped,
    _both_dataset,
    _candidates,
    _config,
    _effective_config,
    _main_module_under_test,
    _run_stages,
    _sourcedata_dataset,
    _stop_config,
    _terminal,
    _two_outputs_dataset,
)


@pytest.fixture
def main_module(monkeypatch, tmp_path):
    return _main_module_under_test(monkeypatch, tmp_path)


# ====================== The input kind prompt ======================


@pytest.mark.unit
def test_both_kinds_prompt_shows_kinds_and_examples(main_module, monkeypatch, tmp_path):
    terminal = _terminal(monkeypatch, "t1")
    root = _both_dataset(tmp_path)
    scan = scan_input(str(root))
    _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root))
    assert terminal.stdin.reads, "no answer was read on a terminal for an input holding both kinds"
    for text in (*KINDS, scan.first_t1, scan.first_graph):
        assert text in terminal.written(), f"prompt lacks {text!r}: {terminal.written()!r}"


@pytest.mark.unit
def test_sourcedata_prompt_shows_first_sourcedata_t1(main_module, monkeypatch, tmp_path):
    terminal = _terminal(monkeypatch, "t1")
    root = _sourcedata_dataset(tmp_path)
    _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root))
    assert terminal.stdin.reads, "no answer was read on a terminal for sourcedata T1 files"
    for text in (*KINDS, sourcedata_t1_files(str(root))[0]):
        assert text in terminal.written(), f"prompt lacks {text!r}: {terminal.written()!r}"


@pytest.mark.unit
@pytest.mark.parametrize("dataset", ["both", "sourcedata"])
def test_prompt_answer_t1_runs_stage1_on_input(main_module, monkeypatch, tmp_path, dataset):
    terminal = _terminal(monkeypatch, "t1")
    root = DATASETS[dataset](tmp_path)
    status, argv = _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root))
    assert terminal.stdin.reads, f"no answer was read on a terminal for the {dataset} input"
    assert STAGE1 in argv, f"stage 1 did not run after the answer t1 (exit status {status})"
    assert argv[STAGE1][1] == str(root), f"stage 1 output root {argv[STAGE1][1]!r}"


@pytest.mark.unit
def test_prompt_answer_morphologist_skips_stage1(main_module, monkeypatch, tmp_path):
    _terminal(monkeypatch, "morphologist")
    root = _both_dataset(tmp_path)
    # A set datasets_root lets stage 1 pass its own validation: only the answer may keep it from running.
    status, argv = _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root, datasets_root=str(root)))
    assert status == 0, f"main() exit status {status}"
    assert STAGE1 not in argv, f"stage 1 ran after the answer morphologist with argv {argv.get(STAGE1)}"


@pytest.mark.unit
def test_prompt_answer_morphologist_stage2_reads_resolved_folder(main_module, monkeypatch, tmp_path):
    _terminal(monkeypatch, "morphologist")
    root = _both_dataset(tmp_path)
    status, argv = _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root))
    assert STAGE2 in argv, f"stage 2 did not run after the answer morphologist (exit status {status})"
    assert argv[STAGE2][0] == str(root.joinpath(*MORPHOLOGIST_6)), f"stage 2 subjects folder {argv[STAGE2][0]!r}"


@pytest.mark.unit
@pytest.mark.parametrize(("dataset", "answer"), [("both", "t1"), ("both", "morphologist"), ("sourcedata", "t1")])
def test_prompt_answer_is_kept_in_input_kind(main_module, monkeypatch, tmp_path, dataset, answer):
    _terminal(monkeypatch, answer)
    config = _config(main_module, tmp_path, DATASETS[dataset](tmp_path))
    assert _effective_config(main_module, monkeypatch, config).dataset.input_kind == answer


@pytest.mark.unit
@pytest.mark.parametrize(
    ("dataset", "answer"),
    [("both", UNKNOWN_KIND), ("sourcedata", UNKNOWN_KIND), ("two_outputs", UNLISTED_FOLDER)],
    ids=["kind_both", "kind_sourcedata", "subjects_folder"],
)
def test_unlisted_prompt_answer_exits_1(main_module, monkeypatch, tmp_path, dataset, answer):
    terminal = _terminal(monkeypatch, answer)
    config = _stop_config(main_module, tmp_path, DATASETS[dataset](tmp_path))
    status, argv = _run_stages(main_module, monkeypatch, config)
    assert terminal.stdin.reads, "no answer was read on a terminal"
    _assert_stopped(status, argv)


@pytest.mark.unit
@pytest.mark.parametrize("dataset", ["both", "sourcedata", "two_outputs"])
def test_end_of_input_at_prompt_exits_1(main_module, monkeypatch, tmp_path, dataset):
    terminal = _terminal(monkeypatch)
    config = _stop_config(main_module, tmp_path, DATASETS[dataset](tmp_path))
    status, argv = _run_stages(main_module, monkeypatch, config)
    assert terminal.stdin.reads, "no answer was read on a terminal"
    _assert_stopped(status, argv)


# ====================== Two Morphologist outputs ======================


@pytest.mark.unit
def test_subjects_folder_prompt_lists_folders(main_module, monkeypatch, tmp_path):
    root = _two_outputs_dataset(tmp_path)
    candidates = _candidates(root)
    terminal = _terminal(monkeypatch, candidates[1])
    _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root))
    assert terminal.stdin.reads, "no answer was read on a terminal for two subjects folders"
    for folder in candidates:
        assert folder in terminal.written(), f"prompt lacks {folder!r}: {terminal.written()!r}"


@pytest.mark.unit
@pytest.mark.parametrize("choice", [0, 1], ids=["first_listed", "second_listed"])
def test_subjects_folder_answer_feeds_stage2(main_module, monkeypatch, tmp_path, choice):
    root = _two_outputs_dataset(tmp_path)
    chosen = _candidates(root)[choice]
    _terminal(monkeypatch, chosen)
    status, argv = _run_stages(main_module, monkeypatch, _config(main_module, tmp_path, root))
    assert STAGE2 in argv, f"stage 2 did not run after a listed subjects folder was chosen (exit status {status})"
    assert argv[STAGE2][0] == os.path.abspath(chosen), f"stage 2 subjects folder {argv[STAGE2][0]!r}"


@pytest.mark.unit
def test_subjects_folder_answer_is_kept_in_morphologist_graphs(main_module, monkeypatch, tmp_path):
    root = _two_outputs_dataset(tmp_path)
    chosen = _candidates(root)[1]
    _terminal(monkeypatch, chosen)
    effective = _effective_config(main_module, monkeypatch, _config(main_module, tmp_path, root))
    assert effective.dataset.morphologist_graphs == os.path.abspath(chosen)
