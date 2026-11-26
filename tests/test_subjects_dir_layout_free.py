"""Subjects-folder resolution without a t1mri folder (TASK-ARGDISC-BDRABCZUK-5ABBA33DB536).

resolve_subjects_dir takes the subject folders of a tree laid out by morphologist-cli from its sub- folders, the run
hands that folder to the stages, a .arg file loose in a non-sub- input stops the run with exit status 2, and a folder
named qc is a subject only when it holds a .arg file. Synthetic trees only: tests/morphologist_layout_fixtures.py.

The modules under test are imported inside each test, so a missing name fails each test on its own.
Requirements: cited in each test (alm links on the Task root).
"""

import dataclasses
import importlib
import os
from pathlib import Path

import pytest
import yaml

from tests.main_input_autodetect_harness import (
    STAGE2,
    _assert_error_contains,
    _assert_stopped,
    _batch,
    _effective_config,
    _main_module_under_test,
    _run_stages,
    _stop_config,
)
from tests.morphologist_layout_fixtures import (
    CLI_SUBJECTS,
    SENIOR_SUBJECTS,
    build_classic_tree,
    build_morphologist_cli_tree,
    build_senior_tree,
)

RESOLVER_MODULE = "champollion_pipeline.utils.input_kind"
RULE_MODULE = "champollion_pipeline.utils.subject_folders"
GRAPH_ITEM_MODULE = "champollion_pipeline.utils.graph_item"
CLASSIC_GRAPH_FOLDER = "t1mri/default_acquisition/default_analysis/folds/3.1"
CLI_GRAPH_FOLDER = "ses-1/run-0/ana-0/folds/3.1"
LOOSE_GRAPH = "Lloose.arg"


def _resolve(path: Path) -> str | None:
    """Return what resolve_subjects_dir gives for ``path``."""
    return importlib.import_module(RESOLVER_MODULE).resolve_subjects_dir(str(path))


def _list_subjects(subjects_dir: Path) -> list[str]:
    """Return what list_morphologist_subjects gives for ``subjects_dir``, called with a str path."""
    return importlib.import_module(RULE_MODULE).list_morphologist_subjects(str(subjects_dir))


def _touch(root: Path, relative: str) -> Path:
    """Create the empty file ``relative`` below ``root``, with its folders."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _plain_cli_tree(root: Path) -> Path:
    """morphologist-cli folders under subject names that do not start with sub-, and no t1mri folder."""
    for subject in ("IXI001", "IXI002"):
        for side in ("L", "R"):
            _touch(root, f"{subject}/{CLI_GRAPH_FOLDER}/{subject}_ses-1_run-0_hemi-{side}.arg")
    return root


def _loose_graph_input(root: Path) -> Path:
    """A non-sub- folder holding a morphologist-cli tree and one .arg file directly in it."""
    input_dir = build_morphologist_cli_tree(root / "study")
    _touch(input_dir, LOOSE_GRAPH)
    return input_dir


@pytest.fixture
def main_module(monkeypatch, tmp_path):
    return _main_module_under_test(monkeypatch, tmp_path)


# ====================== resolve_subjects_dir ======================


@pytest.mark.unit
@pytest.mark.parametrize("below_input", ["", "derivatives/morphologist-cli"], ids=["input_is_root", "root_below"])
def test_cli_tree_resolves_to_the_parent_of_its_sub_folders(tmp_path, below_input):
    # REQ-ARGDISC-BDRABCZUK-EE24FF19000D: the parent of the nearest sub- folders is the subjects folder,
    # no t1mri folder needed
    root = build_morphologist_cli_tree(tmp_path / below_input if below_input else tmp_path)
    assert _resolve(tmp_path) == str(root)


@pytest.mark.unit
@pytest.mark.parametrize(
    "graph",
    [f"{CLI_GRAPH_FOLDER}/sub-01_ses-1_run-0_hemi-L.arg", f"anat/{CLASSIC_GRAPH_FOLDER}/Lsub-01.arg"],
    ids=["cli_layout", "t1mri_below_another_folder"],
)
def test_sub_named_input_is_one_subject(tmp_path, graph):
    # REQ-ARGDISC-BDRABCZUK-075FFA4ACE8F: an input named sub-* with no sub- folder below is itself the subject
    subject = tmp_path / "sub-01"
    _touch(subject, graph)
    assert _resolve(subject) is None


@pytest.mark.unit
def test_graphs_without_sub_or_t1mri_folders_resolve_to_the_input(tmp_path):
    # REQ-ARGDISC-BDRABCZUK-C916E5C55893: the subject folders, named after their graphs, share the input as parent
    assert _resolve(_plain_cli_tree(tmp_path)) == str(tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("graph", "subjects_folder"),
    [
        pytest.param(f"data/IXI002/{CLI_GRAPH_FOLDER}/IXI002_ses-1_run-0_hemi-L.arg", "data", id="no_side_letter"),
        pytest.param("data/IXI002/anat/folds/RIXI002.arg", "data", id="side_letter_then_dot"),
        pytest.param(
            "data/IXI002/folds/3.1/sul-0_auto/LIXI002_sul-0_auto.arg", "data", id="side_letter_then_underscore"
        ),
        pytest.param("data/IXI002/copy/IXI002/folds/LIXI002.arg", "data/IXI002/copy", id="nearest_named_folder"),
        pytest.param("data/IXI00/IXI002/folds/LIXI002.arg", "data/IXI00", id="name_without_separator_is_no_match"),
    ],
)
def test_subject_folder_is_the_nearest_folder_named_after_the_graph(tmp_path, graph, subjects_folder):
    # REQ-ARGDISC-BDRABCZUK-D8BACBE58917: the subject folder of a graph is the nearest folder whose name N starts
    # the graph file name as N, LN or RN followed by a dot or an underscore; no t1mri folder needed
    input_dir = tmp_path / "study"
    _touch(input_dir, graph)
    assert _resolve(input_dir) == str(input_dir / subjects_folder)


@pytest.mark.unit
@pytest.mark.parametrize(
    "graph",
    [
        pytest.param("IXI002/other/Lgraph.arg", id="other_names"),
        pytest.param(f"derivatives/morphologist-6.0/IXI002/{CLASSIC_GRAPH_FOLDER}/Lgraph.arg", id="classic_tree"),
    ],
)
def test_graph_named_after_no_folder_takes_the_child_of_the_input(tmp_path, graph):
    # REQ-ARGDISC-BDRABCZUK-6ED8713C70C3: no folder is named after the graph, so the folder directly in the input
    # is its subject folder (the user's rule of 2026-11-26) and the input is the subjects folder
    input_dir = tmp_path / "study"
    _touch(input_dir, graph)
    assert _resolve(input_dir) == str(input_dir)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("graphs", "input_below"),
    [
        pytest.param(
            (
                f"IXI002/{CLI_GRAPH_FOLDER}/IXI002_ses-1_run-0_hemi-L.arg",
                f"IXI002/backup/s1/{CLASSIC_GRAPH_FOLDER}/Ls1.arg",
            ),
            "IXI002",
            id="cli_layout_with_a_subject_beside",
        ),
        pytest.param(("IXI002/folds/3.1/sul-0_auto/LIXI002_sul-0_auto.arg",), "IXI002", id="labelled_graph_only"),
    ],
)
def test_input_named_after_its_graph_is_one_subject(tmp_path, graphs, input_below):
    # REQ-ARGDISC-BDRABCZUK-1E1A74720545: the input is the subject folder of a graph, so it is one subject, whatever
    # other graphs sit below it
    for graph in graphs:
        _touch(tmp_path, graph)
    assert _resolve(tmp_path / input_below) is None


@pytest.mark.unit
@pytest.mark.parametrize(("group", "subject"), SENIOR_SUBJECTS, ids=[group for group, _ in SENIOR_SUBJECTS])
def test_senior_subject_is_one_subject(tmp_path, group, subject):
    # REQ-ARGDISC-BDRABCZUK-1E1A74720545: a SENIOR subject folder, several acquisitions and sessions below it
    build_senior_tree(tmp_path)
    assert _resolve(tmp_path / "derivatives" / group / subject) is None


@pytest.mark.unit
def test_tree_without_t1mri_gives_the_parent_of_its_subject_folders(tmp_path):
    # REQ-ARGDISC-BDRABCZUK-C916E5C55893: a dataset root whose subjects have no t1mri folder
    subjects_dir = _plain_cli_tree(tmp_path / "derivatives" / "morphologist-cli")
    assert _resolve(tmp_path) == str(subjects_dir)


@pytest.mark.unit
@pytest.mark.parametrize("group", [group for group, _ in SENIOR_SUBJECTS])
def test_senior_subjects_folder_gives_itself(tmp_path, group):
    # REQ-ARGDISC-BDRABCZUK-C916E5C55893: each SENIOR subjects folder resolves to itself
    build_senior_tree(tmp_path)
    assert _resolve(tmp_path / "derivatives" / group) == str(tmp_path / "derivatives" / group)


@pytest.mark.unit
@pytest.mark.parametrize("input_below", ["", "derivatives"], ids=["dataset_root", "derivatives"])
def test_senior_derivatives_folder_is_ambiguous(tmp_path, input_below):
    # REQ-ARGDISC-BDRABCZUK-C916E5C55893 needs one shared parent; SENIOR has two subjects folders, an ambiguity
    # (REQ-INPUTKIND-BDRABCZUK-313AEEB1E572) naming both
    build_senior_tree(tmp_path)
    ambiguous = importlib.import_module(RESOLVER_MODULE).AmbiguousSubjectsFolders
    with pytest.raises(ambiguous) as raised:
        _resolve(tmp_path / input_below)
    assert raised.value.candidates == tuple(str(tmp_path / "derivatives" / group) for group, _ in SENIOR_SUBJECTS)


@pytest.mark.unit
def test_loose_graph_in_non_sub_input_is_refused(tmp_path):
    # REQ-ARGDISC-BDRABCZUK-F17ABC006F18: a .arg file directly in a non-sub- input is refused, naming the file
    input_dir = _loose_graph_input(tmp_path)
    loose_graph_error = importlib.import_module(GRAPH_ITEM_MODULE).LooseGraphError
    with pytest.raises(loose_graph_error) as raised:
        _resolve(input_dir)
    assert raised.value.path == os.path.join(str(input_dir), LOOSE_GRAPH)


# ====================== The run ======================


def _assert_exit_before_any_stage(main_module, monkeypatch, tmp_path, status: int) -> None:
    """main() on a loose-graph input exits with `status` before any stage."""
    _batch(monkeypatch)
    config = _stop_config(main_module, tmp_path, _loose_graph_input(tmp_path))
    result = _run_stages(main_module, monkeypatch, config)
    _assert_stopped(*result, expected=status)


@pytest.mark.unit
def test_loose_graph_stops_the_pipeline_with_exit_2(main_module, monkeypatch, tmp_path):
    # REQ-ARGDISC-BDRABCZUK-4BE8B29338A4: exit status 2 before any stage runs
    _assert_exit_before_any_stage(main_module, monkeypatch, tmp_path, 2)


@pytest.mark.unit
def test_loose_graph_stop_logs_the_file(main_module, monkeypatch, tmp_path, pipeline_caplog):
    # REQ-ARGDISC-BDRABCZUK-DC5F23E63C24: one ERROR record names the loose .arg file
    _batch(monkeypatch)
    input_dir = _loose_graph_input(tmp_path)
    _run_stages(main_module, monkeypatch, _stop_config(main_module, tmp_path, input_dir))
    _assert_error_contains(pipeline_caplog, os.path.join(str(input_dir), LOOSE_GRAPH))


@pytest.mark.unit
def test_cli_tree_runs_with_its_root_as_subjects_folder(main_module, monkeypatch, tmp_path):
    # REQ-ARGDISC-BDRABCZUK-C19172678F1B: the stage 1 to stage 2 chain: dataset.morphologist_graphs is the
    # morphologist-cli root
    _batch(monkeypatch)
    root = build_morphologist_cli_tree(tmp_path / "cli")
    content = dataclasses.asdict(_stop_config(main_module, tmp_path, root))
    replay = tmp_path / "replay.yaml"
    replay.write_text(yaml.safe_dump(content))
    config = _effective_config(main_module, monkeypatch, None, "--config", str(replay))
    assert config.dataset.morphologist_graphs == os.path.abspath(root), (
        f"{STAGE2} subjects folder {config.dataset.morphologist_graphs!r}"
    )


# ====================== The qc folder rule ======================


@pytest.mark.unit
@pytest.mark.parametrize(
    "graph",
    ["qc/reports/Lqc.arg", f"qc/{CLASSIC_GRAPH_FOLDER}/Lqc.arg", f"qc/{CLI_GRAPH_FOLDER}/qc_ses-1_run-0_hemi-L.arg"],
    ids=["no_t1mri", "classic_layout", "cli_layout"],
)
def test_qc_folder_holding_a_graph_is_listed(tmp_path, graph):
    # REQ-ARGDISC-BDRABCZUK-372F95A73053: qc is a subject when it holds a .arg file, whatever the folder names below it
    subjects_dir = build_classic_tree(tmp_path)
    _touch(subjects_dir, graph)
    assert "qc" in _list_subjects(subjects_dir)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("build", "qc_content"),
    [
        (build_classic_tree, "qc/t1mri/default_acquisition/"),
        (build_classic_tree, "qc/ses-1/t1mri/default_acquisition/"),
        (build_classic_tree, "qc/t1mri/default_acquisition/qc.nii.gz"),
        (build_morphologist_cli_tree, "qc/qc.tsv"),
        (build_morphologist_cli_tree, "qc/.hidden/Lqc.arg"),
    ],
    ids=["classic_t1mri_folder", "classic_session_t1mri_folder", "classic_t1mri_volume", "cli_report", "dot_folder"],
)
def test_qc_folder_holding_no_graph_is_left_out(tmp_path, build, qc_content):
    # REQ-ARGDISC-BDRABCZUK-8A14A3246F95: qc holding no .arg file (outside dot folders) is no subject,
    # a t1mri folder inside it or not
    subjects_dir = build(tmp_path)
    if qc_content.endswith("/"):
        (subjects_dir / qc_content).mkdir(parents=True, exist_ok=True)
    else:
        _touch(subjects_dir, qc_content)
    assert "qc" not in _list_subjects(subjects_dir)


@pytest.mark.unit
def test_cli_subject_folders_are_listed(tmp_path):
    # REQ-ARGDISC-BDRABCZUK-641010951F00: each non-dot, non-qc folder holding a .arg file is listed,
    # no t1mri folder needed
    subjects_dir = build_morphologist_cli_tree(tmp_path)
    assert _list_subjects(subjects_dir) == list(CLI_SUBJECTS)
