"""Input kind precedence: a .arg file anywhere below the input is a Morphologist graph, T1 files count without one.

Task TASK-ARGDISC-BDRABCZUK-7F0CEB924F51 (epic TASK-ARGDISC-BDRABCZUK-D91B3C7ACD31). The user's rule, recorded
verbatim as REQ-ARGDISC-BDRABCZUK-E22C25537224: "find .arg files if any, else .nii.gz files if any". The folder
named t1mri no longer decides what a graph is; dot folders stay out. A tree holding graphs and T1 files still reads
both (user answer Q4), and the walk goes past the first T1 file to find a later graph (answer Q5: full depth-first
walk, names only).

Layouts come from tests/morphologist_layout_fixtures.py (names only, no data). Requirements:

* REQ-ARGDISC-BDRABCZUK-E22C25537224 (user rule) - test_classic_tree_reads_morphologist,
  test_morphologist_cli_tree_reads_morphologist, test_senior_tree_reads_morphologist, test_raw_t1_bids_tree_reads_t1
* REQ-ARGDISC-BDRABCZUK-24DD7014356A (.arg anywhere is a graph) - test_arg_outside_t1mri_is_a_graph,
  test_mixed_tree_reads_both, test_scan_names_first_graph_outside_t1mri
* REQ-ARGDISC-BDRABCZUK-6731DADF9510 (.arg at any depth without T1 reads morphologist) -
  test_arg_in_plain_folder_reads_morphologist, test_morphologist_cli_tree_reads_morphologist
* REQ-ARGDISC-BDRABCZUK-80E144705B28 (graph after the first T1 reads both) - test_graph_after_first_t1_reads_both
* REQ-ARGDISC-BDRABCZUK-33275E72AB55 (no folder read after the first T1 and the first graph) -
  test_nothing_is_read_after_first_t1_and_first_graph
* REQ-ARGDISC-BDRABCZUK-1786F42F6CE1 (no .arg: .nii.gz files below t1mri are T1 input, derived volumes left out;
  user answer 2026-11-25) - test_t1mri_tree_without_arg_reads_t1, test_t1mri_tree_without_arg_lists_raw_t1_files,
  test_t1mri_plain_tree_without_arg_lists_raw_t1_files, test_t1mri_tree_without_arg_runs_stage_1; reworded on
  user answer Q1 2026-11-25 (a folder named derivatives or qc keeps its files out) -
  test_t1mri_below_derivatives_or_qc_without_arg_is_no_t1, test_senior_tree_without_arg_reads_by_input_folder
* REQ-INPUTKIND-BDRABCZUK-AE205BE00702 (a sub-* file without _T1w is no T1 file, outside t1mri only; user answer
  Q2 2026-11-25) - test_bids_named_raw_copy_below_t1mri_without_arg_is_t1,
  test_t1w_rule_applies_outside_t1mri_only
* REQ-INPUTKIND-BDRABCZUK-3B0E2458A77E (with a .arg, the T1 list leaves out t1mri) -
  test_classic_tree_lists_no_t1_file
"""

import os
from pathlib import Path

import pytest

from champollion_pipeline.utils.input_kind import InputKind, detect_input_kind, find_t1_files, scan_input
from tests.input_kind_fixtures import header_reads, main_module, read_folders  # noqa: F401 - fixtures
from tests.main_input_autodetect_harness import STAGE1, _batch, _run_stages, _stop_config
from tests.morphologist_layout_fixtures import (
    CLASSIC_SUBJECTS,
    SENIOR_ACQUISITIONS,
    build_classic_tree,
    build_mixed_tree,
    build_morphologist_cli_tree,
    build_raw_t1_bids_tree,
    build_senior_tree,
)

CLI_FIRST_GRAPH = "sub-01/ses-1/run-0/ana-0/folds/3.1/sub-01_ses-1_run-0_hemi-L.arg"
CLASSIC_GRAPH = "zz/t1mri/default_acquisition/default_analysis/folds/3.1/Lzz.arg"


def _touch(root: Path, relative: str) -> Path:
    """Create an empty file at ``root/relative`` with its parent folders."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


# ====================== Acceptance layouts (user rule) ======================


@pytest.mark.parametrize("relative", ["", "IXI001-Guys-0001"])
def test_classic_tree_reads_morphologist(tmp_path, relative):
    root = build_classic_tree(tmp_path)

    assert detect_input_kind(str(root / relative)) is InputKind.MORPHOLOGIST


@pytest.mark.parametrize("relative", ["", "sub-01", "sub-02"])
def test_morphologist_cli_tree_reads_morphologist(tmp_path, relative):
    root = build_morphologist_cli_tree(tmp_path)

    assert detect_input_kind(str(root / relative)) is InputKind.MORPHOLOGIST


@pytest.mark.parametrize("relative", ["", "derivatives", "derivatives/subjects_f"])
def test_senior_tree_reads_morphologist(tmp_path, relative):
    root = build_senior_tree(tmp_path)

    assert detect_input_kind(str(root / relative)) is InputKind.MORPHOLOGIST


def test_raw_t1_bids_tree_reads_t1(tmp_path):
    root = build_raw_t1_bids_tree(tmp_path)

    assert detect_input_kind(str(root)) is InputKind.T1


# ====================== A .arg file anywhere is a graph ======================


def test_arg_outside_t1mri_is_a_graph(tmp_path):
    _touch(tmp_path, "sub-01/graphs/Lsub-01.arg")
    _touch(tmp_path, "sub-01.nii.gz")

    scan = scan_input(str(tmp_path))

    assert scan.first_graph == "sub-01/graphs/Lsub-01.arg"
    assert scan.kind is InputKind.BOTH


def test_mixed_tree_reads_both(tmp_path):
    root = build_mixed_tree(tmp_path)

    assert detect_input_kind(str(root)) is InputKind.BOTH


def test_scan_names_first_graph_outside_t1mri(tmp_path):
    root = build_morphologist_cli_tree(tmp_path)

    assert scan_input(str(root)).first_graph == CLI_FIRST_GRAPH


@pytest.mark.parametrize(
    "relative",
    ["Lsub-01.arg", "graphs/Lsub-01.arg", "sub-01/ses-1/run-0/ana-0/folds/3.1/sub-01_ses-1_run-0_hemi-L.arg"],
)
def test_arg_in_plain_folder_reads_morphologist(tmp_path, relative):
    _touch(tmp_path, relative)

    assert detect_input_kind(str(tmp_path)) is InputKind.MORPHOLOGIST


# ====================== The walk goes past the first T1 file ======================


@pytest.mark.parametrize(
    ("t1", "graph"),
    [
        ("a.nii.gz", CLASSIC_GRAPH),
        ("data/IXI002.nii.gz", CLASSIC_GRAPH),
        ("a.nii.gz", f"z/derivatives/morphologist-6.0/{CLASSIC_GRAPH}"),
        ("a.nii.gz", "zz/ses-1/run-0/ana-0/folds/3.1/zz_ses-1_run-0_hemi-L.arg"),
    ],
)
def test_graph_after_first_t1_reads_both(tmp_path, t1, graph):
    _touch(tmp_path, t1)
    _touch(tmp_path, graph)

    scan = scan_input(str(tmp_path))

    assert (scan.first_t1, scan.first_graph) == (t1, graph)
    assert detect_input_kind(str(tmp_path)) is InputKind.BOTH


def test_nothing_is_read_after_first_t1_and_first_graph(tmp_path, read_folders):  # noqa: F811 - imported fixture
    for relative in ("a/b/IXI002.nii.gz", "a/c/IXI012.nii.gz", "m/Lm.arg", "z/IXI013.nii.gz"):
        _touch(tmp_path, relative)

    assert detect_input_kind(str(tmp_path)) is InputKind.BOTH
    assert [os.path.relpath(folder, tmp_path) for folder in read_folders] == [".", "a", "a/b", "a/c", "m"]


# ====================== No .arg file: .nii.gz files below t1mri are T1 input ======================


def _classic_tree_without_graphs(root: Path) -> Path:
    """The classic BrainVISA tree with every .arg graph and its .data folder removed."""
    build_classic_tree(root)
    for graph in sorted(root.rglob("*.arg")):
        graph.unlink()
        graph.with_suffix(".data").rmdir()
    return root


def _raw_copy(subject: str) -> str:
    """Path, relative to the subject folder, of the raw T1 copy Morphologist keeps in a classic tree."""
    return f"t1mri/default_acquisition/{subject}.nii.gz"


@pytest.mark.parametrize("relative", ["", CLASSIC_SUBJECTS[0]])
def test_t1mri_tree_without_arg_reads_t1(tmp_path, relative):
    root = _classic_tree_without_graphs(tmp_path / "in")

    assert detect_input_kind(str(root / relative)) is InputKind.T1


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        ("", [f"{subject}/{_raw_copy(subject)}" for subject in CLASSIC_SUBJECTS]),
        (CLASSIC_SUBJECTS[0], [_raw_copy(CLASSIC_SUBJECTS[0])]),
    ],
)
def test_t1mri_tree_without_arg_lists_raw_t1_files(tmp_path, relative, expected):
    # The raw copies are T1 input; nobias_* and the segmentation volumes are derived volumes, left out
    root = _classic_tree_without_graphs(tmp_path / "in")

    assert find_t1_files(str(root / relative)) == expected


@pytest.mark.parametrize(
    "names",
    [
        ("s01/t1mri/s01.nii.gz", "s01/t1mri/segmentation/brain_s01.nii.gz"),
        ("s01/t1mri/acq/s01.nii", "s01/t1mri/acq/s01_skeleton.nii.gz"),
    ],
)
def test_t1mri_plain_tree_without_arg_lists_raw_t1_files(tmp_path, names):
    for name in names:
        _touch(tmp_path, name)

    assert find_t1_files(str(tmp_path)) == [names[0]]
    assert detect_input_kind(str(tmp_path)) is InputKind.T1


def test_t1mri_tree_without_arg_runs_stage_1(main_module, monkeypatch, tmp_path, header_reads):  # noqa: F811 - imported fixture
    _batch(monkeypatch)
    root = _classic_tree_without_graphs(tmp_path / "in")

    status, argv = _run_stages(main_module, monkeypatch, _stop_config(main_module, tmp_path, root))

    assert status != 2, "main() stopped with exit status 2 on .nii.gz files below t1mri with no .arg file"
    assert STAGE1 in argv, f"stage 1 did not run on the T1 files below t1mri (exit status {status})"


@pytest.mark.parametrize("relative", ["", CLASSIC_SUBJECTS[0]])
def test_classic_tree_lists_no_t1_file(tmp_path, relative):
    root = build_classic_tree(tmp_path)

    assert find_t1_files(str(root / relative)) == []


# ====================== No .arg file: derivatives and qc keep their files out (user answer Q1) ======================


def _senior_tree_without_graphs(root: Path) -> Path:
    """The SENIOR tree with every .arg graph, its .data folder and its .minf sibling removed."""
    build_senior_tree(root)
    for graph in sorted(root.rglob("*.arg")):
        graph.unlink()
        graph.with_suffix(".data").rmdir()
        graph.with_name(f"{graph.name}.minf").unlink()
    return root


@pytest.mark.parametrize(
    "raw_copy",
    [
        "derivatives/subjects_f/sf0001/t1mri/3TV0/sf0001.nii.gz",
        "derivatives/morphologist/sub-01/t1mri/default_acquisition/sub-01.nii.gz",
        "qc/sub-01/t1mri/default_acquisition/sub-01.nii.gz",
    ],
)
def test_t1mri_below_derivatives_or_qc_without_arg_is_no_t1(tmp_path, raw_copy):
    _touch(tmp_path, raw_copy)

    assert find_t1_files(str(tmp_path)) == []
    assert detect_input_kind(str(tmp_path)) is InputKind.NEITHER


@pytest.mark.parametrize(
    ("relative", "subjects"),
    [
        ("", ()),
        ("derivatives", (("subjects_f/sf0001", "sf0001"), ("subjects_h/sh0001", "sh0001"))),
        ("derivatives/subjects_f", (("sf0001", "sf0001"),)),
        ("derivatives/subjects_f/sf0001", (("", "sf0001"),)),
    ],
)
def test_senior_tree_without_arg_reads_by_input_folder(tmp_path, relative, subjects):
    # A folder named derivatives below the input folder keeps the raw copies out; from derivatives down, none is left
    root = _senior_tree_without_graphs(tmp_path / "in")
    expected = sorted(
        os.path.join(folder, "t1mri", acquisition, f"{subject}.nii.gz")
        for folder, subject in subjects
        for acquisition in SENIOR_ACQUISITIONS
    )

    assert find_t1_files(str(root / relative)) == expected
    assert detect_input_kind(str(root / relative)) is (InputKind.T1 if expected else InputKind.NEITHER)


# ====================== No .arg file: the _T1w rule stays outside t1mri (user answer Q2) ======================


BRAINVISA_RAW_COPY = "sub-01/t1mri/default_acquisition/sub-01.nii.gz"


@pytest.mark.parametrize(("relative", "expected"), [("", BRAINVISA_RAW_COPY), ("sub-01", BRAINVISA_RAW_COPY[7:])])
def test_bids_named_raw_copy_below_t1mri_without_arg_is_t1(tmp_path, relative, expected):
    # Morphologist names its raw copy after the subject, without _T1w; the derived nobias volume stays out
    _touch(tmp_path, BRAINVISA_RAW_COPY)
    _touch(tmp_path, "sub-01/t1mri/default_acquisition/default_analysis/nobias_sub-01.nii.gz")

    assert find_t1_files(str(tmp_path / relative)) == [expected]
    assert detect_input_kind(str(tmp_path / relative)) is InputKind.T1


def test_t1w_rule_applies_outside_t1mri_only(tmp_path):
    _touch(tmp_path, BRAINVISA_RAW_COPY)
    _touch(tmp_path, "sub-02/anat/sub-02.nii.gz")

    assert find_t1_files(str(tmp_path)) == [BRAINVISA_RAW_COPY]
