"""Labelling QC finds each subject's graph folder through graph discovery (TASK-ARGDISC-BDRABCZUK-698DE44C8B80).

Every check builds a synthetic folder tree (names only) in ``tmp_path``; no graph is read.

Requirements (ids in the ledger, titles here):
    discovery    without --path_to_graph, the graph folder of a subject comes from graph discovery, any layout
    first only   a subject with several graph folders uses the first in natural order (temporary rule)
    override     --path_to_graph given: used as before
    loose graph  a .arg directly in a non-sub- subject folder stops the QC with exit status 2
"""

import csv
import logging
from pathlib import Path

from champollion_pipeline.generate_labelling_qc import GenerateLabellingQC, find_graph_dir
from tests.morphologist_layout_fixtures import build_classic_tree, build_morphologist_cli_tree

CLI_SET = "sul-0_auto"


def _run(subjects: Path, output: Path, *extra: str) -> tuple[int, list[dict]]:
    script = GenerateLabellingQC()
    script.parse_args([str(subjects), str(output), *extra])
    code = script.run()
    rows = []
    if output.exists():
        with open(output, newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
    return code, rows


def _row(rows: list[dict], subject: str) -> dict:
    return next(r for r in rows if r["participant_id"] == subject)


def _touch(root: Path, *names: str) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_classic_layout_found_without_path_to_graph(tmp_path):
    subjects = build_classic_tree(tmp_path / "subjects")
    code, rows = _run(subjects, tmp_path / "qc.tsv")
    assert code == 0
    assert [(r["participant_id"], r["qc"], r["labelled_L"], r["labelled_R"], r["reason"]) for r in rows] == [
        ("IXI001-Guys-0001", "1", "True", "True", ""),
        ("IXI002-HH-0002", "1", "True", "True", ""),
    ]


def test_morphologist_cli_layout_without_t1mri_found_without_path_to_graph(tmp_path):
    subjects = build_morphologist_cli_tree(tmp_path / "subjects")
    code, rows = _run(subjects, tmp_path / "qc.tsv", "--labelling_session", CLI_SET)
    assert code == 0
    assert [r["participant_id"] for r in rows] == ["sub-01", "sub-02"]
    assert _row(rows, "sub-01")["qc"] == "1"
    assert (_row(rows, "sub-02")["qc"], _row(rows, "sub-02")["reason"]) == ("0", "no_session_dir")


def test_cli_layout_one_hemisphere_labelled_names_the_missing_side(tmp_path):
    subjects = tmp_path / "subjects"
    _touch(subjects, f"sub-01/ses-1/run-0/ana-0/folds/3.1/{CLI_SET}/sub-01_ses-1_run-0_sul-0_hemi-L_auto.arg")
    _, rows = _run(subjects, tmp_path / "qc.tsv", "--labelling_session", CLI_SET)
    assert (_row(rows, "sub-01")["qc"], _row(rows, "sub-01")["reason"]) == ("0", "missing_R")


def test_several_graph_folders_use_the_first_in_natural_order(tmp_path):
    subjects = tmp_path / "subjects"
    both = ("L", "R")
    for acquisition, sides in (("3TV10", ("R",)), ("3TV2", both)):
        for side in sides:
            _touch(
                subjects,
                f"s1/t1mri/{acquisition}/default_analysis/folds/3.1/default_session_auto/{side}s1_default_session_auto.arg",
            )
    _, rows = _run(subjects, tmp_path / "qc.tsv")
    assert _row(rows, "s1")["qc"] == "1"  # 3TV2 before 3TV10; 3TV10 alone lacks L


def test_first_graph_folder_decides_even_when_a_later_one_is_complete(tmp_path):
    subjects = tmp_path / "subjects"
    for acquisition, sides in (("3TV2", ("R",)), ("3TV10", ("L", "R"))):
        for side in sides:
            _touch(
                subjects,
                f"s1/t1mri/{acquisition}/default_analysis/folds/3.1/default_session_auto/{side}s1_default_session_auto.arg",
            )
    _, rows = _run(subjects, tmp_path / "qc.tsv")
    assert (_row(rows, "s1")["qc"], _row(rows, "s1")["reason"]) == ("0", "missing_L")


def test_path_to_graph_given_overrides_discovery(tmp_path):
    subjects = build_classic_tree(tmp_path / "subjects")
    _, rows = _run(subjects, tmp_path / "qc.tsv", "--path_to_graph", "t1mri/nowhere")
    assert {r["reason"] for r in rows} == {"no_session_dir"}


def test_find_graph_dir_without_path_to_graph_is_the_discovered_folder(tmp_path):
    subjects = build_classic_tree(tmp_path / "subjects")
    subject_dir = subjects / "IXI001-Guys-0001"
    expected = subject_dir / "t1mri" / "default_acquisition" / "default_analysis" / "folds" / "3.1"
    assert Path(find_graph_dir(str(subject_dir), None, False)) == expected


def test_find_graph_dir_without_any_graph_is_none(tmp_path):
    (tmp_path / "s1" / "t1mri").mkdir(parents=True)
    assert find_graph_dir(str(tmp_path / "s1"), None, False) is None


def test_loose_graph_in_non_sub_subject_folder_stops_with_exit_2(tmp_path, caplog):
    subjects = tmp_path / "subjects"
    _touch(subjects, "s1/Ls1.arg")
    with caplog.at_level(logging.ERROR):
        code, _ = _run(subjects, tmp_path / "qc.tsv")
    assert code == 2
    assert str(subjects / "s1" / "Ls1.arg") in caplog.text


def test_loose_graph_in_sub_subject_folder_is_not_refused(tmp_path):
    subjects = tmp_path / "subjects"
    _touch(subjects, "sub-03/Lsub-03.arg")
    code, rows = _run(subjects, tmp_path / "qc.tsv")
    assert code == 1  # no subject passed
    assert _row(rows, "sub-03")["reason"] == "no_session_dir"
