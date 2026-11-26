"""generate_labelling_qc lists subjects through the shared subject-folder rule.

Requirements (INPUTKIND, recorded by the red stage of the labelling QC listing Task):
    one row per folder that list_morphologist_subjects returns, in that order
    no row for a qc folder that holds no .arg file outside dot folders (REQ-INPUTKIND-BDRABCZUK-B08D813F2622, live)
    a row for a qc folder that holds a .arg file outside dot folders (REQ-INPUTKIND-BDRABCZUK-60181CA6AE49, live)
    no row for a dot-folder
    a row for a folder named ra

The qc rule is layout-free since TASK-ARGDISC-BDRABCZUK-5ABBA33DB536: a graph, not a t1mri folder, makes qc a subject.
"""

from champollion_pipeline.utils.subject_folders import list_morphologist_subjects
from tests.test_generate_labelling_qc import _make_subject, _run, output, subjects  # noqa: F401 - fixtures


def test_rows_are_the_folders_of_the_shared_rule_in_its_order(subjects, output):  # noqa: F811
    for name in ("s03", "s01", "s02"):
        _make_subject(subjects, name)
    (subjects / "qc").mkdir()
    (subjects / "notes.txt").write_text("x")
    rows = _run(subjects, output)
    assert [r["participant_id"] for r in rows] == list_morphologist_subjects(subjects) == ["s01", "s02", "s03"]


def test_qc_folder_without_t1mri_is_no_row(subjects, output):  # noqa: F811
    _make_subject(subjects, "s01")
    (subjects / "qc").mkdir()
    (subjects / "qc" / "report.tsv").write_text("x")
    rows = _run(subjects, output)
    assert [r["participant_id"] for r in rows] == ["s01"]


def test_qc_folder_with_t1mri_is_a_row(subjects, output):  # noqa: F811
    _make_subject(subjects, "qc")
    _make_subject(subjects, "s01")
    rows = _run(subjects, output)
    assert [r["participant_id"] for r in rows] == ["qc", "s01"]


def test_qc_folder_with_t1mri_and_no_graph_is_no_row(subjects, output):  # noqa: F811
    _make_subject(subjects, "s01")
    _make_subject(subjects, "qc", sides=())
    rows = _run(subjects, output)
    assert [r["participant_id"] for r in rows] == ["s01"]


def test_qc_folder_with_a_graph_outside_t1mri_is_a_row(subjects, output):  # noqa: F811
    _make_subject(subjects, "s01")
    graph = subjects / "qc" / "ses-1" / "run-0" / "ana-0" / "folds" / "3.1" / "qc_ses-1_run-0_hemi-L.arg"
    graph.parent.mkdir(parents=True)
    graph.write_text("graph")
    rows = _run(subjects, output)
    assert [r["participant_id"] for r in rows] == ["qc", "s01"]


def test_dot_folder_is_no_row(subjects, output):  # noqa: F811
    _make_subject(subjects, "s01")
    (subjects / ".hidden").mkdir()
    rows = _run(subjects, output)
    assert [r["participant_id"] for r in rows] == ["s01"]


def test_ra_folder_is_a_row(subjects, output):  # noqa: F811
    _make_subject(subjects, "s01")
    (subjects / "ra").mkdir()
    rows = _run(subjects, output)
    assert [r["participant_id"] for r in rows] == ["ra", "s01"]
