"""list_morphologist_subjects: which entries of a Morphologist subjects folder are subjects.

The rule is narrow: a folder named exactly qc is left out only when it holds no .arg file outside dot folders
(REQ-ARGDISC-BDRABCZUK-8A14A3246F95, -372F95A73053), a dot-folder always is, and every other folder stays listed, so
that a failed subject keeps its row in the input QC. A t1mri folder in qc without a graph no longer keeps qc listed.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

GRAPH_FOLDER = "default_acquisition/default_analysis/folds/3.3/base"
QC_REPORT = "morphologist_qc.tsv"
RULE_MODULE = "champollion_pipeline.utils.subject_folders"
HEAVY_MODULES = ("torch", "soma.aims", "numpy", "pandas")


def load_rule():
    """Return the rule, importing it here so that a missing module fails each test, not the collection."""
    return importlib.import_module(RULE_MODULE).list_morphologist_subjects


def list_subjects(subjects_dir) -> list:
    """Call the rule on a str path, as the stage scripts do."""
    return load_rule()(str(subjects_dir))


def add_graphs(t1mri_parent: Path, name: str) -> None:
    """Write one graph per hemisphere under ``<t1mri_parent>/t1mri``, as Morphologist does."""
    graph_dir = t1mri_parent / "t1mri" / GRAPH_FOLDER
    graph_dir.mkdir(parents=True)
    for side in "LR":
        (graph_dir / f"{side}{name}_default_session_auto.arg").touch()


def add_subject(subjects_dir: Path, name: str) -> Path:
    """Create the subject folder ``name`` with its graphs and return it."""
    subject_dir = subjects_dir / name
    add_graphs(subject_dir, name)
    return subject_dir


class ScanResult(list):
    """A listing of entries usable like the result of os.scandir, with or without a with block."""

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False


def reverse_order(entries: list) -> list:
    return sorted(entries, key=os.fspath, reverse=True)


def rotated_order(entries: list) -> list:
    ordered = sorted(entries, key=os.fspath)
    return ordered[2:] + ordered[:2]


def impose_listing_order(monkeypatch, order) -> None:
    """Make os.listdir and os.scandir give their entries in ``order``, as a filesystem is free to."""
    real_listdir, real_scandir = os.listdir, os.scandir

    def listdir(path="."):
        return order(real_listdir(path))

    def scandir(path="."):
        with real_scandir(path) as entries:
            return ScanResult(order(list(entries)))

    monkeypatch.setattr(os, "listdir", listdir)
    monkeypatch.setattr(os, "scandir", scandir)


@pytest.fixture
def subjects_dir(temp_dir) -> Path:
    """A subjects folder holding the subject sub-001."""
    root = Path(temp_dir) / "morphologist-6.0"
    add_subject(root, "sub-001")
    return root


@pytest.mark.parametrize("content", ["empty", "report", "t1mri_file", "link_without_t1mri"])
def test_qc_folder_is_left_out(subjects_dir, content):
    qc_folder = subjects_dir / "qc"
    if content == "link_without_t1mri":
        target = subjects_dir.parent / "elsewhere"
        target.mkdir()
        qc_folder.symlink_to(target, target_is_directory=True)
    else:
        qc_folder.mkdir()
    if content == "report":
        (qc_folder / QC_REPORT).touch()
    if content == "t1mri_file":
        (qc_folder / "t1mri").touch()

    assert list_subjects(subjects_dir) == ["sub-001"]


def test_qc_folder_with_a_nested_report_is_left_out(subjects_dir):
    report_dir = subjects_dir / "qc" / "reports" / "sub-001"
    report_dir.mkdir(parents=True)
    (report_dir / QC_REPORT).touch()

    assert list_subjects(subjects_dir) == ["sub-001"]


@pytest.mark.parametrize(
    "t1mri_folder",
    ["qc/reports/t1mri", "qc/ses-1/anat/t1mri"],
    ids=["below_another_folder", "below_a_session_subfolder"],
)
def test_t1mri_deeper_in_qc_is_left_out(subjects_dir, t1mri_folder):
    (subjects_dir / t1mri_folder).mkdir(parents=True)

    assert list_subjects(subjects_dir) == ["sub-001"]


@pytest.mark.parametrize("t1mri_parent", ["qc", "qc/ses-1"], ids=["graphs", "session_graphs"])
def test_qc_folder_holding_t1mri_is_listed(subjects_dir, t1mri_parent):
    # REQ-ARGDISC-BDRABCZUK-372F95A73053: qc holding a .arg file is a subject
    add_graphs(subjects_dir / t1mri_parent, "qc")

    assert list_subjects(subjects_dir) == ["qc", "sub-001"]


@pytest.mark.parametrize("t1mri_parent", ["qc", "qc/ses-1"], ids=["t1mri", "session_t1mri"])
def test_qc_folder_holding_t1mri_without_graph_is_left_out(subjects_dir, t1mri_parent):
    # REQ-ARGDISC-BDRABCZUK-8A14A3246F95: was the no_graph case of the test above; a t1mri folder is no graph
    (subjects_dir / t1mri_parent / "t1mri").mkdir(parents=True)

    assert list_subjects(subjects_dir) == ["sub-001"]


@pytest.mark.parametrize(
    "folder, inner_folder",
    [("sub-002", ""), ("sub-002", "t1mri/default_acquisition"), ("ra", ""), ("logs", "")],
    ids=["empty", "no_graph_in_t1mri", "ra", "other_name"],
)
def test_folder_without_graphs_is_listed(subjects_dir, folder, inner_folder):
    (subjects_dir / folder / inner_folder).mkdir(parents=True)

    assert sorted(list_subjects(subjects_dir)) == sorted(["sub-001", folder])


def test_dot_folder_is_left_out(subjects_dir):
    (subjects_dir / ".snapshot").mkdir()

    assert list_subjects(subjects_dir) == ["sub-001"]


def test_dot_folder_with_graphs_is_left_out(subjects_dir):
    add_subject(subjects_dir, ".sub-002")

    assert list_subjects(subjects_dir) == ["sub-001"]


@pytest.mark.parametrize("kind", ["file", "dangling_link"])
def test_file_is_left_out(subjects_dir, kind):
    if kind == "file":
        (subjects_dir / "input_qc.tsv").touch()
    else:
        (subjects_dir / "sub-002").symlink_to(subjects_dir / "gone")

    assert list_subjects(subjects_dir) == ["sub-001"]


def test_subject_with_graphs_is_listed(subjects_dir):
    add_subject(subjects_dir, "ixi002")

    assert sorted(list_subjects(subjects_dir)) == ["ixi002", "sub-001"]


@pytest.mark.parametrize("name", ["QC", "Qc", "qc01", "qc_control", "sub-qc", "qc.1"])
def test_name_close_to_qc_is_listed(subjects_dir, name):
    add_subject(subjects_dir, name)

    assert sorted(list_subjects(subjects_dir)) == sorted(["sub-001", name])


def test_session_layout_subject_is_listed(subjects_dir):
    add_graphs(subjects_dir / "sub-002" / "ses-1", "sub-002")

    assert sorted(list_subjects(subjects_dir)) == ["sub-001", "sub-002"]


@pytest.mark.parametrize("order", [None, reverse_order, rotated_order], ids=["filesystem", "reverse", "rotated"])
def test_names_are_in_ascending_order(subjects_dir, monkeypatch, order):
    for name in ("sub-010", "sub-002", "ixi003"):
        add_subject(subjects_dir, name)
    with monkeypatch.context() as scoped:  # undone here: temp_dir's teardown lists folders too
        if order:
            impose_listing_order(scoped, order)
        names = list_subjects(subjects_dir)

    assert names == ["ixi003", "sub-001", "sub-002", "sub-010"]


def test_link_to_a_subject_folder_is_listed(subjects_dir, temp_dir):
    target = add_subject(Path(temp_dir) / "elsewhere", "sub-002")
    (subjects_dir / "sub-002").symlink_to(target, target_is_directory=True)

    assert sorted(list_subjects(subjects_dir)) == ["sub-001", "sub-002"]


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_path_that_is_no_folder_gives_no_subject(temp_dir, kind):
    path = Path(temp_dir) / "subjects"
    if kind == "file":
        path.touch()

    assert list_subjects(path) == []


def test_empty_subjects_folder_gives_no_subject(temp_dir):
    assert list_subjects(temp_dir) == []


def test_path_argument_is_accepted(subjects_dir):
    assert load_rule()(subjects_dir) == ["sub-001"]


def test_import_loads_no_heavy_module():
    code = f"import sys, {RULE_MODULE}; print(sorted(set(sys.modules) & set({HEAVY_MODULES!r})))"

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180, check=False)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_unreadable_subjects_folder_keeps_the_oserror(subjects_dir, monkeypatch):
    def refuse(path="."):
        raise PermissionError(13, "Permission denied", os.fspath(path))

    with monkeypatch.context() as scoped:  # undone here: temp_dir's teardown lists folders too
        scoped.setattr(os, "listdir", refuse)
        scoped.setattr(os, "scandir", refuse)
        with pytest.raises(OSError):
            list_subjects(subjects_dir)
