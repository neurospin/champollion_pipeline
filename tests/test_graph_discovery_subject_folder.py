"""Graph discovery tags each graph with the subject folder the resolver takes (TASK-ARGDISC-BDRABCZUK-5ABBA33DB536).

REQ-ARGDISC-BDRABCZUK-CFA5580A0627 (live revision, user decision 2026-11-26 Q3): discover_graphs sets an item's
subject attribute to the name of the subject folder that resolve_subjects_dir takes for that graph when the folder
given to discover_graphs is the input path. So the subject folder of a graph is the nearest sub- folder
(REQ-ARGDISC-BDRABCZUK-EE24FF19000D), else the nearest folder named after the graph file
(REQ-ARGDISC-BDRABCZUK-D8BACBE58917, the given folder included: REQ-ARGDISC-BDRABCZUK-1E1A74720545), else the
folder directly in the given folder (REQ-ARGDISC-BDRABCZUK-6ED8713C70C3), never a rule of discovery's own.

Expected values are written out here from the layouts, independently of the code under test.
"""

import importlib
import os
from pathlib import Path

import pytest

from champollion_pipeline.file_indexer.pipeline_checks import FileIndexStageIndexer
from tests.morphologist_layout_fixtures import _create

pytestmark = pytest.mark.unit  # noqa: V107

CLASSIC_FOLDS = "t1mri/default_acquisition/default_analysis/folds/3.1"
CLI_FOLDS = "ses-1/run-0/ana-0/folds/3.1"
# A graph below the given folder "in", and the subject folder of that graph, relative to "in".
SUBJECT_FOLDERS = [
    pytest.param("data/IXI002/folds/LIXI002.arg", "data/IXI002", id="named_folder_below_a_child"),
    pytest.param(
        f"data/IXI002/{CLASSIC_FOLDS}/default_session_auto/LIXI002_default_session_auto.arg",
        "data/IXI002",
        id="labelled_graph_below_a_named_folder",
    ),
    pytest.param(f"data/IXI003/{CLI_FOLDS}/IXI003_ses-1_run-0_hemi-L.arg", "data/IXI003", id="cli_graph"),
    pytest.param("IXI002/copy/IXI005/folds/RIXI005.arg", "IXI002/copy/IXI005", id="nearest_named_folder"),
    pytest.param("data/x/Lgraph.arg", "data", id="named_after_no_folder"),
    pytest.param("sub-01/IXI002/LIXI002.arg", "sub-01", id="sub_folder_first"),
]
# A subject folder given to discover_graphs, holding its own graph.
OWN_GRAPHS = [
    pytest.param("IXI002", f"{CLASSIC_FOLDS}/LIXI002.arg", id="classic"),
    pytest.param("IXI003", f"{CLI_FOLDS}/IXI003_ses-1_run-0_hemi-R.arg", id="cli"),
]


def _subjects(folder: Path) -> dict[str, str]:
    """Subject attribute of each item of discover_graphs(folder), keyed by item path."""
    items = FileIndexStageIndexer(str(folder.parent / "indexes")).discover_graphs(str(folder))
    return {item.path: item.attributes["subject"] for item in items}


@pytest.mark.parametrize(("graph", "subject_folder"), SUBJECT_FOLDERS)
def test_subject_is_the_name_of_the_graph_subject_folder(tmp_path, graph, subject_folder):
    _create(tmp_path / "in", [graph])

    assert _subjects(tmp_path / "in") == {graph: os.path.basename(subject_folder)}


@pytest.mark.parametrize(("subject", "graph"), OWN_GRAPHS)
def test_subject_of_a_given_subject_folder_is_its_name(tmp_path, subject, graph):
    _create(tmp_path / subject, [graph])

    assert _subjects(tmp_path / subject) == {graph: subject}


@pytest.mark.parametrize(("graph", "subject_folder"), SUBJECT_FOLDERS)
def test_subject_sits_in_the_subjects_folder_of_the_resolver(tmp_path, graph, subject_folder):
    # The resolver returns the parent of the subject folders; discovery names one of them: the two must agree.
    _create(tmp_path / "in", [graph])
    resolve = importlib.import_module("champollion_pipeline.utils.input_kind").resolve_subjects_dir

    subjects_dir = resolve(str(tmp_path / "in"))

    assert subjects_dir == str(tmp_path / "in" / os.path.dirname(subject_folder)).rstrip(os.sep)
    assert _subjects(tmp_path / "in")[graph] == os.path.basename(subject_folder)
