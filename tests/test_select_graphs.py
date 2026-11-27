"""Graph selection behind the StageIndexer port (TASK-ARGDISC-BDRABCZUK-30BA6F0292BA).

select_graphs(folder, filters) returns the graphs discover_graphs finds below folder, each with the attributes
subject, session, run and analysis, narrowed by filters in the filter API of capsul's DataSelector: a regular
expression per attribute, applied with re.match (anchored at the start only); an item lacking a filtered attribute
is left out. With no filter every graph is kept, so every sub-*/ses-*/run-*/ana-* folder is its own scan. Each
behaviour test runs over every adapter in ADAPTERS, so a capsul.datasel adapter can join later.

Expected values are written out here from the layouts, independently of the code under test.

* Port declares select_graphs - test_port_declares_select_graphs
* No filter keeps every discovered graph - test_no_filter_returns_discovered_paths_in_order,
  test_empty_filters_return_discovered_paths_in_order, test_no_filter_keeps_every_scan
* discover_graphs attribute values kept - test_items_keep_discovered_attributes
* Session attribute - test_session_is_nearest_ses_folder, test_classic_tree_has_no_session_run_or_analysis
* Run attribute - test_run_is_nearest_run_folder, test_classic_tree_has_no_session_run_or_analysis
* Analysis attribute - test_analysis_is_nearest_ana_folder, test_classic_tree_has_no_session_run_or_analysis
* Filters matched at the start - test_session_filter_keeps_matching_items, test_subject_filter_keeps_matching_items,
  test_run_filter_keeps_matching_items, test_analysis_filter_keeps_matching_items,
  test_filter_is_anchored_at_start_only, test_filters_combine, test_filtered_items_are_the_matching_discovered_items
* None value of a filtered attribute - test_item_without_filtered_attribute_is_left_out
* Unknown filter attribute - test_unknown_filter_attribute_is_refused
* Invalid regular expression - test_invalid_regular_expression_is_refused
* Same exception as discover_graphs - test_missing_folder_raises_not_a_directory,
  test_loose_graph_raises_loose_graph_error

The Requirement ids are in the ledger (alm pack TASK-ARGDISC-BDRABCZUK-30BA6F0292BA).
"""

import inspect
import re
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from champollion_pipeline._stage_base import StageIndexer
from champollion_pipeline.file_indexer.pipeline_checks import FileIndexStageIndexer
from champollion_pipeline.utils.graph_item import LooseGraphError
from tests.morphologist_layout_fixtures import (
    CLI_LABELLED_SUBJECTS,
    CLI_SUBJECTS,
    HEMISPHERES,
    _create,
    build_classic_tree,
    build_morphologist_cli_tree,
)

ADAPTERS = (FileIndexStageIndexer,)
FILTER_ATTRIBUTES = ("subject", "session", "run", "analysis")

# Scans added to sub-01 of the morphologist-cli fixture: another session, another run, another analysis.
EXTRA_SCANS = (("ses-2", "run-0", "ana-0"), ("ses-1", "run-1", "ana-0"), ("ses-1", "run-0", "ana-1"))
FIRST_SCAN = ("ses-1", "run-0", "ana-0")


def _scan_folds(subject: str, session: str, run: str, analysis: str) -> str:
    return f"{subject}/{session}/{run}/{analysis}/folds/3.1"


def _scan_graphs(subject: str, session: str, run: str, analysis: str) -> list[str]:
    """Unlabelled graph paths of one scan, as morphologist-cli names them."""
    folds = _scan_folds(subject, session, run, analysis)
    return [f"{folds}/{subject}_{session}_{run}_hemi-{hemi}.arg" for hemi in HEMISPHERES]


def _labelled_graphs(subject: str) -> list[str]:
    """Labelled graph paths of the fixture's first scan of subject."""
    folds = _scan_folds(subject, *FIRST_SCAN)
    return [f"{folds}/sul-0_auto/{subject}_ses-1_run-0_sul-0_hemi-{hemi}_auto.arg" for hemi in HEMISPHERES]


def _build_multi_scan_tree(root: Path) -> Path:
    """The shared morphologist-cli fixture plus three more scans of its first subject."""
    build_morphologist_cli_tree(root)
    return _create(root, [path for scan in EXTRA_SCANS for path in _scan_graphs(CLI_SUBJECTS[0], *scan)])


def _expected_paths(subject: str, session: str, run: str, analysis: str) -> set[str]:
    """Every graph path of one scan of the multi-scan tree."""
    paths = set(_scan_graphs(subject, session, run, analysis))
    if subject in CLI_LABELLED_SUBJECTS and (session, run, analysis) == FIRST_SCAN:
        paths |= set(_labelled_graphs(subject))
    return paths


@pytest.fixture(params=ADAPTERS, ids=lambda adapter: adapter.__name__)
def indexer(request, tmp_path):
    """One StageIndexer adapter; its own index files go to a folder apart from the trees under test."""
    return request.param(str(tmp_path / "indexes"))


@pytest.fixture
def multi_scan_tree(tmp_path) -> Path:
    return _build_multi_scan_tree(tmp_path / "in")


def _paths(items) -> list[str]:
    return [item.path for item in items]


def _by_path(items) -> dict:
    by_path = {item.path: item for item in items}
    assert len(by_path) == len(items)
    return by_path


def test_port_declares_select_graphs():
    operation = getattr(StageIndexer, "select_graphs", None)
    assert callable(operation)
    assert list(inspect.signature(operation).parameters)[1:] == ["folder", "filters"]


def test_no_filter_returns_discovered_paths_in_order(indexer, multi_scan_tree):
    discovered = _paths(indexer.discover_graphs(str(multi_scan_tree)))
    assert _paths(indexer.select_graphs(str(multi_scan_tree), None)) == discovered


def test_empty_filters_return_discovered_paths_in_order(indexer, multi_scan_tree):
    discovered = _paths(indexer.discover_graphs(str(multi_scan_tree)))
    assert _paths(indexer.select_graphs(str(multi_scan_tree), {})) == discovered


def test_no_filter_keeps_every_scan(indexer, multi_scan_tree):
    items = indexer.select_graphs(str(multi_scan_tree), None)
    scans = {tuple(item.attributes[name] for name in FILTER_ATTRIBUTES) for item in items}
    expected = {(subject, *FIRST_SCAN) for subject in CLI_SUBJECTS}
    expected |= {(CLI_SUBJECTS[0], *scan) for scan in EXTRA_SCANS}
    assert scans == expected


def test_items_keep_discovered_attributes(indexer, multi_scan_tree):
    discovered = _by_path(indexer.discover_graphs(str(multi_scan_tree)))
    selected = _by_path(indexer.select_graphs(str(multi_scan_tree), None))
    assert set(selected) == set(discovered)
    for path, item in selected.items():
        for name in ("subject", "hemisphere", "labelling_set"):
            assert item.attributes[name] == discovered[path].attributes[name], (path, name)


def test_session_is_nearest_ses_folder(indexer, multi_scan_tree):
    subject = CLI_SUBJECTS[0]
    items = _by_path(indexer.select_graphs(str(multi_scan_tree), None))
    for path in _scan_graphs(subject, "ses-2", "run-0", "ana-0"):
        assert items[path].attributes["session"] == "ses-2"
    for path in _labelled_graphs(subject):
        assert items[path].attributes["session"] == "ses-1"
    inside_subject = _by_path(indexer.select_graphs(str(multi_scan_tree / subject), None))
    assert {item.attributes["session"] for item in inside_subject.values()} == {"ses-1", "ses-2"}


def test_run_is_nearest_run_folder(indexer, multi_scan_tree):
    items = _by_path(indexer.select_graphs(str(multi_scan_tree), None))
    for path in _scan_graphs(CLI_SUBJECTS[0], "ses-1", "run-1", "ana-0"):
        assert items[path].attributes["run"] == "run-1"
    for path in _scan_graphs(CLI_SUBJECTS[1], *FIRST_SCAN):
        assert items[path].attributes["run"] == "run-0"


def test_analysis_is_nearest_ana_folder(indexer, multi_scan_tree):
    items = _by_path(indexer.select_graphs(str(multi_scan_tree), None))
    for path in _scan_graphs(CLI_SUBJECTS[0], "ses-1", "run-0", "ana-1"):
        assert items[path].attributes["analysis"] == "ana-1"
    for path in _labelled_graphs(CLI_SUBJECTS[0]):
        assert items[path].attributes["analysis"] == "ana-0"


def test_classic_tree_has_no_session_run_or_analysis(indexer, tmp_path):
    build_classic_tree(tmp_path / "in")
    items = indexer.select_graphs(str(tmp_path / "in"), None)
    assert items
    for item in items:
        assert (item.attributes["session"], item.attributes["run"], item.attributes["analysis"]) == (None, None, None)


def test_session_filter_keeps_matching_items(indexer, multi_scan_tree):
    items = indexer.select_graphs(str(multi_scan_tree), {"session": "ses-2"})
    assert set(_paths(items)) == _expected_paths(CLI_SUBJECTS[0], "ses-2", "run-0", "ana-0")


def test_subject_filter_keeps_matching_items(indexer, multi_scan_tree):
    items = indexer.select_graphs(str(multi_scan_tree), {"subject": "sub-02"})
    assert set(_paths(items)) == _expected_paths(CLI_SUBJECTS[1], *FIRST_SCAN)


def test_run_filter_keeps_matching_items(indexer, multi_scan_tree):
    items = indexer.select_graphs(str(multi_scan_tree), {"run": "run-1"})
    assert set(_paths(items)) == _expected_paths(CLI_SUBJECTS[0], "ses-1", "run-1", "ana-0")


def test_analysis_filter_keeps_matching_items(indexer, multi_scan_tree):
    items = indexer.select_graphs(str(multi_scan_tree), {"analysis": "ana-1"})
    assert set(_paths(items)) == _expected_paths(CLI_SUBJECTS[0], "ses-1", "run-0", "ana-1")


def test_filter_is_anchored_at_start_only(indexer, multi_scan_tree):
    folder = str(multi_scan_tree)
    every_path = set(_paths(indexer.discover_graphs(folder)))
    assert indexer.select_graphs(folder, {"session": "2"}) == []
    assert set(_paths(indexer.select_graphs(folder, {"session": "ses-"}))) == every_path
    assert set(_paths(indexer.select_graphs(folder, {"subject": "sub-0"}))) == every_path


def test_filters_combine(indexer, multi_scan_tree):
    filters = {"session": "ses-1$", "run": "run-0$", "analysis": "ana-0$"}
    items = indexer.select_graphs(str(multi_scan_tree), filters)
    expected = _expected_paths(CLI_SUBJECTS[0], *FIRST_SCAN) | _expected_paths(CLI_SUBJECTS[1], *FIRST_SCAN)
    assert set(_paths(items)) == expected


@pytest.fixture(scope="module")
def shared_multi_scan_tree(tmp_path_factory) -> Path:
    """One multi-scan tree for the property test; select_graphs only reads it."""
    return _build_multi_scan_tree(tmp_path_factory.mktemp("select") / "in")


PATTERNS = {
    "subject": ("sub-0", "sub-01", "sub-02$", "sub", "01", ".*"),
    "session": ("ses-", "ses-1", "ses-2$", "[sx]es-1", "1", ".*"),
    "run": ("run-", "run-0", "run-1$", "run-.$", "0", ".*"),
    "analysis": ("ana-", "ana-0", "ana-1$", "ana-[01]", "1", ".*"),
}
FILTER_DICTS = st.dictionaries(
    st.sampled_from(FILTER_ATTRIBUTES), st.just(None), max_size=len(FILTER_ATTRIBUTES)
).flatmap(lambda chosen: st.fixed_dictionaries({name: st.sampled_from(PATTERNS[name]) for name in chosen}))


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(filters=FILTER_DICTS)
def test_filtered_items_are_the_matching_discovered_items(indexer, shared_multi_scan_tree, filters):
    folder = str(shared_multi_scan_tree)
    every_item = indexer.select_graphs(folder, None)
    expected = [
        item.path
        for item in every_item
        if all(re.match(pattern, item.attributes[name]) for name, pattern in filters.items())
    ]
    assert _paths(indexer.select_graphs(folder, filters)) == expected


def test_item_without_filtered_attribute_is_left_out(indexer, tmp_path):
    build_classic_tree(tmp_path / "in")
    folder = str(tmp_path / "in")
    assert indexer.select_graphs(folder, {"session": ".*"}) == []
    assert indexer.select_graphs(folder, {"run": ""}) == []
    assert indexer.select_graphs(folder, {"analysis": ".*"}) == []


def test_unknown_filter_attribute_is_refused(indexer, multi_scan_tree):
    with pytest.raises(ValueError) as refusal:
        indexer.select_graphs(str(multi_scan_tree), {"session": "ses-1", "hemisphere": "L", "acq": "x"})
    message = str(refusal.value)
    assert "hemisphere" in message
    assert "acq" in message


def test_invalid_regular_expression_is_refused(indexer, multi_scan_tree):
    with pytest.raises(ValueError, match="session"):
        indexer.select_graphs(str(multi_scan_tree), {"session": "ses-(1"})


def test_missing_folder_raises_not_a_directory(indexer, tmp_path):
    missing = str(tmp_path / "absent")
    with pytest.raises(NotADirectoryError):
        indexer.discover_graphs(missing)
    with pytest.raises(NotADirectoryError):
        indexer.select_graphs(missing, None)


def test_loose_graph_raises_loose_graph_error(indexer, tmp_path):
    folder = _create(tmp_path / "loose", ["L01.arg"])
    with pytest.raises(LooseGraphError):
        indexer.discover_graphs(str(folder))
    with pytest.raises(LooseGraphError):
        indexer.select_graphs(str(folder), None)
