"""Graph discovery behind the StageIndexer port (TASK-ARGDISC-BDRABCZUK-28EF58093F93).

discover_graphs walks a folder and returns one item per .arg file, whatever the folder names: the item's
path relative to the folder, and its attributes subject, hemisphere and labelling_set, taken from where the
file was found. Items have the backend-neutral shape (path, attributes), so another adapter (a capsul.datasel
back end) can implement the same operation; each behaviour test runs over every adapter in ADAPTERS.

Expected values are written out here from the layouts, independently of the code under test.

* REQ-ARGDISC-BDRABCZUK-75B32C1E287B - test_port_declares_discover_graphs, test_items_hold_path_and_attributes
* REQ-ARGDISC-BDRABCZUK-2211A740AA79 - test_classic_tree_gives_one_item_per_graph,
  test_morphologist_cli_tree_gives_one_item_per_graph, test_graph_below_unnamed_folders_gives_an_item,
  test_graph_below_dot_folder_gives_no_item, test_dot_named_graph_gives_no_item
* REQ-ARGDISC-BDRABCZUK-3124F5675A06 - test_item_path_is_relative_with_slashes
* REQ-ARGDISC-BDRABCZUK-B9004EDD1017 - test_subject_is_nearest_sub_folder, test_subject_of_given_sub_folder
* REQ-ARGDISC-BDRABCZUK-CFA5580A0627 - test_subject_without_sub_folder_is_first_folder_below
* REQ-ARGDISC-BDRABCZUK-7A642AA5FC5B - test_hemisphere_from_hemi_entity, test_hemisphere_from_first_letter
* REQ-ARGDISC-BDRABCZUK-B36E59256F4D - test_hemisphere_none_without_side
* REQ-ARGDISC-BDRABCZUK-66E6B1383D2B - test_labelling_set_of_labelled_graph, test_unlabelled_graph_has_no_labelling_set
* REQ-ARGDISC-BDRABCZUK-4D3712978137 - test_items_in_natural_order
* REQ-ARGDISC-BDRABCZUK-FF60BC9D8672 - test_discovery_opens_no_file
* REQ-ARGDISC-BDRABCZUK-8C528FA74D0F - test_graph_directly_in_non_sub_folder_is_refused,
  test_graph_directly_in_given_sub_folder_gives_an_item
* REQ-ARGDISC-BDRABCZUK-1B7904DE6E04 - test_unreadable_folder_warns_during_discovery
* REQ-ARGDISC-BDRABCZUK-1D28D44EE0A8 - test_discovery_continues_past_unreadable_folder
* REQ-ARGDISC-BDRABCZUK-1CC8A2D73229 - test_folder_link_is_searched
* REQ-ARGDISC-BDRABCZUK-0DC678E51B6D - test_link_to_folder_on_search_path_is_not_followed

Requirement ids are the roots; REQ-ARGDISC-BDRABCZUK-2211A740AA79 and REQ-ARGDISC-BDRABCZUK-CFA5580A0627 are read
at their live revisions (dot-named files give no item; the subject rule applies below a child folder of the given
folder).
"""

import builtins
import inspect
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path

import pytest

from champollion_pipeline._stage_base import StageIndexer
from champollion_pipeline.file_indexer.pipeline_checks import FileIndexStageIndexer
from tests.morphologist_layout_fixtures import (
    CLASSIC_SUBJECTS,
    CLI_LABELLED_SUBJECTS,
    CLI_SUBJECTS,
    HEMISPHERES,
    SENIOR_ACQUISITIONS,
    SENIOR_SESSIONS,
    _create,
    build_classic_tree,
    build_morphologist_cli_tree,
    build_senior_tree,
)
from tests.test_input_kind import _input_with_blocked_folder, _unreadable, needs_permissions

ADAPTERS = (FileIndexStageIndexer,)
ATTRIBUTES = {"subject", "hemisphere", "labelling_set"}


@pytest.fixture(params=ADAPTERS, ids=lambda adapter: adapter.__name__)
def indexer(request, tmp_path):
    """One StageIndexer adapter; its own index files go to a folder apart from the trees under test."""
    return request.param(str(tmp_path / "indexes"))


def _discover(indexer, folder: Path) -> dict:
    """Items of discover_graphs(folder), keyed by path, after checking that no path comes twice."""
    items = indexer.discover_graphs(str(folder))
    by_path = {item.path: item for item in items}
    assert len(by_path) == len(items)
    return by_path


def _classic_graphs(subject: str) -> set[str]:
    folds = f"{subject}/t1mri/default_acquisition/default_analysis/folds/3.1"
    unlabelled = {f"{folds}/{hemi}{subject}.arg" for hemi in HEMISPHERES}
    labelled = {f"{folds}/default_session_auto/{hemi}{subject}_default_session_auto.arg" for hemi in HEMISPHERES}
    return unlabelled | labelled


def _cli_graphs(subject: str) -> set[str]:
    folds = f"{subject}/ses-1/run-0/ana-0/folds/3.1"
    graphs = {f"{folds}/{subject}_ses-1_run-0_hemi-{hemi}.arg" for hemi in HEMISPHERES}
    if subject in CLI_LABELLED_SUBJECTS:
        graphs |= {f"{folds}/sul-0_auto/{subject}_ses-1_run-0_sul-0_hemi-{hemi}_auto.arg" for hemi in HEMISPHERES}
    return graphs


def _natural_key(path: str) -> tuple:
    """Digit runs compare as numbers."""
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path))


def test_port_declares_discover_graphs():
    operation = getattr(StageIndexer, "discover_graphs", None)
    assert callable(operation)
    assert list(inspect.signature(operation).parameters)[1:] == ["folder"]


def test_items_hold_path_and_attributes(indexer, tmp_path):
    build_morphologist_cli_tree(tmp_path / "in")
    items = indexer.discover_graphs(str(tmp_path / "in"))
    assert items
    for item in items:
        assert isinstance(item.path, str)
        assert isinstance(item.attributes, Mapping)
        assert set(item.attributes) >= ATTRIBUTES


def test_classic_tree_gives_one_item_per_graph(indexer, tmp_path):
    build_classic_tree(tmp_path / "in")
    expected = set().union(*(_classic_graphs(subject) for subject in CLASSIC_SUBJECTS))
    assert set(_discover(indexer, tmp_path / "in")) == expected


def test_morphologist_cli_tree_gives_one_item_per_graph(indexer, tmp_path):
    build_morphologist_cli_tree(tmp_path / "in")
    expected = set().union(*(_cli_graphs(subject) for subject in CLI_SUBJECTS))
    assert set(_discover(indexer, tmp_path / "in")) == expected


def test_graph_below_unnamed_folders_gives_an_item(indexer, tmp_path):
    _create(tmp_path / "in", ["a/b/c/Lx.arg", "a/notes.txt"])
    assert set(_discover(indexer, tmp_path / "in")) == {"a/b/c/Lx.arg"}


def test_graph_below_dot_folder_gives_no_item(indexer, tmp_path):
    _create(tmp_path / "in", [".hidden/Lx.arg", "sub-01/.cache/Ly.arg", "sub-01/Lz.arg"])
    assert set(_discover(indexer, tmp_path / "in")) == {"sub-01/Lz.arg"}


def test_item_path_is_relative_with_slashes(indexer, tmp_path):
    root = build_morphologist_cli_tree(tmp_path / "in")
    for path in _discover(indexer, root):
        assert not os.path.isabs(path)
        assert "\\" not in path
        assert (root / Path(*path.split("/"))).is_file()


def test_subject_is_nearest_sub_folder(indexer, tmp_path):
    _create(tmp_path / "in", ["study/sub-07/ses-1/sub-07_hemi-L.arg", "sub-A/derivatives/sub-B/anat/sub-B_hemi-R.arg"])
    build_morphologist_cli_tree(tmp_path / "in" / "cli")
    items = _discover(indexer, tmp_path / "in")
    assert items["study/sub-07/ses-1/sub-07_hemi-L.arg"].attributes["subject"] == "sub-07"
    assert items["sub-A/derivatives/sub-B/anat/sub-B_hemi-R.arg"].attributes["subject"] == "sub-B"
    for subject in CLI_SUBJECTS:
        for path in _cli_graphs(subject):
            assert items[f"cli/{path}"].attributes["subject"] == subject


def test_subject_of_given_sub_folder(indexer, tmp_path):
    root = build_morphologist_cli_tree(tmp_path / "in")
    items = _discover(indexer, root / "sub-01")
    assert items
    assert {item.attributes["subject"] for item in items.values()} == {"sub-01"}


def test_subject_without_sub_folder_is_first_folder_below(indexer, tmp_path):
    build_classic_tree(tmp_path / "classic")
    for subject in CLASSIC_SUBJECTS:
        classic = _discover(indexer, tmp_path / "classic")
        assert {classic[path].attributes["subject"] for path in _classic_graphs(subject)} == {subject}
    build_senior_tree(tmp_path / "senior")
    senior = _discover(indexer, tmp_path / "senior" / "derivatives" / "subjects_f")
    assert senior
    assert {item.attributes["subject"] for item in senior.values()} == {"sf0001"}


def test_hemisphere_from_hemi_entity(indexer, tmp_path):
    items = _discover(indexer, build_morphologist_cli_tree(tmp_path / "in"))
    assert items
    for path, item in items.items():
        expected = "L" if "_hemi-L" in path.rsplit("/", 1)[-1] else "R"
        assert item.attributes["hemisphere"] == expected, path


def test_hemisphere_from_first_letter(indexer, tmp_path):
    items = _discover(indexer, build_classic_tree(tmp_path / "in"))
    assert items
    for path, item in items.items():
        assert item.attributes["hemisphere"] == path.rsplit("/", 1)[-1][0], path


def test_hemisphere_none_without_side(indexer, tmp_path):
    _create(tmp_path / "in", ["sub-01/graph.arg", "sub-01/sub-01_hemi-X.arg"])
    items = _discover(indexer, tmp_path / "in")
    assert set(items) == {"sub-01/graph.arg", "sub-01/sub-01_hemi-X.arg"}
    for item in items.values():
        assert item.attributes["hemisphere"] is None


def test_labelling_set_of_labelled_graph(indexer, tmp_path):
    build_morphologist_cli_tree(tmp_path / "in" / "cli")
    build_classic_tree(tmp_path / "in" / "classic")
    _create(tmp_path / "in", ["manual/sub-09/folds/session_manual/Lsub-09_session_manual.arg"])
    items = _discover(indexer, tmp_path / "in")
    labelled = {path: item.attributes["labelling_set"] for path, item in items.items() if "_auto/" in path}
    assert labelled
    assert {value for path, value in labelled.items() if path.startswith("cli/")} == {"sul-0_auto"}
    assert {value for path, value in labelled.items() if path.startswith("classic/")} == {"default_session_auto"}
    manual = items["manual/sub-09/folds/session_manual/Lsub-09_session_manual.arg"]
    assert manual.attributes["labelling_set"] == "session_manual"


def test_unlabelled_graph_has_no_labelling_set(indexer, tmp_path):
    _create(tmp_path / "in", ["sub-01/autograph/Lsub-01.arg"])
    build_morphologist_cli_tree(tmp_path / "in" / "cli")
    items = _discover(indexer, tmp_path / "in")
    unlabelled = [item for path, item in items.items() if "/folds/3.1/sub-" in path or path.startswith("sub-01/")]
    assert len(unlabelled) == 1 + 2 * len(CLI_SUBJECTS)
    for item in unlabelled:
        assert item.attributes["labelling_set"] is None


def test_items_in_natural_order(indexer, tmp_path):
    build_senior_tree(tmp_path / "in")
    paths = [item.path for item in indexer.discover_graphs(str(tmp_path / "in" / "derivatives" / "subjects_f"))]
    assert len(paths) == len(SENIOR_ACQUISITIONS) * len(HEMISPHERES) * (1 + len(SENIOR_SESSIONS))
    assert paths == sorted(paths, key=_natural_key)
    acquisitions = [path.split("/")[2] for path in paths]
    assert acquisitions.index("3TV2") < acquisitions.index("3TV10")


def test_discovery_opens_no_file(indexer, tmp_path, monkeypatch):
    root = build_morphologist_cli_tree(tmp_path / "in")

    def refuse(*args, **_kwargs):
        raise AssertionError(f"file opened: {args[0]!r}")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(os, "open", refuse)
    items = indexer.discover_graphs(str(root))
    assert len(items) == sum(len(_cli_graphs(subject)) for subject in CLI_SUBJECTS)


def test_dot_named_graph_gives_no_item(indexer, tmp_path):
    _create(tmp_path / "in", [".Lloose.arg", "sub-01/._Lsub-01.arg", "sub-01/.Rsub-01.arg", "sub-01/Lsub-01.arg"])
    assert set(_discover(indexer, tmp_path / "in")) == {"sub-01/Lsub-01.arg"}


def test_graph_directly_in_non_sub_folder_is_refused(indexer, tmp_path):
    _create(tmp_path / "in", ["Lloose.arg", "sub-02/Lsub-02.arg"])
    with pytest.raises(ValueError, match=re.escape("Lloose.arg")):
        indexer.discover_graphs(str(tmp_path / "in"))


def test_graph_directly_in_given_sub_folder_gives_an_item(indexer, tmp_path):
    _create(tmp_path / "sub-03", ["Lsub-03.arg"])
    items = _discover(indexer, tmp_path / "sub-03")
    assert set(items) == {"Lsub-03.arg"}
    assert items["Lsub-03.arg"].attributes["subject"] == "sub-03"


@needs_permissions
@pytest.mark.parametrize("through_link", [False, True])
def test_unreadable_folder_warns_during_discovery(indexer, tmp_path, pipeline_caplog, through_link):
    input_dir, target = _input_with_blocked_folder(tmp_path, through_link)
    _create(target, ["sub-01/Lsub-01.arg"])

    with _unreadable(target):
        indexer.discover_graphs(str(input_dir))

    pipeline_caplog.assert_logged(logging.WARNING, "blocked")


@needs_permissions
@pytest.mark.parametrize("through_link", [False, True])
def test_discovery_continues_past_unreadable_folder(indexer, tmp_path, through_link):
    input_dir, target = _input_with_blocked_folder(tmp_path, through_link)
    _create(target, ["sub-01/Lsub-01.arg"])
    _create(input_dir, ["z/sub-02/Lsub-02.arg"])

    with _unreadable(target):
        assert set(_discover(indexer, input_dir)) == {"z/sub-02/Lsub-02.arg"}


def test_folder_link_is_searched(indexer, tmp_path):
    _create(tmp_path / "store", ["sub-01/Lsub-01.arg"])
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "linked").symlink_to(tmp_path / "store", target_is_directory=True)
    items = _discover(indexer, tmp_path / "in")
    assert set(items) == {"linked/sub-01/Lsub-01.arg"}
    assert items["linked/sub-01/Lsub-01.arg"].attributes["subject"] == "sub-01"


def test_link_to_folder_on_search_path_is_not_followed(indexer, tmp_path):
    root = _create(tmp_path / "in", ["sub-01/ses-1/Lsub-01.arg"])
    (root / "sub-01" / "ses-1" / "to_root").symlink_to(root, target_is_directory=True)
    (root / "sub-01" / "ses-1" / "to_subject").symlink_to(root / "sub-01", target_is_directory=True)
    assert set(_discover(indexer, root)) == {"sub-01/ses-1/Lsub-01.arg"}
