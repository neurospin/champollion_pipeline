"""Stage 2 pre-check on real Morphologist trees (TASK-PRECHECK-BDRABCZUK-E89EBABE58FB).

DEF-PRECHECK-BDRABCZUK-761792C8EAF3: the IXI run logged "0 of 2 scans eligible" though the graphs existed.
path_to_graph and path_sk_with_hull are relative to the subject folder, the checker's patterns to the scan
folder, so <subject>/t1mri/<acquisition> came out doubled in front of each pattern.

REQ-PRECHECK-BDRABCZUK-3760929352FB: a graph is missing exactly when absent from the scan's own folder.
REQ-PRECHECK-BDRABCZUK-05899543409B: a skeleton is missing exactly when absent from the scan's own folder.
REQ-PRECHECK-BDRABCZUK-853A20036CEC: on a one-scan subject whose path_to_graph lies inside the scan folder, the
pre-check and input_qc.tsv agree on missing graphs.
REQ-PRECHECK-BDRABCZUK-E5BA26B20CC8: ineligible scans come in ScanId order, however the file system lists folders.
REQ-PRECHECK-BDRABCZUK-6898C699FF81: the index file holds the same bytes, however the file system lists folders.

Each test goes through PipelineOrchestrator.run() with the crop step mocked and reads the records the user
reads, so it holds wherever the fix sits. Each tree also holds a complete subject: that subject is what the
Defect turns ineligible.
"""

import importlib.util
import json
import logging
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.run_cortical_tiles import (
    REASON_MISSING_GRAPH_STEM,
    REASON_SEPARATOR,
    fetch_input_qc_rows,
)
from tests.listing_helpers import file_system_listing_reversed
from tests.log_assertions import OUTPUT

pytestmark = pytest.mark.unit  # noqa: V107

CLI_PY = Path(__file__).resolve().parent.parent / "src" / "champollion_pipeline" / "cli.py"

ANALYSIS = "default_analysis"
ACQUISITION = "default_acquisition"
GRAPH_SUBPATH = "folds/3.1"
SKELETON_SUBPATH = "segmentation"
# The values of the IXI run on rosette.
GRAPH_PATH = f"t1mri/{ACQUISITION}/{ANALYSIS}/{GRAPH_SUBPATH}"
SKELETON_PATH = f"t1mri/{ACQUISITION}/{ANALYSIS}/{SKELETON_SUBPATH}"
# The README's wildcard, and the one the streaming stages ask for when acquisition folders differ.
ANY_ANALYSIS_GRAPH_PATH = f"t1mri/{ACQUISITION}/*/{GRAPH_SUBPATH}"
ANY_ACQUISITION_GRAPH_PATH = f"t1mri/*/{ANALYSIS}/{GRAPH_SUBPATH}"
ANY_ACQUISITION_SKELETON_PATH = f"t1mri/*/{ANALYSIS}/{SKELETON_SUBPATH}"

COMPLETE_SUBJECT = "sub-001"
OTHER_SUBJECT = "sub-002"
SUMMARY_PATTERN = re.compile(r"^Pre-check run_cortical_tiles: (\d+) of (\d+) scans eligible")
INELIGIBLE_MARK = " ineligible scans: "
INDEX_FILE = "index_pre_run_cortical_tiles.json"
# BTree(t=64) holds 127 keys in its root: more files than that split it, and only then can its shape show the order.
FILLER_FILES_PER_SUBJECT = 100
GRAPH_SIDE_PATTERN = re.compile(r"/([LR])\*\.arg$")


@dataclass
class PreCheck:
    """What a pre-check told the user: (eligible, total), per ineligible scan its missing patterns, the raw details."""

    counts: tuple[int, int]
    missing: dict[str, set[str]]
    details: str


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _scan_dir(subjects: Path, subject: str, acquisition: str = ACQUISITION) -> Path:
    return subjects / subject / "t1mri" / acquisition / ANALYSIS


def _add_graphs(scan_dir: Path, subject: str, sides: str = "LR") -> None:
    for side in sides:
        _touch(scan_dir / GRAPH_SUBPATH / f"{side}{subject}.arg")


def _add_skeletons(scan_dir: Path, subject: str, sides: str = "LR") -> None:
    for side in sides:
        _touch(scan_dir / SKELETON_SUBPATH / f"{side}skeleton_{subject}.nii.gz")


def _skeleton_pattern(side: str) -> str:
    return f"{SKELETON_PATH}/{side}skeleton*.nii.gz"


def _add_complete_scan(subjects: Path, subject: str, acquisition: str = ACQUISITION) -> None:
    scan_dir = _scan_dir(subjects, subject, acquisition)
    _add_graphs(scan_dir, subject)
    _add_skeletons(scan_dir, subject)


def _parse_ineligible(details: str) -> dict[str, set[str]]:
    """Read "<scan> missing <pattern>, <pattern>; <scan> missing <pattern>"."""
    scans = (entry.split(" missing ", 1) for entry in details.split("; "))
    return {scan: set(patterns.split(", ")) for scan, patterns in scans}


@pytest.fixture
def main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_pre_check_under_test", CLI_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def subjects(tmp_path) -> Path:
    folder = tmp_path / "subjects"
    folder.mkdir()
    return folder


@pytest.fixture
def run_pre_check(main_module, subjects, tmp_path, pipeline_caplog):
    """Run the sequential stage 2 on ``subjects`` with its crop step mocked; return its PreCheck."""

    def run(path_to_graph: str = GRAPH_PATH, path_sk_with_hull: str = SKELETON_PATH, bids: bool = False) -> PreCheck:
        config = main_module.PipelineConfig(
            root_path=str(tmp_path),
            data_path=str(tmp_path / "data"),
            models_path=str(tmp_path / "models"),
            outputs_path=str(tmp_path / "outputs"),
        )
        config.stages = {"run_cortical_tiles": True}
        config.log_to_file = False  # noqa: V101
        config.log_dir = None  # noqa: V101
        dataset = config.dataset
        dataset.morphologist_graphs = str(subjects)  # noqa: V101
        dataset.cortical_tiles_output = str(tmp_path / "tiles")  # noqa: V101
        dataset.path_to_graph = path_to_graph  # noqa: V101
        dataset.path_sk_with_hull = path_sk_with_hull  # noqa: V101
        dataset.bids = bids  # noqa: V101
        records_before = len(pipeline_caplog.records)
        with patch("champollion_pipeline.run_cortical_tiles.RunCorticalTiles.run", return_value=0):
            assert main_module.PipelineOrchestrator(config).run() == 0
        messages = [(record.levelno, record.getMessage()) for record in pipeline_caplog.records[records_before:]]
        summaries = [SUMMARY_PATTERN.match(text) for level, text in messages if level == OUTPUT]
        summaries = [match for match in summaries if match]
        assert len(summaries) == 1, f"expected one pre-check summary, got {messages}"
        warnings = [text for level, text in messages if level == logging.WARNING and INELIGIBLE_MARK in text]
        details = warnings[0].split(INELIGIBLE_MARK, 1)[1] if warnings else ""
        missing = _parse_ineligible(details) if details else {}
        return PreCheck((int(summaries[0][1]), int(summaries[0][2])), missing, details)

    return run


# ====================== REQ-PRECHECK-BDRABCZUK-3760929352FB: graphs ======================


@pytest.mark.parametrize(
    ("path_to_graph", "path_sk_with_hull", "bids"),
    [
        pytest.param(GRAPH_PATH, SKELETON_PATH, False, id="defect-tree"),
        pytest.param(GRAPH_PATH, SKELETON_PATH, True, id="defect-tree-bids"),
        pytest.param(ANY_ANALYSIS_GRAPH_PATH, SKELETON_PATH, False, id="wildcard-analysis"),
        pytest.param(ANY_ACQUISITION_GRAPH_PATH, ANY_ACQUISITION_SKELETON_PATH, False, id="wildcard-acquisition"),
    ],
)
def test_present_graphs_are_not_missing(subjects, run_pre_check, path_to_graph, path_sk_with_hull, bids):
    """The tree of the Defect, then the same tree through wildcard paths: 2 of 2 eligible, nothing missing."""
    _add_complete_scan(subjects, COMPLETE_SUBJECT)
    _add_complete_scan(subjects, OTHER_SUBJECT)

    result = run_pre_check(path_to_graph, path_sk_with_hull, bids)

    assert result.missing == {}
    assert result.counts == (2, 2)


@pytest.mark.parametrize(("absent", "present"), [("L", "R"), ("R", "L")])
def test_absent_hemisphere_graph_is_missing(subjects, run_pre_check, absent, present):
    """Only the absent hemisphere's pattern is reported, and only for the scan lacking it."""
    _add_complete_scan(subjects, COMPLETE_SUBJECT)
    scan_dir = _scan_dir(subjects, OTHER_SUBJECT)
    _add_graphs(scan_dir, OTHER_SUBJECT, sides=present)
    _add_skeletons(scan_dir, OTHER_SUBJECT)

    result = run_pre_check()

    assert result.missing == {OTHER_SUBJECT: {f"{GRAPH_PATH}/{absent}*.arg"}}
    assert result.counts == (1, 2)


def test_graph_of_another_acquisition_does_not_count(subjects, run_pre_check):
    """Through t1mri/*/..., a graph in ses-1 says nothing of ses-2: each scan is judged on its own folder."""
    _add_complete_scan(subjects, COMPLETE_SUBJECT, "ses-1")
    _add_skeletons(_scan_dir(subjects, COMPLETE_SUBJECT, "ses-2"), COMPLETE_SUBJECT)

    result = run_pre_check(ANY_ACQUISITION_GRAPH_PATH, ANY_ACQUISITION_SKELETON_PATH, bids=True)

    expected = {f"{ANY_ACQUISITION_GRAPH_PATH}/{side}*.arg" for side in "LR"}
    assert result.missing == {f"{COMPLETE_SUBJECT}_ses-2": expected}
    assert result.counts == (1, 2)


def test_literal_path_names_one_acquisition(subjects, run_pre_check):
    """A path naming default_acquisition does not reach a graph of other_acquisition, as a streaming worker sees it."""
    _add_complete_scan(subjects, COMPLETE_SUBJECT)
    _add_complete_scan(subjects, COMPLETE_SUBJECT, "other_acquisition")

    result = run_pre_check()

    expected = {f"{GRAPH_PATH}/{side}*.arg" for side in "LR"} | {_skeleton_pattern(side) for side in "LR"}
    assert result.missing == {COMPLETE_SUBJECT: expected}
    assert result.counts == (1, 2)


def test_symlinked_subject_is_eligible(subjects, tmp_path, run_pre_check):
    """A subject folder that is a symbolic link holds its graphs, for the pre-check as for input_qc.tsv."""
    _add_complete_scan(subjects, COMPLETE_SUBJECT)
    elsewhere = tmp_path / "elsewhere"
    _add_complete_scan(elsewhere, OTHER_SUBJECT)
    (subjects / OTHER_SUBJECT).symlink_to(elsewhere / OTHER_SUBJECT)

    result = run_pre_check()

    assert [row["qc"] for row in fetch_input_qc_rows(str(subjects), GRAPH_PATH, None, False, True)] == [1, 1]
    assert result.missing == {}
    assert result.counts == (2, 2)


@pytest.mark.parametrize("folder", [pytest.param("3.?", id="question-mark"), pytest.param("3.[1]", id="bracket")])
def test_glob_characters_in_path_are_literal(subjects, run_pre_check, folder):
    """Only '*' is a wildcard: the path reaches the folder of that very name, not folds/3.1 which it would glob to."""
    path_to_graph = f"t1mri/{ACQUISITION}/{ANALYSIS}/folds/{folder}"
    literal_dir = _scan_dir(subjects, COMPLETE_SUBJECT)
    _graphs_both_sides(literal_dir / "folds" / folder, COMPLETE_SUBJECT)
    _add_skeletons(literal_dir, COMPLETE_SUBJECT)
    globbed_dir = _scan_dir(subjects, OTHER_SUBJECT)
    _add_graphs(globbed_dir, OTHER_SUBJECT)
    _add_skeletons(globbed_dir, OTHER_SUBJECT)

    result = run_pre_check(path_to_graph)

    assert result.missing == {OTHER_SUBJECT: {f"{path_to_graph}/{side}*.arg" for side in "LR"}}
    assert result.counts == (1, 2)


def test_trailing_slash_in_path(subjects, run_pre_check):
    """A '/' closing path_to_graph changes nothing."""
    _add_complete_scan(subjects, COMPLETE_SUBJECT)
    _add_complete_scan(subjects, OTHER_SUBJECT)

    result = run_pre_check(f"{GRAPH_PATH}/")

    assert result.missing == {}
    assert result.counts == (2, 2)


def test_bids_session_path_is_reported_verbatim(subjects, run_pre_check):
    """With bids, a path starting with ses-* is what the user configured: reported whole, no segment stripped."""
    session_path = f"ses-*/anat/{GRAPH_SUBPATH}"
    _add_skeletons(_scan_dir(subjects, OTHER_SUBJECT), OTHER_SUBJECT)

    result = run_pre_check(session_path, bids=True)

    assert result.missing == {OTHER_SUBJECT: {f"{session_path}/{side}*.arg" for side in "LR"}}


def test_graph_outside_scan_folder_is_missing(subjects, run_pre_check):
    """Outside t1mri/<acquisition> a graph belongs to no scan: missing here, though input_qc.tsv finds it."""
    outside_path = "graphs"
    _graphs_both_sides(subjects / OTHER_SUBJECT / outside_path, OTHER_SUBJECT)
    _add_skeletons(_scan_dir(subjects, OTHER_SUBJECT), OTHER_SUBJECT)

    result = run_pre_check(outside_path)

    assert result.missing == {OTHER_SUBJECT: {f"{outside_path}/{side}*.arg" for side in "LR"}}


# ====================== REQ-PRECHECK-BDRABCZUK-05899543409B: skeleton ======================


@pytest.mark.parametrize(
    ("path_to_graph", "path_sk_with_hull"),
    [
        pytest.param(GRAPH_PATH, SKELETON_PATH, id="literal"),
        pytest.param(ANY_ACQUISITION_GRAPH_PATH, ANY_ACQUISITION_SKELETON_PATH, id="wildcard-acquisition"),
    ],
)
def test_present_skeleton_is_not_missing(subjects, run_pre_check, path_to_graph, path_sk_with_hull):
    """A scan with a skeleton but no graph is reported for its graphs alone."""
    _add_complete_scan(subjects, COMPLETE_SUBJECT)
    _add_skeletons(_scan_dir(subjects, OTHER_SUBJECT), OTHER_SUBJECT)

    result = run_pre_check(path_to_graph, path_sk_with_hull)

    assert result.missing == {OTHER_SUBJECT: {f"{path_to_graph}/{side}*.arg" for side in "LR"}}


@pytest.mark.parametrize(("absent", "present"), [("L", "R"), ("R", "L")])
def test_absent_skeleton_is_missing(subjects, run_pre_check, absent, present):
    """A scan lacking a skeleton is reported for it alone; the other hemisphere's does not stand in for it."""
    _add_complete_scan(subjects, COMPLETE_SUBJECT)
    scan_dir = _scan_dir(subjects, OTHER_SUBJECT)
    _add_graphs(scan_dir, OTHER_SUBJECT)
    _add_skeletons(scan_dir, OTHER_SUBJECT, sides=present)

    result = run_pre_check()

    assert result.missing == {OTHER_SUBJECT: {_skeleton_pattern(absent)}}
    assert result.counts == (1, 2)


def test_other_volume_is_not_a_skeleton(subjects, run_pre_check):
    """Morphologist writes other R*.nii.gz volumes beside the skeleton: none of them stands in for it."""
    _add_complete_scan(subjects, COMPLETE_SUBJECT)
    scan_dir = _scan_dir(subjects, OTHER_SUBJECT)
    _add_graphs(scan_dir, OTHER_SUBJECT)
    _add_skeletons(scan_dir, OTHER_SUBJECT, sides="L")
    _touch(scan_dir / SKELETON_SUBPATH / f"Rroots_{OTHER_SUBJECT}.nii.gz")

    result = run_pre_check()

    assert result.missing == {OTHER_SUBJECT: {_skeleton_pattern("R")}}


# ====================== REQ-PRECHECK-BDRABCZUK-853A20036CEC: the two views ======================


def _graphs_both_sides(graph_dir: Path, subject: str) -> None:
    _touch(graph_dir / f"L{subject}.arg")
    _touch(graph_dir / f"R{subject}.arg")


def _graph_right_only(graph_dir: Path, subject: str) -> None:
    _touch(graph_dir / f"R{subject}.arg")


def _graph_left_only(graph_dir: Path, subject: str) -> None:
    _touch(graph_dir / f"L{subject}.arg")


def _no_graph(graph_dir: Path, _subject: str) -> None:
    graph_dir.mkdir(parents=True)


def _labelled_graphs_only(graph_dir: Path, subject: str) -> None:
    """Graphs of a labelling session only, one level below the folder input_qc.tsv reads."""
    _graphs_both_sides(graph_dir / "deepcnn_session_auto", subject)


def _labelled_graphs_in_side_named_session(graph_dir: Path, subject: str) -> None:
    """As above, in session folders whose names alone could satisfy L*.arg or R*.arg across a '/'."""
    _touch(graph_dir / "L_session" / f"L{subject}.arg")
    _touch(graph_dir / "R_session" / f"R{subject}.arg")


def _pre_check_sides(result: PreCheck, subject: str) -> set[str]:
    """Hemispheres whose graph pattern the pre-check reported missing for ``subject``."""
    matches = (GRAPH_SIDE_PATTERN.search(pattern) for pattern in result.missing.get(subject, ()))
    return {match[1] for match in matches if match}


def _input_qc_sides(row: dict) -> set[str]:
    """Hemispheres named by the missing_graph_ reasons of an input_qc.tsv row."""
    reasons = str(row["reason"]).split(REASON_SEPARATOR)
    return {reason.removeprefix(REASON_MISSING_GRAPH_STEM) for reason in reasons if reason}


@pytest.mark.parametrize(
    "lay_out_graphs",
    [
        pytest.param(_graphs_both_sides, id="both-hemispheres"),
        pytest.param(_graph_right_only, id="right-only"),
        pytest.param(_graph_left_only, id="left-only"),
        pytest.param(_no_graph, id="no-graph"),
        pytest.param(_labelled_graphs_only, id="labelled-only"),
        pytest.param(_labelled_graphs_in_side_named_session, id="labelled-in-side-named-session"),
    ],
)
def test_pre_check_agrees_with_input_qc(subjects, run_pre_check, lay_out_graphs):
    """Per subject, the pre-check and fetch_input_qc_rows name the same missing hemispheres."""
    _add_complete_scan(subjects, COMPLETE_SUBJECT)
    scan_dir = _scan_dir(subjects, OTHER_SUBJECT)
    lay_out_graphs(scan_dir / GRAPH_SUBPATH, OTHER_SUBJECT)
    _add_skeletons(scan_dir, OTHER_SUBJECT)

    result = run_pre_check()
    rows = fetch_input_qc_rows(str(subjects), GRAPH_PATH, None, False, True)

    input_qc = {row["participant_id"]: _input_qc_sides(row) for row in rows}
    pre_check = {subject: _pre_check_sides(result, subject) for subject in input_qc}
    assert pre_check == input_qc


# ====================== REQ-PRECHECK-BDRABCZUK-E5BA26B20CC8: order of the ineligible scans ======================


@pytest.mark.parametrize(
    "listing",
    [
        pytest.param(nullcontext, id="listing-as-is"),
        pytest.param(file_system_listing_reversed, id="listing-reversed"),
    ],
)
def test_ineligible_scans_keep_their_order(subjects, run_pre_check, listing):
    """Scans in ScanId order, each scan's patterns in pattern order, whichever way the folders are listed."""
    third_subject = "sub-003"
    _add_complete_scan(subjects, COMPLETE_SUBJECT)
    no_graph_dir = _scan_dir(subjects, OTHER_SUBJECT)
    _add_skeletons(no_graph_dir, OTHER_SUBJECT)
    no_skeleton_dir = _scan_dir(subjects, third_subject)
    _add_graphs(no_skeleton_dir, third_subject)

    with listing():
        result = run_pre_check()

    assert result.details == (
        f"{OTHER_SUBJECT} missing {GRAPH_PATH}/L*.arg, {GRAPH_PATH}/R*.arg; "
        f"{third_subject} missing {_skeleton_pattern('L')}, {_skeleton_pattern('R')}"
    )


# ====================== REQ-PRECHECK-BDRABCZUK-6898C699FF81: the index file ======================


def test_index_bytes_keep_their_order(subjects, tmp_path, run_pre_check):
    """Listing the same tree the other way round writes the same index file."""
    for subject in (COMPLETE_SUBJECT, OTHER_SUBJECT):
        _add_complete_scan(subjects, subject)
        for number in range(FILLER_FILES_PER_SUBJECT):
            _touch(_scan_dir(subjects, subject) / "filler" / f"{number:03d}.txt")
    index_file = tmp_path / "outputs" / INDEX_FILE

    run_pre_check()
    as_listed = index_file.read_bytes()
    with file_system_listing_reversed():
        run_pre_check()

    assert index_file.read_bytes() == as_listed


def test_index_pre_file_is_still_written(subjects, tmp_path, run_pre_check):
    """The pre-check keeps saving its index, whichever way it matches the patterns."""
    _add_complete_scan(subjects, COMPLETE_SUBJECT)

    run_pre_check()

    assert json.loads((tmp_path / "outputs" / INDEX_FILE).read_text())["size"] == len("LR") * 2
