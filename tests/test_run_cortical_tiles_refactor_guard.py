"""
Guard for the split of src/champollion_pipeline/run_cortical_tiles.py into modules (TASK-CODE-BDRABCZUK-8B3DA5AC4B20,
refactor, behaviour-preserving; user decision 2026-11-24: the new modules hold the code, run_cortical_tiles.py
delegates and re-exports, so existing imports keep working).

The reference is src/champollion_pipeline/run_cortical_tiles.py at commit 3039b13 (byte-identical at WIP 52d7e16).
Its module-level names are frozen in REFERENCE_NAMES below; the differential tests load that version with git show
and compare the moved helpers against it on generated arguments. Both pass on the reference itself.

Requirements:
REQ-RCTSPLIT-BDRABCZUK-620928E02B39 (run_cortical_tiles.py at most 750 lines; budget run_cortical_tiles_source_lines),
REQ-RCTSPLIT-BDRABCZUK-DE19F610E904 (each 3039b13 module-level name importable from run_cortical_tiles),
REQ-RCTSPLIT-BDRABCZUK-C2EAC9EA65F3 (each listed helper returns the 3039b13 value for equal arguments).
"""

import importlib.util
import json
import os
import string
import subprocess
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import champollion_pipeline.run_cortical_tiles as current

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = "src/champollion_pipeline/run_cortical_tiles.py"
REFERENCE_COMMIT = "3039b13e36a62ccb0feaa1f2847372f831e6efa7"
MAX_SOURCE_LINES = 750
MEASURE = "run_cortical_tiles_source_lines"
EXCLUSION_REASONS = ("no_graph_folder", "several_graph_folders", "no_labelled_graph")
FS_SETTINGS = settings(max_examples=40, deadline=None)

# Functions, classes and constants defined at module level in run_cortical_tiles.py at commit 3039b13, in file order.
REFERENCE_NAMES = (
    "logger",
    "LABELLING_SESSION_DEFAULT",
    "WHOLE_BRAIN_SRC_FILENAME",
    "WHOLE_BRAIN_OUT_FILENAME",
    "GRAPH_WILDCARD",
    "EXCLUDED_NO_GRAPH_FOLDER",
    "EXCLUDED_SEVERAL_GRAPH_FOLDERS",
    "EXCLUDED_NO_LABELLED_GRAPH",
    "EXCLUSION_REASONS",
    "LABELLING_QC_FILENAME",
    "LABELLING_QC_EXTRA_COLUMNS",
    "PREFIX_PAIRS_SHOWN",
    "TRANSFORM_QC_FILENAME",
    "INPUT_QC_FILENAME",
    "GRAPH_SIDES",
    "SK_QC_TSV_MARK",
    "REASON_MISSING_GRAPH_STEM",
    "REASON_NOT_IN_SK_QC",
    "REASON_EXCLUDED_BY_SK_QC",
    "REASON_NO_SUBJECT_FOLDER",
    "REASON_INVALID_SK_QC",
    "OUTPUT_QC_FILENAME",
    "CROP_STATUS_PARTIAL",
    "SkipLists",
    "_ExplicitLabellingSession",
    "ROOT_FORBIDDEN_SUBSTRINGS",
    "list_side_volumes",
    "call_whole_brain_step",
    "labelled_session_dir",
    "resolve_labelling_session",
    "resolve_graph_paths",
    "resolve_bids_graph_paths",
    "fetch_labelling_qc_row",
    "list_input_subjects",
    "fetch_sk_qc_ids",
    "has_hemisphere_graph",
    "compute_sk_qc_reason",
    "compute_end_status",
    "fetch_input_qc_rows",
    "find_dotted_ids",
    "find_prefix_id_pairs",
    "exclusion_warning",
    "RunCorticalTiles",
    "main",
)

ids = st.text(alphabet=string.ascii_letters + string.digits + "._-", min_size=1, max_size=8).filter(
    lambda name: name not in (".", "..")
)
texts = st.text(alphabet=string.ascii_letters + string.digits + "/_-*.", max_size=12)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """run_cortical_tiles.py at REFERENCE_COMMIT, read with git show and loaded as a separate module."""
    shown = subprocess.run(
        ["git", "show", f"{REFERENCE_COMMIT}:{SOURCE_PATH}"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert shown.returncode == 0, f"reference {REFERENCE_COMMIT} not readable: {shown.stderr}"
    path = tmp_path_factory.mktemp("reference") / "run_cortical_tiles_reference.py"
    path.write_text(shown.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("run_cortical_tiles_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_folders(root: str, names: list[str]) -> None:
    for name in names:
        os.makedirs(os.path.join(root, name), exist_ok=True)


def test_run_cortical_tiles_source_at_most_750_lines():
    """run_cortical_tiles.py has at most 750 lines; the count goes to $ALM_MEASURES."""
    lines = len((REPO_ROOT / SOURCE_PATH).read_text(encoding="utf-8").splitlines())
    measures = os.environ.get("ALM_MEASURES")
    if measures:
        with open(measures, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({MEASURE: lines}) + "\n")
    assert lines <= MAX_SOURCE_LINES, f"{SOURCE_PATH} has {lines} lines, over {MAX_SOURCE_LINES}"


def test_reference_module_names_importable():
    """Each 3039b13 module-level name still resolves on champollion_pipeline.run_cortical_tiles."""
    missing = [name for name in REFERENCE_NAMES if not hasattr(current, name)]
    assert len(set(REFERENCE_NAMES)) == 44
    assert missing == [], f"names no longer importable from champollion_pipeline.run_cortical_tiles: {missing}"


@given(subject=ids, sk_qc=st.none() | st.dictionaries(ids, st.integers(0, 1), max_size=4), is_refused=st.booleans())
def test_compute_sk_qc_reason_matches_reference(reference, subject, sk_qc, is_refused):
    expected = reference.compute_sk_qc_reason(subject, sk_qc, is_refused)
    assert current.compute_sk_qc_reason(subject, sk_qc, is_refused) == expected


@given(status=st.integers(-3, 10), crop_qc_abs=texts)
def test_compute_end_status_matches_reference(reference, status, crop_qc_abs):
    assert current.compute_end_status(status, crop_qc_abs) == reference.compute_end_status(status, crop_qc_abs)


@given(
    excluded=st.dictionaries(st.sampled_from(EXCLUSION_REASONS), st.lists(ids, max_size=3), max_size=3),
    path_to_graph=texts,
    candidates=st.tuples(ids, ids),
)
def test_exclusion_warning_matches_reference(reference, excluded, path_to_graph, candidates):
    expected = reference.exclusion_warning(excluded, path_to_graph, candidates)
    assert current.exclusion_warning(excluded, path_to_graph, candidates) == expected


@given(morpho_dir=texts, path_to_graph=texts, subject=ids, session=ids, bids=st.booleans())
def test_labelled_session_dir_matches_reference(reference, morpho_dir, path_to_graph, subject, session, bids):  # noqa: PLR0913 - one argument per generated value
    args = (morpho_dir, path_to_graph, subject, session, bids)
    assert current.labelled_session_dir(*args) == reference.labelled_session_dir(*args)


@FS_SETTINGS
@given(
    side=st.sampled_from(["F", "L", "R"]),
    filename=st.sampled_from(["resampled_skeleton", "skeleton", "x"]),
    suffixes=st.lists(ids, max_size=4, unique=True),
    others=st.lists(ids, max_size=3, unique=True),
)
def test_list_side_volumes_matches_reference(reference, side, filename, suffixes, others):
    with tempfile.TemporaryDirectory() as volumes_dir:
        side_dir = os.path.join(volumes_dir, side)
        os.makedirs(side_dir)
        for name in [f"{side}{filename}{suffix}.nii.gz" for suffix in suffixes] + others:
            Path(side_dir, name).touch()
        expected = reference.list_side_volumes(volumes_dir, filename, side)
        assert current.list_side_volumes(volumes_dir, filename, side) == expected


@FS_SETTINGS
@given(names=st.lists(ids, max_size=6, unique=True))
def test_find_dotted_ids_matches_reference(reference, names):
    with tempfile.TemporaryDirectory() as input_dir:
        _make_folders(input_dir, names)
        assert current.find_dotted_ids(input_dir) == reference.find_dotted_ids(input_dir)


@FS_SETTINGS
@given(names=st.lists(ids, max_size=6, unique=True))
def test_find_prefix_id_pairs_matches_reference(reference, names):
    with tempfile.TemporaryDirectory() as input_dir:
        _make_folders(input_dir, names)
        assert current.find_prefix_id_pairs(input_dir) == reference.find_prefix_id_pairs(input_dir)
