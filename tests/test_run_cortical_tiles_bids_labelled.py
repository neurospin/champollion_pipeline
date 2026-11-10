#!/usr/bin/env python3
"""
--bids labelled-graph lookups of the whole-brain step of run_cortical_tiles.py (TASK-INPUTKIND-BDRABCZUK-277A9E5DC923).

A --bids stage 1 tree holds each scan at <input>/sub-<label>/t1mri/<acquisition folder>/, the folder text taken verbatim
from the file name, and stage 2 names the fused volume after cortical_tiles' session_key: sub-<label> followed by the
ses, acq and run entities found in the graph path, in that order. The folder text and the key part of the id differ for
default_acquisition (no key part), rec-x_run-1 (key run-1) and run-1_ses-2 (key ses-2_run-1), so rebuilding the folder
from the id finds nothing there. The labelled graphs must be looked up in the folder whose session_key gives the id.
"""

from pathlib import Path

import pytest

from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from champollion_pipeline.utils.qc_report import read_qc_tsv
from tests.stage2_layout import add_graphs, add_labelled_sessions
from tests.test_run_cortical_tiles import (  # noqa: F401
    _make_skeletons,
    _run_whole_brain_step,
    _stub_whole_brain_functions,  # noqa: V103 - autouse fixture, as in the source module
)

pytestmark = pytest.mark.unit  # noqa: V107

_PATH_TO_GRAPH = "t1mri/*/default_analysis/folds/3.1"
_DEEPCNN = "deepcnn_session_auto"
_DEFAULT_SESSION = "default_session_auto"

_ID_AND_FOLDER = pytest.mark.parametrize(
    ("fused_id", "folder"),
    [
        ("sub-01", "default_acquisition"),
        ("sub-01_ses-V1", "ses-V1"),
        ("sub-01_run-1", "rec-x_run-1"),
        ("sub-01_ses-2_run-1", "run-1_ses-2"),
    ],
    ids=["default_acquisition", "folder_text_is_the_key", "entity_outside_the_key", "non_canonical_order"],
)


def _make_bids_scans(input_dir, output_dir, fused_id, folders):
    """Lay out the fused volume ``fused_id`` of a --bids subject and its acquisition folders.

    ``folders`` is ``{acquisition folder: [labelling sessions holding both labelled graphs]}``; each folder lies at
    ``<input>/<subject>/t1mri/<folder>/default_analysis/folds/3.1`` with one graph per hemisphere, and its labelled
    graphs at ``<session>/<side><subject>_<session>.arg`` below, as Morphologist names them after the subject alone.
    """
    subject = fused_id.split("_", 1)[0]
    (Path(input_dir) / subject).mkdir(parents=True, exist_ok=True)
    for folder, sessions in folders.items():
        graph_dir = _PATH_TO_GRAPH.replace("*", folder)
        add_graphs(input_dir, subject, graph_dir)
        add_labelled_sessions(input_dir, subject, graph_dir, sessions)
    _make_skeletons(output_dir, fused_id)


def _run_bids_step(temp_dir, volumes):
    """Run the whole-brain step with --bids on ``volumes`` (``{fused id: folders of _make_bids_scans}``)."""
    return _run_whole_brain_step(
        temp_dir, volumes, ["--bids"], path_to_graph=_PATH_TO_GRAPH, make_subject=_make_bids_scans
    )


def _labelling_qc_rows(temp_dir):
    """Rows of <output>/derivatives/labelling_qc.tsv, keyed by participant_id."""
    path = Path(temp_dir) / "output" / DERIVATIVES_FOLDER / "labelling_qc.tsv"
    return {row["participant_id"]: row for row in read_qc_tsv(path)}


@_ID_AND_FOLDER
def test_labelling_qc_row_reads_the_folder_of_the_volume_id(temp_dir, fused_id, folder):
    """REQ-INPUTKIND-BDRABCZUK-070BCE15177A: both labelled graphs of the one matching folder give qc 1."""
    _run_bids_step(temp_dir, {fused_id: {folder: [_DEEPCNN]}})

    row = _labelling_qc_rows(temp_dir)[fused_id]
    assert (row["qc"], row["reason"]) == (1, "")


@_ID_AND_FOLDER
def test_remove_ventricle_gets_the_session_of_the_volume_id_folder(temp_dir, fused_id, folder):
    """REQ-INPUTKIND-BDRABCZUK-860D5A37707C: the session found in that folder, here the second candidate."""
    recorder, _, _ = _run_bids_step(temp_dir, {fused_id: {folder: [_DEFAULT_SESSION]}})

    assert recorder.sessions_by_subject == {fused_id: [_DEFAULT_SESSION]}


@pytest.mark.parametrize(
    "folders",
    [
        {"ses-V2": [_DEEPCNN]},
        {"ses-V1_rec-a": [_DEEPCNN], "ses-V1_rec-b": [_DEEPCNN]},
        {"ses-V1": [_DEEPCNN], "ses-V1_rec-a": [_DEEPCNN]},
    ],
    ids=["no_folder_of_the_key", "two_folders_of_the_key", "key_text_folder_and_another_folder_of_the_key"],
)
def test_no_single_folder_of_the_volume_id_gives_qc_0(temp_dir, folders):
    """REQ-INPUTKIND-BDRABCZUK-6C880C2C9EF7: none or several folders of the key, never a guess between them."""
    _run_bids_step(temp_dir, {"sub-01_ses-V1": folders})

    row = _labelling_qc_rows(temp_dir)["sub-01_ses-V1"]
    assert (row["qc"], row["reason"]) == (0, "no_session_dir")
