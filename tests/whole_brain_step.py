"""Laying out subjects for, and running, the whole-brain step of RunCorticalTiles.run() with its cortical_tiles
calls faked.

Shared by tests/test_run_cortical_tiles.py, tests/test_run_cortical_tiles_whole_brain.py and their sibling
test_run_cortical_tiles_*.py files (moved out of test_run_cortical_tiles.py by TASK-CODE-BDRABCZUK-5F39ABE65D2D).
"""

import logging
import re
import warnings
from pathlib import Path
from unittest.mock import patch

from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from tests.stage2_layout import (
    REMOVE_VENTRICLE_SIGNATURE,
    add_graphs,
    add_labelled_sessions,
    write_whole_brain_volumes,
)

_GRAPH_SUBPATH = "graphs/path"
_FIXTURE_SUBJECT = "subj01"


def _skeleton_dir(output_dir):
    """Where generate_sulcal_regions.py writes L/R skeletons and the fusion writes F/."""
    return Path(output_dir).resolve() / DERIVATIVES_FOLDER / "skeletons" / "2mm"


def _make_skeletons(output_dir, subject):
    """Write the subject's L, R and fused F resampled skeletons under the skeleton directory."""
    for side in ("L", "R", "F"):
        side_dir = _skeleton_dir(output_dir) / side
        side_dir.mkdir(parents=True, exist_ok=True)
        (side_dir / f"{side}resampled_skeleton_{subject}.nii.gz").touch()


def _make_subject(input_dir, output_dir, subject, sessions):
    """Lay out one subject on disk as run() finds it after the crop and fusion steps.

    The subject holds its two Morphologist graphs in <path_to_graph>. ``sessions``
    lists the Morphologist labelling sessions holding a labelled graph for the
    subject, at
    ``<input>/<subject>/<path_to_graph>/<session>/<side><subject>_<session>.arg``
    (the layout cortical_tiles' remove_ventricle reads). The subject's L, R and
    fused F resampled skeletons are written under the skeleton directory.
    """
    add_graphs(input_dir, subject, _GRAPH_SUBPATH)
    add_labelled_sessions(input_dir, subject, _GRAPH_SUBPATH, sessions)
    _make_skeletons(output_dir, subject)


def _add_fixture_subject(input_dir, output_dir):
    """One subject labelled under the default and the '0_auto' sessions, so the
    whole-brain step has a subject to process whichever session is requested."""
    _make_subject(input_dir, output_dir, _FIXTURE_SUBJECT, ["deepcnn_session_auto", "0_auto"])


def _parsed_fixture_script(temp_dir, extra_args=None):
    """Lay the fixture subject out under temp_dir/input and temp_dir/output, then return
    (script, input_dir, output_dir) with a RunCorticalTiles parsed on them plus extra_args."""
    input_dir, output_dir = (Path(temp_dir) / name for name in ("input", "output"))
    for folder in (input_dir, output_dir):
        folder.mkdir()
    _add_fixture_subject(input_dir, output_dir)
    script = RunCorticalTiles()
    positional = [str(input_dir), str(output_dir)]
    script.parse_args(
        positional + ["--path_to_graph", _GRAPH_SUBPATH, "--path_sk_with_hull", "skeleton/path"] + (extra_args or [])
    )
    return script, input_dir, output_dir


class _RemoveVentricleRecorder:
    """Fake ``remove_ventricle``: records, per subject, the labelling session of
    every call that would process that subject.

    A call processes the subjects whose fused volume
    ``<src_dir>/<side>/<side><src_filename>_<subject>.nii.gz`` exists when it
    is made -- the same rule cortical_tiles' ``RemoveVentricleFromVolume.compute``
    applies. This keeps the tests independent of how run() groups subjects
    (one call, one call per session group, staged source directories).
    """

    def __init__(self):
        self.sessions_by_subject = {}
        self.path_to_graph_by_subject = {}

    def __call__(self, *args, **kwargs):
        write_whole_brain_volumes(*args, **kwargs)
        bound = REMOVE_VENTRICLE_SIGNATURE.bind(*args, **kwargs)
        bound.apply_defaults()
        params = bound.arguments
        side = params["side"]
        side_dir = Path(params["src_dir"]) / side
        pattern = re.compile(rf"^{re.escape(side + params['src_filename'])}(.*)\.nii\.gz$")
        if not side_dir.is_dir():
            return
        for path in sorted(side_dir.iterdir()):
            match = pattern.match(path.name)
            if match:
                subject = match[1][1:] if match[1].startswith("_") else match[1]
                self.sessions_by_subject.setdefault(subject, []).append(params["labelling_session"])
                self.path_to_graph_by_subject.setdefault(subject, []).append(params["path_to_graph"])


def _run_whole_brain_step(temp_dir, subjects, extra_args=None, *, path_to_graph=_GRAPH_SUBPATH, make_subject=None):
    """Run run() with the crop subprocess and fusion mocked, ``subjects`` laid out
    on disk (``{subject: [sessions holding a labelled graph]}``), and
    ``remove_ventricle`` replaced by a ``_RemoveVentricleRecorder``.

    ``make_subject(input_dir, output_dir, subject, value)`` lays out one subject
    (default ``_make_subject``); ``path_to_graph`` is the --path_to_graph value.

    Returns the recorder, the Python warnings emitted during run() and run()'s return value.
    """
    input_dir = Path(temp_dir) / "input"
    output_dir = Path(temp_dir) / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    for subject, value in subjects.items():
        (make_subject or _make_subject)(input_dir, output_dir, subject, value)

    script = RunCorticalTiles()
    script.parse_args(
        [
            str(input_dir),
            str(output_dir),
            "--path_to_graph",
            path_to_graph,
            "--path_sk_with_hull",
            "skeleton/path",
        ]
        + (extra_args or [])
    )
    recorder = _RemoveVentricleRecorder()

    with (
        warnings.catch_warnings(record=True) as caught,
        patch.object(script, "validate_paths", return_value=True),
        patch.object(script, "execute_command", return_value=0),
        patch.object(script, "_generate_mask_npys"),
        patch("champollion_pipeline.run_cortical_tiles.add_left_and_right_volumes.add_left_and_right_volumes"),
        patch("champollion_pipeline.run_cortical_tiles.remove_ventricle.remove_ventricle", side_effect=recorder),
    ):
        warnings.simplefilter("always")
        result = script.run()

    return recorder, [str(w.message) for w in caught], result


_LOGGER_NAME = "champollion_pipeline.run_cortical_tiles"


def _warnings_mentioning(text, pipeline_caplog):
    """Messages of the WARNING records of the run_cortical_tiles logger that mention ``text``
    (case-insensitive). Printed lines and Python warnings no longer count (TASK-LOGGING-BDRABCZUK-940DB0BFBA89)."""
    logged = [
        r.getMessage() for r in pipeline_caplog.records if r.name == _LOGGER_NAME and r.levelno == logging.WARNING
    ]
    return [m for m in logged if text.lower() in m.lower()]


_WILDCARD_PATH_TO_GRAPH = "t1mri/*/default_analysis/folds/3.1"
_WILDCARD_SESSION = "deepcnn_session_auto"


def _resolved_graph_path(acquisition):
    """``_WILDCARD_PATH_TO_GRAPH`` with its ``*`` replaced by ``acquisition``."""
    return _WILDCARD_PATH_TO_GRAPH.replace("*", acquisition)


def _make_wildcard_subject(input_dir, output_dir, subject, acquisitions):
    """Lay out one subject whose graphs and labelled graphs sit under one folder per entry of
    ``acquisitions``, each matching ``_WILDCARD_PATH_TO_GRAPH``:
    ``<input>/<subject>/t1mri/<acquisition>/default_analysis/folds/3.1/<session>/<side><subject>_<session>.arg``.
    The subject folder exists even when ``acquisitions`` is empty. The L, R and
    fused F resampled skeletons are written under the skeleton directory.
    """
    (Path(input_dir) / subject).mkdir(parents=True)
    for acquisition in acquisitions:
        add_graphs(input_dir, subject, _resolved_graph_path(acquisition))
        session_dir = Path(input_dir) / subject / _resolved_graph_path(acquisition) / _WILDCARD_SESSION
        session_dir.mkdir(parents=True)
        for side in ("L", "R"):
            (session_dir / f"{side}{subject}_{_WILDCARD_SESSION}.arg").touch()
    _make_skeletons(output_dir, subject)


def _run_wildcard_step(temp_dir, subjects):
    """``_run_whole_brain_step`` without --bids, --path_to_graph holding a ``*``,
    ``subjects`` given as ``{subject: [folder names the * matches]}``."""
    return _run_whole_brain_step(
        temp_dir, subjects, path_to_graph=_WILDCARD_PATH_TO_GRAPH, make_subject=_make_wildcard_subject
    )
