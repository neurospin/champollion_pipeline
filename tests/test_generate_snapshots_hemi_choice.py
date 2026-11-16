"""Deterministic hemisphere graph choice of GenerateSnapshots._run_sulcal (no hemi_graphs[0])."""

import logging
from unittest.mock import patch

from champollion_pipeline import generate_snapshots as gs
from champollion_pipeline.generate_snapshots import GenerateSnapshots


def _script(tmp_path, morpho, extra=None):
    args = ["--output_dir", str(tmp_path / "out"), "--morphologist_dir", str(morpho)] + (extra or [])
    script = GenerateSnapshots()
    script.parse_args(args)
    return script


def _folds(morpho, acq, files, subject=None):
    base = morpho / subject if subject else morpho
    folds = base / "t1mri" / acq / "default_analysis" / "folds"
    folds.mkdir(parents=True)
    for name in files:
        (folds / name).touch()
    return folds


def _run(script):
    with patch.object(gs, "generate_sulcal_graph_snapshot", side_effect=lambda *a, **k: a[1]) as render:
        return script._run_sulcal((800, 600)), render


class TestAmbiguousHemisphere:
    def test_no_snapshot_for_ambiguous_hemisphere(self, tmp_path):
        morpho = tmp_path / "morpho"
        _folds(morpho, "wk30", ["Lsulci.arg", "Rsulci.arg"])
        _folds(morpho, "wk40", ["Lsulci.arg"])
        snaps, _ = _run(_script(tmp_path, morpho))
        assert [s for s in snaps if s.endswith("_left.png")] == []

    def test_error_names_graph_paths_and_option(self, tmp_path, caplog):
        morpho = tmp_path / "morpho"
        a = _folds(morpho, "wk30", ["Lsulci.arg"])
        b = _folds(morpho, "wk40", ["Lsulci.arg"])
        with caplog.at_level(logging.DEBUG):
            _run(_script(tmp_path, morpho))
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        msg = errors[0].getMessage()
        assert str(a / "Lsulci.arg") in msg and str(b / "Lsulci.arg") in msg
        assert "--acquisition" in msg

    def test_unambiguous_hemisphere_is_still_rendered(self, tmp_path):
        morpho = tmp_path / "morpho"
        _folds(morpho, "wk30", ["Lsulci.arg", "Rsulci.arg"])
        _folds(morpho, "wk40", ["Lsulci.arg"])
        snaps, _ = _run(_script(tmp_path, morpho))
        assert [s for s in snaps if s.endswith("_right.png")]

    def test_acquisition_selects_single_graph(self, tmp_path, caplog):
        morpho = tmp_path / "morpho"
        _folds(morpho, "wk30", ["Lsulci.arg"])
        _folds(morpho, "wk40", ["Lsulci.arg"])
        with caplog.at_level(logging.DEBUG):
            snaps, _ = _run(_script(tmp_path, morpho, ["--acquisition", "wk40"]))
        assert len(snaps) == 1
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


DEEPCNN = "deepcnn_session_auto"
DEFAULT = "default_session_auto"


def _session_folds(morpho, subject, sessions, sides=("L", "R")):
    """Morphologist folds/3.1 of ``subject``: an unlabelled graph per side plus one graph per labelling session."""
    folds = morpho / subject / "t1mri" / "default_acquisition" / "default_analysis" / "folds" / "3.1"
    folds.mkdir(parents=True)
    for side in sides:
        (folds / f"{side}{subject}.arg").touch()
        for session in sessions:
            (folds / session).mkdir(exist_ok=True)
            (folds / session / f"{side}{subject}_{session}.arg").touch()
    return folds


def _rendered_graphs(render):
    """Graph paths the stand-in sulcal render was called with, in call order."""
    return [str(call.args[0]) for call in render.call_args_list]


class TestRehearsalLayout:
    """REQ-SNAPSULC-BDRABCZUK-9577162BD735: the rehearsal layout (unlabelled + deepcnn graph) is rendered."""

    def test_both_hemispheres_rendered_from_the_session_graph(self, tmp_path):
        morpho = tmp_path / "morpho"
        folds = _session_folds(morpho, "sub-002", (DEEPCNN,))
        _, render = _run(_script(tmp_path, morpho))
        assert sorted(_rendered_graphs(render)) == sorted(
            str(folds / DEEPCNN / f"{side}sub-002_{DEEPCNN}.arg") for side in ("L", "R")
        )


class TestLabellingSessionSelection:
    """Labelling session selection of GenerateSnapshots._run_sulcal."""

    def test_given_session_graph_is_rendered(self, tmp_path):
        """REQ-SNAPSULC-BDRABCZUK-9BC15E8B468E: --labelling_session picks that session's graph."""
        morpho = tmp_path / "morpho"
        folds = _session_folds(morpho, "sub-002", (DEEPCNN, DEFAULT), sides=("L",))
        _, render = _run(_script(tmp_path, morpho, ["--labelling_session", DEFAULT]))
        assert _rendered_graphs(render) == [str(folds / DEFAULT / f"Lsub-002_{DEFAULT}.arg")]

    def test_auto_prefers_the_deepcnn_session(self, tmp_path):
        """REQ-SNAPSULC-BDRABCZUK-2D134733B8A2: without the option, deepcnn_session_auto comes first."""
        morpho = tmp_path / "morpho"
        folds = _session_folds(morpho, "sub-002", (DEEPCNN, DEFAULT), sides=("L",))
        _, render = _run(_script(tmp_path, morpho))
        assert _rendered_graphs(render) == [str(folds / DEEPCNN / f"Lsub-002_{DEEPCNN}.arg")]

    def test_auto_takes_the_default_session_without_deepcnn(self, tmp_path):
        """REQ-SNAPSULC-BDRABCZUK-2D134733B8A2: without deepcnn_session_auto, default_session_auto is taken."""
        morpho = tmp_path / "morpho"
        folds = _session_folds(morpho, "sub-002", (DEFAULT,), sides=("L",))
        _, render = _run(_script(tmp_path, morpho))
        assert _rendered_graphs(render) == [str(folds / DEFAULT / f"Lsub-002_{DEFAULT}.arg")]


class TestAmbiguityMessage:
    """REQ-SNAPSULC-BDRABCZUK-192418778A2A: the ambiguity ERROR lists the rendered subject's acquisitions."""

    def test_error_lists_the_rendered_subject_acquisitions(self, tmp_path, caplog):
        morpho = tmp_path / "morpho"
        graphs = [_folds(morpho, acq, ["Lsulci.arg"], subject="sub-001") / "Lsulci.arg" for acq in ("wk30", "wk40")]
        with caplog.at_level(logging.DEBUG):
            _run(_script(tmp_path, morpho))
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1, errors
        # The graph paths hold the acquisition names too: only the text outside them counts.
        listing = errors[0]
        for graph in graphs:
            listing = listing.replace(str(graph), "")
        assert "wk30" in listing and "wk40" in listing, errors[0]
