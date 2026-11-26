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
    """Several graphs of a hemisphere: only the first in natural path order is rendered, for now.

    REQ-ARGDISC-BDRABCZUK-C1E227A48539, -3A36B34AB8C8 (supersede REQ-PATTERNS C3BF41ED43D3, 0D651D3AE998).
    """

    def test_first_graph_of_the_hemisphere_is_rendered(self, tmp_path):
        morpho = tmp_path / "morpho"
        a = _folds(morpho, "wk30", ["Lsulci.arg", "Rsulci.arg"])
        _folds(morpho, "wk40", ["Lsulci.arg"])
        _, render = _run(_script(tmp_path, morpho))
        left = [str(call.args[0]) for call in render.call_args_list if str(call.args[1]).endswith("_left.png")]
        assert left == [str(a / "Lsulci.arg")]

    def test_left_out_graph_is_logged_at_info_and_no_error(self, tmp_path, caplog):
        morpho = tmp_path / "morpho"
        _folds(morpho, "wk30", ["Lsulci.arg"])
        b = _folds(morpho, "wk40", ["Lsulci.arg"])
        with caplog.at_level(logging.DEBUG):
            _run(_script(tmp_path, morpho))
        infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert [m for m in infos if str(b / "Lsulci.arg") in m]
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

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
