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
