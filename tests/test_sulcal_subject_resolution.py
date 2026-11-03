"""Subject folder resolution of find_sulcal_graphs (audit M2/M3).

TASK-PATTERNS-BDRABCZUK-5497053A90EA; requirements named per class.
Synthetic folders and empty files only.
"""

import logging

from champollion_pipeline.snapshot_types.sulcal import find_sulcal_graphs

LOGGER = "champollion_pipeline.snapshot_types.sulcal"


def _graph(base, name="Lsulci.arg", acquisition="acq1"):
    """Create an empty graph under ``base/t1mri/<acquisition>/.../folds`` and return its path."""
    folds = base / "t1mri" / acquisition / "default_analysis" / "folds" / "3.1"
    folds.mkdir(parents=True, exist_ok=True)
    path = folds / name
    path.touch()
    return str(path)


class TestAmbiguousSubject:
    """REQ-PATTERNS-BDRABCZUK-60FFC36FED21."""

    def test_two_matches_return_empty_and_warn_once_with_both(self, tmp_path, caplog):
        a = tmp_path / "site_a" / "sub1"
        b = tmp_path / "site_b" / "sub1"
        _graph(a)
        _graph(b)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert find_sulcal_graphs(str(tmp_path), "sub1") == []
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == LOGGER]
        assert len(warnings) == 1
        assert str(a) in warnings[0].getMessage()
        assert str(b) in warnings[0].getMessage()

    def test_single_match_is_used(self, tmp_path):
        graph = _graph(tmp_path / "site_a" / "sub1")
        (tmp_path / "site_b").mkdir()
        assert find_sulcal_graphs(str(tmp_path), "sub1") == [graph]

    def test_direct_child_wins_over_deeper_namesake(self, tmp_path):
        graph = _graph(tmp_path / "sub1")
        _graph(tmp_path / "site_b" / "sub1")
        assert find_sulcal_graphs(str(tmp_path), "sub1") == [graph]


class TestLiteralSubject:
    """REQ-PATTERNS-BDRABCZUK-771916FAC82B."""

    def test_star_is_not_a_wildcard(self, tmp_path):
        _graph(tmp_path / "site" / "sub1")
        assert find_sulcal_graphs(str(tmp_path), "sub*") == []

    def test_bracket_name_matches_itself(self, tmp_path):
        graph = _graph(tmp_path / "site" / "sub[1]")
        assert find_sulcal_graphs(str(tmp_path), "sub[1]") == [graph]

    def test_bracket_class_is_not_expanded(self, tmp_path):
        _graph(tmp_path / "site" / "sub1")
        assert find_sulcal_graphs(str(tmp_path), "sub[1]") == []


class TestSortedOrder:
    """REQ-PATTERNS-BDRABCZUK-F9AB07926C96."""

    def test_graphs_are_sorted(self, tmp_path):
        base = tmp_path / "sub1"
        paths = [_graph(base, name, acq) for name, acq in [("Rb.arg", "z"), ("Lb.arg", "m"), ("La.arg", "a")]]
        assert find_sulcal_graphs(str(tmp_path), "sub1") == sorted(paths)
        assert find_sulcal_graphs(str(tmp_path)) == sorted(paths)


class TestFiltersBelowRoot:
    """REQ-PATTERNS-BDRABCZUK-F1ADD276899B."""

    def test_root_name_with_folds_does_not_admit_other_graphs(self, tmp_path):
        root = tmp_path / "folds_root"
        base = root / "sub1" / "t1mri" / "acq1" / "default_analysis" / "segmentation"
        base.mkdir(parents=True)
        (base / "Lgraph.arg").touch()
        assert find_sulcal_graphs(str(root), "sub1") == []

    def test_root_name_with_acquisition_does_not_admit_other_acquisitions(self, tmp_path):
        root = tmp_path / "wk30"
        _graph(root / "sub1", acquisition="wk40")
        assert find_sulcal_graphs(str(root), "sub1", "wk30") == []

    def test_acquisition_below_root_still_selected(self, tmp_path):
        graph = _graph(tmp_path / "sub1", acquisition="wk30")
        _graph(tmp_path / "sub1", name="Lother.arg", acquisition="wk40")
        assert find_sulcal_graphs(str(tmp_path), "sub1", "wk30") == [graph]
