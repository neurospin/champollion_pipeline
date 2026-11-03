"""Hemisphere of a sulcal graph from the leading side letter of its file name.

Morphologist names a sulcal graph ``<L|R><subject>[_<session>].arg``
(see generate_labelling_qc). The side is the first character of the file name,
never a ``_r``/``right`` substring elsewhere in it (audit H4).
TASK-PATTERNS-BDRABCZUK-C53583AE2653; requirements named per class.
Synthetic file names and empty files only.
"""

import logging

import pytest

from champollion_pipeline.generate_snapshots import _detect_hemi
from champollion_pipeline.snapshot_types.sulcal import find_sulcal_graphs_by_hemi, find_white_mesh

SUBJECT = "sub_r001"

# Left graphs whose name holds "_r", "right" or a lowercase leading "r" after the side letter.
LEFT_NAMES = ["Lsub_r001.arg", "Lsub-01_run-1.arg", "Lright_sulci.arg", "Lrsub.arg"]
RIGHT_NAMES = ["Rsub_l001.arg", "Rsub-01_run-1.arg", "Rleft_sulci.arg"]
# No capital L or R as first character.
NO_SIDE_NAMES = ["sub_Rwhite.arg", "right_folds.arg", "rsub-01.arg", "lsub-01.arg", "graph.arg", "_Lsub.arg"]


def _graph(morphologist_dir, name, subject=SUBJECT):
    """Create an empty graph under a Morphologist 6.0 analysis folder and return its path."""
    analysis = morphologist_dir / subject / "t1mri" / "default_acquisition" / "0"
    folds = analysis / "folds" / "3.1"
    folds.mkdir(parents=True, exist_ok=True)
    path = folds / name
    path.touch()
    return path


def _both_meshes(graph_path):
    """Create the left and right white meshes of the graph's analysis folder."""
    mesh_dir = graph_path.parent.parent.parent / "segmentation" / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    (mesh_dir / f"{SUBJECT}_Lwhite.gii").touch()
    (mesh_dir / f"{SUBJECT}_Rwhite.gii").touch()


class TestDetectHemi:
    """REQ-PATTERNS-BDRABCZUK-FBD2685DECAC, -1511DE259EA2: side from the leading L/R only.

    Moved from test_generate_snapshots_rendering.py (old cases sub_Rwhite/right_folds now read as no side).
    """

    @pytest.mark.parametrize("name", ["Rsulci.arg", "Rleft_folds.arg", *RIGHT_NAMES])
    def test_right_is_detected(self, name):
        assert _detect_hemi(f"/a/b/{name}") == "right"

    @pytest.mark.parametrize("name", ["Lsulci.arg", "Lright_folds.arg", *LEFT_NAMES])
    def test_left_is_detected(self, name):
        assert _detect_hemi(f"/a/b/{name}") == "left"

    @pytest.mark.parametrize("name", NO_SIDE_NAMES)
    def test_no_side_letter_is_none(self, name):
        assert _detect_hemi(f"/a/b/{name}") is None


class TestGroupingBySideLetter:
    """REQ-PATTERNS-BDRABCZUK-FBD2685DECAC."""

    @pytest.mark.parametrize("name", LEFT_NAMES)
    def test_leading_l_is_left(self, tmp_path, name):
        graph = _graph(tmp_path, name)
        assert find_sulcal_graphs_by_hemi(str(tmp_path), SUBJECT) == {"left": [str(graph)], "right": []}

    @pytest.mark.parametrize("name", RIGHT_NAMES)
    def test_leading_r_is_right(self, tmp_path, name):
        graph = _graph(tmp_path, name)
        assert find_sulcal_graphs_by_hemi(str(tmp_path), SUBJECT) == {"left": [], "right": [str(graph)]}


class TestWhiteMeshBySideLetter:
    """REQ-PATTERNS-BDRABCZUK-05B561EEB610."""

    @pytest.mark.parametrize("name", LEFT_NAMES)
    def test_leading_l_selects_lwhite(self, tmp_path, name):
        graph = _graph(tmp_path, name)
        _both_meshes(graph)
        assert find_white_mesh(str(graph)).endswith(f"{SUBJECT}_Lwhite.gii")

    @pytest.mark.parametrize("name", RIGHT_NAMES)
    def test_leading_r_selects_rwhite(self, tmp_path, name):
        graph = _graph(tmp_path, name)
        _both_meshes(graph)
        assert find_white_mesh(str(graph)).endswith(f"{SUBJECT}_Rwhite.gii")


class TestGraphWithoutSideLetter:
    """REQ-PATTERNS-BDRABCZUK-1511DE259EA2, -C6CAEB2FB7C7, -80C80C99DA42."""

    @pytest.mark.parametrize("name", NO_SIDE_NAMES)
    def test_left_out_of_both_lists(self, tmp_path, name):
        """REQ-PATTERNS-BDRABCZUK-1511DE259EA2."""
        _graph(tmp_path, name)
        assert find_sulcal_graphs_by_hemi(str(tmp_path), SUBJECT) == {"left": [], "right": []}

    @pytest.mark.parametrize("name", NO_SIDE_NAMES)
    def test_one_warning_names_the_graph(self, tmp_path, name, pipeline_caplog):
        """REQ-PATTERNS-BDRABCZUK-C6CAEB2FB7C7."""
        graph = _graph(tmp_path, name)
        find_sulcal_graphs_by_hemi(str(tmp_path), SUBJECT)
        warnings = [
            r
            for r in pipeline_caplog.records
            if r.levelno == logging.WARNING
            and r.name == "champollion_pipeline.snapshot_types.sulcal"
            and str(graph) in r.getMessage()
        ]
        assert len(warnings) == 1

    def test_sided_graph_gets_no_warning(self, tmp_path, pipeline_caplog):
        """REQ-PATTERNS-BDRABCZUK-C6CAEB2FB7C7: the warning is for side-less names only."""
        _graph(tmp_path, "Lsub_r001.arg")
        _graph(tmp_path, "Rsub_r001.arg")
        find_sulcal_graphs_by_hemi(str(tmp_path), SUBJECT)
        assert not [r for r in pipeline_caplog.records if r.levelno == logging.WARNING]

    @pytest.mark.parametrize("name", NO_SIDE_NAMES)
    def test_white_mesh_is_none(self, tmp_path, name):
        """REQ-PATTERNS-BDRABCZUK-80C80C99DA42."""
        graph = _graph(tmp_path, name)
        _both_meshes(graph)
        assert find_white_mesh(str(graph)) is None
