"""Listing order of the pipeline files other Tasks also edit (TASK-READORDER-BDRABCZUK-44FA08B1465D).

User rule: files must always be read in the same order, for reproducibility. A real directory
gives one arbitrary order, so these tests replace what os.scandir and os.listdir yield, below
glob, and stay valid whichever listing call the code uses.

Audit of generate_labelling_qc.py, run_cortical_tiles.py and snapshot_types/sulcal.py: every
listing there is sorted or read order-free (a length, an any()), except one:

* REQ-READORDER-BDRABCZUK-1B48B916974A - find_white_mesh returns the sorted-first white mesh.

Synthetic subject IDs and synthetic trees only.
"""

import glob
import os
from pathlib import Path

import pytest

from champollion_pipeline.snapshot_types.sulcal import find_white_mesh
from tests.listing_order_helpers import descending as _descending
from tests.listing_order_helpers import rotated as _rotated
from tests.test_listing_order_remaining_files import _listing_in

pytestmark = pytest.mark.unit  # noqa: V107

SUBJECT = "sub-01"
LEFT_MESHES = ("a_Lwhite.gii", "b_Lwhite.gii", "c_Lwhite.gii")

unsorted_orders = pytest.mark.parametrize("order", [_descending, _rotated], ids=["descending", "rotated"])


def _left_graph_with_meshes(root: Path) -> Path:
    """A left sulcal graph path whose segmentation/mesh folder holds the three LEFT_MESHES."""
    analysis = root / SUBJECT / "t1mri" / "default_acquisition" / "default_analysis"
    mesh_dir = analysis / "segmentation" / "mesh"
    mesh_dir.mkdir(parents=True)
    for name in LEFT_MESHES:
        (mesh_dir / name).touch()
    graph = analysis / "folds" / "3.1" / "default_session_auto" / f"L{SUBJECT}_default_session_auto.arg"
    graph.parent.mkdir(parents=True)
    graph.touch()
    return graph


class TestListingOrderControl:
    """The order control above must really reach glob, or the test below proves nothing."""

    @unsorted_orders
    def test_scandir_order_reaches_mesh_glob(self, tmp_path, order):
        graph = _left_graph_with_meshes(tmp_path)
        mesh_dir = graph.parents[3] / "segmentation" / "mesh"
        with _listing_in(order):
            listed = [os.path.basename(p) for p in glob.glob(str(mesh_dir / "*Lwhite.gii"))]
        assert listed == order(list(LEFT_MESHES))


class TestFindWhiteMeshOrder:
    """REQ-READORDER-BDRABCZUK-1B48B916974A."""

    @unsorted_orders
    def test_returns_the_sorted_first_mesh(self, tmp_path, order):
        graph = _left_graph_with_meshes(tmp_path)
        expected = str(graph.parents[3] / "segmentation" / "mesh" / sorted(LEFT_MESHES)[0])
        with _listing_in(order):
            assert find_white_mesh(str(graph)) == expected
