"""Listing order of models/checkpoint.py and generate_masks.py (TASK-READORDER-BDRABCZUK-82FC63FE0B88).

User rule: files must always be read in the same order, for reproducibility. A real directory
gives one arbitrary order, so these tests replace what os.scandir and os.listdir yield, below
glob and pathlib, and stay valid whichever listing call the code uses.

* REQ-READORDER-BDRABCZUK-58F21D92692F - weights_source fallback names the sorted-first checkpoint.
* REQ-READORDER-BDRABCZUK-CD37E1955F33 - find_epoch_checkpoint does not depend on listing order.
* REQ-READORDER-BDRABCZUK-22C052E7570F - find_region_model_dirs does not depend on listing order.
* REQ-READORDER-BDRABCZUK-2ACA5ECE46B9 - _graph_matches returns sorted paths.
"""

import contextlib
import glob
import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline import generate_masks as gm
from champollion_pipeline.models.checkpoint import find_epoch_checkpoint, find_region_model_dirs, weights_source
from tests.listing_order_helpers import Order
from tests.listing_order_helpers import ascending as _ascending
from tests.listing_order_helpers import descending as _descending
from tests.listing_order_helpers import rotated as _rotated

EPOCH = 80
VOXEL_SIZES = (2.0, 2.0, 2.0)

unsorted_orders = pytest.mark.parametrize("order", [_descending, _rotated], ids=["descending", "rotated"])
every_order = pytest.mark.parametrize(
    "order", [_ascending, _descending, _rotated], ids=["ascending", "descending", "rotated"]
)


@contextlib.contextmanager
def _listing_in(order: Order) -> Iterator[None]:
    """Make os.scandir and os.listdir yield their entries in the given order of names."""
    real_scandir = os.scandir
    real_listdir = os.listdir

    @contextlib.contextmanager
    def scandir(path="."):
        with real_scandir(path) as entries:
            by_name = {entry.name: entry for entry in entries}
        yield iter([by_name[name] for name in order(list(by_name))])

    def listdir(path="."):
        return order(real_listdir(path))

    with patch("os.scandir", scandir), patch("os.listdir", listdir):
        yield


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return str(path)


def _ckpt_path(model_dir: Path, version: str, name: str) -> Path:
    return model_dir / "logs" / "lightning_logs" / version / "checkpoints" / name


@pytest.mark.unit
class TestListingOrderControl:
    """The order control above must really reach the calls, or every test below proves nothing."""

    @unsorted_orders
    def test_scandir_order_reaches_glob_and_pathlib(self, tmp_path, order):
        names = ["a.ckpt", "b.ckpt", "c.ckpt"]
        for name in names:
            _touch(tmp_path / name)
        with _listing_in(order):
            from_glob = [os.path.basename(p) for p in glob.glob(str(tmp_path / "*.ckpt"))]
            from_pathlib = [p.name for p in tmp_path.glob("*.ckpt")]
        assert from_glob == order(names)
        assert from_pathlib == order(names)

    @unsorted_orders
    def test_listdir_order_is_controlled(self, tmp_path, order):
        names = ["a", "b", "c"]
        for name in names:
            _touch(tmp_path / name)
        with _listing_in(order):
            assert os.listdir(tmp_path) == order(names)


@pytest.mark.unit
class TestWeightsSourceFallback:
    """checkpoint.py weights_source: ckpts[0] of an unsorted glob is the one call whose order reaches a result."""

    @unsorted_orders
    def test_sorted_first_checkpoint_is_named(self, tmp_path, order):
        paths = [
            _touch(_ckpt_path(tmp_path, "version_0", name))
            for name in ("epoch=3-step=10.ckpt", "epoch=3-step=9.ckpt", "last.ckpt")
        ]
        with _listing_in(order):
            named = weights_source(str(tmp_path), EPOCH)
        assert named == sorted(paths)[0]


@pytest.mark.unit
class TestFindEpochCheckpointOrder:
    """checkpoint.py find_epoch_checkpoint: max() over a total rank, so its unsorted glob stays harmless."""

    @every_order
    def test_step_tie_resolved_by_version_index(self, tmp_path, order):
        name = f"epoch={EPOCH}-step=100.ckpt"
        for version in ("version_0", "version_1", "version_2"):
            _touch(_ckpt_path(tmp_path, version, name))
        winner = _touch(_ckpt_path(tmp_path, "version_10", name))
        _touch(_ckpt_path(tmp_path, "version_11", f"epoch={EPOCH}-step=99.ckpt"))
        with _listing_in(order):
            found = find_epoch_checkpoint(str(tmp_path), EPOCH)
        assert str(found) == winner

    @every_order
    def test_version_index_tie_resolved_by_path(self, tmp_path, order):
        name = f"epoch={EPOCH}-step=100.ckpt"
        _touch(_ckpt_path(tmp_path, "version_01", name))
        winner = _touch(_ckpt_path(tmp_path, "version_1", name))
        with _listing_in(order):
            found = find_epoch_checkpoint(str(tmp_path), EPOCH)
        assert str(found) == winner


def _region_tree_with_aliases(models: Path) -> None:
    """Two regions; one is reachable through two symlinks, of which the walk keeps a single alias."""
    (models / "Other_right" / "logs").mkdir(parents=True)
    (models / "real" / "Region_left" / "logs").mkdir(parents=True)
    (models / "a_alias").symlink_to(models / "real")
    (models / "z_alias").symlink_to(models / "real")


@pytest.mark.unit
class TestFindRegionModelDirsOrder:
    """checkpoint.py find_region_model_dirs: the sorted listing also decides which symlink alias is kept."""

    @unsorted_orders
    def test_symlink_alias_choice_is_stable(self, tmp_path, order):
        _region_tree_with_aliases(tmp_path)
        with _listing_in(_ascending):
            expected = find_region_model_dirs(str(tmp_path))
        with _listing_in(order):
            found = find_region_model_dirs(str(tmp_path))
        assert found == expected

    @unsorted_orders
    def test_result_is_sorted(self, tmp_path, order):
        regions = ("Z_left", "M_right", "A_group/B_left")  # the walk meets the nested region last, sorted puts it first
        for region in regions:
            (tmp_path / region / "logs").mkdir(parents=True)
        with _listing_in(order):
            found = find_region_model_dirs(str(tmp_path))
        assert found == sorted(str(tmp_path / region) for region in regions)


def _subject_with_three_graphs(database: Path) -> tuple[dict, list[str]]:
    """A subject of the database whose graph glob matches three files; return (sub, their paths)."""
    subject = "sub-1"
    graph_dir = database / subject / "folds"
    paths = [_touch(graph_dir / f"L{subject}{suffix}.arg") for suffix in ("_a", "_b", "")]
    sub = {
        "subject": subject,
        "side": "L",
        "dir": str(database),
        "graph_file": gm._graph_glob(subject, "L", "folds"),
    }
    return sub, paths


@pytest.mark.unit
class TestGraphMatchesOrder:
    """generate_masks.py _graph_matches: sorted one line after the glob, so the lint rule cannot see it."""

    @unsorted_orders
    def test_matches_are_sorted(self, tmp_path, order):
        sub, paths = _subject_with_three_graphs(tmp_path)
        with _listing_in(order):
            matches = gm._graph_matches(sub)
        assert matches == sorted(paths)

    @unsorted_orders
    def test_ambiguity_error_lists_sorted_paths(self, tmp_path, order):
        sub, paths = _subject_with_three_graphs(tmp_path)
        with _listing_in(order), patch.object(gm, "aims", MagicMock()), pytest.raises(RuntimeError) as raised:
            gm._load_and_extract_subject(sub, set(), VOXEL_SIZES)
        assert str(raised.value).endswith(", ".join(sorted(paths)))
