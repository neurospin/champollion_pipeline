#!/usr/bin/env python3
"""
Tests for ``champollion_pipeline.file_indexer.btree.BTree``.

The B-tree indexes relative paths (string keys, dict values) so the
eligibility checker can answer prefix range queries per scan. Random
insert and delete sequences at minimum degree 2 and 3 force node splits,
borrows and merges.

Covers REQ-STREAMING-BDRABCZUK-872DD77EBCB0 (search after insert),
-543F6F34ADC6 (absent key), -0311B0F112B9 (duplicate insert),
-8235325CAD21 (ordered iteration), -A58FC71229E7 (range query),
-68E610E937CD (delete), -C0579BE1BCCF (delete of absent key),
-1AFD37473B9E (len), -261EA669234A (minimum degree), -F7DFC1713553
(JSON round trip), -242E1CA48208 (node occupancy) and -8516AEEB7662
(leaf depth).
"""

import json
import random

import pytest

from champollion_pipeline.file_indexer.btree import BTree

DEGREES = [2, 3]


def _keys(count: int, seed: int = 0) -> list[str]:
    keys = [f"sub-{i:04d}/ses-01/anat/sub-{i:04d}_T1w.nii.gz" for i in range(count)]
    random.Random(seed).shuffle(keys)
    return keys


def _filled(count: int, t: int = 2, seed: int = 0) -> tuple[BTree, dict[str, dict]]:
    tree = BTree(t=t)
    pairs = {key: {"index": i} for i, key in enumerate(_keys(count, seed))}
    for key, value in pairs.items():
        tree.insert(key, value)
    return tree, pairs


def _delete_half(tree: BTree, pairs: dict[str, dict], seed: int = 1) -> dict[str, dict]:
    doomed = sorted(pairs)
    random.Random(seed).shuffle(doomed)
    doomed = doomed[: len(doomed) // 2]
    for key in doomed:
        tree.delete(key)
    return {key: value for key, value in pairs.items() if key not in set(doomed)}


def _letters() -> BTree:
    tree = BTree(t=2)
    for key in "gbefcadh":
        tree.insert(key, {"key": key})
    return tree


def _nodes(node: dict, depth: int = 0):
    yield node, depth
    for child in node["children"]:
        yield from _nodes(child, depth + 1)


@pytest.mark.unit
class TestBTreeInsertSearch:
    """REQ-STREAMING-BDRABCZUK-872DD77EBCB0, -543F6F34ADC6 and -0311B0F112B9."""

    @pytest.mark.parametrize("t", DEGREES)
    def test_search_returns_inserted_value(self, t):
        tree, pairs = _filled(200, t=t)
        for key, value in pairs.items():
            assert tree.search(key) == value

    def test_search_absent_key_returns_none(self):
        tree, _ = _filled(50)
        assert tree.search("sub-9999/ses-01/anat/missing.nii.gz") is None
        assert tree.search("") is None

    def test_search_on_empty_tree_returns_none(self):
        assert BTree().search("sub-0001") is None

    @pytest.mark.parametrize("t", DEGREES)
    def test_duplicate_insert_replaces_value(self, t):
        tree, pairs = _filled(100, t=t)
        key = sorted(pairs)[37]
        tree.insert(key, {"index": "replaced"})
        assert tree.search(key) == {"index": "replaced"}
        assert len(tree) == len(pairs)


@pytest.mark.unit
class TestBTreeOrder:
    """REQ-STREAMING-BDRABCZUK-8235325CAD21: iteration yields pairs in ascending key order."""

    @pytest.mark.parametrize("t", DEGREES)
    def test_iteration_ascending(self, t):
        tree, pairs = _filled(150, t=t)
        assert list(tree) == sorted(pairs.items())

    def test_empty_tree_iterates_nothing(self):
        assert list(BTree()) == []


@pytest.mark.unit
class TestBTreeRangeQuery:
    """REQ-STREAMING-BDRABCZUK-A58FC71229E7: inclusive bounds, None leaves a side open."""

    def test_bounded_inclusive(self):
        assert [k for k, _ in _letters().range_query("b", "e")] == ["b", "c", "d", "e"]

    def test_bounds_not_stored(self):
        assert [k for k, _ in _letters().range_query("bb", "ee")] == ["c", "d", "e"]

    def test_open_start(self):
        assert [k for k, _ in _letters().range_query(None, "c")] == ["a", "b", "c"]

    def test_open_end(self):
        assert [k for k, _ in _letters().range_query("f")] == ["f", "g", "h"]

    def test_both_open(self):
        assert _letters().range_query() == [(k, {"key": k}) for k in "abcdefgh"]

    def test_empty_range(self):
        assert _letters().range_query("x", "z") == []

    @pytest.mark.parametrize("t", DEGREES)
    def test_path_prefix_query(self, t):
        tree, pairs = _filled(300, t=t)
        prefix = "sub-0042/"
        expected = sorted((k, v) for k, v in pairs.items() if k.startswith(prefix))
        assert expected
        assert tree.range_query(prefix, prefix + "\xff") == expected


@pytest.mark.unit
class TestBTreeDelete:
    """REQ-STREAMING-BDRABCZUK-68E610E937CD and -C0579BE1BCCF."""

    @pytest.mark.parametrize("t", DEGREES)
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_delete_removes_only_that_key(self, t, seed):
        tree, pairs = _filled(300, t=t, seed=seed)
        remaining = _delete_half(tree, pairs, seed=seed)
        assert list(tree) == sorted(remaining.items())
        for key in pairs.keys() - remaining.keys():
            assert tree.search(key) is None

    @pytest.mark.parametrize("t", DEGREES)
    def test_delete_every_key_empties_tree(self, t):
        tree, pairs = _filled(120, t=t)
        for key in _keys(120, seed=5):
            tree.delete(key)
        assert list(tree) == []
        assert len(tree) == 0
        assert len(pairs) == 120

    def test_delete_absent_key_raises_keyerror(self):
        tree, _ = _filled(40)
        with pytest.raises(KeyError):
            tree.delete("sub-9999/missing")

    def test_delete_on_empty_tree_raises_keyerror(self):
        with pytest.raises(KeyError):
            BTree().delete("sub-0001")


@pytest.mark.unit
class TestBTreeLength:
    """REQ-STREAMING-BDRABCZUK-1AFD37473B9E: len() is the number of distinct stored keys."""

    def test_empty(self):
        assert len(BTree()) == 0

    def test_tracks_inserts_duplicates_and_deletes(self):
        tree, pairs = _filled(64)
        assert len(tree) == 64
        tree.insert(next(iter(pairs)), {"index": "again"})
        assert len(tree) == 64
        remaining = _delete_half(tree, pairs)
        assert len(tree) == len(remaining) == 32


@pytest.mark.unit
class TestBTreeMinimumDegree:
    """REQ-STREAMING-BDRABCZUK-261EA669234A: t below 2 raises ValueError."""

    @pytest.mark.parametrize("t", [1, 0, -3])
    def test_below_two_raises(self, t):
        with pytest.raises(ValueError):
            BTree(t=t)

    def test_two_accepted(self):
        assert len(BTree(t=2)) == 0


@pytest.mark.unit
class TestBTreeSerialisation:
    """REQ-STREAMING-BDRABCZUK-F7DFC1713553: to_dict / from_dict JSON round trip."""

    @pytest.mark.parametrize("t", DEGREES)
    def test_json_round_trip(self, t):
        tree, pairs = _filled(150, t=t)
        _delete_half(tree, pairs)
        restored = BTree.from_dict(json.loads(json.dumps(tree.to_dict())))
        assert restored.to_dict()["t"] == t
        assert len(restored) == len(tree)
        assert list(restored) == list(tree)

    def test_empty_round_trip(self):
        restored = BTree.from_dict(json.loads(json.dumps(BTree(t=4).to_dict())))
        assert restored.to_dict()["t"] == 4
        assert list(restored) == []


@pytest.mark.unit
class TestBTreeBalance:
    """REQ-STREAMING-BDRABCZUK-242E1CA48208 (occupancy) and -8516AEEB7662 (leaf depth)."""

    @pytest.mark.parametrize("t", DEGREES)
    @pytest.mark.parametrize("delete", [False, True])
    def test_node_occupancy(self, t, delete):
        tree, pairs = _filled(500, t=t, seed=7)
        if delete:
            _delete_half(tree, pairs, seed=8)
        root = tree.to_dict()["root"]
        assert len(root["keys"]) <= 2 * t - 1
        for node, depth in _nodes(root):
            if depth:
                assert t - 1 <= len(node["keys"]) <= 2 * t - 1

    @pytest.mark.parametrize("t", DEGREES)
    @pytest.mark.parametrize("delete", [False, True])
    def test_leaves_share_depth(self, t, delete):
        tree, pairs = _filled(500, t=t, seed=7)
        if delete:
            _delete_half(tree, pairs, seed=8)
        leaf_depths = {depth for node, depth in _nodes(tree.to_dict()["root"]) if not node["children"]}
        assert len(leaf_depths) == 1
