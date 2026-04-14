#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the file_indexer module (BTree + IndexFileSystem).
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from file_indexer.btree import BTree, BTreeNode
from file_indexer.indexer import IndexFileSystem


# =========================================================================== #
# TestBTree
# =========================================================================== #

class TestBTree:

    def test_insert_and_search(self):
        tree = BTree(t=3)
        for i in range(100):
            tree.insert(f"file_{i:03d}.txt", {"i": i})
        for i in range(100):
            result = tree.search(f"file_{i:03d}.txt")
            assert result is not None
            assert result["i"] == i

    def test_search_missing_key_returns_none(self):
        tree = BTree(t=3)
        tree.insert("exists.txt", {"x": 1})
        assert tree.search("missing.txt") is None

    def test_insert_duplicate_overwrites_value(self):
        tree = BTree(t=3)
        tree.insert("key.txt", {"v": 1})
        tree.insert("key.txt", {"v": 2})
        assert tree.search("key.txt") == {"v": 2}
        assert len(tree) == 1

    def test_ordered_iteration(self):
        tree = BTree(t=3)
        keys = [f"file_{i:03d}.txt" for i in range(30)]
        for k in keys:
            tree.insert(k, {})
        iterated = [k for k, _ in tree]
        assert iterated == sorted(keys)

    def test_range_query_bounded(self):
        tree = BTree(t=3)
        for code in range(97, 123):  # 'a' .. 'z'
            tree.insert(chr(code), {"code": code})
        result = tree.range_query("c", "f")
        assert [k for k, _ in result] == ["c", "d", "e", "f"]

    def test_range_query_open_start(self):
        tree = BTree(t=3)
        for code in range(97, 123):
            tree.insert(chr(code), {})
        result = tree.range_query(end="c")
        assert [k for k, _ in result] == ["a", "b", "c"]

    def test_range_query_open_end(self):
        tree = BTree(t=3)
        for code in range(97, 123):
            tree.insert(chr(code), {})
        result = tree.range_query(start="x")
        assert [k for k, _ in result] == ["x", "y", "z"]

    def test_delete_leaf_key(self):
        tree = BTree(t=3)
        keys = ["alpha", "beta", "gamma", "delta", "epsilon"]
        for k in keys:
            tree.insert(k, {"k": k})
        initial_len = len(tree)
        tree.delete("gamma")
        assert tree.search("gamma") is None
        assert len(tree) == initial_len - 1

    def test_delete_missing_key_raises_keyerror(self):
        tree = BTree(t=3)
        tree.insert("present", {})
        with pytest.raises(KeyError):
            tree.delete("absent")

    def test_len_tracks_insertions_and_deletions(self):
        tree = BTree(t=3)
        assert len(tree) == 0
        for i in range(10):
            tree.insert(f"k{i}", {})
        assert len(tree) == 10
        tree.delete("k5")
        assert len(tree) == 9
        # Duplicate insert should not increase size
        tree.insert("k0", {"updated": True})
        assert len(tree) == 9

    def test_serialisation_roundtrip(self):
        tree = BTree(t=4)
        for i in range(50):
            tree.insert(f"path/to/file_{i:03d}.py", {"i": i})
        restored = BTree.from_dict(tree.to_dict())
        assert len(restored) == 50
        for i in range(50):
            result = restored.search(f"path/to/file_{i:03d}.py")
            assert result is not None
            assert result["i"] == i

    def test_large_insert_forces_splits(self):
        tree = BTree(t=2)
        keys = [f"file_{i:04d}.txt" for i in range(500)]
        for k in keys:
            tree.insert(k, {"key": k})
        for k in keys:
            assert tree.search(k) is not None, f"Missing key: {k}"
        assert len(tree) == 500


# =========================================================================== #
# TestIndexFileSystem
# =========================================================================== #

def _make_input_dir(temp_dir: str) -> Path:
    """Create a standard input directory with some files and a subdir."""
    inp = Path(temp_dir) / "input"
    inp.mkdir(parents=True, exist_ok=True)
    (inp / "file_a.txt").write_text("hello")
    (inp / "file_b.py").write_text("print()")
    sub = inp / "subdir"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested")
    return inp


class TestIndexFileSystem:

    def test_missing_path_raises_systemexit(self):
        script = IndexFileSystem()
        with pytest.raises(SystemExit):
            script.parse_args([])

    def test_default_output_path(self, temp_dir):
        inp = _make_input_dir(temp_dir)
        script = IndexFileSystem()
        script.parse_args([str(inp)])
        script.run()
        assert (inp / ".file_index.json").exists()

    def test_custom_output_path(self, temp_dir):
        inp = _make_input_dir(temp_dir)
        out = Path(temp_dir) / "custom.json"
        script = IndexFileSystem()
        script.parse_args([str(inp), "--output", str(out)])
        script.run()
        assert out.exists()

    def test_run_creates_valid_json(self, temp_dir):
        inp = _make_input_dir(temp_dir)
        out = Path(temp_dir) / "index.json"
        script = IndexFileSystem()
        script.parse_args([str(inp), "--output", str(out)])
        script.run()
        data = json.loads(out.read_text())
        assert "t" in data
        assert "size" in data
        assert "root" in data

    def test_indexed_file_entry_fields(self, temp_dir):
        inp = _make_input_dir(temp_dir)
        out = Path(temp_dir) / "index.json"
        script = IndexFileSystem()
        script.parse_args([str(inp), "--output", str(out)])
        script.run()
        tree = BTree.from_dict(json.loads(out.read_text()))
        entry = tree.search("file_a.txt")
        assert entry is not None
        assert entry["type"] == "file"
        assert "size" in entry
        assert "modified" in entry
        assert "ext" in entry

    def test_indexed_dir_entry_fields(self, temp_dir):
        inp = _make_input_dir(temp_dir)
        out = Path(temp_dir) / "index.json"
        script = IndexFileSystem()
        script.parse_args([str(inp), "--output", str(out)])
        script.run()
        tree = BTree.from_dict(json.loads(out.read_text()))
        # Root dir is indexed as "."
        entry = tree.search(".")
        assert entry is not None
        assert entry["type"] == "dir"
        assert entry["size"] == 0
        assert "modified" in entry

    def test_files_only_excludes_dirs(self, temp_dir):
        inp = _make_input_dir(temp_dir)
        out = Path(temp_dir) / "index.json"
        script = IndexFileSystem()
        script.parse_args([str(inp), "--output", str(out), "--files-only"])
        script.run()
        tree = BTree.from_dict(json.loads(out.read_text()))
        for k, v in tree:
            assert v["type"] != "dir", f"Unexpected dir entry: {k}"

    def test_max_depth_zero(self, temp_dir):
        inp = _make_input_dir(temp_dir)
        out = Path(temp_dir) / "index.json"
        script = IndexFileSystem()
        script.parse_args([str(inp), "--output", str(out), "--max-depth", "0"])
        script.run()
        tree = BTree.from_dict(json.loads(out.read_text()))
        keys = [k for k, _ in tree]
        # nested.txt lives in subdir (depth 1) — must not appear
        assert "subdir/nested.txt" not in keys
        assert not any("subdir" in k and k != "subdir" for k in keys)

    def test_hidden_excluded_by_default(self, temp_dir):
        inp = _make_input_dir(temp_dir)
        (inp / ".secret").write_text("hidden")
        out = Path(temp_dir) / "index.json"
        script = IndexFileSystem()
        script.parse_args([str(inp), "--output", str(out)])
        script.run()
        tree = BTree.from_dict(json.loads(out.read_text()))
        assert tree.search(".secret") is None

    def test_include_hidden_flag(self, temp_dir):
        inp = _make_input_dir(temp_dir)
        (inp / ".secret").write_text("hidden")
        out = Path(temp_dir) / "index.json"
        script = IndexFileSystem()
        script.parse_args([str(inp), "--output", str(out), "--include-hidden"])
        script.run()
        tree = BTree.from_dict(json.loads(out.read_text()))
        assert tree.search(".secret") is not None

    def test_invalid_path_raises_valueerror(self, temp_dir):
        script = IndexFileSystem()
        script.parse_args(["/nonexistent_path_xyz_123"])
        with pytest.raises(ValueError):
            script.run()

    def test_btree_order_two_accepted(self, temp_dir):
        inp = _make_input_dir(temp_dir)
        out = Path(temp_dir) / "index.json"
        script = IndexFileSystem()
        script.parse_args([str(inp), "--output", str(out), "--btree-order", "2"])
        result = script.run()
        assert result == 0
        data = json.loads(out.read_text())
        assert data["t"] == 2
