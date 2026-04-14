#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to index a directory tree into a B-tree and save the result to JSON.

Usage
-----
    python3 src/file_indexer/indexer.py /path/to/dir [options]

Output
------
A JSON file (default ``<path>/.file_index.json``) with schema::

    {
      "t":    <int>,    # B-tree minimum degree
      "size": <int>,    # total number of indexed entries
      "root": { ... }   # serialised B-tree root node
    }

Each entry records::

    {
      "type":     "file" | "dir",
      "size":     <bytes>,
      "modified": <unix timestamp>,
      "ext":      "<.ext>"          # files only
    }
"""

import json
import os
from os.path import abspath, join, relpath, splitext

from champollion_utils.script_builder import ScriptBuilder

from .btree import BTree


class IndexFileSystem(ScriptBuilder):
    """Walk a directory tree and store all paths in a B-tree index."""

    def __init__(self) -> None:
        super().__init__(
            script_name="index_filesystem",
            description="Index a directory tree into a B-tree and save to JSON.",
        )
        (self
         .add_argument(
             "path",
             help="Absolute path of the directory to index.")
         .add_optional_argument(
             "--output",
             "Output JSON file path. Defaults to <path>/.file_index.json.",
             default=None)
         .add_optional_argument(
             "--max-depth",
             "Maximum traversal depth (-1 = unlimited).",
             default=-1, type_=int)
         .add_optional_argument(
             "--btree-order",
             "B-tree minimum degree t (>= 2). Higher = wider nodes.",
             default=64, type_=int)
         .add_flag(
             "--files-only",
             "Index only regular files; skip directory entries.")
         .add_flag(
             "--include-hidden",
             "Include hidden files and directories (dot-prefixed names)."))

    def run(self) -> int:
        """Walk and index the directory. Returns 0 on success."""
        path = abspath(self.args.path)
        if not self.validate_paths([path]):
            raise ValueError(f"Path does not exist: {path}")

        tree = BTree(t=self.args.btree_order)
        n = self._walk_and_index(path, tree)

        output = self.args.output or join(path, ".file_index.json")
        with open(output, "w") as fh:
            json.dump(tree.to_dict(), fh, indent=2)

        print(f"Indexed {n} entries → {output}", flush=True)
        return 0

    def _walk_and_index(self, root: str, tree: BTree) -> int:
        count = 0
        for dirpath, dirnames, filenames in os.walk(root):
            rel = relpath(dirpath, root)
            depth = 0 if rel == "." else len(rel.split(os.sep))

            if self.args.max_depth >= 0 and depth > self.args.max_depth:
                dirnames.clear()
                continue

            if not self.args.include_hidden:
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]

            if not self.args.files_only:
                s = os.stat(dirpath)
                key = "." if rel == "." else rel
                tree.insert(key, {"type": "dir", "size": 0,
                                  "modified": s.st_mtime})
                count += 1

            for fname in filenames:
                if not self.args.include_hidden and fname.startswith("."):
                    continue
                fpath = join(dirpath, fname)
                key = relpath(fpath, root)
                try:
                    s = os.stat(fpath)
                    tree.insert(key, {
                        "type": "file",
                        "size": s.st_size,
                        "modified": s.st_mtime,
                        "ext": splitext(fname)[1],
                    })
                    count += 1
                except OSError:
                    pass

        return count


def main() -> int:
    script = IndexFileSystem()
    return script.build().print_args().run()


if __name__ == "__main__":
    raise SystemExit(main())
