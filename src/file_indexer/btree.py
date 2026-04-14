#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B-tree data structure for file-system path indexing.

Keys are strings (relative paths). Values are arbitrary dicts (file metadata).
"""
from __future__ import annotations
from typing import Iterator


class BTreeNode:
    """A single node in the B-tree."""

    def __init__(self, t: int, leaf: bool = False) -> None:
        self.t = t
        self.keys: list[str] = []
        self.values: list[dict] = []
        self.children: list[BTreeNode] = []
        self.leaf = leaf

    def is_full(self) -> bool:
        return len(self.keys) == 2 * self.t - 1

    def to_dict(self) -> dict:
        return {
            "leaf": self.leaf,
            "keys": self.keys,
            "values": self.values,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict, t: int) -> BTreeNode:
        node = cls(t, leaf=data["leaf"])
        node.keys = data["keys"]
        node.values = data["values"]
        node.children = [cls.from_dict(c, t) for c in data["children"]]
        return node


class BTree:
    """
    B-tree of minimum degree *t*.

    Each internal node holds between t-1 and 2t-1 keys.
    Keys are strings sorted lexicographically.
    Duplicate inserts overwrite the existing value.
    """

    def __init__(self, t: int = 3) -> None:
        if t < 2:
            raise ValueError("Minimum degree t must be >= 2")
        self.t = t
        self.root = BTreeNode(t, leaf=True)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[tuple[str, dict]]:
        yield from self._inorder(self.root)

    def search(self, key: str) -> dict | None:
        return self._search(self.root, key)

    def insert(self, key: str, value: dict) -> None:
        if self._update(self.root, key, value):
            return
        self._size += 1
        root = self.root
        if root.is_full():
            new_root = BTreeNode(self.t, leaf=False)
            new_root.children.append(root)
            self._split_child(new_root, 0)
            self.root = new_root
        self._insert_non_full(self.root, key, value)

    def delete(self, key: str) -> None:
        if not self._delete(self.root, key):
            raise KeyError(key)
        self._size -= 1
        if not self.root.keys and not self.root.leaf:
            self.root = self.root.children[0]

    def range_query(
        self, start: str | None = None, end: str | None = None
    ) -> list[tuple[str, dict]]:
        result: list[tuple[str, dict]] = []
        for k, v in self._inorder(self.root):
            if start is not None and k < start:
                continue
            if end is not None and k > end:
                break
            result.append((k, v))
        return result

    def to_dict(self) -> dict:
        return {"t": self.t, "size": self._size, "root": self.root.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> BTree:
        tree = cls(t=data["t"])
        tree._size = data["size"]
        tree.root = BTreeNode.from_dict(data["root"], data["t"])
        return tree

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _inorder(self, node: BTreeNode) -> Iterator[tuple[str, dict]]:
        if node.leaf:
            yield from zip(node.keys, node.values)
            return
        for i, (k, v) in enumerate(zip(node.keys, node.values)):
            yield from self._inorder(node.children[i])
            yield k, v
        yield from self._inorder(node.children[-1])

    def _search(self, node: BTreeNode, key: str) -> dict | None:
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and key == node.keys[i]:
            return node.values[i]
        if node.leaf:
            return None
        return self._search(node.children[i], key)

    def _update(self, node: BTreeNode, key: str, value: dict) -> bool:
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and key == node.keys[i]:
            node.values[i] = value
            return True
        if node.leaf:
            return False
        return self._update(node.children[i], key, value)

    def _insert_non_full(self, node: BTreeNode, key: str, value: dict) -> None:
        i = len(node.keys) - 1
        if node.leaf:
            node.keys.append("")
            node.values.append({})
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                node.values[i + 1] = node.values[i]
                i -= 1
            node.keys[i + 1] = key
            node.values[i + 1] = value
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            if node.children[i].is_full():
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key, value)

    def _split_child(self, parent: BTreeNode, i: int) -> None:
        t = self.t
        child = parent.children[i]
        new_node = BTreeNode(t, leaf=child.leaf)
        mid = t - 1
        parent.keys.insert(i, child.keys[mid])
        parent.values.insert(i, child.values[mid])
        parent.children.insert(i + 1, new_node)
        new_node.keys = child.keys[mid + 1:]
        new_node.values = child.values[mid + 1:]
        if not child.leaf:
            new_node.children = child.children[mid + 1:]
            child.children = child.children[:mid + 1]
        child.keys = child.keys[:mid]
        child.values = child.values[:mid]

    def _delete(self, node: BTreeNode, key: str) -> bool:
        t = self.t
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and key == node.keys[i]:
            if node.leaf:
                node.keys.pop(i)
                node.values.pop(i)
                return True
            return self._delete_internal(node, i)
        if node.leaf:
            return False
        deficit = len(node.children[i].keys) < t
        if deficit:
            self._fill(node, i)
            return self._delete(node, key)
        return self._delete(node.children[i], key)

    def _delete_internal(self, node: BTreeNode, i: int) -> bool:
        t = self.t
        key = node.keys[i]
        left, right = node.children[i], node.children[i + 1]
        if len(left.keys) >= t:
            pred_k, pred_v = self._get_predecessor(left)
            node.keys[i] = pred_k
            node.values[i] = pred_v
            return self._delete(left, pred_k)
        elif len(right.keys) >= t:
            succ_k, succ_v = self._get_successor(right)
            node.keys[i] = succ_k
            node.values[i] = succ_v
            return self._delete(right, succ_k)
        else:
            self._merge(node, i)
            return self._delete(left, key)

    def _get_predecessor(self, node: BTreeNode) -> tuple[str, dict]:
        while not node.leaf:
            node = node.children[-1]
        return node.keys[-1], node.values[-1]

    def _get_successor(self, node: BTreeNode) -> tuple[str, dict]:
        while not node.leaf:
            node = node.children[0]
        return node.keys[0], node.values[0]

    def _fill(self, parent: BTreeNode, i: int) -> None:
        t = self.t
        if i > 0 and len(parent.children[i - 1].keys) >= t:
            self._borrow_from_prev(parent, i)
        elif i < len(parent.children) - 1 and len(parent.children[i + 1].keys) >= t:
            self._borrow_from_next(parent, i)
        else:
            if i < len(parent.children) - 1:
                self._merge(parent, i)
            else:
                self._merge(parent, i - 1)

    def _borrow_from_prev(self, parent: BTreeNode, i: int) -> None:
        child, sibling = parent.children[i], parent.children[i - 1]
        child.keys.insert(0, parent.keys[i - 1])
        child.values.insert(0, parent.values[i - 1])
        if not sibling.leaf:
            child.children.insert(0, sibling.children.pop())
        parent.keys[i - 1] = sibling.keys.pop()
        parent.values[i - 1] = sibling.values.pop()

    def _borrow_from_next(self, parent: BTreeNode, i: int) -> None:
        child, sibling = parent.children[i], parent.children[i + 1]
        child.keys.append(parent.keys[i])
        child.values.append(parent.values[i])
        if not sibling.leaf:
            child.children.append(sibling.children.pop(0))
        parent.keys[i] = sibling.keys.pop(0)
        parent.values[i] = sibling.values.pop(0)

    def _merge(self, parent: BTreeNode, i: int) -> None:
        left, right = parent.children[i], parent.children[i + 1]
        left.keys.append(parent.keys.pop(i))
        left.values.append(parent.values.pop(i))
        left.keys.extend(right.keys)
        left.values.extend(right.values)
        if not left.leaf:
            left.children.extend(right.children)
        parent.children.pop(i + 1)
