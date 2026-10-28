"""B-tree keyed by relative path strings, used for prefix range queries.

A CLRS B-tree of minimum degree ``t``: every non-root node holds t-1 to 2t-1
keys and all leaves sit at the same depth. ``to_dict`` keeps the JSON layout
of the earlier index files so they load unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator


class BTreeNode:
    """One B-tree node: sorted keys, their values and, for an internal node, len(keys) + 1 children."""

    def __init__(self, t: int, leaf: bool = False) -> None:
        self.t = t
        self.keys: list[str] = []
        self.values: list[dict] = []
        self.children: list[BTreeNode] = []
        self.leaf = leaf

    def is_full(self) -> bool:
        """True when the node holds the maximum 2t-1 keys."""
        return len(self.keys) == 2 * self.t - 1

    def to_dict(self) -> dict:
        """Return a JSON-serialisable copy of the node and its subtree."""
        return {
            "leaf": self.leaf,
            "keys": list(self.keys),
            "values": list(self.values),
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict, t: int) -> BTreeNode:
        """Rebuild a node and its subtree from ``to_dict`` output."""
        node = cls(t, leaf=data["leaf"])
        node.keys = list(data["keys"])
        node.values = list(data["values"])
        node.children = [cls.from_dict(child, t) for child in data["children"]]
        return node


class BTree:
    """B-tree of minimum degree ``t`` mapping string keys to dict values."""

    def __init__(self, t: int = 3) -> None:
        if t < 2:
            raise ValueError(f"BTree minimum degree t must be >= 2, got {t}")
        self.t = t
        self.root = BTreeNode(t, leaf=True)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[tuple[str, dict]]:
        yield from self._iter_node(self.root)

    def _iter_node(self, node: BTreeNode) -> Iterator[tuple[str, dict]]:
        """Yield the pairs of ``node``'s subtree in key order."""
        for i, key in enumerate(node.keys):
            if not node.leaf:
                yield from self._iter_node(node.children[i])
            yield key, node.values[i]
        if not node.leaf:
            yield from self._iter_node(node.children[-1])

    def _find(self, key: str) -> tuple[BTreeNode, int] | None:
        """Return the node holding ``key`` and the key's index there, or None."""
        node = self.root
        while True:
            i = 0
            while i < len(node.keys) and key > node.keys[i]:
                i += 1
            if i < len(node.keys) and node.keys[i] == key:
                return node, i
            if node.leaf:
                return None
            node = node.children[i]

    def search(self, key: str) -> dict | None:
        """Return the value stored under ``key``, or ``None``."""
        found = self._find(key)
        return found[0].values[found[1]] if found else None

    def insert(self, key: str, value: dict) -> None:
        """Store ``value`` under ``key``, replacing the value of an existing key."""
        found = self._find(key)
        if found:
            node, i = found
            node.values[i] = value
            return
        self._size += 1
        if self.root.is_full():
            new_root = BTreeNode(self.t)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
        self._insert_non_full(self.root, key, value)

    def _split_child(self, parent: BTreeNode, i: int) -> None:
        """Split the full ``parent.children[i]``, moving its median key up into ``parent``."""
        t = self.t
        child = parent.children[i]
        sibling = BTreeNode(t, leaf=child.leaf)
        parent.keys.insert(i, child.keys[t - 1])
        parent.values.insert(i, child.values[t - 1])
        parent.children.insert(i + 1, sibling)
        sibling.keys, child.keys = child.keys[t:], child.keys[: t - 1]
        sibling.values, child.values = child.values[t:], child.values[: t - 1]
        if not child.leaf:
            sibling.children, child.children = child.children[t:], child.children[:t]

    def _insert_non_full(self, node: BTreeNode, key: str, value: dict) -> None:
        """Insert a new ``key`` below the non-full ``node``, splitting full children on the way down."""
        while True:
            i = len(node.keys)
            while i > 0 and key < node.keys[i - 1]:
                i -= 1
            if node.leaf:
                node.keys.insert(i, key)
                node.values.insert(i, value)
                return
            if node.children[i].is_full():
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            node = node.children[i]

    def delete(self, key: str) -> None:
        """Remove the pair stored under ``key``; KeyError, with the tree unchanged, when it is absent."""
        if self._find(key) is None:
            raise KeyError(key)
        self._size -= 1
        self._delete_from(self.root, key)
        if not self.root.keys and not self.root.leaf:
            self.root = self.root.children[0]

    def _delete_from(self, node: BTreeNode, key: str) -> None:
        """Delete ``key``, known to be in ``node``'s subtree, in one top-down pass."""
        while True:
            i = 0
            while i < len(node.keys) and key > node.keys[i]:
                i += 1
            if i < len(node.keys) and node.keys[i] == key:
                if node.leaf:
                    node.keys.pop(i)
                    node.values.pop(i)
                    return
                node, key = self._delete_internal(node, i)
                continue
            if len(node.children[i].keys) < self.t:
                i = self._fill(node, i)
            node = node.children[i]

    def _delete_internal(self, node: BTreeNode, i: int) -> tuple[BTreeNode, str]:
        """Remove key ``i`` of internal ``node``; return the subtree and key the delete continues with."""
        key = node.keys[i]
        left, right = node.children[i], node.children[i + 1]
        if len(left.keys) >= self.t:
            pred_key, pred_value = self._predecessor(left)
            node.keys[i], node.values[i] = pred_key, pred_value
            return left, pred_key
        if len(right.keys) >= self.t:
            succ_key, succ_value = self._successor(right)
            node.keys[i], node.values[i] = succ_key, succ_value
            return right, succ_key
        self._merge(node, i)
        return left, key

    def _fill(self, parent: BTreeNode, i: int) -> int:
        """Give ``parent.children[i]`` at least t keys; return the index of the child that holds its keys now."""
        if i > 0 and len(parent.children[i - 1].keys) >= self.t:
            self._borrow_from_prev(parent, i)
            return i
        if i < len(parent.keys) and len(parent.children[i + 1].keys) >= self.t:
            self._borrow_from_next(parent, i)
            return i
        if i < len(parent.keys):
            self._merge(parent, i)
            return i
        self._merge(parent, i - 1)
        return i - 1

    def _borrow_from_prev(self, parent: BTreeNode, i: int) -> None:
        """Rotate one key from the left sibling of ``parent.children[i]`` through ``parent``."""
        child, sibling = parent.children[i], parent.children[i - 1]
        child.keys.insert(0, parent.keys[i - 1])
        child.values.insert(0, parent.values[i - 1])
        parent.keys[i - 1] = sibling.keys.pop()
        parent.values[i - 1] = sibling.values.pop()
        if not sibling.leaf:
            child.children.insert(0, sibling.children.pop())

    def _borrow_from_next(self, parent: BTreeNode, i: int) -> None:
        """Rotate one key from the right sibling of ``parent.children[i]`` through ``parent``."""
        child, sibling = parent.children[i], parent.children[i + 1]
        child.keys.append(parent.keys[i])
        child.values.append(parent.values[i])
        parent.keys[i] = sibling.keys.pop(0)
        parent.values[i] = sibling.values.pop(0)
        if not sibling.leaf:
            child.children.append(sibling.children.pop(0))

    def _merge(self, parent: BTreeNode, i: int) -> None:
        """Merge ``parent.children[i + 1]`` and key ``i`` of ``parent`` into ``parent.children[i]``."""
        child, sibling = parent.children[i], parent.children.pop(i + 1)
        child.keys.append(parent.keys.pop(i))
        child.values.append(parent.values.pop(i))
        child.keys.extend(sibling.keys)
        child.values.extend(sibling.values)
        child.children.extend(sibling.children)

    @staticmethod
    def _predecessor(node: BTreeNode) -> tuple[str, dict]:
        """Return the largest pair of ``node``'s subtree."""
        while not node.leaf:
            node = node.children[-1]
        return node.keys[-1], node.values[-1]

    @staticmethod
    def _successor(node: BTreeNode) -> tuple[str, dict]:
        """Return the smallest pair of ``node``'s subtree."""
        while not node.leaf:
            node = node.children[0]
        return node.keys[0], node.values[0]

    def range_query(self, start: str | None = None, end: str | None = None) -> list[tuple[str, dict]]:
        """Return the pairs whose key lies in ``[start, end]``, ascending; a None bound is open."""
        result = []
        for key, value in self:
            if start is not None and key < start:
                continue
            if end is not None and key > end:
                break
            result.append((key, value))
        return result

    def to_dict(self) -> dict:
        """Return a JSON-serialisable copy of the tree."""
        return {"t": self.t, "size": self._size, "root": self.root.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> BTree:
        """Rebuild a tree from ``to_dict`` output."""
        tree = cls(t=data["t"])
        tree._size = data["size"]
        tree.root = BTreeNode.from_dict(data["root"], data["t"])
        return tree
