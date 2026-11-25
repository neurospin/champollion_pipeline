"""GraphItem, one discovered Morphologist graph, and LooseGraphError, a graph outside any subject folder.

Standard library only, no I/O: the item type of the StageIndexer port's discover_graphs, shared by the port and its
adapters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class GraphItem:
    """One graph found by a StageIndexer's discover_graphs: a path plus an attribute mapping.

    ``path`` is POSIX ("/"), relative to the discovered folder. ``attributes`` maps attribute names to values; it is
    copied into a read-only mapping at construction, so the item never shares or exposes a mutable dict. Equality
    reads both fields; the hash reads ``path`` only.
    """

    path: str
    attributes: Mapping[str, str | None] = field(hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))


class LooseGraphError(ValueError):
    """Raised when a .arg file sits directly in a given folder whose name does not start with sub-.

    ``path`` is the graph file, as the given folder joined with its name.
    """

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.path = path

    def __str__(self) -> str:
        return (
            f"Graph file {self.path} sits directly in the given folder, outside any subject folder: "
            "give the subject folder holding it, or the folder above the subject folders"
        )
