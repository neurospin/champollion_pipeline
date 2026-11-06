"""Reading of one input path as T1 files, Morphologist outputs, neither or both.

Pure classification: what the pipeline does with the result is decided elsewhere. Detection and the T1 listing share
one lazy walk and one set of predicates, so the two cannot drift.
"""

import logging
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum

T1_SUFFIXES: tuple[str, ...] = (".nii.gz", ".nii")
NON_T1_FOLDERS: frozenset[str] = frozenset({"derivatives", "qc", "t1mri"})

_DERIVATIVES_FOLDER = "derivatives"
_GRAPH_FOLDER = "t1mri"
_GRAPH_SUFFIX = ".arg"
_BIDS_PREFIX = "sub-"
_BIDS_T1_ENDING = "_T1w"

_Parts = tuple[str, ...]

logger = logging.getLogger(__name__)


class InputKind(Enum):
    """What one input path holds."""

    T1 = "t1"
    MORPHOLOGIST = "morphologist"
    NEITHER = "neither"
    BOTH = "both"


@dataclass(frozen=True)
class InputScan:
    """The first T1 file and the first Morphologist graph a walk reached, relative to the input."""

    first_t1: str | None
    first_graph: str | None

    @property
    def kind(self) -> InputKind:
        """Return the kind these two findings make."""
        if self.first_t1 is not None and self.first_graph is not None:
            return InputKind.BOTH
        if self.first_t1 is not None:
            return InputKind.T1
        if self.first_graph is not None:
            return InputKind.MORPHOLOGIST
        return InputKind.NEITHER


@dataclass(frozen=True)
class _Entry:
    """A folder or a regular file of a listing, typed through links."""

    name: str
    path: str
    is_folder: bool
    is_link: bool


def _typed_entry(entry: os.DirEntry) -> _Entry | None:
    """Return ``entry`` typed, or None when it is neither a folder nor a regular file or cannot be typed."""
    try:
        is_folder = entry.is_dir()
        if not is_folder and not entry.is_file():
            return None
        return _Entry(entry.name, entry.path, is_folder, entry.is_symlink())
    except OSError as error:
        logger.warning("Skipping entry %s, its type cannot be read: %s", entry.path, error)
        return None


def _walk_rank(entry: _Entry) -> tuple[bool, str]:
    """Return the sort key of ``entry``: a folder named derivatives first, then plain name order."""
    return (not (entry.is_folder and entry.name == _DERIVATIVES_FOLDER), entry.name)


def _read_folder(folder: str) -> list[_Entry]:
    """Return the entries of ``folder`` that are not hidden, in walk order; none when it cannot be read."""
    try:
        with os.scandir(folder) as listing:
            entries = [entry for entry in listing if not entry.name.startswith(".")]
    except OSError as error:
        logger.warning("Skipping folder %s, it cannot be read: %s", folder, error)
        return []
    typed = [_typed_entry(entry) for entry in entries]
    return sorted((entry for entry in typed if entry is not None), key=_walk_rank)


def _walk_folder(
    folder: str, parts: _Parts, real_path: str, search_path: frozenset[str], wanted: Callable[[_Parts], bool]
) -> Iterator[tuple[_Parts, str]]:
    """Yield the regular files below ``folder`` depth first; see ``_walk``."""
    for entry in _read_folder(folder):
        if not entry.is_folder:
            yield parts, entry.name
            continue
        entry_parts = (*parts, entry.name)
        if not wanted(entry_parts):
            continue
        entry_real = os.path.realpath(entry.path) if entry.is_link else os.path.join(real_path, entry.name)
        if entry_real not in search_path:
            yield from _walk_folder(entry.path, entry_parts, entry_real, search_path | {entry_real}, wanted)


def _walk(path: str, wanted: Callable[[_Parts], bool]) -> Iterator[tuple[_Parts, str]]:
    """Yield ``(folder parts, file name)`` for each regular file below ``path``, one at a time, in walk order.

    ``parts`` are the names on disk relative to ``path``, through links. ``wanted`` is asked just before a folder is
    read, so a rule that changes during the walk applies to the folders not yet entered. A link to a folder already on
    its own search path stays unfollowed; two links to one folder both count.
    """
    real_path = os.path.realpath(path)
    return _walk_folder(path, (), real_path, frozenset({real_path}), wanted)


def _can_hold_t1(parts: _Parts) -> bool:
    """Return whether a folder at ``parts`` may hold a T1 file."""
    return NON_T1_FOLDERS.isdisjoint(parts)


def _is_t1_name(parts: _Parts, name: str) -> bool:
    """Return whether a file ``name`` in the folder at ``parts`` is named as a T1 file."""
    suffix = next((suffix for suffix in T1_SUFFIXES if name.endswith(suffix)), None)
    if suffix is None:
        return False
    if name.startswith(_BIDS_PREFIX) and any(part.startswith(_BIDS_PREFIX) for part in parts):
        return name.removesuffix(suffix).endswith(_BIDS_T1_ENDING)
    return True


def _is_graph(parts: _Parts, name: str) -> bool:
    """Return whether a file ``name`` in the folder at ``parts`` is a Morphologist graph."""
    return _GRAPH_FOLDER in parts and name.endswith(_GRAPH_SUFFIX)


def scan_input(path: str) -> InputScan:
    """Walk ``path`` up to its first T1 file and return what the walk reached.

    The walk stops there, so a T1 file with no graph does not prove that no graph exists. After the first graph, no
    folder that cannot hold a T1 file is read.
    """
    if not os.path.isdir(path):
        return InputScan(None, None)
    first_graph: str | None = None

    def wanted(parts: _Parts) -> bool:
        return first_graph is None or _can_hold_t1(parts)

    for parts, name in _walk(path, wanted):
        if first_graph is None and _is_graph(parts, name):
            first_graph = os.path.join(*parts, name)
        if _can_hold_t1(parts) and _is_t1_name(parts, name):
            return InputScan(os.path.join(*parts, name), first_graph)
    return InputScan(None, first_graph)


def detect_input_kind(path: str) -> InputKind:
    """Return the kind of ``path``."""
    return scan_input(path).kind


def find_t1_files(directory: str) -> list[str]:
    """Return every T1 file of ``directory`` as a sorted list of relative paths, each as found on disk."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(*parts, name) for parts, name in _walk(directory, _can_hold_t1) if _is_t1_name(parts, name)
    )
