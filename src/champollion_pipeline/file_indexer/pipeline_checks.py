"""Per-scan eligibility checks over a B-tree index of a subjects directory.

The streaming arrival barrier polls ``SubjectEligibilityChecker`` through
``list_scans``, ``scan_has_pattern`` and ``refresh``; the sequential pre-stage
report uses ``check``. Patterns and paths come from the user's configuration,
so they are checked, never trusted.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from os.path import join, splitext

import champollion_utils.logger  # noqa: F401 - installs the OUTPUT level helpers
from champollion_utils.logger import OUTPUT

from .btree import BTree
from .scan_id import ScanId

logger = logging.getLogger(__name__)

BIDS_SUBJECT_PREFIX = "sub-"
BIDS_SESSION_PREFIX = "ses-"
BIDS_ACQUISITION_PREFIX = "acq-"
WILDCARD_CHARS = "*?["
# Upper bound of a range query: every key starting with a literal sorts below literal + this.
_KEY_MAX_SUFFIX = "\U0010ffff"


def _check_relative(pattern: str) -> None:
    """Raise ValueError, naming the pattern, when it is absolute or has a '..' path segment."""
    if os.path.isabs(pattern) or ".." in pattern.split("/"):
        raise ValueError(f"Pattern must be a relative path inside the scan folder: {pattern!r}")


def _is_hidden(name: str) -> bool:
    return name.startswith(".")


@dataclass
class SubjectEligibilityReport:
    """Result of a per-scan eligibility check before a pipeline stage."""

    stage_name: str
    input_dir: str
    eligible: list[ScanId] = field(default_factory=list)
    ineligible: dict[ScanId, list[str]] = field(default_factory=dict)

    def log_summary(self) -> None:
        """Log the summary record, and one warning when a scan is ineligible."""
        total = len(self.eligible) + len(self.ineligible)
        logger.log(
            OUTPUT,
            "Pre-check %s: %d of %d scans eligible in %s",
            self.stage_name,
            len(self.eligible),
            total,
            self.input_dir,
        )
        if self.ineligible:
            details = "; ".join(f"{scan} missing {', '.join(missing)}" for scan, missing in self.ineligible.items())
            logger.warning("Pre-check %s: %d ineligible scans: %s", self.stage_name, len(self.ineligible), details)


class SubjectEligibilityChecker:
    """Index ``subjects_dir`` and check the required patterns of each scan."""

    def __init__(  # noqa: PLR0913 - mirrors the options main.py forwards
        self,
        subjects_dir: str,
        required_patterns: list[str],
        stage_name: str = "",
        bids: bool = False,
        path_to_graph: str = "",
        btree_order: int = 64,
        include_hidden: bool = False,
    ) -> None:
        if isinstance(required_patterns, str):
            raise TypeError("required_patterns must be a list of patterns, not a single string")
        for pattern in required_patterns:
            if not isinstance(pattern, str):
                raise TypeError(f"Each required pattern must be a string, got {pattern!r}")
            _check_relative(pattern)
        self.subjects_dir = subjects_dir
        self.required_patterns = list(required_patterns)
        self.stage_name = stage_name
        self.bids = bids
        self.path_to_graph = path_to_graph
        self.btree_order = btree_order
        self.include_hidden = include_hidden
        self._tree: BTree | None = None

    def list_scans(self) -> list[ScanId]:
        """Return the scans found under ``subjects_dir``, sorted."""
        if not os.path.isdir(self.subjects_dir):
            logger.warning("Subjects directory not found or not a directory: %s", self.subjects_dir)
            return []
        scans: set[ScanId] = set()
        for name in self._visible_dirs(self.subjects_dir):
            if not self.bids:
                scans.add(ScanId(name))
            elif name.startswith(BIDS_SUBJECT_PREFIX):
                scans.update(self._bids_scans(name))
        return sorted(scans)

    def scan_has_pattern(self, scan_id: ScanId, pattern: str) -> bool:
        """Return True when an indexed file matches ``<path_prefix>/<pattern>``."""
        _check_relative(pattern)
        full_pattern = self._full_pattern(scan_id, pattern)
        tree = self._index()
        cut = min((i for i, char in enumerate(full_pattern) if char in WILDCARD_CHARS), default=None)
        if cut is None:
            return tree.search(full_pattern) is not None
        literal = full_pattern[:cut]
        return any(fnmatchcase(key, full_pattern) for key, _ in tree.range_query(literal, literal + _KEY_MAX_SUFFIX))

    def refresh(self) -> None:
        """Rebuild the index from the files present under ``subjects_dir``."""
        tree = BTree(t=self.btree_order)
        for root, dirs, files in os.walk(self.subjects_dir):
            if not self.include_hidden:
                dirs[:] = [d for d in dirs if not _is_hidden(d)]
                files = [f for f in files if not _is_hidden(f)]
            for fname in files:
                path = join(root, fname)
                try:
                    stat = os.stat(path)
                except OSError:
                    logger.debug("Cannot stat %s; left out of the index", path, exc_info=True)
                    continue
                key = os.path.relpath(path, self.subjects_dir).replace(os.sep, "/")
                tree.insert(
                    key,
                    {"type": "file", "size": stat.st_size, "modified": stat.st_mtime, "ext": splitext(fname)[1]},
                )
        self._tree = tree

    def check(self, save_index_to: str | None = None) -> SubjectEligibilityReport:
        """Re-index, then sort each listed scan into eligible or ineligible."""
        self.refresh()
        report = SubjectEligibilityReport(stage_name=self.stage_name, input_dir=self.subjects_dir)
        for scan in self.list_scans():
            missing = [p for p in self.required_patterns if not self.scan_has_pattern(scan, p)]
            if missing:
                report.ineligible[scan] = missing
            else:
                report.eligible.append(scan)
        if save_index_to is not None:
            self._save_index(save_index_to)
        return report

    def _index(self) -> BTree:
        if self._tree is None:
            self.refresh()
        return self._tree

    def _visible_dirs(self, directory: str) -> list[str]:
        return sorted(
            entry.name
            for entry in os.scandir(directory)
            if entry.is_dir() and (self.include_hidden or not _is_hidden(entry.name))
        )

    def _bids_scans(self, subject: str) -> set[ScanId]:
        """Scans of one BIDS subject: path_to_graph matches, else ses- folders, else the subject alone."""
        subject_dir = join(self.subjects_dir, subject)
        scans: set[ScanId] = set()
        if self.path_to_graph:
            for match in glob.glob(self.path_to_graph, root_dir=subject_dir, include_hidden=self.include_hidden):
                segments = match.replace(os.sep, "/").split("/")
                session = next((s for s in segments if s.startswith(BIDS_SESSION_PREFIX)), None)
                acquisition = next((s for s in segments if s.startswith(BIDS_ACQUISITION_PREFIX)), None)
                scans.add(ScanId(subject, session, acquisition))
        if not scans:
            sessions = [d for d in self._visible_dirs(subject_dir) if d.startswith(BIDS_SESSION_PREFIX)]
            scans = {ScanId(subject, session) for session in sessions} or {ScanId(subject)}
        return scans

    @staticmethod
    def _full_pattern(scan_id: ScanId, pattern: str) -> str:
        """The single place a scan and a pattern are composed (acquisition is not part of the path yet)."""
        return f"{scan_id.path_prefix}/{pattern}"

    def _save_index(self, path: str) -> None:
        """Write the index as BTree.to_dict() JSON, atomically."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(self._index().to_dict(), f, indent=2)
        os.replace(tmp_path, path)
        logger.log(OUTPUT, "Index saved to %s", path)
