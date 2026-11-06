"""Per-scan eligibility checks over a B-tree index of a subjects directory.

The streaming arrival barrier polls ``SubjectEligibilityChecker`` through
``list_scans``, ``scan_has_pattern`` and ``refresh``, with patterns relative to
the scan folder; the sequential pre-stage report uses ``check_subject_patterns``,
with patterns relative to the subject folder. Patterns and paths come from the
user's configuration, so they are checked, never trusted.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from os.path import join, splitext

import champollion_utils.logger  # noqa: F401 - installs the OUTPUT level helpers
from champollion_utils.logger import OUTPUT

from champollion_pipeline.utils.subject_folders import list_morphologist_subjects

from .btree import BTree
from .scan_id import ScanId

logger = logging.getLogger(__name__)

BIDS_SUBJECT_PREFIX = "sub-"
# Morphologist's per-subject T1 folder; must equal the middle segment of ScanId.path_prefix.
T1MRI_DIR = "t1mri"
WILDCARD_CHARS = "*?["
# The only wildcard of a subject-relative pattern, as in run_cortical_tiles.resolve_graph_paths.
SUBJECT_PATTERN_WILDCARD = "*"
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
    """Index ``subjects_dir`` and check the required patterns of each scan.

    A scan is one Morphologist acquisition folder ``<subject>/t1mri/<acquisition>``. Patterns are
    relative to it for ``scan_has_pattern`` and ``check``, and to the subject folder for
    ``scan_has_subject_pattern`` and ``check_subject_patterns``. ``path_to_graph`` only records the
    dataset's graph sub-path for the caller and is read by no method. Not thread-safe: use one
    instance per polling thread.
    """

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
        self._warned_subjects: set[str] = set()

    def list_scans(self) -> list[ScanId]:
        """Return the scans of the subject folders under ``subjects_dir``, sorted.

        The subjects come from ``list_morphologist_subjects``; an OSError from listing them reaches the caller.
        """
        if not os.path.isdir(self.subjects_dir):
            logger.warning("Subjects directory not found or not a directory: %s", self.subjects_dir)
            return []
        scans: list[ScanId] = []
        for subject in list_morphologist_subjects(self.subjects_dir):
            if self.bids and not subject.startswith(BIDS_SUBJECT_PREFIX):
                continue
            acquisitions = self._acquisitions(subject)
            if not acquisitions:
                self._warn_no_acquisition(subject)
                continue
            scans.extend(ScanId(subject, acquisition, bids=self.bids) for acquisition in acquisitions)
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
            # The BTree's shape follows insertion order; top-down os.walk descends in the order of dirs.
            dirs.sort()
            for fname in sorted(files):
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
        """Judge each scan on patterns relative to the scan folder."""
        return self._sort_scans(self.scan_has_pattern, save_index_to)

    def scan_has_subject_pattern(self, scan_id: ScanId, pattern: str) -> bool:
        """Return True when a file inside the folder of ``scan_id`` matches ``<subject>/<pattern>``.

        Reads the file system as input_qc.tsv does, so the two views agree: ``*`` is the only
        wildcard and stays inside one path segment, and symbolic links are followed.
        """
        _check_relative(pattern)
        escaped = SUBJECT_PATTERN_WILDCARD.join(glob.escape(part) for part in pattern.split(SUBJECT_PATTERN_WILDCARD))
        full = join(glob.escape(join(self.subjects_dir, scan_id.subject)), escaped)
        scan_dir = os.path.normpath(join(self.subjects_dir, scan_id.path_prefix)) + os.sep
        return any(
            os.path.normpath(path).startswith(scan_dir) and os.path.isfile(path)
            for path in sorted(glob.glob(full, include_hidden=self.include_hidden))
        )

    def check_subject_patterns(  # noqa: V105 - used by main.py
        self, save_index_to: str | None = None
    ) -> SubjectEligibilityReport:
        """Judge each scan on patterns relative to the subject folder."""
        return self._sort_scans(self.scan_has_subject_pattern, save_index_to)

    def _sort_scans(
        self, has_pattern: Callable[[ScanId, str], bool], save_index_to: str | None
    ) -> SubjectEligibilityReport:
        """Re-index, then sort each listed scan into eligible or ineligible."""
        self.refresh()
        report = SubjectEligibilityReport(stage_name=self.stage_name, input_dir=self.subjects_dir)
        for scan in self.list_scans():
            missing = [p for p in self.required_patterns if not has_pattern(scan, p)]
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

    def _acquisitions(self, subject: str) -> list[str]:
        """Acquisition folder names under ``<subject>/t1mri``, verbatim; empty when there is none."""
        t1mri = join(self.subjects_dir, subject, T1MRI_DIR)
        if not os.path.isdir(t1mri):
            return []
        try:
            return self._visible_dirs(t1mri)
        except OSError:
            logger.debug("Cannot list %s; no acquisition taken from it", t1mri, exc_info=True)
            return []

    def _warn_no_acquisition(self, subject: str) -> None:
        """Warn once per subject, per checker, that it holds no acquisition folder."""
        if subject in self._warned_subjects:
            return
        self._warned_subjects.add(subject)
        logger.warning(
            "Subject folder %s holds no %s/<acquisition> folder: no scan listed for it",
            join(self.subjects_dir, subject),
            T1MRI_DIR,
        )

    @staticmethod
    def _full_pattern(scan_id: ScanId, pattern: str) -> str:
        """The single place a scan and a pattern are composed: ``<subject>/t1mri/<acquisition>/<pattern>``."""
        return f"{scan_id.path_prefix}/{pattern}"

    def _save_index(self, path: str) -> None:
        """Write the index as BTree.to_dict() JSON, atomically."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(self._index().to_dict(), f, indent=2)
        os.replace(tmp_path, path)
        logger.log(OUTPUT, "Index saved to %s", path)
