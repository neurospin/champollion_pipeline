"""Per-scan eligibility checks over a B-tree index of a subjects directory.

The streaming arrival barrier polls ``SubjectEligibilityChecker`` through
``list_scans``, ``scan_has_pattern`` and ``refresh``, with patterns relative to
the scan folder; the sequential pre-stage report uses ``check_subject_patterns``,
with patterns relative to the subject folder. After a sequential stage,
``build_output_report`` indexes the stage's output directory and counts its files
by extension. ``FileIndexStageIndexer`` is the default StageIndexer adapter main.py
uses for the sequential stages' pre and post index files. Patterns and paths come
from the user's configuration, so they are checked, never trusted.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from os.path import join, splitext

import champollion_utils.logger  # noqa: F401 - installs the OUTPUT level helpers
from champollion_utils.logger import OUTPUT

from champollion_pipeline.utils.graph_item import GraphItem, LooseGraphError
from champollion_pipeline.utils.input_kind import graph_files
from champollion_pipeline.utils.skeleton_qc import ANY_DEPTH_SEGMENT, SKELETON_ALTERNATIVE
from champollion_pipeline.utils.subject_folders import find_graph_subject_folder, list_morphologist_subjects

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
# Shown for the empty extension in the post-stage summary.
NO_EXT_LABEL = "(no ext)"
# Graph discovery: a labelled graph's own folder ends with one of these.
_LABELLING_SET_ENDINGS = ("_auto", "_manual")
# Graph discovery: hemisphere letters, as the first letter of a classic graph name.
_SIDES = ("L", "R")
# Graph discovery: the hemi-L / hemi-R entity of a graph name.
_HEMI_ENTITY = re.compile(r"(?:^|_)hemi-([LR])(?=_|\.arg$)")
_DIGIT_RUN = re.compile(r"([0-9]+)")


def _check_relative(pattern: str) -> None:
    """Raise ValueError, naming the pattern, when it is absolute or has a '..' path segment."""
    if os.path.isabs(pattern) or ".." in pattern.split("/"):
        raise ValueError(f"Pattern must be a relative path inside the scan folder: {pattern!r}")


def _is_hidden(name: str) -> bool:
    return name.startswith(".")


def _fetch_file_index(root: str, btree_order: int, include_hidden: bool = False) -> BTree:
    """Index the files under root: key = POSIX path relative to root, inserted in code-point walk order."""
    tree = BTree(t=btree_order)
    _walk_into(tree, root, root, include_hidden)
    return tree


def _walk_into(tree: BTree, root: str, key_root: str, include_hidden: bool) -> None:
    """Insert the files under root into tree: key = POSIX path relative to key_root, in code-point walk order."""
    for folder, dirs, files in os.walk(root):
        # The BTree's shape follows insertion order; top-down os.walk descends in the order of dirs.
        dirs[:] = sorted(d for d in dirs if include_hidden or not _is_hidden(d))
        for fname in sorted(files):
            if not include_hidden and _is_hidden(fname):
                continue
            path = join(folder, fname)
            try:
                stat = os.stat(path)
            except OSError:
                logger.debug("Cannot stat %s; left out of the index", path, exc_info=True)
                continue
            key = os.path.relpath(path, key_root).replace(os.sep, "/")
            tree.insert(
                key, {"type": "file", "size": stat.st_size, "modified": stat.st_mtime, "ext": splitext(fname)[1]}
            )


def _save_index_json(tree: BTree, path: str) -> None:
    """Write tree as BTree.to_dict() JSON, atomically, creating the parent folders."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(tree.to_dict(), f, indent=2)
    os.replace(tmp_path, path)
    logger.log(OUTPUT, "Index saved to %s", path)


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

    A scan is one Morphologist acquisition folder ``<subject>/t1mri/<acquisition>``, or, for a subject with no such
    folder but a graph below it (a morphologist-cli tree), the subject folder itself. Patterns are
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
        graph_subjects: frozenset[str] | None = None
        for subject in list_morphologist_subjects(self.subjects_dir):
            if self.bids and not subject.startswith(BIDS_SUBJECT_PREFIX):
                continue
            acquisitions = self._acquisitions(subject)
            if not acquisitions:
                if graph_subjects is None:
                    graph_subjects = self._graph_subjects()
                if subject in graph_subjects:
                    scans.append(ScanId(subject, "", bids=self.bids))
                else:
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
        self._tree = _fetch_file_index(self.subjects_dir, self.btree_order, self.include_hidden)

    def check(self, save_index_to: str | None = None) -> SubjectEligibilityReport:
        """Judge each scan on patterns relative to the scan folder."""
        return self._sort_scans(self.scan_has_pattern, save_index_to)

    def scan_has_subject_pattern(self, scan_id: ScanId, pattern: str) -> bool:
        """Return True when one alternative of ``pattern`` (split on ``|``) matches a file of ``scan_id``.

        An alternative starting with ``**/`` matches a file at any depth below the subject folder, by the walk
        rule of skeleton_qc.list_subject_skeleton_files: the checker's index, which leaves out dot-named entries
        and does not enter a symlinked folder. Any other alternative is matched by ``_has_scan_file``. A pattern
        holding neither marker gives the result it gave before. O(entries below the subject folder).
        """
        _check_relative(pattern)
        return any(
            self._has_any_depth_file(scan_id, alternative[len(ANY_DEPTH_SEGMENT) :])
            if alternative.startswith(ANY_DEPTH_SEGMENT)
            else self._has_scan_file(scan_id, alternative)
            for alternative in pattern.split(SKELETON_ALTERNATIVE)
        )

    def _has_any_depth_file(self, scan_id: ScanId, tail: str) -> bool:
        """True when an indexed file below ``<subject>/`` ends with the segments of ``tail``, ``*`` in one segment."""
        regex = re.compile(
            "(?:^|/)" + "[^/]*".join(re.escape(part) for part in tail.split(SUBJECT_PATTERN_WILDCARD)) + "$"
        )
        prefix = f"{scan_id.subject}/"
        return any(
            regex.search(key[len(prefix) :]) for key, _ in self._index().range_query(prefix, prefix + _KEY_MAX_SUFFIX)
        )

    def _has_scan_file(self, scan_id: ScanId, pattern: str) -> bool:
        """True when a file inside the folder of ``scan_id`` matches ``<subject>/<pattern>``.

        Reads the file system as input_qc.tsv does, so the two views agree: ``*`` is the only
        wildcard and stays inside one path segment, and symbolic links are followed.
        """
        escaped = SUBJECT_PATTERN_WILDCARD.join(glob.escape(part) for part in pattern.split(SUBJECT_PATTERN_WILDCARD))
        full = join(glob.escape(join(self.subjects_dir, scan_id.subject)), escaped)
        scan_dir = os.path.normpath(join(self.subjects_dir, self._scan_folder(scan_id))) + os.sep
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
                report.ineligible[scan] = [self._describe_missing(scan, p) for p in missing]
            else:
                report.eligible.append(scan)
        if save_index_to is not None:
            self._save_index(save_index_to)
        return report

    def _describe_missing(self, scan_id: ScanId, pattern: str) -> str:
        """Text a report names for a missing ``pattern``: never the raw alternatives string.

        A pattern with alternatives gives its first (classic) one when the subject has a t1mri folder, else its
        last (morphologist-cli) one; any other pattern is returned unchanged. O(1) plus one stat.
        """
        alternatives = pattern.split(SKELETON_ALTERNATIVE)
        if len(alternatives) == 1:
            return pattern
        has_t1mri = os.path.isdir(join(self.subjects_dir, scan_id.subject, T1MRI_DIR))
        return alternatives[0] if has_t1mri else alternatives[-1]

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

    def _graph_subjects(self) -> frozenset[str]:
        """Names of the child folders of ``subjects_dir`` that hold a graph, by ``find_graph_subject_folder``.

        A loose graph, or one of ``subjects_dir`` itself, names no child subject and is skipped. O(entries below).
        """
        root_name = os.path.basename(os.path.abspath(self.subjects_dir))
        found = (find_graph_subject_folder(root_name, parts, name) for parts, name in graph_files(self.subjects_dir))
        return frozenset(folder[0] for folder in found if folder)

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
    def _scan_folder(scan_id: ScanId) -> str:
        """Relative folder of a scan: ``<subject>/t1mri/<acquisition>``, or the subject folder of a graph-only scan."""
        return scan_id.path_prefix if scan_id.acquisition else scan_id.subject

    @staticmethod
    def _full_pattern(scan_id: ScanId, pattern: str) -> str:
        """The single place a scan and a pattern are composed: ``<scan folder>/<pattern>``."""
        return f"{SubjectEligibilityChecker._scan_folder(scan_id)}/{pattern}"

    def _save_index(self, path: str) -> None:
        """Write the index as BTree.to_dict() JSON, atomically."""
        _save_index_json(self._index(), path)


@dataclass
class OutputIndexReport:
    """Summary of the files a pipeline stage left in its output directory."""

    stage_name: str
    output_dir: str
    total_files: int
    by_ext: dict[str, int] = field(default_factory=dict)

    def log_summary(self) -> None:
        """Log the post-stage summary record: file total, then one <extension>=<count> entry per by_ext item."""
        entries = ", ".join(f"{ext or NO_EXT_LABEL}={count}" for ext, count in self.by_ext.items())
        logger.log(
            OUTPUT,
            "Post-check %s: %d files in %s%s",
            self.stage_name,
            self.total_files,
            self.output_dir,
            f" ({entries})" if entries else "",
        )


def build_output_report(
    output_dir: str,
    stage_name: str,
    save_index_to: str | None = None,
    btree_order: int = 64,
) -> OutputIndexReport:
    """Index ``output_dir`` and count its files by extension; save the index to ``save_index_to`` when given.

    Dot-named files and folders are left out, as is a file ``os.stat`` cannot read. O(files).
    """
    tree = _fetch_file_index(output_dir, btree_order)
    if save_index_to is not None:
        _save_index_json(tree, save_index_to)
    counts = Counter(value["ext"] for _, value in tree)
    return OutputIndexReport(
        stage_name=stage_name, output_dir=output_dir, total_files=len(tree), by_ext=dict(sorted(counts.items()))
    )


def build_input_index(folders: list[str], save_index_to: str, btree_order: int = 64) -> int:
    """Index the files of the existing folders among ``folders`` into one JSON index at ``save_index_to``.

    Keys are POSIX paths relative to the folders' common parent (the folder itself when there is one); a missing
    folder adds no key; dot-named entries are left out. Returns the number of files indexed. O(files).
    """
    key_root = os.path.commonpath([os.path.abspath(folder) for folder in folders])
    tree = BTree(t=btree_order)
    for folder in folders:
        if os.path.isdir(folder):
            _walk_into(tree, os.path.abspath(folder), key_root, include_hidden=False)
    _save_index_json(tree, save_index_to)
    return len(tree)


class FileIndexStageIndexer:  # noqa: V102 - used by main.py
    """Default StageIndexer adapter: B-tree JSON indexes index_<phase>_<stage>.json under outputs_path."""

    def __init__(self, outputs_path: str, bids: bool = False, path_to_graph: str = "") -> None:
        self.outputs_path = outputs_path
        self.bids = bids
        self.path_to_graph = path_to_graph

    def _index_path(self, phase: str, stage_name: str) -> str | None:
        """Path of the pre/post index JSON of a stage, or None without outputs_path."""
        if not self.outputs_path:
            return None
        return join(self.outputs_path, f"index_{phase}_{stage_name}.json")

    def index_pre(  # noqa: V105 - used by main.py
        self, stage_name: str, input_dirs: list[str], required_patterns: list[str]
    ) -> None:
        """Eligibility report and index when patterns are given; else the index of the input folders only.

        Raises ValueError when patterns are given with other than one input folder.
        """
        index_path = self._index_path("pre", stage_name)
        if required_patterns:
            if len(input_dirs) != 1:
                raise ValueError(f"The eligibility pre-check takes one subjects folder, got {input_dirs!r}")
            if self.outputs_path:
                os.makedirs(self.outputs_path, exist_ok=True)
            checker = SubjectEligibilityChecker(
                input_dirs[0],
                required_patterns,
                stage_name=stage_name,
                bids=self.bids,
                path_to_graph=self.path_to_graph,
            )
            checker.check_subject_patterns(save_index_to=index_path).log_summary()
            return
        if index_path is None or not any(os.path.isdir(folder) for folder in input_dirs):
            return
        build_input_index(input_dirs, index_path)

    def index_post(self, stage_name: str, output_dir: str) -> None:  # noqa: V105 - used by main.py
        """Index output_dir and log the post-stage summary."""
        build_output_report(output_dir, stage_name, save_index_to=self._index_path("post", stage_name)).log_summary()

    def discover_graphs(self, folder: str) -> list[GraphItem]:  # noqa: V105 - the port, called by graph readers
        """Return one item per graph (.arg) below folder, attributes read from names alone, in natural path order.

        Each item's path is POSIX, relative to folder; its attributes are subject (name of the graph's subject folder
        from find_graph_subject_folder, folder's own name when that is folder itself), hemisphere (hemi-L/hemi-R
        entity, else a first letter L or R, else None) and labelling_set (its own folder when named *_auto or *_manual,
        else None). No file is opened, no index written.

        Raises:
            NotADirectoryError: folder is not a folder.
            LooseGraphError: a graph sits directly in folder and folder's name does not start with sub-.

        Complexity: O(entries below folder) for the walk, O(n log n) for the sort.
        """
        if not os.path.isdir(folder):
            raise NotADirectoryError(f"Graph discovery needs a folder, got {folder!r}")
        root_name = os.path.basename(os.path.abspath(folder))
        items = []
        for parts, name in graph_files(folder):
            subject_folder = find_graph_subject_folder(root_name, parts, name)
            if subject_folder is None:
                raise LooseGraphError(os.path.join(folder, name))
            attributes = {
                "subject": subject_folder[-1] if subject_folder else root_name,
                "hemisphere": _hemisphere(name),
                "labelling_set": _labelling_set(root_name, parts),
            }
            items.append(GraphItem("/".join((*parts, name)), attributes))
        result = sorted(items, key=lambda item: natural_key(item.path))
        assert len({x.path for x in result}) == len(result), (
            "FileIndexStageIndexer.discover_graphs post: len(set(x.path for x in result)) == len(result)"
        )
        assert all(sorted(x.attributes) == ["hemisphere", "labelling_set", "subject"] for x in result), (
            "FileIndexStageIndexer.discover_graphs post: attribute keys are hemisphere, labelling_set, subject"
        )
        assert all(x.attributes["hemisphere"] in (*_SIDES, None) for x in result), (
            "FileIndexStageIndexer.discover_graphs post: hemisphere in L, R, None"
        )
        assert all(x.attributes["subject"] is not None for x in result), (
            "FileIndexStageIndexer.discover_graphs post: subject is not None"
        )
        return result


def _hemisphere(name: str) -> str | None:
    """Letter of the hemi-L/hemi-R entity of a graph name, else its first letter when L or R, else None."""
    match = _HEMI_ENTITY.search(name)
    if match:
        return match[1]
    return name[0] if name[:1] in _SIDES else None


def _labelling_set(root_name: str, parts: tuple[str, ...]) -> str | None:
    """Name of the graph's own folder when it ends with _auto or _manual, else None."""
    own = parts[-1] if parts else root_name
    return own if own.endswith(_LABELLING_SET_ENDINGS) else None


def natural_key(path: str) -> tuple[tuple[str | int, ...], str]:
    """Sort key comparing runs of digits as numbers; the plain path breaks ties, so the order is total."""
    pieces = _DIGIT_RUN.split(path)
    return (tuple(int(piece) if index % 2 else piece for index, piece in enumerate(pieces)), path)
