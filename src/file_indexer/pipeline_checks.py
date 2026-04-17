#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre- and post-stage pipeline checks using the B-tree file indexer.

Before a stage runs:
    SubjectEligibilityChecker indexes the input subjects directory and
    reports which scans have all required files.  In BIDS mode the unit
    of work is a (subject, session, acquisition) triple (``ScanId``);
    in non-BIDS mode it is simply a subject directory.

After a stage runs:
    build_output_report indexes the output directory and summarises
    what was generated (file count by extension).
"""

import fnmatch
import glob as glob_mod
import json
import os
import re
from dataclasses import dataclass, field
from os.path import join, relpath, splitext
from typing import Dict, List, Optional

from .btree import BTree
from .scan_id import ScanId


# ---------------------------------------------------------------------------
# Pre-stage eligibility check
# ---------------------------------------------------------------------------

@dataclass
class SubjectEligibilityReport:
    """Result of a per-scan eligibility check before a pipeline stage."""

    stage_name: str
    input_dir: str
    eligible: List[ScanId]
    ineligible: Dict[ScanId, List[str]]  # scan → list of missing patterns

    def print(self) -> None:
        n_total = len(self.eligible) + len(self.ineligible)
        sep = "-" * 40
        lines = [
            "",
            f"--- Pre-check: {self.stage_name} ---",
            f"  Input directory : {self.input_dir}",
            f"  Total scans     : {n_total}",
            f"  Eligible        : {len(self.eligible)}",
        ]
        if self.ineligible:
            lines.append(f"  Ineligible ({len(self.ineligible)}):")
            for scan_id in sorted(self.ineligible):
                for pattern in self.ineligible[scan_id]:
                    lines.append(f"    {scan_id}: missing {pattern}")
        lines.append(sep)
        print("\n".join(lines), flush=True)


class SubjectEligibilityChecker:
    """Index a subjects directory with a B-tree and check per-scan file requirements.

    Parameters
    ----------
    subjects_dir:
        Root directory whose first-level sub-directories are subject folders.
    required_patterns:
        Glob patterns relative to each scan's ``path_prefix`` that must all
        match at least one indexed file for the scan to be eligible.
    stage_name:
        Human-readable label used in the eligibility report.
    bids:
        When ``True``, enumerate (subject, session, acquisition) triples
        instead of bare subject directories.  Patterns are expected to be
        relative to ``{subject}/{session}/``.
    path_to_graph:
        Glob pattern relative to the subject directory used to discover BIDS
        scan directories (e.g. ``"ses-*/t1mri/acq-*/default_analysis/folds/3.1"``).
        Only used when ``bids=True``.
    btree_order:
        B-tree minimum degree (default 64).
    include_hidden:
        Include hidden files and directories when walking the filesystem.
    """

    def __init__(
        self,
        subjects_dir: str,
        required_patterns: list,
        stage_name: str = "",
        bids: bool = False,
        path_to_graph: str = "",
        btree_order: int = 64,
        include_hidden: bool = False,
    ) -> None:
        self.subjects_dir = subjects_dir
        self.required_patterns = required_patterns
        self.stage_name = stage_name
        self.bids = bids
        self.path_to_graph = path_to_graph
        self.btree_order = btree_order
        self.include_hidden = include_hidden
        self._tree: Optional[BTree] = None

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self) -> BTree:
        tree = BTree(t=self.btree_order)
        for dirpath, dirnames, filenames in os.walk(self.subjects_dir):
            if not self.include_hidden:
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fname in filenames:
                if not self.include_hidden and fname.startswith("."):
                    continue
                fpath = join(dirpath, fname)
                key = relpath(fpath, self.subjects_dir)
                try:
                    s = os.stat(fpath)
                    tree.insert(key, {
                        "type": "file",
                        "size": s.st_size,
                        "modified": s.st_mtime,
                        "ext": splitext(fname)[1],
                    })
                except OSError:
                    pass
        return tree

    # ------------------------------------------------------------------
    # Scan enumeration
    # ------------------------------------------------------------------

    def _get_scan_ids(self) -> List[ScanId]:
        """Return the list of scans to check.

        Non-BIDS: one ``ScanId(subject=d)`` per top-level directory.
        BIDS: one ``ScanId(subject, session, acquisition)`` per unique
              scan discovered via ``path_to_graph`` glob.
        """
        if self.bids:
            return self._enumerate_bids_scans()

        try:
            return [
                ScanId(subject=d)
                for d in sorted(os.listdir(self.subjects_dir))
                if os.path.isdir(join(self.subjects_dir, d))
                and (self.include_hidden or not d.startswith("."))
            ]
        except OSError:
            return []

    def _enumerate_bids_scans(self) -> List[ScanId]:
        """Walk *subjects_dir* and enumerate BIDS (subject, session, acquisition) triples.

        Uses ``self.path_to_graph`` as a glob pattern relative to each subject
        directory to locate actual scan directories, then extracts ``ses-*`` and
        ``acq-*`` BIDS entities via regex — mirroring what ``cortical_tiles`` does
        internally with the same pattern.

        Falls back to listing ``ses-*`` subdirectories when no glob matches are
        found (e.g. ``path_to_graph`` is empty or no graphs exist yet).
        """
        try:
            subjects = sorted(
                d for d in os.listdir(self.subjects_dir)
                if os.path.isdir(join(self.subjects_dir, d))
                and (self.include_hidden or not d.startswith("."))
            )
        except OSError:
            return []

        scans: List[ScanId] = []
        for subject in subjects:
            sub_dir = join(self.subjects_dir, subject)
            scans.extend(self._bids_scans_for_subject(sub_dir, subject))
        return scans

    def _bids_scans_for_subject(self, sub_dir: str, subject: str) -> List[ScanId]:
        """Return all BIDS ``ScanId`` objects for one subject.

        Globs ``{sub_dir}/{path_to_graph}`` and extracts ``ses-*`` / ``acq-*``
        entities from each matching path.  When no matches are found, falls
        back to enumerating ``ses-*`` session directories directly.
        """
        if self.path_to_graph:
            pattern = join(sub_dir, self.path_to_graph)
            matches = sorted(glob_mod.glob(pattern))
        else:
            matches = []

        if not matches:
            # Fallback: list ses-* dirs without acquisition detail
            try:
                sessions = sorted(
                    d for d in os.listdir(sub_dir)
                    if os.path.isdir(join(sub_dir, d)) and d.startswith("ses-")
                )
            except OSError:
                sessions = []
            if sessions:
                return [ScanId(subject=subject, session=s) for s in sessions]
            return [ScanId(subject=subject)]

        seen: set = set()
        result: List[ScanId] = []
        for match in matches:
            # Make path relative to the subject directory for entity extraction
            rel = match[len(sub_dir):].lstrip("/\\")
            ses_m = re.search(r"(ses-[^/\\]+)", rel)
            acq_m = re.search(r"(acq-[^/\\]+)", rel)
            scan_id = ScanId(
                subject=subject,
                session=ses_m.group(1) if ses_m else None,
                acquisition=acq_m.group(1) if acq_m else None,
            )
            if scan_id not in seen:
                seen.add(scan_id)
                result.append(scan_id)

        return sorted(result)

    # ------------------------------------------------------------------
    # Pattern matching
    # ------------------------------------------------------------------

    def _scan_has_pattern(self, scan_id: ScanId, pattern: str) -> bool:
        """Return True if any indexed key matches ``{scan_id.path_prefix}/{pattern}``.

        Uses ``scan_id.path_prefix`` as the range-query lower bound so the
        B-tree only visits keys under that scan's directory.
        """
        full_pattern = f"{scan_id.path_prefix}/{pattern}"
        if "*" in full_pattern or "?" in full_pattern:
            prefix = full_pattern.split("*")[0].split("?")[0]
            candidates = self._tree.range_query(prefix, prefix + "\xff")
            return any(fnmatch.fnmatch(key, full_pattern) for key, _ in candidates)
        return self._tree.search(full_pattern) is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, save_index_to: Optional[str] = None) -> SubjectEligibilityReport:
        """Build index, check each scan, optionally save index JSON."""
        self._tree = self._build_index()

        if save_index_to:
            os.makedirs(os.path.dirname(save_index_to) or ".", exist_ok=True)
            with open(save_index_to, "w") as fh:
                json.dump(self._tree.to_dict(), fh, indent=2)

        scan_ids = self._get_scan_ids()
        eligible: List[ScanId] = []
        ineligible: Dict[ScanId, List[str]] = {}

        for scan_id in scan_ids:
            missing = [
                p for p in self.required_patterns
                if not self._scan_has_pattern(scan_id, p)
            ]
            if missing:
                ineligible[scan_id] = missing
            else:
                eligible.append(scan_id)

        return SubjectEligibilityReport(
            stage_name=self.stage_name,
            input_dir=self.subjects_dir,
            eligible=eligible,
            ineligible=ineligible,
        )


# ---------------------------------------------------------------------------
# Post-stage output report
# ---------------------------------------------------------------------------

@dataclass
class OutputIndexReport:
    """Summary of files generated by a pipeline stage."""

    stage_name: str
    output_dir: str
    total_files: int
    by_ext: dict  # extension → count

    def print(self) -> None:
        sep = "-" * 40
        ext_str = "  ".join(
            f"{ext or '(no ext)'}={count}"
            for ext, count in sorted(self.by_ext.items())
        )
        lines = [
            "",
            f"--- Post-check: {self.stage_name} ---",
            f"  Output directory: {self.output_dir}",
            f"  Total files     : {self.total_files}",
        ]
        if ext_str:
            lines.append(f"  By extension    : {ext_str}")
        lines.append(sep)
        print("\n".join(lines), flush=True)


def build_output_report(
    output_dir: str,
    stage_name: str,
    save_index_to: Optional[str] = None,
    btree_order: int = 64,
) -> OutputIndexReport:
    """Walk *output_dir*, build a B-tree index, count files by extension.

    Parameters
    ----------
    output_dir:
        Directory produced by the pipeline stage.
    stage_name:
        Human-readable stage label used in the report.
    save_index_to:
        If provided, write the serialised B-tree to this JSON path.
    btree_order:
        B-tree minimum degree (default 64).
    """
    tree = BTree(t=btree_order)
    by_ext: dict = {}
    total = 0

    for dirpath, dirnames, filenames in os.walk(output_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fname in filenames:
            if fname.startswith("."):
                continue
            fpath = join(dirpath, fname)
            key = relpath(fpath, output_dir)
            ext = splitext(fname)[1]
            try:
                s = os.stat(fpath)
                tree.insert(key, {
                    "type": "file",
                    "size": s.st_size,
                    "modified": s.st_mtime,
                    "ext": ext,
                })
                by_ext[ext] = by_ext.get(ext, 0) + 1
                total += 1
            except OSError:
                pass

    if save_index_to:
        os.makedirs(os.path.dirname(save_index_to) or ".", exist_ok=True)
        with open(save_index_to, "w") as fh:
            json.dump(tree.to_dict(), fh, indent=2)

    return OutputIndexReport(
        stage_name=stage_name,
        output_dir=output_dir,
        total_files=total,
        by_ext=by_ext,
    )
