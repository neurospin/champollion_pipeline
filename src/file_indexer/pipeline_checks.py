#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre- and post-stage pipeline checks using the B-tree file indexer.

Before a stage runs:
    SubjectEligibilityChecker indexes the input subjects directory and
    reports which subjects have all required files.

After a stage runs:
    build_output_report indexes the output directory and summarises
    what was generated (file count by extension).
"""

import fnmatch
import json
import os
from dataclasses import dataclass, field
from os.path import join, relpath, splitext
from typing import Optional

from .btree import BTree


# ---------------------------------------------------------------------------
# Pre-stage eligibility check
# ---------------------------------------------------------------------------

@dataclass
class SubjectEligibilityReport:
    """Result of a per-subject eligibility check before a pipeline stage."""

    stage_name: str
    input_dir: str
    eligible: list
    ineligible: dict  # subject → list of missing patterns

    def print(self) -> None:
        n_total = len(self.eligible) + len(self.ineligible)
        sep = "-" * 40
        lines = [
            "",
            f"--- Pre-check: {self.stage_name} ---",
            f"  Input directory : {self.input_dir}",
            f"  Total subjects  : {n_total}",
            f"  Eligible        : {len(self.eligible)}",
        ]
        if self.ineligible:
            lines.append(f"  Ineligible ({len(self.ineligible)}):")
            for sub in sorted(self.ineligible):
                for pattern in self.ineligible[sub]:
                    lines.append(f"    {sub}: missing {pattern}")
        lines.append(sep)
        print("\n".join(lines), flush=True)


class SubjectEligibilityChecker:
    """Index a subjects directory with a B-tree and check per-subject file requirements."""

    def __init__(
        self,
        subjects_dir: str,
        required_patterns: list,
        stage_name: str = "",
        btree_order: int = 64,
        include_hidden: bool = False,
    ) -> None:
        self.subjects_dir = subjects_dir
        self.required_patterns = required_patterns
        self.stage_name = stage_name
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

    def _get_subjects(self) -> list:
        """Return sorted first-level subdirectory names as subject IDs."""
        try:
            return sorted([
                d for d in os.listdir(self.subjects_dir)
                if os.path.isdir(join(self.subjects_dir, d))
                and (self.include_hidden or not d.startswith("."))
            ])
        except OSError:
            return []

    # ------------------------------------------------------------------
    # Pattern matching
    # ------------------------------------------------------------------

    def _subject_has_pattern(self, subject: str, pattern: str) -> bool:
        """Return True if any indexed key matches <subject>/<pattern>."""
        full_pattern = f"{subject}/{pattern}"
        if "*" in full_pattern or "?" in full_pattern:
            prefix = full_pattern.split("*")[0].split("?")[0]
            candidates = self._tree.range_query(prefix, prefix + "\xff")
            return any(fnmatch.fnmatch(key, full_pattern) for key, _ in candidates)
        return self._tree.search(full_pattern) is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, save_index_to: Optional[str] = None) -> SubjectEligibilityReport:
        """Build index, check each subject, optionally save index JSON."""
        self._tree = self._build_index()

        if save_index_to:
            os.makedirs(os.path.dirname(save_index_to) or ".", exist_ok=True)
            with open(save_index_to, "w") as fh:
                json.dump(self._tree.to_dict(), fh, indent=2)

        subjects = self._get_subjects()
        eligible = []
        ineligible = {}

        for subject in subjects:
            missing = [
                p for p in self.required_patterns
                if not self._subject_has_pattern(subject, p)
            ]
            if missing:
                ineligible[subject] = missing
            else:
                eligible.append(subject)

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
