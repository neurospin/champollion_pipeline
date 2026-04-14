#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IndexedScriptBuilder: ScriptBuilder subclass that runs pre/post index reports.

Subclass this instead of ScriptBuilder when your script should report:
  - which subjects have the required input files (pre-check)
  - what files were generated in the output directory (post-check)

Usage
-----
Override three methods in your subclass:

    def subject_input_dir(self) -> str | None:
        return self.args.input   # or None to skip pre-check

    def required_file_patterns(self) -> list[str] | None:
        return ["t1mri/t1/default_analysis/folds/3.3/*.arg"]

    def script_output_dir(self) -> str | None:
        return self.args.output  # or None to skip post-report

Pass ``--index-dir /path/to/dir`` at the command line to save the index
JSON files to disk.  Omit it to print the reports only.
"""

import os
from os.path import join, exists
from typing import Optional

from champollion_utils.script_builder import ScriptBuilder


class IndexedScriptBuilder(ScriptBuilder):
    """ScriptBuilder subclass that runs pre/post B-tree index reports around run()."""

    def __init__(self, script_name: str, description: str):
        super().__init__(script_name, description)
        self.add_optional_argument(
            "--index-dir",
            "Directory to save pre/post index JSON files. Omit to skip saving.",
            default=None,
        )

    # ------------------------------------------------------------------
    # Override these in subclasses
    # ------------------------------------------------------------------

    def subject_input_dir(self) -> Optional[str]:
        """Return the subjects input directory, or None to skip the pre-check."""
        return None

    def required_file_patterns(self) -> Optional[list]:
        """Return per-subject glob patterns required for eligibility, or None to skip."""
        return None

    def script_output_dir(self) -> Optional[str]:
        """Return the script output directory, or None to skip the post-report."""
        return None

    # ------------------------------------------------------------------
    # Lifecycle hooks (called by ScriptBuilder.main)
    # ------------------------------------------------------------------

    def before_run(self) -> None:
        input_dir = self.subject_input_dir()
        patterns = self.required_file_patterns()
        if not input_dir or not patterns:
            return
        from file_indexer.pipeline_checks import SubjectEligibilityChecker
        index_path = self._index_path("pre")
        checker = SubjectEligibilityChecker(
            input_dir, patterns, stage_name=self.script_name
        )
        checker.check(save_index_to=index_path).print()

    def after_run(self, success: bool) -> None:
        if not success:
            return
        out_dir = self.script_output_dir()
        if not out_dir or not exists(out_dir):
            return
        from file_indexer.pipeline_checks import build_output_report
        index_path = self._index_path("post")
        build_output_report(out_dir, self.script_name, save_index_to=index_path).print()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_path(self, stage: str) -> Optional[str]:
        """Return the JSON save path if --index-dir was given, else None."""
        index_dir = getattr(self.args, "index_dir", None)
        if not index_dir:
            return None
        os.makedirs(index_dir, exist_ok=True)
        return join(index_dir, f"index_{stage}_{self.script_name}.json")
