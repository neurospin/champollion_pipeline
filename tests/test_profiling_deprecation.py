#!/usr/bin/env python3
"""Deprecation of generate_embeddings ``--profiling`` (TASK-TASK-BDRABCZUK-35E1301A7918).

User decision 2026-11-24: ``--profiling`` is deprecated in 1.1.0 and removed in 2.0. It keeps
running cProfile, warns with a FutureWarning naming the replacement option ``--profile``, and
its row in the Stage 4 options table of ``docs/usage.md`` says it is deprecated.

This file is not named ``tests/test_docs_*.py`` on purpose: the verify selection skips those.

Requirements pinned here (one test each):

* "--profiling emits a deprecation FutureWarning" - TestProfilingDeprecationWarning.
* "Docs: usage.md --profiling row says deprecated, replaced by --profile" - TestUsageProfilingRow.

No model, no evaluate.py, no network: the real parser is used and the profiled run is mocked.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from tests.docs_test_helpers import md_section, read_text
from tests.usage_option_rows import option_table_rows

pytestmark = pytest.mark.unit  # noqa: V107

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USAGE = PROJECT_ROOT / "docs" / "usage.md"

# ``--profile`` as a whole option, not the prefix of ``--profiling`` or ``--profile-mode``.
PROFILE_OPTION = re.compile(r"(?<![\w-])--profile(?![\w-])")


def _stage4_option_rows() -> dict[str, str]:
    """Map the first-cell ``--flag`` of each '| Option |' table row of the docs/usage.md Stage 4 section to the row."""
    return option_table_rows(md_section(read_text(USAGE), r"Stage 4\b[^\n]*"))


class TestProfilingDeprecationWarning:
    """--profiling emits a FutureWarning whose message names --profile as the replacement option."""

    def test_profiling_warns_naming_profile(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # a real cProfile dump, if any, lands here, never in the repo
        script = GenerateEmbeddings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            script.parse_args(["/models", str(tmp_path / "TEST01"), "--masks", "canonical_25", "--profiling"])
            with (
                patch.object(script, "_run_with_profiling", return_value=0) as profiled,
                patch.object(script, "_run_normal", return_value=0),
            ):
                script.run()
        # The flag still profiles in 1.1.0: the warning is added, the dispatch is kept.
        profiled.assert_called_once()
        messages = [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]
        assert any(PROFILE_OPTION.search(message) for message in messages), messages


class TestUsageProfilingRow:
    """The --profiling row of the Stage 4 options table says deprecated, with --profile as the replacement."""

    def test_row_marks_profiling_deprecated_naming_profile(self):
        row = _stage4_option_rows().get("--profiling")
        assert row, "docs/usage.md Stage 4 options table has no --profiling row"
        assert re.search(r"\bdeprecated\b", row, re.IGNORECASE), f"--profiling row does not say deprecated: {row}"
        assert PROFILE_OPTION.search(row), f"--profiling row does not name --profile: {row}"
