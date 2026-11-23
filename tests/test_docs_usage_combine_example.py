#!/usr/bin/env python3
"""Documentation guard for the stage-5 ``champollion-combine`` example in ``docs/usage.md``.

``put_together_embeddings.py`` (``PutTogetherEmbeddings``) takes a positional
``embeddings_source`` plus a required ``--output_path``. The ``docs/usage.md``
stage-5 example must show that interface, not flags the parser rejects.

Requirements pinned here (one test per requirement):

* REQ-EMBVER-BDRABCZUK-FA4D24096DF2 - the example passes the versioned
  ``region_embeddings/`` directory as the positional ``embeddings_source``.
* REQ-EMBVER-BDRABCZUK-465E3A7A5E89 - every flag in the example is an option
  string declared by the ``PutTogetherEmbeddings`` argument parser.

``docs/usage.md`` is read as plain text. ``PutTogetherEmbeddings`` is only
instantiated to inspect its ``argparse`` parser (as in
``tests/test_readme_accuracy.py``'s REQ-DOCS-10 checks): no argument parsing,
no filesystem access, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings
from tests.docs_test_helpers import md_section, norm, option_strings, positional_args, read_text, required_examples

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USAGE = PROJECT_ROOT / "docs" / "usage.md"

COMMAND = "champollion-combine"
EXPECTED_SOURCE = "/data/myproject/derivatives/champollion_V1/canonical_25/region_embeddings/"


def _combine_examples() -> list[list[str]]:
    """Argument tokens (after the command name) of every ``pixi run champollion-combine`` in stage 5."""
    stage5 = md_section(read_text(USAGE), r"Stage 5\b[^\n]*")
    return required_examples(stage5, COMMAND, "docs/usage.md stage 5")


def _parser():
    """The real ``argparse.ArgumentParser`` of ``PutTogetherEmbeddings`` (built, never parsed)."""
    return PutTogetherEmbeddings().parser


@pytest.mark.smoke
class TestUsageCombineSource:
    """REQ-EMBVER-BDRABCZUK-FA4D24096DF2."""

    def test_combine_source_is_versioned_region_embeddings(self):
        for tokens in _combine_examples():
            positionals = positional_args(tokens, _parser())
            assert [norm(p) for p in positionals] == [norm(EXPECTED_SOURCE)], (
                f"docs/usage.md stage-5 {COMMAND} example must pass exactly one positional "
                f"embeddings_source, {EXPECTED_SOURCE}; found positionals {positionals} in {tokens}"
            )


@pytest.mark.smoke
class TestUsageCombineFlags:
    """REQ-EMBVER-BDRABCZUK-465E3A7A5E89."""

    def test_combine_flags_exist_in_real_parser(self):
        real = option_strings(_parser())
        documented = {tok.split("=", 1)[0] for tokens in _combine_examples() for tok in tokens if tok.startswith("-")}
        bogus = sorted(documented - real)
        assert not bogus, (
            f"docs/usage.md stage-5 {COMMAND} example uses flags that PutTogetherEmbeddings' "
            f"argparse does not declare: {bogus} (declared: {sorted(real)})"
        )
