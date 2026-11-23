#!/usr/bin/env python3
"""Documentation guards for the stage-4 ``champollion-embeddings`` example in ``docs/usage.md``.

Run by hand (``verify_ignore`` skips ``tests/test_docs_*.py``):
``pixi run pytest tests/test_docs_usage_embeddings_example.py``.

``GenerateEmbeddings`` takes two positionals, ``models_path`` and ``datasets_root``. The
``docs/usage.md`` stage-4 example must show that interface, not flags the parser rejects, and
must say where the models cache lives and how to move it.

Requirements pinned here (one test per requirement):

* REQ-DATASETS-BDRABCZUK-D689A0FC6384 - each example passes exactly the positionals
  ``neurospin/Champollion_V1`` and ``/data/myproject/``.
* REQ-DATASETS-BDRABCZUK-82DA8AD5FF6A - the options of the examples are declared by the real parser.
* REQ-DATASETS-BDRABCZUK-DA337CA59AD5 - the stage-4 section states the default cache path,
  ``--models-cache`` and ``dataset.models_cache``.

``docs/usage.md`` is read as plain text. ``GenerateEmbeddings`` is only instantiated to inspect its
``argparse`` parser: no argument parsing, no filesystem access, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from tests.docs_test_helpers import md_section, option_strings, positional_args, read_text, required_examples

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USAGE = PROJECT_ROOT / "docs" / "usage.md"

COMMAND = "champollion-embeddings"
EXPECTED_POSITIONALS = ["neurospin/Champollion_V1", "/data/myproject"]
CACHE_PATH = "{datasets_root}/derivatives/champollion_V1/models_cache/"


def _stage4_section() -> str:
    """Body of the ``## Stage 4 ...`` section of ``docs/usage.md``."""
    return md_section(read_text(USAGE), r"Stage 4\b[^\n]*")


def _examples() -> list[list[str]]:
    """Argument tokens (after the command name) of every ``pixi run champollion-embeddings`` in usage.md."""
    return required_examples(read_text(USAGE), COMMAND, "docs/usage.md")


def _parser():
    """The real ``argparse.ArgumentParser`` of ``GenerateEmbeddings`` (built, never parsed)."""
    return GenerateEmbeddings().parser


@pytest.mark.smoke
class TestUsageEmbeddingsPositionals:
    """REQ-DATASETS-BDRABCZUK-D689A0FC6384."""

    def test_examples_pass_models_path_and_datasets_root(self):
        for tokens in _examples():
            positionals = [p.rstrip("/") for p in positional_args(tokens, _parser())]
            assert positionals == EXPECTED_POSITIONALS, (
                f"docs/usage.md {COMMAND} example must pass exactly {EXPECTED_POSITIONALS}; "
                f"found positionals {positionals} in {tokens}"
            )


@pytest.mark.smoke
class TestUsageEmbeddingsFlags:
    """REQ-DATASETS-BDRABCZUK-82DA8AD5FF6A."""

    def test_flags_exist_in_real_parser(self):
        documented = {tok.split("=", 1)[0] for tokens in _examples() for tok in tokens if tok.startswith("-")}
        bogus = sorted(documented - option_strings(_parser()))
        assert not bogus, f"docs/usage.md {COMMAND} examples use options the real parser rejects: {bogus}"


@pytest.mark.smoke
class TestUsageModelsCacheNote:
    """REQ-DATASETS-BDRABCZUK-DA337CA59AD5."""

    @pytest.mark.parametrize("needle", [CACHE_PATH, "--models-cache", "dataset.models_cache"])
    def test_stage4_section_states(self, needle):
        assert needle in _stage4_section(), f"docs/usage.md stage-4 section must contain {needle!r}"
