#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import re
import shlex
from pathlib import Path

import pytest

from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USAGE = PROJECT_ROOT / "docs" / "usage.md"

COMMAND = "champollion-combine"
EXPECTED_SOURCE = "/data/myproject/derivatives/champollion_V1/canonical_25/region_embeddings/"


def _read(path: Path) -> str:
    assert path.is_file(), f"expected file to exist: {path}"
    return path.read_text(encoding="utf-8")


def _stage5_section() -> str:
    """Body of the ``## Stage 5 ...`` section of ``docs/usage.md``."""
    text = _read(USAGE)
    m = re.search(r"^##\s+Stage 5\b[^\n]*$(.*?)(?=^##\s|\Z)", text, re.MULTILINE | re.DOTALL)
    assert m, "docs/usage.md has no '## Stage 5' section"
    return m.group(1)


def _strip_comment(line: str) -> str:
    if line.lstrip().startswith("#"):
        return ""
    return re.sub(r"\s+#.*$", "", line)


def _combine_examples() -> list[list[str]]:
    """Argument tokens (after the command name) of every ``pixi run champollion-combine`` in stage 5."""
    found: list[list[str]] = []
    for block in re.findall(r"```bash\n(.*?)```", _stage5_section(), re.DOTALL):
        lines = block.splitlines()
        i = 0
        while i < len(lines):
            line = _strip_comment(lines[i]).rstrip()
            if re.search(rf"\bpixi run {re.escape(COMMAND)}(?:\s|\\|$)", line):
                parts = [line]
                while parts[-1].endswith("\\") and i + 1 < len(lines):
                    i += 1
                    parts.append(_strip_comment(lines[i]).rstrip())
                joined = " ".join(p[:-1] if p.endswith("\\") else p for p in parts)
                tokens = shlex.split(joined)
                found.append(tokens[tokens.index(COMMAND) + 1 :])
            i += 1
    assert found, f"docs/usage.md stage 5: no 'pixi run {COMMAND}' example found (test parser sanity check)"
    return found


def _parser():
    """The real ``argparse.ArgumentParser`` of ``PutTogetherEmbeddings`` (built, never parsed)."""
    return PutTogetherEmbeddings().parser


def _real_option_strings() -> set[str]:
    """Every option string declared on the real parser."""
    return {opt for action in _parser()._actions for opt in action.option_strings}


def _value_taking_options() -> set[str]:
    """Option strings of the real parser that consume a following value."""
    return {
        opt
        for action in _parser()._actions
        if action.option_strings and action.nargs != 0
        for opt in action.option_strings
    }


def _positionals(tokens: list[str]) -> list[str]:
    """Tokens that are neither a ``--flag`` nor the value of a value-taking flag."""
    takes_value = _value_taking_options()
    positionals: list[str] = []
    skip_next = False
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue
        if tok.startswith("-"):
            name = tok.split("=", 1)[0]
            # An unknown flag is assumed to take a value, as the usage.md examples write ``--flag value``.
            skip_next = "=" not in tok and (name in takes_value or name not in _real_option_strings())
            continue
        positionals.append(tok)
    return positionals


def _norm(path: str) -> str:
    return path.rstrip("/")


@pytest.mark.smoke
class TestUsageCombineSource:
    """REQ-EMBVER-BDRABCZUK-FA4D24096DF2."""

    def test_combine_source_is_versioned_region_embeddings(self):
        for tokens in _combine_examples():
            positionals = _positionals(tokens)
            assert [_norm(p) for p in positionals] == [_norm(EXPECTED_SOURCE)], (
                f"docs/usage.md stage-5 {COMMAND} example must pass exactly one positional "
                f"embeddings_source, {EXPECTED_SOURCE}; found positionals {positionals} in {tokens}"
            )


@pytest.mark.smoke
class TestUsageCombineFlags:
    """REQ-EMBVER-BDRABCZUK-465E3A7A5E89."""

    def test_combine_flags_exist_in_real_parser(self):
        real = _real_option_strings()
        documented = {tok.split("=", 1)[0] for tokens in _combine_examples() for tok in tokens if tok.startswith("-")}
        bogus = sorted(documented - real)
        assert not bogus, (
            f"docs/usage.md stage-5 {COMMAND} example uses flags that PutTogetherEmbeddings' "
            f"argparse does not declare: {bogus} (declared: {sorted(real)})"
        )
