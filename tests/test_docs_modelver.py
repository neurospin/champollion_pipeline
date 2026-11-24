#!/usr/bin/env python3
"""Documentation guards for the model-version work (epic MODELVER).

``generate_embeddings`` now takes ``--model-version`` (the Hugging Face model-set
subfolder), keeps ``--masks-version`` as a deprecated alias, defaults ``--masks``
to the model version, refuses a ``--masks``/model-version mismatch unless
``--allow-version-mismatch`` is given, checks a local model set against its
``model_set.json`` marker and writes ``provenance.json`` next to the region
embeddings. The Stage 4 section of ``docs/usage.md`` must say so (moved from README.md section
"5. Generate Embeddings" by TASK-DOCS-BDRABCZUK-84BE8AB6FC73).

Every check reads the facts from the live code (the ``GenerateEmbeddings``
parser, ``PROVENANCE_FILE_NAME``
and the provenance dict ``generate_embeddings`` builds), never from hard-coded prose.

Requirements pinned here (one test per requirement; the ``--masks`` row, REQ-MODELVER-BDRABCZUK-6A7EC2D5389F,
which supersedes 37B69C8D1B88, is pinned in tests/test_masks_default_corrected.py):

* REQ-MODELVER-BDRABCZUK-B431CAC3145D - options table rows for the version options.
* REQ-MODELVER-BDRABCZUK-36B042A2A3EC - ``--masks-version`` row: deprecated alias of ``--model-version``.
* ``model_set.json`` subsection: moved to tests/test_usage_modelver_per_region.py with its superseding
  Requirement (TASK-MODELVER-BDRABCZUK-6E24A08667F9), since tests/test_docs_*.py are outside the verify selection.
* REQ-MODELVER-BDRABCZUK-CE3FBFAB1A03 - ``provenance.json`` subsection names every key.

No network and no BrainVISA: the parser is only built (and, for the provenance
keys, parsed with a Hugging Face repo id that is never fetched).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from champollion_pipeline import generate_embeddings as ge
from champollion_pipeline.generate_embeddings import (
    PROVENANCE_FILE_NAME,
    GenerateEmbeddings,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USAGE = PROJECT_ROOT / "docs" / "usage.md"

VERSION_OPTIONS = ("--model-version", "--masks-version", "--allow-version-mismatch")


def _read(path: Path) -> str:
    assert path.is_file(), f"expected file to exist: {path}"
    return path.read_text(encoding="utf-8")


def _embeddings_section() -> str:
    """Text of the docs/usage.md section '## Stage 4 ...', up to the next '##' heading."""
    m = re.search(r"^##\s+Stage 4\b[^\n]*$(.*?)(?=^##\s|\Z)", _read(USAGE), re.MULTILINE | re.DOTALL)
    assert m, "docs/usage.md has no '## Stage 4' section"
    return m.group(1)


def _options_rows() -> dict[str, str]:
    """Map the first-cell ``--flag`` of each row of the Stage 4 section's 'Option' tables to the whole row."""
    rows: dict[str, str] = {}
    in_table = False
    for line in _embeddings_section().splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            in_table = False
            continue
        if re.match(r"^\|\s*Option\s*\|", stripped):
            in_table = True
            continue
        if in_table:
            first = stripped.strip("|").split("|", 1)[0]
            flag = re.search(r"--[A-Za-z][A-Za-z0-9_-]*", first)
            if flag:
                rows[flag.group(0)] = stripped
    assert rows, "Stage 4 section has no '| Option |' table rows (scope helper broken?)"
    return rows


def _subsection(name: str) -> str:
    """Text of the first '###' subsection of the Stage 4 section whose heading contains ``name``."""
    m = re.search(
        rf"^###\s+[^\n]*{re.escape(name)}[^\n]*$(.*?)(?=^###?\s|\Z)", _embeddings_section(), re.MULTILINE | re.DOTALL
    )
    assert m, f"Stage 4 section of docs/usage.md has no '###' subsection whose heading names {name}"
    return m.group(1)


def _names_word(text: str, word: str) -> bool:
    """``word`` occurs as a standalone token (not inside ``--masks`` or ``model_version_x``)."""
    return re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", text) is not None


def _parser_options() -> set[str]:
    return {opt for action in GenerateEmbeddings().parser._actions for opt in action.option_strings}


@pytest.mark.smoke
class TestReadmeVersionOptionRows:
    """REQ-MODELVER-BDRABCZUK-B431CAC3145D."""

    @pytest.mark.parametrize("option", VERSION_OPTIONS)
    def test_version_option_has_row(self, option):
        assert option in _parser_options(), f"{option} is no longer a generate_embeddings option"
        assert option in _options_rows(), f"docs/usage.md Stage 4 options table has no row for {option}"


@pytest.mark.smoke
class TestReadmeMasksVersionRow:
    """REQ-MODELVER-BDRABCZUK-36B042A2A3EC."""

    def test_row_states_deprecated_alias_of_model_version(self):
        row = _options_rows().get("--masks-version")
        assert row, "docs/usage.md Stage 4 options table has no --masks-version row"
        assert re.search(r"\bdeprecated\b", row, re.IGNORECASE), f"--masks-version row does not say deprecated: {row}"
        assert re.search(r"\balias\b", row, re.IGNORECASE), f"--masks-version row does not say alias: {row}"
        assert "--model-version" in row, f"--masks-version row does not name --model-version: {row}"


@pytest.mark.smoke
class TestReadmeProvenanceSubsection:
    """REQ-MODELVER-BDRABCZUK-CE3FBFAB1A03."""

    def test_subsection_names_every_provenance_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ge, "read_pipeline_commit", lambda: None)
        builder = GenerateEmbeddings()
        builder.parse_args(["neurospin/Champollion_V1", str(tmp_path)])
        keys = list(builder._build_provenance(str(tmp_path)))
        assert keys, "generate_embeddings builds an empty provenance dict (helper broken?)"
        body = _subsection(PROVENANCE_FILE_NAME)
        missing = [k for k in keys if not _names_word(body, k)]
        assert not missing, f"{PROVENANCE_FILE_NAME} subsection does not name the key(s) {missing}"
