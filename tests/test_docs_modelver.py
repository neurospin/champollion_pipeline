#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Documentation guards for the model-version work (epic MODELVER).

``generate_embeddings`` now takes ``--model-version`` (the Hugging Face model-set
subfolder), keeps ``--masks-version`` as a deprecated alias, defaults ``--masks``
to the model version, refuses a ``--masks``/model-version mismatch unless
``--allow-version-mismatch`` is given, checks a local model set against its
``model_set.json`` marker and writes ``provenance.json`` next to the region
embeddings. Section "5. Generate Embeddings" of ``README.md`` must say so.

Every check reads the facts from the live code (the ``GenerateEmbeddings``
parser, ``MODEL_SET_FILE_NAME``, ``PROVENANCE_FILE_NAME``, ``ModelSet`` fields,
``DEFAULT_MASKS_VERSION`` and the provenance dict ``generate_embeddings``
builds), never from hard-coded prose.

Requirements pinned here (one test per requirement):

* REQ-MODELVER-BDRABCZUK-B431CAC3145D - options table rows for the version options.
* REQ-MODELVER-BDRABCZUK-36B042A2A3EC - ``--masks-version`` row: deprecated alias of ``--model-version``.
* REQ-MODELVER-BDRABCZUK-37B69C8D1B88 - ``--masks`` row: defaults to the model version, else canonical_25.
* REQ-MODELVER-BDRABCZUK-3389BCA1D219 - ``model_set.json`` subsection.
* REQ-MODELVER-BDRABCZUK-CE3FBFAB1A03 - ``provenance.json`` subsection names every key.

No network and no BrainVISA: the parser is only built (and, for the provenance
keys, parsed with a Hugging Face repo id that is never fetched).
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from champollion_pipeline import generate_embeddings as ge
from champollion_pipeline.generate_embeddings import (
    DEFAULT_MASKS_VERSION,
    MODEL_SET_FILE_NAME,
    PROVENANCE_FILE_NAME,
    GenerateEmbeddings,
    ModelSet,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = PROJECT_ROOT / "README.md"

VERSION_OPTIONS = ("--model-version", "--masks-version", "--allow-version-mismatch")


def _read(path: Path) -> str:
    assert path.is_file(), f"expected file to exist: {path}"
    return path.read_text(encoding="utf-8")


def _embeddings_section() -> str:
    """Text of README section '## 5. Generate Embeddings', up to the next '##' heading."""
    m = re.search(r"^##\s+5\.\s+Generate Embeddings\s*$(.*?)(?=^##\s|\Z)", _read(README), re.MULTILINE | re.DOTALL)
    assert m, "README.md has no '## 5. Generate Embeddings' section"
    return m.group(1)


def _options_rows() -> dict[str, str]:
    """Map the first-cell ``--flag`` of each row of section 5's 'Option' tables to the whole row."""
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
    assert rows, "section 5 has no '| Option |' table rows (scope helper broken?)"
    return rows


def _subsection(name: str) -> str:
    """Text of the first '###' subsection of section 5 whose heading contains ``name``."""
    m = re.search(
        rf"^###\s+[^\n]*{re.escape(name)}[^\n]*$(.*?)(?=^###?\s|\Z)", _embeddings_section(), re.MULTILINE | re.DOTALL
    )
    assert m, f"section 5 of README.md has no '###' subsection whose heading names {name}"
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
        assert option in _options_rows(), f"README section 5 options table has no row for {option}"


@pytest.mark.smoke
class TestReadmeMasksVersionRow:
    """REQ-MODELVER-BDRABCZUK-36B042A2A3EC."""

    def test_row_states_deprecated_alias_of_model_version(self):
        row = _options_rows().get("--masks-version")
        assert row, "README section 5 options table has no --masks-version row"
        assert re.search(r"\bdeprecated\b", row, re.IGNORECASE), f"--masks-version row does not say deprecated: {row}"
        assert re.search(r"\balias\b", row, re.IGNORECASE), f"--masks-version row does not say alias: {row}"
        assert "--model-version" in row, f"--masks-version row does not name --model-version: {row}"


@pytest.mark.smoke
class TestReadmeMasksRow:
    """REQ-MODELVER-BDRABCZUK-37B69C8D1B88."""

    def test_row_states_model_version_default(self):
        row = _options_rows().get("--masks")
        assert row, "README section 5 options table has no --masks row"
        assert "--model-version" in row, f"--masks row does not name --model-version as its default: {row}"
        assert DEFAULT_MASKS_VERSION in row, f"--masks row does not name the {DEFAULT_MASKS_VERSION} fallback: {row}"


@pytest.mark.smoke
class TestReadmeModelSetSubsection:
    """REQ-MODELVER-BDRABCZUK-3389BCA1D219."""

    def test_subsection_describes_marker(self):
        body = _subsection(MODEL_SET_FILE_NAME)
        required_keys = [f.name for f in dataclasses.fields(ModelSet) if f.default is dataclasses.MISSING]
        missing = [k for k in required_keys if not _names_word(body, k)]
        assert not missing, f"{MODEL_SET_FILE_NAME} subsection does not name the key(s) {missing}"
        assert "--allow-version-mismatch" in _parser_options()
        assert "--allow-version-mismatch" in body, (
            f"{MODEL_SET_FILE_NAME} subsection does not name --allow-version-mismatch"
        )
        assert re.search(r"\bwarning\b", body, re.IGNORECASE), (
            f"{MODEL_SET_FILE_NAME} subsection does not describe the warning for a local models_path without the marker"
        )
        assert "Hugging Face" in body, (
            f"{MODEL_SET_FILE_NAME} subsection does not say a Hugging Face download writes it"
        )


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
