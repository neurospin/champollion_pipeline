#!/usr/bin/env python3
"""Documentation guards for the per-region Mask versions of a local model set (TASK-MODELVER-BDRABCZUK-6E24A08667F9).

Stage 4 (``champollion-embeddings``) checks each region model name of a local models_path against the Mask
version of its crops: the per-region Mask version resolver reads ``model_set.json``, then ``PROVENANCE.csv``
(columns ``region`` and ``mask_version_trained_on``), then the folder name (TASK-MODELVER-BDRABCZUK-5F2E9945DE10).
A mismatch exits with status 2 unless ``--allow-version-mismatch`` is given (then one WARNING log record per
region model name); unresolved names give one WARNING log record; a local folder without ``model_set.json`` is no
longer "unchecked" (TASK-MODELVER-BDRABCZUK-86405DAD28CE). ``champollion-pipeline`` without ``--masks-version`` uses
the Mask version of each region model name of a local model set (REQ-MODELVER-BDRABCZUK-1621C738EAB8).

The file names, column names and options come from the live code, never from hard-coded prose. Text only:
no filesystem writes, no network.

Requirements pinned here, one test each:

* REQ-MODELVER-BDRABCZUK-8D0812FB3A27 successor - ``model_set.json`` subsection names the marker keys.
* REQ-MODELVER-BDRABCZUK-4A40F4148A0E - no "cannot be checked" statement for a folder without the marker.
* REQ-MODELVER-BDRABCZUK-E0204B0C3647 - the three Mask version sources, in order of precedence.
* REQ-MODELVER-BDRABCZUK-A8865122B1BC - the two ``PROVENANCE.csv`` columns read.
* REQ-MODELVER-BDRABCZUK-19B0B9791C37 - one WARNING log record on unresolved region model names.
* REQ-MODELVER-BDRABCZUK-C6C84AB0518C - exit 2 on two ``PROVENANCE.csv`` Mask versions for one name.
* REQ-MODELVER-BDRABCZUK-7676D0E7DA9B - ``--allow-version-mismatch`` row: per-region exit 2.
* REQ-MODELVER-BDRABCZUK-95CEB2DB7576 - ``--allow-version-mismatch`` row: one WARNING per region model name.
* REQ-MODELVER-BDRABCZUK-7C4F485F1A06 - README ``--masks-version`` row: per-region Mask versions.
* REQ-MODELVER-BDRABCZUK-6379B2B63170 - docs/usage.md champollion-pipeline ``--masks-version`` row: same.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from champollion_pipeline.generate_embeddings import MODEL_SET_FILE_NAME, GenerateEmbeddings, ModelSet
from champollion_pipeline.models.version_options import (
    PROVENANCE_FILE_NAME as PROVENANCE_CSV,
)
from champollion_pipeline.models.version_options import (
    PROVENANCE_REGION_COLUMN,
    PROVENANCE_VERSION_COLUMN,
)
from tests.docs_test_helpers import md_section, read_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = PROJECT_ROOT / "README.md"
USAGE = PROJECT_ROOT / "docs" / "usage.md"

ALLOW = "--allow-version-mismatch"
MASKS_VERSION = "--masks-version"


def _names_word(text: str, word: str) -> bool:
    """``word`` occurs as a standalone token (not inside ``--masks`` or ``model_version_x``)."""
    return re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", text) is not None


def _stage4_section() -> str:
    return md_section(read_text(USAGE), r"Stage 4\b[^\n]*")


def _model_set_subsection() -> str:
    """Body of the Stage 4 '###' subsection whose heading names model_set.json, up to the next heading."""
    m = re.search(
        rf"^###\s+[^\n]*{re.escape(MODEL_SET_FILE_NAME)}[^\n]*$(.*?)(?=^###?\s|\Z)",
        _stage4_section(),
        re.MULTILINE | re.DOTALL,
    )
    assert m, f"Stage 4 section of docs/usage.md has no '###' subsection whose heading names {MODEL_SET_FILE_NAME}"
    return m.group(1)


def _paragraphs(text: str) -> list[str]:
    """Paragraphs and list items of ``text``: blocks split at blank lines and at '- ' bullet starts."""
    blocks: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        blocks.extend(part for part in re.split(r"\n(?=\s*[-*]\s)", block) if part.strip())
    return blocks


def _in_precedence_order(paragraph: str) -> bool:
    """model_set.json, then PROVENANCE.csv, then 'folder name' appear in that order in ``paragraph``."""
    first = paragraph.index(MODEL_SET_FILE_NAME)
    second = paragraph.index(PROVENANCE_CSV)
    return first < second and re.search(r"\bfolder name\b", paragraph[second:]) is not None


def _option_row(text: str, option: str, where: str) -> str:
    """The table row of ``text`` whose first cell names ``option``."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            first = stripped.strip("|").split("|", 1)[0]
            if re.search(rf"(?<![\w-]){re.escape(option)}(?![\w-])", first):
                return stripped
    pytest.fail(f"{where} has no option table row for {option}")


def _stage4_options() -> set[str]:
    return {opt for action in GenerateEmbeddings().parser._actions for opt in action.option_strings}


@pytest.mark.smoke
class TestModelSetSubsection:
    """The model_set.json subsection of the Stage 4 section of docs/usage.md."""

    def test_subsection_describes_marker(self):
        """Successor of REQ-MODELVER-BDRABCZUK-8D0812FB3A27 (moved from tests/test_docs_modelver.py)."""
        body = _model_set_subsection()
        required_keys = [f.name for f in dataclasses.fields(ModelSet) if f.default is dataclasses.MISSING]
        missing = [k for k in required_keys if not _names_word(body, k)]
        assert not missing, f"{MODEL_SET_FILE_NAME} subsection does not name the key(s) {missing}"
        assert ALLOW in _stage4_options()
        assert ALLOW in body, f"{MODEL_SET_FILE_NAME} subsection does not name {ALLOW}"
        assert "Hugging Face" in body, (
            f"{MODEL_SET_FILE_NAME} subsection does not say a Hugging Face download writes it"
        )

    def test_subsection_drops_unchecked_warning(self):
        """REQ-MODELVER-BDRABCZUK-4A40F4148A0E."""
        body = _model_set_subsection()
        stale = re.findall(r"[^.\n]*\b(?:cannot be checked|unchecked)\b[^.\n]*", body, re.IGNORECASE)
        assert not stale, (
            f"{MODEL_SET_FILE_NAME} subsection still says a local models_path without the marker "
            f"cannot be checked: {stale}"
        )

    def test_subsection_names_sources_in_precedence_order(self):
        """REQ-MODELVER-BDRABCZUK-E0204B0C3647."""
        candidates = [
            p
            for p in _paragraphs(_model_set_subsection())
            if MODEL_SET_FILE_NAME in p and PROVENANCE_CSV in p and re.search(r"\bfolder name\b", p)
        ]
        assert candidates, (
            f"{MODEL_SET_FILE_NAME} subsection has no paragraph naming {MODEL_SET_FILE_NAME}, {PROVENANCE_CSV} "
            "and the folder name as the Mask version sources"
        )
        ordered = [p for p in candidates if _in_precedence_order(p)]
        assert ordered, (
            f"no paragraph names {MODEL_SET_FILE_NAME}, then {PROVENANCE_CSV}, then the folder name: {candidates}"
        )

    def test_subsection_names_provenance_columns(self):
        """REQ-MODELVER-BDRABCZUK-A8865122B1BC."""
        body = _model_set_subsection()
        assert PROVENANCE_CSV in body, f"{MODEL_SET_FILE_NAME} subsection does not name {PROVENANCE_CSV}"
        for column in (PROVENANCE_REGION_COLUMN, PROVENANCE_VERSION_COLUMN):
            assert f"`{column}`" in body, (
                f"{MODEL_SET_FILE_NAME} subsection does not name the {PROVENANCE_CSV} column `{column}`"
            )

    def test_subsection_states_unresolved_warning(self):
        """REQ-MODELVER-BDRABCZUK-19B0B9791C37."""
        hits = [
            p
            for p in _paragraphs(_model_set_subsection())
            if re.search(r"\bunresolved\b", p, re.IGNORECASE) and re.search(r"\bone\s+`?WARNING\b", p)
        ]
        assert hits, (
            f"{MODEL_SET_FILE_NAME} subsection does not state that unresolved region model names give one "
            "WARNING log record and the run goes on"
        )

    def test_subsection_states_conflict_exit_2(self):
        """REQ-MODELVER-BDRABCZUK-C6C84AB0518C."""
        hits = [
            p
            for p in _paragraphs(_model_set_subsection())
            if PROVENANCE_CSV in p
            and re.search(r"\bstatus 2\b", p)
            and re.search(r"\b(?:two|different|conflict\w*)\b", p, re.IGNORECASE)
        ]
        assert hits, (
            f"{MODEL_SET_FILE_NAME} subsection does not state the exit with status 2 when {PROVENANCE_CSV} names "
            "two different Mask versions for one region model name"
        )


@pytest.mark.smoke
class TestAllowVersionMismatchRow:
    """The --allow-version-mismatch row of the Stage 4 option table of docs/usage.md."""

    def _row(self) -> str:
        assert ALLOW in _stage4_options(), f"{ALLOW} is no longer a generate_embeddings option"
        return _option_row(_stage4_section(), ALLOW, "Stage 4 section of docs/usage.md")

    def test_row_states_per_region_exit_2(self):
        """REQ-MODELVER-BDRABCZUK-7676D0E7DA9B."""
        row = self._row()
        assert re.search(r"\bstatus 2\b", row), f"{ALLOW} row does not state the exit with status 2: {row}"
        assert re.search(r"\bregion model name\b", row), (
            f"{ALLOW} row does not state the mismatch per region model name: {row}"
        )

    def test_row_states_one_warning_per_region(self):
        """REQ-MODELVER-BDRABCZUK-95CEB2DB7576."""
        row = self._row()
        assert re.search(r"\bone\s+`?WARNING\b", row), f"{ALLOW} row does not state one WARNING log record: {row}"
        assert re.search(r"\b(?:per|each)\s+(?:mismatching\s+)?region model name\b", row), (
            f"{ALLOW} row does not state one WARNING per mismatching region model name: {row}"
        )


def _states_per_region_versions(row: str) -> bool:
    return bool(re.search(r"\bregion model name\b", row) and re.search(r"\blocal model set\b", row, re.IGNORECASE))


@pytest.mark.smoke
class TestPipelineMasksVersionRows:
    """The champollion-pipeline --masks-version rows of README.md and docs/usage.md."""

    def test_readme_row_states_per_region_versions(self):
        """REQ-MODELVER-BDRABCZUK-7C4F485F1A06."""
        row = _option_row(read_text(README), MASKS_VERSION, "README.md")
        assert _states_per_region_versions(row), (
            f"README.md {MASKS_VERSION} row does not state that a local model set without {MASKS_VERSION} uses "
            f"the Mask version of each region model name: {row}"
        )

    def test_usage_row_states_per_region_versions(self):
        """REQ-MODELVER-BDRABCZUK-6379B2B63170."""
        section = md_section(read_text(USAGE), r"champollion-pipeline reference")
        row = _option_row(section, MASKS_VERSION, "champollion-pipeline reference of docs/usage.md")
        assert _states_per_region_versions(row), (
            f"docs/usage.md champollion-pipeline {MASKS_VERSION} row does not state that a local model set "
            f"without {MASKS_VERSION} uses the Mask version of each region model name: {row}"
        )
