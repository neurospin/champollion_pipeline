#!/usr/bin/env python3
"""Documentation guards for the skeleton seed (skel_seed) of crop generation (epic SKELSEED, task 9/9).

Behaviour documented (read from the code, WIP a367ad5):

* ``run_cortical_tiles --skel-seed N`` forwards ``--skel_seed N`` to cortical_tiles'
  ``generate_sulcal_regions.py``, which uses N instead of the ``skel_seed`` key of
  ``{output}/pipeline_loop_2mm.json``; without the option nothing is forwarded and the
  JSON value applies (cortical_tiles ``DEFAULT_SKEL_SEED`` = 42 when the key is absent).
* The shipped ``pipeline_loop_2mm.json`` template sets ``"skel_seed": 42``; run_cortical_tiles
  copies it to ``{output}/`` only when no copy exists there.
* ``main.py`` ``DatasetConfig.skel_seed`` (YAML key ``dataset.skel_seed``, default ``None``) is
  forwarded as ``--skel-seed`` only when it is not ``None``.
* A fixed seed makes the 2mm re-skeletonization, and so every crop, reproducible; crops built
  before the seed existed came from clock-seeded skeletons, so they and their embeddings differ.

The README.md section 3 text moved to the Stage 2 section of docs/usage.md (TASK-DOCS-BDRABCZUK-84BE8AB6FC73);
the Requirements below name that place since.

Requirements pinned here (one test per requirement):

* REQ-SKELSEED-BDRABCZUK-E48CDFD68B58 - the section 3 option table has one ``--skel-seed`` row.
* REQ-SKELSEED-BDRABCZUK-A0431BE9EAF0 - that row names the pipeline config key ``dataset.skel_seed``.
* REQ-SKELSEED-BDRABCZUK-29921F2361A8 - that row states that, without the option, the ``skel_seed``
  key of ``{output}/pipeline_loop_2mm.json`` sets the seed.
* REQ-SKELSEED-BDRABCZUK-AF52CD789B59 - that row states that a null ``dataset.skel_seed`` leaves the
  ``pipeline_loop_2mm.json`` value in effect.
* REQ-SKELSEED-BDRABCZUK-2F4E2A6409D3 - the "Skeleton seed" subsection states 42 as the template value.
* REQ-SKELSEED-BDRABCZUK-73155499C72A - that subsection states that the same input and seed give
  identical crops.
* REQ-SKELSEED-BDRABCZUK-2C0A2394DE52 - that subsection states that crops and embeddings of datasets
  built before the skel_seed key differ from those of a seeded run.

docs/usage.md is read as plain text: no package import, no Sphinx build, no network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

USAGE = PROJECT_ROOT / "docs" / "usage.md"

CROPS_SECTION_HEADING = r"Stage 2\b[^\n]*"
OPTION_NAME = "--skel-seed"
JSON_NAME = "pipeline_loop_2mm.json"
OUTPUT_JSON = "{output}/pipeline_loop_2mm.json"
CONFIG_KEY_RE = re.compile(r"(?<![\w.])dataset\.skel_seed(?!\w)")
JSON_KEY_RE = re.compile(r"(?<![\w.])skel_seed(?!\w)")


def _read(path: Path) -> str:
    assert path.is_file(), f"expected file to exist: {path}"
    return path.read_text(encoding="utf-8")


def _section(text: str, heading_regex: str) -> str:
    """Body of the ``## <heading>`` section, up to the next level-2 heading."""
    m = re.search(rf"^##\s+{heading_regex}\s*$(.*?)(?=^##\s|\Z)", text, re.MULTILINE | re.DOTALL)
    assert m, f"no '## {heading_regex}' section found in docs/usage.md"
    return m.group(1)


def _crops_section() -> str:
    return _section(_read(USAGE), CROPS_SECTION_HEADING)


def _cells(line: str) -> list[str]:
    """Cells of one Markdown table line, stripped of surrounding whitespace."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _option_table_rows(section: str) -> list[list[str]]:
    """Body rows of the table whose header row starts with ``Option``."""
    rows: list[list[str]] = []
    in_table = False
    for line in section.splitlines():
        if not line.lstrip().startswith("|"):
            in_table = False
            continue
        cells = _cells(line)
        if not in_table:
            in_table = cells[0] == "Option"
            continue
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
            continue  # separator row
        rows.append(cells)
    return rows


def _skel_seed_rows() -> tuple[list[list[str]], list[list[str]]]:
    """(all option rows, ``--skel-seed`` rows) of the section 3 option table."""
    rows = _option_table_rows(_crops_section())
    assert rows, "docs/usage.md section 'Stage 2' has no table whose header row starts with 'Option'"
    matches = [r for r in rows if r[0].strip("`").split()[0:1] == [OPTION_NAME]]
    return rows, matches


def _skel_seed_row_description() -> str:
    """Description cell of the ``--skel-seed`` row of the section 3 option table."""
    rows, matches = _skel_seed_rows()
    assert len(matches) == 1, (
        f"docs/usage.md Stage 2 option table must contain exactly one '{OPTION_NAME}' row; "
        f"option cells found: {[r[0] for r in rows]}"
    )
    return " | ".join(matches[0][1:])


def _skeleton_seed_subsection() -> str:
    """Text of the ``### Skeleton seed`` subsection of the Stage 2 section, up to the next heading."""
    section = _crops_section()
    m = re.search(r"^###\s+Skeleton seed\s*$(.*?)(?=^#{1,3}\s|\Z)", section, re.MULTILINE | re.DOTALL)
    assert m, "docs/usage.md section 'Stage 2' has no '### Skeleton seed' subsection"
    return m.group(1)


@pytest.mark.smoke
class TestReadmeSkelSeedOptionRow:
    """REQ-SKELSEED-BDRABCZUK-E48CDFD68B58."""

    def test_option_table_has_skel_seed_row(self):
        rows, matches = _skel_seed_rows()
        assert len(matches) == 1, (
            f"docs/usage.md Stage 2 option table must contain exactly one '{OPTION_NAME}' row; "
            f"option cells found: {[r[0] for r in rows]}"
        )


@pytest.mark.smoke
class TestReadmeSkelSeedConfigKey:
    """REQ-SKELSEED-BDRABCZUK-A0431BE9EAF0."""

    def test_option_row_names_config_key(self):
        description = _skel_seed_row_description()
        assert CONFIG_KEY_RE.search(description), (
            f"docs/usage.md Stage 2 '{OPTION_NAME}' row must name the pipeline config key dataset.skel_seed; "
            f"description: {description!r}"
        )


@pytest.mark.smoke
class TestReadmeSkelSeedDefaultSource:
    """REQ-SKELSEED-BDRABCZUK-29921F2361A8."""

    def test_option_row_names_json_default(self):
        description = _skel_seed_row_description()
        assert (
            OUTPUT_JSON in description
            and JSON_KEY_RE.search(description)
            and re.search(r"\bwithout\b", description, re.IGNORECASE)
        ), (
            f"docs/usage.md Stage 2 '{OPTION_NAME}' row must state that, without the option, the skel_seed key of "
            f"{OUTPUT_JSON} sets the skeleton seed; description: {description!r}"
        )


@pytest.mark.smoke
class TestReadmeSkelSeedNullConfig:
    """REQ-SKELSEED-BDRABCZUK-AF52CD789B59."""

    def test_option_row_states_null_config(self):
        description = _skel_seed_row_description()
        assert (
            re.search(r"\bnull\b", description, re.IGNORECASE)
            and CONFIG_KEY_RE.search(description)
            and JSON_NAME in description
        ), (
            f"docs/usage.md Stage 2 '{OPTION_NAME}' row must state that a null dataset.skel_seed leaves the "
            f"skel_seed value of {JSON_NAME} in effect; description: {description!r}"
        )


@pytest.mark.smoke
class TestReadmeSkelSeedTemplateValue:
    """REQ-SKELSEED-BDRABCZUK-2F4E2A6409D3."""

    def test_subsection_states_template_seed_42(self):
        prose = _skeleton_seed_subsection()
        assert (
            re.search(r"(?<![\w.])42(?!\w|\.\d)", prose)
            and JSON_NAME in prose
            and re.search(r"\btemplate\b", prose, re.IGNORECASE)
        ), (
            f"docs/usage.md 'Skeleton seed' subsection must state 42 as the skel_seed value of the {JSON_NAME} "
            f"template; subsection: {prose!r}"
        )


@pytest.mark.smoke
class TestReadmeSkelSeedDeterminism:
    """REQ-SKELSEED-BDRABCZUK-73155499C72A."""

    def test_subsection_states_identical_crops(self):
        prose = _skeleton_seed_subsection()
        assert re.search(r"\bsame\b", prose, re.IGNORECASE) and re.search(
            r"\bidentical\b[^.]*\bcrops\b", prose, re.IGNORECASE
        ), (
            "docs/usage.md 'Skeleton seed' subsection must state that two runs with the same input and skeleton "
            f"seed produce identical crops; subsection: {prose!r}"
        )


@pytest.mark.smoke
class TestReadmeSkelSeedOlderDatasets:
    """REQ-SKELSEED-BDRABCZUK-2C0A2394DE52."""

    def test_subsection_warns_results_differ(self):
        prose = _skeleton_seed_subsection()
        missing = [
            word
            for word in ("crops", "embeddings", "before", "differ")
            if not re.search(rf"\b{word}", prose, re.IGNORECASE)
        ]
        assert not missing, (
            "docs/usage.md 'Skeleton seed' subsection must state that the crops and embeddings of datasets built "
            f"before the skel_seed key existed differ from those of a seeded run; missing words: {missing}"
        )
