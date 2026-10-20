#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Documentation guards for the models_cache location and the --models-cache override (epic DATASETS).

Behaviour documented (read from the code, WIP 5f00c94):

* ``generate_embeddings.py`` caches downloaded/extracted models in
  ``compute_models_cache_dir(datasets_root)`` =
  ``abspath(<datasets_root>/derivatives/champollion_V1/models_cache)`` unless
  ``--models-cache PATH`` is given; a relative PATH resolves against the
  current working directory (``abspath``).
* ``main.py`` ``DatasetConfig.models_cache`` (YAML key ``dataset.models_cache``,
  default ``None``) is forwarded as ``--models-cache``.

Requirements pinned here (one test per requirement):

* REQ-DATASETS-BDRABCZUK-591B1EC86494 - the section 5 option table has a ``--models-cache`` row
  stating the default ``{datasets_root}/derivatives/champollion_V1/models_cache/``.
* REQ-DATASETS-BDRABCZUK-CEF567C3003A - that row names the pipeline config key ``dataset.models_cache``.
* REQ-DATASETS-BDRABCZUK-636BB2AF1532 - that row states a relative path resolves against the
  current working directory.
* REQ-DATASETS-BDRABCZUK-8BD1AEC2DABE - the "Model sources" subsection prose (before its
  collapsible option table) states the default cache directory.
* REQ-DATASETS-BDRABCZUK-6BC4F1FFE744 - README.md has no stale
  ``data/{dataset}/derivatives/champollion_V1/models_cache`` path.

README.md is read as plain text: no package import, no Sphinx build, no network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

README = PROJECT_ROOT / "README.md"

EMBEDDINGS_SECTION_HEADING = r"\d+\.\s+Generate Embeddings"
CACHE_PATH_RE = re.compile(r"\{datasets_root\}/derivatives/champollion_V1/models_cache/?(?![A-Za-z0-9_])")
STALE_CACHE_PATH = "data/{dataset}/derivatives/champollion_V1/models_cache"
OPTION_NAME = "--models-cache"


def _read(path: Path) -> str:
    assert path.is_file(), f"expected file to exist: {path}"
    return path.read_text(encoding="utf-8")


def _section(text: str, heading_regex: str) -> str:
    """Body of the ``## <heading>`` section, up to the next level-2 heading."""
    m = re.search(rf"^##\s+{heading_regex}\s*$(.*?)(?=^##\s|\Z)", text, re.MULTILINE | re.DOTALL)
    assert m, f"no '## {heading_regex}' section found in README.md"
    return m.group(1)


def _embeddings_section() -> str:
    return _section(_read(README), EMBEDDINGS_SECTION_HEADING)


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


def _models_cache_row_description() -> str:
    """Description cell of the ``--models-cache`` row of the section 5 option table."""
    rows = _option_table_rows(_embeddings_section())
    assert rows, "README.md section '5. Generate Embeddings' has no table whose header row starts with 'Option'"
    matches = [r for r in rows if r[0].strip("`").split()[0:1] == [OPTION_NAME]]
    assert len(matches) == 1, (
        f"README.md section 5 option table must contain exactly one '{OPTION_NAME}' row; "
        f"option cells found: {[r[0] for r in rows]}"
    )
    return " | ".join(matches[0][1:])


def _model_sources_prose() -> str:
    """Text of the ``### Model sources`` subsection before its collapsible ``<details>`` option table."""
    section = _embeddings_section()
    m = re.search(r"^###\s+Model sources\s*$(.*?)(?=^#{1,3}\s|\Z)", section, re.MULTILINE | re.DOTALL)
    assert m, "README.md section '5. Generate Embeddings' has no '### Model sources' subsection"
    return m.group(1).split("<details>", 1)[0]


@pytest.mark.smoke
class TestReadmeModelsCacheOptionRow:
    """REQ-DATASETS-BDRABCZUK-591B1EC86494."""

    def test_option_table_row_states_default(self):
        description = _models_cache_row_description()
        assert CACHE_PATH_RE.search(description), (
            f"README.md section 5 '{OPTION_NAME}' row must state the default directory "
            f"{{datasets_root}}/derivatives/champollion_V1/models_cache/; description: {description!r}"
        )


@pytest.mark.smoke
class TestReadmeModelsCacheConfigKey:
    """REQ-DATASETS-BDRABCZUK-CEF567C3003A."""

    def test_option_row_names_config_key(self):
        description = _models_cache_row_description()
        assert re.search(r"(?<![\w.])dataset\.models_cache(?!\w)", description), (
            f"README.md section 5 '{OPTION_NAME}' row must name the pipeline config key dataset.models_cache; "
            f"description: {description!r}"
        )


@pytest.mark.smoke
class TestReadmeModelsCacheRelativePath:
    """REQ-DATASETS-BDRABCZUK-636BB2AF1532."""

    def test_option_row_states_relative_path_resolution(self):
        description = _models_cache_row_description()
        assert re.search(r"\brelative\b", description, re.IGNORECASE) and re.search(
            r"\bcurrent working directory\b", description, re.IGNORECASE
        ), (
            f"README.md section 5 '{OPTION_NAME}' row must state that a relative path resolves against the "
            f"current working directory; description: {description!r}"
        )


@pytest.mark.smoke
class TestReadmeModelSourcesCachePath:
    """REQ-DATASETS-BDRABCZUK-8BD1AEC2DABE."""

    def test_model_sources_states_default_cache_dir(self):
        prose = _model_sources_prose()
        assert CACHE_PATH_RE.search(prose), (
            "README.md 'Model sources' subsection (before its option table) must state "
            "{datasets_root}/derivatives/champollion_V1/models_cache/ as the default directory of downloaded models"
        )


@pytest.mark.smoke
class TestReadmeNoStaleCachePath:
    """REQ-DATASETS-BDRABCZUK-6BC4F1FFE744."""

    def test_readme_has_no_stale_cache_path(self):
        text = _read(README)
        stale = [n for n, line in enumerate(text.splitlines(), 1) if STALE_CACHE_PATH in line]
        assert not stale, f"README.md still contains the stale path '{STALE_CACHE_PATH}' on line(s) {stale}"
