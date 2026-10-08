#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Documentation guards for the models_cache sync/archive exclusion note (epic DATASETS).

USER DECISION 2026-10-08: the downloaded/extracted model cache lives inside the
dataset tree at ``{datasets_root}/derivatives/champollion_V1/models_cache/``
(REQ-DATASETS-BDRABCZUK-C5669FCE4AB8). It is multi-GB and regenerable, so it must
be kept out of any dataset sync or archive. No sync/archive tool exists in the
Champollion repos, so the deliverable is a ``README.md`` note: a level-3
subsection of section "5. Generate Embeddings" whose heading names
``models_cache``.

Requirements pinned here (one test per requirement):

* REQ-DATASETS-BDRABCZUK-BB97F11D9595 - section 5 has exactly one ``### ...models_cache...`` subsection.
* REQ-DATASETS-BDRABCZUK-A125F8E88891 - the subsection states the cache path
  ``{datasets_root}/derivatives/champollion_V1/models_cache/``.
* REQ-DATASETS-BDRABCZUK-89F7EA7F27C0 - the subsection contains the word "regenerable".
* REQ-DATASETS-BDRABCZUK-C76F6D8A92B4 - a fenced bash block has an ``rsync`` command whose
  ``--exclude`` names models_cache.
* REQ-DATASETS-BDRABCZUK-EF17E13F4D4D - a fenced bash block has a ``tar`` command whose
  ``--exclude`` names models_cache.
* REQ-DATASETS-BDRABCZUK-CF12570BA65B - the subsection contains the option name ``--models-cache``.

README.md is read as plain text: no package import, no Sphinx build, no network.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

README = PROJECT_ROOT / "README.md"

EMBEDDINGS_SECTION_HEADING = r"\d+\.\s+Generate Embeddings"
CACHE_PATH_RE = re.compile(r"\{datasets_root\}/derivatives/champollion_V1/models_cache/?(?![A-Za-z0-9_])")
SHELL_OPERATORS = {"|", "||", "&&", ";", "&"}


def _read(path: Path) -> str:
    assert path.is_file(), f"expected file to exist: {path}"
    return path.read_text(encoding="utf-8")


def _section(text: str, heading_regex: str) -> str:
    """Body of the ``## <heading>`` section, up to the next level-2 heading."""
    m = re.search(rf"^##\s+{heading_regex}\s*$(.*?)(?=^##\s|\Z)", text, re.MULTILINE | re.DOTALL)
    assert m, f"no '## {heading_regex}' section found in README.md"
    return m.group(1)


def _subsections(section: str) -> list[tuple[str, str]]:
    """``(heading, body)`` of every level-3 subsection, ignoring ``#`` lines inside code fences."""
    found: list[tuple[str, list[str]]] = []
    in_fence = False
    for line in section.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not in_fence:
            heading = re.match(r"^###\s+(.*?)\s*$", line)
            if heading:
                found.append((heading.group(1), []))
                continue
            if re.match(r"^#{1,3}\s", line):
                found.append(("", []))  # a shallower heading closes the current subsection
                continue
        if found:
            found[-1][1].append(line)
    return [(h, "\n".join(body)) for h, body in found if h]


def _models_cache_subsections() -> list[tuple[str, str]]:
    section = _section(_read(README), EMBEDDINGS_SECTION_HEADING)
    return [(h, body) for h, body in _subsections(section) if "models_cache" in h]


def _models_cache_note() -> str:
    """Body of the single models_cache subsection of section 5."""
    matches = _models_cache_subsections()
    assert len(matches) == 1, (
        "README.md section '5. Generate Embeddings' must have exactly one level-3 subsection whose "
        f"heading contains 'models_cache'; found headings {[h for h, _ in matches]}"
    )
    return matches[0][1]


def _bash_blocks(text: str) -> list[str]:
    """Bodies of every fenced ``bash`` code block."""
    return re.findall(r"```bash\n(.*?)```", text, re.DOTALL)


def _strip_comment(line: str) -> str:
    """Drop a shell comment: a whole comment line, or a `` # ...`` tail after whitespace."""
    if line.lstrip().startswith("#"):
        return ""
    return re.sub(r"\s+#.*$", "", line)


def _command_invocations(text: str, command: str) -> list[list[str]]:
    """Argument tokens of every ``command`` invocation in the fenced bash blocks of ``text``."""
    found: list[list[str]] = []
    for block in _bash_blocks(text):
        logical: list[str] = []
        current = ""
        for raw in block.splitlines():
            line = _strip_comment(raw).rstrip()
            if line.endswith("\\"):
                current += line[:-1] + " "
                continue
            logical.append(current + line)
            current = ""
        if current:
            logical.append(current)
        for line in logical:
            try:
                tokens = shlex.split(line)
            except ValueError:
                continue
            for idx, tok in enumerate(tokens):
                if tok == command:
                    args: list[str] = []
                    for arg in tokens[idx + 1 :]:
                        if arg in SHELL_OPERATORS:
                            break
                        args.append(arg)
                    found.append(args)
    return found


def _exclude_values(args: list[str]) -> list[str]:
    """Values given to ``--exclude`` (``--exclude value`` or ``--exclude=value``)."""
    values = []
    for idx, tok in enumerate(args):
        if tok == "--exclude" and idx + 1 < len(args):
            values.append(args[idx + 1])
        elif tok.startswith("--exclude="):
            values.append(tok.split("=", 1)[1])
    return values


def _assert_exclude_example(command: str) -> None:
    note = _models_cache_note()
    invocations = _command_invocations(note, command)
    assert invocations, f"README.md models_cache subsection has no fenced bash block with a '{command}' command"
    excludes = [v for args in invocations for v in _exclude_values(args)]
    assert any("models_cache" in v for v in excludes), (
        f"README.md models_cache subsection: no '{command}' command passes an --exclude value containing "
        f"'models_cache'; --exclude values found: {excludes}"
    )


@pytest.mark.smoke
class TestReadmeModelsCacheSubsection:
    """REQ-DATASETS-BDRABCZUK-BB97F11D9595."""

    def test_embeddings_section_has_exactly_one_models_cache_subsection(self):
        matches = _models_cache_subsections()
        assert len(matches) == 1, (
            "README.md section '5. Generate Embeddings' must contain exactly one level-3 subsection whose "
            f"heading contains 'models_cache'; found headings {[h for h, _ in matches]}"
        )


@pytest.mark.smoke
class TestReadmeModelsCachePath:
    """REQ-DATASETS-BDRABCZUK-A125F8E88891."""

    def test_subsection_states_cache_path(self):
        assert CACHE_PATH_RE.search(_models_cache_note()), (
            "README.md models_cache subsection must state the cache path "
            "{datasets_root}/derivatives/champollion_V1/models_cache/"
        )


@pytest.mark.smoke
class TestReadmeModelsCacheRegenerable:
    """REQ-DATASETS-BDRABCZUK-89F7EA7F27C0."""

    def test_subsection_says_regenerable(self):
        assert re.search(r"\bregenerable\b", _models_cache_note(), re.IGNORECASE), (
            "README.md models_cache subsection must contain the word 'regenerable'"
        )


@pytest.mark.smoke
class TestReadmeModelsCacheRsyncExclude:
    """REQ-DATASETS-BDRABCZUK-C76F6D8A92B4."""

    def test_subsection_has_rsync_exclude_example(self):
        _assert_exclude_example("rsync")


@pytest.mark.smoke
class TestReadmeModelsCacheTarExclude:
    """REQ-DATASETS-BDRABCZUK-EF17E13F4D4D."""

    def test_subsection_has_tar_exclude_example(self):
        _assert_exclude_example("tar")


@pytest.mark.smoke
class TestReadmeModelsCacheOverrideOption:
    """REQ-DATASETS-BDRABCZUK-CF12570BA65B."""

    def test_subsection_names_models_cache_option(self):
        assert re.search(r"--models-cache(?![\w-])", _models_cache_note()), (
            "README.md models_cache subsection must contain the champollion-embeddings option name --models-cache"
        )
