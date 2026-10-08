#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Documentation guards for the versioned embeddings output layout (epic EMBVER).

The code now writes the stage 4-6 outputs under a mask-version folder:

    <datasets_root>/derivatives/champollion_V1/<masks_version>/region_embeddings  (stage 4)
    <datasets_root>/derivatives/champollion_V1/<masks_version>/embeddings         (stage 5)
    <datasets_root>/derivatives/champollion_V1/<masks_version>/snapshots          (stage 6)

``README.md`` and ``docs/usage.md`` must show that layout and no longer show the
unversioned ``derivatives/champollion_V1/{embeddings,snapshots}`` paths nor the
legacy ``{parent_of_datasets_root}/{dataset_name}embeddings`` default.

Requirements pinned here (one test per requirement):

* REQ-EMBVER-BDRABCZUK-648E7C808A66 - README ``--output`` row default.
* REQ-EMBVER-BDRABCZUK-A8C00CDCC4BA - README has no unversioned champollion_V1 output path.
* REQ-EMBVER-BDRABCZUK-8FAAE269F6A8 - README has no legacy ``{dataset}embeddings`` directory.
* REQ-EMBVER-BDRABCZUK-9701F325E029 - README combine source is versioned region_embeddings.
* REQ-EMBVER-BDRABCZUK-2B45967C0A58 - README combine ``--output_path`` is versioned embeddings.
* REQ-EMBVER-BDRABCZUK-12297D8A3F5E - README snapshots ``--embeddings_dir`` is versioned embeddings.
* REQ-EMBVER-BDRABCZUK-E6C7F51760B3 - README snapshots ``--output_dir`` is versioned snapshots.
* REQ-EMBVER-BDRABCZUK-4D7EBDA99E07 - usage.md has no unversioned champollion_V1 output path.
* REQ-EMBVER-BDRABCZUK-828BC3CDCCF6 - usage.md combine ``--output_path`` is versioned embeddings.
* REQ-EMBVER-BDRABCZUK-838D29CDB0B4 - usage.md snapshots ``--embeddings_dir`` is versioned embeddings.
* REQ-EMBVER-BDRABCZUK-8B768C32B6E6 - usage.md snapshots ``--output_dir`` is versioned snapshots.

Both files are read as plain text: no package import, no Sphinx build, no network.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

README = PROJECT_ROOT / "README.md"
USAGE = PROJECT_ROOT / "docs" / "usage.md"

EXAMPLE_ROOT = "/data/myproject/"
VERSIONED_ROOT = "/data/myproject/derivatives/champollion_V1/canonical_25"
VERSIONED_REGION_EMBEDDINGS = f"{VERSIONED_ROOT}/region_embeddings"
VERSIONED_EMBEDDINGS = f"{VERSIONED_ROOT}/embeddings"
VERSIONED_SNAPSHOTS = f"{VERSIONED_ROOT}/snapshots"

DEFAULT_OUTPUT_TEMPLATE = "{datasets_root}/derivatives/champollion_V1/{masks}/region_embeddings"

UNVERSIONED_PATH_RE = re.compile(
    r"derivatives/champollion_V1/(?:region_embeddings|embeddings|snapshots)(?![A-Za-z0-9_])"
)
LEGACY_SUBSTRINGS = ("{dataset}embeddings", "{dataset_name}embeddings", "myprojectembeddings")


def _read(path: Path) -> str:
    assert path.is_file(), f"expected file to exist: {path}"
    return path.read_text(encoding="utf-8")


def _norm(path: str) -> str:
    """Compare directories regardless of a trailing slash."""
    return path.rstrip("/")


def _bash_blocks(text: str) -> list[str]:
    """Bodies of every fenced ``bash`` code block."""
    return re.findall(r"```bash\n(.*?)```", text, re.DOTALL)


def _strip_comment(line: str) -> str:
    """Drop a shell comment: a whole comment line, or a `` # ...`` tail after whitespace."""
    if line.lstrip().startswith("#"):
        return ""
    return re.sub(r"\s+#.*$", "", line)


def _commands(text: str, command: str) -> list[list[str]]:
    """Argument tokens (after the command name) of every ``pixi run <command>`` example."""
    found: list[list[str]] = []
    for block in _bash_blocks(text):
        lines = block.splitlines()
        i = 0
        while i < len(lines):
            line = _strip_comment(lines[i]).rstrip()
            if re.search(rf"\bpixi run {re.escape(command)}(?:\s|\\|$)", line):
                parts = [line]
                while parts[-1].endswith("\\") and i + 1 < len(lines):
                    i += 1
                    parts.append(_strip_comment(lines[i]).rstrip())
                joined = " ".join(p[:-1] if p.endswith("\\") else p for p in parts)
                tokens = shlex.split(joined)
                start = tokens.index(command) + 1
                found.append(tokens[start:])
            i += 1
    return found


def _option_values(tokens: list[str], option: str) -> list[str]:
    """Values given to ``option`` in a token list (``--opt value`` or ``--opt=value``)."""
    values = []
    for idx, tok in enumerate(tokens):
        if tok == option and idx + 1 < len(tokens):
            values.append(tokens[idx + 1])
        elif tok.startswith(option + "="):
            values.append(tok.split("=", 1)[1])
    return values


def _offending_lines(text: str, pattern: re.Pattern[str]) -> list[str]:
    return [f"line {n}: {line.strip()}" for n, line in enumerate(text.splitlines(), 1) if pattern.search(line)]


def _section(text: str, heading_regex: str) -> str:
    m = re.search(rf"^##\s+{heading_regex}\s*$(.*?)(?=^##\s|\Z)", text, re.MULTILINE | re.DOTALL)
    assert m, f"no '## {heading_regex}' section found"
    return m.group(1)


def _assert_examples_use(path: Path, command: str, option: str, expected: str, *, under_example_root: bool) -> None:
    examples = _commands(_read(path), command)
    assert examples, f"{path.name}: no 'pixi run {command}' example found (test parser sanity check)"
    values = [v for tokens in examples for v in _option_values(tokens, option)]
    if under_example_root:
        values = [v for v in values if v.startswith(EXAMPLE_ROOT)]
    assert values, f"{path.name}: no {command} example sets {option} under {EXAMPLE_ROOT}"
    wrong = [v for v in values if _norm(v) != _norm(expected)]
    assert not wrong, f"{path.name}: {command} {option} must be {expected}/ (versioned layout); found {wrong}"


# ---------------------------------------------------------------------------
# README.md
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestReadmeEmbeddingsOutputDefault:
    """REQ-EMBVER-BDRABCZUK-648E7C808A66."""

    def test_output_row_states_versioned_region_embeddings_default(self):
        section = _section(_read(README), r"\d+\.\s+Generate Embeddings")
        rows = [ln for ln in section.splitlines() if re.match(r"^\|\s*`--output`\s*\|", ln)]
        assert len(rows) == 1, f"expected one `--output` row in the embeddings options table, found {rows}"
        assert re.search(re.escape(DEFAULT_OUTPUT_TEMPLATE) + r"/?", rows[0]), (
            f"the champollion-embeddings `--output` row must state {DEFAULT_OUTPUT_TEMPLATE}/ "
            f"as the default output directory; row is: {rows[0].strip()}"
        )


@pytest.mark.smoke
class TestReadmeNoUnversionedLayout:
    """REQ-EMBVER-BDRABCZUK-A8C00CDCC4BA and REQ-EMBVER-BDRABCZUK-8FAAE269F6A8."""

    def test_no_unversioned_champollion_v1_output_path(self):
        offending = _offending_lines(_read(README), UNVERSIONED_PATH_RE)
        assert not offending, (
            "README.md still shows derivatives/champollion_V1/{region_embeddings,embeddings,snapshots} "
            "without the <masks_version> folder:\n" + "\n".join(offending)
        )

    def test_no_legacy_dataset_embeddings_directory(self):
        legacy_re = re.compile("|".join(re.escape(s) for s in LEGACY_SUBSTRINGS))
        offending = _offending_lines(_read(README), legacy_re)
        assert not offending, (
            "README.md still names the legacy unversioned {parent_of_datasets_root}/{dataset_name}embeddings "
            "output directory:\n" + "\n".join(offending)
        )


@pytest.mark.smoke
class TestReadmeCombineExamples:
    """REQ-EMBVER-BDRABCZUK-9701F325E029 and REQ-EMBVER-BDRABCZUK-2B45967C0A58."""

    def test_combine_source_is_versioned_region_embeddings(self):
        examples = _commands(_read(README), "champollion-combine")
        assert examples, "README.md: no 'pixi run champollion-combine' example found (test parser sanity check)"
        sources = [tokens[0] if tokens and not tokens[0].startswith("--") else None for tokens in examples]
        wrong = [s for s in sources if s is None or _norm(s) != _norm(VERSIONED_REGION_EMBEDDINGS)]
        assert not wrong, (
            f"README.md: champollion-combine positional embeddings source must be "
            f"{VERSIONED_REGION_EMBEDDINGS}/; found {wrong}"
        )

    def test_combine_output_path_is_versioned_embeddings(self):
        _assert_examples_use(
            README, "champollion-combine", "--output_path", VERSIONED_EMBEDDINGS, under_example_root=False
        )


@pytest.mark.smoke
class TestReadmeSnapshotsExamples:
    """REQ-EMBVER-BDRABCZUK-12297D8A3F5E and REQ-EMBVER-BDRABCZUK-E6C7F51760B3."""

    def test_snapshots_embeddings_dir_is_versioned_embeddings(self):
        _assert_examples_use(
            README, "champollion-snapshots", "--embeddings_dir", VERSIONED_EMBEDDINGS, under_example_root=True
        )

    def test_snapshots_output_dir_is_versioned_snapshots(self):
        _assert_examples_use(
            README, "champollion-snapshots", "--output_dir", VERSIONED_SNAPSHOTS, under_example_root=True
        )


# ---------------------------------------------------------------------------
# docs/usage.md
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestUsageNoUnversionedLayout:
    """REQ-EMBVER-BDRABCZUK-4D7EBDA99E07."""

    def test_no_unversioned_champollion_v1_output_path(self):
        offending = _offending_lines(_read(USAGE), UNVERSIONED_PATH_RE)
        assert not offending, (
            "docs/usage.md still shows derivatives/champollion_V1/{region_embeddings,embeddings,snapshots} "
            "without the <masks_version> folder:\n" + "\n".join(offending)
        )


@pytest.mark.smoke
class TestUsageCombineExamples:
    """REQ-EMBVER-BDRABCZUK-828BC3CDCCF6."""

    def test_combine_output_path_is_versioned_embeddings(self):
        _assert_examples_use(
            USAGE, "champollion-combine", "--output_path", VERSIONED_EMBEDDINGS, under_example_root=False
        )


@pytest.mark.smoke
class TestUsageSnapshotsExamples:
    """REQ-EMBVER-BDRABCZUK-838D29CDB0B4 and REQ-EMBVER-BDRABCZUK-8B768C32B6E6."""

    def test_snapshots_embeddings_dir_is_versioned_embeddings(self):
        _assert_examples_use(
            USAGE, "champollion-snapshots", "--embeddings_dir", VERSIONED_EMBEDDINGS, under_example_root=True
        )

    def test_snapshots_output_dir_is_versioned_snapshots(self):
        _assert_examples_use(
            USAGE, "champollion-snapshots", "--output_dir", VERSIONED_SNAPSHOTS, under_example_root=True
        )
