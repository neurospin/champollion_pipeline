#!/usr/bin/env python3
"""Documentation guards: the docs/usage.md Stage sections hold the per-stage detail moved from README.md.

TASK-DOCS-BDRABCZUK-84BE8AB6FC73 (user decision 2026-11-23, 'Move to usage.md, short table'): README sections
2 to 7 move into the matching Stage 1 to Stage 6 sections of docs/usage.md, with their subsections and their
'All options' tables; every moved example must parse (README section 2's champollion-morphologist line does not).

Requirements pinned here (one test each):

* REQ-DOCS-BDRABCZUK-83937EF76C1C - Stage 1 has a level-3 subsection naming soma-workflow.
* REQ-DOCS-BDRABCZUK-317BA21D7C56 - Stage 2 subsections (``test_stage_subsections[2]``).
* REQ-DOCS-BDRABCZUK-8EAA85E85085 - Stage 3 subsections (``[3]``).
* REQ-DOCS-BDRABCZUK-CE049F944846 - Stage 4 subsections (``[4]``).
* REQ-DOCS-BDRABCZUK-6BA0668459D8 - Stage 6 subsections (``[6]``).
* REQ-DOCS-BDRABCZUK-197FC3EF7A1B - Stage 3 option table (``test_stage_has_option_table[3]``).
* REQ-DOCS-BDRABCZUK-314DCB5658CE - Stage 6 option table (``[6]``).
* REQ-DOCS-BDRABCZUK-D2C22D54EB5C - every stage command line of docs/usage.md parses.
* REQ-DOCS-BDRABCZUK-8B57BA988AB2 - Stage 1 shows a pixi run champollion-morphologist line.
* REQ-DOCS-BDRABCZUK-963DD11A0882 - no command line runs morphologist-cli directly.
* REQ-DOCS-BDRABCZUK-1863D0CBD9CC - heading links of README.md and docs/usage.md resolve.
* REQ-DOCS-BDRABCZUK-9676D5ECFD6A - quoted "<section>" in the README references name README.md headings.

Named outside ``tests/test_docs_*.py`` so ``verify_ignore`` keeps it in every full run. The parsers are only
built and given the documented tokens: nothing runs, no network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from champollion_pipeline.generate_champollion_config import GenerateChampollionConfig
from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from champollion_pipeline.generate_morphologist_graphs import GenerateMorphologistGraphs
from champollion_pipeline.generate_snapshots import GenerateSnapshots
from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings
from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from tests.docs_test_helpers import md_section, pixi_run_examples, read_text, refused_command_lines
from tests.test_readme_stage_table import _code_lines, _headings, _outside_fences, slug

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
USAGE = ROOT / "docs" / "usage.md"

STAGE_SUBSECTIONS = {
    2: ("Whole-brain volume", "QC File", "Skeleton seed", "Mask versions"),
    3: ("Non-standard derivatives layout", "Read-only containers (Apptainer / Docker)"),
    4: (
        "Checkpoint selection",
        "Models of one region",
        "Model sources",
        "Model set marker: model_set.json",
        "Run record: provenance.json",
        "Excluding models_cache/ from dataset sync and archives",
    ),
    6: ("UMAP Visualization", "Multiple acquisitions per subject"),
}
STAGE_BUILDERS = {
    "champollion-morphologist": GenerateMorphologistGraphs,
    "champollion-cortical-tiles": RunCorticalTiles,
    "champollion-config": GenerateChampollionConfig,
    "champollion-embeddings": GenerateEmbeddings,
    "champollion-combine": PutTogetherEmbeddings,
    "champollion-snapshots": GenerateSnapshots,
}
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
# The link targets resolved here, per source file: the href path before '#' -> the target file.
LINK_TARGETS = {
    README: {"": README, "README.md": README, "docs/usage.md": USAGE},
    USAGE: {"": USAGE, "usage.md": USAGE, "../README.md": README},
}
README_QUOTE_RE = re.compile(r'"([^"\n]+)"\s+in the README\b')


def _stage(number: int) -> str:
    return md_section(read_text(USAGE), rf"Stage {number}\b[^\n]*")


def _anchors(path: Path) -> set[str]:
    text = read_text(path)
    return {slug(h) for level in range(1, 7) for h in _headings(text, level)}


@pytest.mark.smoke
class TestStageSubsections:
    """REQ-DOCS-BDRABCZUK-83937EF76C1C, -317BA21D7C56, -8EAA85E85085, -CE049F944846, -6BA0668459D8."""

    def test_stage_1_soma_workflow_subsection(self):
        headings = _headings(_stage(1), 3)
        assert any("soma-workflow" in h for h in headings), (
            f"docs/usage.md Stage 1 has no level-3 subsection naming soma-workflow; level-3 headings: {headings}"
        )

    @pytest.mark.parametrize("number", sorted(STAGE_SUBSECTIONS))
    def test_stage_subsections(self, number):
        headings = _headings(_stage(number), 3)
        missing = [h for h in STAGE_SUBSECTIONS[number] if h not in headings]
        assert not missing, f"docs/usage.md Stage {number} lacks the level-3 subsections {missing}; has {headings}"


@pytest.mark.smoke
class TestStageOptionTables:
    """REQ-DOCS-BDRABCZUK-197FC3EF7A1B and REQ-DOCS-BDRABCZUK-314DCB5658CE."""

    @pytest.mark.parametrize("number", [3, 6])
    def test_stage_has_option_table(self, number):
        lines = _outside_fences(_stage(number))
        assert any(re.match(r"^\s*\|\s*Option\s*\|", ln) for ln in lines), (
            f"docs/usage.md Stage {number} has no table whose header row starts with Option"
        )


@pytest.mark.smoke
class TestUsageCommandLinesParse:
    """REQ-DOCS-BDRABCZUK-D2C22D54EB5C."""

    def test_stage_command_lines_parse(self):
        seen, refused = refused_command_lines(read_text(USAGE), STAGE_BUILDERS)
        assert seen, "no 'pixi run champollion-<stage>' line in docs/usage.md (test parser sanity check)"
        assert not refused, "docs/usage.md command lines refused by their argument parser:\n" + "\n".join(refused)


@pytest.mark.smoke
class TestStage1Wrapper:
    """REQ-DOCS-BDRABCZUK-8B57BA988AB2 and REQ-DOCS-BDRABCZUK-963DD11A0882 (one test each)."""

    def test_stage_1_shows_wrapper_command_line(self):
        assert pixi_run_examples(_stage(1), "champollion-morphologist"), (
            "docs/usage.md Stage 1 shows no 'pixi run champollion-morphologist' command line"
        )

    def test_no_raw_morphologist_cli_command_line(self):
        raw = [
            ln.strip()
            for ln in _code_lines(read_text(USAGE))
            if re.match(r"^\s*(?:pixi run\s+)?morphologist-cli(?![\w-])", ln)
        ]
        assert not raw, f"docs/usage.md runs morphologist-cli directly instead of champollion-morphologist: {raw}"


@pytest.mark.smoke
class TestHeadingLinks:
    """REQ-DOCS-BDRABCZUK-1863D0CBD9CC."""

    def test_heading_links_resolve(self):
        anchors = {path: _anchors(path) for path in (README, USAGE)}
        broken: list[str] = []
        checked = 0
        for source, targets in LINK_TARGETS.items():
            for line in _outside_fences(read_text(source)):
                for href in LINK_RE.findall(line):
                    if "#" not in href or "://" in href:
                        continue
                    path, anchor = href.split("#", 1)
                    target = targets.get(path)
                    if target is None:
                        continue
                    checked += 1
                    if anchor not in anchors[target]:
                        broken.append(f"{source.name}: {href}")
        assert checked, "no heading link found in README.md or docs/usage.md (test parser sanity check)"
        assert not broken, "links to a heading anchor that no heading produces:\n" + "\n".join(broken)


@pytest.mark.smoke
class TestReadmeReferences:
    """REQ-DOCS-BDRABCZUK-9676D5ECFD6A."""

    def test_quoted_readme_sections_exist(self):
        readme = read_text(README)
        titles = {h for level in range(1, 7) for h in _headings(readme, level)}
        titles |= {re.sub(r"^\d+\.\s+", "", t) for t in titles}
        quoted = README_QUOTE_RE.findall(read_text(USAGE))
        missing = [q for q in quoted if q not in titles]
        assert not missing, f"docs/usage.md refers to README sections that README.md does not hold: {missing}"
