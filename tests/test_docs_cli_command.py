#!/usr/bin/env python3
"""Documentation guards: the docs run the pipeline as ``champollion-pipeline`` (TASK-CLICMD docs half).

Run by hand (``verify_ignore`` skips ``tests/test_docs_*.py``):
``pixi run pytest tests/test_docs_cli_command.py``.

Requirements pinned here (one test each):

* REQ-CLICMD-BDRABCZUK-3AEC0C259D33 - the first fenced bash block of README's first level-2 section runs
  ``champollion-pipeline`` and no per-stage ``champollion-*`` command.
* REQ-CLICMD-BDRABCZUK-46BC5168AD98 (supersedes 578A2F5DCF4B) - the ``<details>`` element of that section holding
  the per-stage commands has a fenced bash block listing the six stage commands.
* REQ-CLICMD-BDRABCZUK-217305FB444D - README, usage and workflow show no ``python main.py`` command.
* REQ-CLICMD-BDRABCZUK-BADEC38D0262 - each of the three files names ``main.py`` at most once.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.docs_test_helpers import stage_details

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOCS = [README, ROOT / "docs" / "usage.md", ROOT / "docs" / "workflow.md"]

STAGE_COMMANDS = [
    "champollion-morphologist",
    "champollion-cortical-tiles",
    "champollion-config",
    "champollion-embeddings",
    "champollion-combine",
    "champollion-snapshots",
]


def _first_section() -> str:
    text = README.read_text(encoding="utf-8")
    m = re.search(r"^##\s+.*?$(.*?)(?=^##\s)", text, re.MULTILINE | re.DOTALL)
    assert m, "README.md has no level-2 section"
    return m.group(1)


def _bash_blocks(text: str) -> list[str]:
    return re.findall(r"```bash\n(.*?)```", text, re.DOTALL)


def _code_lines(block: str) -> list[str]:
    return [ln for ln in block.splitlines() if not ln.lstrip().startswith("#")]


def test_first_section_block_runs_champollion_pipeline():
    section = _first_section()
    advanced = re.search(r"^###\s+Advanced", section, re.MULTILINE)
    head = section[: advanced.start()] if advanced else section
    blocks = _bash_blocks(head)
    assert blocks, "no fenced bash block before the Advanced subsection"
    code = "\n".join(_code_lines(blocks[0]))
    assert re.search(r"(?<![\w-])champollion-pipeline(?![\w-])", code), (
        "first block has no champollion-pipeline command"
    )
    stage = [c for c in STAGE_COMMANDS if c in code]
    assert not stage, f"first block still runs per-stage commands: {stage}"


def test_first_section_details_lists_six_stage_commands():
    blocks = _bash_blocks(stage_details(_first_section()).group("body"))
    assert blocks, "the <details> element holding the stage commands has no fenced bash block"
    code = "\n".join(_code_lines(blocks[0]))
    missing = [c for c in STAGE_COMMANDS if c not in code]
    assert not missing, f"Advanced <details> block lacks stage commands: {missing}"


def test_docs_show_no_python_main_py_command():
    offenders = [
        f"{p.name}:{i}"
        for p in DOCS
        for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"python\d?\s+(?:\./)?main\.py", ln)
    ]
    assert not offenders, f"python main.py command lines remain: {offenders}"


def test_docs_mention_main_py_at_most_once():
    counts = {p.name: len(re.findall(r"main\.py", p.read_text(encoding="utf-8"))) for p in DOCS}
    too_many = {k: v for k, v in counts.items() if v > 1}
    assert not too_many, f"main.py is named more than once: {too_many}"
