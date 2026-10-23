#!/usr/bin/env python3
"""Documentation guards for the run log written by ``--log`` (epic RUNLOG, docs task).

Behaviour documented (read from the code on branch req-runlog-bdrabczuk-f2f6a21fe6bb, d83a8a0,
and champollion_utils >= 0.3.0):

* Every stage script (ScriptBuilder subclass) accepts ``--log [PATH]``. A bare ``--log`` writes
  the run log at the script's ``default_log_path()``, named ``<stage>_<YYYYmmdd-HHMMSS>.log``
  in a ``.log`` folder; ``--log=PATH`` writes that file.
* Bare ``--log`` folders: morphologist ``<root>/derivatives/morphologist-6.0/.log``; cortical tiles
  ``<output>/cortical_tiles-2026/.log``; config, embeddings and training
  ``derivatives/champollion_V1/<masks version>/.log`` (training: ``canonical_25``); combine
  ``<--output_path>/.log``; snapshots ``<--output_dir>/.log``; prune and purge
  ``<dataset>/derivatives/.log``; generate_masks ``<--output_dir>/.log``; labelling QC ``.log`` next
  to the output TSV.
* The run log holds everything written to stdout/stderr, child processes included, between a
  header (argv, cwd, start time, pid) and a footer (exit code, elapsed time); the terminal still
  shows the output unchanged.
* A bare ``--log`` placed before a positional argument takes that argument as its PATH.
* ``main.py``: bare ``--log`` = each stage's default folder; ``--log=DIR`` writes
  ``DIR/<stage>_<YYYYmmdd-HHMMSS>.log``; top-level config key ``log``.

Requirements pinned here (one test per requirement):

* REQ-RUNLOG-BDRABCZUK-7E5CFA7FCA11 - README "Run logs" section names ``--log [PATH]``.
* REQ-RUNLOG-BDRABCZUK-486BE4E4CF6C - README table row per script with the bare ``--log`` folder.
* REQ-RUNLOG-BDRABCZUK-E0F3C9E1AC68 - README states the run-log file name pattern.
* REQ-RUNLOG-BDRABCZUK-786069CB7740 - README states ``--log=PATH``.
* REQ-RUNLOG-BDRABCZUK-D7FA3FC736BE - README warns about a bare ``--log`` before a positional.
* REQ-RUNLOG-BDRABCZUK-2945CBCDE672 - README states child-process output capture.
* REQ-RUNLOG-BDRABCZUK-1A7EDA30393C - README describes the header.
* REQ-RUNLOG-BDRABCZUK-2DDD5F9BD843 - README describes the footer.
* REQ-RUNLOG-BDRABCZUK-E80972233159 - README states the terminal output is kept.
* REQ-RUNLOG-BDRABCZUK-184CD12DF56E - README documents ``main.py`` bare ``--log``.
* REQ-RUNLOG-BDRABCZUK-DDF465E360CE - README documents ``main.py --log=DIR``.
* REQ-RUNLOG-BDRABCZUK-4AFACDFFB7DE - README names the ``log`` configuration key.
* REQ-RUNLOG-BDRABCZUK-7A260D79A5D6 - usage.md "Run logs" section shows a ``--log=PATH`` example.
* REQ-RUNLOG-BDRABCZUK-B4F23F62AAAB - usage.md states the bare ``--log`` folder of the six stages.
* REQ-RUNLOG-BDRABCZUK-5C5D9A90274C - usage.md warns about a bare ``--log`` before a positional.

Markdown is read as plain text (backticks removed): no package import, no Sphinx build, no network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

README = PROJECT_ROOT / "README.md"
USAGE = PROJECT_ROOT / "docs" / "usage.md"

SECTION_HEADING = r"Run logs"
FILE_NAME_PATTERN = "<stage>_<YYYYmmdd-HHMMSS>.log"
CHAMPOLLION_LOG_DIR = "derivatives/champollion_V1/"

# Script name -> text fragments its bare --log location must state.
README_LOCATIONS: dict[str, tuple[str, ...]] = {
    "champollion-morphologist": ("derivatives/morphologist-6.0/.log",),
    "champollion-cortical-tiles": ("cortical_tiles-2026/.log",),
    "champollion-config": (CHAMPOLLION_LOG_DIR, "/.log"),
    "champollion-embeddings": (CHAMPOLLION_LOG_DIR, "/.log"),
    "champollion-combine": ("--output_path", ".log"),
    "champollion-snapshots": ("--output_dir", ".log"),
    "champollion-train": ("derivatives/champollion_V1/canonical_25/.log",),
    "champollion-prune": ("derivatives/.log",),
    "champollion-purge": ("derivatives/.log",),
    "champollion-labelling-qc": ("TSV", ".log"),
    "generate_masks.py": ("--output_dir", ".log"),
}

# The six stages of docs/usage.md, by console script.
USAGE_STAGE_SCRIPTS: tuple[str, ...] = (
    "champollion-morphologist",
    "champollion-cortical-tiles",
    "champollion-config",
    "champollion-embeddings",
    "champollion-combine",
    "champollion-snapshots",
)


def _read(path: Path) -> str:
    assert path.is_file(), f"expected file to exist: {path}"
    return path.read_text(encoding="utf-8")


def _run_logs_section(path: Path, *, keep_backticks: bool = False) -> str:
    """Body of the level-2 ``## Run logs`` section, up to the next level-2 heading.

    Backticks are removed unless ``keep_backticks`` (needed to find fenced code blocks).
    """
    m = re.search(
        rf"^##\s+{SECTION_HEADING}\s*$(.*?)(?=^##\s|\Z)",
        _read(path),
        re.MULTILINE | re.DOTALL,
    )
    assert m, f"no '## {SECTION_HEADING}' section found in {path.name}"
    return m.group(1) if keep_backticks else m.group(1).replace("`", "")


def _paragraphs(section: str) -> list[str]:
    """Blank-line separated blocks of the section, each joined on one line and lower-cased."""
    return [" ".join(block.split()).lower() for block in re.split(r"\n\s*\n", section) if block.strip()]


def _has_paragraph(section: str, *patterns: str) -> bool:
    """True when one paragraph matches every regex in ``patterns`` (case-insensitive)."""
    return any(all(re.search(p, para) for p in patterns) for para in _paragraphs(section))


def _table_rows(section: str) -> list[str]:
    """Body lines of the Markdown tables of the section (separator rows dropped)."""
    return [
        line
        for line in section.splitlines()
        if line.lstrip().startswith("|") and not re.fullmatch(r"\s*\|[\s:|-]*\|?\s*", line)
    ]


def _assert_fragments(line: str, fragments: tuple[str, ...], where: str) -> None:
    missing = [f for f in fragments if f not in line]
    assert not missing, f"{where} lacks {missing}: {line.strip()!r}"


class TestReadmeRunLogSection:
    """REQ-RUNLOG-BDRABCZUK-7E5CFA7FCA11."""

    def test_section_names_log_option(self):
        section = _run_logs_section(README)
        assert "--log [PATH]" in section, "README 'Run logs' section does not name the --log [PATH] option"
        assert "generate_masks.py" in section, "README 'Run logs' section does not name generate_masks.py"


class TestReadmeRunLogLocations:
    """REQ-RUNLOG-BDRABCZUK-486BE4E4CF6C."""

    @pytest.mark.parametrize("script", sorted(README_LOCATIONS))
    def test_row_states_bare_log_folder(self, script):
        rows = [
            row
            for row in _table_rows(_run_logs_section(README))
            if re.search(rf"(?<![\w-]){re.escape(script)}(?![\w-])", row)
        ]
        assert len(rows) == 1, f"expected one README 'Run logs' table row for {script}, found {len(rows)}"
        _assert_fragments(rows[0], README_LOCATIONS[script], f"README row of {script}")


class TestReadmeRunLogFileName:
    """REQ-RUNLOG-BDRABCZUK-E0F3C9E1AC68."""

    def test_section_states_file_name_pattern(self):
        assert FILE_NAME_PATTERN in _run_logs_section(README)


class TestReadmeRunLogExplicitPath:
    """REQ-RUNLOG-BDRABCZUK-786069CB7740."""

    def test_section_states_log_equals_path(self):
        assert _has_paragraph(_run_logs_section(README), r"--log=path\b", r"\bfile\b")


class TestReadmeRunLogPositionalWarning:
    """REQ-RUNLOG-BDRABCZUK-D7FA3FC736BE."""

    def test_section_warns_bare_log_before_positional(self):
        assert _has_paragraph(_run_logs_section(README), r"\bbare --log\b", r"\bpositional\b")


class TestReadmeRunLogCapture:
    """REQ-RUNLOG-BDRABCZUK-2945CBCDE672."""

    def test_section_states_child_process_capture(self):
        assert _has_paragraph(
            _run_logs_section(README),
            r"standard output|\bstdout\b",
            r"standard error|\bstderr\b",
            r"child process|subprocess",
        )


class TestReadmeRunLogHeader:
    """REQ-RUNLOG-BDRABCZUK-1A7EDA30393C."""

    def test_section_describes_header(self):
        assert _has_paragraph(
            _run_logs_section(README),
            r"\bheader\b",
            r"command line|\bargv\b",
            r"working directory|\bcwd\b",
            r"\bstart",
            r"process id|\bpid\b",
        )


class TestReadmeRunLogFooter:
    """REQ-RUNLOG-BDRABCZUK-2DDD5F9BD843."""

    def test_section_describes_footer(self):
        assert _has_paragraph(_run_logs_section(README), r"\bfooter\b", r"exit code", r"\belapsed\b")


class TestReadmeRunLogTerminal:
    """REQ-RUNLOG-BDRABCZUK-E80972233159."""

    def test_section_states_terminal_output_kept(self):
        assert _has_paragraph(_run_logs_section(README), r"\bterminal\b", r"\bunchanged\b")


class TestReadmeRunLogMainBare:
    """REQ-RUNLOG-BDRABCZUK-184CD12DF56E."""

    def test_section_documents_main_bare_log(self):
        assert _has_paragraph(_run_logs_section(README), r"\bmain\.py\b", r"\bbare --log\b", r"\bstage\b")


class TestReadmeRunLogMainDir:
    """REQ-RUNLOG-BDRABCZUK-DDF465E360CE."""

    def test_section_documents_main_log_dir(self):
        section = _run_logs_section(README)
        assert "main.py --log=DIR" in section, "README 'Run logs' section shows no main.py --log=DIR"
        assert f"DIR/{FILE_NAME_PATTERN}" in section, "README 'Run logs' section does not state DIR/<stage>_<ts>.log"


class TestReadmeRunLogConfigKey:
    """REQ-RUNLOG-BDRABCZUK-4AFACDFFB7DE."""

    def test_section_names_log_config_key(self):
        assert _has_paragraph(
            _run_logs_section(README),
            r"\bconfig",
            r"(?:^|\s)log:(?:\s|$)|\bkey log\b",
        ), "README 'Run logs' section does not name the top-level configuration key log"


class TestUsageRunLogSection:
    """REQ-RUNLOG-BDRABCZUK-7A260D79A5D6."""

    def test_section_shows_log_equals_path_example(self):
        section = _run_logs_section(USAGE, keep_backticks=True)
        blocks = re.findall(r"^```[^\n]*\n(.*?)^```", section, re.MULTILINE | re.DOTALL)
        assert any(re.search(r"champollion-[\w-]+", b) and re.search(r"--log=\S", b) for b in blocks), (
            "docs/usage.md 'Run logs' section has no code example passing --log=PATH to a champollion-* script"
        )


class TestUsageRunLogLocations:
    """REQ-RUNLOG-BDRABCZUK-B4F23F62AAAB."""

    @pytest.mark.parametrize("script", USAGE_STAGE_SCRIPTS)
    def test_section_states_bare_log_folder(self, script):
        lines = [
            line
            for line in _run_logs_section(USAGE).splitlines()
            if re.search(rf"(?<![\w-]){re.escape(script)}(?![\w-])", line) and ".log" in line
        ]
        assert lines, f"docs/usage.md 'Run logs' section states no bare --log folder for {script}"
        _assert_fragments(lines[0], README_LOCATIONS[script], f"docs/usage.md line of {script}")


class TestUsageRunLogPositionalWarning:
    """REQ-RUNLOG-BDRABCZUK-5C5D9A90274C."""

    def test_section_warns_bare_log_before_positional(self):
        assert _has_paragraph(_run_logs_section(USAGE), r"\bbare --log\b", r"\bpositional\b")
