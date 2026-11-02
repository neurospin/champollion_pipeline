"""Documentation of ``main.py --mode streaming`` (TASK-STREAMING-BDRABCZUK-7C6982046A54).

The artifacts under test are the level-2 "Streaming mode" sections of ``docs/usage.md`` (user-facing: command,
options, logs, resume, exit code, limits) and ``docs/workflow.md`` (internal: producer/consumer design, arrival
barrier), plus a mention in ``README.md``.

Names and values that the code defines (log file names, the completion marker, default timeouts, the run log
folder of main.py) are read from the sources by static AST inspection, never imported, so a rename in the code
turns these tests red until the docs follow. A statement counts as present when one block of the section (a
paragraph, a list item or a table row) matches every pattern the test names.
"""

import ast
import functools
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
USAGE_MD = ROOT / "docs" / "usage.md"
WORKFLOW_MD = ROOT / "docs" / "workflow.md"
README_MD = ROOT / "README.md"
MAIN_PY = ROOT / "main.py"
STREAMING_DIR = ROOT / "src" / "champollion_pipeline" / "streaming"

SECTION_TITLE = "Streaming mode"
EXIT_1 = r"(?:code|exits?(?:\s+with)?|returns?)\s+`?1`?(?![\d.])"
EXIT_0 = r"(?:code|exits?(?:\s+with)?|returns?)\s+`?0`?(?![\d.])"


# --------------------------------------------------------------------------- #
# Static extraction helpers
# --------------------------------------------------------------------------- #


def _module_constant(path: Path, name: str):
    """Value of the module-level constant ``name = <literal>`` in ``path``."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{path} defines no module constant {name!r}")


def _section(path: Path, title: str = SECTION_TITLE) -> str:
    """Body of the first level-2 section of ``path`` whose heading contains ``title``; fails when absent."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    in_fence = False
    start = None
    for index, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line.startswith("## "):
            continue
        if start is not None:
            return "\n".join(lines[start:index])
        if title.lower() in line.lower():
            start = index + 1
    if start is not None:
        return "\n".join(lines[start:])
    raise AssertionError(f"{path.name} has no level-2 section whose heading contains {title!r}")


def _blocks(section: str) -> list[str]:
    """Paragraphs, list items and table rows of a section, each as one whitespace-normalised string."""
    blocks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            blocks.append(" ".join(" ".join(current).split()))
            current.clear()

    for line in section.splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
        elif re.match(r"^(?:[-*+]|\d+[.)])\s+|^\|", stripped):
            flush()
            current.append(stripped)
            if stripped.startswith("|"):
                flush()
        else:
            current.append(stripped)
    flush()
    return blocks


def _assert_block(section: str, *patterns: str, where: str) -> None:
    """Assert that one block of ``section`` matches every regex of ``patterns`` (case-insensitive)."""
    for block in _blocks(section):
        if all(re.search(pattern, block, re.IGNORECASE) for pattern in patterns):
            return
    raise AssertionError(f"no paragraph, list item or table row of {where} matches all of {patterns}")


def _code_blocks(section: str) -> list[str]:
    return re.findall(r"```[^\n]*\n(.*?)```", section, re.DOTALL)


@pytest.fixture(scope="module")
def usage_section():
    """Getter of the docs/usage.md section: a missing section fails each test rather than erroring its setup."""
    return functools.partial(_section, USAGE_MD)


@pytest.fixture(scope="module")
def workflow_section():
    """Getter of the docs/workflow.md section: a missing section fails each test rather than erroring its setup."""
    return functools.partial(_section, WORKFLOW_MD)


# --------------------------------------------------------------------------- #
# docs/usage.md
# --------------------------------------------------------------------------- #


class TestUsageStreamingSection:
    where = "the docs/usage.md Streaming mode section"

    def test_example_command(self, usage_section):
        """REQ: a level-2 Streaming mode section with a python main.py --mode streaming --config example."""
        section = usage_section()
        commands = [" ".join(b.replace("\\\n", " ").split()) for b in _code_blocks(section)]
        assert any("main.py" in c and re.search(r"--mode[ =]streaming", c) and "--config" in c for c in commands), (
            f"no code block of {self.where} runs main.py with --mode streaming and --config: {commands}"
        )

    def test_embeddings_only_guard(self, usage_section):
        """REQ: without dataset embeddings_only: true the run exits with code 1 before any scan."""
        section = usage_section()
        _assert_block(section, r"embeddings_only", EXIT_1, where=self.where)

    def test_subjects_directory(self, usage_section):
        """REQ: dataset input_path, else dataset morphologist_graphs, is the subjects directory read."""
        section = usage_section()
        _assert_block(section, r"input_path", r"morphologist_graphs", where=self.where)

    def test_no_morphologist_stage(self, usage_section):
        """REQ: a streaming run executes no Morphologist stage."""
        section = usage_section()
        _assert_block(
            section,
            r"Morphologist",
            r"\b(?:no|not|never|without|skips?)\b.{0,40}Morphologist|Morphologist.{0,40}\b(?:not|never)\b",
            where=self.where,
        )

    def test_n_workers(self, usage_section):
        """REQ: --n-workers sets the producer worker processes, one per CPU by default."""
        section = usage_section()
        _assert_block(section, r"--n-workers", r"\bCPUs?\b", where=self.where)

    def test_worker_timeout(self, usage_section):
        """REQ: a producer worker running longer than --worker-timeout (7200 s default) ends FAILED."""
        section = usage_section()
        default = _module_constant(STREAMING_DIR / "settings.py", "DEFAULT_WORKER_TIMEOUT")
        _assert_block(section, r"--worker-timeout", rf"\b{default}\b", r"\bFAILED\b", where=self.where)

    def test_dry_run(self, usage_section):
        """REQ: --dry-run lists the scans without processing them."""
        section = usage_section()
        _assert_block(section, r"--dry-run", r"\blists?\b", where=self.where)

    def test_log_dir_option(self, usage_section):
        """REQ: with --log DIR, DIR is the streaming log directory."""
        section = usage_section()
        _assert_block(section, r"--log[ =]`?DIR\b", where=self.where)

    def test_log_dir_bare_log(self, usage_section):
        """REQ: a bare --log gives ./.log/champollion_pipeline_<YYYYmmdd-HHMMSS>/ as the log directory."""
        section = usage_section()
        name = _module_constant(MAIN_PY, "SCRIPT_NAME")
        expected = rf"\./\.log/{re.escape(name)}_<YYYYmmdd-HHMMSS>/?(?!\.log)"
        _assert_block(section, r"\bbare\b", expected, where=self.where)

    def test_log_dir_default(self, usage_section):
        """REQ: without --log, the YAML log value, else log_dir, else <outputs_path>/logs."""
        section = usage_section()
        _assert_block(section, r"\blog_dir\b", r"<outputs_path>/logs", where=self.where)

    def test_worker_log_path(self, usage_section):
        """REQ: each scan's producer worker log is <log directory>/<subject>/t1mri/<acquisition>/worker.log."""
        section = usage_section()
        name = _module_constant(STREAMING_DIR / "worker.py", "WORKER_LOG_NAME")
        assert re.search(rf"<[^<>/]+>/<subject>/t1mri/<acquisition>/{re.escape(name)}", section), (
            f"{self.where} gives no <log directory>/<subject>/t1mri/<acquisition>/{name} path"
        )

    def test_consumer_log_path(self, usage_section):
        """REQ: the inference consumer log is <log directory>/consumer.log."""
        section = usage_section()
        name = _module_constant(STREAMING_DIR / "consumer.py", "CONSUMER_LOG_NAME")
        assert re.search(rf"<[^<>/]+>/{re.escape(name)}", section), f"{self.where} gives no <log directory>/{name} path"

    def test_worker_resume(self, usage_section):
        """REQ: a rerun skips the worker of a scan whose completion marker records the run's masks version."""
        section = usage_section()
        marker = _module_constant(STREAMING_DIR / "worker.py", "TILES_COMPLETE_MARKER")
        _assert_block(section, re.escape(marker), r"\bmasks?\b", r"\bskip", where=self.where)

    def test_consumer_resume(self, usage_section):
        """REQ: a rerun skips the embedding of a scan already holding a row in each region's full_embeddings.csv."""
        section = usage_section()
        name = _module_constant(STREAMING_DIR / "consumer.py", "_EMBEDDINGS_FILENAME")
        _assert_block(section, re.escape(name), r"\bskip", where=self.where)

    def test_exit_code_failed_scan(self, usage_section):
        """REQ: a streaming run exits with code 1 when at least one scan ends FAILED."""
        section = usage_section()
        _assert_block(section, r"\bFAILED\b", EXIT_1, where=self.where)

    def test_exit_code_success(self, usage_section):
        """REQ: a streaming run exits with code 0 when no scan ends FAILED and the combine succeeds."""
        section = usage_section()
        _assert_block(section, EXIT_0, r"\bcombine", where=self.where)

    def test_embeddings_path_refused(self, usage_section):
        """REQ: a non-empty dataset embeddings_path makes a streaming run exit with code 1."""
        section = usage_section()
        _assert_block(section, r"embeddings_path", EXIT_1, where=self.where)


# --------------------------------------------------------------------------- #
# docs/workflow.md
# --------------------------------------------------------------------------- #


class TestWorkflowStreamingSection:
    where = "the docs/workflow.md Streaming mode section"

    def test_entry_point(self, workflow_section):
        """REQ: the section names run_parallel_pipeline of champollion_pipeline.streaming.runner."""
        section = workflow_section()
        assert "run_parallel_pipeline" in section, f"{self.where} does not name run_parallel_pipeline"
        assert re.search(r"champollion_pipeline\.streaming\.runner|streaming/runner\.py", section), (
            f"{self.where} does not name champollion_pipeline.streaming.runner"
        )

    def test_producer_workers(self, workflow_section):
        """REQ: each producer worker runs cortical_tiles for one scan in a spawn process pool."""
        section = workflow_section()
        _assert_block(
            section,
            r"cortical_tiles",
            r"\bspawn\b",
            r"\b(?:one|each|per|single) scan\b",
            where=self.where,
        )

    def test_dataset_config_once(self, workflow_section):
        """REQ: the dataset config is generated once, from the crops of the first OK or SKIPPED scan."""
        section = workflow_section()
        _assert_block(section, r"\bconfig", r"\bonce\b", r"\bfirst\b", r"\bOK\b", r"\bSKIPPED\b", where=self.where)

    def test_resident_consumer(self, workflow_section):
        """REQ: one consumer process keeps a single EmbeddingEngine resident for the whole run."""
        section = workflow_section()
        _assert_block(section, r"\bconsumer\b", r"EmbeddingEngine", r"\bresident\b", where=self.where)

    def test_consumer_overlaps_producers(self, workflow_section):
        """REQ: the consumer embeds scans while the producer workers still run."""
        section = workflow_section()
        _assert_block(
            section,
            r"\bconsumer\b",
            r"\bproducer",
            r"\b(?:while|alongside|concurrently|in parallel)\b",
            where=self.where,
        )

    def test_arrival_barrier(self, workflow_section):
        """REQ: the arrival barrier waits for the masks-version completion marker plus each region's skeleton crops."""
        section = workflow_section()
        marker = _module_constant(STREAMING_DIR / "worker.py", "TILES_COMPLETE_MARKER")
        _assert_block(
            section,
            r"arrival barrier",
            re.escape(marker),
            r"\bmasks?\b",
            r"skeleton",
            where=self.where,
        )

    def test_never_ready_scan(self, workflow_section):
        """REQ: a scan still pending when a drain poll finds no ready scan is marked FAILED."""
        section = workflow_section()
        _assert_block(
            section,
            r"\bFAILED\b",
            r"\b(?:not|never) (?:yet )?ready\b|\bstill pending\b",
            r"\bproducer",
            where=self.where,
        )

    def test_consumer_stall_timeout(self, workflow_section):
        """REQ: the consumer stops waiting after 7200 seconds without a newly ready scan."""
        section = workflow_section()
        default = _module_constant(STREAMING_DIR / "settings.py", "DEFAULT_CONSUMER_STALL_TIMEOUT")
        _assert_block(
            section,
            r"\bconsumer\b",
            rf"\b{default}\b",
            r"\bstall|\bwithout a (?:new|newly)",
            where=self.where,
        )

    def test_combine_once(self, workflow_section):
        """REQ: a streaming run combines the per-region embeddings once, after the consumer returns."""
        section = workflow_section()
        _assert_block(section, r"\bcombine", r"\bonce\b", r"\bafter\b", r"\bconsumer\b", where=self.where)


# --------------------------------------------------------------------------- #
# README.md
# --------------------------------------------------------------------------- #


class TestReadmeStreamingMention:
    def test_mentions_streaming_with_usage_link(self):
        """REQ: README.md mentions python main.py --mode streaming with a link to docs/usage.md."""
        text = README_MD.read_text(encoding="utf-8")
        assert re.search(r"main\.py[^\n]*--mode[ =]streaming", text), (
            "README.md does not mention main.py --mode streaming"
        )
        assert re.search(r"\]\((?:\./)?docs/usage\.md(?:#[^)]*)?\)", text), "README.md has no link to docs/usage.md"
