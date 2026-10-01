"""Tests for REQ-TEST-SPEED-07 — the docs-conf build test never touches docs/_build.

``tests/test_docs_conf.py::TestDocsConfBuild`` deletes and rebuilds the real
``docs/_build/html``, so concurrent suite runs, xdist workers, or a suite run
during ``pixi run docs-build`` collide on it.

Same technique as REQ-TEST-SPEED-02 (``test_docs_pages_build_isolation.py``,
whose recorder plugin and argv helpers are reused here): the class runs in a
child pytest with a plugin that swaps ``sphinx-build`` for ``true`` and logs
its argv plus every ``shutil.rmtree`` target. The output and doctree
directories must lie beneath the child's ``--basetemp``, and nothing under
``docs/_build`` may have been removed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_docs_pages_build_isolation import (
    LOG_ENV,
    PLUGIN_NAME,
    RECORDER_PLUGIN,
    _is_within,
    _positionals_and_doctree,
    _sphinx_args,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_BUILD = REPO_ROOT / "docs" / "_build"
TARGET = "tests/test_docs_conf.py::TestDocsConfBuild"


@pytest.fixture(scope="module")
def child_run(tmp_path_factory) -> dict:
    """Run ``TestDocsConfBuild`` in a child pytest with the recorder plugin."""
    if shutil.which("pixi") is None:
        pytest.skip("pixi is not on PATH; the docs-conf build fixture would skip before building")

    work = tmp_path_factory.mktemp("docs_conf_confinement")
    plugin_dir = work / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / f"{PLUGIN_NAME}.py").write_text(RECORDER_PLUGIN, encoding="utf-8")
    log = work / "record.jsonl"
    log.touch()
    basetemp = (work / "basetemp").resolve()

    env = {
        **os.environ,
        LOG_ENV: str(log),
        "PYTHONPATH": os.pathsep.join(filter(None, [str(plugin_dir), os.environ.get("PYTHONPATH")])),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            TARGET,
            "-p",
            PLUGIN_NAME,
            "-p",
            "no:cacheprovider",
            "-p",
            "no:xdist",
            "-o",
            "addopts=",
            "-q",
            "--basetemp",
            str(basetemp),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    sphinx_calls = [r for r in records if r["kind"] == "sphinx"]
    if not sphinx_calls and "skipped" in result.stdout and " passed" not in result.stdout:
        pytest.skip(f"TestDocsConfBuild skipped in the child run:\n{result.stdout[-2000:]}")
    assert sphinx_calls, (
        "the child run launched no sphinx-build subprocess, so the docs-conf build cannot be observed\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n--- stderr ---\n{result.stderr[-4000:]}"
    )
    return {
        "basetemp": basetemp,
        "sphinx_calls": sphinx_calls,
        "rmtree_paths": [Path(r["path"]) for r in records if r["kind"] == "rmtree"],
    }


@pytest.mark.smoke
class TestDocsConfBuildConfinement:
    """REQ-TEST-SPEED-07."""

    def test_output_dir_is_under_pytest_basetemp(self, child_run):
        """Every sphinx-build output directory lies beneath the child's --basetemp."""
        for call in child_run["sphinx_calls"]:
            positionals, _doctree = _positionals_and_doctree(_sphinx_args(call["argv"]))
            assert len(positionals) >= 2, f"sphinx-build argv has no output directory: {call['argv']!r}"
            out_dir = (Path(call["cwd"]) / positionals[1]).resolve()
            assert _is_within(out_dir, child_run["basetemp"]), (
                f"sphinx-build writes to {out_dir}, outside the pytest temporary base directory {child_run['basetemp']}"
            )

    def test_doctree_dir_is_under_pytest_basetemp(self, child_run):
        """Every doctree directory (explicit ``-d`` or Sphinx's ``<out>/.doctrees`` default) lies beneath --basetemp."""
        for call in child_run["sphinx_calls"]:
            positionals, doctree = _positionals_and_doctree(_sphinx_args(call["argv"]))
            assert len(positionals) >= 2, f"sphinx-build argv has no output directory: {call['argv']!r}"
            doctree_dir = (Path(call["cwd"]) / (doctree or Path(positionals[1]) / ".doctrees")).resolve()
            assert _is_within(doctree_dir, child_run["basetemp"]), (
                f"sphinx-build writes doctrees to {doctree_dir}, outside the pytest temporary base "
                f"directory {child_run['basetemp']}"
            )

    def test_nothing_under_docs_build_is_deleted(self, child_run):
        """No ``shutil.rmtree`` call targets ``docs/_build`` or anything beneath it."""
        offenders = [p for p in child_run["rmtree_paths"] if _is_within(p.resolve(), DOCS_BUILD.resolve())]
        assert not offenders, f"the docs-conf build test deleted paths under docs/_build: {offenders}"
