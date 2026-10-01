"""Tests for REQ-TEST-SPEED-02 — docs strict-build tests never touch docs/_build.

``tests/test_docs_pages.py::TestDocsStrictBuild`` used to delete and rebuild
the real ``docs/_build/html``, so two concurrent suite runs (or two xdist
workers) clobbered each other's build.

This module runs that class in a child pytest process with a small recorder
plugin. The plugin swaps any ``sphinx-build`` subprocess for ``true`` (so no
real build happens and the check stays fast), logging the argv and cwd it was
given, and logs every ``shutil.rmtree`` target. The assertions then read that
log: the output and doctree directories must lie beneath the child's
``--basetemp``, and nothing under ``docs/_build`` may have been removed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_BUILD = REPO_ROOT / "docs" / "_build"
TARGET = "tests/test_docs_pages.py::TestDocsStrictBuild"
PLUGIN_NAME = "docs_build_recorder"
LOG_ENV = "DOCS_BUILD_RECORD"

RECORDER_PLUGIN = '''\
"""Child-run plugin: fake sphinx-build, log its argv and every rmtree target."""
import json
import os
import shlex
import shutil
import subprocess

_LOG = os.environ["DOCS_BUILD_RECORD"]
_ORIG_POPEN_INIT = subprocess.Popen.__init__
_ORIG_RMTREE = shutil.rmtree


def _append(record):
    with open(_LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\\n")


def _argv(args):
    if isinstance(args, (str, bytes)):
        return shlex.split(os.fsdecode(args))
    return [os.fsdecode(a) for a in args]


def _is_sphinx(argv):
    for i, token in enumerate(argv):
        if os.path.basename(token) == "sphinx-build":
            return True
        if token == "-m" and i + 1 < len(argv) and argv[i + 1].split(".")[0] == "sphinx":
            return True
    return False


def _popen_init(self, args, *a, **kw):
    argv = _argv(args)
    if _is_sphinx(argv):
        cwd = kw.get("cwd")
        _append({"kind": "sphinx", "argv": argv, "cwd": os.fsdecode(cwd) if cwd else os.getcwd()})
        args = "true" if kw.get("shell") else ["true"]
    _ORIG_POPEN_INIT(self, args, *a, **kw)


def _rmtree(path, *a, **kw):
    _append({"kind": "rmtree", "path": os.path.abspath(os.fsdecode(path))})
    return _ORIG_RMTREE(path, *a, **kw)


subprocess.Popen.__init__ = _popen_init
shutil.rmtree = _rmtree
'''

# sphinx-build options that consume the following token as their value.
SPHINX_VALUE_OPTIONS = {
    "-b",
    "--builder",
    "-M",
    "-d",
    "--doctree-dir",
    "-c",
    "--conf-dir",
    "-D",
    "--define",
    "-A",
    "--html-define",
    "-t",
    "--tag",
    "-j",
    "--jobs",
    "-w",
    "--warning-file",
}


def _sphinx_args(argv: list[str]) -> list[str]:
    """The arguments after the ``sphinx-build`` executable / ``-m sphinx`` module."""
    for i, token in enumerate(argv):
        if os.path.basename(token) == "sphinx-build":
            return argv[i + 1 :]
        if token == "-m" and i + 1 < len(argv) and argv[i + 1].split(".")[0] == "sphinx":
            return argv[i + 2 :]
    raise AssertionError(f"not a sphinx invocation: {argv!r}")


def _positionals_and_doctree(args: list[str]) -> tuple[list[str], str | None]:
    positionals, doctree, skip_for = [], None, None
    for token in args:
        if skip_for is not None:
            if skip_for in ("-d", "--doctree-dir"):
                doctree = token
            skip_for = None
            continue
        if token.startswith("--doctree-dir="):
            doctree = token.split("=", 1)[1]
            continue
        if token.startswith("-"):
            if token in SPHINX_VALUE_OPTIONS:
                skip_for = token
            continue
        positionals.append(token)
    return positionals, doctree


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


@pytest.fixture(scope="module")
def child_run(tmp_path_factory) -> dict:
    """Run ``TestDocsStrictBuild`` in a child pytest with the recorder plugin."""
    if shutil.which("pixi") is None:
        pytest.skip("pixi is not on PATH; the strict-build fixture would skip before building")

    work = tmp_path_factory.mktemp("docs_confinement")
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
        pytest.skip(f"TestDocsStrictBuild skipped in the child run:\n{result.stdout[-2000:]}")
    assert sphinx_calls, (
        "the child run launched no sphinx-build subprocess, so the strict build cannot be observed\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n--- stderr ---\n{result.stderr[-4000:]}"
    )
    return {
        "basetemp": basetemp,
        "sphinx_calls": sphinx_calls,
        "rmtree_paths": [Path(r["path"]) for r in records if r["kind"] == "rmtree"],
    }


@pytest.mark.smoke
class TestDocsStrictBuildConfinement:
    """REQ-TEST-SPEED-02."""

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
        assert not offenders, f"the docs strict-build tests deleted paths under docs/_build: {offenders}"
