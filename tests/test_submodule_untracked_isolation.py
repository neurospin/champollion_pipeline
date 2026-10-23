#!/usr/bin/env python3
"""
REQ-TESTISOL-02: the submodule-isolation guard
(tests/test_submodule_checkout_isolation.py) must fail when its child test
run creates a new untracked, non-gitignored file under external/champollion_V1
or external/cortical_tiles, while ignoring files that were already untracked
before the run and files covered by a .gitignore.

The guard derives every path from its own location (``REPO_ROOT`` is the
parent of its ``tests/`` directory), so these tests copy it into a throwaway
fake repo root:

    <tmp>/tests/test_submodule_checkout_isolation.py           (copied guard)
    <tmp>/tests/test_generate_champollion_config_internals.py  (dummy child)
    <tmp>/external/champollion_V1/   (git repo: tracked + pre-existing untracked)
    <tmp>/external/cortical_tiles/   (git repo: tracked + pre-existing untracked)

and run the copied guard in a subprocess. Nothing under the real external/ is
touched.
"""

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GUARD = REPO_ROOT / "tests" / "test_submodule_checkout_isolation.py"
CHILD_MODULE = "test_generate_champollion_config_internals.py"
CHECKOUTS = ("champollion_V1", "cortical_tiles")

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")  # noqa: V107


# REQ-TESTCFG-BDRABCZUK-AB35C683B7CC / REQ-TESTCFG-BDRABCZUK-186ADD534A7F:
# a ledger commit made through a temporary index exports GIT_INDEX_FILE (and a
# worktree or hook may export GIT_DIR & co.); the helpers below must not let
# such repo-selecting variables redirect the git/pytest subprocesses they start.
REPO_SELECTING_GIT_VARS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR")


def _clean_git_env():
    return {k: v for k, v in os.environ.items() if k not in REPO_SELECTING_GIT_VARS}


def _git(cwd, *args):
    subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        env={
            **_clean_git_env(),
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    )


def _make_checkout(path):
    """Throwaway git repo standing in for a submodule checkout."""
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    (path / "tracked.txt").write_text("tracked\n")
    (path / ".gitignore").write_text("*.ignored\n")
    _git(path, "add", "tracked.txt", ".gitignore")
    _git(path, "commit", "-q", "-m", "init")
    # Pre-existing local untracked file (mirrors .codegraph/.gitignore in the
    # real checkouts): must not be reported by the guard.
    (path / ".codegraph").mkdir()
    (path / ".codegraph" / ".gitignore").write_text("*\n!.gitignore\n")


def _fake_repo(tmp_path, child_body):
    root = tmp_path / "fake_repo"
    (root / "tests").mkdir(parents=True)
    shutil.copy(GUARD, root / "tests" / GUARD.name)
    for name in CHECKOUTS:
        _make_checkout(root / "external" / name)
    (root / "tests" / CHILD_MODULE).write_text(
        textwrap.dedent(
            """
            from pathlib import Path

            EXTERNAL = Path(__file__).resolve().parent.parent / "external"


            def test_dummy():
            """
        )
        + textwrap.indent(textwrap.dedent(child_body), "    ")
    )
    return root


def _run_guard(root):
    env = {k: v for k, v in _clean_git_env().items() if k not in ("PYTEST_ADDOPTS", "COVERAGE_PROCESS_START")}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"tests/{GUARD.name}"],
        cwd=root,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


def _assert_guard_flags_new_untracked(tmp_path, checkout):
    leaked = f"configs/leaked_{checkout}.yaml"
    root = _fake_repo(
        tmp_path,
        f"""
        target = EXTERNAL / "{checkout}" / "{leaked}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("dataset_folder: /tmp/dead\\n")
        """,
    )
    proc = _run_guard(root)
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0 and leaked in output, (
        f"Guard did not flag new untracked file external/{checkout}/{leaked} (rc={proc.returncode}).\n{output[-3000:]}"
    )


def test_guard_fails_on_new_untracked_file_in_champollion_v1(tmp_path):
    _assert_guard_flags_new_untracked(tmp_path, "champollion_V1")


def test_guard_fails_on_new_untracked_file_in_cortical_tiles(tmp_path):
    _assert_guard_flags_new_untracked(tmp_path, "cortical_tiles")


def test_guard_ignores_preexisting_untracked_and_gitignored_files(tmp_path):
    root = _fake_repo(
        tmp_path,
        """
        for name in ("champollion_V1", "cortical_tiles"):
            (EXTERNAL / name / "scratch.ignored").write_text("ignored\\n")
        """,
    )
    proc = _run_guard(root)
    assert proc.returncode == 0, (
        "Guard failed although the child only wrote gitignored files and the only "
        f"other untracked files pre-existed.\n{(proc.stdout + proc.stderr)[-3000:]}"
    )


@pytest.mark.parametrize("var", REPO_SELECTING_GIT_VARS)
def test_git_helper_ignores_inherited_repo_selecting_var(tmp_path, monkeypatch, var):
    """REQ-TESTCFG-BDRABCZUK-AB35C683B7CC: _git drops ``var`` from its subprocess env."""
    bogus = tmp_path / "bogus" / var.lower()
    monkeypatch.setenv(var, str(bogus))
    checkout = tmp_path / "checkout"
    try:
        _make_checkout(checkout)
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"_git honoured inherited {var}={bogus}: {exc.stderr.decode(errors='replace').strip()}")
    listed = subprocess.run(
        ["git", "-C", str(checkout), "ls-files"], capture_output=True, text=True, env=_clean_git_env()
    )
    assert listed.stdout.split() == [".gitignore", "tracked.txt"], (
        f"_git wrote outside the checkout's own repository under inherited {var}: {listed.stdout}{listed.stderr}"
    )
    assert not bogus.exists(), f"_git wrote to the inherited {var} path {bogus}"


def test_run_guard_ignores_inherited_repo_selecting_vars(tmp_path, monkeypatch):
    """REQ-TESTCFG-BDRABCZUK-186ADD534A7F: _run_guard drops the vars from its pytest subprocess env."""
    root = tmp_path / "probe_repo"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / GUARD.name).write_text(
        textwrap.dedent(
            f"""
            import os


            def test_no_repo_selecting_git_vars():
                leaked = sorted(v for v in {REPO_SELECTING_GIT_VARS!r} if v in os.environ)
                assert not leaked, f"inherited: {{leaked}}"
            """
        )
    )
    for var in REPO_SELECTING_GIT_VARS:
        monkeypatch.setenv(var, str(tmp_path / "bogus" / var.lower()))
    proc = _run_guard(root)
    assert proc.returncode == 0, f"_run_guard passed repo-selecting git vars on:\n{(proc.stdout + proc.stderr)[-3000:]}"
