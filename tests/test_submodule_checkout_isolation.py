#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guard: pipeline tests must not write into the external/ submodule checkouts
(REQ-TESTISOL-01, REQ-TESTISOL-02).

generate_champollion_config.run() writes YAMLs to the dataset's own derivatives
tree by default (``<D>/<dataset>/derivatives/champollion_V1/configs``), so a
default run no longer touches ``--champollion_loc``.  Writing inside the
checkout is opt-in only, via explicit ``--output`` / ``--external-config``
paths.  Tests that exercise run() use isolated tmp paths and/or the autouse
fixture that redirects ``_DEFAULT_CHAMPOLLION_LOC``, so nothing is ever written
into the real ``external/champollion_V1`` checkout.

This guard runs the module known to exercise that code path in a child
pytest process, then checks every initialised checkout under ``external/``:

- every git-tracked file must be byte-identical to its pre-run content
  (REQ-TESTISOL-01);
- no untracked, non-gitignored file may appear that was not already
  untracked before the run (REQ-TESTISOL-02).

Before asserting, the guard undoes what it found: changed tracked files are
written back to their pre-run bytes, and newly created untracked files are
deleted. Files that were already untracked before the run are never touched.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKOUTS = (
    REPO_ROOT / "external" / "champollion_V1",
    REPO_ROOT / "external" / "cortical_tiles",
)
EXERCISED_MODULE = "tests/test_generate_champollion_config_internals.py"
pytestmark = pytest.mark.serial  # noqa: V107


def _git_paths(checkout, *ls_files_args):
    """Run ``git ls-files -z`` in a checkout and return absolute paths."""
    out = subprocess.run(
        ["git", "-C", str(checkout), "ls-files", "-z", *ls_files_args],
        check=True,
        capture_output=True,
    ).stdout
    return [checkout / rel.decode() for rel in out.split(b"\0") if rel]


def _tracked_files(checkout):
    """Return the git-tracked files of a checkout as absolute paths."""
    return _git_paths(checkout)


def _untracked_files(checkout):
    """Return the untracked, non-gitignored files of a checkout."""
    return set(_git_paths(checkout, "--others", "--exclude-standard"))


def _snapshot(paths):
    """Map each existing path to its current bytes."""
    return {p: p.read_bytes() for p in paths if p.is_file()}


def test_config_generation_tests_leave_external_checkouts_unchanged():
    checkouts = [c for c in CHECKOUTS if (c / ".git").exists()]
    if not checkouts:
        pytest.skip("no external/ submodule is initialized")

    before = _snapshot(p for c in checkouts for p in _tracked_files(c))
    untracked_before = {c: _untracked_files(c) for c in checkouts}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", EXERCISED_MODULE],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
    finally:
        after = _snapshot(before)
        changed = sorted(str(p.relative_to(REPO_ROOT)) for p, data in before.items() if after.get(p) != data)
        for p, data in before.items():
            if after.get(p) != data:
                p.write_bytes(data)
        leaked = sorted(p for c in checkouts for p in _untracked_files(c) - untracked_before[c])
        for p in leaked:
            if p.is_file() or p.is_symlink():
                p.unlink()
        leaked = [str(p.relative_to(REPO_ROOT)) for p in leaked]

    assert proc.returncode == 0, f"{EXERCISED_MODULE} itself failed:\n{proc.stdout[-2000:]}"
    assert changed == [], f"Running {EXERCISED_MODULE} rewrote tracked submodule files: {changed}"
    assert leaked == [], f"Running {EXERCISED_MODULE} created untracked submodule files: {leaked}"
