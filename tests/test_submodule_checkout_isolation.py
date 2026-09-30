#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guard: pipeline tests must not rewrite tracked files of the champollion_V1
submodule checkout (REQ-TESTISOL-01).

generate_champollion_config.run() writes
``<champollion_loc>/champollion/configs/dataset_localization/<name>.yaml``
unless ``--external-config`` is given, and ``--champollion_loc`` defaults to
the real ``external/champollion_V1`` checkout. Tests that exercise run()
without redirecting that write leave a dead ``/tmp`` ``dataset_folder`` in the
submodule's ``local.yaml``.

This guard runs the module known to exercise that code path in a child
pytest process, then compares every git-tracked file of the submodule
byte-for-byte with its pre-run content. Any file found changed is written
back to its pre-run bytes before the assertion, so the guard never leaves
the checkout polluted itself.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAMPOLLION_V1 = REPO_ROOT / "external" / "champollion_V1"
EXERCISED_MODULE = "tests/test_generate_champollion_config_internals.py"


def _tracked_files(checkout):
    """Return the git-tracked files of a checkout as absolute paths."""
    out = subprocess.run(
        ["git", "-C", str(checkout), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    return [checkout / rel.decode() for rel in out.split(b"\0") if rel]


def _snapshot(paths):
    """Map each existing path to its current bytes."""
    return {p: p.read_bytes() for p in paths if p.is_file()}


def test_config_generation_tests_leave_champollion_v1_checkout_unchanged():
    tracked = _tracked_files(CHAMPOLLION_V1) if (CHAMPOLLION_V1 / ".git").exists() else []
    if not tracked:
        pytest.skip("external/champollion_V1 submodule is not initialized")

    before = _snapshot(tracked)
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

    assert proc.returncode == 0, f"{EXERCISED_MODULE} itself failed:\n{proc.stdout[-2000:]}"
    assert changed == [], f"Running {EXERCISED_MODULE} rewrote tracked submodule files: {changed}"
