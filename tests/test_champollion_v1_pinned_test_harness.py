"""Tests for REQ-CHAMPTEST-25 — the pinned champollion_V1 commit carries the test harness.

``pixi run test-champollion`` runs ``external/champollion_V1/test``. From a fresh
clone (after ``git submodule update``) that directory is whatever the pipeline's
``external/champollion_V1`` gitlink records, so the gitlink staged in this repo's
index must point at a champollion_V1 commit whose ``test/`` tree contains every
harness module added by TASK-098 and TASK-100..TASK-104.

Offline: reads the gitlink via ``git ls-files -s`` and the tree via
``git ls-tree <sha> test/`` inside the local submodule; skips if the pinned commit
is not available locally.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMODULE = "external/champollion_V1"

EXPECTED_TEST_FILES = (
    "test/conftest.py",
    "test/test_harness_isolation.py",
    "test/test_train_model_summary.py",
    "test/test_smoke_train_step.py",
    "test/test_smoke_evaluate.py",
    "test/test_unit_augmentations.py",
    "test/test_unit_losses.py",
    "test/test_unit_model_paths.py",
)


def _git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)


@pytest.fixture(scope="module")
def pinned_test_tree() -> tuple[str, set[str]]:
    entry = _git("ls-files", "-s", SUBMODULE, cwd=REPO_ROOT).stdout.split()
    if len(entry) < 2 or entry[0] != "160000":
        pytest.skip(f"{SUBMODULE} is not a gitlink in this checkout")
    sha = entry[1]
    submodule_dir = REPO_ROOT / SUBMODULE
    if not (submodule_dir / ".git").exists():
        pytest.skip(f"{SUBMODULE} is not checked out")
    if _git("cat-file", "-e", f"{sha}^{{commit}}", cwd=submodule_dir).returncode != 0:
        pytest.skip(f"pinned commit {sha} not available locally")
    listed = _git("ls-tree", "-r", "--name-only", sha, "test/", cwd=submodule_dir)
    assert listed.returncode == 0, listed.stderr
    return sha, set(listed.stdout.split())


@pytest.mark.parametrize("path", EXPECTED_TEST_FILES)
def test_pinned_champollion_v1_contains_harness_file(pinned_test_tree, path):
    sha, files = pinned_test_tree
    assert path in files, (
        f"pinned champollion_V1 commit {sha[:7]} lacks {path} — bump the {SUBMODULE} "
        "gitlink to a champollion_V1 commit carrying the full test harness"
    )
