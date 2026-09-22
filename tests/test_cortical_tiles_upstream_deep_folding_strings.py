#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pins the absence of the legacy package name on upstream ``cortical_tiles`` main.

REQ-UPSTREAM-01 (TASK-040) landed the directory rename ``deep_folding`` ->
``cortical_tiles`` on upstream ``main`` via merge ``a194dc1``. The old name
survived that rename as plain string content in four notebooks and as one doc
image file name. Each surviving occurrence is a fresh trap for the next
consumer to copy a stale import or path out of -- exactly the failure class
TASK-031/TASK-032/TASK-034 already had to repair downstream.

These tests observe the *remote* branch, not the local checkout: the
requirement is about what upstream ``main`` carries, so a cleanup performed
locally but never pushed must not turn them green.

The cleanup itself is upstream work that this repository is not allowed to do
-- AGENTS.md ("Operational Guardrails") forbids editing ``external/cortical_tiles``
from here, and the leftovers are notebook prose and one image file name, none of
which this pipeline imports. So both checks are marked ``xfail(strict=True)``:
the suite stays green while the gap is open, and the moment upstream lands the
cleanup they report XPASS, which is a failure, prompting removal of the marker.

Covers REQ-UPSTREAM-02.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMODULE = REPO_ROOT / "external" / "cortical_tiles"

# The legacy package identifier, as an exact literal. Prose spellings
# ("Deep folding") are deliberately out of scope -- see REQ-UPSTREAM-02.
LEGACY_NAME = "deep_folding"

REMOTE_MAIN = "refs/remotes/origin/main"

#: Why both checks are expected to fail. Remove the marker -- not the tests --
#: once neurospin/cortical_tiles@main carries the cleanup.
UPSTREAM_GAP = (
    "neurospin/cortical_tiles@main still carries the legacy 'deep_folding' name in "
    "notebook content and in docs/deep_folding.png; the cleanup belongs upstream and "
    "cannot be made from this repository (AGENTS.md: Operational Guardrails)"
)


def _git(*args):
    """Run git inside the submodule, returning (returncode, stdout)."""
    completed = subprocess.run(
        ["git", "-C", str(SUBMODULE), *args],
        capture_output=True,
        text=True,
    )
    return completed.returncode, completed.stdout.strip()


@pytest.fixture(scope="module")
def upstream_main():
    """Refresh ``origin/main`` from the remote and return its resolved SHA.

    Skips (never fails) when the submodule is not checked out or the remote is
    unreachable: an offline machine has no evidence either way, and a green
    verdict from stale local refs would be worse than no verdict.
    """
    if not (SUBMODULE / ".git").exists():
        pytest.skip(f"submodule not checked out at {SUBMODULE}")

    returncode, _ = _git("fetch", "--quiet", "origin", "main")
    if returncode != 0:
        pytest.skip("cannot reach git@github.com:neurospin/cortical_tiles.git")

    returncode, sha = _git("rev-parse", REMOTE_MAIN)
    if returncode != 0:
        pytest.skip(f"{REMOTE_MAIN} not available after fetch")
    return sha


@pytest.mark.integration
@pytest.mark.xfail(strict=True, reason=UPSTREAM_GAP)
class TestCorticalTilesUpstreamNoDeepFoldingStrings:
    """Upstream ``main`` must not mention ``deep_folding`` in paths or content."""

    def test_no_tracked_path_mentions_deep_folding(self, upstream_main):
        """No file or directory name on origin/main contains deep_folding."""
        returncode, listing = _git("ls-tree", "-r", "--name-only", upstream_main)
        assert returncode == 0, f"cannot list tree of {upstream_main}"

        offenders = sorted(path for path in listing.splitlines() if LEGACY_NAME in path)
        assert not offenders, (
            f"upstream main ({upstream_main[:7]}) still has {len(offenders)} tracked "
            f"path(s) naming '{LEGACY_NAME}': {offenders}"
        )

    def test_no_tracked_text_blob_contains_deep_folding(self, upstream_main):
        """No tracked text file's content on origin/main contains deep_folding."""
        # -I skips binary blobs; -l lists matching files only. Exit code 1 means
        # "no match", which is the state this test wants.
        returncode, matches = _git("grep", "-Il", LEGACY_NAME, upstream_main)
        assert returncode in (0, 1), f"git grep failed against {upstream_main}: rc={returncode}"

        offenders = sorted(line.split(":", 1)[1] for line in matches.splitlines() if ":" in line)
        assert not offenders, (
            f"upstream main ({upstream_main[:7]}) still has {len(offenders)} tracked "
            f"file(s) whose content mentions '{LEGACY_NAME}': {offenders}"
        )
