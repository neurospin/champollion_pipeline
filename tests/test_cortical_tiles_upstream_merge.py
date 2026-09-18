#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pins the upstream merge state of the ``external/cortical_tiles`` submodule.

The package rename ``deep_folding`` -> ``cortical_tiles`` lives on the upstream
branch ``rename-deep-folding-to-cortical-tiles`` (tip ``fc7dee1``), while
``origin/main`` has kept moving on a diverged line (tip ``acc82fe``). Until the
two lines are merged, anything that pins this submodule to upstream ``main``
still gets the old ``deep_folding`` package tree, which is exactly the class of
breakage TASK-031/TASK-032/TASK-034 already had to repair downstream.

These tests observe the *remote* branch, not the local checkout: the
requirement is about what upstream ``main`` carries, so a local merge that was
never pushed must not turn them green.

Covers REQ-UPSTREAM-01.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMODULE = REPO_ROOT / "external" / "cortical_tiles"

# Tip of ``rename-deep-folding-to-cortical-tiles``: the commit that renamed the
# package directory. This is also the commit this repo currently pins.
RENAME_TIP = "fc7dee1b079bd1d7bb188d05eac635f2ed7c0681"

# Tip of upstream ``main`` before the merge. Asserting this stays reachable is
# what distinguishes a merge from a force-push that discards main's own three
# diverged commits (acc82fe, 9ad2c80, 6292597).
PREMERGE_MAIN_TIP = "acc82feb0ca3b4008e926a586d3d9cdb5ad9f975"

REMOTE_MAIN = "refs/remotes/origin/main"


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
class TestCorticalTilesUpstreamMainMerged:
    """Upstream ``main`` must carry the rename without losing its own history."""

    def test_upstream_main_contains_rename_commit(self, upstream_main):
        """origin/main reaches fc7dee1, the commit that renamed the package."""
        returncode, _ = _git("merge-base", "--is-ancestor", RENAME_TIP, upstream_main)
        assert returncode == 0, (
            f"upstream main ({upstream_main[:7]}) does not contain "
            f"{RENAME_TIP[:7]}; branch rename-deep-folding-to-cortical-tiles "
            f"is still unmerged"
        )

    def test_upstream_main_still_contains_its_own_diverged_tip(self, upstream_main):
        """origin/main still reaches acc82fe, so the merge kept main's commits."""
        returncode, _ = _git("merge-base", "--is-ancestor", PREMERGE_MAIN_TIP, upstream_main)
        assert returncode == 0, (
            f"upstream main ({upstream_main[:7]}) no longer contains "
            f"{PREMERGE_MAIN_TIP[:7]}; main's three diverged commits were "
            f"discarded rather than merged"
        )

    def test_upstream_main_tree_exposes_cortical_tiles_package(self, upstream_main):
        """origin/main's top level holds cortical_tiles/ and no deep_folding/."""
        returncode, listing = _git("ls-tree", "--name-only", upstream_main)
        assert returncode == 0, f"cannot list tree of {upstream_main}"

        entries = set(listing.splitlines())
        assert "cortical_tiles" in entries, (
            f"upstream main has no top-level cortical_tiles/ directory; top level is {sorted(entries)}"
        )
        assert "deep_folding" not in entries, "upstream main still has a top-level deep_folding/ directory"
