#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pins the absence of the legacy package name from upstream ``cortical_tiles``
main's tracked *paths*.

REQ-UPSTREAM-01 (TASK-040) landed the directory rename ``deep_folding`` ->
``cortical_tiles`` on upstream ``main`` via merge ``a194dc1``. One stray file
name, ``docs/deep_folding.png``, survived that rename -- fixed by
REQ-UPSTREAM-03 (TASK-085, commit 17b5d4d on upstream ``main``).

This test observes the *remote* branch, not the local checkout: the
requirement is about what upstream ``main`` carries, so a cleanup performed
locally but never pushed would not turn it green.

We hold admin/push ownership on ``neurospin/cortical_tiles`` (AGENTS.md:13),
so this cleanup was made directly there, not just reported.

Note: this test only checks tracked *paths*, not text content -- the four
notebooks' ``/neurospin/dico/data/deep_folding/...`` references are correct
as written (a real, permanent Neurospin data-storage directory, unrelated to
the package rename; confirmed to exist on rosette). A content-mentions check
was dropped from REQ-UPSTREAM-02/03's scope for that reason -- see
REQ-UPSTREAM-03's "Narrowed" note in elm/REQUIREMENTS.md.

Covers REQ-UPSTREAM-02.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMODULE = REPO_ROOT / "external" / "cortical_tiles"

# The legacy package identifier, as an exact literal.
LEGACY_NAME = "deep_folding"

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
class TestCorticalTilesUpstreamNoDeepFoldingStrings:
    """Upstream ``main`` must not mention ``deep_folding`` in tracked paths."""

    def test_no_tracked_path_mentions_deep_folding(self, upstream_main):
        """No file or directory name on origin/main contains deep_folding."""
        returncode, listing = _git("ls-tree", "-r", "--name-only", upstream_main)
        assert returncode == 0, f"cannot list tree of {upstream_main}"

        offenders = sorted(path for path in listing.splitlines() if LEGACY_NAME in path)
        assert not offenders, (
            f"upstream main ({upstream_main[:7]}) still has {len(offenders)} tracked "
            f"path(s) naming '{LEGACY_NAME}': {offenders}"
        )
