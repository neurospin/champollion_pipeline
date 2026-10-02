"""Tests for REQ-UPDATE-05..06 — the history guard recognises a shallow clone.

TASK-158. ``scripts/check_update_history.sh`` (TASK-142) runs ``git merge-base
HEAD FETCH_HEAD``. In a shallow clone the common ancestor may lie beyond the
shallow boundary, so ``merge-base`` exits 1 and the script wrongly reports
"unrelated histories" with destructive reset/re-clone recovery steps.

The script under test is this repository's own ``scripts/check_update_history.sh``,
executed inside throw-away git repositories built with the ``_Sandbox`` harness of
``tests/test_pixi_update_task_safety.py`` — never against this repository. No
network access is used: the "remote" is a local repository cloned over
``file://`` (``--depth`` is ignored for plain local paths).

Shallow scenario: upstream ``A -> B``; the clone is ``--depth 1`` (only ``B``);
upstream ``main`` is then reset to ``A`` and gets a new commit ``C``, so a plain
``git fetch`` (as the update tasks run it) brings ``C`` whose ancestor ``A`` is
not an ancestor of the grafted ``B`` locally — ``merge-base`` exits 1 although
the histories are related.

- REQ-UPDATE-05: shallow + no merge base -> non-zero exit, output names
  ``git fetch --unshallow``.
- REQ-UPDATE-06: shallow + no merge base -> output has neither ``unrelated
  histories`` nor ``git reset --hard origin/main``.
- Guards (behaviour that must not change): a shallow clone whose fetched commit
  does share a merge base still exits 0; a full (non-shallow) clone facing a
  truly unrelated upstream still gets the unrelated-histories message, without
  the shallow hint.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.test_pixi_update_task_safety import REAL_GIT, SCRIPTS_DIR, _Sandbox

HISTORY_SCRIPT = SCRIPTS_DIR / "check_update_history.sh"
UNSHALLOW_HINT = "git fetch --unshallow"
RESET_STEP = "git reset --hard origin/main"
UNRELATED_PHRASE = "unrelated histories"

pytestmark = pytest.mark.skipif(REAL_GIT is None, reason="git executable not available")  # noqa: V107


class _ShallowSandbox(_Sandbox):
    """``_Sandbox`` whose local clone is ``--depth 1`` over ``file://``."""

    def create(self) -> None:
        self.upstream.mkdir()
        self.git(self.upstream, "init", "-q", "-b", "main")
        self._seed_tree(self.upstream, "original history")
        self.git(self.upstream, "add", "-A")
        self.git(self.upstream, "commit", "-q", "-m", "Commit A")
        (self.upstream / "README.md").write_text("commit B\n")
        self.git(self.upstream, "commit", "-q", "-am", "Commit B")
        self.local.parent.mkdir(parents=True)
        self.git(self.root, "clone", "-q", "--depth", "1", self.upstream.as_uri(), str(self.local))

    def diverge_beyond_shallow_boundary(self) -> None:
        """Related history whose merge base (commit A) is outside the shallow clone."""
        self.git(self.upstream, "reset", "-q", "--hard", "HEAD~1")
        (self.upstream / "README.md").write_text("commit C\n")
        self.git(self.upstream, "commit", "-q", "-am", "Commit C")


def _fetch(box: _Sandbox) -> None:
    box.git(box.local, "fetch", "-q")


def _run_history_check(box: _Sandbox) -> tuple[int, str]:
    result = subprocess.run(
        ["bash", str(HISTORY_SCRIPT)],
        cwd=box.local,
        env=box.task_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.returncode, result.stdout + result.stderr


def _merge_base_status(box: _Sandbox) -> int:
    return subprocess.run(
        [REAL_GIT, "merge-base", "HEAD", "FETCH_HEAD"],
        cwd=box.local,
        env=box.setup_env,
        capture_output=True,
        text=True,
    ).returncode


@pytest.fixture
def shallow_no_merge_base(tmp_path: Path) -> _ShallowSandbox:
    """Shallow clone after ``git fetch``, where ``merge-base HEAD FETCH_HEAD`` exits 1."""
    box = _ShallowSandbox(tmp_path)
    box.create()
    box.diverge_beyond_shallow_boundary()
    _fetch(box)
    assert box.git(box.local, "rev-parse", "--is-shallow-repository") == "true", "fixture clone is not shallow"
    assert _merge_base_status(box) == 1, "fixture did not reproduce a missing merge base in the shallow clone"
    return box


@pytest.mark.integration
class TestShallowCloneRecommendsUnshallow:
    """REQ-UPDATE-05: shallow clone without a reachable merge base -> non-zero exit and unshallow hint."""

    def test_exits_non_zero_and_recommends_unshallow(self, shallow_no_merge_base):
        code, output = _run_history_check(shallow_no_merge_base)

        assert code != 0, f"REQ-UPDATE-05: check_update_history.sh exited 0 on a shallow clone.\nOutput:\n{output}"
        assert UNSHALLOW_HINT in output, (
            f"REQ-UPDATE-05: output on a shallow clone with no reachable merge base lacks {UNSHALLOW_HINT!r}.\n"
            f"Output:\n{output}"
        )


@pytest.mark.integration
class TestShallowCloneOmitsUnrelatedRecovery:
    """REQ-UPDATE-06: shallow clone without a reachable merge base -> no unrelated-histories recovery text."""

    def test_output_omits_unrelated_histories_and_reset(self, shallow_no_merge_base):
        _, output = _run_history_check(shallow_no_merge_base)

        present = []
        if UNRELATED_PHRASE in output.lower():
            present.append(repr(UNRELATED_PHRASE))
        if RESET_STEP in output:
            present.append(repr(RESET_STEP))
        assert not present, (
            f"REQ-UPDATE-06: output on a shallow clone still contains {', '.join(present)}.\nOutput:\n{output}"
        )


@pytest.mark.integration
class TestHistoryGuardUnchangedOutsideShallowCase:
    """Guards: behaviour outside the shallow/no-merge-base case stays as TASK-142 left it."""

    def test_shallow_clone_with_merge_base_exits_zero(self, tmp_path):
        box = _ShallowSandbox(tmp_path)
        box.create()
        box.advance_upstream()
        _fetch(box)
        assert box.git(box.local, "rev-parse", "--is-shallow-repository") == "true"

        code, output = _run_history_check(box)

        assert code == 0, f"shallow clone with a reachable merge base was rejected.\nOutput:\n{output}"

    def test_full_clone_unrelated_history_keeps_recovery_text(self, tmp_path):
        box = _Sandbox(tmp_path)
        box.create()
        box.rewrite_upstream()
        _fetch(box)
        assert box.git(box.local, "rev-parse", "--is-shallow-repository") == "false"

        code, output = _run_history_check(box)

        assert code == 1, f"full clone on unrelated history exited {code}.\nOutput:\n{output}"
        assert UNRELATED_PHRASE in output.lower() and RESET_STEP in output, (
            f"full clone on unrelated history lost its recovery text.\nOutput:\n{output}"
        )
        assert UNSHALLOW_HINT not in output, f"full clone was given the shallow hint.\nOutput:\n{output}"
