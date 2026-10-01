"""Tests for REQ-UPDATE-01..04 — pixi update tasks fail safely and explicitly.

TASK-142. After ``origin/main`` was force-pushed (rewritten history), an old
clone running ``pixi run update`` hit ``fatal: refusing to merge unrelated
histories`` with no recovery guidance, and submodule fetches prompted for
credentials against a stale submodule URL.

The tasks under test are read from the real ``pixi.toml`` but executed (through
``bash``, as an approximation of pixi's task shell) inside throw-away git
repositories under ``tmp_path`` — never against this repository. A ``git``
wrapper on ``PATH`` records every invocation (and its ``GIT_TERMINAL_PROMPT``
value) before delegating to the real ``git``; a ``pip`` stub records install
attempts without installing anything. No network access is used: the "remote"
is a local repository.

- REQ-UPDATE-01: unrelated history -> non-zero exit, ``HEAD`` unchanged, no
  ``pip`` invocation.
- REQ-UPDATE-02: unrelated history -> message naming the problem and the
  recovery steps.
- REQ-UPDATE-03: every ``git`` process runs with ``GIT_TERMINAL_PROMPT=0``.
- REQ-UPDATE-04: ``git submodule sync --recursive`` precedes the first
  ``git submodule update``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIXI_TOML = PROJECT_ROOT / "pixi.toml"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
REAL_GIT = shutil.which("git")

# (task scope, task name): "workspace" is [tasks], "embeddings" is
# [feature.embeddings.tasks].
UPDATE_PIPELINE = ("workspace", "update-pipeline")
UPDATE_SUBMODULES = ("workspace", "update-submodules")
EMBEDDINGS_UPDATE = ("embeddings", "update")

HISTORY_GUARDED_TASKS = [UPDATE_PIPELINE, EMBEDDINGS_UPDATE]
NO_PROMPT_TASKS = [UPDATE_PIPELINE, UPDATE_SUBMODULES, EMBEDDINGS_UPDATE]
SUBMODULE_SYNC_TASKS = [UPDATE_SUBMODULES, EMBEDDINGS_UPDATE]

pytestmark = pytest.mark.skipif(REAL_GIT is None, reason="git executable not available")  # noqa: V107


def _task_id(task: tuple[str, str]) -> str:
    scope, name = task
    return name if scope == "workspace" else f"{scope}:{name}"


@pytest.fixture(scope="module")
def pixi_config() -> dict:
    with PIXI_TOML.open("rb") as handle:
        return tomllib.load(handle)


def _lookup_task(config: dict, scope: str, name: str):
    """Resolve a task the way pixi does for an environment: feature first, then [tasks]."""
    if scope != "workspace":
        feature_tasks = config.get("feature", {}).get(scope, {}).get("tasks", {})
        if name in feature_tasks:
            return feature_tasks[name]
    workspace_tasks = config.get("tasks", {})
    assert name in workspace_tasks, f"pixi.toml defines no task named {name!r} (scope {scope!r})"
    return workspace_tasks[name]


def _run_task(config: dict, scope: str, name: str, cwd: Path, env: dict[str, str]) -> tuple[int, str]:
    """Run a pixi task (dependencies first) via bash; stop at the first failure.

    Returns the exit status of the last command run and the combined output.
    """
    task = _lookup_task(config, scope, name)
    output = ""
    if isinstance(task, dict):
        for dep in task.get("depends-on", []) or []:
            dep_name = dep["task"] if isinstance(dep, dict) else dep
            code, dep_output = _run_task(config, scope, dep_name, cwd, env)
            output += dep_output
            if code != 0:
                return code, output
        cmd = task.get("cmd", "")
        task_env = {key: str(value) for key, value in (task.get("env") or {}).items()}
    else:
        cmd = task
        task_env = {}
    if isinstance(cmd, list):
        cmd = " ".join(cmd)
    if not cmd:
        return 0, output
    result = subprocess.run(
        ["bash", "-c", cmd],
        cwd=cwd,
        env={**env, **task_env},
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.returncode, output + result.stdout + result.stderr


class _Sandbox:
    """Throw-away upstream + local clone, with git/pip instrumentation on PATH."""

    def __init__(self, root: Path):
        self.root = root
        self.upstream = root / "upstream"
        self.local = root / "work" / "champollion_pipeline"
        self.git_log = root / "git-calls.log"
        self.pip_log = root / "pip-calls.log"
        home = root / "home"
        home.mkdir()
        bin_dir = root / "bin"
        bin_dir.mkdir()

        git_wrapper = bin_dir / "git"
        git_wrapper.write_text(
            "#!/bin/sh\n"
            'printf \'%s\\t%s\\n\' "${GIT_TERMINAL_PROMPT-<unset>}" "$*" >> "$CHAMP_TEST_GIT_LOG"\n'
            f'exec "{REAL_GIT}" "$@"\n'
        )
        pip_stub = '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$CHAMP_TEST_PIP_LOG"\nexit 0\n'
        for executable, body in ((git_wrapper, None), (bin_dir / "pip", pip_stub), (bin_dir / "pip3", pip_stub)):
            if body is not None:
                executable.write_text(body)
            executable.chmod(0o755)

        base_env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        identity = {
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
            "GIT_CONFIG_NOSYSTEM": "1",
        }
        self.setup_env = {**base_env, **identity, "HOME": str(home)}
        self.task_env = {
            **self.setup_env,
            "PATH": f"{bin_dir}{os.pathsep}{base_env.get('PATH', '')}",
            "PIXI_PROJECT_ROOT": str(self.local),
            "CHAMP_TEST_GIT_LOG": str(self.git_log),
            "CHAMP_TEST_PIP_LOG": str(self.pip_log),
        }

    def git(self, cwd: Path, *args: str) -> str:
        result = subprocess.run(
            [REAL_GIT, *args], cwd=cwd, env=self.setup_env, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    def _seed_tree(self, repo: Path, marker: str) -> None:
        shutil.copy2(PIXI_TOML, repo / "pixi.toml")
        (repo / "pixi.lock").write_text(f"version: 7\n# {marker}\n")
        (repo / "README.md").write_text(f"{marker}\n")
        scripts = repo / "scripts"
        scripts.mkdir(exist_ok=True)
        for entry in SCRIPTS_DIR.iterdir():
            if entry.is_file() and entry.suffix in {".sh", ".py"}:
                shutil.copy2(entry, scripts / entry.name)

    def create(self) -> None:
        self.upstream.mkdir()
        self.git(self.upstream, "init", "-q", "-b", "main")
        self._seed_tree(self.upstream, "original history")
        self.git(self.upstream, "add", "-A")
        self.git(self.upstream, "commit", "-q", "-m", "Initial commit")
        self.local.parent.mkdir(parents=True)
        self.git(self.root, "clone", "-q", str(self.upstream), str(self.local))

    def advance_upstream(self) -> None:
        """Related history: one new commit on top of what the clone has."""
        (self.upstream / "README.md").write_text("upstream change\n")
        self.git(self.upstream, "commit", "-q", "-am", "Upstream change")

    def rewrite_upstream(self) -> None:
        """Simulate a force-push that replaced main with an unrelated root commit."""
        self.git(self.upstream, "checkout", "-q", "--orphan", "rewritten")
        self._seed_tree(self.upstream, "rewritten history")
        self.git(self.upstream, "add", "-A")
        self.git(self.upstream, "commit", "-q", "-m", "Rewritten root commit")
        self.git(self.upstream, "branch", "-M", "main")

    def head(self) -> str:
        return self.git(self.local, "rev-parse", "HEAD")

    def git_calls(self) -> list[tuple[str, str]]:
        if not self.git_log.exists():
            return []
        calls = []
        for line in self.git_log.read_text().splitlines():
            prompt, _, args = line.partition("\t")
            calls.append((prompt, args))
        return calls

    def pip_calls(self) -> list[str]:
        return self.pip_log.read_text().splitlines() if self.pip_log.exists() else []


@pytest.fixture
def sandbox(tmp_path) -> _Sandbox:
    box = _Sandbox(tmp_path)
    box.create()
    return box


@pytest.mark.integration
class TestUnrelatedHistoryStopsUpdate:
    """REQ-UPDATE-01: unrelated upstream history stops the update before any change."""

    @pytest.mark.parametrize("task", HISTORY_GUARDED_TASKS, ids=_task_id)
    def test_exits_non_zero_without_moving_head_or_installing(self, pixi_config, sandbox, task):
        sandbox.rewrite_upstream()
        head_before = sandbox.head()

        code, output = _run_task(pixi_config, *task, cwd=sandbox.local, env=sandbox.task_env)

        assert code != 0, (
            f"REQ-UPDATE-01: {_task_id(task)} exited 0 although origin/main shares no merge base "
            f"with HEAD.\nOutput:\n{output}"
        )
        assert sandbox.head() == head_before, f"REQ-UPDATE-01: {_task_id(task)} moved HEAD on unrelated history"
        assert sandbox.pip_calls() == [], (
            f"REQ-UPDATE-01: {_task_id(task)} invoked pip after detecting unrelated history: {sandbox.pip_calls()}"
        )


@pytest.mark.integration
class TestUnrelatedHistoryRecoveryMessage:
    """REQ-UPDATE-02: unrelated upstream history yields an explanation plus recovery steps."""

    @pytest.mark.parametrize("task", HISTORY_GUARDED_TASKS, ids=_task_id)
    def test_message_names_problem_and_recovery_steps(self, pixi_config, sandbox, task):
        sandbox.rewrite_upstream()

        _, output = _run_task(pixi_config, *task, cwd=sandbox.local, env=sandbox.task_env)

        missing = []
        if "unrelated" not in output.lower():
            missing.append("the word 'unrelated'")
        if not re.search(r"back ?up", output, re.IGNORECASE):
            missing.append("an instruction to back up local changes")
        if "git reset --hard origin/main" not in output:
            missing.append("'git reset --hard origin/main'")
        if "git clone" not in output:
            missing.append("'git clone'")
        assert not missing, (
            f"REQ-UPDATE-02: {_task_id(task)} output on unrelated history lacks {', '.join(missing)}.\n"
            f"Output:\n{output}"
        )


@pytest.mark.integration
class TestGitNeverPromptsForCredentials:
    """REQ-UPDATE-03: every git process started by an update task has GIT_TERMINAL_PROMPT=0."""

    @pytest.mark.parametrize("task", NO_PROMPT_TASKS, ids=_task_id)
    def test_every_git_call_disables_terminal_prompt(self, pixi_config, sandbox, task):
        sandbox.advance_upstream()

        _run_task(pixi_config, *task, cwd=sandbox.local, env=sandbox.task_env)

        calls = sandbox.git_calls()
        assert calls, f"{_task_id(task)} started no git process; the instrumentation saw nothing to check"
        prompting = [args for prompt, args in calls if prompt != "0"]
        assert not prompting, (
            f"REQ-UPDATE-03: {_task_id(task)} ran git without GIT_TERMINAL_PROMPT=0: {prompting}\n"
            f"All calls (prompt value, args): {calls}"
        )


@pytest.mark.integration
class TestSubmoduleSyncBeforeUpdate:
    """REQ-UPDATE-04: submodule URLs are re-synced before submodules are updated."""

    @pytest.mark.parametrize("task", SUBMODULE_SYNC_TASKS, ids=_task_id)
    def test_sync_recursive_precedes_first_submodule_update(self, pixi_config, sandbox, task):
        sandbox.advance_upstream()

        _run_task(pixi_config, *task, cwd=sandbox.local, env=sandbox.task_env)

        args_in_order = [args for _, args in sandbox.git_calls()]
        update_positions = [i for i, args in enumerate(args_in_order) if re.search(r"\bsubmodule update\b", args)]
        sync_positions = [
            i
            for i, args in enumerate(args_in_order)
            if re.search(r"\bsubmodule sync\b", args) and "--recursive" in args.split()
        ]
        assert update_positions, f"{_task_id(task)} ran no 'git submodule update'; calls: {args_in_order}"
        assert sync_positions and sync_positions[0] < update_positions[0], (
            f"REQ-UPDATE-04: {_task_id(task)} did not run 'git submodule sync --recursive' before its first "
            f"'git submodule update'. git calls in order: {args_in_order}"
        )
