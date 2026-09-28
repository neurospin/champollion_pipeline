"""Tests for REQ-DEADCODE-01 — a pixi task that scans for dead Python code.

The artifact under test is ``pixi.toml`` itself. The requirement states that
the ``[tasks]`` table shall define a ``dead-code`` task that runs ``vulture``
against ``src/`` and ``tests/``, with ``vulture`` declared under
``[dependencies]`` so the task runs in the default pixi environment (the same
place the existing ``lint``/``format`` tasks and ``ruff`` live).

These tests parse the manifest only; they do not execute vulture, so they
need no network access and no solved environment.
"""

import shlex
from pathlib import Path

import pytest
import tomllib

PIXI_TOML = Path(__file__).resolve().parents[1] / "pixi.toml"

TASK_NAME = "dead-code"
TOOL = "vulture"
SCANNED_PATHS = ("src/", "tests/")


@pytest.fixture(scope="module")
def pixi_config() -> dict:
    """Parsed contents of the pipeline's ``pixi.toml``."""
    with PIXI_TOML.open("rb") as handle:
        return tomllib.load(handle)


def _task_command(task) -> str:
    """Return a task's shell command, whether it is a string or a table."""
    if isinstance(task, str):
        return task
    if isinstance(task, dict):
        return task.get("cmd", "")
    raise TypeError(f"unexpected pixi task type: {type(task)!r}")


def _normalise_path(token: str) -> str:
    """Treat ``src`` and ``src/`` (and ``./src``) as the same scan target."""
    token = token.removeprefix("./")
    return token if token.endswith("/") else f"{token}/"


@pytest.mark.smoke
class TestDeadCodeTask:
    """REQ-DEADCODE-01: ``pixi run dead-code`` runs vulture on src/ and tests/."""

    def test_vulture_is_a_default_environment_dependency(self, pixi_config):
        """``vulture`` is declared under the shared ``[dependencies]`` table."""
        dependencies = pixi_config.get("dependencies", {})
        assert TOOL in dependencies, (
            f"REQ-DEADCODE-01: '{TOOL}' is not declared under [dependencies] in pixi.toml, "
            "so the default environment cannot run it"
        )

    def test_dead_code_task_is_defined(self, pixi_config):
        """A ``dead-code`` task exists under the shared ``[tasks]`` table."""
        tasks = pixi_config.get("tasks", {})
        assert TASK_NAME in tasks, f"REQ-DEADCODE-01: pixi.toml defines no '{TASK_NAME}' task under [tasks]"

    def test_dead_code_task_runs_vulture_on_src_and_tests(self, pixi_config):
        """The ``dead-code`` task invokes vulture with both src/ and tests/ as targets."""
        tasks = pixi_config.get("tasks", {})
        assert TASK_NAME in tasks, f"REQ-DEADCODE-01: pixi.toml defines no '{TASK_NAME}' task under [tasks]"
        tokens = shlex.split(_task_command(tasks[TASK_NAME]))
        if tokens[:3] == ["python", "-m", TOOL]:
            arguments = tokens[3:]
        elif tokens[:1] == [TOOL]:
            arguments = tokens[1:]
        else:
            pytest.fail(f"REQ-DEADCODE-01: '{TASK_NAME}' task does not invoke '{TOOL}' (tokens: {tokens!r})")
        targets = {_normalise_path(token) for token in arguments if not token.startswith("-")}
        missing = [path for path in SCANNED_PATHS if path not in targets]
        assert not missing, f"REQ-DEADCODE-01: '{TASK_NAME}' task does not scan {missing!r} (tokens: {tokens!r})"
