"""Tests for REQ-SUBMOD-03 — routine pixi update tasks keep submodules pinned.

The artifact under test is ``pixi.toml`` itself. The requirement states that
the ``update`` task under ``[feature.embeddings.tasks]`` and the
``update-submodules`` task under ``[tasks]`` shall not pass ``--remote`` to any
``git submodule update`` invocation, so each submodule is checked out at the
commit recorded in the parent repository rather than at the unpinned tip of
its upstream branch.

These tests complement, and do not replace, REQ-SUBMOD-02's guards in
``tests/test_pixi_tasks.py`` (``--force`` required, ``--rebase`` forbidden):
``--force`` without ``--remote`` still checks out the pinned commit.
"""

from pathlib import Path

import pytest
import tomllib

PIXI_TOML = Path(__file__).resolve().parents[1] / "pixi.toml"

CHAMPOLLION_SUBMODULE = "external/champollion_V1"
CORTICAL_TILES_SUBMODULE = "external/cortical_tiles"


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


def _submodule_update_fragments(command: str) -> list[str]:
    """Every ``git submodule update`` fragment in a multi-line ``&&`` task body."""
    fragments: list[str] = []
    for line in command.splitlines():
        fragments.extend(line.split("&&"))
    return [fragment.strip() for fragment in fragments if "git submodule update" in fragment]


@pytest.fixture(scope="module")
def embeddings_update_command(pixi_config) -> str:
    """The ``update`` task defined under ``[feature.embeddings.tasks]``."""
    tasks = pixi_config["feature"]["embeddings"]["tasks"]
    assert "update" in tasks, "pixi.toml no longer defines an 'update' task under [feature.embeddings.tasks]"
    return _task_command(tasks["update"])


@pytest.fixture(scope="module")
def update_submodules_command(pixi_config) -> str:
    """The ``update-submodules`` task defined under ``[tasks]``."""
    tasks = pixi_config["tasks"]
    assert "update-submodules" in tasks, "pixi.toml no longer defines an 'update-submodules' task under [tasks]"
    return _task_command(tasks["update-submodules"])


@pytest.mark.smoke
class TestRoutineUpdateTasksKeepSubmodulesPinned:
    """REQ-SUBMOD-03: routine update tasks never pass --remote to git submodule update."""

    def test_embeddings_update_task_does_not_pull_upstream_tip(self, embeddings_update_command):
        """The embeddings 'update' task checks champollion_V1 out at its pinned commit."""
        fragments = _submodule_update_fragments(embeddings_update_command)
        assert fragments, "the 'update' task under [feature.embeddings.tasks] has no 'git submodule update' command"
        offending = [fragment for fragment in fragments if "--remote" in fragment]
        assert not offending, (
            "REQ-SUBMOD-03: the 'update' task under [feature.embeddings.tasks] passes --remote, "
            f"pulling the unpinned upstream tip instead of the recorded commit: {offending}"
        )

    @pytest.mark.parametrize("submodule", [CHAMPOLLION_SUBMODULE, CORTICAL_TILES_SUBMODULE])
    def test_update_submodules_task_does_not_pull_upstream_tip(self, update_submodules_command, submodule):
        """'update-submodules' checks each declared submodule out at its pinned commit."""
        fragments = [f for f in _submodule_update_fragments(update_submodules_command) if submodule in f]
        assert fragments, f"the 'update-submodules' task has no 'git submodule update' command for {submodule}"
        offending = [fragment for fragment in fragments if "--remote" in fragment]
        assert not offending, (
            f"REQ-SUBMOD-03: the 'update-submodules' task passes --remote for {submodule}, "
            f"pulling the unpinned upstream tip instead of the recorded commit: {offending}"
        )
