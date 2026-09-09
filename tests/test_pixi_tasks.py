"""Tests for REQ-SUBMOD-02 — submodule update flags in pixi tasks.

The artifact under test is ``pixi.toml`` itself. The requirement states that
the ``update`` and ``update-submodules`` tasks shall invoke
``git submodule update`` with ``--force`` and never with ``--rebase``, so a
submodule whose checkout has diverged from the remote is reset rather than
stopping the task on a rebase/merge conflict.

These are regression guards for the fix in commit 480bdf0: they are expected
to pass against the current manifest and to fail if ``--rebase`` ever comes
back.

Note that ``pixi.toml`` defines two tasks named ``update``: a plain-string one
under ``[feature.embeddings.tasks]`` that carries the submodule invocation, and
a ``depends-on`` aggregate under ``[tasks]`` that carries no command of its
own. The requirement targets the former.
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
    """Return a task's shell command, whether it is a string or a table.

    A pixi task is either a bare command string or a table with a ``cmd`` key
    (possibly alongside ``depends-on``). Tasks that only declare dependencies
    have no command of their own and yield an empty string.
    """
    if isinstance(task, str):
        return task
    if isinstance(task, dict):
        return task.get("cmd", "")
    raise TypeError(f"unexpected pixi task type: {type(task)!r}")


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


def _submodule_update_lines(command: str, submodule: str) -> list[str]:
    """Every ``git submodule update`` fragment naming ``submodule``.

    The task bodies are multi-line ``&&``-joined shell strings, so a fragment
    is isolated by splitting on both newlines and ``&&`` before matching. This
    keeps an assertion about one submodule from being satisfied by a flag that
    belongs to another command in the same task.
    """
    fragments: list[str] = []
    for line in command.splitlines():
        fragments.extend(line.split("&&"))
    return [fragment.strip() for fragment in fragments if "git submodule update" in fragment and submodule in fragment]


@pytest.mark.smoke
class TestUpdateTaskSubmoduleFlags:
    """REQ-SUBMOD-02: pixi update tasks force submodule updates, never rebase."""

    def test_pixi_toml_exists(self):
        """The manifest under test is present at the pipeline root."""
        assert PIXI_TOML.is_file(), f"{PIXI_TOML} not found"

    def test_embeddings_update_task_does_not_use_rebase(self, embeddings_update_command):
        """The embeddings 'update' task never passes --rebase."""
        assert "--rebase" not in embeddings_update_command, (
            "the 'update' task under [feature.embeddings.tasks] uses --rebase; "
            "REQ-SUBMOD-02 requires --force so a diverged submodule is reset "
            "instead of aborting the update"
        )

    def test_embeddings_update_task_forces_champollion_submodule(self, embeddings_update_command):
        """The embeddings 'update' task updates champollion_V1 with --force."""
        lines = _submodule_update_lines(embeddings_update_command, CHAMPOLLION_SUBMODULE)
        assert lines, (
            "the 'update' task under [feature.embeddings.tasks] has no "
            f"'git submodule update' command for {CHAMPOLLION_SUBMODULE}"
        )
        for line in lines:
            assert "--force" in line, f"submodule update for {CHAMPOLLION_SUBMODULE} lacks --force: {line}"

    def test_update_submodules_task_does_not_use_rebase(self, update_submodules_command):
        """The 'update-submodules' task never passes --rebase."""
        assert "--rebase" not in update_submodules_command, (
            "the 'update-submodules' task uses --rebase; REQ-SUBMOD-02 requires --force"
        )

    @pytest.mark.parametrize("submodule", [CHAMPOLLION_SUBMODULE, CORTICAL_TILES_SUBMODULE])
    def test_update_submodules_task_forces_each_submodule(self, update_submodules_command, submodule):
        """'update-submodules' uses --force for both declared submodules."""
        lines = _submodule_update_lines(update_submodules_command, submodule)
        assert lines, f"the 'update-submodules' task has no 'git submodule update' command for {submodule}"
        for line in lines:
            assert "--force" in line, f"submodule update for {submodule} lacks --force: {line}"
