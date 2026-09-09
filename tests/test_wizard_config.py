"""Tests for REQ-WIZARD-01 — pixi.toml setup task, rich dependency, training environment.

The artifact under test is ``pixi.toml`` itself: the requirement states that
it shall declare a ``setup`` task running ``python scripts/setup_wizard.py``,
add ``rich >=13.0`` to ``[pypi-dependencies]``, and add a ``training``
environment built from the ``embeddings`` feature with
``solve-group = "embeddings"``.

The requirement fixes the wizard's *invocation*, not its behaviour — the
``scripts/setup_wizard.py`` module itself belongs to REQ-WIZARD-02, so these
tests assert only on the manifest.
"""

# TASK-009

from pathlib import Path

import pytest
import tomllib

PIXI_TOML = Path(__file__).resolve().parents[1] / "pixi.toml"

WIZARD_SCRIPT = "scripts/setup_wizard.py"


@pytest.fixture(scope="module")
def pixi_config() -> dict:
    """Parsed contents of the pipeline's ``pixi.toml``."""
    with PIXI_TOML.open("rb") as handle:
        return tomllib.load(handle)


@pytest.mark.smoke
class TestWizardConfig:
    """REQ-WIZARD-01: pixi.toml declares the setup task, rich, and the training env."""

    def test_setup_task_exists(self, pixi_config):
        """``[tasks]`` declares a ``setup`` task."""
        tasks = pixi_config.get("tasks", {})
        assert "setup" in tasks, f"[tasks] defines no 'setup' key; found {sorted(tasks)}"

    def test_setup_task_runs_wizard(self, pixi_config):
        """The ``setup`` task invokes the setup wizard script."""
        setup_task = pixi_config.get("tasks", {}).get("setup")
        assert setup_task is not None, "[tasks] defines no 'setup' key, so its command cannot be checked"
        command = setup_task if isinstance(setup_task, str) else setup_task.get("cmd", "")
        assert WIZARD_SCRIPT in command, f"[tasks].setup must run {WIZARD_SCRIPT!r}; got {command!r}"

    def test_rich_in_dependencies(self, pixi_config):
        """``rich`` is declared in [dependencies] or [pypi-dependencies]."""
        deps = pixi_config.get("dependencies", {})
        pypi_deps = pixi_config.get("pypi-dependencies", {})
        assert "rich" in deps or "rich" in pypi_deps, (
            f"'rich' missing from both [dependencies] and [pypi-dependencies]; "
            f"deps={sorted(deps)}, pypi={sorted(pypi_deps)}"
        )

    def test_training_environment_exists(self, pixi_config):
        """``[environments]`` declares a ``training`` environment."""
        environments = pixi_config.get("environments", {})
        assert "training" in environments, (
            f"[environments] defines no 'training' key; found {sorted(environments)}"
        )

    def test_training_env_uses_embeddings_feature(self, pixi_config):
        """The ``training`` environment is built from the ``embeddings`` feature."""
        training_environment = pixi_config.get("environments", {}).get("training", {})
        assert "embeddings" in training_environment.get("features", []), (
            f"environments.training must list 'embeddings' in its features; got {training_environment!r}"
        )

    def test_training_env_solve_group(self, pixi_config):
        """The ``training`` environment shares the ``embeddings`` solve-group."""
        training_environment = pixi_config.get("environments", {}).get("training", {})
        assert training_environment.get("solve-group") == "embeddings", (
            f'environments.training must set solve-group = "embeddings"; got {training_environment!r}'
        )
