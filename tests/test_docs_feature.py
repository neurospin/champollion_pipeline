"""Tests for REQ-DOCS-01 — pixi.toml docs feature.

The artifact under test is ``pixi.toml`` itself: the requirement states that
it shall define a ``docs`` feature (with the Sphinx toolchain as
pypi-dependencies and ``docs-build`` / ``docs-serve`` tasks) plus a ``docs``
environment declared with ``no-default-feature = true``.
"""

from pathlib import Path

import pytest
import tomllib

PIXI_TOML = Path(__file__).resolve().parents[1] / "pixi.toml"

REQUIRED_DOCS_DEPENDENCIES = (
    "sphinx",
    "furo",
    "myst-parser",
    "sphinx-autodoc-typehints",
)

REQUIRED_DOCS_TASKS = (
    "docs-build",
    "docs-serve",
)


@pytest.fixture(scope="module")
def pixi_config() -> dict:
    """Parsed contents of the pipeline's ``pixi.toml``."""
    with PIXI_TOML.open("rb") as handle:
        return tomllib.load(handle)


@pytest.mark.smoke
class TestDocsFeature:
    """REQ-DOCS-01: pixi.toml declares the docs feature, tasks and environment."""

    def test_pixi_toml_exists(self, pixi_config):
        """The pixi manifest under test is present and parseable."""
        assert pixi_config, f"{PIXI_TOML} parsed to an empty document"

    def test_docs_feature_block_exists(self, pixi_config):
        """A ``[feature.docs]`` block is defined."""
        assert "docs" in pixi_config.get("feature", {}), "pixi.toml defines no [feature.docs] block"

    @pytest.mark.parametrize("dependency", REQUIRED_DOCS_DEPENDENCIES)
    def test_docs_feature_declares_pypi_dependency(self, pixi_config, dependency):
        """Each Sphinx toolchain package is a pypi-dependency of the docs feature."""
        docs_feature = pixi_config.get("feature", {}).get("docs", {})
        pypi_dependencies = docs_feature.get("pypi-dependencies", {})
        assert dependency in pypi_dependencies, (
            f"[feature.docs.pypi-dependencies] is missing {dependency!r}; found {sorted(pypi_dependencies)}"
        )

    @pytest.mark.parametrize("task_name", REQUIRED_DOCS_TASKS)
    def test_docs_feature_declares_task(self, pixi_config, task_name):
        """Each documentation task is defined under ``[feature.docs.tasks]``."""
        docs_feature = pixi_config.get("feature", {}).get("docs", {})
        tasks = docs_feature.get("tasks", {})
        assert task_name in tasks, f"[feature.docs.tasks] is missing {task_name!r}; found {sorted(tasks)}"

    def test_docs_environment_exists(self, pixi_config):
        """``[environments]`` declares a ``docs`` environment."""
        environments = pixi_config.get("environments", {})
        assert "docs" in environments, f"[environments] defines no 'docs' key; found {sorted(environments)}"

    def test_docs_environment_sets_no_default_feature(self, pixi_config):
        """The ``docs`` environment is isolated via ``no-default-feature = true``."""
        docs_environment = pixi_config.get("environments", {}).get("docs", {})
        assert docs_environment.get("no-default-feature") is True, (
            f"environments.docs must set no-default-feature = true; got {docs_environment!r}"
        )

    def test_docs_environment_includes_docs_feature(self, pixi_config):
        """The ``docs`` environment is built from the ``docs`` feature."""
        docs_environment = pixi_config.get("environments", {}).get("docs", {})
        assert "docs" in docs_environment.get("features", []), (
            f"environments.docs must list 'docs' in its features; got {docs_environment!r}"
        )
