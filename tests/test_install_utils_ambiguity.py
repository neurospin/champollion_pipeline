"""Tests for REQ-PIXICFG-03 — install-utils task ambiguity in the default environment.

The artifact under test is ``pixi.toml`` itself. ``pixi run install-all``
(the exact command README.md's Setup section instructs every new user to
run) fails on a fresh clone with:

    Error:   x the task 'install-utils' is ambiguous
      help: These environments provide the task 'install-utils': default,
            embeddings, training, cortical-tiles, brainvisa

Root cause: ``install-utils`` is defined three times — once in the shared
``[tasks]`` table (always active, since pixi implicitly includes the
default/base feature unless ``no-default-feature = true``), once in
``[feature.brainvisa.tasks]``, and once in ``[feature.embeddings.tasks]``.
The ``default`` environment (``features = ["brainvisa", "cortical-tiles",
"embeddings"]``) does not set ``no-default-feature = true``, so it
implicitly also carries the base feature — meaning it sees all three
``install-utils`` bodies simultaneously and pixi refuses to pick one.

This is checked statically (parsing ``pixi.toml`` with ``tomllib``) rather
than by shelling out to a real ``pixi run``, so the test stays fast and
does not depend on a pixi installation or environment solve being
available in CI. It reproduces the ambiguity as pixi itself derives it:
count how many task tables reachable by the ``default`` environment define
an ``install-utils`` key.
"""

from pathlib import Path

import pytest
import tomllib

PIXI_TOML = Path(__file__).resolve().parents[1] / "pixi.toml"

TASK_KEY = "install-utils"


@pytest.fixture(scope="module")
def pixi_config() -> dict:
    """Parsed contents of the pipeline's ``pixi.toml``."""
    with PIXI_TOML.open("rb") as handle:
        return tomllib.load(handle)


def _reachable_task_tables(pixi_config: dict, environment: str) -> list[dict]:
    """Every task table pixi consults when resolving a task name for ``environment``.

    Mirrors pixi's own environment/task resolution: the shared ``[tasks]``
    table is included unless the environment sets
    ``no-default-feature = true``, plus each feature's own ``[feature.<f>.tasks]``
    table for every feature listed in ``features``.
    """
    env_spec = pixi_config["environments"][environment]
    tables = []

    if not env_spec.get("no-default-feature", False):
        tables.append(pixi_config.get("tasks", {}))

    for feature_name in env_spec.get("features", []):
        feature_tasks = pixi_config.get("feature", {}).get(feature_name, {}).get("tasks", {})
        tables.append(feature_tasks)

    return tables


def _definitions_of(pixi_config: dict, environment: str, task_key: str) -> list[dict]:
    """Task tables reachable by ``environment`` that define ``task_key``."""
    return [table for table in _reachable_task_tables(pixi_config, environment) if task_key in table]


class TestInstallUtilsAmbiguity:
    """REQ-PIXICFG-03: install-utils must resolve unambiguously in every environment."""

    def test_install_utils_resolves_unambiguously_in_default_environment(self, pixi_config):
        """REQ-PIXICFG-03: `install-utils` must resolve to exactly one body in `default`.

        This is the actual requirement assertion: `pixi run install-all`
        (which depends on `install-utils`) must not hit pixi's "is
        ambiguous" refusal in the `default` environment.
        """
        definitions = _definitions_of(pixi_config, "default", TASK_KEY)
        assert len(definitions) <= 1, (
            f"'{TASK_KEY}' is defined in {len(definitions)} task tables reachable "
            f"by the 'default' environment; pixi will refuse to run it as "
            f"ambiguous. Reachable definitions: {definitions}"
        )

    @pytest.mark.parametrize("environment", ["default", "embeddings", "training", "cortical-tiles", "brainvisa"])
    def test_install_utils_resolves_unambiguously_in_every_environment(self, pixi_config, environment):
        """No environment that can reach `install-utils` may see more than one definition."""
        definitions = _definitions_of(pixi_config, environment, TASK_KEY)
        assert len(definitions) <= 1, (
            f"'{TASK_KEY}' is defined in {len(definitions)} task tables reachable "
            f"by the '{environment}' environment; pixi will refuse to run it as "
            f"ambiguous. Reachable definitions: {definitions}"
        )

    def test_embeddings_environment_keeps_no_build_isolation(self, pixi_config):
        """Non-regression guard: whatever fix lands must not drop --no-build-isolation
        from the body that actually reaches the `embeddings`/`training` environments —
        those environments set `no-default-feature = true` and therefore lack the base
        [dependencies] hatchling/editables build backend (TASK-051), so
        --no-build-isolation is load-bearing there.
        """
        definitions = _definitions_of(pixi_config, "embeddings", TASK_KEY)
        assert definitions, "no install-utils definition reaches the 'embeddings' environment at all"
        bodies = []
        for table in definitions:
            task = table[TASK_KEY]
            body = task if isinstance(task, str) else task.get("cmd", "")
            bodies.append(body)
        assert any("--no-build-isolation" in body for body in bodies), (
            "none of the install-utils definitions reachable by 'embeddings' retain "
            "--no-build-isolation; TASK-051's fix for Jeff's ModuleNotFoundError "
            "regression depends on this flag being present there"
        )
