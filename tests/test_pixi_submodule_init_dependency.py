"""Tests for REQ-INSTALL-06 — install-champollion initialises submodules first.

REQ-INSTALL-06: In every pixi.toml environment exposing ``install-champollion``,
that task shall depend, directly or transitively, on an ``init-submodules``
task resolvable in that same environment.

Why pixi.toml (fix approach "b") rather than setup_wizard.build_plan() ("a"):
``install-champollion`` runs ``pip install -e external/champollion_V1``, which
fails whenever that submodule directory is still empty. The wizard is only one
caller: README instructions, SLURM scripts and users typing
``pixi run -e embeddings install-embeddings`` by hand hit the same failure.
Declaring the dependency on the task itself covers every caller, and pixi's
``depends-on`` guarantees ordering, which a wizard-side command list does not
enforce for anyone else.

Environment scoping matters: ``embeddings`` and ``training`` are declared with
``no-default-feature = true``, so the shared ``[tasks]`` table (the default
feature, where ``init-submodules`` lives today) is NOT visible there —
confirmed with ``pixi task list -e embeddings`` (pixi 0.80.0). The resolver
below models that, so a fix that only adds a ``depends-on`` pointing at a task
the environment cannot see does not pass.

# TASK-075
"""

from __future__ import annotations

import importlib.util
import shlex
from pathlib import Path

import pytest
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIXI_TOML = PROJECT_ROOT / "pixi.toml"
WIZARD = PROJECT_ROOT / "scripts" / "setup_wizard.py"

INIT_TASK = "init-submodules"
CHAMPOLLION_TASK = "install-champollion"


@pytest.fixture(scope="module")
def pixi_config() -> dict:
    with PIXI_TOML.open("rb") as handle:
        return tomllib.load(handle)


@pytest.fixture(scope="module")
def wizard():
    spec = importlib.util.spec_from_file_location("setup_wizard", WIZARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _env_tasks(config: dict, env: str) -> dict:
    """Tasks visible in ``env``: default-feature ``[tasks]`` (unless the
    environment sets ``no-default-feature``), overlaid by each listed
    feature's own tasks."""
    if env == "default" and env not in config.get("environments", {}):
        env_def: dict = {"features": []}
    else:
        env_def = config["environments"][env]
        if isinstance(env_def, list):
            env_def = {"features": env_def}
    tasks: dict = {}
    if not env_def.get("no-default-feature", False):
        tasks.update(config.get("tasks", {}))
    for feature in env_def.get("features", []):
        tasks.update(config.get("feature", {}).get(feature, {}).get("tasks", {}))
    return tasks


def _depends_on(task) -> list[str]:
    if isinstance(task, dict):
        deps = task.get("depends-on", task.get("depends_on", []))
        if isinstance(deps, str):
            deps = [deps]
        return [d if isinstance(d, str) else d.get("task", "") for d in deps]
    return []


def _closure(tasks: dict, name: str) -> set[str]:
    """Names of every task ``name`` depends on, transitively (excluding
    itself). Unresolvable names are kept so they can be reported, but they
    are not expanded."""
    seen: set[str] = set()
    stack = list(_depends_on(tasks.get(name)))
    while stack:
        dep = stack.pop()
        if dep in seen:
            continue
        seen.add(dep)
        stack.extend(_depends_on(tasks.get(dep)))
    return seen


def _envs_exposing_champollion(config: dict) -> list[str]:
    return [env for env in config.get("environments", {}) if CHAMPOLLION_TASK in _env_tasks(config, env)]


def _all_envs() -> list[str]:
    with PIXI_TOML.open("rb") as handle:
        config = tomllib.load(handle)
    return list(config.get("environments", {}))


@pytest.mark.smoke
class TestInstallChampollionInitialisesSubmodules:
    """REQ-INSTALL-06: pixi.toml task graph."""

    def test_embeddings_and_training_expose_install_champollion(self, pixi_config):
        """Anchor: the environments the bug report names do expose the task,
        so the parametrized check below is not vacuous for them."""
        exposing = _envs_exposing_champollion(pixi_config)
        assert "embeddings" in exposing
        assert "training" in exposing

    @pytest.mark.parametrize("env", _all_envs())
    def test_install_champollion_depends_on_init_submodules(self, pixi_config, env):
        tasks = _env_tasks(pixi_config, env)
        if CHAMPOLLION_TASK not in tasks:
            pytest.skip(f"{CHAMPOLLION_TASK} not exposed in environment {env!r}")
        deps = _closure(tasks, CHAMPOLLION_TASK)
        assert INIT_TASK in deps, (
            f"environment {env!r}: {CHAMPOLLION_TASK} does not depend on "
            f"{INIT_TASK} (transitive depends-on: {sorted(deps) or 'none'}); "
            "external/champollion_V1 may still be empty when pip installs it"
        )
        assert INIT_TASK in tasks, (
            f"environment {env!r}: {CHAMPOLLION_TASK} names {INIT_TASK} in "
            f"depends-on, but no {INIT_TASK} task is visible in this "
            "environment (no-default-feature hides the shared [tasks] table)"
        )


# (location, use_case) pairs from the bug report whose plans call
# install-embeddings without a preceding init-submodules.
AFFECTED_PLANS = [
    ("2", "1"),  # Jean-Zay, full pipeline
    ("1", "2"),  # local, embeddings inference
    ("2", "2"),  # Jean-Zay, embeddings inference
    ("1", "3"),  # local, training
    ("2", "3"),  # Jean-Zay, training
    ("2", "5"),  # Jean-Zay, everything
]


def _parse_pixi_run(cmd: str) -> tuple[str, str] | None:
    """Return (environment, task) for a ``pixi run [-e env] task`` command."""
    argv = shlex.split(cmd)
    if argv[:2] != ["pixi", "run"]:
        return None
    rest = argv[2:]
    env = "default"
    if rest and rest[0] in {"-e", "--environment"}:
        env, rest = rest[1], rest[2:]
    return (env, rest[0]) if rest else None


@pytest.mark.smoke
class TestWizardPlansInitialiseSubmodules:
    """REQ-INSTALL-06 as seen through the wizard's emitted plans: every
    command whose task graph reaches install-champollion also reaches
    init-submodules."""

    @pytest.mark.parametrize(
        "location,use_case",
        AFFECTED_PLANS + [("1", "1"), ("3", "5")],  # install-all paths: control
    )
    def test_plan_reaches_init_submodules(self, pixi_config, wizard, location, use_case):
        plan = wizard.build_plan(location=location, use_case=use_case, gpu=True)
        checked = 0
        for cmd in plan.commands:
            parsed = _parse_pixi_run(cmd)
            if parsed is None:
                continue
            env, task = parsed
            tasks = _env_tasks(pixi_config, env)
            closure = _closure(tasks, task) | {task}
            if CHAMPOLLION_TASK not in closure:
                continue
            checked += 1
            assert INIT_TASK in closure and INIT_TASK in tasks, (
                f"wizard plan (location={location}, use_case={use_case}) runs "
                f"{cmd!r}, which reaches {CHAMPOLLION_TASK} but never "
                f"{INIT_TASK} in environment {env!r}"
            )
        assert checked, "plan never reaches install-champollion; test is vacuous"
