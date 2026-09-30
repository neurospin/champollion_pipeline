"""Tests for the cortical_tiles test-harness wiring (TASK-115, epic TASK-114).

Pipeline-side requirements:

- REQ-CTILESTEST-01 — ``pixi.toml`` defines a ``test-cortical-tiles`` task,
  available in the ``default`` environment (the only one with cortical_tiles
  installed editable and BrainVISA/aims), that runs pytest on
  ``external/cortical_tiles/tests`` with coverage measured on the
  ``cortical_tiles`` package and reported to the terminal.
- REQ-CTILESTEST-02 — that task runs pytest with the working directory set to
  the ``external/cortical_tiles`` checkout root (the upstream tests resolve
  their data paths relative to the cwd, e.g. ``data/mask/1mm/...``).
- REQ-CTILESTEST-03 — the cortical_tiles checkout carries no pixi manifest,
  lock file or ``[tool.pixi]`` table (user constraint, 2026-09-30: the runner
  lives in this repo's ``pixi.toml``, never upstream).

The pipeline ``test`` task scope guard (TASK-115 scope item c) is already
REQ-CHAMPTEST-04, tested in ``tests/test_champollion_v1_test_harness.py``.

Offline: only parses ``pixi.toml``/``pyproject.toml`` and inspects the
submodule working tree.
"""

import shlex
from pathlib import Path

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
PIXI_TOML = REPO_ROOT / "pixi.toml"
CORTICAL_TILES = REPO_ROOT / "external" / "cortical_tiles"

TASK_NAME = "test-cortical-tiles"
UPSTREAM_TEST_DIR = CORTICAL_TILES / "tests"
COVERAGE_PACKAGE = "cortical_tiles"
COVERAGE_PACKAGE_DIR = CORTICAL_TILES / "cortical_tiles"


@pytest.fixture(scope="module")
def pixi_config() -> dict:
    with PIXI_TOML.open("rb") as handle:
        return tomllib.load(handle)


def _tasks_visible_in(config: dict, env: str) -> dict:
    """Tasks pixi exposes in ``env``: workspace ``[tasks]`` plus its features' tasks."""
    env_spec = config["environments"][env]
    features = env_spec["features"] if isinstance(env_spec, dict) else env_spec
    tasks = dict(config.get("tasks", {}))
    for feature in features:
        tasks.update(config.get("feature", {}).get(feature, {}).get("tasks", {}))
    return tasks


def _resolve_invocation(task) -> tuple[Path, list[str]]:
    """Return ``(effective cwd, pytest argv)`` for a pixi task.

    The effective cwd starts at the manifest directory, honours the task
    table's ``cwd`` field, then any leading ``cd <dir> &&`` steps.
    """
    if isinstance(task, str):
        command, cwd = task, REPO_ROOT
    elif isinstance(task, dict):
        command = task.get("cmd", "")
        cwd = (REPO_ROOT / task["cwd"]) if "cwd" in task else REPO_ROOT
    else:
        raise TypeError(f"unexpected pixi task type: {type(task)!r}")

    tokens = shlex.split(command)
    while len(tokens) >= 3 and tokens[0] == "cd" and tokens[2] == "&&":
        cwd = cwd / tokens[1]
        tokens = tokens[3:]

    for i, token in enumerate(tokens):
        if token == "pytest" or token.endswith("/pytest"):
            return cwd.resolve(), tokens[i + 1 :]
        if token == "-m" and i + 1 < len(tokens) and tokens[i + 1] == "pytest":
            return cwd.resolve(), tokens[i + 2 :]
    raise AssertionError(f"task command does not invoke pytest: {command!r}")


def _option_values(argv: list[str], option: str) -> list[str]:
    """Values of ``option`` given as ``--opt=value`` or ``--opt value``."""
    values = []
    for i, token in enumerate(argv):
        if token.startswith(option + "="):
            values.append(token.split("=", 1)[1])
        elif token == option and i + 1 < len(argv):
            values.append(argv[i + 1])
    return values


def _positional_paths(argv: list[str]) -> list[str]:
    """Positional (non-option) arguments, skipping values of options that take one."""
    takes_value = {"--cov", "--cov-report", "--cov-config", "-m", "-k", "-c", "--rootdir", "-p", "--basetemp"}
    paths, skip = [], False
    for token in argv:
        if skip:
            skip = False
            continue
        if token.startswith("-"):
            skip = token in takes_value
            continue
        paths.append(token)
    return paths


def _cortical_tiles_invocation(pixi_config) -> tuple[Path, list[str]]:
    """``(cwd, pytest argv)`` of the ``test-cortical-tiles`` task (fails the calling test if absent)."""
    tasks = _tasks_visible_in(pixi_config, "default")
    assert TASK_NAME in tasks, (
        f"pixi.toml defines no {TASK_NAME!r} task visible in the default environment "
        "(REQ-CTILESTEST-01: the cortical_tiles test runner lives in the pipeline's pixi.toml)"
    )
    return _resolve_invocation(tasks[TASK_NAME])


@pytest.mark.smoke
class TestCorticalTilesTestTask:
    """REQ-CTILESTEST-01."""

    def test_task_targets_upstream_tests_directory(self, pixi_config):
        cwd, argv = _cortical_tiles_invocation(pixi_config)
        paths = [(cwd / p).resolve() for p in _positional_paths(argv)]
        assert paths == [UPSTREAM_TEST_DIR.resolve()], (
            f"{TASK_NAME} must run pytest on exactly external/cortical_tiles/tests; "
            f"got positional args {_positional_paths(argv)!r} from cwd {cwd}"
        )

    def test_task_measures_coverage_on_cortical_tiles_package(self, pixi_config):
        cwd, argv = _cortical_tiles_invocation(pixi_config)
        targets = [t.rstrip("/") for t in _option_values(argv, "--cov")]

        def _is_package(target: str) -> bool:
            return target == COVERAGE_PACKAGE or (cwd / target).resolve() == COVERAGE_PACKAGE_DIR.resolve()

        assert targets and all(_is_package(t) for t in targets), (
            f"{TASK_NAME} must pass --cov for the cortical_tiles package only; got {targets!r} from cwd {cwd}"
        )

    def test_task_reports_coverage_to_terminal(self, pixi_config):
        _, argv = _cortical_tiles_invocation(pixi_config)
        reports = _option_values(argv, "--cov-report")
        # pytest-cov prints a terminal report by default when no --cov-report is given.
        assert not reports or any(r.split(":", 1)[0] in {"term", "term-missing"} for r in reports), (
            f"{TASK_NAME} must report coverage to the terminal; got --cov-report {reports!r}"
        )


@pytest.mark.smoke
class TestCorticalTilesTaskWorkingDirectory:
    """REQ-CTILESTEST-02."""

    def test_task_runs_from_cortical_tiles_checkout_root(self, pixi_config):
        cwd, _ = _cortical_tiles_invocation(pixi_config)
        assert cwd == CORTICAL_TILES.resolve(), (
            f"{TASK_NAME} must run pytest with cwd external/cortical_tiles (task `cwd` field or a "
            f"leading `cd external/cortical_tiles &&`); effective cwd is {cwd}"
        )


@pytest.mark.smoke
class TestNoPixiConfigInCorticalTiles:
    """REQ-CTILESTEST-03 — encodes the user constraint; passes today by design."""

    @pytest.fixture(autouse=True)
    def _require_checkout(self):
        if not (CORTICAL_TILES / "setup.py").exists():
            pytest.skip("external/cortical_tiles is not checked out")

    @pytest.mark.parametrize("name", ["pixi.toml", "pixi.lock"])
    def test_no_pixi_file_at_submodule_root(self, name):
        assert not (CORTICAL_TILES / name).exists(), (
            f"external/cortical_tiles/{name} must not exist: pixi tasks/config for cortical_tiles "
            "live in champollion_pipeline/pixi.toml"
        )

    def test_no_tool_pixi_table_in_submodule_pyproject(self):
        pyproject = CORTICAL_TILES / "pyproject.toml"
        if not pyproject.exists():
            return
        with pyproject.open("rb") as handle:
            tool = tomllib.load(handle).get("tool", {})
        assert "pixi" not in tool, "external/cortical_tiles/pyproject.toml must not carry a [tool.pixi] table"
