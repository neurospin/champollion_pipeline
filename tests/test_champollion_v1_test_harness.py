"""Tests for the champollion_V1 test-harness wiring (TASK-100, epic TASK-099).

Pipeline-side requirements:

- REQ-CHAMPTEST-01 — ``pixi.toml`` defines a ``test-champollion`` task,
  available in the ``default`` environment (the only one with champollion_V1
  installed editable), that runs pytest on ``external/champollion_V1/test``
  with coverage measured on the ``champollion`` package and reported to the
  terminal.
- REQ-CHAMPTEST-03 — the champollion_V1 checkout carries no pixi manifest or
  lock file (user constraint, 2026-09-30: the runner lives in this repo's
  ``pixi.toml``, never upstream).
- REQ-CHAMPTEST-04 — the pipeline's own ``test`` task keeps collecting only
  ``tests/`` (adding the upstream harness must not leak into it).

Coverage configuration of the task (TASK-105):

- REQ-CHAMPTEST-21 — ``test-champollion`` passes ``--cov-config`` naming a
  dedicated pipeline file that coverage.py never reads by default, so other
  coverage runs (``test-cov --cov=src``) are unaffected.
- REQ-CHAMPTEST-22 — that file omits exactly the two dead-code globs
  (decision pending in TASK-120).
- REQ-CHAMPTEST-23 — that file enables multiprocessing collection, so lines run
  in forked DataLoader workers are counted.
- REQ-CHAMPTEST-24 — ``test-champollion`` passes ``--cov-fail-under`` with an
  integer between 61 and 100.

REQ-CHAMPTEST-02 (per-test cwd isolation) is verified upstream in
``external/champollion_V1/test/test_harness_isolation.py``.

Offline: only parses ``pixi.toml``/``pyproject.toml`` and inspects the
submodule working tree.
"""

import configparser
import shlex
from pathlib import Path

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
PIXI_TOML = REPO_ROOT / "pixi.toml"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"
CHAMPOLLION_V1 = REPO_ROOT / "external" / "champollion_V1"

TASK_NAME = "test-champollion"
UPSTREAM_TEST_DIR = "external/champollion_V1/test"
COVERAGE_TARGETS = {"champollion", "external/champollion_V1/champollion"}

# Files coverage.py reads on its own when no --rcfile/--cov-config is given.
COVERAGE_DEFAULT_CONFIG_FILES = {".coveragerc", "setup.cfg", "tox.ini", "pyproject.toml"}
DEAD_CODE_OMITS = {
    "*/champollion/config_manager/*",
    "*/champollion/utils/create_dataset_config_files.py",
}
# Total champollion coverage with the dead-code omits but without worker
# collection, measured 2026-09-30 on champollion_V1 WIP f16990d2: 61.3%.
# A lower threshold would not protect what TASK-101..TASK-104 added.
MIN_FAIL_UNDER = 61


@pytest.fixture(scope="module")
def pixi_config() -> dict:
    with PIXI_TOML.open("rb") as handle:
        return tomllib.load(handle)


def _task_command(task) -> str:
    if isinstance(task, str):
        return task
    if isinstance(task, dict):
        return task.get("cmd", "")
    raise TypeError(f"unexpected pixi task type: {type(task)!r}")


def _tasks_visible_in(config: dict, env: str) -> dict:
    """Tasks pixi exposes in ``env``: workspace ``[tasks]`` plus its features' tasks."""
    env_spec = config["environments"][env]
    features = env_spec["features"] if isinstance(env_spec, dict) else env_spec
    tasks = dict(config.get("tasks", {}))
    for feature in features:
        tasks.update(config.get("feature", {}).get(feature, {}).get("tasks", {}))
    return tasks


def _pytest_argv(command: str) -> list[str]:
    """Return the argv after the ``pytest`` executable in a task command."""
    tokens = shlex.split(command)
    for i, token in enumerate(tokens):
        if token == "pytest" or token.endswith("/pytest"):
            return tokens[i + 1 :]
        if token == "-m" and i + 1 < len(tokens) and tokens[i + 1] == "pytest":
            return tokens[i + 2 :]
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
    takes_value = {
        "--cov",
        "--cov-report",
        "--cov-config",
        "--cov-fail-under",
        "-m",
        "-k",
        "-c",
        "--rootdir",
        "-p",
        "--basetemp",
    }
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


def _champollion_test_argv(pixi_config) -> list[str]:
    """pytest argv of the ``test-champollion`` task (fails the calling test if absent)."""
    tasks = _tasks_visible_in(pixi_config, "default")
    assert TASK_NAME in tasks, (
        f"pixi.toml defines no {TASK_NAME!r} task visible in the default environment "
        "(REQ-CHAMPTEST-01: the champollion_V1 test runner lives in the pipeline's pixi.toml)"
    )
    return _pytest_argv(_task_command(tasks[TASK_NAME]))


@pytest.mark.smoke
class TestChampollionTestTask:
    """REQ-CHAMPTEST-01."""

    def test_task_targets_upstream_test_directory(self, pixi_config):
        champollion_test_argv = _champollion_test_argv(pixi_config)
        paths = [p.rstrip("/") for p in _positional_paths(champollion_test_argv)]
        assert paths == [UPSTREAM_TEST_DIR], (
            f"{TASK_NAME} must run pytest on exactly {UPSTREAM_TEST_DIR!r}; got positional args {paths!r}"
        )

    def test_task_measures_coverage_on_champollion_package(self, pixi_config):
        champollion_test_argv = _champollion_test_argv(pixi_config)
        targets = {t.rstrip("/") for t in _option_values(champollion_test_argv, "--cov")}
        assert targets and targets <= COVERAGE_TARGETS, (
            f"{TASK_NAME} must pass --cov for the champollion package only "
            f"(one of {sorted(COVERAGE_TARGETS)}); got {sorted(targets)!r}"
        )

    def test_task_reports_coverage_to_terminal(self, pixi_config):
        champollion_test_argv = _champollion_test_argv(pixi_config)
        reports = _option_values(champollion_test_argv, "--cov-report")
        # pytest-cov prints a terminal report by default when no --cov-report is given.
        assert not reports or any(r.split(":", 1)[0] in {"term", "term-missing"} for r in reports), (
            f"{TASK_NAME} must report coverage to the terminal; got --cov-report {reports!r}"
        )


@pytest.mark.smoke
class TestNoPixiConfigInChampollionV1:
    """REQ-CHAMPTEST-03 — encodes the user constraint; passes today by design."""

    @pytest.fixture(autouse=True)
    def _require_checkout(self):
        if not (CHAMPOLLION_V1 / "setup.py").exists():
            pytest.skip("external/champollion_V1 is not checked out")

    @pytest.mark.parametrize("name", ["pixi.toml", "pixi.lock"])
    def test_no_pixi_file_at_submodule_root(self, name):
        assert not (CHAMPOLLION_V1 / name).exists(), (
            f"external/champollion_V1/{name} must not exist: pixi tasks/config for champollion_V1 "
            "live in champollion_pipeline/pixi.toml"
        )

    def test_no_tool_pixi_table_in_submodule_pyproject(self):
        pyproject = CHAMPOLLION_V1 / "pyproject.toml"
        if not pyproject.exists():
            return
        with pyproject.open("rb") as handle:
            tool = tomllib.load(handle).get("tool", {})
        assert "pixi" not in tool, "external/champollion_V1/pyproject.toml must not carry a [tool.pixi] table"


@pytest.mark.smoke
class TestPipelineTestTaskScope:
    """REQ-CHAMPTEST-04 — regression guard; passes today by design."""

    def test_test_task_runs_pytest_on_tests_dir_only(self, pixi_config):
        argv = _pytest_argv(_task_command(pixi_config["tasks"]["test"]))
        paths = [p.rstrip("/") for p in _positional_paths(argv)]
        assert paths == ["tests"], f"pixi task 'test' must collect only tests/; got positional args {paths!r}"

    def test_pytest_testpaths_is_tests_only(self):
        with PYPROJECT_TOML.open("rb") as handle:
            ini = tomllib.load(handle)["tool"]["pytest"]["ini_options"]
        assert ini.get("testpaths") == ["tests"]


def _champollion_cov_config_path(pixi_config) -> Path:
    """The single ``--cov-config`` file of ``test-champollion``, resolved from the pipeline root."""
    values = _option_values(_champollion_test_argv(pixi_config), "--cov-config")
    assert len(values) == 1, f"{TASK_NAME} must pass exactly one --cov-config (REQ-CHAMPTEST-21); got {values!r}"
    return (REPO_ROOT / values[0]).resolve()


def _champollion_cov_run_section(pixi_config) -> configparser.SectionProxy:
    path = _champollion_cov_config_path(pixi_config)
    assert path.is_file(), f"--cov-config file {path} does not exist"
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    assert parser.has_section("run"), f"{path.name} has no [run] section"
    return parser["run"]


def _list_setting(value: str) -> list[str]:
    """coverage.py list option: comma- and/or newline-separated."""
    return [item.strip() for line in value.splitlines() for item in line.split(",") if item.strip()]


@pytest.mark.smoke
class TestChampollionCoverageConfigFile:
    """REQ-CHAMPTEST-21."""

    def test_task_passes_a_dedicated_cov_config_inside_the_pipeline(self, pixi_config):
        path = _champollion_cov_config_path(pixi_config)
        assert path.is_file(), f"--cov-config file {path} does not exist"
        rel = path.relative_to(REPO_ROOT)  # raises if outside the pipeline repo
        assert rel.parts[0] != "external", (
            f"--cov-config must live in champollion_pipeline, not in a submodule; got {rel}"
        )
        assert path.name not in COVERAGE_DEFAULT_CONFIG_FILES, (
            f"--cov-config must not be a file coverage.py reads by default "
            f"({sorted(COVERAGE_DEFAULT_CONFIG_FILES)}), or it would also apply to test-cov; got {rel}"
        )


@pytest.mark.smoke
class TestChampollionCoverageOmits:
    """REQ-CHAMPTEST-22."""

    def test_cov_config_omits_exactly_the_dead_code_modules(self, pixi_config):
        run = _champollion_cov_run_section(pixi_config)
        omits = _list_setting(run.get("omit", ""))
        assert sorted(omits) == sorted(DEAD_CODE_OMITS), (
            f"[run] omit must be exactly {sorted(DEAD_CODE_OMITS)} (TASK-120 dead code); got {omits!r}"
        )


@pytest.mark.smoke
class TestChampollionCoverageWorkerCollection:
    """REQ-CHAMPTEST-23."""

    def test_cov_config_collects_multiprocessing_children(self, pixi_config):
        run = _champollion_cov_run_section(pixi_config)
        concurrency = _list_setting(run.get("concurrency", ""))
        assert "multiprocessing" in concurrency, (
            "[run] concurrency must include 'multiprocessing' so forked DataLoader workers are measured; "
            f"got {concurrency!r}"
        )


@pytest.mark.smoke
class TestChampollionCoverageThreshold:
    """REQ-CHAMPTEST-24."""

    def test_task_enforces_an_integer_coverage_floor(self, pixi_config):
        values = _option_values(_champollion_test_argv(pixi_config), "--cov-fail-under")
        assert len(values) == 1, f"{TASK_NAME} must pass exactly one --cov-fail-under; got {values!r}"
        assert values[0].isdigit(), f"--cov-fail-under must be an integer; got {values[0]!r}"
        threshold = int(values[0])
        assert MIN_FAIL_UNDER <= threshold <= 100, (
            f"--cov-fail-under must be between {MIN_FAIL_UNDER} and 100; got {threshold}"
        )
