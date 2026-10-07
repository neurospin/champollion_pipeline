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

Coverage configuration (TASK-130):

- REQ-CTILESTEST-24 — that task passes ``--cov-config`` naming the
  pipeline-root ``.coveragerc-cortical-tiles`` (resolved from the task cwd).
- REQ-CTILESTEST-139 (supersedes REQ-CTILESTEST-25, TASK-137) — that file's
  ``[run] omit`` is exactly the 22 dead cortical_tiles modules (the 16 of
  REQ-CTILESTEST-25 plus the three extremities modules and three modules
  with no live importer; keep-or-delete pending in TASK-131); the
  distbottom modules are not omitted.
- REQ-CTDEADCODE-3 (supersedes REQ-CTILESTEST-139, TASK-131) — that file's
  ``[run] omit`` equals the REQ-CTILESTEST-139 omit set minus the nine
  modules deleted from cortical_tiles by REQ-CTDEADCODE-1 (user decision
  2026-10-06: delete everything but what's inside brainvisa/utils); the
  seven kept brainvisa/utils modules stay omitted.
- REQ-CTILESTEST-26 — superseded by REQ-COVISO-03 (TASK-125): the shared
  pipeline-root ``.coverage`` is replaced by per-task data files; location,
  naming and gitignore are tested in ``tests/test_coverage_data_file_isolation.py``.

Coverage threshold (TASK-119):

- REQ-CTILESTEST-167 — that task passes ``--cov-fail-under`` exactly once,
  with the integer value 85 (user decision 2026-10-01).

The pipeline ``test`` task scope guard (TASK-115 scope item c) is already
REQ-CHAMPTEST-04, tested in ``tests/test_champollion_v1_test_harness.py``.

Offline: only parses ``pixi.toml``/``pyproject.toml`` and inspects the
submodule working tree.
"""

import configparser
import re
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

COV_CONFIG_FILE = REPO_ROOT / ".coveragerc-cortical-tiles"
# REQ-CTDEADCODE-1 (TASK-131): deleted from cortical_tiles, so no longer omitted.
DELETED_MODULES = (
    "preprocessing/transforms.py",
    "preprocessing/pynet_transforms.py",
    "preprocessing/create_sets.py",
    "preprocessing/datasets.py",
    "brainvisa/benchmark_pipeline.py",
    "brainvisa/put_together_datasets.py",
    "brainvisa/dataset_to_sparse.py",
    "brainvisa/generate_sparse_dataset.py",
    "utils/split_train_test.py",
)
# REQ-CTDEADCODE-3 Omit set: REQ-CTILESTEST-139's 22 minus DELETED_MODULES
# (paths inside the cortical_tiles package).
DEAD_MODULES = (
    # Kept by the TASK-131 user decision (everything inside brainvisa/utils).
    "brainvisa/utils/generate_spam_graph.py",
    "brainvisa/utils/convert_volume_to_bucket.py",
    "brainvisa/utils/display_reconstructions.py",
    "brainvisa/utils/mask_qc.py",
    "brainvisa/utils/write_distance_map.py",
    "brainvisa/utils/generate_spam_sulcal_region.py",
    "brainvisa/utils/suppress_files_from_csv.py",
    # Added by REQ-CTILESTEST-139 (TASK-137): extremities, dead for
    # champollion_V1 since champollion_V1 commit 770a5b74.
    "brainvisa/generate_extremities.py",
    "brainvisa/mask_resampled_extremities.py",
    "brainvisa/utils/skeleton_extremities.py",
    # Added by REQ-CTILESTEST-139 (TASK-137): no live importer.
    "utils/pytorchtools.py",
    "preprocessing/generate_numpy_array.py",
    "utils/save_results.py",
)
DEAD_MODULE_OMITS = tuple(f"*/cortical_tiles/{module}" for module in DEAD_MODULES)
# REQ-CTILESTEST-139: still reachable, so explicitly NOT omitted.
LIVE_DISTBOTTOM_MODULES = (
    "brainvisa/generate_distbottom_crops.py",
    "brainvisa/utils/distbottom.py",
)


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

    @pytest.fixture(autouse=True)  # noqa: V105
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


def _expand_pixi_root(value: str) -> str:
    """Expand ``$PIXI_PROJECT_ROOT``/``${PIXI_PROJECT_ROOT}`` as pixi's task shell does."""
    return re.sub(r"\$\{PIXI_PROJECT_ROOT\}|\$PIXI_PROJECT_ROOT\b", str(REPO_ROOT), value)


def _cortical_tiles_cov_config_path(pixi_config) -> Path:
    """The single ``--cov-config`` of ``test-cortical-tiles``, resolved from the task cwd."""
    cwd, argv = _cortical_tiles_invocation(pixi_config)
    values = _option_values(argv, "--cov-config")
    assert len(values) == 1, f"{TASK_NAME} must pass exactly one --cov-config (REQ-CTILESTEST-24); got {values!r}"
    return (cwd / _expand_pixi_root(values[0])).resolve()


def _cortical_tiles_cov_run_section(pixi_config) -> configparser.SectionProxy:
    path = _cortical_tiles_cov_config_path(pixi_config)
    assert path.is_file(), f"--cov-config file {path} does not exist"
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    assert parser.has_section("run"), f"{path.name} has no [run] section"
    return parser["run"]


def _list_setting(value: str) -> list[str]:
    """coverage.py list option: comma- and/or newline-separated."""
    return [item.strip() for line in value.splitlines() for item in line.split(",") if item.strip()]


@pytest.mark.smoke
class TestCorticalTilesCoverageConfigFile:
    """REQ-CTILESTEST-24."""

    def test_task_passes_pipeline_root_coveragerc_cortical_tiles(self, pixi_config):
        path = _cortical_tiles_cov_config_path(pixi_config)
        assert path == COV_CONFIG_FILE.resolve(), (
            f"{TASK_NAME} --cov-config must resolve (from cwd external/cortical_tiles) to {COV_CONFIG_FILE}; got {path}"
        )
        assert path.is_file(), f"--cov-config file {path} does not exist"


@pytest.mark.smoke
class TestCorticalTilesCoverageOmits:
    """REQ-CTDEADCODE-3 (supersedes REQ-CTILESTEST-139, which superseded REQ-CTILESTEST-25)."""

    def test_cov_config_omits_exactly_the_dead_modules(self, pixi_config):
        omits = _list_setting(_cortical_tiles_cov_run_section(pixi_config).get("omit", ""))
        omitted_distbottom = [
            omit for omit in omits if any(omit.endswith(module) for module in LIVE_DISTBOTTOM_MODULES)
        ]
        assert not omitted_distbottom, (
            f"distbottom modules must stay measured (REQ-CTILESTEST-139); omitted: {omitted_distbottom!r}"
        )
        omitted_deleted = [omit for omit in omits if any(omit.endswith(module) for module in DELETED_MODULES)]
        assert not omitted_deleted, (
            f"modules deleted from cortical_tiles (REQ-CTDEADCODE-1) must leave the omit list "
            f"(REQ-CTDEADCODE-3, TASK-131); still omitted: {omitted_deleted!r}"
        )
        assert sorted(omits) == sorted(DEAD_MODULE_OMITS), (
            f"[run] omit must be exactly the {len(DEAD_MODULE_OMITS)} dead-module globs "
            f"{sorted(DEAD_MODULE_OMITS)} (REQ-CTDEADCODE-3, TASK-131); got {omits!r}"
        )


COVERAGE_FAIL_UNDER = 85


@pytest.mark.smoke
class TestCorticalTilesCoverageThreshold:
    """REQ-CTILESTEST-167."""

    def test_task_enforces_85_percent_coverage_floor(self, pixi_config):
        _, argv = _cortical_tiles_invocation(pixi_config)
        values = _option_values(argv, "--cov-fail-under")
        assert len(values) == 1, (
            f"{TASK_NAME} must pass exactly one --cov-fail-under (REQ-CTILESTEST-167); got {values!r}"
        )
        assert values[0] == str(COVERAGE_FAIL_UNDER), (
            f"--cov-fail-under must be the integer {COVERAGE_FAIL_UNDER} (REQ-CTILESTEST-167); got {values[0]!r}"
        )
