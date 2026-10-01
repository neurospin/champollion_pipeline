"""Per-task coverage data files for the three coverage-measuring pixi tasks (TASK-125).

``test-cov``, ``test-champollion`` and ``test-cortical-tiles`` all wrote and
erased the same pipeline-root ``.coverage``, so concurrent runs corrupted each
other's totals (2026-09-30: ``test-cov`` TOTAL 63%, polluted by a parallel
``test-cortical-tiles`` run).

- REQ-COVISO-01 — the three resolved data files are pairwise different paths.
- REQ-COVISO-02 — no data file basename is ``.coverage``, begins with
  ``.coverage.``, or begins with another task's basename followed by ``.``
  (coverage.py combines and erases ``<basename>.*`` siblings; the
  ``test-champollion`` config runs in parallel mode, so its combine/erase
  would otherwise swallow or delete another task's file).
- REQ-COVISO-03 (supersedes REQ-CTILESTEST-26) — each data file lies directly
  in the champollion_pipeline root (never inside a submodule checkout).
- REQ-COVISO-04 — the pipeline's git ignore rules cover each data file and
  its ``<basename>.<suffix>`` parallel-mode shards.

"Resolved data file" follows coverage.py 7.15 / pytest-cov semantics, from the
task's effective working directory: a ``COVERAGE_FILE`` set in the task's
``env`` table or as a leading ``COVERAGE_FILE=... `` assignment wins;
otherwise ``data_file`` from the ``[run]`` settings of the ``--cov-config``
file (or, without one, the first of ``.coveragerc``/``setup.cfg``/
``tox.ini``/``pyproject.toml`` coverage.py finds in the cwd); otherwise
``.coverage``. ``${VAR}``/``${VAR-default}`` are substituted with
``PIXI_PROJECT_ROOT`` set to the pipeline root.

Offline: parses ``pixi.toml`` and the coverage config files, and asks
``git check-ignore --no-index`` about paths that need not exist.
"""

import configparser
import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
PIXI_TOML = REPO_ROOT / "pixi.toml"

COVERAGE_TASKS = ("test-cov", "test-champollion", "test-cortical-tiles")
DEFAULT_DATA_FILE = ".coverage"
# Representative parallel-mode shard suffix: <host>.<pid>.<random>.
SHARD_SUFFIX = ".nodename.12345.XaBcDeFg"
ENV_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)


@pytest.fixture(scope="module")
def pixi_config() -> dict:
    with PIXI_TOML.open("rb") as handle:
        return tomllib.load(handle)


@pytest.fixture(scope="module")
def data_files(pixi_config) -> dict[str, Path]:
    return {name: _resolved_data_file(pixi_config, name) for name in COVERAGE_TASKS}


def _tasks_visible_in(config: dict, env: str) -> dict:
    env_spec = config["environments"][env]
    features = env_spec["features"] if isinstance(env_spec, dict) else env_spec
    tasks = dict(config.get("tasks", {}))
    for feature in features:
        tasks.update(config.get("feature", {}).get(feature, {}).get("tasks", {}))
    return tasks


def _expand_pixi_root(value: str) -> str:
    """Expand ``$PIXI_PROJECT_ROOT``/``${PIXI_PROJECT_ROOT}`` as pixi's task shell does."""
    return re.sub(r"\$\{PIXI_PROJECT_ROOT\}|\$PIXI_PROJECT_ROOT\b", str(REPO_ROOT), value)


def _invocation(task) -> tuple[Path, dict[str, str], list[str]]:
    """``(effective cwd, task env, pytest argv)`` for a pixi task definition."""
    if isinstance(task, str):
        command, cwd, env = task, REPO_ROOT, {}
    elif isinstance(task, dict):
        command = task.get("cmd", "")
        cwd = (REPO_ROOT / task["cwd"]) if "cwd" in task else REPO_ROOT
        env = {k: _expand_pixi_root(str(v)) for k, v in task.get("env", {}).items()}
    else:
        raise TypeError(f"unexpected pixi task type: {type(task)!r}")
    if isinstance(command, list):
        command = " ".join(command)

    tokens = shlex.split(command)
    while True:
        if len(tokens) >= 3 and tokens[0] == "cd" and tokens[2] == "&&":
            cwd = cwd / _expand_pixi_root(tokens[1])
            tokens = tokens[3:]
            continue
        if tokens and tokens[0] == "export" and len(tokens) >= 3 and tokens[2] == "&&":
            match = ENV_ASSIGNMENT.match(tokens[1])
            if match:
                env[match.group(1)] = _expand_pixi_root(match.group(2))
                tokens = tokens[3:]
                continue
        if tokens and ENV_ASSIGNMENT.match(tokens[0]):
            match = ENV_ASSIGNMENT.match(tokens[0])
            env[match.group(1)] = _expand_pixi_root(match.group(2))
            tokens = tokens[1:]
            continue
        break

    for i, token in enumerate(tokens):
        if token == "pytest" or token.endswith("/pytest"):
            return cwd.resolve(), env, tokens[i + 1 :]
        if token == "-m" and i + 1 < len(tokens) and tokens[i + 1] == "pytest":
            return cwd.resolve(), env, tokens[i + 2 :]
    raise AssertionError(f"task command does not invoke pytest: {command!r}")


def _option_values(argv: list[str], option: str) -> list[str]:
    values = []
    for i, token in enumerate(argv):
        if token.startswith(option + "="):
            values.append(token.split("=", 1)[1])
        elif token == option and i + 1 < len(argv):
            values.append(argv[i + 1])
    return values


def _data_file_setting(config_file: Path, specified: bool) -> str | None:
    """Raw ``[run] data_file`` from one coverage config file, or None if it sets none."""
    if config_file.suffix == ".toml":
        with config_file.open("rb") as handle:
            run = tomllib.load(handle).get("tool", {}).get("coverage", {}).get("run", {})
        return run.get("data_file")
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(config_file, encoding="utf-8")
    # coverage.py: .coveragerc / a specified file use [run]; setup.cfg / tox.ini use [coverage:run].
    sections = ["run", "coverage:run"] if (specified or config_file.name == ".coveragerc") else ["coverage:run"]
    for section in sections:
        if parser.has_option(section, "data_file"):
            return parser.get(section, "data_file")
    return None


def _resolved_data_file(pixi_config: dict, task_name: str) -> Path:
    from coverage.misc import substitute_variables

    tasks = _tasks_visible_in(pixi_config, "default")
    assert task_name in tasks, f"pixi.toml defines no {task_name!r} task visible in the default environment"
    cwd, task_env, argv = _invocation(tasks[task_name])

    variables = {**os.environ, **task_env, "PIXI_PROJECT_ROOT": str(REPO_ROOT)}
    if task_env.get("COVERAGE_FILE"):
        raw = task_env["COVERAGE_FILE"]
    else:
        configs = _option_values(argv, "--cov-config")
        assert len(configs) <= 1, f"{task_name} passes --cov-config more than once: {configs!r}"
        if configs:
            config_file = (cwd / _expand_pixi_root(configs[0])).resolve()
            assert config_file.is_file(), f"{task_name} --cov-config file {config_file} does not exist"
            raw = _data_file_setting(config_file, specified=True)
        else:
            raw = None
            for name in (".coveragerc", "setup.cfg", "tox.ini", "pyproject.toml"):
                candidate = cwd / name
                if candidate.is_file():
                    raw = _data_file_setting(candidate, specified=False)
                    if raw is not None:
                        break
        raw = raw or DEFAULT_DATA_FILE
    expanded = os.path.expanduser(substitute_variables(raw, variables))
    return (cwd / expanded).resolve()


def _git_ignores(relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", "--no-index", "-q", relative_path],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), f"git check-ignore failed: {result.stderr.strip()}"
    return result.returncode == 0


@pytest.mark.smoke
class TestCoverageDataFilesPairwiseDistinct:
    """REQ-COVISO-01."""

    def test_data_files_of_the_three_coverage_tasks_are_pairwise_distinct(self, data_files):
        by_path: dict[Path, list[str]] = {}
        for task, path in data_files.items():
            by_path.setdefault(path, []).append(task)
        shared = {str(path): tasks for path, tasks in by_path.items() if len(tasks) > 1}
        assert not shared, (
            f"coverage data files must be pairwise different (REQ-COVISO-01); shared: {shared!r}; "
            f"resolved: { {t: str(p) for t, p in data_files.items()}!r}"
        )


@pytest.mark.smoke
class TestCoverageDataFileNames:
    """REQ-COVISO-02."""

    @pytest.mark.parametrize("task", COVERAGE_TASKS)
    def test_data_file_basename_is_outside_default_and_sibling_shard_namespaces(self, task, data_files):
        name = data_files[task].name
        assert name != DEFAULT_DATA_FILE, (
            f"{task} data file must not be named {DEFAULT_DATA_FILE!r} (REQ-COVISO-02); resolved {data_files[task]}"
        )
        assert not name.startswith(DEFAULT_DATA_FILE + "."), (
            f"{task} data file {name!r} must not begin with '.coverage.' (REQ-COVISO-02)"
        )
        clashes = [other for other in COVERAGE_TASKS if other != task and name.startswith(data_files[other].name + ".")]
        assert not clashes, (
            f"{task} data file {name!r} lies in the '<basename>.*' shard namespace of {clashes!r} (REQ-COVISO-02)"
        )


@pytest.mark.smoke
class TestCoverageDataFileLocation:
    """REQ-COVISO-03 (supersedes REQ-CTILESTEST-26)."""

    @pytest.mark.parametrize("task", COVERAGE_TASKS)
    def test_data_file_is_directly_in_pipeline_root(self, task, data_files):
        assert data_files[task].parent == REPO_ROOT.resolve(), (
            f"{task} data file resolves to {data_files[task]}; it must lie directly in {REPO_ROOT} (REQ-COVISO-03)"
        )


@pytest.mark.smoke
class TestCoverageDataFilesGitignored:
    """REQ-COVISO-04."""

    @pytest.mark.parametrize("task", COVERAGE_TASKS)
    def test_data_file_and_its_shards_are_gitignored(self, task, data_files):
        name = data_files[task].name
        for candidate in (name, name + SHARD_SUFFIX):
            assert _git_ignores(candidate), (
                f"{task}: pipeline git ignore rules must ignore root-level {candidate!r} (REQ-COVISO-04)"
            )
