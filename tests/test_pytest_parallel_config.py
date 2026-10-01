"""Tests for REQ-TEST-SPEED-03..06 — parallel test runs via pytest-xdist.

* REQ-TEST-SPEED-03: ``pytest-xdist`` is declared wherever ``pytest`` is.
* REQ-TEST-SPEED-04: a ``test-parallel`` pixi task runs every non-``serial``
  test in one ``-n auto`` pytest invocation.
* REQ-TEST-SPEED-05: the submodule checkout isolation guard is ``serial``.
* REQ-TEST-SPEED-06: ``test-parallel`` runs ``serial`` tests only in a pytest
  invocation without ``-n``.

Everything except REQ-TEST-SPEED-05 is checked by parsing ``pixi.toml``; a
nested ``pytest -n`` run of the whole suite would be far too slow. Marker
expressions (``-m``) are evaluated with pytest's own expression engine, so
``"not serial"``, ``"not (serial or slow)"`` etc. are all understood.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib
from _pytest.mark.expression import Expression

REPO_ROOT = Path(__file__).resolve().parents[1]
PIXI_TOML = REPO_ROOT / "pixi.toml"
PARALLEL_TASK = "test-parallel"
SERIAL_MARKER = "serial"
GUARD_MODULE = "tests/test_submodule_checkout_isolation.py"

# Marker sets an ordinary (non-serial) test in this suite may carry.
ORDINARY_MARKER_SETS = (frozenset(), frozenset({"smoke"}), frozenset({"unit"}), frozenset({"integration"}))
SEPARATORS = {"&&", "||", ";", "&", "|"}
# pytest options that consume the following token as their value.
PYTEST_VALUE_OPTIONS = {
    "-m",
    "-k",
    "-n",
    "--numprocesses",
    "-c",
    "-p",
    "-o",
    "--rootdir",
    "--basetemp",
    "--cov",
    "--cov-report",
    "--cov-config",
    "--cov-fail-under",
    "--dist",
    "--maxprocesses",
    "--tb",
    "--durations",
}


@pytest.fixture(scope="module")
def pixi_config() -> dict:
    with PIXI_TOML.open("rb") as handle:
        return tomllib.load(handle)


def _dependency_tables(config: dict):
    """Yield ``(label, table)`` for every dependency table in the manifest."""
    for key in ("dependencies", "pypi-dependencies"):
        if key in config:
            yield key, config[key]
    for name, feature in config.get("feature", {}).items():
        for key in ("dependencies", "pypi-dependencies"):
            if key in feature:
                yield f"feature.{name}.{key}", feature[key]


def _task_commands(config: dict, name: str, seen: frozenset = frozenset()) -> list[str]:
    """Shell commands a pixi task runs: its ``depends-on`` tasks first, then its own ``cmd``."""
    tasks = dict(config.get("tasks", {}))
    for feature in config.get("feature", {}).values():
        tasks.update(feature.get("tasks", {}))
    assert name in tasks, f"pixi.toml defines no {name!r} task"
    assert name not in seen, f"cyclic depends-on through {name!r}"
    task = tasks[name]
    if isinstance(task, str):
        return [task]
    commands = []
    depends = task.get("depends-on", task.get("depends_on", []))
    for dep in [depends] if isinstance(depends, str) else depends:
        dep_name = dep if isinstance(dep, str) else dep["task"]
        commands += _task_commands(config, dep_name, seen | {name})
    cmd = task.get("cmd")
    if cmd:
        commands.append(cmd if isinstance(cmd, str) else shlex.join(cmd))
    return commands


def _pytest_invocations(commands: list[str]) -> list[list[str]]:
    """Argv (after ``pytest``) of every pytest invocation in the given shell commands."""
    invocations = []
    for command in commands:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True  # noqa: V101 (read by shlex itself)
        segment: list[str] = []
        for token in [*lexer, "&&"]:
            if token in SEPARATORS:
                for i, word in enumerate(segment):
                    if word == "pytest" or word.endswith("/pytest"):
                        invocations.append(segment[i + 1 :])
                        break
                    if word == "-m" and i + 1 < len(segment) and segment[i + 1] == "pytest":
                        invocations.append(segment[i + 2 :])
                        break
                segment = []
            else:
                segment.append(token)
    return invocations


def _option_values(argv: list[str], short: str | None, long: str | None) -> list[str]:
    values = []
    for i, token in enumerate(argv):
        for opt in filter(None, (short, long)):
            if token == opt and i + 1 < len(argv):
                values.append(argv[i + 1])
            elif token.startswith(opt + "="):
                values.append(token.split("=", 1)[1])
            elif short and opt == short and token.startswith(short) and len(token) > len(short):
                if not token.startswith("--"):
                    values.append(token[len(short) :].lstrip("="))
    return values


def _numprocesses(argv: list[str]) -> list[str]:
    return _option_values(argv, "-n", "--numprocesses")


def _selects(argv: list[str], markers: frozenset) -> bool:
    """Whether this invocation's ``-m`` expressions select a test carrying exactly ``markers``."""
    return all(
        Expression.compile(expr).evaluate(lambda name, **_kwargs: name in markers)
        for expr in _option_values(argv, "-m", None)
    )


def _positional_paths(argv: list[str]) -> list[str]:
    paths, skip = [], False
    for token in argv:
        if skip:
            skip = False
            continue
        if token.startswith("-"):
            skip = token in PYTEST_VALUE_OPTIONS
            continue
        paths.append(token.rstrip("/"))
    return paths


def _parallel_invocations(pixi_config: dict) -> list[list[str]]:
    """Pytest invocations of the parallel task (asserting inside a test, so a gap reads as FAILED)."""
    invocations = _pytest_invocations(_task_commands(pixi_config, PARALLEL_TASK))
    assert invocations, f"pixi task {PARALLEL_TASK!r} runs no pytest invocation"
    return invocations


@pytest.mark.smoke
class TestXdistDeclared:
    """REQ-TEST-SPEED-03."""

    def test_pytest_xdist_declared_wherever_pytest_is(self, pixi_config):
        tables_with_pytest = [(label, t) for label, t in _dependency_tables(pixi_config) if "pytest" in t]
        assert tables_with_pytest, "pixi.toml declares pytest in no dependency table"
        missing = [label for label, table in tables_with_pytest if "pytest-xdist" not in table]
        assert not missing, f"pytest-xdist is not declared alongside pytest in: {missing}"


@pytest.mark.smoke
class TestParallelTask:
    """REQ-TEST-SPEED-04 and REQ-TEST-SPEED-06."""

    def test_parallel_task_runs_non_serial_tests_with_n_auto(self, pixi_config):
        """REQ-TEST-SPEED-04: one ``-n auto`` invocation covers every non-serial test under tests/."""
        parallel_invocations = _parallel_invocations(pixi_config)
        covering = [
            argv
            for argv in parallel_invocations
            if "auto" in _numprocesses(argv)
            and not _option_values(argv, "-k", None)
            and set(_positional_paths(argv)) <= {"tests"}
            and all(_selects(argv, markers) for markers in ORDINARY_MARKER_SETS)
        ]
        assert covering, (
            f"no pytest invocation in {PARALLEL_TASK!r} passes '-n auto' while selecting every "
            f"non-serial test under tests/; invocations: {parallel_invocations!r}"
        )

    def test_parallel_invocations_deselect_serial_tests(self, pixi_config):
        """REQ-TEST-SPEED-06: no invocation passing ``-n`` selects a ``serial`` test."""
        parallel_invocations = _parallel_invocations(pixi_config)
        serial = frozenset({SERIAL_MARKER})
        offenders = [argv for argv in parallel_invocations if _numprocesses(argv) and _selects(argv, serial)]
        assert not offenders, f"these '-n' invocations of {PARALLEL_TASK!r} also run serial tests: {offenders!r}"

    def test_a_non_parallel_invocation_selects_serial_tests(self, pixi_config):
        """REQ-TEST-SPEED-06: serial tests do run, in an invocation without ``-n``."""
        parallel_invocations = _parallel_invocations(pixi_config)
        serial = frozenset({SERIAL_MARKER})
        runners = [argv for argv in parallel_invocations if not _numprocesses(argv) and _selects(argv, serial)]
        assert runners, (
            f"no pytest invocation of {PARALLEL_TASK!r} without '-n' selects serial-marked tests; "
            f"invocations: {parallel_invocations!r}"
        )


def _collected_ids(*extra: str) -> list[str]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            GUARD_MODULE,
            "--collect-only",
            "-q",
            "--strict-markers",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            *extra,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode in (0, 5), (
        f"collection of {GUARD_MODULE} failed (exit {result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    return sorted(line for line in result.stdout.splitlines() if "::" in line)


@pytest.mark.smoke
class TestSerialMarker:
    """REQ-TEST-SPEED-05."""

    def test_isolation_guard_tests_are_selected_by_serial_marker(self):
        every_test = _collected_ids()
        assert every_test, f"no tests collected from {GUARD_MODULE}"
        serial_tests = _collected_ids("-m", SERIAL_MARKER)
        unmarked = sorted(set(every_test) - set(serial_tests))
        assert not unmarked, f"tests in {GUARD_MODULE} not selected by '-m {SERIAL_MARKER}': {unmarked}"
