#!/usr/bin/env python3
"""
Root-level tool configuration: a bare ``pytest`` or a bare ``vulture`` run from
the repository root must behave like the pixi tasks that pass explicit paths.

Covers:

- REQ-TESTCFG-BDRABCZUK-CDFF52F4EEBE: a bare ``pytest`` from the root collects
  test items only from ``tests/`` (never from the ``external/`` submodules).
- REQ-TESTCFG-BDRABCZUK-D8F6C3FE2142: the root holds exactly one pytest
  configuration section (``pytest.ini`` / ``pyproject.toml`` / ``tox.ini`` /
  ``setup.cfg``), so no section is silently shadowed by another.
- REQ-TESTCFG-BDRABCZUK-38FF93B6F8E7: the root pytest configuration registers
  the ``serial`` marker (``test-parallel``'s second pass selects on it).
- REQ-TESTCFG-BDRABCZUK-29BCF2DD74E2: a bare ``vulture`` from the root reports
  the same findings as ``vulture src/ tests/`` (the ``dead-code`` task).

Every check runs the real tool in a subprocess from the repository root this
file lives in (a worktree root when run from a worktree).
"""

import configparser
import os
import re
import subprocess
import sys
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent

# Env vars that would make a child pytest/coverage run differ from a plain one.
_CHILD_ENV_DROP = ("PYTEST_ADDOPTS", "COVERAGE_PROCESS_START")

# ``<path>.py::<name>`` (collected item) or ``ERROR <path>.py`` (import failure).
_NODE_LINE = re.compile(r"^(?:ERROR )?([\w./-]+\.py)(?:::|$)")


def _child_env():
    return {k: v for k, v in os.environ.items() if k not in _CHILD_ENV_DROP}


def _run_from_root(*args, timeout=600):
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=_child_env(),
        timeout=timeout,
    )


def _pytest_config_sections():
    """Return the root files that carry a pytest configuration section."""
    found = []
    pytest_ini = REPO_ROOT / "pytest.ini"
    if pytest_ini.is_file():
        found.append("pytest.ini [pytest]")
    pyproject = REPO_ROOT / "pyproject.toml"
    if pyproject.is_file():
        with pyproject.open("rb") as handle:
            if "ini_options" in tomllib.load(handle).get("tool", {}).get("pytest", {}):
                found.append("pyproject.toml [tool.pytest.ini_options]")
    for name, section in (("tox.ini", "pytest"), ("setup.cfg", "tool:pytest")):
        path = REPO_ROOT / name
        if path.is_file():
            parser = configparser.ConfigParser(interpolation=None)
            parser.read(path, encoding="utf-8")
            if parser.has_section(section):
                found.append(f"{name} [{section}]")
    return found


def test_bare_pytest_from_root_collects_only_tests_dir():
    """REQ-TESTCFG-BDRABCZUK-CDFF52F4EEBE."""
    proc = _run_from_root("-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider")
    output = proc.stdout + proc.stderr
    # ``-q`` prints one ``<path>::<name>`` line per item and ``ERROR <path>``
    # per module that failed to import; take the path part of each.
    paths = {m.group(1) for m in map(_NODE_LINE.match, output.splitlines()) if m}
    outside = sorted(p for p in paths if not p.startswith("tests/"))
    assert any(p.startswith("tests/") for p in paths), f"no item collected from tests/:\n{output[-3000:]}"
    assert not outside, (
        f"bare `pytest` from {REPO_ROOT} collected {len(outside)} file(s) outside tests/, e.g. {outside[:5]}"
    )


def test_root_has_exactly_one_pytest_config_section():
    """REQ-TESTCFG-BDRABCZUK-D8F6C3FE2142."""
    sections = _pytest_config_sections()
    assert len(sections) == 1, f"expected exactly one pytest configuration section at the root, found {sections}"


def test_root_pytest_config_registers_serial_marker():
    """REQ-TESTCFG-BDRABCZUK-38FF93B6F8E7."""
    proc = _run_from_root("-m", "pytest", "--markers", "-p", "no:cacheprovider", timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "@pytest.mark.serial:" in proc.stdout, "the `serial` marker is not registered:\n" + proc.stdout[-3000:]


def test_bare_vulture_from_root_matches_dead_code_task():
    """REQ-TESTCFG-BDRABCZUK-29BCF2DD74E2."""
    bare = _run_from_root("-m", "vulture")
    explicit = _run_from_root("-m", "vulture", "src/", "tests/")
    assert (bare.returncode, sorted(bare.stdout.splitlines())) == (
        explicit.returncode,
        sorted(explicit.stdout.splitlines()),
    ), (
        f"bare `vulture` (rc={bare.returncode}) differs from `vulture src/ tests/` (rc={explicit.returncode}).\n"
        f"bare stdout+stderr:\n{(bare.stdout + bare.stderr)[-2000:]}\n"
        f"explicit stdout+stderr:\n{(explicit.stdout + explicit.stderr)[-2000:]}"
    )
