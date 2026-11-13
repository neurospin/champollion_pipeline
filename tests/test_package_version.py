#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package metadata in `champollion_pipeline/__init__.py` agrees with the manifests.

Run with: pixi run test-specific tests/test_package_version.py

REQ-INSTALL-07: every package fact stated in
`src/champollion_pipeline/__init__.py` (its `__version__` value and each
dependency its comments name) shall agree with the project manifests:
`__version__` equals `pyproject.toml`'s `[project] version`, and each named
dependency appears in `pyproject.toml` or `pixi.toml`.

The version check reads `__version__` from the imported module at runtime, so
both a literal string and a value derived via `importlib.metadata` satisfy it.

REQ-INSTALL-BDRABCZUK-4745885059E0: when the bump-version workflow bumps the
patch version, its version-bump commit contains
`src/champollion_pipeline/__init__.py` with `__version__` equal to the new
version (DEF-INSTALL-BDRABCZUK-C31D2053EF9D).
"""

import io
import os
import re
import shutil
import subprocess
import tokenize
from pathlib import Path

import pytest
import tomllib
import yaml

import champollion_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
PIXI_TOML = REPO_ROOT / "pixi.toml"
INIT_PY = REPO_ROOT / "src" / "champollion_pipeline" / "__init__.py"
BUMP_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "bump-version.yml"
INIT_PY_REL = "src/champollion_pipeline/__init__.py"

# A parenthesised, comma-separated list following the word "dependencies" in a
# comment, e.g. "optional dependencies (brainvisa, torch…)".
_DEPENDENCY_LIST = re.compile(r"dependenc\w*\s*\(([^)]*)\)", re.IGNORECASE)


def _pyproject_version() -> str:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def _comment_text(path: Path) -> str:
    """Return every `#` comment in a Python file, joined by newlines."""
    source = path.read_text(encoding="utf-8")
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return "\n".join(tok.string for tok in tokens if tok.type == tokenize.COMMENT)


def _dependencies_named_in_comments(path: Path) -> list[str]:
    """Return every dependency name listed in a "dependencies (...)" comment."""
    # Join comment lines so a list wrapped across two lines is still matched.
    text = " ".join(line.lstrip("#").strip() for line in _comment_text(path).splitlines())
    names = []
    for match in _DEPENDENCY_LIST.finditer(text):
        for raw in match.group(1).split(","):
            name = raw.strip().strip("…").rstrip(".").strip()
            if name:
                names.append(name)
    return names


@pytest.mark.smoke
class TestPackageMetadataMatchesManifests:
    """REQ-INSTALL-07: `__init__.py` states no package fact the manifests contradict."""

    def test_version_matches_pyproject(self):

        expected = _pyproject_version()
        assert champollion_pipeline.__version__ == expected, (
            f"champollion_pipeline.__version__ is {champollion_pipeline.__version__!r}, "
            f"but pyproject.toml declares version = {expected!r}"
        )

    def test_dependencies_named_in_comments_are_declared_in_manifests(self):
        manifests = (PYPROJECT.read_text(encoding="utf-8") + "\n" + PIXI_TOML.read_text(encoding="utf-8")).lower()
        named = _dependencies_named_in_comments(INIT_PY)
        undeclared = [name for name in named if not re.search(rf"\b{re.escape(name.lower())}\b", manifests)]
        assert not undeclared, (
            f"{INIT_PY.relative_to(REPO_ROOT)} comments name dependencies {undeclared} "
            "that appear in neither pyproject.toml nor pixi.toml"
        )


def _bump_steps() -> list[dict]:
    """Return the steps of the bump-version workflow's `bump` job."""
    workflow = yaml.safe_load(BUMP_WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"]["bump"]["steps"]


def _bump_step_script() -> str:
    """Return the `run` script of the step with id `bump`."""
    (step,) = [step for step in _bump_steps() if step.get("id") == "bump"]
    return step["run"]


def _commit_step_script() -> str:
    """Return the `run` script of the step that makes the version-bump commit."""
    (step,) = [step for step in _bump_steps() if "git commit" in step.get("run", "")]
    return step["run"]


class TestBumpVersionWorkflow:
    """REQ-INSTALL-BDRABCZUK-4745885059E0: the bump commit carries the new `__version__`."""

    def test_bump_step_writes_new_version_to_init_py(self, tmp_path):
        bash = shutil.which("bash")
        assert bash, "bash is needed to run the workflow's bump step"
        (tmp_path / "pixi.toml").write_text('[workspace]\nname = "x"\nversion = "1.2.3"\n', encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "1.2.3"\n', encoding="utf-8")
        init_py = tmp_path / INIT_PY_REL
        init_py.parent.mkdir(parents=True)
        init_py.write_text('"""Package."""\n\n__version__ = "1.2.3"\n', encoding="utf-8")
        env = {**os.environ, "GITHUB_OUTPUT": str(tmp_path / "github_output")}

        result = subprocess.run(
            [bash, "-e", "-c", _bump_step_script()],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, f"bump step failed: {result.stderr}"
        assert 'version = "1.2.4"' in (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
        assert '__version__ = "1.2.4"' in init_py.read_text(encoding="utf-8"), (
            f"the bump step left {INIT_PY_REL} at {init_py.read_text(encoding='utf-8')!r}; expected 1.2.4"
        )

    def test_commit_step_stages_init_py(self):
        add_lines = [line.split() for line in _commit_step_script().splitlines() if line.strip().startswith("git add")]
        staged = [path for line in add_lines for path in line[2:]]
        assert INIT_PY_REL in staged, f"the commit step stages {staged}, not {INIT_PY_REL}"
