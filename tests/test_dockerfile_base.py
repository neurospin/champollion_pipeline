#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`.github/Dockerfile.base` runs no pixi task that reaches `init-submodules`.

Run with: pixi run test-specific tests/test_dockerfile_base.py

`.dockerignore` drops `.git/`, so `git submodule update` (the
`init-submodules` task) exits 128 inside the image; the submodules are
already in the build context. A `pixi run <task>` there must pass
`--skip-deps` whenever the task depends, directly or through other
tasks, on `init-submodules`.
"""

import re
import shlex
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / ".github" / "Dockerfile.base"
FORBIDDEN = "init-submodules"


def _all_tasks() -> dict:
    data = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))
    tasks = dict(data.get("tasks", {}))
    for feature in data.get("feature", {}).values():
        tasks.update(feature.get("tasks", {}))
    return tasks


def _deps(task) -> list[str]:
    if isinstance(task, dict):
        deps = task.get("depends-on", [])
        return [deps] if isinstance(deps, str) else list(deps)
    return []


def _reaches(tasks: dict, name: str, seen=None) -> bool:
    seen = seen if seen is not None else set()
    if name == FORBIDDEN:
        return True
    if name in seen or name not in tasks:
        return False
    seen.add(name)
    return any(_reaches(tasks, dep, seen) for dep in _deps(tasks[name]))


def _pixi_run_calls() -> list[tuple[str, bool]]:
    """(task name, skips deps) for each named `pixi run` in the Dockerfile."""
    text = DOCKERFILE.read_text(encoding="utf-8").replace("\\\n", " ")
    calls = []
    for segment in re.split(r"&&|\n", text):
        if "pixi run" not in segment:
            continue
        words = shlex.split(segment.split("pixi run", 1)[1])
        if "--" in words:
            continue
        skip = "--skip-deps" in words
        names = [w for w in words if not w.startswith("-")]
        if names:
            calls.append((names[0], skip))
    return calls


def test_dockerfile_calls_pixi_tasks():
    assert _pixi_run_calls(), "no named `pixi run <task>` found in Dockerfile.base"


@pytest.mark.parametrize("name,skip", _pixi_run_calls())
def test_pixi_run_task_does_not_reach_init_submodules(name, skip):
    tasks = _all_tasks()
    assert name in tasks, f"{name} is not a pixi task"
    assert skip or not _reaches(tasks, name), (
        f"`pixi run {name}` reaches {FORBIDDEN} through depends-on; pass --skip-deps (the image has no .git)"
    )
