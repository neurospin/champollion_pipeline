#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CI workflows install a pixi that satisfies the project's `requires-pixi`.

Run with: pixi run test-specific tests/test_ci_workflows.py

Every `prefix-dev/setup-pixi` step that pins `pixi-version` in
`.github/workflows` must pin a version at or above the minimum that
`pixi.toml` declares in `requires-pixi`; otherwise pixi refuses the
project and CI fails before any test runs.
"""

import re
from pathlib import Path

import pytest
import tomllib
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))


def _version(text: str) -> tuple[int, ...]:
    match = re.search(r"(\d+(?:\.\d+)*)", text)
    assert match, f"no version in {text!r}"
    return tuple(int(part) for part in match.group(1).split("."))


def _required_minimum() -> tuple[int, ...]:
    data = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))
    spec = data["workspace"]["requires-pixi"] if "workspace" in data else data["project"]["requires-pixi"]
    assert spec.startswith(">="), f"unexpected requires-pixi form: {spec!r}"
    return _version(spec)


def _pinned_versions() -> list[tuple[str, str]]:
    pins = []
    for workflow in WORKFLOWS:
        document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
        for job in (document.get("jobs") or {}).values():
            for step in job.get("steps") or []:
                if str(step.get("uses", "")).startswith("prefix-dev/setup-pixi"):
                    pinned = (step.get("with") or {}).get("pixi-version")
                    if pinned:
                        pins.append((workflow.name, str(pinned)))
    return pins


def test_workflows_pin_a_pixi_version() -> None:
    assert _pinned_versions(), "no setup-pixi step pins pixi-version"


@pytest.mark.parametrize("workflow,pinned", _pinned_versions())
def test_pinned_pixi_satisfies_requires_pixi(workflow: str, pinned: str) -> None:
    assert _version(pinned) >= _required_minimum(), (
        f"{workflow} pins pixi {pinned}, below requires-pixi minimum in pixi.toml"
    )
