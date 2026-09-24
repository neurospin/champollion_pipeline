#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python version floor consistency across pyproject.toml, pixi.toml and README.md.

Run with: pixi run test-specific tests/test_python_version_floor.py

REQ-INSTALL-05: every minimum Python version declared in `pixi.toml` or
stated in `README.md` shall be at least 3.10, the floor set by
`pyproject.toml`'s `requires-python`.

3.10 is the genuine floor, not a stale constraint: source modules evaluate
PEP 604 unions of runtime types in function signatures without
`from __future__ import annotations` (e.g. `utils/lib.py::are_paths_valid`
`-> bool | None`, `generate_champollion_config.py::_get_crop_size`
`-> tuple[int, int, int] | None`), which raises `TypeError` at import time
on Python < 3.10; and the hard dependency `champollion-utils` itself
declares `requires-python = ">=3.10"`.
"""

import re
from pathlib import Path

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
PIXI_TOML = REPO_ROOT / "pixi.toml"
README = REPO_ROOT / "README.md"

FLOOR = (3, 10)

_LOWER_BOUND = re.compile(r"(?:>=|==|~=|>)\s*(\d+)\.(\d+)")


def _lower_bound(spec: str) -> tuple[int, int] | None:
    """Return the (major, minor) lower bound of a version spec, or None if it has none."""
    bounds = [(int(major), int(minor)) for major, minor in _LOWER_BOUND.findall(spec)]
    return max(bounds) if bounds else None


def _pixi_python_specs() -> list[tuple[str, str]]:
    """Return (table, spec) for every `python` dependency declared in pixi.toml."""
    with PIXI_TOML.open("rb") as handle:
        data = tomllib.load(handle)

    tables: list[tuple[str, dict]] = [("[dependencies]", data.get("dependencies", {}))]
    for name, feature in data.get("feature", {}).items():
        tables.append((f"[feature.{name}.dependencies]", feature.get("dependencies", {})))
        for target, body in feature.get("target", {}).items():
            tables.append((f"[feature.{name}.target.{target}.dependencies]", body.get("dependencies", {})))
    for target, body in data.get("target", {}).items():
        tables.append((f"[target.{target}.dependencies]", body.get("dependencies", {})))

    specs = []
    for table, deps in tables:
        if "python" in deps:
            spec = deps["python"]
            if isinstance(spec, dict):
                spec = spec.get("version", "")
            specs.append((table, str(spec)))
    return specs


_PIXI_SPECS = _pixi_python_specs()

_README_FLOOR = re.compile(
    r"python(?:-|\s*)(?:%E2%89%A5|≥|>=)\s*(\d+)\.(\d+)",
    re.IGNORECASE,
)


class TestPythonVersionFloor:
    """No stated/declared Python floor may undercut pyproject.toml's requires-python."""

    @pytest.mark.smoke
    def test_pyproject_requires_python_floor_is_3_10(self):
        """pyproject.toml's requires-python is the reference floor, 3.10."""
        with PYPROJECT.open("rb") as handle:
            requires_python = tomllib.load(handle)["project"]["requires-python"]
        assert _lower_bound(requires_python) == FLOOR, (
            f"pyproject.toml requires-python is {requires_python!r}; expected a 3.10 floor "
            "(source code uses runtime PEP 604 unions, champollion-utils requires >=3.10)"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize(("table", "spec"), _PIXI_SPECS, ids=[table for table, _ in _PIXI_SPECS])
    def test_pixi_python_floor_at_least_3_10(self, table, spec):
        """Each pixi.toml `python` dependency has a lower bound of at least 3.10."""
        bound = _lower_bound(spec)
        assert bound is not None and bound >= FLOOR, (
            f"pixi.toml {table} declares python = {spec!r}; its floor must be at least "
            "3.10 to match pyproject.toml's requires-python"
        )

    @pytest.mark.smoke
    def test_readme_python_floor_at_least_3_10(self):
        """Every Python floor stated in README.md (badge and prose) is at least 3.10."""
        text = README.read_text(encoding="utf-8")
        stated = [
            (text.count("\n", 0, match.start()) + 1, (int(match.group(1)), int(match.group(2))))
            for match in _README_FLOOR.finditer(text)
        ]
        too_low = [(line, f"{major}.{minor}") for line, (major, minor) in stated if (major, minor) < FLOOR]
        assert not too_low, (
            f"README.md states a Python floor below 3.10 at (line, version) {too_low!r}; "
            "pyproject.toml's requires-python is >=3.10"
        )
