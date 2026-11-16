"""Environment check: the top-level PyPI dependencies of ``pixi.toml`` are present.

Standard library plus ``packaging`` only; no stage code, no ``champollion_V1``, no ``soma``.
Modules are located with ``find_spec``, never imported: ``dico_toolbox`` imports ``soma``.
Used by ``tests/conftest.py`` (session start) and ``tests/test_pixi_pypi_dependencies_importable.py``
(``pixi run check-env``).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import metadata
from importlib.machinery import PathFinder
from pathlib import Path

import tomllib
from packaging.requirements import InvalidRequirement, Requirement

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class EnvironmentReport:
    """What the active environment lacks, by distribution (names are PEP 503 normalised)."""

    missing_distributions: dict[str, str] = field(default_factory=dict)  # distribution -> required by
    missing_modules: dict[str, list[str]] = field(default_factory=dict)  # distribution -> modules
    required_by: dict[str, str] = field(default_factory=dict)


def normalize(name: str) -> str:
    """PEP 503 name: lower case, runs of ``-_.`` become one ``-``."""
    return re.sub(r"[-_.]+", "-", name).lower()


def declared_distributions(pixi_toml: Path) -> list[str]:
    """Distributions of the top-level ``[pypi-dependencies]`` table (features are not read)."""
    with pixi_toml.open("rb") as handle:
        table = tomllib.load(handle).get("pypi-dependencies", {})
    return [normalize(name) for name in table]


def _required(dist, distribution: Callable) -> list[str]:
    """Names a distribution requires unconditionally (extras and failing markers left out)."""
    names = []
    for line in dist.requires or []:
        try:
            req = Requirement(line)
        except InvalidRequirement:
            continue
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        names.append(normalize(req.name))
    return names


def find_missing(
    pixi_toml: Path,
    *,
    distribution: Callable = metadata.distribution,
    packages_distributions: Callable = metadata.packages_distributions,
    find_spec: Callable = PathFinder.find_spec,
) -> EnvironmentReport:
    """Check each declared distribution and the distributions it requires."""
    providers: dict[str, list[str]] = {}
    for module, dists in packages_distributions().items():
        for dist_name in dists:
            providers.setdefault(normalize(dist_name), []).append(module)

    report = EnvironmentReport()
    queue = [(name, "") for name in declared_distributions(pixi_toml)]
    seen: set[str] = set()
    while queue:
        name, parent = queue.pop(0)
        if name in seen:
            continue
        seen.add(name)
        try:
            dist = distribution(name)
        except metadata.PackageNotFoundError:
            report.missing_distributions[name] = parent
            continue
        # A name that is no identifier (an empty shim such as sklearn 0.0) is no module to locate.
        modules = sorted(m for m in providers.get(name, []) if m.isidentifier())
        absent = [m for m in modules if find_spec(m) is None]
        if absent:
            report.missing_modules[name] = absent
            if parent:
                report.required_by[name] = parent
        if not parent:
            queue.extend((req, name) for req in _required(dist, distribution))
    return report


def render(report: EnvironmentReport, worktree: Path) -> str | None:
    """One line naming every missing distribution and module and the fix; None when nothing is missing."""
    parts = []
    for name, parent in report.missing_distributions.items():
        parts.append(f"{name} (not installed, required by {parent})" if parent else f"{name} (not installed)")
    for name, modules in report.missing_modules.items():
        parent = report.required_by.get(name)
        suffix = f", required by {parent}" if parent else ""
        parts.append(f"{name} (module {', '.join(modules)} not found{suffix})")
    if not parts:
        return None
    return (
        f"The environment lacks {len(parts)} PyPI package(s) of pixi.toml: {'; '.join(parts)}. "
        f"Fix: run 'pixi install' in {worktree}."
    )


def check_environment(pixi_toml: Path = ROOT / "pixi.toml") -> str | None:
    """The message for the active environment, or None when it is complete."""
    return render(find_missing(pixi_toml), pixi_toml.parent)
