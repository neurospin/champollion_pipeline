"""cortical_tiles runtime dependencies locked in the default pixi environment.

Task TASK-E2ECI-BDRABCZUK-38487724394D (epic TASK-E2ECI-BDRABCZUK-F3B0138F58F3).
The env SIF installs the default environment from ``pixi.lock``; the
cortical_tiles ``REQUIRES`` entries that conda does not cover must therefore be
PyPI dependencies of that environment, locked, and reproducible.

- REQ-E2ECI-BDRABCZUK-F6F7753E7051 -- the features composing the default
  environment of ``pixi.toml`` (top-level default feature included) declare
  p_tqdm, dico_toolbox, colorado and point-cloud-pattern-mining as PyPI
  dependencies (PEP 503 names).
- REQ-E2ECI-BDRABCZUK-1107408F2F52 -- on each platform locked for the default
  environment, ``pixi.lock`` lists each of the four as a PyPI package.
- REQ-E2ECI-BDRABCZUK-38E6C39F22DC -- the point-cloud-pattern-mining
  dependency pins its git source to a 40-hex-character rev (not ``@main``).

Offline: reads ``pixi.toml`` and ``pixi.lock`` as text; no pixi call, no network.
"""

import re
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parent.parent
PIXI_TOML = ROOT / "pixi.toml"
PIXI_LOCK = ROOT / "pixi.lock"

CORTICAL_TILES_PYPI_DEPS = (
    "p_tqdm",
    "dico_toolbox",
    "colorado",
    "point-cloud-pattern-mining",
)
PCPM = "point-cloud-pattern-mining"
COMMIT_HASH = re.compile(r"^[0-9a-fA-F]{40}$")


def _normalise(name):
    """PEP 503 normalised distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _manifest():
    with PIXI_TOML.open("rb") as handle:
        return tomllib.load(handle)


def _default_env_pypi_dependencies():
    """Map normalised name -> spec over the features of the default env."""
    manifest = _manifest()
    env = manifest.get("environments", {}).get("default", {})
    if isinstance(env, list):
        env = {"features": env}
    tables = []
    if not env.get("no-default-feature", False):
        tables.append(manifest.get("pypi-dependencies", {}))
    features = manifest.get("feature", {})
    for feature in env.get("features", []):
        tables.append(features.get(feature, {}).get("pypi-dependencies", {}))
    merged = {}
    for table in tables:
        for name, spec in table.items():
            merged[_normalise(name)] = spec
    return merged


def _is_top_level_key(line):
    """True for a column-0 YAML mapping key (``packages:``), not a ``- `` item."""
    return bool(line) and not line.startswith((" ", "- ", "#"))


def _lock_default_env_pypi_names():
    """Map platform -> normalised PyPI names locked for the default env.

    Line-based reading of the pixi.lock (v6+) layout: the ``environments``
    block lists ``- pypi: <url>`` per platform, and the top-level
    ``packages`` block maps each ``- pypi: <url>`` to a ``name:`` line.
    """
    lines = PIXI_LOCK.read_text(encoding="utf-8").splitlines()
    url_to_name = {}
    in_packages = False
    current_url = None
    for line in lines:
        if _is_top_level_key(line):
            in_packages = line == "packages:"
            current_url = None
            if in_packages:
                continue
        if not in_packages:
            continue
        if line.startswith("- "):
            current_url = line[len("- pypi: ") :] if line.startswith("- pypi: ") else None
        elif current_url and line.startswith("  name: "):
            url_to_name[current_url] = _normalise(line[len("  name: ") :].strip())
            current_url = None

    per_platform = {}
    section = None
    env = None
    platform = None
    for line in lines:
        if _is_top_level_key(line):
            section = line
            continue
        if section != "environments:":
            continue
        if re.match(r"^  \S", line):
            env = line.strip().rstrip(":")
            platform = None
        elif env == "default" and re.match(r"^      \S.*:$", line):
            platform = line.strip().rstrip(":")
            per_platform.setdefault(platform, set())
        elif env == "default" and platform and line.startswith("      - pypi: "):
            url = line[len("      - pypi: ") :].strip()
            per_platform[platform].add(url_to_name.get(url, url))
    return per_platform


@pytest.mark.parametrize("name", CORTICAL_TILES_PYPI_DEPS)
def test_default_env_declares_pypi_dependency(name):
    """REQ-E2ECI-BDRABCZUK-F6F7753E7051."""
    declared = _default_env_pypi_dependencies()
    assert _normalise(name) in declared, (
        f"{name} is not a PyPI dependency of any feature of the default "
        f"environment in pixi.toml (declared: {sorted(declared)})"
    )


@pytest.mark.parametrize("name", CORTICAL_TILES_PYPI_DEPS)
def test_default_env_locks_pypi_package(name):
    """REQ-E2ECI-BDRABCZUK-1107408F2F52."""
    per_platform = _lock_default_env_pypi_names()
    assert per_platform, "pixi.lock locks no platform for the default environment"
    missing = sorted(p for p, names in per_platform.items() if _normalise(name) not in names)
    assert not missing, f"{name} is not locked as a PyPI package of the default environment on {missing}"


def test_pcpm_pinned_to_commit_hash():
    """REQ-E2ECI-BDRABCZUK-38E6C39F22DC."""
    spec = _default_env_pypi_dependencies().get(PCPM)
    assert isinstance(spec, dict), f"{PCPM} is not declared with a git table in pixi.toml: {spec!r}"
    assert spec.get("git"), f"{PCPM} declares no git source: {spec!r}"
    rev = str(spec.get("rev", ""))
    assert COMMIT_HASH.match(rev), f"{PCPM} git rev is not a 40-hex commit hash: {spec!r}"
