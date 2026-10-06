"""Tests for REQ-PIXIDEPS-01 — every pixi environment that can run
``src/compare.py`` declares scikit-learn and matplotlib directly.

``src/compare.py`` imports ``sklearn`` and ``matplotlib`` at module top for
every mode, so each of the ``default``, ``embeddings``, ``training``,
``brainvisa`` and ``cortical-tiles`` environments must declare both packages
through the features composing it — not merely receive them as transitive
dependencies of something else. The top-level ``[dependencies]`` table is
pixi's default feature and only counts for an environment that does not set
``no-default-feature = true``.

Each declared constraint must also admit the versions ``pixi.lock`` resolves
today across those environments and platforms (scikit-learn 1.9.0 and 1.9.1,
matplotlib 3.9.1 and 3.11.2), so the declaration forces no re-solve conflict.

These tests parse ``pixi.toml`` only: no network, no solved environment.
"""

import re
from pathlib import Path

import pytest
import tomllib

PIXI_TOML = Path(__file__).resolve().parents[1] / "pixi.toml"

COMPARE_ENVIRONMENTS = ("default", "embeddings", "training", "brainvisa", "cortical-tiles")
LOCKED_VERSIONS = {
    "scikit-learn": ("1.9.0", "1.9.1"),
    "matplotlib": ("3.9.1", "3.11.2"),
}

_CLAUSE = re.compile(r"^(>=|<=|==|!=|>|<|=)?\s*([0-9][0-9A-Za-z.*]*)$")


@pytest.fixture(scope="module")
def pixi_config() -> dict:
    """Parsed contents of the pipeline's ``pixi.toml``."""
    with PIXI_TOML.open("rb") as handle:
        return tomllib.load(handle)


def _environment_features(config: dict, env_name: str) -> list[str]:
    """Return the feature names composing an environment, in declared order."""
    env = config.get("environments", {})[env_name]
    if isinstance(env, list):
        features, no_default = list(env), False
    else:
        features = list(env.get("features", []))
        no_default = bool(env.get("no-default-feature", False))
    return features if no_default else ["<default>", *features]


def _feature_dependencies(config: dict, feature: str) -> dict:
    """Return a feature's conda dependency table (top-level for the default feature)."""
    if feature == "<default>":
        return config.get("dependencies", {})
    return config.get("feature", {}).get(feature, {}).get("dependencies", {})


def _declared_specs(config: dict, env_name: str, package: str) -> list[tuple[str, str]]:
    """Return ``(feature, version spec)`` for each feature of the env declaring the package."""
    specs = []
    for feature in _environment_features(config, env_name):
        spec = _feature_dependencies(config, feature).get(package)
        if spec is None:
            continue
        if isinstance(spec, dict):
            spec = spec.get("version", "*")
        specs.append((feature, str(spec)))
    return specs


def _version_key(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".") if part.isdigit())


def _clause_admits(clause: str, version: str) -> bool:
    """Evaluate one conda match-spec version clause (``>=3.5``, ``3.9.*``, ...)."""
    clause = clause.strip()
    if clause in ("", "*"):
        return True
    match = _CLAUSE.match(clause)
    if match is None:
        raise ValueError(f"unsupported version clause {clause!r}")
    operator, bound = match.group(1), match.group(2)
    if operator in (None, "=") or bound.endswith("*"):
        # Conda fuzzy match: bare ``3.9``, ``=3.9`` and ``3.9.*`` all mean "3.9.x".
        prefix = _version_key(bound.rstrip(".*"))
        inside = _version_key(version)[: len(prefix)] == prefix
        return not inside if operator == "!=" else inside
    have, want = _version_key(version), _version_key(bound)
    return {
        ">=": have >= want,
        "<=": have <= want,
        ">": have > want,
        "<": have < want,
        "==": have == want,
        "!=": have != want,
    }[operator]


def _spec_admits(spec: str, version: str) -> bool:
    """A comma-separated spec admits a version when every clause does."""
    return all(_clause_admits(clause, version) for clause in spec.split(","))


ENV_PACKAGE_PAIRS = [(env, pkg) for env in COMPARE_ENVIRONMENTS for pkg in LOCKED_VERSIONS]


@pytest.mark.smoke
class TestComparePixiEnvironmentDependencies:
    """REQ-PIXIDEPS-01: compare-capable environments declare sklearn and matplotlib."""

    @pytest.mark.parametrize(("env_name", "package"), ENV_PACKAGE_PAIRS)
    def test_environment_declares_package(self, pixi_config, env_name, package):
        """The features composing the environment declare the package directly."""
        features = _environment_features(pixi_config, env_name)
        assert _declared_specs(pixi_config, env_name, package), (
            f"REQ-PIXIDEPS-01: pixi environment '{env_name}' (features {features}) does not "
            f"declare '{package}' in any of its features' dependency tables; src/compare.py "
            "imports it at module top, so the environment only gets it transitively"
        )

    @pytest.mark.parametrize(("env_name", "package"), ENV_PACKAGE_PAIRS)
    def test_declared_constraint_admits_locked_versions(self, pixi_config, env_name, package):
        """Every declaration of the package admits the versions locked today."""
        specs = _declared_specs(pixi_config, env_name, package)
        if not specs:
            pytest.fail(
                f"REQ-PIXIDEPS-01: pixi environment '{env_name}' declares no '{package}' "
                "constraint to check against the locked versions"
            )
        for feature, spec in specs:
            for version in LOCKED_VERSIONS[package]:
                assert _spec_admits(spec, version), (
                    f"REQ-PIXIDEPS-01: '{package} = \"{spec}\"' in feature '{feature}' of "
                    f"environment '{env_name}' rejects locked version {version}"
                )
