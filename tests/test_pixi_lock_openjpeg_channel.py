"""Tests for REQ-LOCK-01 — embeddings/training locks must not take openjpeg from neuro-forge.

The artifact under test is the committed ``pixi.lock`` (lock format v7).

Background (TASK-092): ``pixi.lock`` pinned
``https://brainvisa.info/neuro-forge/linux-64/openjpeg-2.5.2-hb0f4dca_0.conda``
in the ``embeddings`` and ``training`` environments on both locked platforms.
That artifact returns HTTP 404 (removed server-side while still listed in the
channel's repodata), so ``pixi install -e training`` fails on any fresh
machine. ``openjpeg`` is a transitive dependency; it resolves from neuro-forge
only because that channel is listed first in ``[workspace] channels`` — the
embeddings/training environments have no BrainVISA dependency at all.

REQ-LOCK-02 (TASK-093) generalises this to every environment: the same dead
artifact is still locked in ``default``, ``brainvisa`` and ``cortical-tiles``,
so a fresh ``pixi install`` of any of those fails too.

These tests are offline: they parse the lock file (and, for REQ-LOCK-02, the
manifest's environment list) and never contact a channel.
"""

from pathlib import Path

import pytest
import tomllib
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PIXI_LOCK = REPO_ROOT / "pixi.lock"
PIXI_TOML = REPO_ROOT / "pixi.toml"

# REQ-LOCK-02: conda artifacts known to be unavailable upstream (HTTP 404).
KNOWN_UNAVAILABLE_ARTIFACTS = ("https://brainvisa.info/neuro-forge/linux-64/openjpeg-2.5.2-hb0f4dca_0.conda",)


def _load_manifest() -> dict:
    with PIXI_TOML.open("rb") as handle:
        return tomllib.load(handle)


def _manifest_environments() -> list[str]:
    """Every environment declared in ``pixi.toml`` (``default`` included)."""
    return sorted(_load_manifest()["environments"])


TARGET_ENVIRONMENTS = ("embeddings", "training")
FORBIDDEN_CHANNEL = "https://brainvisa.info/neuro-forge/"
PACKAGE_NAME = "openjpeg"


@pytest.fixture(scope="module")
def pixi_lock() -> dict:
    """Parsed contents of the pipeline's ``pixi.lock``."""
    with PIXI_LOCK.open() as handle:
        return yaml.safe_load(handle)


def _conda_urls(pixi_lock: dict, environment: str) -> dict[str, list[str]]:
    """Conda package URLs in ``environment``'s locked closure, keyed by platform."""
    packages_by_platform = pixi_lock["environments"][environment]["packages"]
    return {
        platform: [entry["conda"] for entry in entries if "conda" in entry]
        for platform, entries in packages_by_platform.items()
    }


def _is_package(url: str, name: str) -> bool:
    """True if the conda artifact at ``url`` is package ``name`` (any version/build)."""
    basename = url.rsplit("/", 1)[-1]
    return basename.startswith(f"{name}-") and basename[len(name) + 1 : len(name) + 2].isdigit()


@pytest.mark.smoke
class TestOpenjpegNotLockedFromNeuroForge:
    """REQ-LOCK-01: embeddings/training must not resolve openjpeg from neuro-forge."""

    @pytest.mark.parametrize("environment", TARGET_ENVIRONMENTS)
    def test_environment_locks_every_declared_platform(self, pixi_lock, environment):
        """Guard: the environment is locked for every workspace platform (no vacuous pass)."""
        declared = {platform["name"] for platform in pixi_lock["platforms"]}
        locked = set(_conda_urls(pixi_lock, environment))
        assert declared and locked == declared, (
            f"pixi.lock '{environment}' environment is locked for {sorted(locked)}, "
            f"expected every workspace platform {sorted(declared)}"
        )

    @pytest.mark.parametrize("environment", TARGET_ENVIRONMENTS)
    def test_openjpeg_not_resolved_from_neuro_forge(self, pixi_lock, environment):
        """No locked platform of the environment takes openjpeg from brainvisa.info/neuro-forge."""
        offending = {
            platform: url
            for platform, urls in _conda_urls(pixi_lock, environment).items()
            for url in urls
            if _is_package(url, PACKAGE_NAME) and url.startswith(FORBIDDEN_CHANNEL)
        }
        assert not offending, (
            f"pixi.lock '{environment}' environment resolves '{PACKAGE_NAME}' from "
            f"{FORBIDDEN_CHANNEL} (artifact is HTTP 404 upstream, breaking fresh "
            f"`pixi install -e {environment}`): {offending}"
        )


def _expected_platforms(pixi_lock: dict, environment: str) -> set[str]:
    """Platforms ``environment`` must be locked for.

    Every workspace platform, narrowed by any ``platforms`` restriction on the
    environment's features (e.g. ``feature.docs`` is ``linux-64`` only).
    """
    manifest = _load_manifest()
    expected = {platform["name"] for platform in pixi_lock["platforms"]}
    env_spec = manifest["environments"][environment]
    features = env_spec["features"] if isinstance(env_spec, dict) else env_spec
    for feature in features:
        restriction = manifest.get("feature", {}).get(feature, {}).get("platforms")
        if restriction is not None:
            expected &= set(restriction)
    return expected


@pytest.mark.smoke
class TestNoKnownUnavailableArtifactLocked:
    """REQ-LOCK-02: no environment, on any platform, locks a known-dead artifact."""

    @pytest.mark.parametrize("environment", _manifest_environments())
    def test_environment_locks_every_expected_platform(self, pixi_lock, environment):
        """Guard: every manifest environment is locked for all of its platforms."""
        assert environment in pixi_lock["environments"], (
            f"pixi.toml environment '{environment}' is missing from pixi.lock"
        )
        expected = _expected_platforms(pixi_lock, environment)
        locked = set(_conda_urls(pixi_lock, environment))
        assert expected and locked == expected, (
            f"pixi.lock '{environment}' environment is locked for {sorted(locked)}, expected {sorted(expected)}"
        )

    @pytest.mark.parametrize("environment", _manifest_environments())
    def test_no_known_unavailable_artifact(self, pixi_lock, environment):
        """No locked platform of the environment references a known-unavailable URL."""
        offending = {
            platform: url
            for platform, urls in _conda_urls(pixi_lock, environment).items()
            for url in urls
            if url in KNOWN_UNAVAILABLE_ARTIFACTS
        }
        assert not offending, (
            f"pixi.lock '{environment}' environment references artifact(s) known "
            f"to be unavailable upstream (HTTP 404, breaking fresh "
            f"`pixi install -e {environment}`): {offending}"
        )
