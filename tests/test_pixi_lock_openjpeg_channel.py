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

These tests are offline: they parse the lock file and never contact a channel.
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PIXI_LOCK = REPO_ROOT / "pixi.lock"

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
