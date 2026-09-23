"""Tests for REQ-TORCHSUMMARY-01 — migrate the embeddings feature off torch-summary.

The artifact under test is ``pixi.toml`` (the ``[feature.embeddings]`` conda
and pypi dependency tables) plus the ``external/champollion_V1`` git submodule
pointer recorded in this repo's own index.

Background: the only caller of ``torch-summary``/``torchsummary`` was
``external/champollion_V1/champollion/train.py``, which imported
``from torchsummary import summary``. Upstream fixed that import to
``from torchinfo import summary`` in commit ``5f08c9f1`` on
``champollion_V1``'s ``main`` branch. This repo's submodule pointer has not
been bumped to (or past) that commit yet, and ``pixi.toml`` still lists
``torch-summary`` as a pypi-dependency with ``torchinfo`` commented out.

Satisfying REQ-TORCHSUMMARY-01 requires all of:
  (a) the ``external/champollion_V1`` submodule pointer recorded in this
      repo's git index is at or after commit ``5f08c9f1``;
  (b) ``torchinfo`` is declared as an active (non-commented)
      ``[feature.embeddings.dependencies]`` conda dependency;
  (c) ``torch-summary``/``torchsummary`` no longer appears anywhere in
      ``pixi.toml``.
"""

import subprocess
from pathlib import Path

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
PIXI_TOML = REPO_ROOT / "pixi.toml"
CHAMPOLLION_SUBMODULE = "external/champollion_V1"

# The commit on champollion_V1's main branch that replaced the torchsummary
# import in train.py with torchinfo.
TORCHINFO_FIX_COMMIT = "5f08c9f136cfcc386418ab85b786adf39818acbf"


@pytest.fixture(scope="module")
def pixi_config() -> dict:
    """Parsed contents of the pipeline's ``pixi.toml``."""
    with PIXI_TOML.open("rb") as handle:
        return tomllib.load(handle)


@pytest.fixture(scope="module")
def pixi_toml_text() -> str:
    """Raw text of ``pixi.toml``, including comments."""
    return PIXI_TOML.read_text()


def _pinned_submodule_commit(submodule: str) -> str:
    """The commit SHA this repo's git index currently pins ``submodule`` at.

    Read from ``git ls-tree HEAD`` rather than the submodule's own checked-out
    HEAD, since a submodule's working tree can be updated locally (e.g. by an
    upstream fetch) without the superproject's pointer having been bumped and
    committed yet — it is the index-recorded pointer this requirement is
    about.
    """
    output = subprocess.run(
        ["git", "ls-tree", "HEAD", "--", submodule],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert output, f"git ls-tree HEAD reports no entry for {submodule}"
    # Format: "160000 commit <sha>\t<path>"
    return output.split()[2]


@pytest.mark.smoke
class TestChampollionV1SubmodulePointer:
    """REQ-TORCHSUMMARY-01: submodule pointer is at or past the torchinfo fix."""

    def test_submodule_path_exists(self):
        assert (REPO_ROOT / CHAMPOLLION_SUBMODULE / ".git").exists(), (
            f"{CHAMPOLLION_SUBMODULE} submodule is not checked out"
        )

    def test_pinned_commit_is_at_or_past_torchinfo_fix(self):
        """This repo's index pins champollion_V1 at/after commit 5f08c9f1.

        Verified via ``git merge-base --is-ancestor``: the fix commit must be
        an ancestor of (or equal to) the pinned commit. Today the pin is at
        the older commit 00bb86df, which predates the fix, so this is
        expected to fail.
        """
        pinned_commit = _pinned_submodule_commit(CHAMPOLLION_SUBMODULE)
        submodule_dir = REPO_ROOT / CHAMPOLLION_SUBMODULE
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", TORCHINFO_FIX_COMMIT, pinned_commit],
            cwd=submodule_dir,
        )
        assert result.returncode == 0, (
            f"champollion_V1 is pinned at {pinned_commit}, which is not at or "
            f"after the torchinfo-fix commit {TORCHINFO_FIX_COMMIT}; the "
            "submodule pointer must be bumped"
        )


@pytest.mark.smoke
class TestPixiTomlTorchinfoMigration:
    """REQ-TORCHSUMMARY-01: pixi.toml depends on torchinfo, not torch-summary."""

    def test_pixi_toml_exists(self):
        assert PIXI_TOML.is_file(), f"{PIXI_TOML} not found"

    def test_torchinfo_is_an_active_embeddings_conda_dependency(self, pixi_config):
        """torchinfo is declared (uncommented) under [feature.embeddings.dependencies]."""
        deps = pixi_config["feature"]["embeddings"]["dependencies"]
        assert "torchinfo" in deps, (
            "pixi.toml's [feature.embeddings.dependencies] has no active "
            "'torchinfo' entry; it is still commented out"
        )

    def test_torch_summary_pypi_dependency_removed(self, pixi_config):
        """torch-summary is no longer an embeddings pypi-dependency."""
        pypi_deps = pixi_config.get("feature", {}).get("embeddings", {}).get("pypi-dependencies", {})
        assert "torch-summary" not in pypi_deps, (
            "pixi.toml's [feature.embeddings.pypi-dependencies] still lists "
            "'torch-summary'"
        )

    def test_torch_summary_string_absent_from_manifest(self, pixi_toml_text):
        """Neither 'torch-summary' nor 'torchsummary' appears anywhere in pixi.toml."""
        lowered = pixi_toml_text.lower()
        assert "torch-summary" not in lowered, "pixi.toml still contains 'torch-summary'"
        assert "torchsummary" not in lowered, "pixi.toml still contains 'torchsummary'"
