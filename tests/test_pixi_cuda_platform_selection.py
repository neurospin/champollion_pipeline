"""Tests for REQ-PIXICFG-02 — restore CUDA-vs-CPU pytorch selection in pixi.toml.

The artifact under test is ``pixi.toml``'s ``[workspace] platforms`` list
(and, for the docs-env-scoping clause, ``[feature.docs]``'s own
``platforms`` list).

Background: commit ``8af06e4``/``8af0c04`` (see REQ-PIXICFG-01/TASK-047's
ledger note) simplified the workspace's dual-platform declaration:

.. code-block:: toml

    platforms = [
      { name = "linux-64-cuda", platform = "linux-64", cuda = "13.2" },
      { name = "linux-64u", platform = "linux-64" }]

down to a single bare-string entry:

.. code-block:: toml

    platforms = ["linux-64"]

That removed pixi's only mechanism for choosing a ``pytorch-*-cuda*`` conda
build variant over a ``pytorch-*-cpu_*`` one for the plain ``pytorch``
dependency already declared under ``[feature.embeddings.dependencies]`` —
confirmed this session: with the single-platform manifest, ``pytorch``
always resolves to a ``cpu_mkl`` build, even on a CUDA-capable machine.

Satisfying REQ-PIXICFG-02 requires:
  (a) ``[workspace] platforms`` is restored to more than one platform entry,
      with at least one being a named-platform table carrying an inline
      ``cuda = "..."`` virtual-package floor (pixi's documented, current
      mechanism for CUDA/CPU build-variant selection — NOT the deprecated
      ``[system-requirements]`` table);
  (b) ``[feature.docs]`` declares its own single-entry ``platforms`` list,
      so the ``docs`` environment (which has no CUDA-relevant dependency)
      keeps solving against exactly one platform instead of being forced
      to solve against every platform in the now-multi-platform workspace
      list — the regression that (incorrectly) motivated the original
      single-platform simplification.
"""

from pathlib import Path

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
PIXI_TOML = REPO_ROOT / "pixi.toml"


@pytest.fixture(scope="module")
def pixi_config() -> dict:
    """Parsed contents of the pipeline's ``pixi.toml``."""
    with PIXI_TOML.open("rb") as handle:
        return tomllib.load(handle)


def _has_cuda_platform_entry(platforms: list) -> bool:
    """True if any entry in a ``platforms`` list is a table with a ``cuda`` key."""
    return any(isinstance(entry, dict) and "cuda" in entry for entry in platforms)


@pytest.mark.smoke
class TestWorkspaceCudaPlatformSelection:
    """REQ-PIXICFG-02: [workspace] platforms restores CUDA-vs-CPU selection."""

    def test_pixi_toml_exists(self):
        assert PIXI_TOML.is_file(), f"{PIXI_TOML} not found"

    def test_workspace_declares_more_than_one_platform(self, pixi_config):
        """[workspace] platforms is not the single bare-string ["linux-64"] list."""
        platforms = pixi_config["workspace"]["platforms"]
        assert len(platforms) > 1, (
            "pixi.toml's [workspace] platforms has only one entry "
            f"({platforms!r}); the CUDA-vs-CPU platform declaration removed "
            "by the single-platform simplification has not been restored"
        )

    def test_workspace_has_a_named_platform_with_cuda_floor(self, pixi_config):
        """At least one [workspace] platforms entry is a table with a 'cuda' key.

        This is pixi's current, documented mechanism (a named platform with
        an inline ``cuda = "X.Y"`` virtual-package requirement) for letting
        the solver pick a CUDA build variant of the plain ``pytorch``
        dependency on CUDA-capable machines. The old ``[system-requirements]``
        table is a different, deprecated mechanism and does not satisfy this.
        """
        platforms = pixi_config["workspace"]["platforms"]
        assert _has_cuda_platform_entry(platforms), (
            "pixi.toml's [workspace] platforms has no named-platform table "
            f"entry carrying a 'cuda' key ({platforms!r})"
        )


@pytest.mark.smoke
class TestDocsFeaturePlatformScoping:
    """REQ-PIXICFG-02: [feature.docs] stays scoped to a single platform.

    Restoring a multi-platform [workspace] list must not force the docs
    environment (which has no CUDA-relevant dependency) to solve against
    more than one platform — that regression is what (incorrectly)
    motivated the original single-platform simplification.
    """

    def test_docs_feature_declares_its_own_single_platform(self, pixi_config):
        docs_feature = pixi_config.get("feature", {}).get("docs", {})
        docs_platforms = docs_feature.get("platforms")
        assert docs_platforms is not None, (
            "pixi.toml's [feature.docs] does not declare its own 'platforms' "
            "list; once [workspace] platforms has more than one entry, the "
            "docs environment would be solved against all of them"
        )
        assert len(docs_platforms) == 1, (
            f"pixi.toml's [feature.docs] platforms is {docs_platforms!r}; "
            "expected exactly one platform so the docs environment isn't "
            "solved against CUDA-selection platforms it doesn't need"
        )
