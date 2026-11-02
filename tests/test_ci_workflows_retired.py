#!/usr/bin/env python3
"""
Retirement of the DockerHub champollion-base build and push workflow.

Run with: pixi run test-specific tests/test_ci_workflows_retired.py

User decision (2026-10-28, confirmed 2026-10-30): the GitHub workflow that
built `.github/Dockerfile.base` and pushed `bdrabczuk/champollion-base` is
retired, not aligned with the rewritten env-only Dockerfile.base (which now
needs PIXI_VERSION and PIXI_SHA256 build arguments that workflow never passed).

REQ-E2ECI-BDRABCZUK-4E9CB67D078A: no file under .github/workflows/ references
the path .github/Dockerfile.base.
REQ-E2ECI-BDRABCZUK-054CBC9EF112: no file under .github/workflows/ references
the champollion-base image name.
"""

from pathlib import Path

import pytest

WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / ".github" / "workflows"

DOCKERFILE_BASE_PATH = ".github/Dockerfile.base"
BASE_IMAGE_NAME = "champollion-base"


def _workflow_files() -> list[Path]:
    """Every file under .github/workflows/, recursively."""
    return sorted(path for path in WORKFLOWS_DIR.rglob("*") if path.is_file())


def _files_containing(needle: str) -> list[str]:
    """Paths (relative to .github/workflows/) of files whose text contains needle."""
    return [
        str(path.relative_to(WORKFLOWS_DIR))
        for path in _workflow_files()
        if needle in path.read_text(encoding="utf-8", errors="replace")
    ]


@pytest.mark.smoke
def test_workflows_directory_holds_ci_workflow():
    """Anchor: the scan is not vacuous; ci.yml stays in .github/workflows/."""
    names = [path.name for path in _workflow_files()]
    assert "ci.yml" in names, f".github/workflows/ holds no ci.yml; found {names}"


@pytest.mark.smoke
def test_no_workflow_references_dockerfile_base():
    """REQ-E2ECI-BDRABCZUK-4E9CB67D078A: no workflow file names .github/Dockerfile.base."""
    offenders = _files_containing(DOCKERFILE_BASE_PATH)
    assert not offenders, f"Workflow files referencing {DOCKERFILE_BASE_PATH}: {offenders}"


@pytest.mark.smoke
def test_no_workflow_references_champollion_base_image():
    """REQ-E2ECI-BDRABCZUK-054CBC9EF112: no workflow file names the champollion-base image."""
    offenders = _files_containing(BASE_IMAGE_NAME)
    assert not offenders, f"Workflow files referencing the {BASE_IMAGE_NAME} image: {offenders}"
