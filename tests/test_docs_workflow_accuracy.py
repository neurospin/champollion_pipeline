"""Tests for REQ-DOCS-14 and REQ-DOCS-15 — docs/workflow.md accuracy.

REQ-DOCS-14: every Hydra platform config named in ``docs/*.md`` (as
``platform/<name>.yaml`` or ``platform=<name>``) exists in the pinned
``external/champollion_V1`` checkout under ``champollion/configs/platform/``.

REQ-DOCS-15: no ``docs/*.md`` page contains ``create_dataset_config_files``
(that script no longer exists; region-YAML generation is inline in
``generate_champollion_config.py``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke  # noqa: V107

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
WORKFLOW_MD = DOCS_DIR / "workflow.md"
PLATFORM_DIR = PROJECT_ROOT / "external" / "champollion_V1" / "champollion" / "configs" / "platform"

PLATFORM_REF_RE = re.compile(r"platform/(?P<yaml>[\w.-]+?)\.yaml|platform=(?P<eq>[\w.-]+)")


def _doc_pages() -> list[Path]:
    return sorted(DOCS_DIR.glob("*.md"))


def _platform_refs() -> list[tuple[str, int, str]]:
    refs = []
    for page in _doc_pages():
        for lineno, line in enumerate(page.read_text().splitlines(), start=1):
            for match in PLATFORM_REF_RE.finditer(line):
                name = match.group("yaml") or match.group("eq")
                refs.append((page.name, lineno, name))
    return refs


@pytest.fixture
def platform_dir() -> Path:
    if not PLATFORM_DIR.is_dir():
        pytest.skip(f"champollion_V1 submodule not checked out: {PLATFORM_DIR}")
    return PLATFORM_DIR


class TestDocsPlatformConfigsExist:
    """REQ-DOCS-14."""

    def test_pinned_platform_config_dir_is_populated(self, platform_dir: Path) -> None:
        assert list(platform_dir.glob("*.yaml")), f"no platform configs in {platform_dir}; check would pass vacuously"

    def test_every_documented_platform_config_exists(self, platform_dir: Path) -> None:
        available = sorted(p.stem for p in platform_dir.glob("*.yaml"))
        missing = [
            f"docs/{page}:{lineno}: platform '{name}'"
            for page, lineno, name in _platform_refs()
            if not (platform_dir / f"{name}.yaml").is_file()
        ]
        assert not missing, (
            "docs reference platform configs absent from the pinned "
            f"champollion_V1 (available: {available}):\n" + "\n".join(missing)
        )


class TestDocsRegionYamlAttribution:
    """REQ-DOCS-15."""

    def test_no_doc_mentions_create_dataset_config_files(self) -> None:
        hits = [
            f"docs/{page.name}:{lineno}: {line.strip()}"
            for page in _doc_pages()
            for lineno, line in enumerate(page.read_text().splitlines(), start=1)
            if "create_dataset_config_files" in line
        ]
        assert not hits, (
            "docs name the removed create_dataset_config_files script; region "
            "YAMLs are generated inline by generate_champollion_config.py:\n" + "\n".join(hits)
        )

    def test_workflow_still_names_generate_champollion_config(self) -> None:
        assert "generate_champollion_config.py" in WORKFLOW_MD.read_text()
