"""Tests for REQ-DOCS-16 / REQ-DOCS-17 — setup docs and the SLURM template.

# TASK-095

``docs/setup.md`` routes Jean-Zay users; it must point at the tracked
training template and stop pointing at the gitignored, cluster-local
``slurm/`` directory that no clone contains.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke  # noqa: V107

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETUP_MD = PROJECT_ROOT / "docs" / "setup.md"
TEMPLATE_REL = "scripts/templates/train_champollion.slurm.example"


class TestSetupDocsSlurmTemplate:
    def test_setup_md_names_training_template(self) -> None:
        """REQ-DOCS-16: setup.md names the tracked template path."""
        assert TEMPLATE_REL in SETUP_MD.read_text(encoding="utf-8")

    def test_setup_md_has_no_untracked_slurm_dir_reference(self) -> None:
        """REQ-DOCS-17: setup.md has no bare ``slurm/`` path token."""
        text = SETUP_MD.read_text(encoding="utf-8")
        hits = [m.group(0) for m in re.finditer(r"(?<![\w./])slurm/", text)]
        assert not hits, "docs/setup.md still references the untracked `slurm/` directory"
