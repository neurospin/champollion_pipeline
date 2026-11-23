#!/usr/bin/env python3
"""
Stage 6 UMAP stays opt-in on a user-named reference (TASK-MODELVER-BDRABCZUK-39D727185A86; user decision
2026-11-19 12:13: the UMAP reference is fitted on UK Biobank embeddings, can never be published and stays local).

- REQ-MODELVER-BDRABCZUK-1F9769CEF5B3 (stakeholder): UMAP snapshots render only from a reference directory the
  user names.
- REQ-MODELVER-BDRABCZUK-BB34F3469BFD: GenerateSnapshots run without --reference_data_dir warns once that UMAP
  is skipped, naming --reference_data_dir.
- REQ-MODELVER-BDRABCZUK-2C4BE1D9C176: main.py with no reference_data_path passes no --reference_data_dir.
- REQ-MODELVER-BDRABCZUK-8ED5271F31DB: the repository tracks no umap_<region>_<hemisphere> reference file.
- REQ-MODELVER-BDRABCZUK-FCBF2DDB548A: the docs/usage.md UMAP Visualization subsection (moved from README.md by
  TASK-DOCS-BDRABCZUK-84BE8AB6FC73) says UMAP snapshots need a
  locally fitted UMAP reference of the same Mask version as the embeddings (user decision 2026-11-19 13:05).

The main.py stage path without reference data is covered by tests/test_main_snapshots_skip_no_reference.py
(REQ-INPUTKIND-BDRABCZUK-5B93C6EB8F51 and siblings). Synthetic embeddings only; the UMAP renderer is a mock.
"""

import logging
import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.generate_snapshots import GenerateSnapshots
from tests.main_stage_helpers import load_main_module

pytestmark = pytest.mark.unit  # noqa: V107

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOTS_LOGGER = "champollion_pipeline.generate_snapshots"
_RENDER_UMAP = "champollion_pipeline.generate_snapshots.generate_umap_snapshot"
_REFERENCE_FILE = re.compile(r"^umap_.+_(left|right)(\.pkl|_coords\.npy)$")
_UMAP_SUBSECTION = re.compile(r"^###\s+UMAP Visualization\s*$(.*?)(?=^##|\Z)", re.MULTILINE | re.DOTALL)


def _embeddings_dir(root: Path) -> Path:
    """A combined embeddings folder holding one region CSV."""
    embeddings = root / "embeddings"
    embeddings.mkdir(parents=True)
    (embeddings / "S.Or._left_embeddings.csv").write_text("ID,dim1,dim2\nsub-01,0.1,0.2\n")
    return embeddings


def _run_umap_only(root: Path):
    """Run GenerateSnapshots --umap-only on a real embeddings folder, no --reference_data_dir; return the mock."""
    script = GenerateSnapshots()
    script.parse_args(
        ["--output_dir", str(root / "out"), "--embeddings_dir", str(_embeddings_dir(root)), "--umap-only"]
    )
    with patch(_RENDER_UMAP, return_value=[]) as render_umap:
        script.run()
    return render_umap


class TestUmapOnlyFromNamedReference:
    """REQ-MODELVER-BDRABCZUK-1F9769CEF5B3: no reference directory named -> no UMAP render."""

    def test_no_umap_render_without_reference(self, tmp_path):
        assert not _run_umap_only(tmp_path).called


class TestSnapshotsWarnWithoutReference:
    """REQ-MODELVER-BDRABCZUK-BB34F3469BFD: one WARNING naming UMAP and --reference_data_dir."""

    def test_one_warning_names_reference_option(self, tmp_path, pipeline_caplog):
        _run_umap_only(tmp_path)
        warnings = [
            record.getMessage()
            for record in pipeline_caplog.records
            if record.levelno == logging.WARNING
            and record.name == SNAPSHOTS_LOGGER
            and "UMAP" in record.getMessage()
            and "--reference_data_dir" in record.getMessage()
        ]
        assert len(warnings) == 1, warnings


class TestMainPassesNoReference:
    """REQ-MODELVER-BDRABCZUK-2C4BE1D9C176: default main.py config -> no --reference_data_dir for stage 6."""

    def test_stage6_argv_has_no_reference_option(self, tmp_path):
        main_module = load_main_module("_champollion_main_umap_reference_opt_in_under_test")
        config = main_module.PipelineConfig(models_path="/models")
        config.dataset.datasets_root = str(tmp_path / "MYDATASET")  # noqa: V101
        stage = main_module.GenerateSnapshotsStage("generate_snapshots", config, logging.getLogger(__name__))
        argv = stage.build_args()
        assert not [token for token in argv if token.startswith("--reference_data_dir")], argv


class TestNoTrackedReference:
    """REQ-MODELVER-BDRABCZUK-8ED5271F31DB: git tracks no umap_<region>_<hemisphere> .pkl or _coords.npy file."""

    def test_no_tracked_reference_file(self):
        listing = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files"], capture_output=True, text=True, check=True
        ).stdout.splitlines()
        assert listing, "git ls-files listed nothing"
        tracked = [path for path in listing if _REFERENCE_FILE.match(Path(path).name)]
        assert not tracked, tracked


class TestReadmeUmapReferenceVersion:
    """REQ-MODELVER-BDRABCZUK-FCBF2DDB548A: docs/usage.md says the UMAP reference is fitted locally on the version."""

    def test_umap_section_states_local_same_version_reference(self):
        section = _UMAP_SUBSECTION.search((REPO_ROOT / "docs" / "usage.md").read_text(encoding="utf-8"))
        assert section, "docs/usage.md has no '### UMAP Visualization' subsection"
        sentences = re.split(r"(?<=[.!?])\s+", " ".join(section.group(1).split()))
        stating = [
            sentence
            for sentence in sentences
            if re.search(r"\blocally fitted\b", sentence, re.IGNORECASE)
            and re.search(r"\bUMAP reference\b", sentence, re.IGNORECASE)
            and re.search(r"\bsame Mask version\b", sentence, re.IGNORECASE)
        ]
        assert stating, "no UMAP Visualization sentence names a locally fitted UMAP reference of the same Mask version"
