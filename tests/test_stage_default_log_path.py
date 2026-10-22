#!/usr/bin/env python3
"""
Run-log path of a bare --log for each pipeline stage script (RUNLOG epic, "run-log stage set").

champollion_utils ScriptBuilder (>= 0.3.0) owns --log [PATH]; a bare --log writes to
default_log_path(). Each stage script overrides it so the run log lands in the stage's derivatives
folder: <derivatives folder>/.log/<stage>_<YYYYmmdd-HHMMSS>.log (local start time), where the
derivatives folder comes from derivatives_layout.compute_run_log_derivatives_dir:

- generate_morphologist_graphs: <output>/derivatives/morphologist-6.0;
- run_cortical_tiles: <output>/cortical_tiles-<CORTICAL_TILES_VERSION> (<output> is the derivatives root);
- generate_champollion_config: <D>/<dataset>/derivatives/champollion_V1/<masks>;
- generate_embeddings: <datasets_root>/derivatives/champollion_V1/<masks> (masks as resolved at parse time);
- train_champollion: <pipeline>/data/<dataset>/derivatives/champollion_V1/<DEFAULT_MASKS_VERSION>.

put_together_embeddings and generate_snapshots parse no dataset root or masks version (user decision
2026-10-22): their run log lands in <output>/.log/<stage>_<YYYYmmdd-HHMMSS>.log, <output> being
--output_path (combine) or --output_dir (snapshots).

Each test parses a command line, then calls default_log_path(); nothing runs and nothing is written.
The start time is bracketed by two clock reads, so the expected file name is one of two stamps.
"""

from datetime import datetime
from os.path import join

import pytest

from champollion_pipeline.derivatives_layout import (
    DEFAULT_MASKS_VERSION,
    compute_run_log_derivatives_dir,
    compute_run_log_path,
)
from champollion_pipeline.generate_champollion_config import GenerateChampollionConfig
from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from champollion_pipeline.generate_morphologist_graphs import GenerateMorphologistGraphs
from champollion_pipeline.generate_snapshots import GenerateSnapshots
from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings
from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from champollion_pipeline.train_champollion import _PIPELINE_DATA_DIR, TrainChampollion

ROOT = "/data/myproject"
MASK_VERSION_DIR = join(ROOT, "derivatives", "champollion_V1", "canonical_26")
HF_REPO = "neurospin/Champollion_V1"


def _default_log_path_and_candidates(script, argv, stage, derivatives_dir):
    """Parse argv, call default_log_path(); return it and the paths allowed for a start time around the call."""
    script.parse_args(argv)
    before = datetime.now().astimezone()
    path = script.default_log_path()
    after = datetime.now().astimezone()
    candidates = {compute_run_log_path(derivatives_dir, stage, moment) for moment in (before, after)}
    return path, candidates


# ---------------------------------------------------------------------------
# TASK-RUNLOG-BDRABCZUK-C60AD2ED7B4D: generate_morphologist_graphs
# ---------------------------------------------------------------------------


# REQ-RUNLOG-BDRABCZUK-EA3AC88C3719
def test_morphologist_bare_log_writes_under_morphologist_derivatives():
    derivatives_dir = compute_run_log_derivatives_dir("morphologist", ROOT)
    path, candidates = _default_log_path_and_candidates(
        GenerateMorphologistGraphs(), ["/raw/t1", ROOT, "--log"], "morphologist", derivatives_dir
    )
    assert path in candidates


# ---------------------------------------------------------------------------
# TASK-RUNLOG-BDRABCZUK-4472DFFF87D5: run_cortical_tiles
# ---------------------------------------------------------------------------


# REQ-RUNLOG-BDRABCZUK-2EFE8BB038B1
@pytest.mark.parametrize("output", [join(ROOT, "derivatives"), join(ROOT, "derivatives") + "/"])
def test_cortical_tiles_bare_log_writes_under_cortical_tiles_derivatives(output):
    derivatives_dir = compute_run_log_derivatives_dir("cortical_tiles", ROOT)
    argv = [
        join(ROOT, "derivatives", "morphologist-6.0", "subjects"),
        output,
        "--path_to_graph=t1mri/default_acquisition/default_analysis/folds/3.3/base",
        "--path_sk_with_hull=t1mri/default_acquisition/default_analysis/segmentation/mesh",
        "--log",
    ]
    path, candidates = _default_log_path_and_candidates(RunCorticalTiles(), argv, "cortical_tiles", derivatives_dir)
    assert path in candidates


# ---------------------------------------------------------------------------
# TASK-RUNLOG-BDRABCZUK-51209E0FEC5E: generate_champollion_config
# ---------------------------------------------------------------------------


# REQ-RUNLOG-BDRABCZUK-C75563E157BF
def test_config_bare_log_writes_under_mask_version_dir():
    derivatives_dir = compute_run_log_derivatives_dir("config", ROOT, masks_version="canonical_26")
    argv = [
        join(ROOT, "derivatives", "cortical_tiles-2026", "crops", "canonical_26", "2mm"),
        "--dataset=myproject",
        "--masks=canonical_26",
        "--log",
    ]
    path, candidates = _default_log_path_and_candidates(GenerateChampollionConfig(), argv, "config", derivatives_dir)
    assert path in candidates


# ---------------------------------------------------------------------------
# TASK-RUNLOG-BDRABCZUK-54C0C8EE3CCE: generate_embeddings
# ---------------------------------------------------------------------------


# REQ-RUNLOG-BDRABCZUK-ACFD5CA6ED3F
def test_embeddings_bare_log_writes_under_mask_version_dir():
    derivatives_dir = compute_run_log_derivatives_dir("embeddings", ROOT, masks_version="canonical_26")
    argv = [HF_REPO, ROOT, "--masks=canonical_26", "--log"]
    path, candidates = _default_log_path_and_candidates(GenerateEmbeddings(), argv, "embeddings", derivatives_dir)
    assert path in candidates


# REQ-RUNLOG-BDRABCZUK-ACFD5CA6ED3F
def test_embeddings_bare_log_uses_model_version_when_masks_absent():
    derivatives_dir = compute_run_log_derivatives_dir("embeddings", ROOT, masks_version="canonical_26")
    argv = [HF_REPO, ROOT, "--model-version=canonical_26", "--log"]
    path, candidates = _default_log_path_and_candidates(GenerateEmbeddings(), argv, "embeddings", derivatives_dir)
    assert path in candidates


# ---------------------------------------------------------------------------
# TASK-RUNLOG-BDRABCZUK-1412641FF6E6: train_champollion
# ---------------------------------------------------------------------------


# REQ-RUNLOG-BDRABCZUK-142A48F67165
def test_training_bare_log_writes_under_pipeline_data_mask_version_dir():
    derivatives_dir = compute_run_log_derivatives_dir(
        "training", join(_PIPELINE_DATA_DIR, "myproject"), masks_version=DEFAULT_MASKS_VERSION
    )
    argv = ["--dataset=myproject", "--region=cingulate_left", "--log"]
    path, candidates = _default_log_path_and_candidates(TrainChampollion(), argv, "training", derivatives_dir)
    assert path in candidates


# ---------------------------------------------------------------------------
# TASK-RUNLOG-BDRABCZUK-60F6A90AF3BB: put_together_embeddings
# ---------------------------------------------------------------------------


# REQ-RUNLOG-BDRABCZUK-625A2309610E
def test_combine_bare_log_writes_under_output_path():
    output_path = join(MASK_VERSION_DIR, "embeddings")
    argv = [join(MASK_VERSION_DIR, "region_embeddings"), f"--output_path={output_path}", "--log"]
    path, candidates = _default_log_path_and_candidates(PutTogetherEmbeddings(), argv, "combine", output_path)
    assert path in candidates


# ---------------------------------------------------------------------------
# TASK-RUNLOG-BDRABCZUK-3AAAAC19D96C: generate_snapshots
# ---------------------------------------------------------------------------


# REQ-RUNLOG-BDRABCZUK-7640FBBEE5EB
def test_snapshots_bare_log_writes_under_output_dir():
    output_dir = join(MASK_VERSION_DIR, "snapshots")
    argv = [f"--output_dir={output_dir}", f"--embeddings_dir={join(MASK_VERSION_DIR, 'embeddings')}", "--log"]
    path, candidates = _default_log_path_and_candidates(GenerateSnapshots(), argv, "snapshots", output_dir)
    assert path in candidates
