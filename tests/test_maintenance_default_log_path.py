#!/usr/bin/env python3
"""
Run-log path of a bare --log for the maintenance scripts (RUNLOG epic, TASK-RUNLOG-BDRABCZUK-11891E63CED1).

champollion_utils ScriptBuilder (>= 0.3.0) owns --log [PATH]; a bare --log writes to default_log_path().
User decision 2026-10-22: a maintenance script's bare --log writes to the dataset-level folder
<dataset>/derivatives/.log/<script name>_<YYYYmmdd-HHMMSS>.log (local start time).

- prune_failed_subjects: its output argument is the <dataset>/derivatives folder (the --output of
  run_cortical_tiles), so the run log lands in <output>/.log;
- purge_subject: its derivatives argument is <dataset>/derivatives/cortical_tiles-<version>/, so the
  run log lands in the parent of that folder, a trailing separator ignored.

generate_masks and generate_labelling_qc take no dataset root (user decision 2026-10-22): their run log
lands next to the output, as for combine and snapshots:

- generate_masks: <--output_dir>/.log/generate_masks_<YYYYmmdd-HHMMSS>.log;
- generate_labelling_qc: dirname(<output TSV>)/.log/generate_labelling_qc_<YYYYmmdd-HHMMSS>.log.

Each test parses a command line, then calls default_log_path(); nothing runs and nothing is written.
The start time is bracketed by two clock reads, so the expected file name is one of two stamps.
"""

from datetime import datetime
from os.path import join

import pytest

from champollion_pipeline.generate_labelling_qc import GenerateLabellingQC
from champollion_pipeline.generate_masks import GenerateMasks
from champollion_pipeline.prune_failed_subjects import PruneFailedSubjects
from champollion_pipeline.purge_subject import PurgeSubject

ROOT = "/data/myproject"
DATASET_DERIVATIVES = join(ROOT, "derivatives")
DATASET_LOG_DIR = join(DATASET_DERIVATIVES, ".log")
CORTICAL_TILES_DERIVATIVES = join(DATASET_DERIVATIVES, "cortical_tiles-2026")


def _default_log_path_and_candidates(script, argv, script_name, log_dir=DATASET_LOG_DIR):
    """Parse argv, call default_log_path(); return it and the paths allowed for a start time around the call."""
    script.parse_args(argv)
    before = datetime.now().astimezone()
    path = script.default_log_path()
    after = datetime.now().astimezone()
    candidates = {join(log_dir, f"{script_name}_{moment.strftime('%Y%m%d-%H%M%S')}.log") for moment in (before, after)}
    return path, candidates


# REQ-RUNLOG-BDRABCZUK-F0BE41134943
@pytest.mark.parametrize("output", [DATASET_DERIVATIVES, DATASET_DERIVATIVES + "/"])
def test_prune_bare_log_writes_under_dataset_derivatives(output):
    argv = [output, "--qc=/data/myproject/qc.tsv", "--log"]
    path, candidates = _default_log_path_and_candidates(PruneFailedSubjects(), argv, "prune_failed_subjects")
    assert path in candidates


# REQ-RUNLOG-BDRABCZUK-878DA1BE3917
@pytest.mark.parametrize("derivatives", [CORTICAL_TILES_DERIVATIVES, CORTICAL_TILES_DERIVATIVES + "/"])
def test_purge_bare_log_writes_under_dataset_derivatives(derivatives):
    argv = [derivatives, "--subject=sub-123456", "--log"]
    path, candidates = _default_log_path_and_candidates(PurgeSubject(), argv, "purge_subject")
    assert path in candidates


# REQ-RUNLOG-BDRABCZUK-F3B8400CBE97
@pytest.mark.parametrize("masks_option", [[], ["--masks=canonical_26"]])
def test_masks_bare_log_writes_under_output_dir(masks_option):
    output_dir = "/data/masks"
    argv = [
        "--labeled_subjects_dir=/data/manually_labeled/pclean/all",
        "--path_to_graph_supervised=t1mri/t1/default_analysis/folds/3.3/base2026_manual",
        f"--output_dir={output_dir}",
        *masks_option,
        "--log",
    ]
    path, candidates = _default_log_path_and_candidates(
        GenerateMasks(), argv, "generate_masks", log_dir=join(output_dir, ".log")
    )
    assert path in candidates


# REQ-RUNLOG-BDRABCZUK-B8789790A2D1
def test_labelling_qc_bare_log_writes_next_to_output_tsv():
    qc_dir = join(ROOT, "qc")
    argv = [
        join(DATASET_DERIVATIVES, "morphologist-6.0", "subjects"),
        join(qc_dir, "labelling_qc.tsv"),
        "--path_to_graph=t1mri/default_acquisition/default_analysis/folds/3.3",
        "--log",
    ]
    path, candidates = _default_log_path_and_candidates(
        GenerateLabellingQC(), argv, "generate_labelling_qc", log_dir=join(qc_dir, ".log")
    )
    assert path in candidates
