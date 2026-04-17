#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI wrapper for the scan-centric parallel pipeline (Strategy 2).

Launches ``run_parallel_pipeline`` from ``parallel_runner`` with all parameters
supplied as command-line arguments — no YAML config file required.  This script
is the subprocess entry point used by the ``champollion_sulcal_mcp`` server's
``start_streaming`` tool.

Usage::

    python run_streaming.py <subjects_dir> <output_dir> \\
        --dataset MY_COHORT \\
        --path-to-graph "t1mri/default_acquisition/default_analysis/folds/3.1" \\
        --path-sk-with-hull "t1mri/default_acquisition/default_analysis/segmentation" \\
        --n-workers 8 --bids

See PARALLEL_PIPELINE.md for the full design.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the path so relative imports work
sys.path.insert(0, str(Path(__file__).parent))


def _build_config(args: argparse.Namespace):
    """Build a lightweight config namespace from parsed CLI arguments."""

    class _Config:
        pass

    cfg = _Config()
    cfg.name = args.dataset
    cfg.path_to_graph = args.path_to_graph
    cfg.path_sk_with_hull = args.path_sk_with_hull
    cfg.sk_qc_path = args.sk_qc_path or ""
    cfg.bids = args.bids
    cfg.regions = []
    cfg.embeddings_only = True  # streaming always runs inference only
    cfg.models_path = args.models_path or ""
    cfg.dataset_localization = args.dataset_localization
    cfg.datasets_root = args.datasets_root or ""
    cfg.short_name = args.short_name
    cfg.embeddings_path = args.embeddings_path or "champollion_V1"
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan-centric parallel Champollion pipeline (streaming mode).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional
    parser.add_argument(
        "subjects_dir",
        help="Root directory containing subject folders (Morphologist output).",
    )
    parser.add_argument(
        "output_dir",
        help="Root output directory for all pipeline stages.",
    )

    # Dataset identity
    parser.add_argument("--dataset", required=True, help="Short dataset name (e.g. MY_COHORT).")
    parser.add_argument(
        "--path-to-graph",
        required=True,
        dest="path_to_graph",
        help="Relative path to .arg graph inside each subject folder.",
    )
    parser.add_argument(
        "--path-sk-with-hull",
        required=True,
        dest="path_sk_with_hull",
        help="Relative path to skeleton directory inside each subject folder.",
    )
    parser.add_argument("--sk-qc-path", dest="sk_qc_path", default=None,
                        help="TSV QC file with participant_id and qc columns.")

    # BIDS
    parser.add_argument("--bids", action="store_true",
                        help="Input follows BIDS layout (sub-*/ses-*/...).")

    # Parallelism / timeouts
    parser.add_argument(
        "--n-workers", type=int, default=0, dest="n_workers",
        help="Number of parallel scan workers. 0 = os.cpu_count() (default: 0).",
    )
    parser.add_argument(
        "--worker-timeout", type=int, default=7200, dest="worker_timeout",
        help="Per-worker timeout in seconds waiting for prerequisite files (default: 7200).",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=10, dest="poll_interval",
        help="Seconds between filesystem polls in each worker (default: 10).",
    )

    # Embeddings params
    parser.add_argument("--models-path", dest="models_path", default="",
                        help="Path to model weights (local dir, archive, or HF repo ID).")
    parser.add_argument("--dataset-localization", dest="dataset_localization", default="local")
    parser.add_argument("--datasets-root", dest="datasets_root", default="")
    parser.add_argument("--short-name", dest="short_name", default="eval",
                        help="Run tag used in output folder names.")
    parser.add_argument("--embeddings-path", dest="embeddings_path", default="champollion_V1",
                        help="Relative path for combined embeddings output.")

    # Dry-run / resume
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Enumerate scans and report without processing.",
    )

    args = parser.parse_args()

    from parallel_runner import run_parallel_pipeline

    config = _build_config(args)
    log_dir = str(Path(args.output_dir) / "logs")

    return run_parallel_pipeline(
        subjects_dir=args.subjects_dir,
        output_dir=args.output_dir,
        config=config,
        n_workers=args.n_workers,
        log_dir=log_dir,
        poll_interval=args.poll_interval,
        worker_timeout=args.worker_timeout,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
