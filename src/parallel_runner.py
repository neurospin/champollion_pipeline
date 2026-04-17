#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan-centric parallel runner for the Champollion pipeline (Strategy 2).

Each worker owns one ScanId and runs stages 2-4 (cortical_tiles ->
champollion_config -> embeddings) sequentially, using file-presence barriers
between stages.  Stage 5 (put_together_embeddings) runs once after all workers
drain.

Usage via main.py::

    python main.py --mode streaming --n-workers 8 --bids

See PARALLEL_PIPELINE.md for the full design.
"""

import glob
import logging
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from main import DatasetConfig

# Module-level import so tests can patch parallel_runner.SubjectEligibilityChecker.
# This runs in the main (orchestrator) process, not in worker processes.
sys.path.insert(0, str(Path(__file__).parent))
from file_indexer.pipeline_checks import SubjectEligibilityChecker  # noqa: E402
from file_indexer.scan_id import ScanId  # noqa: E402


# ---------------------------------------------------------------------------
# Per-worker logger
# ---------------------------------------------------------------------------

def _worker_logger(scan_id: "ScanId", log_dir: str) -> logging.Logger:
    """Create a per-worker file logger. Never configure the same logger twice."""
    name = f"parallel_runner.{scan_id}"
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    worker_log_dir = os.path.join(log_dir, str(scan_id))
    os.makedirs(worker_log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(worker_log_dir, "worker.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# File-presence barrier
# ---------------------------------------------------------------------------

def _wait_for_scan_files(
    base_dir: str,
    scan_id: "ScanId",
    patterns: List[str],
    logger: logging.Logger,
    poll_interval: int,
    timeout: int,
) -> None:
    """Poll until all patterns resolve for this scan, or raise TimeoutError.

    Rebuilds the B-tree index on every poll so there is no shared mutable state.
    """
    # Workers run in a fresh process — re-insert src/ and import locally.
    _src = str(Path(__file__).parent)
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from file_indexer.pipeline_checks import SubjectEligibilityChecker as _Checker

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        checker = _Checker(
            base_dir,
            patterns,
            bids=scan_id.session is not None,
        )
        checker._tree = checker._build_index()
        all_present = all(checker._scan_has_pattern(scan_id, p) for p in patterns)
        if all_present:
            return
        logger.debug("waiting for %s — patterns: %s", scan_id, patterns)
        time.sleep(poll_interval)
    raise TimeoutError(f"{scan_id}: timed out after {timeout}s waiting for {patterns}")


# ---------------------------------------------------------------------------
# Single-subject temp directory
# ---------------------------------------------------------------------------

def _make_single_subject_dir(scan_id: "ScanId", subjects_dir: str) -> tempfile.TemporaryDirectory:
    """Create a TemporaryDirectory with a symlink scoping to one subject.

    Usage::

        with _make_single_subject_dir(scan_id, subjects_dir) as tmp_path:
            # tmp_path/{scan_id.subject} -> subjects_dir/{scan_id.subject}
            ...
    """
    tmp = tempfile.TemporaryDirectory()
    src = os.path.join(subjects_dir, scan_id.subject)
    dst = os.path.join(tmp.name, scan_id.subject)
    os.symlink(src, dst)
    return tmp


# ---------------------------------------------------------------------------
# Find crops root
# ---------------------------------------------------------------------------

def _find_crops_root(output_dir: str) -> Optional[str]:
    """Return the latest cortical_tiles crops/2mm directory, or None."""
    matches = sorted(glob.glob(join(output_dir, "cortical_tiles-*", "crops", "2mm")))
    return matches[-1] if matches else None


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def _run_cortical_tiles(scan_id: "ScanId", subjects_dir: str, output_dir: str, config: "DatasetConfig", njobs_per_worker: int) -> None:
    """Run cortical_tiles for a single scan."""
    sys.path.insert(0, str(Path(__file__).parent))
    from run_cortical_tiles import RunCorticalTiles

    with _make_single_subject_dir(scan_id, subjects_dir) as tmp_subjects:
        args = [
            tmp_subjects,
            output_dir,
            f"--path_to_graph={config.path_to_graph}",
            f"--path_sk_with_hull={config.path_sk_with_hull}",
        ]
        if njobs_per_worker > 0:
            args.append(f"--njobs={njobs_per_worker}")
        if getattr(config, "sk_qc_path", ""):
            args.append(f"--sk_qc_path={config.sk_qc_path}")
        if getattr(config, "bids", False):
            args.append("--bids")
        if getattr(config, "regions", None):
            args.extend(["--regions"] + config.regions)
        script = RunCorticalTiles()
        script.parse_args(args)
        rc = script.run()
        if rc != 0:
            raise RuntimeError(f"cortical_tiles failed with code {rc} for {scan_id}")


def _run_generate_config(scan_id: "ScanId", output_dir: str, config: "DatasetConfig") -> None:
    """Run generate_champollion_config for a single scan."""
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_champollion_config import GenerateChampollionConfig

    crop_path = _find_crops_root(output_dir)
    if crop_path is None:
        raise RuntimeError(f"No cortical_tiles crops found in {output_dir} for {scan_id}")

    args = [crop_path, f"--dataset={config.name}"]
    script = GenerateChampollionConfig()
    script.parse_args(args)
    rc = script.run()
    if rc != 0:
        raise RuntimeError(f"generate_champollion_config failed with code {rc} for {scan_id}")


def _run_generate_embeddings(scan_id: "ScanId", output_dir: str, config: "DatasetConfig") -> None:
    """Run generate_embeddings (--embeddings_only) for a single scan."""
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_embeddings import GenerateEmbeddings

    models_path = getattr(config, "models_path", "") or output_dir
    dataset_localization = getattr(config, "dataset_localization", "local")
    datasets_root = getattr(config, "datasets_root", "")
    short_name = getattr(config, "short_name", "eval")

    args = [
        models_path,
        dataset_localization,
        datasets_root,
        short_name,
        "--embeddings_only",
    ]

    script = GenerateEmbeddings()
    script.parse_args(args)
    rc = script.run()
    if rc != 0:
        raise RuntimeError(f"generate_embeddings failed with code {rc} for {scan_id}")


def _run_put_together_embeddings(output_dir: str, config: "DatasetConfig") -> int:
    """Run put_together_embeddings once after all workers drain."""
    sys.path.insert(0, str(Path(__file__).parent))
    from put_together_embeddings import PutTogetherEmbeddings

    combined_dir = os.path.join(output_dir, "combined_embeddings")
    os.makedirs(combined_dir, exist_ok=True)

    embeddings_subpath = getattr(config, "embeddings_path", "") or "champollion_V1"

    args = [
        f"--embeddings_subpath={embeddings_subpath}",
        f"--output_path={combined_dir}",
    ]

    script = PutTogetherEmbeddings()
    script.parse_args(args)
    return script.run()


# ---------------------------------------------------------------------------
# Embeddings existence check (resume support)
# ---------------------------------------------------------------------------

def _scan_embeddings_exist(scan_id: "ScanId", output_dir: str) -> bool:
    """Return True if full_embeddings.csv already exists for this scan."""
    pattern = join(output_dir, "champollion_V1", "**", str(scan_id), "**", "full_embeddings.csv")
    return len(glob.glob(pattern, recursive=True)) > 0


# ---------------------------------------------------------------------------
# Per-scan worker
# ---------------------------------------------------------------------------

def run_scan_worker(
    scan_id: "ScanId",
    subjects_dir: str,
    output_dir: str,
    config: "DatasetConfig",
    log_dir: str,
    njobs_per_worker: int = 1,
    poll_interval: int = 10,
    timeout: int = 7200,
    dry_run: bool = False,
) -> None:
    """Full stages 2-4 pipeline for one scan.  Never raises — logs and returns."""
    # Re-insert src/ into sys.path (new process needs it)
    src_dir = str(Path(__file__).parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    logger = _worker_logger(scan_id, log_dir)

    if dry_run:
        logger.info("dry-run: skipping %s", scan_id)
        return

    if _scan_embeddings_exist(scan_id, output_dir):
        logger.info("skipping %s — embeddings already exist", scan_id)
        return

    try:
        # Stage 2: wait for Morphologist .arg graphs, then run cortical_tiles
        _wait_for_scan_files(
            subjects_dir,
            scan_id,
            patterns=[f"{config.path_to_graph}/R*.arg"],
            logger=logger,
            poll_interval=poll_interval,
            timeout=timeout,
        )
        _run_cortical_tiles(scan_id, subjects_dir, output_dir, config, njobs_per_worker)

        # Stage 3: wait for crops, then generate config
        _wait_for_scan_files(
            output_dir,
            scan_id,
            patterns=["cortical_tiles-*/crops/2mm/*"],
            logger=logger,
            poll_interval=poll_interval,
            timeout=timeout,
        )
        _run_generate_config(scan_id, output_dir, config)

        # Stage 4: wait for config yaml, then run inference
        _wait_for_scan_files(
            output_dir,
            scan_id,
            patterns=["champollion_V1/configs/**/reference.yaml"],
            logger=logger,
            poll_interval=poll_interval,
            timeout=timeout,
        )
        _run_generate_embeddings(scan_id, output_dir, config)

        logger.info("complete")
    except TimeoutError as exc:
        logger.error("timed out waiting for prerequisite files: %s", exc)
    except Exception as exc:
        logger.error("failed: %s", exc, exc_info=True)
    # Does NOT re-raise — pool continues with remaining scans


# ---------------------------------------------------------------------------
# Pool orchestrator
# ---------------------------------------------------------------------------

def run_parallel_pipeline(
    subjects_dir: str,
    output_dir: str,
    config: "DatasetConfig",
    n_workers: int,
    log_dir: str,
    poll_interval: int = 10,
    worker_timeout: int = 7200,
    dry_run: bool = False,
) -> int:
    """Spawn N workers (one per scan), then combine embeddings.

    Parameters
    ----------
    subjects_dir:
        Root directory containing subject folders.
    output_dir:
        Root output directory for all pipeline outputs.
    config:
        Dataset configuration (DatasetConfig instance from main.py).
    n_workers:
        Number of parallel worker processes. 0 = os.cpu_count().
    log_dir:
        Directory for per-worker log files.
    poll_interval:
        Seconds between filesystem polls in each worker.
    worker_timeout:
        Per-worker timeout (seconds) waiting for prerequisite files.
    dry_run:
        If True, enumerate scans and report without processing.
    """
    pool_logger = logging.getLogger("parallel_runner")
    if not pool_logger.handlers:
        pool_logger.setLevel(logging.INFO)
        pool_logger.addHandler(logging.StreamHandler())

    if n_workers <= 0:
        n_workers = os.cpu_count() or 1

    njobs_per_worker = max(1, (os.cpu_count() or 1) // n_workers)

    checker = SubjectEligibilityChecker(
        subjects_dir,
        [],
        bids=getattr(config, "bids", False),
        path_to_graph=getattr(config, "path_to_graph", ""),
    )
    checker._tree = checker._build_index()
    scan_ids = checker._get_scan_ids()

    if not scan_ids:
        pool_logger.warning("No scans found in %s — nothing to do.", subjects_dir)
        return 0

    if dry_run:
        print(f"dry-run: {len(scan_ids)} scan(s) found:")
        for sid in scan_ids:
            print(f"  {sid}")
        return 0

    os.makedirs(log_dir, exist_ok=True)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                run_scan_worker,
                sid,
                subjects_dir,
                output_dir,
                config,
                log_dir,
                njobs_per_worker,
                poll_interval,
                worker_timeout,
                dry_run,
            ): sid
            for sid in scan_ids
        }
        for future in as_completed(futures):
            sid = futures[future]
            try:
                future.result()
            except Exception as exc:
                pool_logger.error("%s pool error: %s", sid, exc)

    # Stage 5: combine embeddings once after all workers drain
    return _run_put_together_embeddings(output_dir, config)
