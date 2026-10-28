"""Producer worker of a streaming run: Morphologist then cortical_tiles for ONE scan, never raising.

run_scan_worker runs in a ProcessPool child. It skips a dry run or a scan whose completion marker
matches the masks version, captures the stages' output into <log_dir>/<path_prefix>/worker.log, writes
the completion marker <tiles_dir>/tiles_complete.json once every stage succeeded, and turns any stage
failure into a FAILED WorkerResult. Not thread-safe: RunLogCapture redirects the process-wide fds 1 and 2.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum

import champollion_utils.logger  # noqa: F401 - installs Logger.trace/output
from champollion_utils.run_log import RunLogCapture

from champollion_pipeline.config import DatasetConfig
from champollion_pipeline.file_indexer import ScanId
from champollion_pipeline.models.cache import write_json_atomic
from champollion_pipeline.process_setup import init_pipeline_process

from .settings import StreamingSettings
from .stages import (
    compute_scan_tiles_dir,
    run_cortical_tiles_for_scan,
    run_morphologist_for_scan,
)

logger = logging.getLogger(__name__)

WORKER_LOG_NAME = "worker.log"
TILES_COMPLETE_MARKER = "tiles_complete.json"


class WorkerStatus(str, Enum):
    """Outcome of one scan's producer work."""

    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class WorkerResult:
    """What a worker returns for one scan (picklable)."""

    scan: ScanId
    status: WorkerStatus


def tiles_marker_matches(tiles_dir: str, masks_version: str) -> bool:
    """Return True when tiles_dir holds a completion marker written for masks_version; never raise."""
    try:
        with open(os.path.join(tiles_dir, TILES_COMPLETE_MARKER), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("masks_version") == masks_version


def run_scan_worker(
    scan: ScanId, settings: StreamingSettings, dataset: DatasetConfig, raw_scan: str | None = None
) -> WorkerResult:
    """Run Morphologist (when raw_scan is given) then cortical_tiles for one scan; never raise for a stage failure."""
    init_pipeline_process()
    p = scan.path_prefix
    logger.trace(f"worker: scan {p}: start")
    if settings.dry_run:
        logger.output(f"worker: scan {p} skipped (dry run)")
        return WorkerResult(scan, WorkerStatus.SKIPPED)
    tiles_dir = compute_scan_tiles_dir(dataset.cortical_tiles_output, scan)
    marker = os.path.join(tiles_dir, TILES_COMPLETE_MARKER)
    if tiles_marker_matches(tiles_dir, dataset.masks_version):
        logger.output(f"worker: scan {p} skipped, completion marker {marker} matches masks {dataset.masks_version}")
        return WorkerResult(scan, WorkerStatus.SKIPPED)
    try:
        os.remove(marker)
    except FileNotFoundError:
        pass
    except OSError as error:
        logger.error(f"worker: scan {p} failed: stale completion marker {marker} not removable: {error}")
        return WorkerResult(scan, WorkerStatus.FAILED)
    log_path = os.path.join(settings.log_dir, p, WORKER_LOG_NAME)
    with contextlib.ExitStack() as stack:
        try:
            capture = stack.enter_context(RunLogCapture(log_path))
        except OSError as error:
            logger.error(f"worker: scan {p} failed: worker log {log_path} unusable: {error}")
            return WorkerResult(scan, WorkerStatus.FAILED)
        ok = _run_stages(scan, settings, dataset, raw_scan, p) and _write_marker(marker, dataset.masks_version, p)
        capture.exit_code = 0 if ok else 1
    if not ok:
        return WorkerResult(scan, WorkerStatus.FAILED)
    logger.output(f"worker: scan {p} done, tiles in {tiles_dir}")
    return WorkerResult(scan, WorkerStatus.OK)


def _write_marker(marker: str, masks_version: str, p: str) -> bool:
    """Write the completion marker atomically; log and return False when it cannot be written."""
    try:
        write_json_atomic(marker, {"masks_version": masks_version})
    except OSError as error:
        logger.error(f"worker: scan {p} failed: completion marker {marker} not written: {error}")
        return False
    return True


def _run_stages(
    scan: ScanId, settings: StreamingSettings, dataset: DatasetConfig, raw_scan: str | None, p: str
) -> bool:
    """Run the scan's stages in order; log and return False at the first failure."""
    stages = []
    if raw_scan is not None:
        stages.append(("Morphologist", lambda: run_morphologist_for_scan(raw_scan, settings, dataset)))
    stages.append(("cortical_tiles", lambda: run_cortical_tiles_for_scan(scan, settings, dataset)))
    for stage, call in stages:
        try:
            call()
        except (Exception, SystemExit) as error:
            logger.error(f"worker: scan {p} failed in {stage}: {type(error).__name__}: {error}", exc_info=error)
            return False
    return True
