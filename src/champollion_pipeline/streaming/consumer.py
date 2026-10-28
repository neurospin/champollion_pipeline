"""Streaming consumer: embed scans through one resident EmbeddingEngine as their crops arrive.

One call per process (not thread-safe: RunLogCapture redirects fds 1/2 and the readiness
checker re-indexes the tiles tree at each poll). Scans already embedded are skipped; a bad
scan is FAILED, never raised; the loop returns on drain (producers done and nothing ready)
or on stall.
"""

from __future__ import annotations

import csv
import glob
import logging
import os
import time
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum

import champollion_utils.logger  # noqa: F401 - installs Logger.trace/output
import pandas as pd
from champollion_utils.logger import LOG, OUTPUT
from champollion_utils.run_log import RunLogCapture

from champollion_pipeline.config import DatasetConfig
from champollion_pipeline.derivatives_layout import compute_region_embeddings_dir
from champollion_pipeline.embedding_engine import EmbeddingEngine
from champollion_pipeline.file_indexer import ScanId, SubjectEligibilityChecker
from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from champollion_pipeline.process_setup import init_pipeline_process

from .barrier import poll_ready_scans, scan_embeddings_exist
from .settings import StreamingSettings
from .stages import SCAN_TILES_DIRNAME, compute_scan_tiles_dir, find_scan_crops_root
from .worker import TILES_COMPLETE_MARKER, tiles_marker_matches

logger = logging.getLogger(__name__)

CONSUMER_LOG_NAME = "consumer.log"
_EMBEDDINGS_FILENAME = "full_embeddings.csv"


class ConsumerStatus(str, Enum):
    """Outcome of one scan in the consumer."""

    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class ConsumerResult:
    """Outcome of one scan."""

    scan: ScanId
    status: ConsumerStatus


class _ScanFailure(Exception):
    """A scan cannot be embedded or written; the message says why."""


class _ReadinessChecker(SubjectEligibilityChecker):
    """Ready = completion marker of this masks_version plus every region's crop inputs."""

    def __init__(self, cortical_tiles_output: str, masks_version: str, regions: Sequence[str]) -> None:
        self._cortical_tiles_output = cortical_tiles_output
        self._masks_version = masks_version
        self._region_files: dict[str, tuple[str, str]] = {}
        for region in regions:
            side = "L" if region.endswith("_left") else "R"
            for name in (f"{side}skeleton.npy", f"{side}skeleton_subject.csv"):
                self._region_files[f"{region}/mask/{name}"] = (region, name)
        super().__init__(
            subjects_dir=os.path.join(cortical_tiles_output, SCAN_TILES_DIRNAME),
            required_patterns=[TILES_COMPLETE_MARKER, *self._region_files],
            stage_name="consumer",
        )

    def scan_has_pattern(self, scan_id: ScanId, pattern: str) -> bool:
        tiles = compute_scan_tiles_dir(self._cortical_tiles_output, scan_id)
        if pattern == TILES_COMPLETE_MARKER:
            return super().scan_has_pattern(scan_id, pattern) and tiles_marker_matches(tiles, self._masks_version)
        if pattern in self._region_files:
            region, name = self._region_files[pattern]
            crops = find_scan_crops_root(tiles, self._masks_version)
            if crops is None:
                return False
            crop = GenerateEmbeddings._find_crop_dir(None, crops, region)
            rel = os.path.relpath(os.path.join(crops, crop, "mask", name), tiles)
            return super().scan_has_pattern(scan_id, glob.escape(rel))
        return super().scan_has_pattern(scan_id, pattern)


def _check_inputs(scans: Iterable[ScanId], dataset: DatasetConfig) -> list[ScanId]:
    """The given scans as a list; TypeError / ValueError on unusable scans or dataset layout."""
    if isinstance(scans, str):
        raise TypeError(f"scans must be an iterable of ScanId, not a str: {scans!r}")
    items = list(scans)
    for scan in items:
        if not isinstance(scan, ScanId):
            raise TypeError(f"scans must hold only ScanId, got {type(scan).__name__} {scan!r}")
    for name in ("datasets_root", "cortical_tiles_output"):
        if not getattr(dataset, name):
            raise ValueError(f"DatasetConfig.{name} must not be empty")
    return items


def _skip(results: dict, scan: ScanId, reason: str) -> None:
    logger.log(OUTPUT, "consumer: scan %s skipped (%s)", scan.path_prefix, reason)
    results[scan] = ConsumerResult(scan, ConsumerStatus.SKIPPED)


def _write_rows(scan: ScanId, tables: dict[str, pd.DataFrame], dataset: DatasetConfig) -> None:
    """Append the scan's row to each region's full_embeddings.csv (regions already holding it are skipped)."""
    base = compute_region_embeddings_dir(dataset.datasets_root, dataset.masks_version)
    paths = {region: os.path.join(base, region, _EMBEDDINGS_FILENAME) for region in tables}
    for region, path in paths.items():
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            with open(path, newline="") as handle:
                header = next(csv.reader(handle), [])
            if header != list(tables[region].columns):
                raise _ScanFailure(f"{path} has columns {header}, not {list(tables[region].columns)}")
    for region, table in tables.items():
        if scan_embeddings_exist(scan, dataset.datasets_root, dataset.masks_version, [region]):
            continue
        path = paths[region]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new_file = not os.path.isfile(path) or os.path.getsize(path) == 0
        table.to_csv(path, mode="a", header=new_file, index=False)


def _embed_tables(engine: EmbeddingEngine, regions: Sequence[str], scan: ScanId, crops: str) -> dict:
    """One single-row embedding table per region, ID set to str(scan)."""
    tables = {}
    for region in regions:
        table = engine.embed_crops(region, crops)
        if len(table) != 1:
            raise _ScanFailure(f"region {region} gave {len(table)} embedding rows, expected 1")
        table["ID"] = str(scan)
        dims = table.drop(columns=["ID"])
        if dims.isna().to_numpy().any():
            logger.warning("consumer: scan %s region %s: embedding holds NaN", scan.path_prefix, region)
        tables[region] = table
    return tables


def _embed_scan(
    engine: EmbeddingEngine, regions: Sequence[str], scan: ScanId, dataset: DatasetConfig
) -> ConsumerResult:
    """Embed every region of one ready scan and write its rows; never raises."""
    logger.log(LOG, "consumer: embedding scan %s", scan.path_prefix)
    tiles = compute_scan_tiles_dir(dataset.cortical_tiles_output, scan)
    try:
        crops = find_scan_crops_root(tiles, dataset.masks_version)
        if crops is None:
            raise _ScanFailure(f"no crops for masks version {dataset.masks_version} under {tiles}")
        tables = _embed_tables(engine, regions, scan, crops)
        _write_rows(scan, tables, dataset)
    except (Exception, SystemExit) as error:
        logger.error("consumer: scan %s failed: %s: %s", scan.path_prefix, type(error).__name__, error, exc_info=error)
        return ConsumerResult(scan, ConsumerStatus.FAILED)
    logger.log(OUTPUT, "consumer: scan %s embedded (%d regions)", scan.path_prefix, len(tables))
    return ConsumerResult(scan, ConsumerStatus.OK)


def _poll_loop(context: dict, pending: list[ScanId], producers_done, results: dict) -> list[ScanId]:
    """Embed scans as they become ready; return the scans still pending at drain or stall."""
    settings, engine, regions, dataset, checker = (
        context["settings"],
        context["engine"],
        context["regions"],
        context["dataset"],
        context["checker"],
    )
    last_progress = None
    while pending:
        draining = producers_done.is_set()
        now = time.monotonic()
        ready = poll_ready_scans(checker, pending)
        for scan in ready:
            results[scan] = _embed_scan(engine, regions, scan, dataset)
        pending = [scan for scan in pending if scan not in ready]
        if ready:
            last_progress = now
        elif draining:
            break
        elif last_progress is None:
            last_progress = now
        elif now - last_progress >= settings.consumer_stall_timeout:
            logger.warning(
                "consumer: stalled %s s with %d scans pending", settings.consumer_stall_timeout, len(pending)
            )
            break
        if not pending:
            break
        time.sleep(settings.poll_interval)
    return pending


def _consume(context: dict, ordered: list[ScanId], producers_done, results: dict) -> None:
    """Resume, then poll and embed; every scan of ``ordered`` ends with a result."""
    dataset, regions = context["dataset"], context["regions"]
    pending = []
    for scan in ordered:
        if scan_embeddings_exist(scan, dataset.datasets_root, dataset.masks_version, regions):
            _skip(results, scan, "embeddings already present")
        else:
            pending.append(scan)
    context["checker"] = _ReadinessChecker(dataset.cortical_tiles_output, dataset.masks_version, regions)
    for scan in _poll_loop(context, pending, producers_done, results):
        logger.error(
            "consumer: scan %s failed: never ready (region inputs or %s completion marker missing)",
            scan.path_prefix,
            dataset.masks_version,
        )
        results[scan] = ConsumerResult(scan, ConsumerStatus.FAILED)


def run_consumer(
    settings: StreamingSettings,
    dataset: DatasetConfig,
    scans: Iterable[ScanId],
    producers_done,
    models_path: str,
) -> list[ConsumerResult]:
    """Embed ``scans`` as their crops arrive; one result per distinct scan, in sorted order.

    ``producers_done`` is any object with ``is_set()``; once set, the first poll that finds no
    ready scan ends the run. ``models_path`` is already fetched.
    """
    init_pipeline_process()
    items = _check_inputs(scans, dataset)
    ordered = sorted(set(items))
    results: dict[ScanId, ConsumerResult] = {}
    if settings.dry_run:
        for scan in ordered:
            _skip(results, scan, "dry run")
        return [results[scan] for scan in ordered]

    log_path = os.path.join(settings.log_dir, CONSUMER_LOG_NAME)
    with ExitStack() as stack:
        try:
            capture = stack.enter_context(RunLogCapture(log_path))
        except OSError as error:
            for scan in ordered:
                logger.error("consumer: scan %s failed: run log %s unusable: %s", scan.path_prefix, log_path, error)
                results[scan] = ConsumerResult(scan, ConsumerStatus.FAILED)
            return [results[scan] for scan in ordered]
        engine = EmbeddingEngine(models_path, settings.device, epoch=dataset.checkpoint_epoch)
        regions = engine.regions
        if not regions:
            raise ValueError(f"No region model under models path {models_path}")
        context = {"settings": settings, "engine": engine, "regions": regions, "dataset": dataset}
        try:
            _consume(context, ordered, producers_done, results)
        finally:
            capture.exit_code = 1 if any(r.status is ConsumerStatus.FAILED for r in results.values()) else 0
    return [results[scan] for scan in ordered]
