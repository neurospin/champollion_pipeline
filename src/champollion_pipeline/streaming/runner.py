"""Orchestrator of a streaming run: producer workers, one resident inference consumer, then combine.

Scans of ``settings.subjects_dir`` (graphs layout) run through ``run_scan_worker`` in a spawn process pool. The
dataset config is generated once, from the first OK or SKIPPED scan's crops root (dry run: up front), then one
``run_consumer`` call embeds scans while the workers still run. When the workers drain, ``producers_done`` is set,
the consumer returns, both summaries are logged and the run is combined. Any FAILED scan makes the exit code 1.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import os
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, BrokenExecutor, Future, ProcessPoolExecutor, wait

import champollion_utils.logger  # noqa: F401 - installs Logger.trace/output

from champollion_pipeline.config import DatasetConfig
from champollion_pipeline.derivatives_layout import compute_models_cache_dir
from champollion_pipeline.file_indexer import ScanId, SubjectEligibilityChecker
from champollion_pipeline.models.cache import ensure_models_cache_dir
from champollion_pipeline.models.fetch import fetch_model_set

from .combine import run_combine
from .consumer import CONSUMER_LOG_NAME, ConsumerResult, ConsumerStatus, run_consumer
from .dataset_config import ensure_dataset_config
from .settings import StreamingSettings
from .stages import compute_scan_tiles_dir, find_scan_crops_root
from .worker import WORKER_LOG_NAME, WorkerResult, WorkerStatus, run_scan_worker

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _ProducersDone:
    """Flag file: set once every producer has finished; picklable and readable from any process."""

    path: str

    def is_set(self) -> bool:
        return os.path.exists(self.path)

    def set(self) -> None:
        with open(self.path, "w"):
            pass


@dataclasses.dataclass
class _ProducerState:
    """Bookkeeping of the producer pool for one run."""

    settings: StreamingSettings
    dataset: DatasetConfig
    scans: list[ScanId]
    producers_done: _ProducersDone
    pending: list[ScanId] = dataclasses.field(default_factory=list)
    running: dict[Future, tuple[ScanId, float]] = dataclasses.field(default_factory=dict)
    abandoned: set[Future] = dataclasses.field(default_factory=set)
    results: dict[ScanId, WorkerResult] = dataclasses.field(default_factory=dict)
    crops_root: str | None = None
    failed: bool = False

    def __post_init__(self) -> None:
        self.pending = sorted(self.scans)


def _worker_log(settings: StreamingSettings, scan: ScanId) -> str:
    return os.path.join(settings.log_dir, scan.path_prefix, WORKER_LOG_NAME)


def _list_scans(settings: StreamingSettings, dataset: DatasetConfig) -> list[ScanId] | None:
    """The scans to run, or None when the inputs are refused."""
    if dataset.embeddings_path:
        logger.error(
            "embeddings_path must be empty in streaming mode (the run writes the versioned default): %s",
            dataset.embeddings_path,
        )
        return None
    if not os.path.isdir(settings.subjects_dir):
        logger.error("subjects directory not found: %s", settings.subjects_dir)
        return None
    scans = SubjectEligibilityChecker(settings.subjects_dir, [], bids=dataset.bids).list_scans()
    if not scans:
        logger.warning("no scan under %s", settings.subjects_dir)
    return scans


def _fetch_models(settings: StreamingSettings, dataset: DatasetConfig, models_path: str) -> str | None:
    """Local path of the model set (dry run: the source unchanged), or None when the fetch fails."""
    if settings.dry_run:
        logger.trace("dry run: models not fetched (%s)", models_path)
        return models_path
    try:
        cache = (
            os.path.abspath(dataset.models_cache)
            if dataset.models_cache
            else compute_models_cache_dir(dataset.datasets_root)
        )
        extract_to = ensure_models_cache_dir(cache)
        fetched, _ = fetch_model_set(models_path, extract_to, None, False)
    except Exception as error:
        logger.error("model fetch failed for %s: %s", models_path, error)
        return None
    logger.trace("models fetched to %s", fetched)
    return fetched


def _log_dry_run(scans: list[ScanId]) -> None:
    for scan in sorted(scans):
        logger.output("dry run: scan %s", scan.path_prefix)


def _ensure_config(settings: StreamingSettings, dataset: DatasetConfig) -> bool:
    try:
        root = ensure_dataset_config(settings, dataset)
    except Exception as error:
        logger.error("dataset config failed: %s", error)
        return False
    logger.trace("dataset config ready under %s", root)
    return True


def _capacity(state: _ProducerState) -> int:
    """Producer slots not held by a timed-out call that still runs."""
    return state.settings.producer_workers - len([f for f in state.abandoned if not f.done()])


def _fail_pending(state: _ProducerState, reason: str) -> None:
    for scan in state.pending:
        logger.error(
            "scan %s: not run: %s; no worker log written at %s",
            scan.path_prefix,
            reason,
            _worker_log(state.settings, scan),
        )
        state.results[scan] = WorkerResult(scan, WorkerStatus.FAILED)
        state.failed = True
    state.pending = []


def _submit_ready(state: _ProducerState, pool: ProcessPoolExecutor) -> None:
    """Submit pending scans while a slot is free; in-flight calls never exceed the free slots."""
    while state.pending and len(state.running) < _capacity(state):
        scan = state.pending.pop(0)
        try:
            future = pool.submit(run_scan_worker, scan, state.settings, state.dataset)
        except BrokenExecutor as error:
            state.pending.insert(0, scan)
            _fail_pending(state, str(error))
            return
        state.running[future] = (scan, time.monotonic())
        logger.trace("scan %s: worker submitted", scan.path_prefix)
    if state.pending and _capacity(state) <= 0:
        _fail_pending(state, "every producer slot is held by a timed-out worker")


def _record_done(state: _ProducerState, future: Future) -> None:
    scan, _ = state.running.pop(future)
    log = _worker_log(state.settings, scan)
    try:
        result = future.result()
    except Exception as error:
        logger.error("scan %s: run_scan_worker raised: %s (log %s)", scan.path_prefix, error, log)
        result = WorkerResult(scan, WorkerStatus.FAILED)
    else:
        if result.status is WorkerStatus.FAILED:
            logger.error("scan %s: worker FAILED (log %s)", scan.path_prefix, log)
    state.results[scan] = result
    if result.status is WorkerStatus.FAILED:
        state.failed = True
    elif state.crops_root is None:
        tiles = compute_scan_tiles_dir(state.dataset.cortical_tiles_output, scan)
        state.crops_root = find_scan_crops_root(tiles, state.dataset.masks_version)


def _expire(state: _ProducerState) -> None:
    """Mark FAILED every call running longer than worker_timeout; its slot stays held while it runs."""
    now = time.monotonic()
    for future, (scan, started) in list(state.running.items()):
        if now - started < state.settings.worker_timeout:
            continue
        del state.running[future]
        state.abandoned.add(future)
        state.results[scan] = WorkerResult(scan, WorkerStatus.FAILED)
        state.failed = True
        logger.error(
            "scan %s: worker exceeded worker_timeout=%s s, marked FAILED (log %s)",
            scan.path_prefix,
            state.settings.worker_timeout,
            _worker_log(state.settings, scan),
        )


def _producers_active(state: _ProducerState) -> bool:
    return bool(state.pending or state.running)


def _step(state: _ProducerState, pool: ProcessPoolExecutor) -> None:
    """Wait one poll interval for a call to finish, then record, expire and refill."""
    watched = set(state.running) | {f for f in state.abandoned if not f.done()}
    done, _ = wait(watched, timeout=state.settings.poll_interval, return_when=FIRST_COMPLETED)
    finished = sorted((f for f in done if f in state.running), key=lambda f: state.running[f][0])
    for future in finished:
        _record_done(state, future)
    _expire(state)
    _submit_ready(state, pool)


def _status_counts(label: str, statuses, results) -> str:
    values = [result.status for result in results]
    return f"{label}: " + ", ".join(f"{status.value}={values.count(status)}" for status in statuses)


def _log_worker_summary(state: _ProducerState) -> None:
    logger.output(_status_counts("workers", WorkerStatus, state.results.values()))


def _await_config(state: _ProducerState, pool: ProcessPoolExecutor) -> bool:
    """Generate the dataset config from the first OK or SKIPPED scan's crops root (dry run: done already).

    A dry run waits for its first OK or SKIPPED scan instead; with none, it ends like a real run without a
    crops root: no consumer, no combine.
    """
    if state.settings.dry_run:
        return _await_dry_run_scan(state, pool)
    while state.crops_root is None and _producers_active(state):
        _step(state, pool)
    if state.crops_root is None:
        _log_worker_summary(state)
        logger.error(
            "no OK or SKIPPED scan left a crops root under %s; dataset config not generated",
            state.dataset.cortical_tiles_output,
        )
        return False
    return _ensure_config(state.settings, dataclasses.replace(state.dataset, crops_path=state.crops_root))


def _await_dry_run_scan(state: _ProducerState, pool: ProcessPoolExecutor) -> bool:
    def any_usable() -> bool:
        return any(result.status is not WorkerStatus.FAILED for result in state.results.values())

    while not any_usable() and _producers_active(state):
        _step(state, pool)
    if any_usable():
        return True
    _log_worker_summary(state)
    logger.error(
        "dry run: no OK or SKIPPED scan under %s; consumer and combine not run", state.dataset.cortical_tiles_output
    )
    return False


def _consumer_results(future: Future, settings: StreamingSettings) -> list[ConsumerResult] | None:
    try:
        return future.result()
    except Exception as error:
        logger.error("consumer failed: %s (log %s)", error, os.path.join(settings.log_dir, CONSUMER_LOG_NAME))
        return None


def _log_consumer_results(settings: StreamingSettings, results: list[ConsumerResult]) -> bool:
    """Log each FAILED scan and the consumer summary; return whether any scan FAILED."""
    log = os.path.join(settings.log_dir, CONSUMER_LOG_NAME)
    failed = [result for result in results if result.status is ConsumerStatus.FAILED]
    for result in failed:
        logger.error("scan %s: consumer FAILED (log %s)", result.scan.path_prefix, log)
    logger.output(_status_counts("consumer", ConsumerStatus, results))
    return bool(failed)


def _combine(settings: StreamingSettings, dataset: DatasetConfig) -> bool:
    try:
        logger.output("combined embeddings: %s", run_combine(settings, dataset))
    except Exception as error:
        logger.error("combine failed: %s", error)
        return False
    return True


def _close_worker_pool(state: _ProducerState, pool: ProcessPoolExecutor) -> None:
    """Shut the producer pool down; terminate the children of timed-out calls still running."""
    hung = [f for f in state.abandoned if not f.done()]
    pool.shutdown(wait=not hung, cancel_futures=True)
    if hung:
        logger.error("%d timed-out worker calls still running; terminating the producer processes", len(hung))
        for child in multiprocessing.active_children():
            child.terminate()
            child.join(timeout=5)


def _consume_and_combine(state: _ProducerState, pool: ProcessPoolExecutor, models_path: str, spawn) -> int:
    settings, dataset = state.settings, state.dataset
    with ProcessPoolExecutor(max_workers=1, mp_context=spawn) as consumer_pool:
        future = consumer_pool.submit(run_consumer, settings, dataset, state.scans, state.producers_done, models_path)
        try:
            while _producers_active(state):
                _step(state, pool)
        finally:
            state.producers_done.set()
        consumed = _consumer_results(future, settings)
    _log_worker_summary(state)
    if consumed is None:
        return 1
    if _log_consumer_results(settings, consumed):
        state.failed = True
    if not _combine(settings, dataset):
        return 1
    return 1 if state.failed else 0


def _run_streaming(state: _ProducerState, models_path: str) -> int:
    spawn = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=state.settings.producer_workers, mp_context=spawn)
    try:
        _submit_ready(state, pool)
        if not _await_config(state, pool):
            return 1
        return _consume_and_combine(state, pool, models_path, spawn)
    finally:
        _close_worker_pool(state, pool)


def run_parallel_pipeline(settings: StreamingSettings, dataset: DatasetConfig, models_path: str) -> int:
    """Run one streaming run and return its exit code; ``models_path`` is the models source to fetch once."""
    scans = _list_scans(settings, dataset)
    if scans is None:
        return 1
    if not scans:
        return 0
    fetched = _fetch_models(settings, dataset, models_path)
    if fetched is None:
        return 1
    if settings.dry_run:
        _log_dry_run(scans)
        if not _ensure_config(settings, dataset):
            return 1
    logger.trace("streaming run: %d scans, %d producer workers", len(scans), settings.producer_workers)
    with tempfile.TemporaryDirectory(prefix="champollion_streaming_") as flag_dir:
        state = _ProducerState(settings, dataset, scans, _ProducersDone(os.path.join(flag_dir, "producers_done")))
        return _run_streaming(state, fetched)
