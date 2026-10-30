"""Tests for the streaming orchestrator run_parallel_pipeline.

Task TASK-STREAMING-BDRABCZUK-D667773A2F2C (revises -6F4A62C83DCD and earlier, back to -FA7E85BD7877).
Dataflow (REWORK 2026-10-27):
list the scans of StreamingSettings.subjects_dir, run run_scan_worker per scan in a spawn ProcessPoolExecutor
alongside ONE run_consumer call, generate the dataset's Champollion config once, summarise, then run_combine once.
User decisions 2026-10-27: a non-empty DatasetConfig.embeddings_path is refused; the orchestrator fetches the models
once and passes the local path to the consumer; producers_done is any object with is_set().
User decisions 2026-10-28: graphs layout only (no raw T1, raw_scan never passed); a FAILED scan (worker FAILED,
worker raised, worker timed out, consumer FAILED) still lets the run combine and summarise, then exit 1; a call
running longer than worker_timeout is FAILED, counts as finished for producers_done, and holds its producer slot,
so scans left without a free slot are FAILED; in a real run the config comes from the first OK or SKIPPED scan's
crops root (never a FAILED one) and before run_consumer; in a dry run before the workers.

Nothing real runs: run_scan_worker, run_consumer, ensure_dataset_config, fetch_model_set and run_combine are
replaced in the runner module's namespace by recording fakes, and ProcessPoolExecutor by a thread pool that
records how it was created. The scans are listed by the real SubjectEligibilityChecker on a synthetic
Morphologist layout (<subjects_dir>/<subject>/t1mri/<acquisition>, folders only, no data). The runner module is
loaded inside each test body so a missing implementation fails each test on its own.

Interface the tests expect from champollion_pipeline.streaming.runner:
- run_parallel_pipeline(settings: StreamingSettings, dataset: DatasetConfig, models_path: str) -> int
  models_path is the models source (local path, URL or Hugging Face repo id) given to fetch_model_set.
- module-level names, looked up at call time: ProcessPoolExecutor (called with max_workers and mp_context),
  run_scan_worker, run_consumer, ensure_dataset_config, fetch_model_set (models source as first argument,
  returns (local path, source dict)), run_combine (called as run_combine(settings, dataset)).
  run_consumer is either called in the orchestrator process or submitted through ProcessPoolExecutor; it is
  never started through multiprocessing.Process directly (the fake would not run).
- logger champollion_pipeline.streaming.runner.
- summary records: one OUTPUT record holding ``<status value>=<count>`` for each WorkerStatus value, and one for
  each ConsumerStatus value.

Requirements (id, test):
- REQ-STREAMING-BDRABCZUK-652447A1BF64 test_one_worker_call_per_listed_scan
- REQ-STREAMING-BDRABCZUK-A8C27327136D test_producer_pool_size
- REQ-STREAMING-BDRABCZUK-F92DB6F26A12 test_producer_pool_uses_spawn
- REQ-STREAMING-BDRABCZUK-84CD3799C95A (supersedes -47029C477745, -99226E3ABA9F)
  test_one_consumer_call_with_listed_scans
- REQ-STREAMING-BDRABCZUK-69A7F1D0D41C test_consumer_starts_before_last_worker_finishes
- REQ-STREAMING-BDRABCZUK-C6EE33183C70 (supersedes -5ACD6FDB90E1) test_producers_done_unset_while_worker_runs
- REQ-STREAMING-BDRABCZUK-375CE2D5D564 test_producers_done_set_after_last_worker
- REQ-STREAMING-BDRABCZUK-F2CFB8A30656 test_consumer_gets_fetched_models_path
- REQ-STREAMING-BDRABCZUK-CC71AB8C65A4 test_fetch_failure_returns_1_before_workers
- REQ-STREAMING-BDRABCZUK-4AC332CADBD0 test_dry_run_fetches_no_models
- REQ-STREAMING-BDRABCZUK-0333BF2E716B test_dry_run_lists_scans_as_output
- REQ-STREAMING-BDRABCZUK-1BF0B2C75AB3 (supersedes -2E9EB7CDEA8A) test_dry_run_returns_0
- REQ-STREAMING-BDRABCZUK-A763AEC9BF9E test_embeddings_path_refused
- REQ-STREAMING-BDRABCZUK-880EB86EE90E test_missing_subjects_dir_returns_1
- REQ-STREAMING-BDRABCZUK-EDBF43290781 test_no_scans_warns_and_returns_0
- REQ-STREAMING-BDRABCZUK-68393836DBA7 (supersedes -42930A23C3A3) test_dataset_config_once_from_scan_crops_root
- REQ-STREAMING-BDRABCZUK-06307E0E3B3F test_dry_run_dataset_config_unchanged
- REQ-STREAMING-BDRABCZUK-51DE6EED1EE8 test_dataset_config_failure_returns_1
- REQ-STREAMING-BDRABCZUK-FDADCF64A446 test_consumer_failure_returns_1
- REQ-STREAMING-BDRABCZUK-46C1C41A0F54 test_combine_once_after_consumer
- REQ-STREAMING-BDRABCZUK-BC2A37681713 test_combine_failure_returns_1
- REQ-STREAMING-BDRABCZUK-457EFB45C234 test_success_returns_0
- REQ-STREAMING-BDRABCZUK-B20EDFE8D89C test_worker_summary_output_record
- REQ-STREAMING-BDRABCZUK-B4A21B125795 test_consumer_summary_output_record
- REQ-STREAMING-BDRABCZUK-93E37205681B test_failed_worker_scan_logs_error
- REQ-STREAMING-BDRABCZUK-0CA8B26AA177 test_failed_consumer_scan_logs_error
- REQ-STREAMING-BDRABCZUK-11252B9D71AA test_worker_exception_logs_error
- REQ-STREAMING-BDRABCZUK-8BAB8E819860 (supersedes -D0725801AFAB, -A8D97BD2714B)
  test_failed_worker_scan_combines_then_returns_1
- REQ-STREAMING-BDRABCZUK-589915AF2DE0 test_failed_consumer_scan_combines_then_returns_1
- REQ-STREAMING-BDRABCZUK-0996C0D4CF1C (supersedes -C53A8FB253BD, -E64FD8510F88)
  test_worker_exception_combines_then_returns_1
- REQ-STREAMING-BDRABCZUK-0FC93FECD0C9 test_timed_out_worker_is_failed
- REQ-STREAMING-BDRABCZUK-C4E5739458C0 test_remaining_scans_run_after_timeout
- REQ-STREAMING-BDRABCZUK-E3D4BFAE5BE7 test_scans_without_free_slot_are_failed
- REQ-STREAMING-BDRABCZUK-7DC7E74FA956 test_producers_done_set_with_only_timed_out_calls
- REQ-STREAMING-BDRABCZUK-CE9278D60E4F test_dataset_config_before_consumer
- REQ-STREAMING-BDRABCZUK-B4EADE4C7438 test_dry_run_dataset_config_before_workers
- REQ-STREAMING-BDRABCZUK-52CE446C3A81 (supersedes -CD4C51703F9D), REQ-STREAMING-BDRABCZUK-D7E3001BE808
  (supersedes -863C5D1A02BC) test_no_ok_or_skipped_scan_returns_1_without_consumer,
  test_skipped_scans_without_crops_root_return_1
- REQ-STREAMING-BDRABCZUK-537672FF1735, REQ-STREAMING-BDRABCZUK-004902F07AB2
  test_dry_run_without_ok_or_skipped_scan_returns_1
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import pytest

from champollion_pipeline.config import DatasetConfig
from champollion_pipeline.file_indexer import ScanId, SubjectEligibilityChecker
from champollion_pipeline.streaming import stages
from champollion_pipeline.streaming.consumer import CONSUMER_LOG_NAME, ConsumerResult, ConsumerStatus
from champollion_pipeline.streaming.settings import StreamingSettings
from champollion_pipeline.streaming.worker import WORKER_LOG_NAME, WorkerResult, WorkerStatus
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER

pytestmark = pytest.mark.unit  # noqa: V107

RUNNER_MODULE = "champollion_pipeline.streaming.runner"
OUTPUT_LEVEL = 25
MASKS = "synthetic_masks"
MODELS_SOURCE = "/synthetic/models_source"
FETCHED = "/synthetic/fetched_models"
CONFIGS_ROOT = "/synthetic/configs_root"
COMBINED = "/synthetic/combined"
WAIT = 5.0  # seconds a fake waits for the orchestrator before giving up (never reached when it behaves)
TIMEOUT = 0.3  # worker_timeout of the timeout tests
HANG = 10.0  # longest a hung fake worker blocks; the fixture releases it when the test ends

SCAN_A = ScanId("sub-01", "acq-a")
SCAN_B = ScanId("sub-02", "acq-b")
SCAN_C = ScanId("sub-03", "acq-c")
ALL_SCANS = (SCAN_A, SCAN_B, SCAN_C)


# --- fakes -------------------------------------------------------------------------------------------


class Harness:
    """Recording fakes for the runner's collaborators, and how they misbehave in one test."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.worker_calls: list[tuple] = []
        self.worker_threads: list[str] = []
        self.worker_status: dict[ScanId, WorkerStatus] = {}
        self.worker_errors: dict[ScanId, BaseException] = {}
        self.gated: set[ScanId] = set()  # workers of these scans wait until the consumer is called
        self.delays: dict[ScanId, float] = {}  # seconds a worker runs (default 0.05)
        self.hung: set[ScanId] = set()  # workers of these scans block until release (past any worker_timeout)
        self.release = threading.Event()
        self.running = 0  # calls running, hung ones excluded
        self.hung_running = 0
        self.events: list[str] = []  # "worker <path_prefix>", "config", "consumer", in call order
        self.finished: list[ScanId] = []
        self.consumer_called = threading.Event()
        self.consumer_calls: list[dict] = []
        self.consumer_status: dict[ScanId, ConsumerStatus] = {}
        self.consumer_error: BaseException | None = None
        self.running_at_consumer_call: int | None = None
        self.done_seen_while_running = False
        self.done_seen = False
        self.done_seen_while_hung = False
        self.consumer_returned = False
        self.fetch_calls: list[tuple] = []
        self.fetch_error: BaseException | None = None
        self.config_calls: list[tuple] = []
        self.config_error: BaseException | None = None
        self.combine_calls: list[tuple] = []
        self.combine_after_consumer: list[bool] = []
        self.combine_error: BaseException | None = None
        self.pools: list[FakePool] = []

    def worker(self, scan: ScanId, settings: StreamingSettings, dataset: DatasetConfig, *args, **kwargs):
        """Fake run_scan_worker: an optional raw scan path follows the dataset."""
        with self.lock:
            self.worker_calls.append((scan, settings, dataset, args, kwargs))
            self.worker_threads.append(threading.current_thread().name)
            self.events.append(f"worker {scan.path_prefix}")
            if scan in self.hung:
                self.hung_running += 1
            else:
                self.running += 1
        if scan in self.hung:
            try:
                self.release.wait(HANG)
                return WorkerResult(scan, WorkerStatus.OK)
            finally:
                with self.lock:
                    self.hung_running -= 1
        try:
            if scan in self.gated:
                self.consumer_called.wait(WAIT)
            else:
                time.sleep(self.delays.get(scan, 0.05))
            if scan in self.worker_errors:
                raise self.worker_errors[scan]
            status = self.worker_status.get(scan, WorkerStatus.OK)
            if status is WorkerStatus.OK:
                _crops_root(dataset, scan)  # what a real producer leaves behind
            return WorkerResult(scan, status)
        finally:
            with self.lock:
                self.running -= 1
                self.finished.append(scan)

    def consumer(self, settings, dataset, scans, producers_done, models_path):
        """Fake run_consumer: records the call, then waits for producers_done."""
        scans = list(scans)
        with self.lock:
            self.running_at_consumer_call = self.running
            self.events.append("consumer")
            self.consumer_calls.append(
                {"settings": settings, "dataset": dataset, "scans": scans, "models_path": models_path}
            )
        self._watch(producers_done)
        self.consumer_called.set()
        deadline = time.monotonic() + WAIT
        while not self._watch(producers_done) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.consumer_returned = True
        if self.consumer_error is not None:
            raise self.consumer_error
        return [ConsumerResult(scan, self.consumer_status.get(scan, ConsumerStatus.OK)) for scan in sorted(scans)]

    def _watch(self, producers_done) -> bool:
        with self.lock:
            running = self.running
            hung = self.hung_running
            done = bool(producers_done.is_set())
        if done:
            self.done_seen = True
            if running:
                self.done_seen_while_running = True
            if hung:
                self.done_seen_while_hung = True
        return done

    def fetch(self, *args, **kwargs):
        """Fake fetch_model_set: the models source comes first; returns (local path, source dict)."""
        self.fetch_calls.append((args, kwargs))
        if self.fetch_error is not None:
            raise self.fetch_error
        return FETCHED, {"type": "local", "path": FETCHED}

    def config(self, settings, dataset):
        """Fake ensure_dataset_config."""
        self.config_calls.append((settings, dataset))
        with self.lock:
            self.events.append("config")
        if self.config_error is not None:
            raise self.config_error
        return CONFIGS_ROOT

    def combine(self, settings, dataset):
        """Fake run_combine."""
        self.combine_calls.append((settings, dataset))
        self.combine_after_consumer.append(self.consumer_returned)
        if self.combine_error is not None:
            raise self.combine_error
        return COMBINED

    def fetched_source(self) -> str:
        args, kwargs = self.fetch_calls[0]
        return args[0] if args else kwargs["models_path"]

    def pool_of_workers(self) -> FakePool:
        """The pool whose threads ran the run_scan_worker calls."""
        names = set(self.worker_threads)
        pools = [pool for pool in self.pools if any(name.startswith(pool.prefix) for name in names)]
        assert len(pools) == 1, f"run_scan_worker calls ran in {len(pools)} recorded pools: {names}"
        return pools[0]


class FakePool(ThreadPoolExecutor):
    """Stand-in for ProcessPoolExecutor: threads, wide enough never to deadlock the fakes; records its arguments."""

    harness: Harness

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kwargs) -> None:
        self.requested_max_workers = max_workers
        self.mp_context = mp_context
        self.extra = kwargs
        self.prefix = f"fakepool{len(self.harness.pools)}"
        self.harness.pools.append(self)
        super().__init__(max_workers=16, thread_name_prefix=self.prefix, initializer=initializer, initargs=initargs)


# --- fixtures ----------------------------------------------------------------------------------------


def _make_layout(subjects_dir: Path, scans=ALL_SCANS) -> None:
    for scan in scans:
        (subjects_dir / scan.path_prefix).mkdir(parents=True)


def _crops_root(dataset: DatasetConfig, scan: ScanId) -> str:
    """Create the 2mm crops directory of one scan and return it as find_scan_crops_root reports it."""
    tiles_dir = stages.compute_scan_tiles_dir(dataset.cortical_tiles_output, scan)
    os.makedirs(os.path.join(tiles_dir, DERIVATIVES_FOLDER, "crops", MASKS, "2mm"), exist_ok=True)
    found = stages.find_scan_crops_root(tiles_dir, MASKS)
    assert found is not None
    return found


class _Context:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.subjects_dir = tmp_path / "subjects"
        self.subjects_dir.mkdir()
        _make_layout(self.subjects_dir)
        self.h = Harness()
        self.settings = self.make_settings()
        self.dataset = DatasetConfig(
            name="synthetic_ds",
            datasets_root=str(tmp_path / "synthetic_ds"),
            cortical_tiles_output=str(tmp_path / "synthetic_ds" / "derivatives" / "cortical_tiles"),
            crops_path=str(tmp_path / "configured_crops"),
            masks_version=MASKS,
            embeddings_path="",
        )

    def make_settings(self, **overrides) -> StreamingSettings:
        values = {
            "subjects_dir": str(self.subjects_dir),
            "output_dir": str(self.tmp_path / "out"),
            "log_dir": str(self.tmp_path / "logs"),
            "n_workers": 3,
            "poll_interval": 0.01,
            "worker_timeout": WAIT,
            "consumer_stall_timeout": WAIT,
        }
        values.update(overrides)
        return StreamingSettings(**values)

    def load(self) -> ModuleType:
        runner = importlib.import_module(RUNNER_MODULE)
        FakePool.harness = self.h
        patch = self.monkeypatch.setattr
        # raising=False: the stub defines none of these names yet; green binds them at module level.
        patch(runner, "ProcessPoolExecutor", FakePool, raising=False)
        patch(runner, "run_scan_worker", self.h.worker, raising=False)
        patch(runner, "run_consumer", self.h.consumer, raising=False)
        patch(runner, "fetch_model_set", self.h.fetch, raising=False)
        patch(runner, "ensure_dataset_config", self.h.config, raising=False)
        patch(runner, "run_combine", self.h.combine, raising=False)
        return runner

    def run(self, settings: StreamingSettings | None = None, dataset: DatasetConfig | None = None) -> int:
        runner = self.load()
        return runner.run_parallel_pipeline(settings or self.settings, dataset or self.dataset, MODELS_SOURCE)


@pytest.fixture
def ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    context = _Context(tmp_path, monkeypatch)
    yield context
    context.h.release.set()  # let hung fake workers return


def _listed(subjects_dir: Path) -> list[ScanId]:
    return SubjectEligibilityChecker(str(subjects_dir), []).list_scans()


def _errors(pipeline_caplog) -> list[str]:
    return [r.getMessage() for r in pipeline_caplog.records if r.levelno == logging.ERROR]


# --- scans, pool, consumer -----------------------------------------------------------------------------


def test_one_worker_call_per_listed_scan(ctx):
    ctx.run()
    called = sorted(call[0] for call in ctx.h.worker_calls)
    assert called == sorted(_listed(ctx.subjects_dir))
    for _scan, settings, dataset, _args, _kwargs in ctx.h.worker_calls:
        assert settings is ctx.settings
        assert dataset.name == ctx.dataset.name


def test_producer_pool_size(ctx):
    ctx.settings = ctx.make_settings(n_workers=2)
    ctx.run()
    assert ctx.h.pool_of_workers().requested_max_workers == ctx.settings.producer_workers == 2


def test_producer_pool_uses_spawn(ctx):
    ctx.run()
    context = ctx.h.pool_of_workers().mp_context
    assert context is not None, "ProcessPoolExecutor created without mp_context"
    assert context.get_start_method() == "spawn"


def test_one_consumer_call_with_listed_scans(ctx):
    ctx.run()
    assert len(ctx.h.consumer_calls) == 1
    call = ctx.h.consumer_calls[0]
    assert call["settings"] is ctx.settings
    assert call["dataset"].name == ctx.dataset.name
    assert sorted(call["scans"]) == sorted(_listed(ctx.subjects_dir))


def test_consumer_starts_before_last_worker_finishes(ctx):
    ctx.h.gated = {SCAN_C}
    ctx.run()
    assert ctx.h.consumer_calls, "run_consumer never called"
    assert ctx.h.running_at_consumer_call, "run_consumer was called only after every run_scan_worker call finished"


def test_producers_done_unset_while_worker_runs(ctx):
    ctx.h.gated = {SCAN_C}
    ctx.run()
    assert ctx.h.consumer_calls, "run_consumer never called"
    assert not ctx.h.done_seen_while_running, "producers_done was set while a run_scan_worker call was running"


def test_producers_done_set_after_last_worker(ctx):
    ctx.run()
    assert ctx.h.consumer_calls, "run_consumer never called"
    assert ctx.h.done_seen, f"producers_done never set within {WAIT}s after the workers finished"
    assert len(ctx.h.finished) == len(ALL_SCANS)


# --- models --------------------------------------------------------------------------------------------


def test_consumer_gets_fetched_models_path(ctx):
    ctx.run()
    assert len(ctx.h.fetch_calls) == 1
    assert ctx.h.fetched_source() == MODELS_SOURCE
    assert [call["models_path"] for call in ctx.h.consumer_calls] == [FETCHED]


def test_fetch_failure_returns_1_before_workers(ctx, pipeline_caplog):
    ctx.h.fetch_error = RuntimeError("synthetic fetch failure 41c7")
    assert ctx.run() == 1
    assert ctx.h.worker_calls == []
    pipeline_caplog.assert_logged(logging.ERROR, "synthetic fetch failure 41c7")


def test_dry_run_fetches_no_models(ctx):
    ctx.run(settings=ctx.make_settings(dry_run=True))
    assert ctx.h.fetch_calls == []
    assert [call["models_path"] for call in ctx.h.consumer_calls] == [MODELS_SOURCE]


# --- dry run -------------------------------------------------------------------------------------------


def test_dry_run_lists_scans_as_output(ctx, pipeline_caplog):
    ctx.run(settings=ctx.make_settings(dry_run=True))
    outputs = [r.getMessage() for r in pipeline_caplog.records if r.levelno == OUTPUT_LEVEL]
    for scan in ALL_SCANS:
        hits = [m for m in outputs if scan.path_prefix in m]
        assert len(hits) == 1, f"{scan.path_prefix}: {len(hits)} OUTPUT records in {outputs}"


def test_dry_run_returns_0(ctx):
    assert ctx.run(settings=ctx.make_settings(dry_run=True)) == 0


# --- input checks --------------------------------------------------------------------------------------


def test_embeddings_path_refused(ctx, pipeline_caplog):
    ctx.dataset = dataclasses.replace(ctx.dataset, embeddings_path="/synthetic/explicit_embeddings")
    assert ctx.run() == 1
    assert ctx.h.worker_calls == []
    assert any("embeddings_path" in m for m in _errors(pipeline_caplog)), _errors(pipeline_caplog)


def test_missing_subjects_dir_returns_1(ctx, pipeline_caplog):
    missing = str(ctx.tmp_path / "no_such_subjects")
    assert ctx.run(settings=ctx.make_settings(subjects_dir=missing)) == 1
    assert ctx.h.worker_calls == []
    pipeline_caplog.assert_logged(logging.ERROR, missing)


def test_no_scans_warns_and_returns_0(ctx, pipeline_caplog):
    empty = ctx.tmp_path / "empty_subjects"
    empty.mkdir()
    assert ctx.run(settings=ctx.make_settings(subjects_dir=str(empty))) == 0
    assert ctx.h.worker_calls == []
    pipeline_caplog.assert_logged(logging.WARNING, str(empty))


# --- dataset config ------------------------------------------------------------------------------------


def test_dataset_config_once_from_scan_crops_root(ctx):
    _crops_root(ctx.dataset, SCAN_A)  # partial crops of a FAILED tiles run, finished first: never the source
    crops_b = _crops_root(ctx.dataset, SCAN_B)
    ctx.h.worker_status = {SCAN_A: WorkerStatus.FAILED, SCAN_B: WorkerStatus.SKIPPED, SCAN_C: WorkerStatus.FAILED}
    ctx.h.delays = {SCAN_A: 0.0, SCAN_B: 0.3, SCAN_C: 0.0}
    ctx.run()
    assert len(ctx.h.config_calls) == 1
    settings, dataset = ctx.h.config_calls[0]
    assert settings is ctx.settings
    assert dataset.crops_path == crops_b
    assert dataset.name == ctx.dataset.name


def test_dry_run_dataset_config_unchanged(ctx):
    _crops_root(ctx.dataset, SCAN_B)  # a crops root exists, yet a dry run keeps the configured crops_path
    ctx.run(settings=ctx.make_settings(dry_run=True))
    assert len(ctx.h.config_calls) == 1
    assert ctx.h.config_calls[0][1] == ctx.dataset


def test_dataset_config_failure_returns_1(ctx, pipeline_caplog):
    ctx.h.config_error = RuntimeError("synthetic config failure 9d02")
    assert ctx.run() == 1
    pipeline_caplog.assert_logged(logging.ERROR, "synthetic config failure 9d02")


# --- consumer and combine ------------------------------------------------------------------------------


def test_consumer_failure_returns_1(ctx, pipeline_caplog):
    ctx.h.consumer_error = ValueError("synthetic consumer failure 77ab")
    assert ctx.run() == 1
    pipeline_caplog.assert_logged(logging.ERROR, "synthetic consumer failure 77ab")


def test_combine_once_after_consumer(ctx):
    ctx.run()
    assert len(ctx.h.combine_calls) == 1
    settings, dataset = ctx.h.combine_calls[0]
    assert settings is ctx.settings
    assert dataset.name == ctx.dataset.name
    assert ctx.h.combine_after_consumer == [True]


def test_combine_failure_returns_1(ctx, pipeline_caplog):
    ctx.h.combine_error = FileNotFoundError("synthetic combine failure 3e15")
    assert ctx.run() == 1
    pipeline_caplog.assert_logged(logging.ERROR, "synthetic combine failure 3e15")


def test_success_returns_0(ctx, pipeline_caplog):
    ctx.h.worker_status = {SCAN_B: WorkerStatus.SKIPPED}
    ctx.h.consumer_status = {SCAN_C: ConsumerStatus.SKIPPED}
    assert ctx.run() == 0
    assert len(ctx.h.combine_calls) == 1
    assert _errors(pipeline_caplog) == []


# --- summary and failure records -----------------------------------------------------------------------


def _summary_records(pipeline_caplog, statuses) -> list[str]:
    outputs = [r.getMessage() for r in pipeline_caplog.records if r.levelno == OUTPUT_LEVEL]
    return [m for m in outputs if all(f"{status.value}=" in m for status in statuses)]


def test_worker_summary_output_record(ctx, pipeline_caplog):
    ctx.h.worker_status = {SCAN_B: WorkerStatus.SKIPPED, SCAN_C: WorkerStatus.FAILED}
    ctx.run()
    hits = _summary_records(pipeline_caplog, WorkerStatus)
    hits = [m for m in hits if all(f"{s.value}=1" in m for s in WorkerStatus)]
    assert len(hits) == 1, f"worker summary records: {hits}"


def test_consumer_summary_output_record(ctx, pipeline_caplog):
    ctx.h.consumer_status = {SCAN_A: ConsumerStatus.OK, SCAN_B: ConsumerStatus.OK, SCAN_C: ConsumerStatus.FAILED}
    ctx.run()
    hits = _summary_records(pipeline_caplog, ConsumerStatus)
    hits = [m for m in hits if "ok=2" in m and "skipped=0" in m and "failed=1" in m]
    assert len(hits) == 1, f"consumer summary records: {hits}"


def test_failed_worker_scan_logs_error(ctx, pipeline_caplog):
    ctx.h.worker_status = {SCAN_B: WorkerStatus.FAILED}
    ctx.run()
    log_file = os.path.join(ctx.settings.log_dir, SCAN_B.path_prefix, WORKER_LOG_NAME)
    hits = [m for m in _errors(pipeline_caplog) if SCAN_B.path_prefix in m and log_file in m]
    assert len(hits) == 1, f"ERROR records: {_errors(pipeline_caplog)}"


def test_failed_consumer_scan_logs_error(ctx, pipeline_caplog):
    ctx.h.consumer_status = {SCAN_C: ConsumerStatus.FAILED}
    ctx.run()
    log_file = os.path.join(ctx.settings.log_dir, CONSUMER_LOG_NAME)
    hits = [m for m in _errors(pipeline_caplog) if SCAN_C.path_prefix in m and log_file in m]
    assert len(hits) == 1, f"ERROR records: {_errors(pipeline_caplog)}"


def test_worker_exception_logs_error(ctx, pipeline_caplog):
    ctx.h.worker_errors = {SCAN_B: RuntimeError("synthetic pool failure 5c88")}
    ctx.run()
    hits = [m for m in _errors(pipeline_caplog) if SCAN_B.path_prefix in m and "synthetic pool failure 5c88" in m]
    assert hits, f"ERROR records: {_errors(pipeline_caplog)}"


# --- exit code 1 on failed scans (user decision 2026-10-28) --------------------------------------------


def test_failed_worker_scan_combines_then_returns_1(ctx):
    ctx.h.worker_status = {SCAN_B: WorkerStatus.FAILED}
    assert ctx.run() == 1
    assert len(ctx.h.combine_calls) == 1


def test_failed_consumer_scan_combines_then_returns_1(ctx):
    ctx.h.consumer_status = {SCAN_C: ConsumerStatus.FAILED}
    assert ctx.run() == 1
    assert len(ctx.h.combine_calls) == 1


def test_worker_exception_combines_then_returns_1(ctx):
    ctx.h.worker_errors = {SCAN_B: RuntimeError("synthetic pool failure 61f0")}
    assert ctx.run() == 1
    assert len(ctx.h.combine_calls) == 1


# --- worker_timeout (user decision 2026-10-28) ---------------------------------------------------------


def test_timed_out_worker_is_failed(ctx, pipeline_caplog):
    ctx.settings = ctx.make_settings(n_workers=3, worker_timeout=TIMEOUT)
    ctx.h.hung = {SCAN_A}
    ctx.run()
    hits = [m for m in _summary_records(pipeline_caplog, WorkerStatus) if "ok=2" in m and "failed=1" in m]
    assert len(hits) == 1, f"worker summary records: {_summary_records(pipeline_caplog, WorkerStatus)}"
    log_file = os.path.join(ctx.settings.log_dir, SCAN_A.path_prefix, WORKER_LOG_NAME)
    assert [m for m in _errors(pipeline_caplog) if SCAN_A.path_prefix in m and log_file in m], _errors(pipeline_caplog)


def test_remaining_scans_run_after_timeout(ctx, pipeline_caplog):
    # Two slots, worker_timeout 1 s. SCAN_A hangs in one slot and times out at ~1.0 s. SCAN_B (0.5 s) and then
    # SCAN_C (0.8 s, from ~0.5 s to ~1.3 s) run in the other slot, each well within worker_timeout, so SCAN_C is
    # still running when SCAN_A times out and must still end OK.
    ctx.settings = ctx.make_settings(n_workers=2, worker_timeout=1.0)
    ctx.h.hung = {SCAN_A}
    ctx.h.delays = {SCAN_B: 0.5, SCAN_C: 0.8}
    ctx.run()
    assert sorted(call[0] for call in ctx.h.worker_calls) == list(ALL_SCANS)
    assert SCAN_C in ctx.h.finished
    hits = [m for m in _summary_records(pipeline_caplog, WorkerStatus) if "ok=2" in m and "failed=1" in m]
    assert len(hits) == 1, f"worker summary records: {_summary_records(pipeline_caplog, WorkerStatus)}"


def test_scans_without_free_slot_are_failed(ctx, pipeline_caplog):
    ctx.settings = ctx.make_settings(n_workers=1, worker_timeout=TIMEOUT)
    ctx.h.hung = {SCAN_A}
    assert ctx.run() == 1
    assert [call[0] for call in ctx.h.worker_calls] == [SCAN_A]
    for scan in (SCAN_B, SCAN_C):
        assert [m for m in _errors(pipeline_caplog) if scan.path_prefix in m], (scan, _errors(pipeline_caplog))


def test_producers_done_set_with_only_timed_out_calls(ctx):
    ctx.settings = ctx.make_settings(n_workers=3, worker_timeout=TIMEOUT)
    ctx.h.hung = {SCAN_A}
    ctx.run()
    assert ctx.h.consumer_calls, "run_consumer never called"
    assert ctx.h.done_seen_while_hung, "producers_done not set while only a timed-out call was running"


# --- dataset config order (user decision 2026-10-28) ---------------------------------------------------


def test_dataset_config_before_consumer(ctx):
    ctx.run()
    assert "config" in ctx.h.events, ctx.h.events
    assert "consumer" in ctx.h.events, ctx.h.events
    assert ctx.h.events.index("config") < ctx.h.events.index("consumer"), ctx.h.events


def test_dry_run_dataset_config_before_workers(ctx):
    ctx.run(settings=ctx.make_settings(dry_run=True))
    assert "config" in ctx.h.events, ctx.h.events
    workers = [i for i, event in enumerate(ctx.h.events) if event.startswith("worker ")]
    assert workers, ctx.h.events
    assert ctx.h.events.index("config") < workers[0], ctx.h.events


# --- no OK or SKIPPED scan: no crops root for the dataset config ---------------------------------------


def test_no_ok_or_skipped_scan_returns_1_without_consumer(ctx, pipeline_caplog):
    _crops_root(ctx.dataset, SCAN_A)  # partial crops of a FAILED tiles run: never a config source
    ctx.h.worker_status = dict.fromkeys(ALL_SCANS, WorkerStatus.FAILED)
    assert ctx.run() == 1
    assert ctx.h.consumer_calls == []
    assert ctx.h.combine_calls == []
    assert ctx.h.config_calls == []
    hits = [m for m in _summary_records(pipeline_caplog, WorkerStatus) if "failed=3" in m]
    assert len(hits) == 1, f"worker summary records: {_summary_records(pipeline_caplog, WorkerStatus)}"
    assert [m for m in _errors(pipeline_caplog) if ctx.dataset.cortical_tiles_output in m], _errors(pipeline_caplog)


def test_skipped_scans_without_crops_root_return_1(ctx, pipeline_caplog):
    ctx.h.worker_status = dict.fromkeys(ALL_SCANS, WorkerStatus.SKIPPED)  # resumed, yet no crops directory left
    assert ctx.run() == 1
    assert ctx.h.consumer_calls == []
    assert ctx.h.combine_calls == []
    assert ctx.h.config_calls == []
    hits = [m for m in _summary_records(pipeline_caplog, WorkerStatus) if "skipped=3" in m]
    assert len(hits) == 1, f"worker summary records: {_summary_records(pipeline_caplog, WorkerStatus)}"
    assert [m for m in _errors(pipeline_caplog) if ctx.dataset.cortical_tiles_output in m], _errors(pipeline_caplog)


def test_dry_run_without_ok_or_skipped_scan_returns_1(ctx, pipeline_caplog):
    ctx.h.worker_status = dict.fromkeys(ALL_SCANS, WorkerStatus.FAILED)
    assert ctx.run(settings=ctx.make_settings(dry_run=True)) == 1
    assert ctx.h.consumer_calls == []
    assert ctx.h.combine_calls == []
    assert len(ctx.h.config_calls) == 1  # a dry run generates the config before the workers
    hits = [m for m in _summary_records(pipeline_caplog, WorkerStatus) if "failed=3" in m]
    assert len(hits) == 1, f"worker summary records: {_summary_records(pipeline_caplog, WorkerStatus)}"
    assert [m for m in _errors(pipeline_caplog) if ctx.dataset.cortical_tiles_output in m], _errors(pipeline_caplog)
