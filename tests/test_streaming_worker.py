"""Tests for the per-scan producer worker of a streaming run.

Task TASK-STREAMING-BDRABCZUK-839D9801428D: run_scan_worker runs Morphologist (when given a raw scan) then
cortical_tiles for ONE scan in a ProcessPool child, never raises, returns a WorkerResult and writes a per-worker
log file in the Morphologist layout <log_dir>/<subject>/t1mri/<acquisition>/worker.log.

Morphologist and cortical_tiles never run: the producer stage functions and init_pipeline_process are replaced,
in the worker module's namespace, by recording stand-ins. The worker module is loaded inside each test body
(_Context.load) so a missing implementation fails each test on its own instead of breaking collection.

Interface the tests expect from champollion_pipeline.streaming.worker:
- run_scan_worker(scan, settings, dataset, raw_scan=None) -> WorkerResult
- WorkerResult(scan, status), WorkerStatus.OK / SKIPPED / FAILED
- module-level names init_pipeline_process, run_morphologist_for_scan, run_cortical_tiles_for_scan

Requirements:
- REQ-STREAMING-BDRABCZUK-9E6539A4B814 Morphologist before tiles when a raw scan is given
- REQ-STREAMING-BDRABCZUK-951FF1A3D3FC tiles only without a raw scan
- REQ-STREAMING-BDRABCZUK-6098AF4B5943 success status OK
- REQ-STREAMING-BDRABCZUK-37C991204398 result holds the scan
- REQ-STREAMING-BDRABCZUK-329CD83E2985 stage Exception gives FAILED without raising
- REQ-STREAMING-BDRABCZUK-0A25ADBE3D3D stage SystemExit gives FAILED without raising
- REQ-STREAMING-BDRABCZUK-8BBB3CFFDC39 no tiles after a Morphologist failure
- REQ-STREAMING-BDRABCZUK-CEB42AC25B4F ERROR record on a stage failure
- REQ-STREAMING-BDRABCZUK-967B6F136F92 dry run skips
- REQ-STREAMING-BDRABCZUK-334C117E751A existing crops skip (resume)
- REQ-STREAMING-BDRABCZUK-B40295BAE4FC OUTPUT record on a skip
- REQ-STREAMING-BDRABCZUK-48DEC870F8C8 OUTPUT record naming the per-scan tiles directory on success
- REQ-STREAMING-BDRABCZUK-4299F1D39459 init_pipeline_process once, before the stages
- REQ-STREAMING-BDRABCZUK-3C3CC0AF76DF per-worker log file captures fds 1 and 2, Morphologist layout path
- REQ-STREAMING-BDRABCZUK-E96ED01D6C2D unusable log file gives FAILED without raising
- REQ-STREAMING-BDRABCZUK-C7C6A43D6E5A WorkerResult pickle round trip
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import os
import pickle
from pathlib import Path
from types import ModuleType

import pytest

from champollion_pipeline.config import DatasetConfig
from champollion_pipeline.file_indexer import ScanId
from champollion_pipeline.streaming import stages
from champollion_pipeline.streaming.settings import StreamingSettings
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER

WORKER_MODULE = "champollion_pipeline.streaming.worker"
OUTPUT_LEVEL = 25
MASKS_VERSION = "synthetic_masks"
STDOUT_MARKER = "tiles-stdout-marker-7f3a"
STDERR_MARKER = "tiles-stderr-marker-9c1d"

# One scan per Morphologist flavour: classic (str() is the subject alone) and BIDS (str() is sub-01_ses-1_run-1,
# which must never become a path).
CLASSIC_SCAN = ScanId("subj01", "default_acquisition")
BIDS_SCAN = ScanId("sub-01", "ses-1_run-1", bids=True)
SCANS = pytest.mark.parametrize("scan", [CLASSIC_SCAN, BIDS_SCAN], ids=["classic", "bids"])
STAGE_NAMES = pytest.mark.parametrize("failing", ["morphologist", "tiles"])


class _FakeStages:
    """Recording stand-ins for init_pipeline_process and the two producer stage functions."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.errors: dict[str, BaseException] = {}
        self.write_fds = False

    def init_pipeline_process(self) -> logging.Logger:
        self.calls.append(("init",))
        return logging.getLogger()

    def run_morphologist_for_scan(self, raw_scan: str, settings: StreamingSettings, dataset: DatasetConfig) -> None:
        self.calls.append(("morphologist", raw_scan, settings, dataset))
        if "morphologist" in self.errors:
            raise self.errors["morphologist"]

    def run_cortical_tiles_for_scan(self, scan: ScanId, settings: StreamingSettings, dataset: DatasetConfig) -> None:
        self.calls.append(("tiles", scan, settings, dataset))
        if self.write_fds:
            os.write(1, f"{STDOUT_MARKER}\n".encode())
            os.write(2, f"{STDERR_MARKER}\n".encode())
        if "tiles" in self.errors:
            raise self.errors["tiles"]

    def stage_calls(self) -> list[tuple]:
        return [call for call in self.calls if call[0] != "init"]


@dataclasses.dataclass
class _Context:
    """One test's fake stages, settings, dataset, raw scan and worker-logger records."""

    monkeypatch: pytest.MonkeyPatch
    fake: _FakeStages
    settings: StreamingSettings
    dataset: DatasetConfig
    raw_scan: str
    records: list[logging.LogRecord]

    def load(self) -> ModuleType:
        """Import the worker module and replace its stage functions and process setup with the fake's."""
        worker = importlib.import_module(WORKER_MODULE)
        for name in ("init_pipeline_process", "run_morphologist_for_scan", "run_cortical_tiles_for_scan"):
            self.monkeypatch.setattr(worker, name, getattr(self.fake, name))
        return worker

    def messages(self, level: int) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno == level]


def _settings(tmp_path: Path) -> StreamingSettings:
    subjects = tmp_path / "subjects"
    for scan in (CLASSIC_SCAN, BIDS_SCAN):
        (subjects / scan.path_prefix).mkdir(parents=True)
    return StreamingSettings(
        subjects_dir=str(subjects),
        output_dir=str(tmp_path / "out"),
        log_dir=str(tmp_path / "logs"),
        n_workers=2,
    )


def _dataset(tmp_path: Path) -> DatasetConfig:
    morphologist = tmp_path / "morphologist"
    morphologist.mkdir()
    return DatasetConfig(
        name="synthetic",
        morphologist_graphs=str(morphologist),
        cortical_tiles_output=str(tmp_path / "derivatives"),
        masks_version=MASKS_VERSION,
    )


def _raw_scan(tmp_path: Path) -> str:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw = raw_dir / "sub-01.nii.gz"
    raw.write_bytes(b"synthetic")
    return str(raw)


@pytest.fixture
def ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Context:
    """Fresh context; the worker module logger's records are captured on the logger itself."""
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    worker_logger = logging.getLogger(WORKER_MODULE)
    handler = _ListHandler(level=1)
    level = worker_logger.level
    worker_logger.addHandler(handler)
    worker_logger.setLevel(1)
    yield _Context(monkeypatch, _FakeStages(), _settings(tmp_path), _dataset(tmp_path), _raw_scan(tmp_path), records)
    worker_logger.removeHandler(handler)
    worker_logger.setLevel(level)


def _make_crops(dataset: DatasetConfig, scan: ScanId) -> str:
    """Create the scan's 2mm crops directory, so stages.find_scan_crops_root finds it."""
    tiles_dir = stages.compute_scan_tiles_dir(dataset.cortical_tiles_output, scan)
    os.makedirs(os.path.join(tiles_dir, DERIVATIVES_FOLDER, "crops", MASKS_VERSION, "2mm"))
    crops = stages.find_scan_crops_root(tiles_dir, MASKS_VERSION)
    assert crops is not None
    return crops


# --- stage sequence -------------------------------------------------------------------------------


@SCANS
def test_raw_scan_runs_morphologist_then_tiles(ctx: _Context, scan: ScanId) -> None:
    worker = ctx.load()
    worker.run_scan_worker(scan, ctx.settings, ctx.dataset, raw_scan=ctx.raw_scan)
    assert ctx.fake.stage_calls() == [
        ("morphologist", ctx.raw_scan, ctx.settings, ctx.dataset),
        ("tiles", scan, ctx.settings, ctx.dataset),
    ]


@SCANS
def test_no_raw_scan_runs_tiles_only(ctx: _Context, scan: ScanId) -> None:
    worker = ctx.load()
    worker.run_scan_worker(scan, ctx.settings, ctx.dataset)
    assert ctx.fake.stage_calls() == [("tiles", scan, ctx.settings, ctx.dataset)]


@pytest.mark.parametrize("with_raw", [True, False], ids=["raw_scan", "no_raw_scan"])
def test_success_returns_ok(ctx: _Context, with_raw: bool) -> None:
    worker = ctx.load()
    result = worker.run_scan_worker(BIDS_SCAN, ctx.settings, ctx.dataset, raw_scan=ctx.raw_scan if with_raw else None)
    assert result.status == worker.WorkerStatus.OK


@SCANS
@pytest.mark.parametrize("outcome", ["ok", "skipped", "failed"])
def test_result_holds_scan(ctx: _Context, scan: ScanId, outcome: str) -> None:
    worker = ctx.load()
    settings = dataclasses.replace(ctx.settings, dry_run=True) if outcome == "skipped" else ctx.settings
    if outcome == "failed":
        ctx.fake.errors["tiles"] = RuntimeError("tiles failed")
    result = worker.run_scan_worker(scan, settings, ctx.dataset)
    assert result.scan == scan
    assert result.status == worker.WorkerStatus(outcome)


# --- failures never raise -------------------------------------------------------------------------


@STAGE_NAMES
def test_stage_exception_returns_failed(ctx: _Context, failing: str) -> None:
    worker = ctx.load()
    ctx.fake.errors[failing] = RuntimeError(f"{failing} failed")
    result = worker.run_scan_worker(CLASSIC_SCAN, ctx.settings, ctx.dataset, raw_scan=ctx.raw_scan)
    assert result.status == worker.WorkerStatus.FAILED


@STAGE_NAMES
def test_stage_system_exit_returns_failed(ctx: _Context, failing: str) -> None:
    worker = ctx.load()
    ctx.fake.errors[failing] = SystemExit(2)
    result = worker.run_scan_worker(CLASSIC_SCAN, ctx.settings, ctx.dataset, raw_scan=ctx.raw_scan)
    assert result.status == worker.WorkerStatus.FAILED


@pytest.mark.parametrize("error", [RuntimeError("Morphologist failed"), SystemExit(2)], ids=["exception", "exit"])
def test_morphologist_failure_skips_tiles(ctx: _Context, error: BaseException) -> None:
    worker = ctx.load()
    ctx.fake.errors["morphologist"] = error
    worker.run_scan_worker(CLASSIC_SCAN, ctx.settings, ctx.dataset, raw_scan=ctx.raw_scan)
    assert [call[0] for call in ctx.fake.stage_calls()] == ["morphologist"]


@SCANS
@STAGE_NAMES
def test_stage_failure_logs_error(ctx: _Context, scan: ScanId, failing: str) -> None:
    worker = ctx.load()
    message = f"boom from {failing} 4e1b"
    ctx.fake.errors[failing] = RuntimeError(message)
    worker.run_scan_worker(scan, ctx.settings, ctx.dataset, raw_scan=ctx.raw_scan)
    errors = ctx.messages(logging.ERROR)
    assert any(scan.path_prefix in text and message in text for text in errors), errors


# --- skips ----------------------------------------------------------------------------------------


@SCANS
def test_dry_run_skips(ctx: _Context, scan: ScanId) -> None:
    worker = ctx.load()
    settings = dataclasses.replace(ctx.settings, dry_run=True)
    result = worker.run_scan_worker(scan, settings, ctx.dataset, raw_scan=ctx.raw_scan)
    assert result.status == worker.WorkerStatus.SKIPPED
    assert ctx.fake.stage_calls() == []


@SCANS
def test_existing_crops_skips(ctx: _Context, scan: ScanId) -> None:
    _make_crops(ctx.dataset, scan)
    worker = ctx.load()
    result = worker.run_scan_worker(scan, ctx.settings, ctx.dataset, raw_scan=ctx.raw_scan)
    assert result.status == worker.WorkerStatus.SKIPPED
    assert ctx.fake.stage_calls() == []


@SCANS
@pytest.mark.parametrize("reason", ["dry_run", "existing_crops"])
def test_skip_logs_output(ctx: _Context, scan: ScanId, reason: str) -> None:
    worker = ctx.load()
    settings = ctx.settings
    if reason == "dry_run":
        settings = dataclasses.replace(settings, dry_run=True)
    else:
        _make_crops(ctx.dataset, scan)
    worker.run_scan_worker(scan, settings, ctx.dataset)
    outputs = ctx.messages(OUTPUT_LEVEL)
    assert any(scan.path_prefix in text for text in outputs), outputs


# --- success logging and process setup -------------------------------------------------------------


@SCANS
def test_success_logs_output_tiles_dir(ctx: _Context, scan: ScanId) -> None:
    worker = ctx.load()
    worker.run_scan_worker(scan, ctx.settings, ctx.dataset)
    tiles_dir = stages.compute_scan_tiles_dir(ctx.dataset.cortical_tiles_output, scan)
    outputs = ctx.messages(OUTPUT_LEVEL)
    assert any(tiles_dir in text for text in outputs), outputs


def test_init_pipeline_process_before_stages(ctx: _Context) -> None:
    worker = ctx.load()
    worker.run_scan_worker(CLASSIC_SCAN, ctx.settings, ctx.dataset, raw_scan=ctx.raw_scan)
    assert [call[0] for call in ctx.fake.calls] == ["init", "morphologist", "tiles"]


# --- per-worker log file -----------------------------------------------------------------------------


@SCANS
def test_worker_log_file_captures_output(ctx: _Context, scan: ScanId) -> None:
    worker = ctx.load()
    ctx.fake.write_fds = True
    worker.run_scan_worker(scan, ctx.settings, ctx.dataset)
    log_dir = Path(ctx.settings.log_dir)
    log_file = log_dir / scan.path_prefix / "worker.log"
    assert log_file.is_file()
    text = log_file.read_text()
    assert STDOUT_MARKER in text
    assert STDERR_MARKER in text
    if str(scan) != scan.subject:
        assert not (log_dir / str(scan)).exists()


def test_unusable_log_file_returns_failed(ctx: _Context, tmp_path: Path) -> None:
    worker = ctx.load()
    blocker = tmp_path / "log_dir_is_a_file"
    blocker.write_text("not a directory")
    settings = dataclasses.replace(ctx.settings, log_dir=str(blocker))
    result = worker.run_scan_worker(CLASSIC_SCAN, settings, ctx.dataset)
    assert result.status == worker.WorkerStatus.FAILED


# --- result transport --------------------------------------------------------------------------------


@SCANS
def test_result_pickle_round_trip(ctx: _Context, scan: ScanId) -> None:
    worker = ctx.load()
    result = worker.run_scan_worker(scan, ctx.settings, ctx.dataset)
    assert pickle.loads(pickle.dumps(result)) == result
