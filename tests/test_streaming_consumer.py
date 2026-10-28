"""Tests for the resident inference consumer of a streaming run.

Task TASK-STREAMING-BDRABCZUK-4DA7B0A0F45E (revises -A4213E337F78, -E4994A199B26, -AC3602321C6E): ONE resident
process embeds scans through ONE EmbeddingEngine as their per-scan crops arrive from the producer workers, appends
one row per scan and region to the region full_embeddings.csv files of the derivatives layout, skips scans already
embedded (resume), never raises on a bad scan, and returns after the producers signal completion (drain) or after
consumer_stall_timeout.

Readiness (user decisions 2026-10-27 and 2026-10-28): a scan is ready at a poll that finds every region input AND
the producer's completion marker (worker.TILES_COMPLETE_MARKER, never restated here) in the scan's tiles
directory, holding the run's masks_version. A marker for another masks version, or no marker, keeps the scan
pending; a scan still pending when the consumer returns is FAILED.

Nothing real runs: EmbeddingEngine and init_pipeline_process are replaced, in the consumer module's namespace,
by recording stand-ins, and the clock (the ``time`` module bound in the consumer and barrier modules) by a fake
clock that never sleeps. The consumer module is loaded inside each test body (_Context.load) so a missing
implementation fails each test on its own instead of breaking collection.

Layout: Morphologist's own. The producer writes a scan's tiles to
compute_scan_tiles_dir(cortical_tiles_output, scan) = <ct_out>/scans/<subject>/t1mri/<acquisition>; the 2mm
crops are find_scan_crops_root(that, masks_version), and the completion marker sits in the tiles directory itself.
str(scan) is only the CSV ID, never a path. Crop folder names keep their dots (S.C.-sylv.) while model region
names drop them (SC-sylv_left), as on real data; GenerateEmbeddings._find_crop_dir maps one to the other.

Interface the tests expect from champollion_pipeline.streaming.consumer:
- run_consumer(settings, dataset, scans, producers_done, models_path) -> list[ConsumerResult]
  scans: iterable of ScanId; producers_done: any object with is_set() (threading.Event in most tests, an
  object with is_set() alone in one, a multiprocessing Event or Manager().Event() proxy in the orchestrator);
  models_path: str, already fetched by the orchestrator.
- ConsumerResult(scan, status), ConsumerStatus.OK / SKIPPED / FAILED
- module-level names EmbeddingEngine (called as EmbeddingEngine(models_path, device, epoch=...)),
  init_pipeline_process, and the time module (when the consumer reads the clock itself)
- logger champollion_pipeline.streaming.consumer; log file <log_dir>/consumer.log

Requirements (id, test):
- REQ-STREAMING-BDRABCZUK-4A0FCB1CB2D6 test_one_engine_per_call
- REQ-STREAMING-BDRABCZUK-E012ED952C62 test_waits_for_each_region_input
- REQ-STREAMING-BDRABCZUK-C96D1FBE24B7 test_embeds_from_scan_crops_directory
- REQ-STREAMING-BDRABCZUK-64AC11AA70D2 test_waits_for_matching_completion_marker
- REQ-STREAMING-BDRABCZUK-03880F27E2B9 (supersedes -21183ACE8FCA) test_embeds_scan_at_first_poll_after_arrival,
  test_stale_marker_scan_embeds_once_marker_matches
- REQ-STREAMING-BDRABCZUK-4A6E172F4B98 test_waits_poll_interval_between_polls
- REQ-STREAMING-BDRABCZUK-A92F4CC567DE test_writes_row_with_scan_id_in_derivatives_layout
- REQ-STREAMING-BDRABCZUK-DB32F52053D0 test_keeps_existing_rows
- REQ-STREAMING-BDRABCZUK-6CCBB1C29A5C test_adds_no_duplicate_row
- REQ-STREAMING-BDRABCZUK-1448BC07E094 test_skips_scan_with_existing_embeddings
- REQ-STREAMING-BDRABCZUK-42D1F894D6D5 test_dry_run_skips_without_engine
- REQ-STREAMING-BDRABCZUK-0538E58D0C2D test_engine_failure_gives_failed
- REQ-STREAMING-BDRABCZUK-B848ECB1F603 (supersedes -9A6094BDF49D) test_continues_after_failed_scan
- REQ-STREAMING-BDRABCZUK-0E039BA46724 test_failed_scan_writes_no_row
- REQ-STREAMING-BDRABCZUK-58D1819AC1CE test_engine_row_count_mismatch_gives_failed
- REQ-STREAMING-BDRABCZUK-4D9B869E3DB0 test_failed_scan_logs_error
- REQ-STREAMING-BDRABCZUK-51DD7B61E000 test_drain_returns_after_empty_poll
- REQ-STREAMING-BDRABCZUK-7C513D3E0C5F (supersedes -63BBE35C5D7F) test_drain_embeds_scans_present_at_final_poll
- REQ-STREAMING-BDRABCZUK-D189ED33FA3A (supersedes -F7882C2754EA) test_never_ready_scan_gives_failed
- REQ-STREAMING-BDRABCZUK-33B027B6D72D test_stall_timeout_returns
- REQ-STREAMING-BDRABCZUK-F47E1BEED988 test_results_one_per_scan_sorted
- REQ-STREAMING-BDRABCZUK-31164732A3E6 test_result_pickle_round_trip
- REQ-STREAMING-BDRABCZUK-547C4F8CAD7F test_init_pipeline_process_before_engine
- REQ-STREAMING-BDRABCZUK-89B822EF95C7 test_consumer_log_file_captures_output
- REQ-STREAMING-BDRABCZUK-24BA08EBA83E test_unusable_log_file_gives_failed
- REQ-STREAMING-BDRABCZUK-ABF0853B734A test_progress_log_record
- REQ-STREAMING-BDRABCZUK-6DF8EC34031F test_result_output_record
- REQ-STREAMING-BDRABCZUK-9ADB35E0B215 test_nan_embedding_logs_warning
- REQ-STREAMING-BDRABCZUK-A00FD7BEEEB7 test_nan_embedding_still_ok
- REQ-STREAMING-BDRABCZUK-0149FFF67C6E test_non_scanid_raises_type_error
- REQ-STREAMING-BDRABCZUK-20B2FB89461D test_empty_layout_field_raises_value_error
- REQ-STREAMING-BDRABCZUK-0D40BB7D3B62 test_engine_without_regions_raises_value_error
- REQ-STREAMING-BDRABCZUK-F24FDA16887F test_producers_done_needs_only_is_set
"""

from __future__ import annotations

import csv
import dataclasses
import importlib
import json
import logging
import os
import pickle
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from champollion_pipeline.config import DatasetConfig
from champollion_pipeline.derivatives_layout import compute_region_embeddings_dir
from champollion_pipeline.file_indexer import ScanId
from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from champollion_pipeline.streaming import barrier, stages, worker
from champollion_pipeline.streaming.settings import StreamingSettings
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER

pytestmark = pytest.mark.unit  # noqa: V107

CONSUMER_MODULE = "champollion_pipeline.streaming.consumer"
LOG_LEVEL = 22
OUTPUT_LEVEL = 25
MASKS = "synthetic_masks"
OTHER_MASKS = "other_masks"
MODELS_PATH = "/synthetic/models"
EPOCH = 7
POLL = 10.0
STALL = 1000.0
STDOUT_MARKER = "engine-stdout-marker-5b2e"
STDERR_MARKER = "engine-stderr-marker-c81f"

# Model region name -> crop folder name (dots kept on disk, dropped in model names).
CROP_DIRS = {"SC-sylv_left": "S.C.-sylv.", "OCCIPITAL_right": "OCCIPITAL"}
REGIONS = tuple(CROP_DIRS)

# One scan per Morphologist flavour: classic (str() is the subject alone) and BIDS (str() is sub-01_ses-1_run-1,
# which must never become a path); OTHER sorts after both.
CLASSIC_SCAN = ScanId("subj01", "default_acquisition")
BIDS_SCAN = ScanId("sub-01", "ses-1_run-1", bids=True)
OTHER_SCAN = ScanId("subj02", "default_acquisition")
SCAN_INDEX = {CLASSIC_SCAN: 0, BIDS_SCAN: 1, OTHER_SCAN: 2}
SCANS = pytest.mark.parametrize("scan", [CLASSIC_SCAN, BIDS_SCAN], ids=["classic", "bids"])


def _value(scan: ScanId, region: str) -> float:
    """The synthetic skeleton value of one scan and region; the fake engine embeds it as dim1."""
    return float(1 + 10 * SCAN_INDEX[scan] + REGIONS.index(region))


# --- fakes -------------------------------------------------------------------------------------------


class FakeClock:
    """Stand-in for the ``time`` module: sleep advances monotonic and may run an action."""

    def __init__(self) -> None:
        self.start = 1000.0
        self.now = self.start
        self.sleeps: list[float] = []
        self.on_sleep: dict[int, Callable[[], None]] = {}

    def monotonic(self) -> float:  # noqa: V105 - called as time.monotonic
        return self.now

    def time(self) -> float:  # noqa: V105 - called as time.time
        return self.now

    def perf_counter(self) -> float:  # noqa: V105 - called as time.perf_counter
        return self.now

    def sleep(self, seconds: float) -> None:  # noqa: V105 - called as time.sleep
        self.sleeps.append(seconds)
        if len(self.sleeps) > 1000:
            raise AssertionError("consumer kept polling: more than 1000 waits")
        self.now += seconds
        action = self.on_sleep.get(len(self.sleeps))
        if action is not None:
            action()

    @property
    def elapsed(self) -> float:
        return self.now - self.start


class FakeEngineState:
    """What the fake engines of one test record, and how they misbehave."""

    def __init__(self, clock: FakeClock, events: list[str]) -> None:
        self.clock = clock
        self.events = events
        self.regions: list[str] = list(REGIONS)
        self.instances: list[dict] = []
        self.calls: list[tuple[str, float, int]] = []  # (region, skeleton value, waits so far)
        self.errors: dict[float, BaseException] = {}
        self.nan_values: set[float] = set()
        self.row_counts: dict[float, int] = {}
        self.write_fds = False
        self.batch_sizes: list[int] = []

    def table(self, region: str, skeletons: np.ndarray, ids: list[str]) -> pd.DataFrame:
        rows = []
        for i in range(skeletons.shape[0]):
            value = float(skeletons[i].flat[0])
            self.calls.append((region, value, len(self.clock.sleeps)))
            if self.write_fds:
                os.write(1, f"{STDOUT_MARKER}\n".encode())
                os.write(2, f"{STDERR_MARKER}\n".encode())
            if value in self.errors:
                raise self.errors[value]
            dim1 = float("nan") if value in self.nan_values else value
            rows.extend([[ids[i], dim1, value + 0.5]] * self.row_counts.get(value, 1))
        return pd.DataFrame(rows, columns=["ID", "dim1", "dim2"])


def _fake_engine_class(state: FakeEngineState) -> type:
    class FakeEngine:
        """Stand-in for EmbeddingEngine: same constructor, regions, embed and embed_crops."""

        def __init__(self, models_path: str, device: object, *, epoch: int = 0, batch_size: int = 32) -> None:
            state.events.append("engine")
            state.instances.append({"models_path": models_path, "device": device, "epoch": epoch})
            state.batch_sizes.append(batch_size)

        @property
        def regions(self) -> list[str]:
            return list(state.regions)

        def embed(self, region: str, skeletons: np.ndarray, scan_ids: list[str]) -> pd.DataFrame:
            return state.table(region, np.asarray(skeletons), list(scan_ids))

        def embed_crops(self, region: str, crops_2mm: str) -> pd.DataFrame:
            base = region.rsplit("_", 1)[0]
            crop = next(d for d in os.listdir(crops_2mm) if d.replace(".", "") == base)
            side = "L" if region.endswith("_left") else "R"
            mask = os.path.join(crops_2mm, crop, "mask")
            skeletons = np.load(os.path.join(mask, f"{side}skeleton.npy"))
            subjects = pd.read_csv(os.path.join(mask, f"{side}skeleton_subject.csv"), dtype=str)["Subject"]
            return state.table(region, skeletons, list(subjects))

    return FakeEngine


@dataclasses.dataclass
class _Context:
    """One test's fake engine, clock, settings, dataset, producers_done event and consumer-logger records."""

    monkeypatch: pytest.MonkeyPatch
    clock: FakeClock
    engine: FakeEngineState
    events: list[str]
    settings: StreamingSettings
    dataset: DatasetConfig
    done: threading.Event
    records: list[logging.LogRecord]

    def load(self) -> ModuleType:
        """Import the consumer module and replace its engine, process setup and clock with the fakes."""
        consumer = importlib.import_module(CONSUMER_MODULE)
        self.monkeypatch.setattr(consumer, "EmbeddingEngine", _fake_engine_class(self.engine), raising=False)
        self.monkeypatch.setattr(consumer, "init_pipeline_process", self._init, raising=False)
        self.monkeypatch.setattr(consumer, "time", self.clock, raising=False)
        self.monkeypatch.setattr(barrier, "time", self.clock, raising=False)
        return consumer

    def _init(self) -> logging.Logger:
        self.events.append("init")
        return logging.getLogger()

    def run(
        self, scans: Iterable[object], settings: StreamingSettings | None = None, producers_done: object = None
    ) -> list:
        consumer = self.load()
        done = self.done if producers_done is None else producers_done
        return consumer.run_consumer(settings or self.settings, self.dataset, list(scans), done, MODELS_PATH)

    def status(self, results: list, scan: ScanId) -> str:
        matches = [r.status for r in results if r.scan == scan]
        assert len(matches) == 1, results
        return matches[0].value

    def messages(self, level: int) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno == level]

    def embedded_values(self) -> set[float]:
        return {value for _, value, _ in self.engine.calls}

    # --- synthetic data ---

    def crops_dir(self, scan: ScanId, masks: str = MASKS) -> Path:
        tiles = stages.compute_scan_tiles_dir(self.dataset.cortical_tiles_output, scan)
        return Path(tiles, DERIVATIVES_FOLDER, "crops", masks, "2mm")

    def write_marker(self, scan: ScanId, text: str | None = None, masks: str = MASKS) -> None:
        """Write the scan's completion marker as the producer worker does (masks_version JSON), or the given text."""
        tiles = Path(stages.compute_scan_tiles_dir(self.dataset.cortical_tiles_output, scan))
        tiles.mkdir(parents=True, exist_ok=True)
        content = json.dumps({"masks_version": masks}) if text is None else text
        (tiles / worker.TILES_COMPLETE_MARKER).write_text(content, encoding="utf-8")

    def write_inputs(
        self,
        scan: ScanId,
        regions: Iterable[str] = REGIONS,
        *,
        npy: bool = True,
        subjects: bool = True,
        marker: str | None = MASKS,
    ) -> None:
        """Write the scan's per-region skeleton .npy and skeleton_subject.csv as one cortical_tiles run would,
        then (unless marker is None) the completion marker for that masks version, as the producer worker would."""
        for region in regions:
            side = "L" if region.endswith("_left") else "R"
            mask = self.crops_dir(scan) / CROP_DIRS[region] / "mask"
            mask.mkdir(parents=True, exist_ok=True)
            if npy:
                np.save(mask / f"{side}skeleton.npy", np.full((1, 4, 4, 4, 1), _value(scan, region)))
            if subjects:
                (mask / f"{side}skeleton_subject.csv").write_text(f"Subject\n{scan.subject}\n")
        if marker is not None:
            self.write_marker(scan, masks=marker)

    def embeddings_csv(self, region: str) -> Path:
        base = compute_region_embeddings_dir(self.dataset.datasets_root, self.dataset.masks_version)
        return Path(base, region, "full_embeddings.csv")

    def write_rows(self, region: str, rows: list[tuple[str, float, float]]) -> None:
        path = self.embeddings_csv(region)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "dim1", "dim2"])
            writer.writerows(rows)

    def read_rows(self, region: str) -> pd.DataFrame:
        path = self.embeddings_csv(region)
        if not path.is_file():
            return pd.DataFrame(columns=["ID", "dim1", "dim2"])
        return pd.read_csv(path, dtype={"ID": str})


@pytest.fixture
def ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Context:
    """Fresh context; the consumer module logger's records are captured on the logger itself."""
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    consumer_logger = logging.getLogger(CONSUMER_MODULE)
    handler = _ListHandler(level=1)
    level = consumer_logger.level
    consumer_logger.addHandler(handler)
    consumer_logger.setLevel(1)
    clock = FakeClock()
    events: list[str] = []
    settings = StreamingSettings(
        subjects_dir=str(tmp_path / "subjects"),
        output_dir=str(tmp_path / "out"),
        log_dir=str(tmp_path / "logs"),
        poll_interval=POLL,
        worker_timeout=STALL,
        consumer_stall_timeout=STALL,
        device="cpu",
    )
    dataset = DatasetConfig(
        name="synthetic",
        datasets_root=str(tmp_path / "dataset"),
        cortical_tiles_output=str(tmp_path / "ct_out"),
        masks_version=MASKS,
        checkpoint_epoch=EPOCH,
    )
    yield _Context(
        monkeypatch, clock, FakeEngineState(clock, events), events, settings, dataset, threading.Event(), records
    )
    consumer_logger.removeHandler(handler)
    consumer_logger.setLevel(level)


# --- the engine ------------------------------------------------------------------------------------------


def test_one_engine_per_call(ctx: _Context) -> None:
    ctx.write_inputs(CLASSIC_SCAN)
    ctx.write_inputs(OTHER_SCAN)
    ctx.done.set()
    ctx.run([CLASSIC_SCAN, OTHER_SCAN])
    assert ctx.engine.instances == [{"models_path": MODELS_PATH, "device": "cpu", "epoch": EPOCH}]


def test_engine_without_regions_raises_value_error(ctx: _Context) -> None:
    ctx.engine.regions = []
    ctx.write_inputs(CLASSIC_SCAN)
    ctx.done.set()
    with pytest.raises(ValueError, match=MODELS_PATH):
        ctx.run([CLASSIC_SCAN])
    assert ctx.clock.sleeps == []
    assert ctx.engine.calls == []


def test_init_pipeline_process_before_engine(ctx: _Context) -> None:
    ctx.write_inputs(CLASSIC_SCAN)
    ctx.done.set()
    ctx.run([CLASSIC_SCAN])
    assert ctx.events == ["init", "engine"]


# --- arrival -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["region", "skeleton_npy", "subject_csv"])
def test_waits_for_each_region_input(ctx: _Context, missing: str) -> None:
    ctx.write_inputs(CLASSIC_SCAN, REGIONS[:1])
    if missing == "skeleton_npy":
        ctx.write_inputs(CLASSIC_SCAN, REGIONS[1:], npy=False)
    elif missing == "subject_csv":
        ctx.write_inputs(CLASSIC_SCAN, REGIONS[1:], subjects=False)
    ctx.write_inputs(OTHER_SCAN)
    ctx.done.set()
    ctx.run([CLASSIC_SCAN, OTHER_SCAN])
    classic = {_value(CLASSIC_SCAN, region) for region in REGIONS}
    assert not ctx.embedded_values() & classic, ctx.engine.calls
    assert {_value(OTHER_SCAN, region) for region in REGIONS} <= ctx.embedded_values()


@SCANS
def test_embeds_from_scan_crops_directory(ctx: _Context, scan: ScanId) -> None:
    ctx.write_inputs(scan)
    decoy = ctx.crops_dir(scan, masks="other_masks") / CROP_DIRS[REGIONS[0]] / "mask"
    decoy.mkdir(parents=True)
    np.save(decoy / "Lskeleton.npy", np.full((1, 4, 4, 4, 1), 999.0))
    (decoy / "Lskeleton_subject.csv").write_text(f"Subject\n{scan.subject}\n")
    for region in REGIONS:  # the fixture's crop folders resolve the way the sequential embeddings stage does
        assert GenerateEmbeddings._find_crop_dir(None, str(ctx.crops_dir(scan)), region) == CROP_DIRS[region]
    ctx.done.set()
    ctx.run([scan])
    tiles = stages.compute_scan_tiles_dir(ctx.dataset.cortical_tiles_output, scan)
    assert stages.find_scan_crops_root(tiles, MASKS) == str(ctx.crops_dir(scan))
    assert sorted((region, value) for region, value, _ in ctx.engine.calls) == sorted(
        (region, _value(scan, region)) for region in REGIONS
    )


def test_embeds_scan_at_first_poll_after_arrival(ctx: _Context) -> None:
    ctx.clock.on_sleep[2] = lambda: ctx.write_inputs(CLASSIC_SCAN)
    ctx.clock.on_sleep[3] = ctx.done.set
    results = ctx.run([CLASSIC_SCAN])
    waits = {waits for _, _, waits in ctx.engine.calls}
    assert waits == {2}, ctx.engine.calls
    assert ctx.status(results, CLASSIC_SCAN) == "ok"


@pytest.mark.parametrize("marker", ["missing", "other_masks", "not_json"])
def test_waits_for_matching_completion_marker(ctx: _Context, marker: str) -> None:
    ctx.write_inputs(CLASSIC_SCAN, marker=None)
    if marker == "other_masks":
        ctx.write_marker(CLASSIC_SCAN, masks=OTHER_MASKS)
    elif marker == "not_json":
        ctx.write_marker(CLASSIC_SCAN, text="{not json")
    ctx.write_inputs(OTHER_SCAN)
    ctx.done.set()
    ctx.run([CLASSIC_SCAN, OTHER_SCAN])
    assert not ctx.embedded_values() & {_value(CLASSIC_SCAN, region) for region in REGIONS}, ctx.engine.calls
    assert {_value(OTHER_SCAN, region) for region in REGIONS} <= ctx.embedded_values()


def test_stale_marker_scan_embeds_once_marker_matches(ctx: _Context) -> None:
    ctx.write_inputs(CLASSIC_SCAN, marker=OTHER_MASKS)
    ctx.clock.on_sleep[2] = lambda: ctx.write_marker(CLASSIC_SCAN)
    ctx.clock.on_sleep[3] = ctx.done.set
    results = ctx.run([CLASSIC_SCAN])
    waits = {waits for _, _, waits in ctx.engine.calls}
    assert waits == {2}, ctx.engine.calls
    assert ctx.status(results, CLASSIC_SCAN) == "ok"


def test_waits_poll_interval_between_polls(ctx: _Context) -> None:
    ctx.clock.on_sleep[3] = lambda: ctx.write_inputs(CLASSIC_SCAN)
    ctx.clock.on_sleep[4] = ctx.done.set
    ctx.run([CLASSIC_SCAN, OTHER_SCAN])
    assert len(ctx.clock.sleeps) >= 3
    assert set(ctx.clock.sleeps) == {POLL}


# --- output rows ----------------------------------------------------------------------------------------------


@SCANS
def test_writes_row_with_scan_id_in_derivatives_layout(ctx: _Context, scan: ScanId) -> None:
    ctx.write_inputs(scan)
    ctx.done.set()
    ctx.run([scan])
    for region in REGIONS:
        rows = ctx.read_rows(region)
        assert list(rows.columns) == ["ID", "dim1", "dim2"]
        assert rows["ID"].tolist() == [str(scan)]
        assert rows["dim1"].tolist() == [_value(scan, region)]
        assert rows["dim2"].tolist() == [_value(scan, region) + 0.5]
    if str(scan) != scan.subject:
        assert not any(str(scan) in name for _, dirs, _ in os.walk(ctx.dataset.datasets_root) for name in dirs)


def test_keeps_existing_rows(ctx: _Context) -> None:
    for region in REGIONS:
        ctx.write_rows(region, [("subjX", 3.25, 4.5)])
    ctx.write_inputs(CLASSIC_SCAN)
    ctx.done.set()
    ctx.run([CLASSIC_SCAN])
    for region in REGIONS:
        rows = ctx.read_rows(region)
        assert rows.iloc[0].tolist() == ["subjX", 3.25, 4.5]
        assert rows["ID"].tolist() == ["subjX", str(CLASSIC_SCAN)]


def test_adds_no_duplicate_row(ctx: _Context) -> None:
    first = REGIONS[0]
    ctx.write_rows(first, [(str(CLASSIC_SCAN), _value(CLASSIC_SCAN, first), _value(CLASSIC_SCAN, first) + 0.5)])
    ctx.write_inputs(CLASSIC_SCAN)
    ctx.done.set()
    ctx.run([CLASSIC_SCAN])
    for region in REGIONS:
        assert ctx.read_rows(region)["ID"].tolist() == [str(CLASSIC_SCAN)], region


# --- skips ----------------------------------------------------------------------------------------------------


def test_skips_scan_with_existing_embeddings(ctx: _Context) -> None:
    for region in REGIONS:
        ctx.write_rows(region, [(str(CLASSIC_SCAN), 1.0, 2.0)])
    ctx.write_inputs(OTHER_SCAN)
    ctx.done.set()
    results = ctx.run([CLASSIC_SCAN, OTHER_SCAN])
    assert ctx.status(results, CLASSIC_SCAN) == "skipped"
    assert not ctx.embedded_values() & {_value(CLASSIC_SCAN, region) for region in REGIONS}


def test_dry_run_skips_without_engine(ctx: _Context) -> None:
    ctx.write_inputs(CLASSIC_SCAN)
    results = ctx.run([CLASSIC_SCAN, OTHER_SCAN], dataclasses.replace(ctx.settings, dry_run=True))
    assert [r.status.value for r in results] == ["skipped", "skipped"]
    assert ctx.engine.instances == []
    assert not ctx.embeddings_csv(REGIONS[0]).exists()


# --- failures never raise ----------------------------------------------------------------------------------


@pytest.mark.parametrize("error", [RuntimeError("engine failed"), SystemExit(2)], ids=["exception", "exit"])
def test_engine_failure_gives_failed(ctx: _Context, error: BaseException) -> None:
    ctx.engine.errors[_value(CLASSIC_SCAN, REGIONS[0])] = error
    ctx.write_inputs(CLASSIC_SCAN)
    ctx.done.set()
    results = ctx.run([CLASSIC_SCAN])
    assert ctx.status(results, CLASSIC_SCAN) == "failed"


def test_continues_after_failed_scan(ctx: _Context) -> None:
    ctx.engine.errors[_value(CLASSIC_SCAN, REGIONS[0])] = RuntimeError("engine failed")
    ctx.write_inputs(CLASSIC_SCAN)
    ctx.write_inputs(OTHER_SCAN)
    ctx.done.set()
    results = ctx.run([CLASSIC_SCAN, OTHER_SCAN])
    assert ctx.status(results, OTHER_SCAN) == "ok"
    for region in REGIONS:
        assert str(OTHER_SCAN) in ctx.read_rows(region)["ID"].tolist()


def test_failed_scan_writes_no_row(ctx: _Context) -> None:
    ctx.engine.errors[_value(CLASSIC_SCAN, REGIONS[1])] = RuntimeError("second region failed")
    ctx.write_inputs(CLASSIC_SCAN)
    ctx.done.set()
    ctx.run([CLASSIC_SCAN])
    for region in REGIONS:
        assert str(CLASSIC_SCAN) not in ctx.read_rows(region)["ID"].tolist(), region


@pytest.mark.parametrize("rows", [0, 2])
def test_engine_row_count_mismatch_gives_failed(ctx: _Context, rows: int) -> None:
    ctx.engine.row_counts[_value(CLASSIC_SCAN, REGIONS[0])] = rows
    ctx.write_inputs(CLASSIC_SCAN)
    ctx.done.set()
    results = ctx.run([CLASSIC_SCAN])
    assert ctx.status(results, CLASSIC_SCAN) == "failed"


@SCANS
@pytest.mark.parametrize("reason", ["engine_error", "never_ready"])
def test_failed_scan_logs_error(ctx: _Context, scan: ScanId, reason: str) -> None:
    if reason == "engine_error":
        ctx.engine.errors[_value(scan, REGIONS[0])] = RuntimeError("engine failed")
        ctx.write_inputs(scan)
    ctx.done.set()
    ctx.run([scan])
    errors = ctx.messages(logging.ERROR)
    assert any(scan.path_prefix in text for text in errors), errors


def test_unusable_log_file_gives_failed(ctx: _Context, tmp_path: Path) -> None:
    blocker = tmp_path / "log_dir_is_a_file"
    blocker.write_text("not a directory")
    ctx.write_inputs(CLASSIC_SCAN)
    ctx.done.set()
    results = ctx.run([CLASSIC_SCAN, OTHER_SCAN], dataclasses.replace(ctx.settings, log_dir=str(blocker)))
    assert [r.status.value for r in results] == ["failed", "failed"]


# --- drain and stall -----------------------------------------------------------------------------------------


def test_drain_returns_after_empty_poll(ctx: _Context) -> None:
    ctx.clock.on_sleep[2] = ctx.done.set
    ctx.run([CLASSIC_SCAN])
    assert len(ctx.clock.sleeps) == 2, ctx.clock.sleeps


def test_drain_embeds_scans_present_at_final_poll(ctx: _Context) -> None:
    def arrive_then_done() -> None:
        ctx.write_inputs(CLASSIC_SCAN)
        ctx.done.set()

    ctx.clock.on_sleep[2] = arrive_then_done
    results = ctx.run([CLASSIC_SCAN])
    assert ctx.status(results, CLASSIC_SCAN) == "ok"
    for region in REGIONS:
        assert ctx.read_rows(region)["ID"].tolist() == [str(CLASSIC_SCAN)]


@pytest.mark.parametrize("missing", ["inputs", "marker", "marker_version"])
@pytest.mark.parametrize("ending", ["drain", "stall"])
def test_never_ready_scan_gives_failed(ctx: _Context, ending: str, missing: str) -> None:
    ctx.write_inputs(CLASSIC_SCAN)
    if missing == "marker":
        ctx.write_inputs(OTHER_SCAN, marker=None)
    elif missing == "marker_version":
        ctx.write_inputs(OTHER_SCAN, marker=OTHER_MASKS)
    if ending == "drain":
        ctx.done.set()
    results = ctx.run([CLASSIC_SCAN, OTHER_SCAN])
    assert ctx.status(results, CLASSIC_SCAN) == "ok"
    assert ctx.status(results, OTHER_SCAN) == "failed"


class _IsSetOnly:
    """A producers_done with is_set() and no other attribute (no wait, no set, not an Event)."""

    __slots__ = ("_event",)

    def __init__(self, event: threading.Event) -> None:
        self._event = event

    def is_set(self) -> bool:  # noqa: V105 - called by run_consumer
        return self._event.is_set()


def test_producers_done_needs_only_is_set(ctx: _Context) -> None:
    def arrive_then_done() -> None:
        ctx.write_inputs(CLASSIC_SCAN)
        ctx.done.set()

    ctx.clock.on_sleep[2] = arrive_then_done
    results = ctx.run([CLASSIC_SCAN, OTHER_SCAN], producers_done=_IsSetOnly(ctx.done))
    assert ctx.status(results, CLASSIC_SCAN) == "ok"
    assert ctx.status(results, OTHER_SCAN) == "failed"
    assert ctx.clock.elapsed < STALL, ctx.clock.elapsed  # drained on is_set(), not stalled out


@pytest.mark.parametrize("arrival_wait", [None, 50], ids=["nothing_arrives", "one_arrival"])
def test_stall_timeout_returns(ctx: _Context, arrival_wait: int | None) -> None:
    progress = 0.0
    if arrival_wait is not None:
        ctx.clock.on_sleep[arrival_wait] = lambda: ctx.write_inputs(CLASSIC_SCAN)
        progress = arrival_wait * POLL
    ctx.run([CLASSIC_SCAN, OTHER_SCAN])
    assert progress + STALL <= ctx.clock.elapsed <= progress + STALL + POLL, ctx.clock.elapsed


# --- results ----------------------------------------------------------------------------------------------------


def test_results_one_per_scan_sorted(ctx: _Context) -> None:
    ctx.write_inputs(CLASSIC_SCAN)
    ctx.done.set()
    results = ctx.run([OTHER_SCAN, CLASSIC_SCAN, OTHER_SCAN])
    assert [r.scan for r in results] == [CLASSIC_SCAN, OTHER_SCAN]


def test_result_pickle_round_trip(ctx: _Context) -> None:
    for region in REGIONS:
        ctx.write_rows(region, [(str(BIDS_SCAN), 1.0, 2.0)])
    ctx.write_inputs(CLASSIC_SCAN)
    ctx.done.set()
    results = ctx.run([CLASSIC_SCAN, BIDS_SCAN, OTHER_SCAN])
    assert {r.status.value for r in results} == {"ok", "skipped", "failed"}
    for result in results:
        assert pickle.loads(pickle.dumps(result)) == result


# --- logging ---------------------------------------------------------------------------------------------------


@SCANS
def test_consumer_log_file_captures_output(ctx: _Context, scan: ScanId) -> None:
    ctx.engine.write_fds = True
    ctx.write_inputs(scan)
    ctx.done.set()
    ctx.run([scan])
    log_file = Path(ctx.settings.log_dir, "consumer.log")
    assert log_file.is_file()
    text = log_file.read_text()
    assert STDOUT_MARKER in text
    assert STDERR_MARKER in text


@SCANS
def test_progress_log_record(ctx: _Context, scan: ScanId) -> None:
    ctx.write_inputs(scan)
    ctx.done.set()
    ctx.run([scan])
    progress = ctx.messages(LOG_LEVEL)
    assert any(scan.path_prefix in text for text in progress), progress


@SCANS
@pytest.mark.parametrize("outcome", ["ok", "resume_skip", "dry_run_skip"])
def test_result_output_record(ctx: _Context, scan: ScanId, outcome: str) -> None:
    settings = ctx.settings
    if outcome == "resume_skip":
        for region in REGIONS:
            ctx.write_rows(region, [(str(scan), 1.0, 2.0)])
    elif outcome == "dry_run_skip":
        settings = dataclasses.replace(settings, dry_run=True)
    ctx.write_inputs(scan)
    ctx.done.set()
    ctx.run([scan], settings)
    outputs = ctx.messages(OUTPUT_LEVEL)
    assert any(scan.path_prefix in text for text in outputs), outputs


@SCANS
def test_nan_embedding_logs_warning(ctx: _Context, scan: ScanId) -> None:
    region = REGIONS[1]
    ctx.engine.nan_values.add(_value(scan, region))
    ctx.write_inputs(scan)
    ctx.done.set()
    ctx.run([scan])
    warnings = ctx.messages(logging.WARNING)
    assert any(scan.path_prefix in text and region in text for text in warnings), warnings


def test_nan_embedding_still_ok(ctx: _Context) -> None:
    ctx.engine.nan_values.add(_value(CLASSIC_SCAN, REGIONS[0]))
    ctx.write_inputs(CLASSIC_SCAN)
    ctx.done.set()
    results = ctx.run([CLASSIC_SCAN])
    assert ctx.status(results, CLASSIC_SCAN) == "ok"


# --- input checks --------------------------------------------------------------------------------------------


def test_non_scanid_raises_type_error(ctx: _Context) -> None:
    ctx.done.set()
    with pytest.raises(TypeError, match="subj02"):
        ctx.run([CLASSIC_SCAN, "subj02"])
    assert ctx.engine.instances == []


@pytest.mark.parametrize("field", ["datasets_root", "cortical_tiles_output"])
def test_empty_layout_field_raises_value_error(ctx: _Context, field: str) -> None:
    ctx.dataset = dataclasses.replace(ctx.dataset, **{field: ""})
    ctx.done.set()
    with pytest.raises(ValueError, match=field):
        ctx.run([CLASSIC_SCAN])
    assert ctx.engine.instances == []
