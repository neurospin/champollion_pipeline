#!/usr/bin/env python3
"""
Tests for ``champollion_pipeline.streaming.barrier``.

The arrival barrier serves the single resident inference consumer: it polls a
SubjectEligibilityChecker and yields each scan as soon as its per-region inputs
(one skeleton per region today) are all present, or raises TimeoutError when
nothing arrives for ``timeout`` seconds. ``scan_embeddings_exist`` is the resume
check: a scan whose embeddings are already in every region's full_embeddings.csv
under the derivatives layout is skipped.

Layout: Morphologist's own. A scan is ``ScanId(subject, acquisition)``, its folder
``<subject>/t1mri/<acquisition>`` (``path_prefix``), the acquisition folder name kept
verbatim. The producer writes one scan's tiles to
``compute_scan_tiles_dir(ct_out, scan)`` = ``<ct_out>/scans/<path_prefix>``, so the
checker the consumer passes has ``subjects_dir = <ct_out>/scans`` and patterns
relative to the acquisition folder. ``str(scan)`` is only the CSV ID, never a path.

Clock contract: the barrier reads ``time.monotonic()`` and waits with
``time.sleep()`` through the ``time`` module bound in the barrier module, so the
tests replace ``barrier.time`` with a fake clock and never sleep for real.

Synthetic folders and CSV files only (tmp_path); no real subject data.
"""

import builtins
import csv
import re
from pathlib import Path

import pytest
from champollion_utils.logger import LOG

from champollion_pipeline.derivatives_layout import compute_region_embeddings_dir
from champollion_pipeline.file_indexer import ScanId, SubjectEligibilityChecker
from champollion_pipeline.streaming import barrier
from champollion_pipeline.streaming.stages import SCAN_TILES_DIRNAME, compute_scan_tiles_dir
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER

pytestmark = pytest.mark.unit  # noqa: V107

ACQ = "default_acquisition"
REGIONS = ("SC-sylv_left", "OCCIPITAL_right")
MASKS = "canonical_25"
CROPS = f"{DERIVATIVES_FOLDER}/crops/{MASKS}/2mm"
# Relative to the scan's acquisition folder <subject>/t1mri/<acquisition>/.
PATTERNS = [f"{CROPS}/{region}/mask/skeleton.npy" for region in REGIONS]
BARRIER_LOGGER = "champollion_pipeline.streaming.barrier"


class FakeClock:
    """Stand-in for the ``time`` module: sleep advances monotonic and may run an arrival."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []
        self.on_sleep: dict[int, object] = {}

    def monotonic(self) -> float:  # noqa: V105 - called by the barrier as time.monotonic
        return self.now

    def sleep(self, seconds: float) -> None:  # noqa: V105 - called by the barrier as time.sleep
        self.sleeps.append(seconds)
        if len(self.sleeps) > 1000:
            raise AssertionError("barrier kept polling: more than 1000 waits")
        self.now += seconds
        action = self.on_sleep.get(len(self.sleeps))
        if action is not None:
            action()


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(barrier, "time", fake, raising=False)
    return fake


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _arrive(ct_out: Path, scan: ScanId, regions=REGIONS) -> None:
    """Create one scan's per-region skeletons where the producer writes them."""
    tiles = Path(compute_scan_tiles_dir(str(ct_out), scan))
    for region in regions:
        _touch(tiles / CROPS / region / "mask" / "skeleton.npy")


def _checker(ct_out: Path) -> SubjectEligibilityChecker:
    """The checker the consumer passes: subjects_dir is <ct_out>/scans."""
    return SubjectEligibilityChecker(str(ct_out / SCAN_TILES_DIRNAME), PATTERNS)


def _collect(gen, clock: FakeClock) -> list[tuple[ScanId, float]]:
    """Consume the barrier, recording the fake time at which each scan was yielded."""
    return [(scan, clock.now) for scan in gen]


# ---------------------------------------------------------------------------
# Arrival
# ---------------------------------------------------------------------------


def test_iter_ready_scans_yields_each_scan_once_when_patterns_resolve(tmp_path, clock):
    """REQ-STREAMING-BDRABCZUK-B3096C406A8A: a scan is yielded once, at the first poll where every pattern resolves."""
    first, second = ScanId("sub-01", ACQ), ScanId("sub-02", ACQ)
    _arrive(tmp_path, first)
    _arrive(tmp_path, second, regions=REGIONS[:1])  # one region still missing
    clock.on_sleep[1] = lambda: _arrive(tmp_path, second, regions=REGIONS[1:])
    start = clock.now

    got = _collect(barrier.iter_ready_scans(_checker(tmp_path), [first, second], 10, 600), clock)

    assert got == [(first, start), (second, start + 10)]


def test_iter_ready_scans_yields_duplicate_scan_once(tmp_path, clock):
    """REQ-STREAMING-BDRABCZUK-B3096C406A8A: a scan given twice is still yielded once."""
    first, second = ScanId("sub-01", ACQ), ScanId("sub-02", ACQ)
    _arrive(tmp_path, first)
    _arrive(tmp_path, second)

    got = list(barrier.iter_ready_scans(_checker(tmp_path), [first, second, first], 10, 600))

    assert got == [first, second]


def test_iter_ready_scans_finds_scans_in_morphologist_tiles_layout(tmp_path, clock):
    """REQ-STREAMING-BDRABCZUK-B3096C406A8A: inputs under <ct_out>/scans/<subject>/t1mri/<acq>/ make a scan ready.

    The files are written by literal path, not through compute_scan_tiles_dir, so the test pins the
    Morphologist layout itself: acquisition folder kept verbatim, no sub/ses split, no str(scan) folder.
    """
    scans = [ScanId("sub-01", "ses-1_run-1", bids=True), ScanId("sub-01", "ses-2", bids=True), ScanId("02", ACQ)]
    for literal in ("sub-01/t1mri/ses-1_run-1", "sub-01/t1mri/ses-2", f"02/t1mri/{ACQ}"):
        for pattern in PATTERNS:
            _touch(tmp_path / "scans" / literal / pattern)

    got = list(barrier.iter_ready_scans(_checker(tmp_path), scans, 10, 600))

    assert got == sorted(scans)
    assert clock.sleeps == []


def test_iter_ready_scans_keeps_acquisitions_of_one_subject_apart(tmp_path, clock):
    """REQ-STREAMING-BDRABCZUK-B3096C406A8A: another acquisition's files never make a scan of the same subject ready."""
    done, waiting = ScanId("sub-01", "ses-1"), ScanId("sub-01", "ses-2")
    _arrive(tmp_path, done)
    _arrive(tmp_path, waiting, regions=REGIONS[:1])
    gen = barrier.iter_ready_scans(_checker(tmp_path), [done, waiting], 10, 30)

    assert next(gen) == done
    with pytest.raises(TimeoutError):
        next(gen)


def test_iter_ready_scans_sees_files_created_between_polls(tmp_path, clock):
    """REQ-STREAMING-BDRABCZUK-8AAA069E57B8: each poll judges the files present when it runs, not a stale index."""
    early, late = ScanId("sub-early", ACQ), ScanId("sub-late", ACQ)
    checker = _checker(tmp_path)
    assert not checker.scan_has_pattern(early, PATTERNS[0])  # builds the index before the files exist
    _arrive(tmp_path, early)
    clock.on_sleep[2] = lambda: _arrive(tmp_path, late)
    start = clock.now

    got = _collect(barrier.iter_ready_scans(checker, [early, late], 5, 600), clock)

    assert got == [(early, start), (late, start + 10)]


def test_iter_ready_scans_orders_scans_ready_in_one_poll(tmp_path, clock):
    """REQ-STREAMING-BDRABCZUK-1BD012B30FA8: scans ready in the same poll come out in ScanId sort order."""
    scans = [ScanId("sub-02", ACQ), ScanId("sub-01", "ses-2"), ScanId("sub-01", "ses-1_run-1"), ScanId("sub-01", ACQ)]
    for scan in scans:
        _arrive(tmp_path, scan)

    got = list(barrier.iter_ready_scans(_checker(tmp_path), scans, 10, 600))

    assert got == sorted(scans)


# ---------------------------------------------------------------------------
# Waiting and stopping
# ---------------------------------------------------------------------------


def test_iter_ready_scans_waits_poll_interval_between_polls(tmp_path, clock):
    """REQ-STREAMING-BDRABCZUK-AAA6997D52CE: the barrier waits poll_interval seconds between two polls."""
    scan = ScanId("sub-01", ACQ)
    clock.on_sleep[3] = lambda: _arrive(tmp_path, scan)

    got = list(barrier.iter_ready_scans(_checker(tmp_path), [scan], 7.5, 600))

    assert got == [scan]
    assert clock.sleeps == [7.5, 7.5, 7.5]


def test_iter_ready_scans_stops_without_waiting_after_last_scan(tmp_path, clock):
    """REQ-STREAMING-BDRABCZUK-7AE59DE77F97: once no scan is pending, iteration ends with no further wait."""
    scans = [ScanId("sub-01", ACQ), ScanId("sub-02", ACQ)]
    for scan in scans:
        _arrive(tmp_path, scan)

    got = list(barrier.iter_ready_scans(_checker(tmp_path), scans, 10, 600))

    assert got == scans
    assert clock.sleeps == []


def test_iter_ready_scans_empty_scans_stops_without_waiting(tmp_path, clock):
    """REQ-STREAMING-BDRABCZUK-7AE59DE77F97: with no scan given, iteration ends at once."""
    got = list(barrier.iter_ready_scans(_checker(tmp_path), [], 10, 600))

    assert got == []
    assert clock.sleeps == []


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


def test_iter_ready_scans_raises_timeout_naming_pending_scans(tmp_path, clock):
    """REQ-STREAMING-BDRABCZUK-F6CF31A0E975: TimeoutError names the timeout and each pending scan's path_prefix.

    Two acquisitions of one non-BIDS subject share str(scan) ("sub-41"); only path_prefix tells them apart.
    """
    ready = ScanId("sub-ready", ACQ)
    pending = [ScanId("sub-41", ACQ), ScanId("sub-41", "ses-2"), ScanId("sub-42", "ses-1_run-1", bids=True)]
    _arrive(tmp_path, ready)
    gen = barrier.iter_ready_scans(_checker(tmp_path), [ready, *pending], 10, 30)
    start = clock.now

    assert next(gen) == ready
    with pytest.raises(TimeoutError) as excinfo:
        next(gen)

    message = str(excinfo.value)
    assert re.search(r"(?<!\d)30(?!\d)", message)
    for scan in pending:
        assert scan.path_prefix in message
    assert 30 <= clock.now - start <= 40


def test_iter_ready_scans_timeout_restarts_after_arrival(tmp_path, clock):
    """REQ-STREAMING-BDRABCZUK-F6CF31A0E975: the timeout counts from the last poll that found a ready scan."""
    first, second = ScanId("sub-01", ACQ), ScanId("sub-02", ACQ)
    clock.on_sleep[2] = lambda: _arrive(tmp_path, first)  # found by the poll at +20 s
    clock.on_sleep[5] = lambda: _arrive(tmp_path, second)  # found at +50 s: 50 s after start, 30 s after first

    got = list(barrier.iter_ready_scans(_checker(tmp_path), [first, second], 10, 30))

    assert got == [first, second]


def test_iter_ready_scans_slow_consumer_does_not_time_out_ready_scan(tmp_path, clock):
    """REQ-STREAMING-BDRABCZUK-F6CF31A0E975: a scan ready at the next poll is yielded however long the consumer took."""
    first, second = ScanId("sub-01", ACQ), ScanId("sub-02", ACQ)
    _arrive(tmp_path, first)
    gen = barrier.iter_ready_scans(_checker(tmp_path), [first, second], 10, 30)

    assert next(gen) == first
    clock.now += 100  # the consumer spends 100 s embedding the first scan
    _arrive(tmp_path, second)

    assert next(gen) == second


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_iter_ready_scans_logs_pending_count_at_log_level_before_wait(tmp_path, clock, caplog):
    """REQ-STREAMING-BDRABCZUK-B96DB772EBB2: one LOG-level record with the pending count before each wait."""
    caplog.set_level(LOG, logger=BARRIER_LOGGER)
    scans = [ScanId("sub-01", ACQ), ScanId("sub-02", ACQ), ScanId("sub-03", ACQ)]

    def arrive_all() -> None:
        for scan in scans:
            _arrive(tmp_path, scan)

    clock.on_sleep[2] = arrive_all

    got = list(barrier.iter_ready_scans(_checker(tmp_path), scans, 10, 600))

    assert got == scans
    records = [r for r in caplog.records if r.name == BARRIER_LOGGER and r.levelno == LOG]
    assert len(records) == len(clock.sleeps) == 2
    for record in records:
        assert re.search(r"(?<!\d)3(?!\d)", record.getMessage())


# ---------------------------------------------------------------------------
# Argument checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("poll_interval", "timeout", "name"),
    [(0, 30, "poll_interval"), (-1, 30, "poll_interval"), (10, 0, "timeout"), (10, -5, "timeout")],
)
def test_iter_ready_scans_rejects_non_positive_durations(  # noqa: PLR0913 - pytest fixtures and parameters
    tmp_path, clock, monkeypatch, poll_interval, timeout, name
):
    """REQ-STREAMING-BDRABCZUK-9D7062EFCFE4: ValueError naming the argument at call time, before any poll."""
    checker = _checker(tmp_path)
    polls: list[object] = []
    monkeypatch.setattr(checker, "scan_has_pattern", lambda *args: polls.append(args) or True)

    with pytest.raises(ValueError, match=name):
        barrier.iter_ready_scans(checker, [ScanId("sub-01", ACQ)], poll_interval, timeout)

    assert polls == []
    assert clock.sleeps == []


@pytest.mark.parametrize(
    ("scans", "bad"),
    [
        (["sub-01/t1mri/default_acquisition"], "sub-01/t1mri/default_acquisition"),  # a path string
        ([ScanId("sub-01", ACQ), "sub-02"], "sub-02"),  # a CSV ID string among scans
        ([("sub-03", ACQ)], "sub-03"),  # a bare (subject, acquisition) tuple
    ],
    ids=["path-str", "id-str-mixed", "tuple"],
)
def test_iter_ready_scans_rejects_non_scanid_scan(tmp_path, clock, monkeypatch, scans, bad):
    """REQ-STREAMING-BDRABCZUK-61E3036C11D1: a scan that is not a ScanId is a TypeError naming it, at call time."""
    checker = _checker(tmp_path)
    polls: list[object] = []
    monkeypatch.setattr(checker, "refresh", lambda: polls.append("refresh"))
    monkeypatch.setattr(checker, "scan_has_pattern", lambda *args: polls.append(args) or True)

    with pytest.raises(TypeError, match=re.escape(bad)):
        barrier.iter_ready_scans(checker, scans, 10, 30)

    assert polls == []
    assert clock.sleeps == []


# ---------------------------------------------------------------------------
# Resume check
# ---------------------------------------------------------------------------

RESUME_SCANS = [
    ScanId("sub-01", ACQ),
    ScanId("sub-01", "ses-1_run-1", bids=True),
    ScanId("sub-01", "ses-V1_acq-T1w", bids=True),
]


def _write_region_csv(datasets_root: Path, region: str, header: list[str], ids: list[str], masks=MASKS) -> None:
    path = Path(compute_region_embeddings_dir(str(datasets_root), masks)) / region / "full_embeddings.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for scan_id in ids:
            writer.writerow([scan_id, *([0.5] * (len(header) - 1))])


@pytest.mark.parametrize("scan", RESUME_SCANS, ids=str)
def test_scan_embeddings_exist_true_when_each_region_has_scan_row(tmp_path, scan):
    """REQ-STREAMING-BDRABCZUK-5801750AD11C: True when every given region's CSV holds the str(scan) ID row."""
    for region in REGIONS:
        _write_region_csv(tmp_path, region, ["ID", "dim1", "dim2"], ["sub-00", str(scan)])

    assert barrier.scan_embeddings_exist(scan, str(tmp_path), MASKS, list(REGIONS)) is True


SCAN = ScanId("sub-01", ACQ)


def _missing_row(root: Path) -> ScanId:
    _write_region_csv(root, REGIONS[0], ["ID", "dim1"], [str(SCAN)])
    _write_region_csv(root, REGIONS[1], ["ID", "dim1"], ["sub-02"])
    return SCAN


def _missing_file(root: Path) -> ScanId:
    _write_region_csv(root, REGIONS[0], ["ID", "dim1"], [str(SCAN)])
    return SCAN


def _no_id_column(root: Path) -> ScanId:
    for region in REGIONS:
        _write_region_csv(root, region, ["Subject", "dim1"], [str(SCAN)])
    return SCAN


def _longer_id_only(root: Path) -> ScanId:
    for region in REGIONS:
        _write_region_csv(root, region, ["ID", "dim1"], [f"{SCAN}_ses-1"])
    return SCAN


def _other_masks_version(root: Path) -> ScanId:
    for region in REGIONS:
        _write_region_csv(root, region, ["ID", "dim1"], [str(SCAN)], masks="canonical_corrected_26_1")
    return SCAN


def _path_prefix_row_only(root: Path) -> ScanId:
    """A row keyed by the scan folder is not the scan's ID: paths never stand for IDs."""
    for region in REGIONS:
        _write_region_csv(root, region, ["ID", "dim1"], [SCAN.path_prefix])
    return SCAN


def _subject_row_only_for_bids_scan(root: Path) -> ScanId:
    """A BIDS scan's ID carries its entities: the subject-only row belongs to another scan."""
    scan = ScanId("sub-01", "ses-1_run-1", bids=True)
    for region in REGIONS:
        _write_region_csv(root, region, ["ID", "dim1"], [scan.subject])
    return scan


@pytest.mark.parametrize(
    "layout",
    [
        _missing_row,
        _missing_file,
        _no_id_column,
        _longer_id_only,
        _other_masks_version,
        _path_prefix_row_only,
        _subject_row_only_for_bids_scan,
    ],
)
def test_scan_embeddings_exist_false_when_a_region_lacks_the_scan(tmp_path, layout):
    """REQ-STREAMING-BDRABCZUK-5801750AD11C: False unless every region's CSV has an ID cell equal to str(scan)."""
    scan = layout(tmp_path)

    assert barrier.scan_embeddings_exist(scan, str(tmp_path), MASKS, list(REGIONS)) is False


def test_scan_embeddings_exist_rejects_empty_regions(tmp_path):
    """REQ-STREAMING-BDRABCZUK-AD62DCB0F6AB: an empty region list is a ValueError, never a vacuous True."""
    with pytest.raises(ValueError):
        barrier.scan_embeddings_exist(SCAN, str(tmp_path), MASKS, [])


def test_scan_embeddings_exist_rejects_str_regions(tmp_path):
    """REQ-STREAMING-BDRABCZUK-F984B641C238: a single string is not a region list."""
    with pytest.raises(TypeError):
        barrier.scan_embeddings_exist(SCAN, str(tmp_path), MASKS, "SC-sylv_left")


@pytest.mark.parametrize("scan_id", [str(SCAN), SCAN.path_prefix, (SCAN.subject, SCAN.acquisition)], ids=repr)
def test_scan_embeddings_exist_rejects_non_scanid_scan_id(tmp_path, monkeypatch, scan_id):
    """REQ-STREAMING-BDRABCZUK-1D87FD897A82: a scan_id that is not a ScanId is a TypeError, before any CSV is opened.

    Every region's CSV holds both the ID and the folder string, so a lenient check would answer True.
    """
    for region in REGIONS:
        _write_region_csv(tmp_path, region, ["ID", "dim1"], [str(SCAN), SCAN.path_prefix])
    opened: list[object] = []

    def spy_open(*args, **kwargs):
        opened.append(args)
        return builtins.open(*args, **kwargs)

    monkeypatch.setattr(barrier, "open", spy_open, raising=False)

    with pytest.raises(TypeError):
        barrier.scan_embeddings_exist(scan_id, str(tmp_path), MASKS, list(REGIONS))

    assert opened == []
