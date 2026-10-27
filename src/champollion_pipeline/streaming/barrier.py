"""Streaming arrival barrier and resume check for the resident inference consumer.

iter_ready_scans polls a SubjectEligibilityChecker and yields scans as their inputs arrive;
scan_embeddings_exist tells whether every region already holds a scan's embedding row.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from collections.abc import Iterable, Iterator, Sequence

from champollion_utils.logger import LOG

from champollion_pipeline.derivatives_layout import compute_region_embeddings_dir
from champollion_pipeline.file_indexer import ScanId, SubjectEligibilityChecker

logger = logging.getLogger(__name__)

_EMBEDDINGS_FILENAME = "full_embeddings.csv"
_ID_COLUMN = "ID"


def iter_ready_scans(
    checker: SubjectEligibilityChecker,
    scans: Iterable[ScanId],
    poll_interval: float,
    timeout: float,
) -> Iterator[ScanId]:
    """Yield each given scan once its required patterns all resolve; raise TimeoutError on a stall.

    The timeout is a stall timeout: it runs from the first poll, or from the last poll that found
    a ready scan. Not thread-safe: each poll refreshes the checker's index.
    """
    for name, value in (("poll_interval", poll_interval), ("timeout", timeout)):
        if not (value > 0):
            raise ValueError(f"{name} must be greater than 0, got {value!r}")
    if isinstance(scans, str):
        raise TypeError(f"scans must be an iterable of ScanId, not a str: {scans!r}")
    items = list(scans)
    for scan in items:
        if not isinstance(scan, ScanId):
            raise TypeError(f"scans must hold only ScanId, got {type(scan).__name__} {scan!r}")
    return _poll(checker, sorted(set(items)), poll_interval, timeout)


def _poll(
    checker: SubjectEligibilityChecker, pending: list[ScanId], poll_interval: float, timeout: float
) -> Iterator[ScanId]:
    last_progress: float | None = None
    while pending:
        now = time.monotonic()
        checker.refresh()
        ready = [s for s in pending if all(checker.scan_has_pattern(s, p) for p in checker.required_patterns)]
        if ready:
            last_progress = now
            pending = [s for s in pending if s not in ready]
            yield from ready
        elif last_progress is None:
            last_progress = now
        elif now - last_progress >= timeout:
            names = ", ".join(s.path_prefix for s in pending)
            raise TimeoutError(f"No scan became ready within timeout={timeout} s; pending scans: {names}")
        if not pending:
            return
        logger.log(LOG, "Barrier: %d scans pending; next poll in %s s", len(pending), poll_interval)
        time.sleep(poll_interval)


def scan_embeddings_exist(
    scan_id: ScanId,
    datasets_root: str,
    masks_version: str,
    regions: Sequence[str],
) -> bool:
    """Return True when each region's full_embeddings.csv holds a row for the scan."""
    if not isinstance(scan_id, ScanId):
        raise TypeError(f"scan_id must be a ScanId, got {type(scan_id).__name__} {scan_id!r}")
    if isinstance(regions, str):
        raise TypeError("regions must be a sequence of region names, not a str")
    if len(regions) == 0:
        raise ValueError("regions must not be empty")
    base = compute_region_embeddings_dir(datasets_root, masks_version)
    target = str(scan_id)
    for region in regions:
        path = os.path.join(base, region, _EMBEDDINGS_FILENAME)
        if not os.path.isfile(path):
            return False
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if _ID_COLUMN not in (reader.fieldnames or []):
                return False
            if not any(row[_ID_COLUMN] == target for row in reader):
                return False
    return True
