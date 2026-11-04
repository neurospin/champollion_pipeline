"""Stage 6 output QC: the snapshot files one run expects, checked on the disk after the snapshot steps.

Each step registers its files before its render; ``compute_rows`` then gives one row per
expected file for ``output_qc.tsv``. Nothing here logs: reporting sits in ``generate_snapshots``.
"""

import os.path as osp

OUTPUT_QC_FILENAME = "output_qc.tsv"
MISSING_SNAPSHOT = "missing_snapshot"
EMPTY_SNAPSHOT = "empty_snapshot"


def _find_gap(path: str) -> str:
    """Return the reason the snapshot file ``path`` is unusable, empty text when it holds one or more bytes."""
    if not osp.isfile(path):
        return MISSING_SNAPSHOT
    return EMPTY_SNAPSHOT if osp.getsize(path) == 0 else ""


class SnapshotsOutputQC:
    """Expected snapshot files of one run, each file once, in expectation order."""

    def __init__(self) -> None:
        self._paths: dict[str, str] = {}

    def expect(self, path: str) -> None:
        """Register the snapshot file ``path``; a file name already registered is a no-op."""
        self._paths.setdefault(osp.basename(path), path)

    def compute_rows(self) -> list[dict[str, object]]:
        """Return one {check, qc, reason} row per expected file, from the disk at call time."""
        rows = []
        for check, path in self._paths.items():
            reason = _find_gap(path)
            rows.append({"check": check, "qc": 0 if reason else 1, "reason": reason})
        return rows
