"""ScanId: identity of one processable scan (subject, session, acquisition)."""

from __future__ import annotations

import functools
from dataclasses import dataclass


@functools.total_ordering
@dataclass(frozen=True)
class ScanId:
    """Identity of a single processable scan.

    ``subject`` is the top-level subject folder name; ``session`` and
    ``acquisition`` are BIDS labels with their prefix (``ses-01``, ``acq-T1w``),
    or None when absent. A field is set when it is not None. Scans sort by
    subject, then session, then acquisition, a missing label before any label.
    """

    subject: str
    session: str | None = None
    acquisition: str | None = None

    @property
    def path_prefix(self) -> str:
        """Relative folder of the scan: ``subject/session`` when a session is set, else ``subject``."""
        return f"{self.subject}/{self.session}" if self.session is not None else self.subject

    def __str__(self) -> str:
        parts = [self.subject]
        if self.session is not None:
            parts.append(self.session)
        if self.acquisition is not None:
            parts.append(self.acquisition)
        return "_".join(parts)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ScanId):
            return NotImplemented
        return self._sort_key() < other._sort_key()

    def _sort_key(self) -> tuple[str, bool, str, bool, str]:
        """Comparison key in which a missing session or acquisition sorts before any label."""
        return (
            self.subject,
            self.session is not None,
            self.session or "",
            self.acquisition is not None,
            self.acquisition or "",
        )
