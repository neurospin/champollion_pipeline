#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScanId: minimal unit of work for the Champollion pipeline.

In non-BIDS databases a subject has exactly one scan, so only ``subject``
is set.  In BIDS databases a subject may have multiple sessions and/or
acquisitions, so ``session`` and ``acquisition`` carry that extra context.

The ``path_prefix`` property gives the relative path from *subjects_dir*
to the root of this scan's files:

    non-BIDS  →  ``"sub-001"``
    BIDS      →  ``"sub-001/ses-01"``

``acquisition`` is stored as metadata (e.g. for worker routing and
reporting) but is *not* part of ``path_prefix``: it is encoded in file
paths themselves (``acq-T1w/`` directories) and matched via glob patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, order=True)
class ScanId:
    """Identity of a single processable scan.

    Attributes
    ----------
    subject:
        Top-level subject directory name (e.g. ``"sub-001"``).
    session:
        BIDS session label including prefix (e.g. ``"ses-01"``).
        ``None`` for non-BIDS databases.
    acquisition:
        BIDS acquisition label including prefix (e.g. ``"acq-T1w"``).
        ``None`` when absent.
    """

    subject: str
    session: Optional[str] = None
    acquisition: Optional[str] = None

    @property
    def path_prefix(self) -> str:
        """Relative path prefix under *subjects_dir* for this scan.

        Used as the B-tree range-query prefix in eligibility checks and
        as the file-watching scope in the parallel subject worker.
        """
        if self.session:
            return f"{self.subject}/{self.session}"
        return self.subject

    def __str__(self) -> str:
        parts = [self.subject]
        if self.session:
            parts.append(self.session)
        if self.acquisition:
            parts.append(self.acquisition)
        return "_".join(parts)
