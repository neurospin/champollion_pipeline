"""ScanId: identity of one processable scan in the Morphologist layout (subject, acquisition folder)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Mirrors cortical_tiles (read-only submodule)
# external/cortical_tiles/cortical_tiles/brainvisa/generate_skeletons.py,
# GraphConvert2Skeleton.get_skeleton_filename: with --bids the sequential subject ID is the subject
# followed by '_' + the first match of each pattern, in this fixed order, a missing entity skipped.
# Inlined, not imported: upstream it is an inline method body that needs aims.
_BIDS_ENTITY_PATTERNS: tuple[str, ...] = ("ses-([^_/]+)", "acq-([^_/]+)", "run-([^_/]+)")


@dataclass(frozen=True, order=True)
class ScanId:
    """Identity of a single processable scan, in Morphologist's layout.

    Morphologist writes a scan's graphs under ``<subject>/t1mri/<acquisition>/``. ``subject`` is the
    top-level subject folder name and ``acquisition`` the acquisition folder name, kept verbatim and
    never split (``default_acquisition``, ``ses-V1``, ``ses-1_run-1``). Identity, hash and ordering are
    ``(subject, acquisition)``; ``bids`` (the subjects follow --bids naming) is not part of identity.
    ``str()`` is the sequential pipeline's subject ID: the subject alone, or with ``bids`` the subject
    followed by the ses-, acq- and run- entities found in ``path_prefix``, in that order.
    """

    subject: str
    acquisition: str
    bids: bool = field(default=False, compare=False)

    @property
    def path_prefix(self) -> str:
        """Relative folder of the scan: ``<subject>/t1mri/<acquisition>``."""
        return f"{self.subject}/t1mri/{self.acquisition}"

    def __str__(self) -> str:
        if not self.bids:
            return self.subject
        parts = [self.subject]
        for pattern in _BIDS_ENTITY_PATTERNS:
            match = re.search(pattern, self.path_prefix)
            if match:
                parts.append(match[0])
        return "_".join(parts)
