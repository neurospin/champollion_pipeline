"""Transform QC rows of the crop stage.

TEMPORARY (DEF-CORTILES-BDRABCZUK-8A44DAD7F312): goes with the prefix-ID guard of run_cortical_tiles.py.
"""

import os
from os.path import isdir, join

TRANSFORM_SIDES = ("L", "R")
TRANSFORM_STEM = "transform_to_ICBM2009c_"  # files are <side><TRANSFORM_STEM><ID>.trm


def transform_qc_rows(transforms_dir: str) -> list[dict[str, object]]:
    """QC rows of the ICBM2009c transforms in <transforms_dir>/<side>/, one per subject, sorted by ID.

    A subject fails (qc 0) when its transform file on a side has exactly the same bytes as another
    subject's on that side; L and R are never compared with each other. Pure: no logging, no write.
    """
    flagged: dict[str, list[str]] = {}
    subjects: set[str] = set()
    for side in TRANSFORM_SIDES:
        side_dir = join(transforms_dir, side)
        if not isdir(side_dir):
            continue
        prefix = side + TRANSFORM_STEM
        groups: dict[bytes, list[str]] = {}
        with os.scandir(side_dir) as listing:
            for entry in sorted(listing, key=lambda item: item.name):
                name = entry.name
                if not (entry.is_file() and name.startswith(prefix) and name.endswith(".trm")):
                    continue
                subject = name[len(prefix) : -len(".trm")]
                if not subject:
                    continue
                with open(entry.path, "rb") as handle:
                    groups.setdefault(handle.read(), []).append(subject)
                subjects.add(subject)
        for group in groups.values():
            for subject in group:
                others = sorted(other for other in group if other != subject)
                if others:
                    flagged.setdefault(subject, []).append(f"{side} transform byte-identical to {', '.join(others)}")
    return [
        {"participant_id": subject, "qc": 0 if subject in flagged else 1, "reason": "; ".join(flagged.get(subject, []))}
        for subject in sorted(subjects)
    ]
