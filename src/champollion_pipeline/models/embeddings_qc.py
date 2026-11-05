"""Stage 4 QC of the embeddings: crop rows of a run's region models and rows of its embeddings CSV files."""

import csv
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from os.path import dirname, exists

from champollion_pipeline.utils.qc_report import (
    ID_COLUMN,
    QC_COLUMN,
    REASON_COLUMN,
    REASON_SEPARATOR,
    fetch_crop_qc_rows,
)

INPUT_QC_KINDS = ("skeleton",)  # evaluate.py reads no other crop kind
EMBEDDINGS_FILENAME = "full_embeddings.csv"
EMBEDDINGS_ID_COLUMN = "ID"


def compute_region_side(region: str) -> str:
    """Hemisphere side of a region model: L for *_left, else R."""
    return "L" if region.endswith("_left") else "R"


def compute_crops_by_side(crop_by_region: Mapping[str, str]) -> dict[str, list[str]]:
    """Crop folder name of each region model, by the hemisphere side the model uses."""
    crops_by_side: dict[str, list[str]] = {}
    for region, crop in crop_by_region.items():
        crops_by_side.setdefault(compute_region_side(region), []).append(crop)
    return crops_by_side


def fetch_region_model_qc_rows(
    crops_2mm_dir: str, crops_by_side: Mapping[str, Sequence[str]]
) -> list[dict[str, object]]:
    """One QC row per subject, sorted, over the crop folders of each side: qc 0 with the reasons of all its sides.

    The cohort is the union of the sides, so that a side whose files are wholly absent still shows its gap.
    """
    sides = sorted(crops_by_side)
    cohort = {
        row[ID_COLUMN]
        for side in sides
        for row in fetch_crop_qc_rows(crops_2mm_dir, crops_by_side[side], INPUT_QC_KINDS, sides=(side,))
    }
    reasons: dict[str, list[str]] = {subject: [] for subject in cohort}
    for side in sides:
        for row in fetch_crop_qc_rows(
            crops_2mm_dir, crops_by_side[side], INPUT_QC_KINDS, subjects=cohort, sides=(side,)
        ):
            if row[REASON_COLUMN]:
                reasons[row[ID_COLUMN]].append(row[REASON_COLUMN])
    return [
        {ID_COLUMN: subject, QC_COLUMN: 0 if found else 1, REASON_COLUMN: REASON_SEPARATOR.join(found)}
        for subject, found in sorted(reasons.items())
    ]


def compute_excluded_subjects(rows: Iterable[Mapping[str, object]]) -> frozenset[str]:
    """The participant_ids of the qc 0 rows."""
    return frozenset(row[ID_COLUMN] for row in rows if row[QC_COLUMN] == 0)


def remove_subject_rows(csv_path: str, subjects: frozenset[str]) -> int:
    """Drop the rows of ``subjects`` from the embeddings CSV at ``csv_path``, atomically; return the dropped count.

    The header and the kept rows keep their bytes and the file its mode. An empty file, a header with no ID cell
    or a file holding none of ``subjects`` stays untouched and gives 0; a line too short to hold the ID cell is kept.
    """
    with open(csv_path, encoding="utf-8", newline="") as handle:
        lines = handle.readlines()
    header = next(csv.reader(lines[:1]), [])
    if EMBEDDINGS_ID_COLUMN not in header:
        return 0
    id_index = header.index(EMBEDDINGS_ID_COLUMN)
    kept = lines[:1]
    for line in lines[1:]:
        cells = next(csv.reader([line]), [])
        if len(cells) <= id_index or cells[id_index] not in subjects:
            kept.append(line)
    dropped = len(lines) - len(kept)
    if not dropped:
        return 0
    fd, tmp_path = tempfile.mkstemp(dir=dirname(csv_path), prefix=".embeddings_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.writelines(kept)
        shutil.copymode(csv_path, tmp_path)
        os.replace(tmp_path, csv_path)
    except BaseException:
        if exists(tmp_path):
            os.remove(tmp_path)
        raise
    return dropped
