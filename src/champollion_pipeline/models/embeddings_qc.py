"""QC of the embeddings, stages 4 and 5: crop rows of a run's region models and rows of its embeddings CSV files."""

import csv
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from os.path import dirname, exists, join

import yaml

from champollion_pipeline.models.cache import write_json_atomic
from champollion_pipeline.utils.qc_report import (
    ID_COLUMN,
    QC_COLUMN,
    REASON_COLUMN,
    REASON_SEPARATOR,
    fetch_crop_qc_rows,
)

INPUT_QC_KINDS = ("skeleton",)  # evaluate.py reads no other crop kind
EMBEDDINGS_FILENAME = "full_embeddings.csv"
REGION_COPY_SUFFIX = "_embeddings.csv"
EMBEDDINGS_ID_COLUMN = "ID"
MODEL_CONFIG_PARTS = (".hydra", "config.yaml")
MODEL_WIDTH_KEY = "backbone_output_size"
REASON_MISSING_EMBEDDINGS = "missing_embeddings"
REASON_UNREADABLE_EMBEDDINGS = "unreadable_embeddings"
REASON_UNKNOWN_WIDTH = "unknown_width"
REASON_MISSING_ROW = "missing_row"
REASON_DUPLICATE_ROW = "duplicate_row"
REASON_UNEXPECTED_ROW = "unexpected_row"
REASON_WRONG_WIDTH = "wrong_width"
REASON_NON_NUMERIC = "non_numeric"
REASON_NON_FINITE = "non_finite"
WHOLE_FILE_REASONS = (REASON_MISSING_EMBEDDINGS, REASON_UNREADABLE_EMBEDDINGS, REASON_UNKNOWN_WIDTH)
OUTPUT_QC_FILENAME = "output_qc.tsv"
WIDTH_SIDECAR_FILENAME = "embedding_width.json"
WIDTH_SIDECAR_KEY = "width"


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


def compute_region_copy_path(output_path: str, region: str) -> str:
    """The combined copy of ``region``: <output_path>/<region>_embeddings.csv."""
    return join(output_path, f"{region}{REGION_COPY_SUFFIX}")


def compute_excluded_subjects(rows: Iterable[Mapping[str, object]]) -> frozenset[str]:
    """The participant_ids of the qc 0 rows."""
    return frozenset(row[ID_COLUMN] for row in rows if row[QC_COLUMN] == 0)


def _compute_width(document: object, key: str) -> int | None:
    """The positive integer under ``key`` of ``document``, else None: true and 3.5 are no width."""
    value = document.get(key) if isinstance(document, dict) else None
    return value if type(value) is int and value > 0 else None


def read_model_width(model_path: str) -> int | None:
    """The positive integer backbone_output_size of the .hydra/config.yaml of a region model, else None."""
    try:
        with open(join(model_path, *MODEL_CONFIG_PARTS), encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, ValueError, yaml.YAMLError):
        return None
    return _compute_width(config, MODEL_WIDTH_KEY)


def write_region_widths(output_base: str, region_by_model: Mapping[str, str]) -> None:
    """Write <output_base>/<region>/embedding_width.json from each region model; no usable config writes nothing."""
    for model_path, region in region_by_model.items():
        width = read_model_width(model_path)
        if width is not None:
            os.makedirs(join(output_base, region), exist_ok=True)
            write_json_atomic(join(output_base, region, WIDTH_SIDECAR_FILENAME), {WIDTH_SIDECAR_KEY: width})


def read_region_width(region_dir: str) -> int | None:
    """The positive integer width of the embedding_width.json of ``region_dir``, else None."""
    try:
        with open(join(region_dir, WIDTH_SIDECAR_FILENAME), encoding="utf-8") as handle:
            sidecar = json.load(handle)
    except (OSError, ValueError):
        return None
    return _compute_width(sidecar, WIDTH_SIDECAR_KEY)


def _read_embeddings_table(csv_path: str) -> list[list[str]] | None:
    """The rows of the file, header first, blank lines left out; None when the file reads as no table."""
    try:
        with open(csv_path, encoding="utf-8", newline="") as handle:
            table = [row for row in csv.reader(handle) if row]
    except (OSError, ValueError, csv.Error):
        return None
    return table if table and EMBEDDINGS_ID_COLUMN in table[0] else None


def read_header_width(csv_path: str) -> int | None:
    """The count of header cells after the ID cell of the embeddings CSV, None when it reads as no table."""
    table = _read_embeddings_table(csv_path)
    return None if table is None else len(table[0]) - table[0].index(EMBEDDINGS_ID_COLUMN) - 1


def _read_embeddings_rows(csv_path: str) -> list[tuple[str, list[str]]] | None:
    """The (ID cell, cells after it) of each data row, blank lines left out; None when the file reads as no table."""
    table = _read_embeddings_table(csv_path)
    if table is None:
        return None
    id_index = table[0].index(EMBEDDINGS_ID_COLUMN)
    rows = []
    for row in table[1:]:
        cells = row[id_index:]
        rows.append((cells[0] if cells else "", cells[1:]))
    return rows


def _compute_cell_reasons(cells: Sequence[str], region: str, width: int | None) -> list[str]:
    """Reasons of the value cells of one row: its count against ``width``, then each cell; an empty cell is NaN."""
    reasons = []
    if width is not None and len(cells) != width:
        reasons.append(f"{REASON_WRONG_WIDTH}:{region}")
    for cell in cells:
        try:
            is_finite = bool(cell) and math.isfinite(float(cell))
        except ValueError:
            reasons.append(f"{REASON_NON_NUMERIC}:{region}")
            continue
        if not is_finite:
            reasons.append(f"{REASON_NON_FINITE}:{region}")
    return reasons


def fetch_embeddings_csv_reasons(
    csv_path: str, region: str, expected_subjects: Iterable[str], width: int | None
) -> dict[str, list[str]]:
    """Output QC reasons by subject over the embeddings CSV of ``region``, ``width`` value cells a row, None unknown.

    Distinct reasons in the order found. A subject with no reason has no key; a key outside ``expected_subjects`` is
    an ID of the file. A fault of the whole file reaches each key, and a file that does not read gives no missing_row.
    """
    expected = set(expected_subjects)
    file_reasons = [f"{REASON_UNKNOWN_WIDTH}:{region}"] if width is None else []
    found: dict[str, list[str]] = {}
    rows = None
    if not exists(csv_path):
        file_reasons.append(f"{REASON_MISSING_EMBEDDINGS}:{region}")
    else:
        rows = _read_embeddings_rows(csv_path)
        if rows is None:
            file_reasons.append(f"{REASON_UNREADABLE_EMBEDDINGS}:{region}")
    if rows is not None:
        for subject, cells in rows:
            if not subject:
                file_reasons.append(f"{REASON_UNREADABLE_EMBEDDINGS}:{region}")
                continue
            reasons = _compute_cell_reasons(cells, region, width)
            if subject in found:
                reasons.append(f"{REASON_DUPLICATE_ROW}:{region}")
            if subject not in expected:
                reasons.append(f"{REASON_UNEXPECTED_ROW}:{region}")
            found.setdefault(subject, []).extend(reasons)
        for subject in expected - found.keys():
            found[subject] = [f"{REASON_MISSING_ROW}:{region}"]
    held = {subject: found.get(subject, []) + file_reasons for subject in sorted(expected | found.keys())}
    return {subject: list(dict.fromkeys(reasons)) for subject, reasons in held.items() if reasons}


def fetch_embeddings_ids(source: str, regions: Iterable[str]) -> set[str]:
    """The non-empty IDs of the region files of ``regions``; a file that reads as no table gives none."""
    subjects: set[str] = set()
    for region in regions:
        rows = _read_embeddings_rows(join(source, region, EMBEDDINGS_FILENAME))
        subjects.update(subject for subject, _ in rows or [] if subject)
    return subjects


def fetch_embeddings_qc_rows(
    output_base: str, region_by_model: Mapping[str, str], input_rows: Iterable[Mapping[str, object]]
) -> list[dict[str, object]]:
    """One output QC row per qc 1 subject of ``input_rows`` and per unexpected ID, sorted, over each region model.

    ``region_by_model`` gives the region of each model folder. A reason of a whole file reaches every row.
    """
    expected = {row[ID_COLUMN] for row in input_rows if row[QC_COLUMN] == 1}
    widths = [(region, read_model_width(model_path)) for model_path, region in region_by_model.items()]
    return fetch_region_files_qc_rows(output_base, widths, expected)


def fetch_region_files_qc_rows(
    output_base: str, region_widths: Iterable[tuple[str, int | None]], expected_subjects: Iterable[str]
) -> list[dict[str, object]]:
    """One QC row per expected subject and per unexpected ID, sorted, over the (region, width) pairs.

    A reason of a whole file reaches every row.
    """
    files = [(region, join(output_base, region, EMBEDDINGS_FILENAME), width) for region, width in region_widths]
    return _fetch_csv_files_qc_rows(files, expected_subjects)


def fetch_region_copies_qc_rows(
    output_path: str, region_widths: Iterable[tuple[str, int | None]], input_rows: Iterable[Mapping[str, object]]
) -> list[dict[str, object]]:
    """One output QC row per qc 1 subject of ``input_rows`` and per unexpected ID, sorted, over the combined copies.

    A reason of a whole file reaches every row.
    """
    expected = {row[ID_COLUMN] for row in input_rows if row[QC_COLUMN] == 1}
    files = [(region, compute_region_copy_path(output_path, region), width) for region, width in region_widths]
    return _fetch_csv_files_qc_rows(files, expected)


def _fetch_csv_files_qc_rows(
    region_files: Iterable[tuple[str, str, int | None]], expected_subjects: Iterable[str]
) -> list[dict[str, object]]:
    """One QC row per expected subject and per unexpected ID, sorted, over the (region, CSV path, width) triples."""
    expected = set(expected_subjects)
    reasons: dict[str, dict[str, None]] = {subject: {} for subject in expected}
    for region, csv_path, width in region_files:
        found = fetch_embeddings_csv_reasons(csv_path, region, expected, width)
        for subject, subject_reasons in found.items():
            reasons.setdefault(subject, {}).update(dict.fromkeys(subject_reasons))
    whole_file = dict.fromkeys(
        reason for held in reasons.values() for reason in held if reason.split(":", 1)[0] in WHOLE_FILE_REASONS
    )
    rows: list[dict[str, object]] = []
    for subject, held in sorted(reasons.items()):
        cell = REASON_SEPARATOR.join({**held, **whole_file})
        rows.append({ID_COLUMN: subject, QC_COLUMN: 0 if cell else 1, REASON_COLUMN: cell})
    return rows


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
