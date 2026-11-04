#!/usr/bin/env python3
"""
Shared per-subject QC report: a tab-separated file with the columns
participant_id, qc (1 keep, 0 exclude), optional extra columns, then reason.
A per-run report (write_run_qc_tsv) has one row per check instead: check,
qc, reason.

Every NTUI QC step writes its report through write_qc_tsv and reads one back
through read_qc_tsv. The file stays readable by cortical_tiles --sk_qc_path
and prune_failed_subjects --qc. Standard library only: no toolchain import.
"""

import csv
import numbers
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence

ID_COLUMN = "participant_id"
QC_COLUMN = "qc"
REASON_COLUMN = "reason"
CHECK_COLUMN = "check"
REASON_SEPARATOR = ";"
REASON_MISSING_CROP = "missing_crop"
CROP_KIND_MASK = "mask"
CROP_SIDES = ("L", "R")
CROP_MASK_FOLDER = "mask"  # <crops_dir>/<region>/mask, the cortical_tiles cropping type
CROP_FILE_STEMS = {"foldlabel": "label"}  # kinds whose files carry another stem than the kind name
CROP_SUBJECT_COLUMN = "Subject"
_RESERVED = (ID_COLUMN, QC_COLUMN, REASON_COLUMN)


def _qc_value(value: object) -> str | None:
    """Return "1" or "0" for a valid qc value, else None."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, numbers.Integral):
        return str(int(value)) if value in (0, 1) else None
    if isinstance(value, str) and value in ("0", "1"):
        return value
    return None


def _cell(value: object) -> str:
    """Text written for an extra or reason cell: "" for None."""
    return "" if value is None else str(value)


def _check_extra_columns(extra_columns: Sequence[str]) -> None:
    """Raise ValueError on a reserved or repeated extra column name."""
    seen = set()
    for name in extra_columns:
        if name in _RESERVED:
            raise ValueError(f"extra column {name!r} clashes with a fixed QC column")
        if name in seen:
            raise ValueError(f"extra column {name!r} is repeated")
        seen.add(name)


def _row_cells(index: int, row: Mapping[str, object], seen_ids: set[str]) -> dict[str, object]:
    """Validate row ``index`` and return it with participant_id and qc as written text."""
    subject = row.get(ID_COLUMN)
    if subject is None or str(subject) == "":
        raise ValueError(f"row {index}: missing participant_id")
    subject = str(subject)
    if subject in seen_ids:
        raise ValueError(f"row {index} ({subject}): participant_id repeats an earlier row")
    seen_ids.add(subject)
    if QC_COLUMN not in row:
        raise ValueError(f"row {index} ({subject}): missing qc")
    qc = _qc_value(row[QC_COLUMN])
    if qc is None:
        raise ValueError(f"row {index} ({subject}): qc must be 0 or 1, got {row[QC_COLUMN]!r}")
    return {**row, ID_COLUMN: subject, QC_COLUMN: qc}


def write_qc_tsv(
    path: str | os.PathLike,
    rows: Iterable[Mapping[str, object]],
    extra_columns: Sequence[str] = (),
) -> None:
    """Write one tab-separated row per subject, atomically.

    Every row is validated before the file system is touched; an invalid row
    raises ValueError naming its index and participant_id, and leaves any
    existing file at ``path`` unchanged.
    """
    extra_columns = tuple(extra_columns)
    _check_extra_columns(extra_columns)
    seen_ids: set[str] = set()
    checked = [_row_cells(i, row, seen_ids) for i, row in enumerate(rows)]

    header = [ID_COLUMN, QC_COLUMN, *extra_columns, REASON_COLUMN]
    _write_atomic(
        path,
        header,
        (
            {
                ID_COLUMN: row[ID_COLUMN],
                QC_COLUMN: row[QC_COLUMN],
                **{name: _cell(row.get(name)) for name in (*extra_columns, REASON_COLUMN)},
            }
            for row in checked
        ),
    )


def write_run_qc_tsv(path: str | os.PathLike, rows: Iterable[Mapping[str, object]]) -> None:
    """Write one tab-separated row per check (check, qc, reason), atomically.

    Every row is validated before the file system is touched; a missing or
    repeated check name or a qc other than 0 or 1 raises ValueError naming the
    row index, and leaves any existing file at ``path`` unchanged.
    """
    checked = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        name = row.get(CHECK_COLUMN)
        if name is None or str(name) == "":
            raise ValueError(f"row {index}: missing check")
        name = str(name)
        if name in seen:
            raise ValueError(f"row {index} ({name}): check repeats an earlier row")
        seen.add(name)
        qc = _qc_value(row.get(QC_COLUMN))
        if qc is None:
            raise ValueError(f"row {index} ({name}): qc must be 0 or 1, got {row.get(QC_COLUMN)!r}")
        checked.append({CHECK_COLUMN: name, QC_COLUMN: qc, REASON_COLUMN: _cell(row.get(REASON_COLUMN))})
    _write_atomic(path, [CHECK_COLUMN, QC_COLUMN, REASON_COLUMN], checked)


def _write_atomic(path: str | os.PathLike, header: Sequence[str], rows: Iterable[Mapping[str, str]]) -> None:
    """Write ``rows`` under ``header`` to a temporary file, then move it onto ``path``."""
    directory = os.path.dirname(os.path.abspath(os.fspath(path)))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".qc_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(header), delimiter="\t", lineterminator="\n", extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def read_qc_tsv(path: str | os.PathLike, delimiter: str = "\t") -> list[dict[str, object]]:
    """Return one record per data row of a QC file, tab-separated unless ``delimiter`` says otherwise.

    ``delimiter`` exists for the comma-separated --sk_qc_path files cortical_tiles accepts.
    Values are the cell text, except qc which becomes int 0 or 1. A file
    without a reason column gets ``reason = ""`` on every record. Raises
    ValueError, message starting with the path, on a missing participant_id
    or qc column, a repeated participant_id or a qc cell other than 0 or 1.
    """
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        header = list(reader.fieldnames or [])
        for column in (ID_COLUMN, QC_COLUMN):
            if column not in header:
                raise ValueError(f"{path}: header has no {column} column")
        records = []
        seen_ids = set()
        for line in reader:
            record = {name: line.get(name) or "" for name in header}
            subject = record[ID_COLUMN]
            if subject in seen_ids:
                raise ValueError(f"{path}: participant_id {subject!r} repeats an earlier row")
            seen_ids.add(subject)
            qc = str(record[QC_COLUMN]).strip()
            if qc not in ("0", "1"):
                raise ValueError(f"{path}: qc of {subject!r} must be 0 or 1, got {record[QC_COLUMN]!r}")
            record[QC_COLUMN] = int(qc)
            record.setdefault(REASON_COLUMN, "")
            records.append(record)
    return records


def read_crop_subjects(path: str, is_bids: bool = False) -> set[str]:
    """Non-empty Subject cells of a crop subject CSV, cut to the participant_id with ``is_bids``.

    An absent or unreadable file, or one without a Subject column, gives the empty set.
    """
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            cells = [line.get(CROP_SUBJECT_COLUMN) for line in csv.DictReader(handle)]
    except (OSError, ValueError, csv.Error):
        return set()
    return {cell.partition("_")[0] if is_bids else cell for cell in cells if cell}


def _list_crop_steps(
    crops_dir: str | os.PathLike, regions: Sequence[str], kinds: Sequence[str], is_bids: bool, sides: Sequence[str]
) -> list[tuple[str, bool, set[str] | None]]:
    """(reason, array usable, subjects of the CSV) by region, side, kind; the kind mask has no CSV, so None."""
    steps = []
    for region in regions:
        folder = os.path.join(crops_dir, region, CROP_MASK_FOLDER)
        for side in sides:
            for kind in kinds:
                stem = os.path.join(folder, side + CROP_FILE_STEMS.get(kind, kind))
                array = f"{stem}.npy"
                is_usable = os.path.isfile(array) and os.path.getsize(array) > 0
                listed = None if kind == CROP_KIND_MASK else read_crop_subjects(f"{stem}_subject.csv", is_bids)
                steps.append((f"{REASON_MISSING_CROP}:{kind}:{side}:{region}", is_usable, listed))
    return steps


def fetch_crop_qc_rows(  # noqa: PLR0913 - keyword-only sides, stage 2 positional call kept
    crops_dir: str | os.PathLike,
    regions: Sequence[str],
    kinds: Sequence[str],
    subjects: Iterable[str] | None = None,
    is_bids: bool = False,
    *,
    sides: Sequence[str] = CROP_SIDES,
) -> list[dict[str, object]]:
    """One QC row per subject over the crop files of ``kinds`` in <crops_dir>/<region>/mask, sorted by participant_id.

    The shared crop-completeness check: ``subjects`` None takes the subjects the subject CSVs name, so
    ``kinds`` holding mask alone gives no row. A subject gets a missing_crop reason for an unusable array
    or a CSV not listing it, on the hemisphere ``sides`` only. Read-only; an absent or unreadable file is a
    gap, never an exception.
    """
    steps = _list_crop_steps(crops_dir, regions, kinds, is_bids, sides)
    if subjects is None:
        cohort = set().union(*(listed for _, _, listed in steps if listed is not None))
    else:
        cohort = set(subjects)
    rows = []
    for subject in sorted(cohort):
        reasons = [
            reason
            for reason, is_usable, listed in steps
            if not is_usable or (listed is not None and subject not in listed)
        ]
        rows.append(
            {
                ID_COLUMN: subject,
                QC_COLUMN: 0 if reasons else 1,
                REASON_COLUMN: REASON_SEPARATOR.join(dict.fromkeys(reasons)),
            }
        )
    return rows


def summarise_qc(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Return subject, excluded and excluded-per-reason counts."""
    n_subjects = 0
    n_excluded = 0
    per_reason: dict[str, int] = {}
    for record in records:
        n_subjects += 1
        if int(record[QC_COLUMN]) != 0:
            continue
        n_excluded += 1
        reason = record.get(REASON_COLUMN)
        if reason:
            per_reason[reason] = per_reason.get(reason, 0) + 1
    return {"n_subjects": n_subjects, "n_excluded": n_excluded, "excluded_per_reason": per_reason}
