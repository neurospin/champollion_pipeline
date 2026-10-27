"""Stage 6 input QC: per-subject checks of the snapshot inputs, run once before any render.

Sulcal subjects are checked for one sulcal graph per hemisphere (and a white
mesh, reported but not excluding); embeddings IDs for a finite row in every
selected CSV. Results go to ``snapshots_input_qc.tsv`` through the shared QC
report writer. Region-level gaps (UMAP reference files, crop masks, region
graphs) are reported as single warning lines and never give a row.
"""

import csv
import math
import os
import os.path as osp
from collections.abc import Sequence

from champollion_pipeline.snapshot_types.sulcal import (
    HEMISPHERES,
    find_sulcal_graphs_by_hemi,
    find_white_mesh,
    list_sulcal_subjects,
)
from champollion_pipeline.snapshot_types.tiles import compute_region_graph_path, find_completed_regions
from champollion_pipeline.snapshot_types.umap import compute_umap_reference_paths
from champollion_pipeline.utils.qc_report import write_qc_tsv

QC_FILENAME = "snapshots_input_qc.tsv"
MISSING_WHITE_MESH_COLUMN = "missing_white_mesh"
REASON_SEPARATOR = ";"
EMBEDDING_ID_COLUMN = "ID"
_MASK_SKELETON_FILES = ("Lmask_skeleton.nii.gz", "Rmask_skeleton.nii.gz")


class SnapshotsInputQC:
    """Per-subject accumulator; one row per registered subject, in registration order."""

    def __init__(self) -> None:
        self._reasons: dict[str, list[str]] = {}
        self._missing_white_mesh: dict[str, list[str]] = {}

    def add_subject(self, subject: str) -> None:
        """Register ``subject`` (qc 1 while it has no reason); repeated calls are no-ops."""
        self._reasons.setdefault(subject, [])

    def add_reason(self, subject: str, reason: str) -> None:
        """Register ``subject`` and append ``reason`` unless it is already listed."""
        reasons = self._reasons.setdefault(subject, [])
        if reason not in reasons:
            reasons.append(reason)

    def set_missing_white_mesh(self, subject: str, hemispheres: Sequence[str]) -> None:
        """Register ``subject`` and store the hemispheres whose white mesh is missing."""
        self.add_subject(subject)
        self._missing_white_mesh[subject] = list(hemispheres)

    def excluded(self) -> list[str]:
        """Return the subjects with at least one reason, in registration order."""
        return [subject for subject, reasons in self._reasons.items() if reasons]

    def write(self, output_dir: str) -> str:
        """Write the QC file into ``output_dir`` (header only when no subject) and return its path."""
        path = osp.join(output_dir, QC_FILENAME)
        extra_columns = (MISSING_WHITE_MESH_COLUMN,) if self._missing_white_mesh else ()
        rows = [
            {
                "participant_id": subject,
                "qc": 0 if reasons else 1,
                "reason": REASON_SEPARATOR.join(reasons),
                MISSING_WHITE_MESH_COLUMN: REASON_SEPARATOR.join(self._missing_white_mesh.get(subject, [])),
            }
            for subject, reasons in self._reasons.items()
        ]
        write_qc_tsv(path, rows, extra_columns)
        return path


def check_sulcal_inputs(report, morphologist_dir, subject=None, acquisition=None) -> None:
    """Record a missing sulcal graph per hemisphere (qc 0) and missing white meshes (column only)."""
    for name in list_sulcal_subjects(morphologist_dir, subject):
        report.add_subject(name)
        by_hemi = find_sulcal_graphs_by_hemi(morphologist_dir, name, acquisition)
        missing_mesh = []
        for hemi in HEMISPHERES:
            if not by_hemi[hemi]:
                report.add_reason(name, f"missing_sulcal_graph_{hemi}")
            elif find_white_mesh(by_hemi[hemi][0]) is None:
                missing_mesh.append(hemi)
        report.set_missing_white_mesh(name, missing_mesh)


def _is_finite(value) -> bool:
    """True when ``value`` is a cell text that parses to a finite float."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _read_embedding_rows(csv_path) -> dict[str, bool] | None:
    """Return ``{ID: every value of its rows is finite}``, or None when the CSV has no ID column."""
    with open(csv_path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if EMBEDDING_ID_COLUMN not in (reader.fieldnames or []):
            return None
        finite: dict[str, bool] = {}
        for row in reader:
            subject = row[EMBEDDING_ID_COLUMN]
            row_finite = all(_is_finite(value) for key, value in row.items() if key != EMBEDDING_ID_COLUMN)
            finite[subject] = finite.get(subject, True) and row_finite
    return finite


def check_embedding_inputs(report, csvs: Sequence[tuple[str, str, str]]) -> None:
    """Record, per ID of the selected CSVs, a missing or non-finite embeddings row (qc 0)."""
    usable = []
    for csv_path, region, hemi in csvs:
        try:
            rows = _read_embedding_rows(csv_path)
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            print_warning(f"snapshots input QC could not use {csv_path}", [f"{type(exc).__name__}: {exc}"])
            continue
        if rows is None:
            print_warning(f"snapshots input QC could not use {csv_path}", [f"no {EMBEDDING_ID_COLUMN} column"])
            continue
        usable.append((rows, region, hemi))

    subjects = list(dict.fromkeys(subject for rows, _, _ in usable for subject in rows))
    for subject in subjects:
        report.add_subject(subject)
        for rows, region, hemi in usable:
            if subject not in rows:
                report.add_reason(subject, f"missing_embedding_{region}_{hemi}")
            elif not rows[subject]:
                report.add_reason(subject, f"nonfinite_embedding_{region}_{hemi}")


def find_missing_umap_references(reference_data_dir, csvs) -> list[str]:
    """Return the basenames of the missing UMAP model and coordinates files, in CSV order."""
    missing = []
    for _, region, hemi in csvs:
        for path in compute_umap_reference_paths(reference_data_dir, region, hemi):
            if not osp.exists(path):
                missing.append(osp.basename(path))
    return missing


def find_crop_regions_without_mask(crops_dir) -> list[str]:
    """Return the region folders of ``crops_dir`` whose ``mask/`` holds no mask skeleton file."""
    return [
        name
        for name in sorted(os.listdir(crops_dir))
        if osp.isdir(osp.join(crops_dir, name))
        and not any(osp.exists(osp.join(crops_dir, name, "mask", f)) for f in _MASK_SKELETON_FILES)
    ]


def find_missing_region_graphs(champollion_data_root, crops_dir, level) -> list[str]:
    """Return the missing region graph paths of the hemispheres with at least one completed region."""
    completed = find_completed_regions(crops_dir)
    missing = []
    for hemi, side in (("left", "L"), ("right", "R")):
        if not completed[hemi]:
            continue
        path = compute_region_graph_path(champollion_data_root, side, level)
        if not osp.exists(path):
            missing.append(path)
    return missing


def print_warning(label: str, names: Sequence[str]) -> None:
    """Print one ``Warning: <label>: <names>`` line; nothing when ``names`` is empty."""
    if names:
        print(f"Warning: {label}: {', '.join(names)}")
