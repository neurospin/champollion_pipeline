#!/usr/bin/env python3
"""
Conflicting QC rows in prune_failed_subjects.py (TASK-INPUTKIND-BDRABCZUK-CE7FCF70325D).

REQ-INPUTKIND-BDRABCZUK-84FA31A1181B: when the --qc file has an acquisition column and one crop id
(str(ScanId(participant_id, acquisition, bids=True))) is the crop id of a row with qc 0 and of a row
with a qc value other than 0, PruneFailedSubjects.run raises a ValueError naming every such crop id
in sorted order, before deleting a derivatives file; --dry-run is refused the same way. User decision
2026-11-12, replacing the design's assumed "keep". A file without an acquisition column is unchanged.

Stage 1's write_qc_tsv refuses a repeated (participant_id, acquisition) row, so such a file only
comes from an edited or concatenated TSV: the QC file is written here as plain text. The scan
fixtures come from tests/test_prune_failed_subjects_per_acquisition.py and tests/prune_layout.py.
All subject IDs are synthetic.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from champollion_pipeline.prune_failed_subjects import PruneFailedSubjects
from tests.prune_layout import crop_file, make_aggregate, skeleton_file
from tests.test_prune_failed_subjects_per_acquisition import REGION, SIDE, _make_scans

CONFLICT_ID = "sub-002_ses-V1"
CONFLICT_ROWS = [("sub-002", "ses-V1", 0), ("sub-002", "ses-V1", 1), ("sub-002", "ses-V2", 0)]
SCANS = ["sub-002_ses-V1", "sub-002_ses-V2"]


def _prune_with_raw_rows(base: Path, rows: list[tuple[str, str, int]], dry_run: bool = False) -> int:
    """Write <base>/edited_qc.tsv from (participant_id, acquisition, qc) rows, repeats kept; run prune."""
    qc_path = base / "edited_qc.tsv"
    qc_path.write_text("participant_id\tacquisition\tqc\n" + "".join(f"{s}\t{a}\t{q}\n" for s, a, q in rows))
    script = PruneFailedSubjects()
    script.args = script.parse_args([str(base), "--qc", str(qc_path)] + (["--dry-run"] if dry_run else []))
    return script.run()


# ---------------------------------------------------------------------------
# REQ-INPUTKIND-BDRABCZUK-84FA31A1181B: conflicting rows refuse the QC file
# ---------------------------------------------------------------------------


class TestConflictingRowsRefused:
    def test_conflicting_rows_raise_naming_crop_id(self, tmp_path):
        _make_scans(tmp_path, SCANS)
        with pytest.raises(ValueError, match=CONFLICT_ID):
            _prune_with_raw_rows(tmp_path, CONFLICT_ROWS)

    def test_conflicting_rows_delete_no_file(self, tmp_path):
        """sub-002_ses-V2 fails without conflict, yet neither its files nor its aggregate row go."""
        _make_scans(tmp_path, SCANS)
        npy, csv, arr = make_aggregate(tmp_path, REGION, "Rskeleton", SCANS)
        files = [crop_file(tmp_path, REGION, SIDE, s) for s in SCANS]
        files += [skeleton_file(tmp_path, SIDE, s) for s in SCANS]
        with pytest.raises(ValueError, match=CONFLICT_ID):
            _prune_with_raw_rows(tmp_path, CONFLICT_ROWS)
        assert [p.exists() for p in files] == [True] * len(files)
        assert pd.read_csv(csv, dtype=str)["Subject"].tolist() == SCANS
        assert len(np.load(npy)) == len(arr)

    def test_dry_run_conflicting_rows_raise(self, tmp_path):
        _make_scans(tmp_path, SCANS)
        with pytest.raises(ValueError, match=CONFLICT_ID):
            _prune_with_raw_rows(tmp_path, CONFLICT_ROWS, dry_run=True)

    def test_two_conflicting_crop_ids_named_in_sorted_order(self, tmp_path):
        """Rows list sub-009 first; the error names sub-003_ses-V1 before sub-009_ses-V2."""
        _make_scans(tmp_path, ["sub-003_ses-V1", "sub-009_ses-V2"])
        rows = [
            ("sub-009", "ses-V2", 1),
            ("sub-009", "ses-V2", 0),
            ("sub-003", "ses-V1", 0),
            ("sub-003", "ses-V1", 1),
        ]
        with pytest.raises(ValueError, match=r"(?s)sub-003_ses-V1.*sub-009_ses-V2"):
            _prune_with_raw_rows(tmp_path, rows)

    def test_acquisition_file_without_conflict_prunes(self, tmp_path):
        """Guard: two qc 0 rows of one crop id are no conflict; the failed crop goes, the passing one stays."""
        _make_scans(tmp_path, SCANS)
        rows = [("sub-002", "ses-V1", 0), ("sub-002", "ses-V1", 0), ("sub-002", "ses-V2", 1)]
        code = _prune_with_raw_rows(tmp_path, rows)
        survivors = [crop_file(tmp_path, REGION, SIDE, s).exists() for s in SCANS]
        assert (code, survivors) == (0, [False, True])
