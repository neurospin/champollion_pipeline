#!/usr/bin/env python3
"""
Per-acquisition pruning in prune_failed_subjects.py (TASK-INPUTKIND-BDRABCZUK-CE7FCF70325D).

With a stage 1 --bids input_qc.tsv (one row per scan, with an acquisition column) each row maps to
the crop id cortical_tiles gives that scan: the participant_id followed by the first ses-, acq- and
run- entity of the acquisition folder, in that order (cortical_tiles skip_list.session_key), for
example sub-002 and ses-V1 give sub-002_ses-V1, and default_acquisition gives sub-002.

- REQ-INPUTKIND-BDRABCZUK-8970A7B2FE9D: the crop id of a qc 0 row is pruned.
- REQ-INPUTKIND-BDRABCZUK-1E95DDDBF68A: the files of the crop id of a qc 1 row are kept, also when
  a pruned crop id is a delimited token of their names (sub-006 pruned, sub-006_ses-V1 kept).
- REQ-INPUTKIND-BDRABCZUK-6DD7C71362FB: the aggregate rows of the crop id of a qc 1 row are kept.
- REQ-INPUTKIND-BDRABCZUK-2C05DB90D783: a crop id that no row maps to is pruned.
- REQ-INPUTKIND-BDRABCZUK-36390C8564E1: without an acquisition column, crop ids are compared with
  participant_id, as before.
- DEF-INPUTKIND-BDRABCZUK-0151BA26539C is reproduced by
  TestPassingAcquisitionKept::test_bids_passing_session_crop_is_kept.

All subject IDs are synthetic.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from champollion_pipeline.prune_failed_subjects import PruneFailedSubjects
from champollion_pipeline.utils.qc_report import ACQUISITION_COLUMN, write_qc_tsv
from tests.prune_layout import crop_file, make_aggregate, run_prune, skeleton_file, touch

REGION = "region_a"
SIDE = "R"
FAILED_REASON = "unreadable_nifti"


def _make_scans(base: Path, crop_ids: list[str]) -> None:
    """Create, per crop id, its crop file and its resampled skeleton (real cortical_tiles layout)."""
    for crop_id in crop_ids:
        touch(crop_file(base, REGION, SIDE, crop_id))
        touch(skeleton_file(base, SIDE, crop_id))


def _prune_with_acquisitions(base: Path, rows: list[tuple[str, str, int]]) -> int:
    """Write <base>/input_qc.tsv as stage 1 --bids does from (participant_id, acquisition, qc) rows; run prune."""
    qc_path = base / "input_qc.tsv"
    write_qc_tsv(
        qc_path,
        [
            {
                "participant_id": subject,
                "qc": qc,
                ACQUISITION_COLUMN: acquisition,
                "reason": "" if qc else FAILED_REASON,
            }
            for subject, acquisition, qc in rows
        ],
        extra_columns=(ACQUISITION_COLUMN,),
    )
    script = PruneFailedSubjects()
    script.args = script.parse_args([str(base), "--qc", str(qc_path)])
    return script.run()


def _subject_column(csv_path: Path) -> list[str]:
    return pd.read_csv(csv_path, dtype=str, keep_default_na=False)["Subject"].tolist()


# ---------------------------------------------------------------------------
# REQ-INPUTKIND-BDRABCZUK-8970A7B2FE9D: the crop id of a qc 0 row is pruned
# ---------------------------------------------------------------------------


class TestFailedAcquisitionPruned:
    def test_failed_session_crop_is_deleted(self, tmp_path):
        _make_scans(tmp_path, ["sub-002_ses-V1", "sub-002_ses-V2"])
        _prune_with_acquisitions(tmp_path, [("sub-002", "ses-V1", 0), ("sub-002", "ses-V2", 1)])
        assert not crop_file(tmp_path, REGION, SIDE, "sub-002_ses-V1").exists()

    def test_failed_default_acquisition_crop_is_deleted_beside_passing_session(self, tmp_path):
        _make_scans(tmp_path, ["sub-006", "sub-006_ses-V1"])
        _prune_with_acquisitions(tmp_path, [("sub-006", "default_acquisition", 0), ("sub-006", "ses-V1", 1)])
        assert not crop_file(tmp_path, REGION, SIDE, "sub-006").exists()

    def test_crop_id_orders_entities_ses_acq_run(self, tmp_path):
        """The crop id of acquisition run-1_acq-mp_ses-V1 is sub-004_ses-V1_acq-mp_run-1.

        Checked on the kept side: a crop id no row maps to is pruned anyway, so only a qc 1 row
        tells whether the row maps to the right crop id.
        """
        _make_scans(tmp_path, ["sub-004_ses-V1_acq-mp_run-1", "sub-004_ses-V2"])
        _prune_with_acquisitions(tmp_path, [("sub-004", "run-1_acq-mp_ses-V1", 1), ("sub-004", "ses-V2", 0)])
        assert crop_file(tmp_path, REGION, SIDE, "sub-004_ses-V1_acq-mp_run-1").exists()

    def test_failed_session_aggregate_row_is_removed(self, tmp_path):
        _make_scans(tmp_path, ["sub-002_ses-V1", "sub-002_ses-V2"])
        _, csv, _ = make_aggregate(tmp_path, REGION, "Rskeleton", ["sub-002_ses-V1", "sub-002_ses-V2"])
        _prune_with_acquisitions(tmp_path, [("sub-002", "ses-V1", 0), ("sub-002", "ses-V2", 1)])
        assert "sub-002_ses-V1" not in _subject_column(csv)

    def test_subject_without_passing_acquisition_loses_every_crop(self, tmp_path):
        _make_scans(tmp_path, ["sub-005_ses-V1", "sub-005_ses-V2", "sub-008_ses-V1"])
        _prune_with_acquisitions(
            tmp_path, [("sub-005", "ses-V1", 0), ("sub-005", "ses-V2", 0), ("sub-008", "ses-V1", 1)]
        )
        left = sorted(p.name for p in crop_file(tmp_path, REGION, SIDE, "x").parent.iterdir())
        assert not [name for name in left if name.startswith("sub-005")]


# ---------------------------------------------------------------------------
# REQ-INPUTKIND-BDRABCZUK-1E95DDDBF68A: files of the crop id of a qc 1 row are kept
# REQ-INPUTKIND-BDRABCZUK-6DD7C71362FB: aggregate rows of the crop id of a qc 1 row are kept
# ---------------------------------------------------------------------------


class TestPassingAcquisitionKept:
    def test_bids_passing_session_crop_is_kept(self, tmp_path):
        """DEF-INPUTKIND-BDRABCZUK-0151BA26539C: sub-002_ses-V1 passes, yet was deleted as absent."""
        _make_scans(tmp_path, ["sub-002_ses-V1", "sub-002_ses-V2"])
        _prune_with_acquisitions(tmp_path, [("sub-002", "ses-V1", 1), ("sub-002", "ses-V2", 0)])
        assert crop_file(tmp_path, REGION, SIDE, "sub-002_ses-V1").exists()

    def test_passing_session_files_are_kept_beside_failed_default_acquisition(self, tmp_path):
        """sub-006 is pruned and is a delimited token of the sub-006_ses-V1 names, which pass."""
        _make_scans(tmp_path, ["sub-006", "sub-006_ses-V1"])
        _prune_with_acquisitions(tmp_path, [("sub-006", "default_acquisition", 0), ("sub-006", "ses-V1", 1)])
        kept = [crop_file(tmp_path, REGION, SIDE, "sub-006_ses-V1"), skeleton_file(tmp_path, SIDE, "sub-006_ses-V1")]
        assert [p.exists() for p in kept] == [True, True]

    def test_passing_session_aggregate_row_is_kept(self, tmp_path):
        _make_scans(tmp_path, ["sub-002_ses-V1", "sub-002_ses-V2"])
        npy, csv, arr = make_aggregate(tmp_path, REGION, "Rskeleton", ["sub-002_ses-V1", "sub-002_ses-V2"])
        _prune_with_acquisitions(tmp_path, [("sub-002", "ses-V1", 1), ("sub-002", "ses-V2", 0)])
        assert _subject_column(csv) == ["sub-002_ses-V1"]
        np.testing.assert_array_equal(np.load(npy), arr[[0]])


# ---------------------------------------------------------------------------
# REQ-INPUTKIND-BDRABCZUK-2C05DB90D783: a crop id no row maps to is pruned
# ---------------------------------------------------------------------------


class TestUnmappedCropIdPruned:
    def test_crop_id_no_row_maps_to_is_deleted(self, tmp_path):
        _make_scans(tmp_path, ["sub-007_ses-V1", "sub-007_ses-V9"])
        _prune_with_acquisitions(tmp_path, [("sub-007", "ses-V1", 1)])
        assert not crop_file(tmp_path, REGION, SIDE, "sub-007_ses-V9").exists()


# ---------------------------------------------------------------------------
# REQ-INPUTKIND-BDRABCZUK-36390C8564E1: without an acquisition column, crop ids equal participant_id
# ---------------------------------------------------------------------------


class TestWithoutAcquisitionColumn:
    def test_crop_of_participant_with_qc_0_is_deleted(self, tmp_path):
        _make_scans(tmp_path, ["sub-01", "sub-02"])
        run_prune(tmp_path, [("sub-01", 0), ("sub-02", 1)])
        assert not crop_file(tmp_path, REGION, SIDE, "sub-01").exists()

    def test_crop_of_passing_participant_is_kept(self, tmp_path):
        _make_scans(tmp_path, ["sub-01", "sub-02"])
        run_prune(tmp_path, [("sub-01", 0), ("sub-02", 1)])
        assert crop_file(tmp_path, REGION, SIDE, "sub-02").exists()
