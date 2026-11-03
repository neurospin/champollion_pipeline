#!/usr/bin/env python3
"""
prune_failed_subjects.py on the real cortical_tiles crop layout.

cortical_tiles writes one crop per subject to
crops/<masks>/2mm/<region>/mask/<side>crops/<id>_cropped_skeleton.nii.gz and stacks
them into the aggregated arrays crops/<masks>/2mm/<region>/mask/<side><name>.npy, whose
rows follow crops/<masks>/2mm/<region>/mask/<side><name>_subject.csv (column Subject).

- REQ-PATTERNS-BDRABCZUK-4E9993FCE014: discovery returns exactly the <id> of
  the <side>crops/<id>_cropped_skeleton.nii.gz files.
- REQ-PATTERNS-BDRABCZUK-41EE61ED5E09: the rows of a pruned subject leave each
  subject CSV and its paired .npy array, the other rows staying aligned.
- REQ-PATTERNS-BDRABCZUK-3CB582F38DEE: --dry-run leaves both files unchanged.

All subject IDs are synthetic.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from champollion_pipeline.prune_failed_subjects import PruneFailedSubjects
from tests.prune_layout import CROP_SUFFIX, crops_2mm, make_aggregate, make_crop_files, run_prune, touch


def _csv_subjects(csv: Path) -> list[str]:
    return pd.read_csv(csv, dtype=str)["Subject"].tolist()


# ---------------------------------------------------------------------------
# REQ-PATTERNS-BDRABCZUK-4E9993FCE014: discovery from the real layout
# ---------------------------------------------------------------------------


class TestDiscoverRealLayout:
    def test_returns_ids_of_cropped_skeleton_files_of_both_sides(self, tmp_path):
        make_crop_files(tmp_path, "S.C.-sylv.", "R", ["sub-01", "sub-02"])
        make_crop_files(tmp_path, "S.C.-sylv.", "L", ["sub-02", "sub-03"])
        make_crop_files(tmp_path, "F.I.P.", "R", ["sub-04"])

        result = PruneFailedSubjects()._discover_subjects(str(crops_2mm(tmp_path)))
        assert result == {"sub-01", "sub-02", "sub-03", "sub-04"}

    def test_ids_containing_underscore_and_dot_kept_whole(self, tmp_path):
        make_crop_files(tmp_path, "region_a", "R", ["sub-1_ses-2", "100002.v2"])

        result = PruneFailedSubjects()._discover_subjects(str(crops_2mm(tmp_path)))
        assert result == {"sub-1_ses-2", "100002.v2"}

    def test_other_file_names_yield_no_id(self, tmp_path):
        make_crop_files(tmp_path, "region_a", "R", ["sub-01"])
        make_aggregate(tmp_path, "region_a", "Rskeleton", ["sub-01"])
        mask = crops_2mm(tmp_path) / "region_a" / "mask"
        touch(mask / "Rcrops" / f"sub-01{CROP_SUFFIX}.minf")
        touch(mask / "Rlabels" / "sub-08_cropped_foldlabel.nii.gz")
        touch(mask / "Rextremities" / "sub-09_cropped_extremities.nii.gz")
        touch(mask / "sub_id.npy")

        result = PruneFailedSubjects()._discover_subjects(str(crops_2mm(tmp_path)))
        assert result == {"sub-01"}

    def test_file_directly_under_region_yields_no_id(self, tmp_path):
        make_crop_files(tmp_path, "region_a", "R", ["sub-01"])
        touch(crops_2mm(tmp_path) / "region_a" / "sub-77.nii.gz")

        result = PruneFailedSubjects()._discover_subjects(str(crops_2mm(tmp_path)))
        assert result == {"sub-01"}

    def test_run_deletes_crop_files_of_failed_subject_on_real_layout(self, tmp_path):
        """The defect symptom: on the real layout prune found 0 subjects and deleted nothing."""
        make_crop_files(tmp_path, "region_a", "R", ["sub-01", "sub-02"])
        make_crop_files(tmp_path, "region_a", "L", ["sub-01", "sub-02"])
        run_prune(tmp_path, [("sub-01", 1), ("sub-02", 0)])

        mask = crops_2mm(tmp_path) / "region_a" / "mask"
        remaining = sorted(str(p.relative_to(mask)) for p in mask.rglob("*") if p.is_file())
        assert remaining == [f"Lcrops/sub-01{CROP_SUFFIX}", f"Rcrops/sub-01{CROP_SUFFIX}"]


# ---------------------------------------------------------------------------
# REQ-PATTERNS-BDRABCZUK-41EE61ED5E09: aggregated rows of removed subjects
# ---------------------------------------------------------------------------


class TestAggregateRowsFiltered:
    def test_failed_subject_row_leaves_csv_and_npy(self, tmp_path):
        subjects = ["sub-01", "sub-02", "sub-03"]
        make_crop_files(tmp_path, "region_a", "R", subjects)
        npy, csv, arr = make_aggregate(tmp_path, "region_a", "Rskeleton", subjects)
        run_prune(tmp_path, [("sub-01", 1), ("sub-02", 0), ("sub-03", 1)])

        assert _csv_subjects(csv) == ["sub-01", "sub-03"]
        np.testing.assert_array_equal(np.load(npy), arr[[0, 2]])

    def test_rows_filtered_independently_of_discovery(self, tmp_path, monkeypatch):
        """Discovery stubbed to the real IDs, so only the aggregate filtering is exercised."""
        subjects = ["sub-01", "sub-02", "sub-03"]
        make_crop_files(tmp_path, "region_a", "R", subjects)
        npy, csv, arr = make_aggregate(tmp_path, "region_a", "Rskeleton", subjects)
        monkeypatch.setattr(PruneFailedSubjects, "_discover_subjects", lambda _self, _crops: set(subjects))
        run_prune(tmp_path, [("sub-02", 1)])

        assert _csv_subjects(csv) == ["sub-02"]
        np.testing.assert_array_equal(np.load(npy), arr[[1]])

    def test_rows_stay_aligned_in_each_side_and_region(self, tmp_path):
        subjects = ["sub-01", "sub-02", "sub-03", "sub-04"]
        aggregates = []
        for region in ("region_a", "region_b"):
            for side in ("L", "R"):
                make_crop_files(tmp_path, region, side, subjects)
                aggregates.append(make_aggregate(tmp_path, region, f"{side}skeleton", subjects))
        run_prune(tmp_path, [("sub-02", 1), ("sub-03", 1)])

        for npy, csv, arr in aggregates:
            assert _csv_subjects(csv) == ["sub-02", "sub-03"], csv
            np.testing.assert_array_equal(np.load(npy), arr[[1, 2]], err_msg=str(npy))

    def test_numeric_ids_match_whole_value(self, tmp_path):
        subjects = ["1000021", "100002", "2100002"]
        make_crop_files(tmp_path, "region_a", "R", subjects)
        npy, csv, arr = make_aggregate(tmp_path, "region_a", "Rskeleton", subjects)
        run_prune(tmp_path, [("100002", 0), ("1000021", 1), ("2100002", 1)])

        assert _csv_subjects(csv) == ["1000021", "2100002"]
        np.testing.assert_array_equal(np.load(npy), arr[[0, 2]])


# ---------------------------------------------------------------------------
# REQ-PATTERNS-BDRABCZUK-3CB582F38DEE: --dry-run leaves the aggregates unchanged
# ---------------------------------------------------------------------------


class TestAggregateDryRun:
    def test_dry_run_leaves_csv_and_npy_bytes_unchanged(self, tmp_path, monkeypatch):
        """Discovery stubbed so a subject is due for removal whatever the discovery code finds."""
        subjects = ["sub-01", "sub-02", "sub-03"]
        make_crop_files(tmp_path, "region_a", "R", subjects)
        npy, csv, _arr = make_aggregate(tmp_path, "region_a", "Rskeleton", subjects)
        monkeypatch.setattr(PruneFailedSubjects, "_discover_subjects", lambda _self, _crops: set(subjects))
        before = (npy.read_bytes(), csv.read_bytes())
        run_prune(tmp_path, [("sub-01", 1), ("sub-03", 1)], dry_run=True)

        assert (npy.read_bytes(), csv.read_bytes()) == before
