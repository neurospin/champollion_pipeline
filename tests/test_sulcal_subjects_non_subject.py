"""Stage 6 sulcal subject listing follows the shared subject-folder rule.

REQ-INPUTKIND-BDRABCZUK-F9B3F4DDFCCD (list_sulcal_subjects) and
REQ-INPUTKIND-BDRABCZUK-C59879844724 (snapshots input QC rows).
Synthetic folders only.
"""

from champollion_pipeline.snapshot_types.input_qc import SnapshotsInputQC, check_sulcal_inputs
from champollion_pipeline.snapshot_types.sulcal import list_sulcal_subjects
from champollion_pipeline.utils.subject_folders import list_morphologist_subjects


def _graph(subject_dir, name="Lsulci.arg"):
    """Create an empty graph under ``subject_dir/t1mri/acq1/default_analysis/folds/3.1`` and return its path."""
    folds = subject_dir / "t1mri" / "acq1" / "default_analysis" / "folds" / "3.1"
    folds.mkdir(parents=True, exist_ok=True)
    path = folds / name
    path.touch()
    (folds / "Rsulci.arg").touch()
    return path


def _tree(root):
    """Build subjects sub-b, sub-a, a qc folder with no t1mri, a dot-folder, and a file."""
    for name in ("sub-b", "sub-a"):
        _graph(root / name)
    (root / "qc").mkdir()
    (root / "qc" / "report.tsv").touch()
    (root / ".hidden").mkdir()
    (root / "notes.txt").touch()
    return root


class TestListSulcalSubjects:
    """REQ-INPUTKIND-BDRABCZUK-F9B3F4DDFCCD."""

    def test_qc_without_t1mri_is_no_subject(self, tmp_path):
        _tree(tmp_path)
        assert "qc" not in list_sulcal_subjects(str(tmp_path))

    def test_dot_folder_is_no_subject(self, tmp_path):
        _tree(tmp_path)
        assert ".hidden" not in list_sulcal_subjects(str(tmp_path))

    def test_returns_what_the_shared_rule_returns_in_ascending_order(self, tmp_path):
        _tree(tmp_path)
        assert list_sulcal_subjects(str(tmp_path)) == list_morphologist_subjects(str(tmp_path)) == ["sub-a", "sub-b"]

    def test_qc_holding_t1mri_stays_a_subject(self, tmp_path):
        _tree(tmp_path)
        _graph(tmp_path / "qc")
        assert list_sulcal_subjects(str(tmp_path)) == ["qc", "sub-a", "sub-b"]


class TestInputQCSkipsNonSubjects:
    """REQ-INPUTKIND-BDRABCZUK-C59879844724."""

    def test_qc_and_dot_folder_get_no_input_qc_row(self, tmp_path):
        _tree(tmp_path)
        report = SnapshotsInputQC()
        check_sulcal_inputs(report, str(tmp_path))
        assert report.excluded() == []
        written = report.write(str(tmp_path))
        with open(written, encoding="utf-8") as handle:
            ids = [line.split("\t")[0] for line in handle.read().splitlines()[1:]]
        assert ids == ["sub-a", "sub-b"]
