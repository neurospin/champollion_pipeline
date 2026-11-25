"""Stage 2 lists its input subjects through the shared subject-folder rule.

Requirements (INPUTKIND, recorded by the red stage of the stage 2 listing Task):
    list_input_subjects leaves out a qc folder that holds no t1mri folder
    list_input_subjects keeps a qc folder that holds a t1mri folder (pin)
    list_input_subjects keeps a folder named ra and leaves out a dot-folder (pin)
    list_input_subjects keeps the OSError of an unreadable input folder (pin)
    the input QC has no row for a qc folder without t1mri
    the input QC keeps a row for a qc folder with t1mri (pin)
    a qc folder without t1mri is no side of a prefix pair, nor a dotted ID
"""

import os

import pytest

from champollion_pipeline.run_cortical_tiles_inputs import (
    fetch_input_qc_rows,
    find_dotted_ids,
    find_prefix_id_pairs,
    list_input_subjects,
)

pytestmark = pytest.mark.unit  # noqa: V107

GRAPH = "t1mri/default_acquisition/default_analysis/folds/3.1/default_session_auto"


def _subject(root, name, with_graphs=True):
    folder = root / name / GRAPH
    folder.mkdir(parents=True)
    if with_graphs:
        for side in ("L", "R"):
            (folder / f"{side}{name}.arg").write_text("x")
    return root / name


def _qc_folder(root, with_t1mri):
    qc = root / "qc"
    qc.mkdir()
    (qc / "report.tsv").write_text("x")
    if with_t1mri:
        (qc / "t1mri" / "default_acquisition").mkdir(parents=True)
    return qc


def test_qc_folder_without_t1mri_is_no_subject(tmp_path):
    """qc without a t1mri folder is left out of the listing."""
    _subject(tmp_path, "s01")
    _qc_folder(tmp_path, with_t1mri=False)

    assert list_input_subjects(str(tmp_path)) == ["s01"]


def test_qc_folder_with_t1mri_is_a_subject(tmp_path):
    """Pin: a qc folder that holds a t1mri folder stays listed."""
    _subject(tmp_path, "s01")
    _qc_folder(tmp_path, with_t1mri=True)

    assert list_input_subjects(str(tmp_path)) == ["qc", "s01"]


def test_ra_folder_is_a_subject_and_dot_folder_is_not(tmp_path):
    """Pin: a folder named ra is listed, a dot-folder is not."""
    _subject(tmp_path, "ra")
    _subject(tmp_path, "s01")
    (tmp_path / ".cache").mkdir()

    assert list_input_subjects(str(tmp_path)) == ["ra", "s01"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every folder")
def test_unreadable_input_folder_keeps_the_oserror(tmp_path):
    """Pin: an input folder that cannot be listed raises the OSError."""
    folder = tmp_path / "in"
    folder.mkdir()
    folder.chmod(0)
    try:
        with pytest.raises(OSError):
            list_input_subjects(str(folder))
    finally:
        folder.chmod(0o700)


def test_input_qc_has_no_row_for_qc_without_t1mri(tmp_path):
    """The input QC has one row per listed subject: none for a qc folder without t1mri."""
    _subject(tmp_path, "s01")
    _qc_folder(tmp_path, with_t1mri=False)

    rows = fetch_input_qc_rows(str(tmp_path), "default_analysis/folds/3.1/default_session_auto", None, False, False)

    assert [row["participant_id"] for row in rows] == ["s01"]


def test_input_qc_keeps_a_row_for_qc_with_t1mri(tmp_path):
    """Pin: a qc folder that holds a t1mri folder keeps its row."""
    _subject(tmp_path, "s01")
    _qc_folder(tmp_path, with_t1mri=True)

    rows = fetch_input_qc_rows(str(tmp_path), "default_analysis/folds/3.1/default_session_auto", None, False, False)

    assert [row["participant_id"] for row in rows] == ["qc", "s01"]


def test_qc_without_t1mri_is_no_side_of_a_prefix_pair(tmp_path):
    """A qc folder without t1mri does not prefix qc2."""
    _subject(tmp_path, "qc2")
    _qc_folder(tmp_path, with_t1mri=False)

    assert find_prefix_id_pairs(str(tmp_path)) == []


def test_qc_with_t1mri_is_a_side_of_a_prefix_pair(tmp_path):
    """Pin: a qc folder that holds a t1mri folder prefixes qc2."""
    _subject(tmp_path, "qc2")
    _qc_folder(tmp_path, with_t1mri=True)

    assert find_prefix_id_pairs(str(tmp_path)) == [("qc", "qc2")]


def test_dotted_ids_come_from_the_listed_subjects(tmp_path):
    """Dotted IDs are looked for among the listed subjects only."""
    _subject(tmp_path, "A.1")
    _qc_folder(tmp_path, with_t1mri=False)

    assert find_dotted_ids(str(tmp_path)) == ["A.1"]
