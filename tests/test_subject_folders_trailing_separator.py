"""is_non_subject: a path ending in a separator gives the result of the same path without it."""

import os

import pytest

from champollion_pipeline.utils.subject_folders import is_non_subject


def make_qc(tmp_path, with_t1mri: bool):
    qc_dir = tmp_path / "qc"
    qc_dir.mkdir(parents=True)
    if with_t1mri:
        (qc_dir / "t1mri").mkdir()
    return qc_dir


@pytest.mark.parametrize("suffix", ["/", "//", "///"])
def test_qc_folder_without_t1mri_is_non_subject_with_trailing_separators(tmp_path, suffix):
    qc_dir = make_qc(tmp_path, with_t1mri=False)
    assert is_non_subject(str(qc_dir)) is True
    assert is_non_subject(str(qc_dir) + suffix) is True


@pytest.mark.parametrize("suffix", ["/", "//"])
def test_qc_and_ordinary_folders_keep_their_result_with_trailing_separators(tmp_path, suffix):
    plain_qc = make_qc(tmp_path / "a", with_t1mri=False)
    subject_qc = make_qc(tmp_path / "b", with_t1mri=True)
    subject_dir = tmp_path / "sub-01"
    subject_dir.mkdir()
    # the first assertion fails while the qc rule ignores a trailing separator; the others pin the subjects
    assert is_non_subject(str(plain_qc) + suffix) is True
    assert is_non_subject(str(subject_qc) + suffix) is False
    assert is_non_subject(str(subject_dir) + suffix) is False


@pytest.mark.parametrize("suffix", ["/", "//"])
def test_dot_folder_is_non_subject_with_trailing_separators(tmp_path, suffix):
    dot_dir = tmp_path / ".x"
    dot_dir.mkdir()
    assert is_non_subject(str(dot_dir)) is True
    assert is_non_subject(str(dot_dir) + suffix) is True


def test_pathlike_input_still_works(tmp_path):
    dot_dir = tmp_path / ".x"
    dot_dir.mkdir()
    assert is_non_subject(dot_dir) is True
    assert is_non_subject(os.fspath(dot_dir) + os.sep) is True
