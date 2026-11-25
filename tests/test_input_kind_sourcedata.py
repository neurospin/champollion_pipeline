"""Tests of the sourcedata ambiguity check on the stage 1 file list: input_kind.sourcedata_t1_files.

The module under test is imported inside each test, through ``_module``, so a missing function fails each test on its
own instead of stopping collection of the whole file.
"""

import importlib
import os
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

MODULE = "champollion_pipeline.utils.input_kind"
SOURCEDATA = "sourcedata"

# Folder and file names the generated trees draw from: T1 files and non-T1 files, inside and outside sourcedata,
# below folders find_t1_files skips and below look-alike folder names.
_FOLDER_NAMES = (SOURCEDATA, "rawdata", "sub-01", "anat", "derivatives", "qc", "t1mri", "sourcedata_old", ".hidden")
_FILE_NAMES = ("a.nii.gz", "b.nii", "sub-01_T1w.nii.gz", "sub-01_bold.nii.gz", "notes.txt", "sourcedata.nii.gz")
_RELATIVE_FILES = st.lists(
    st.tuples(st.lists(st.sampled_from(_FOLDER_NAMES), max_size=3), st.sampled_from(_FILE_NAMES)).map(
        lambda pair: os.path.join(*pair[0], pair[1])
    ),
    max_size=8,
    unique=True,
)


def _module():
    """Return the module under test."""
    return importlib.import_module(MODULE)


def _sourcedata_t1_files(path: Path) -> list[str]:
    """Return what sourcedata_t1_files reports for ``path``."""
    return _module().sourcedata_t1_files(str(path))


def _touch(root: Path, relative: str) -> Path:
    """Create an empty file at ``root/relative`` with its parent folders."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _has_sourcedata_folder(relative: str) -> bool:
    """Return whether one of the folder names of ``relative`` is sourcedata."""
    return SOURCEDATA in Path(relative).parts[:-1]


@pytest.mark.parametrize(
    "relative",
    [
        "sourcedata/IXI002.nii.gz",
        "sourcedata/IXI002.nii",
        "sourcedata/sub-01/ses-1/anat/sub-01_ses-1_T1w.nii.gz",
        "study/sourcedata/IXI002.nii.gz",
        # No .arg file: a file below t1mri is T1 input (REQ-ARGDISC-BDRABCZUK-1786F42F6CE1)
        "sourcedata/IXI002/t1mri/default_acquisition/IXI002.nii.gz",
    ],
)
def test_t1_below_sourcedata_is_returned(tmp_path, relative):
    _touch(tmp_path, relative)

    assert _sourcedata_t1_files(tmp_path) == [relative]


def test_sourcedata_t1_is_returned_beside_rawdata(tmp_path):
    # rawdata sorts before sourcedata, so the detector stops at the rawdata T1 and never sees this one.
    _touch(tmp_path, "rawdata/sub-01/anat/sub-01_T1w.nii.gz")
    _touch(tmp_path, "sourcedata/sub-01/anat/sub-01_T1w.nii.gz")

    assert _sourcedata_t1_files(tmp_path) == ["sourcedata/sub-01/anat/sub-01_T1w.nii.gz"]


@pytest.mark.parametrize(
    "relative",
    [
        "IXI002.nii.gz",
        "rawdata/sub-01/anat/sub-01_T1w.nii.gz",
        "sourcedata.nii.gz",
        "sourcedata_old/IXI002.nii.gz",
        "mysourcedata/IXI002.nii.gz",
    ],
)
def test_t1_outside_sourcedata_is_left_out(tmp_path, relative):
    _touch(tmp_path, relative)
    assert _module().find_t1_files(str(tmp_path)) == [relative]

    assert _sourcedata_t1_files(tmp_path) == []


@pytest.mark.parametrize(
    ("input_folder", "relative"),
    [
        ("sourcedata", "IXI002.nii.gz"),
        ("sourcedata/study", "sub-01/anat/sub-01_T1w.nii.gz"),
    ],
)
def test_input_inside_sourcedata_returns_none(tmp_path, input_folder, relative):
    input_dir = tmp_path / input_folder
    _touch(input_dir, relative)

    assert _sourcedata_t1_files(input_dir) == []


@pytest.mark.parametrize(
    "relative",
    [
        "sourcedata/derivatives/IXI002.nii.gz",
        "sourcedata/qc/IXI002.nii.gz",
        "sourcedata/.hidden/IXI002.nii.gz",
        "sourcedata/.IXI002.nii.gz",
        "sourcedata/sub-01/anat/sub-01_bold.nii.gz",
        "sourcedata/IXI002.nii.gz.minf",
        "sourcedata/notes.txt",
    ],
)
def test_non_t1_file_below_sourcedata_is_left_out(tmp_path, relative):
    _touch(tmp_path, relative)

    assert _sourcedata_t1_files(tmp_path) == []


def test_missing_input_returns_none(tmp_path):
    assert _sourcedata_t1_files(tmp_path / "missing") == []


def test_paths_are_sorted(tmp_path):
    relatives = [
        "sourcedata/sub-02/anat/sub-02_T1w.nii.gz",
        "study/sourcedata/IXI003.nii",
        "sourcedata/sub-01/anat/sub-01_T1w.nii.gz",
        "sourcedata/IXI002.nii.gz",
    ]
    for relative in relatives:
        _touch(tmp_path, relative)

    found = _sourcedata_t1_files(tmp_path)

    assert isinstance(found, list)
    assert found == sorted(relatives)


@settings(max_examples=60)
@given(relatives=_RELATIVE_FILES)
def test_result_is_within_find_t1_files(relatives):
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for relative in relatives:
            _touch(root, relative)

        found = _sourcedata_t1_files(root)

        assert set(found) <= set(_module().find_t1_files(folder))


@settings(max_examples=60)
@given(relatives=_RELATIVE_FILES)
def test_result_is_find_t1_files_below_sourcedata(relatives):
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for relative in relatives:
            _touch(root, relative)

        expected = sorted(path for path in _module().find_t1_files(folder) if _has_sourcedata_folder(path))

        assert _sourcedata_t1_files(root) == expected
