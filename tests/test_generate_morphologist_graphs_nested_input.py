#!/usr/bin/env python3
"""
Stage 1 reader on nested trees: what _get_input_files returns and how the input QC and morphologist-cli take it.

morphologist-cli names a subject after the last path component of each input and reads no attribute from the folders,
so a nested path must reach it unchanged, and two files of one name in different folders must stop at the input QC.
"""

import os
from pathlib import Path

import pytest

from champollion_pipeline.generate_morphologist_graphs import GenerateMorphologistGraphs
from champollion_pipeline.utils.input_kind import find_t1_files
from tests.test_generate_morphologist_graphs import _qc_by_id, _run_stage, _write_nifti

pytestmark = pytest.mark.unit  # noqa: V107

BIDS_T1 = "sub-01/ses-V1/anat/sub-01_ses-V1_T1w.nii.gz"
DUPLICATE = (0, "duplicate_subject")


@pytest.fixture
def dirs(tmp_path):
    """Return (input_dir, output_dir), both existing."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    return input_dir, output_dir


def _add_file(input_dir: Path, relative_path: str) -> None:
    """Create an empty file at ``relative_path``, with the folders leading to it."""
    path = input_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _add_nifti(input_dir: Path, relative_path: str, **kwargs) -> None:
    """Write a NIfTI file at ``relative_path``, with the folders leading to it."""
    path = input_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_nifti(path, **kwargs)


def _read(input_dir: Path, output_dir: Path) -> list[str]:
    """Return what stage 1's reader returns for ``input_dir``."""
    script = GenerateMorphologistGraphs()
    script.parse_args([str(input_dir), str(output_dir)])
    return script._get_input_files()


@pytest.mark.parametrize(
    "relative_path",
    [
        pytest.param(BIDS_T1, id="bids_session"),
        pytest.param("sub-01/anat/sub-01_T1w.nii", id="bids_uncompressed"),
        pytest.param("scans/s01/t1.nii.gz", id="plain_folders"),
        pytest.param("site A/s01.nii", id="folder_with_a_space"),
    ],
)
def test_nested_t1_file_is_returned_relative_to_input(dirs, relative_path):
    """REQ-INPUTKIND-BDRABCZUK-BEBFE65A7C55: a T1 file below a subfolder comes back as its relative path."""
    input_dir, output_dir = dirs
    _add_file(input_dir, relative_path)
    assert _read(input_dir, output_dir) == [relative_path]


@pytest.mark.parametrize(
    ("files", "links", "expected"),
    [
        pytest.param(["t1.nii.gz", "t2.nii"], [], ["t1.nii.gz", "t2.nii"], id="top_level_only"),
        pytest.param(
            ["top.nii.gz", "b/t.nii.gz", "a/t.nii"],
            [],
            ["a/t.nii", "b/t.nii.gz", "top.nii.gz"],
            id="sorted_nested_and_top_level",
        ),
        pytest.param(
            [BIDS_T1, "sub-01/ses-V1/anat/sub-01_ses-V1_T1w.json", "sub-01/ses-V1/anat/notes.txt"],
            [],
            [BIDS_T1],
            id="other_suffixes_left_out",
        ),
        pytest.param(
            ["sub-01/anat/sub-01_T1w.nii.gz", "sub-01/anat/sub-01_FLAIR.nii.gz"],
            [],
            ["sub-01/anat/sub-01_T1w.nii.gz"],
            id="t1w_needed_below_sub_folder",
        ),
        pytest.param(
            ["sub-01_FLAIR.nii.gz", "scans/sub-02_FLAIR.nii"],
            [],
            ["scans/sub-02_FLAIR.nii", "sub-01_FLAIR.nii.gz"],
            id="t1w_not_needed_outside_sub_folder",
        ),
        pytest.param(
            [
                BIDS_T1,
                "derivatives/morphologist-6.0/t1.nii.gz",
                "qc/t1.nii.gz",
                "sub-02/t1mri/default_acquisition/t1.nii.gz",
            ],
            [],
            [BIDS_T1],
            id="non_t1_folders_left_out",
        ),
        pytest.param(
            [BIDS_T1, ".cache/t1.nii.gz", "sub-01/ses-V1/anat/.sub-01_ses-V1_T1w.nii.gz"],
            [],
            [BIDS_T1],
            id="hidden_entries_left_out",
        ),
        pytest.param(["top.nii.gz", "scan.gz"], [], ["top.nii.gz"], id="bare_gz_left_out"),
        pytest.param(['"quoted.nii.gz'], [], ['"quoted.nii.gz'], id="quote_in_name_kept"),
        pytest.param(
            ["../elsewhere/anat/sub-02_T1w.nii.gz"],
            [("sub-02", "../elsewhere")],
            ["sub-02/anat/sub-02_T1w.nii.gz"],
            id="link_to_folder_followed",
        ),
        pytest.param(
            ["../elsewhere/t1.nii.gz"],
            [("a", "../elsewhere"), ("b", "../elsewhere")],
            ["a/t1.nii.gz", "b/t1.nii.gz"],
            id="two_links_to_one_folder_both_count",
        ),
        pytest.param(
            ["sub-01/anat/sub-01_T1w.nii.gz"],
            [("sub-01/anat/loop", "sub-01")],
            ["sub-01/anat/sub-01_T1w.nii.gz"],
            id="link_cycle_stays_unfollowed",
        ),
    ],
)
def test_reader_returns_the_shared_t1_search_result(dirs, files, links, expected):
    """REQ-INPUTKIND-BDRABCZUK-82DD555149FD: the reader's list is the T1 search's list, order and path text included."""
    input_dir, output_dir = dirs
    for relative_path in files:
        _add_file(input_dir, relative_path)
    for link, target in links:
        os.symlink(input_dir / target, input_dir / link)
    returned = _read(input_dir, output_dir)
    assert returned == expected
    assert returned == find_t1_files(str(input_dir))


@pytest.mark.parametrize(
    ("relative_path", "extra_args"),
    [
        pytest.param(BIDS_T1, (), id="default_format"),
        pytest.param(BIDS_T1, ("--bids",), id="bids_format"),
        pytest.param("site A;$(touch x)/s01.nii.gz", (), id="metacharacters_in_folder"),
    ],
)
def test_nested_file_reaches_cli_relative_to_input(dirs, relative_path, extra_args):
    """REQ-INPUTKIND-BDRABCZUK-22D56B17404E: one argument, the relative path unchanged, valid from the cwd."""
    input_dir, output_dir = dirs
    _add_nifti(input_dir, relative_path)
    _, mock_exec = _run_stage(input_dir, output_dir, extra_args=extra_args)
    mock_exec.assert_called_once()
    command = mock_exec.call_args.args[0]
    assert command[1:4] == [relative_path, str(output_dir), "--"]
    assert (Path(mock_exec.call_args.kwargs["cwd"]) / command[1]).is_file()


def test_cli_receives_files_in_reader_order(dirs):
    """REQ-INPUTKIND-BDRABCZUK-16D8CB79943A: the paths keep the reader's order, which is not participant_id order."""
    input_dir, output_dir = dirs
    in_reader_order = ["a/s3.nii.gz", "b/s2.nii.gz", "s1.nii.gz"]
    for relative_path in in_reader_order:
        _add_nifti(input_dir, relative_path)
    _, mock_exec = _run_stage(input_dir, output_dir)
    mock_exec.assert_called_once()
    assert mock_exec.call_args.args[0][1:4] == in_reader_order


@pytest.mark.parametrize(
    ("relative_path", "extra_args", "expected"),
    [
        pytest.param(BIDS_T1, (), {"sub-01_ses-V1_T1w": (1, "")}, id="bids_tree_default_format"),
        pytest.param("s01/t1.nii", (), {"t1": (1, "")}, id="plain_folder"),
        pytest.param(BIDS_T1, ("--bids",), {"sub-01": (1, "")}, id="bids_entities"),
        pytest.param("sub-01/v1.2/sub-01_T1w.nii.gz", ("--bids",), {"sub-01": (1, "")}, id="folder_with_a_dot"),
        pytest.param("sub-01/anat/scan.nii.gz", ("--bids",), {"scan": (0, "no_bids_subject")}, id="no_bids_file_name"),
    ],
)
def test_participant_id_of_nested_file_ignores_folders(dirs, relative_path, extra_args, expected):
    """REQ-INPUTKIND-BDRABCZUK-ECEC4169EBF5: the existing participant_id rules read the file name, no folder."""
    input_dir, output_dir = dirs
    _add_nifti(input_dir, relative_path)
    _run_stage(input_dir, output_dir, extra_args=extra_args)
    assert _qc_by_id(output_dir) == expected


@pytest.mark.parametrize(
    ("relative_paths", "extra_args", "expected"),
    [
        pytest.param(["a/t1.nii.gz", "b/t1.nii.gz"], (), {"t1": DUPLICATE}, id="default_format"),
        pytest.param(["t1.nii.gz", "b/t1.nii.gz"], (), {"t1": DUPLICATE}, id="top_level_and_nested"),
        pytest.param(
            ["sub-01/ses-1/anat/sub-01_T1w.nii.gz", "sub-01/ses-2/anat/sub-01_T1w.nii.gz"],
            ("--bids",),
            {"sub-01": DUPLICATE},
            id="bids_format",
        ),
    ],
)
def test_same_file_name_in_two_folders_is_a_duplicate(dirs, relative_paths, extra_args, expected):
    """REQ-NTUI-BDRABCZUK-6B48AF1E3A51, REQ-INPUTKIND-BDRABCZUK-ECEC4169EBF5: Morphologist gives both one subject."""
    input_dir, output_dir = dirs
    for relative_path in relative_paths:
        _add_nifti(input_dir, relative_path)
    result, mock_exec = _run_stage(input_dir, output_dir, extra_args=extra_args)
    assert _qc_by_id(output_dir) == expected
    assert result == 1
    mock_exec.assert_not_called()


def test_nested_file_with_qc_0_stays_out_of_cli(dirs):
    """REQ-INPUTKIND-BDRABCZUK-BD6B9FD94909: the unreadable file drops, the readable one reaches morphologist-cli."""
    input_dir, output_dir = dirs
    readable = "sub-01/anat/sub-01_T1w.nii.gz"
    unreadable = "sub-02/anat/sub-02_T1w.nii.gz"
    _add_nifti(input_dir, readable)
    _add_nifti(input_dir, unreadable, content=b"this is not a NIfTI image")
    _, mock_exec = _run_stage(input_dir, output_dir)
    assert _qc_by_id(output_dir)["sub-02_T1w"] == (0, "unreadable_header")
    mock_exec.assert_called_once()
    assert mock_exec.call_args.args[0][1:3] == [readable, str(output_dir)]


def test_bids_nested_bad_session_leaves_good_session_in_cli(dirs):
    """REQ-INPUTKIND-BDRABCZUK-BD6B9FD94909, REQ-NTUI-BDRABCZUK-7752385F171B: the qc of the file's own row counts.

    sub-01 has a qc 0 row (its bad session), yet the good session of sub-01 still reaches morphologist-cli.
    """
    input_dir, output_dir = dirs
    bad = "sub-01/ses-V1/anat/sub-01_ses-V1_T1w.nii.gz"
    good = "sub-01/ses-V2/anat/sub-01_ses-V2_T1w.nii.gz"
    _add_nifti(input_dir, bad, content=b"this is not a NIfTI image")
    _add_nifti(input_dir, good)
    _, mock_exec = _run_stage(input_dir, output_dir, extra_args=("--bids",))
    mock_exec.assert_called_once()
    command = mock_exec.call_args.args[0]
    assert good in command
    assert bad not in command
