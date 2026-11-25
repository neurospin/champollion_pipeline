"""Tests of the input kind detector: does one input path hold T1s, Morphologist outputs, neither or both.

The module under test is imported inside each test, through ``_module``, so a missing module fails each test
on its own instead of stopping collection of the whole file.
"""

import importlib
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.input_kind_fixtures import read_folders  # noqa: F401 - fixture

MODULE = "champollion_pipeline.utils.input_kind"
GRAPH_FOLDER = "default_analysis/folds/3.1"
HEAVY_PACKAGES = ("torch", "soma.aims", "numpy", "pandas")
# Stops a hung detection so the test fails instead of blocking the suite; it measures nothing and promises nothing.
HANG_GUARD_S = 60
RUNNING_AS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0
needs_permissions = pytest.mark.skipif(RUNNING_AS_ROOT, reason="root ignores directory permissions")

_NO_HEAVY_PACKAGE_SCRIPT = f"""
import importlib
import sys

module = importlib.import_module({MODULE!r})
kind = module.detect_input_kind(sys.argv[1])
print(kind.value, *[name for name in {HEAVY_PACKAGES!r} if name in sys.modules])
"""


def _module():
    """Return the module under test."""
    return importlib.import_module(MODULE)


def _kind(path: Path) -> str:
    """Return the value of the InputKind detected for ``path``."""
    module = _module()
    kind = module.detect_input_kind(str(path))
    assert isinstance(kind, module.InputKind)
    return kind.value


def _t1_files(path: Path) -> list[str]:
    """Return the T1 files found under ``path``."""
    return _module().find_t1_files(str(path))


def _scan(path: Path):
    """Return the scan of ``path``: its kind and its first T1 file and graph."""
    return _module().scan_input(str(path))


def _relative(folders: list[str], root: Path) -> list[str]:
    """Return ``folders`` relative to ``root``, in read order."""
    return [os.path.relpath(folder, root) for folder in folders]


class _ReversedListing:
    """Stand-in for an os.scandir iterator that lists entries in reverse name order."""

    def __init__(self, entries: list[os.DirEntry]) -> None:
        self._entries = entries

    def __enter__(self) -> Iterator[os.DirEntry]:
        return iter(self._entries)

    def __exit__(self, *_exc_info: object) -> None:
        return None

    def __iter__(self) -> Iterator[os.DirEntry]:
        return iter(self._entries)


def _reverse_listings(monkeypatch) -> None:
    """Make os.scandir list in reverse name order, so a walk that does not sort cannot pass by luck."""
    real_scandir = os.scandir

    def reversing_scandir(path="."):
        with real_scandir(path) as entries:
            return _ReversedListing(sorted(entries, key=lambda entry: entry.name, reverse=True))

    monkeypatch.setattr(os, "scandir", reversing_scandir)


@contextmanager
def _unreadable(folder: Path) -> Iterator[None]:
    """Remove every permission on ``folder`` for the time of the block."""
    folder.chmod(0)
    try:
        yield
    finally:
        folder.chmod(0o755)


def _input_with_blocked_folder(root: Path, through_link: bool) -> tuple[Path, Path]:
    """Return an input directory holding a folder named ``blocked``, and the real folder to make unreadable."""
    input_dir = root / "input"
    input_dir.mkdir()
    if through_link:
        target = root / "store"
        target.mkdir()
        (input_dir / "blocked").symlink_to(target, target_is_directory=True)
        return input_dir, target
    target = input_dir / "blocked"
    target.mkdir()
    return input_dir, target


def _detect_in_new_process(path: Path) -> list[str]:
    """Run detection in a new process, stopped after HANG_GUARD_S; return the kind then any heavy package loaded."""
    result = subprocess.run(
        [sys.executable, "-c", _NO_HEAVY_PACKAGE_SCRIPT, str(path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=HANG_GUARD_S,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


def _touch(root: Path, relative: str) -> Path:
    """Create an empty file at ``root/relative`` with its parent folders."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _morphologist_subject(root: Path, subject: str, prefix: str = "", graph_folder: str = GRAPH_FOLDER) -> None:
    """Lay out the real tree of one subject: raw T1, segmentation, sulcivoronoi and the two graphs."""
    analysis = f"{prefix}{subject}/t1mri/default_acquisition"
    _touch(root, f"{analysis}/{subject}.nii.gz")
    _touch(root, f"{analysis}/default_analysis/segmentation/brain_{subject}.nii.gz")
    _touch(root, f"{analysis}/default_analysis/folds/3.1/session/sulcivoronoi/Lsulcivoronoi_{subject}.nii.gz")
    for side in "LR":
        _touch(root, f"{analysis}/{graph_folder}/{side}{subject}.arg")


def test_top_level_nii_gz_reads_t1(tmp_path):
    _touch(tmp_path, "IXI002-Guys-0828-T1.nii.gz")

    assert _kind(tmp_path) == "t1"


@pytest.mark.parametrize(
    "relative",
    [
        "sub-01/ses-V1/anat/sub-01_ses-V1_T1w.nii.gz",
        "sub-01/anat/sub-01_T1w.nii.gz",
        "site-a/cohort/sub-01/ses-V1/anat/sub-01_ses-V1_T1w.nii.gz",
    ],
)
def test_nested_nii_gz_reads_t1(tmp_path, relative):
    _touch(tmp_path, relative)

    assert _kind(tmp_path) == "t1"


@pytest.mark.parametrize(
    ("prefix", "graph_folder"),
    [
        ("", GRAPH_FOLDER),
        ("subjects/", GRAPH_FOLDER),
        ("", "default_analysis/folds/3.3/base"),
        ("derivatives/morphologist-6.0/", GRAPH_FOLDER),
        ("derivatives/morphologist-2021/subjects/", GRAPH_FOLDER),
        ("derivatives/cat12/", GRAPH_FOLDER),
    ],
)
def test_morphologist_tree_reads_morphologist(tmp_path, prefix, graph_folder):
    _morphologist_subject(tmp_path, "sub-01", prefix, graph_folder)

    assert _kind(tmp_path) == "morphologist"


def test_find_t1_files_leaves_out_t1mri_tree(tmp_path):
    _morphologist_subject(tmp_path, "IXI002")

    assert _t1_files(tmp_path) == []


def test_subject_without_graph_keeps_tree_morphologist(tmp_path):
    _morphologist_subject(tmp_path, "IXI002")
    _touch(tmp_path, "IXI012/t1mri/default_acquisition/IXI012.nii.gz")

    assert _kind(tmp_path) == "morphologist"


def test_empty_directory_reads_neither(tmp_path):
    assert _kind(tmp_path) == "neither"


def test_unrelated_files_read_neither(tmp_path):
    _touch(tmp_path, "participants.tsv")
    _touch(tmp_path, "sub-01/notes.txt")

    assert _kind(tmp_path) == "neither"


@pytest.mark.parametrize("t1", ["sub-99.nii.gz", "rawdata/sub-99/anat/sub-99_T1w.nii.gz"])
def test_t1_beside_morphologist_tree_reads_both(tmp_path, t1):
    _morphologist_subject(tmp_path, "sub-01", "derivatives/morphologist-6.0/")
    _touch(tmp_path, t1)

    assert _kind(tmp_path) == "both"


def test_non_directory_reads_neither(tmp_path):
    plain_file = _touch(tmp_path, "sub-01.nii.gz")

    assert _kind(tmp_path / "missing") == "neither"
    assert _kind(plain_file) == "neither"


def test_find_t1_files_skips_derivatives_folder(tmp_path):
    _touch(tmp_path, "derivatives/cortical_tiles-2026/crops/sub-01.nii.gz")

    assert _t1_files(tmp_path) == []


def test_find_t1_files_skips_qc_folder(tmp_path):
    _touch(tmp_path, "qc/sub-01.nii.gz")

    assert _t1_files(tmp_path) == []


def test_find_t1_files_skips_dot_folder(tmp_path):
    _touch(tmp_path, ".snapshot/sub-01.nii.gz")

    assert _t1_files(tmp_path) == []


def test_hidden_file_is_no_t1(tmp_path):
    _touch(tmp_path, "._sub-01.nii.gz")

    assert _t1_files(tmp_path) == []


def test_other_gz_file_is_no_t1(tmp_path):
    _touch(tmp_path, "sub-01/func/sub-01_physio.tsv.gz")

    assert _t1_files(tmp_path) == []


def test_sidecar_file_is_no_t1(tmp_path):
    _touch(tmp_path, "sub-01.nii.gz.minf")

    assert _t1_files(tmp_path) == []


def test_find_t1_files_returns_sorted_relative_paths(tmp_path):
    _touch(tmp_path, "sub-02/ses-V1/anat/sub-02_ses-V1_T1w.nii.gz")
    _touch(tmp_path, "IXI002-Guys-0828-T1.nii.gz")
    _touch(tmp_path, "sub-01/ses-V2/anat/sub-01_ses-V2_T1w.nii.gz")

    assert _t1_files(tmp_path) == [
        "IXI002-Guys-0828-T1.nii.gz",
        "sub-01/ses-V2/anat/sub-01_ses-V2_T1w.nii.gz",
        "sub-02/ses-V1/anat/sub-02_ses-V1_T1w.nii.gz",
    ]


def test_symlinked_file_counts_under_its_own_path(tmp_path):
    target = _touch(tmp_path, "store/IXI002-Guys-0828-T1.nii.gz")
    input_dir = tmp_path / "view"
    input_dir.mkdir()
    (input_dir / "sub-IXI002.nii.gz").symlink_to(target)

    assert _t1_files(input_dir) == ["sub-IXI002.nii.gz"]


def test_symlinked_input_directory_is_read(tmp_path):
    target = tmp_path / "store"
    _morphologist_subject(target, "sub-01")
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    assert _kind(link) == "morphologist"


def test_broken_symlink_is_no_t1(tmp_path):
    (tmp_path / "sub-01.nii.gz").symlink_to(tmp_path / "missing.nii.gz")

    assert _t1_files(tmp_path) == []


def test_arg_in_dot_folder_is_no_graph(tmp_path):
    _morphologist_subject(tmp_path, "sub-01", ".snapshot/")
    _touch(tmp_path, "sub-01.nii.gz")

    assert _kind(tmp_path) == "t1"


@pytest.mark.parametrize("folder", ["derivatives", "qc", ".hidden", "t1mri"])
def test_folders_above_input_path_do_not_matter(tmp_path, folder):
    input_dir = tmp_path / folder / "dataset"
    _touch(input_dir, "sub-01.nii.gz")

    assert _kind(input_dir) == "t1"
    assert _t1_files(input_dir) == ["sub-01.nii.gz"]


def test_detection_loads_no_heavy_package(tmp_path):
    _touch(tmp_path, "sub-01.nii.gz")

    assert _detect_in_new_process(tmp_path) == ["t1"]


@pytest.mark.parametrize(
    "relative",
    [
        "IXI002.nii",
        "sub-01/ses-V1/anat/sub-01_ses-V1_T1w.nii",
        # No .arg file: a file below t1mri is T1 input (REQ-ARGDISC-BDRABCZUK-1786F42F6CE1)
        "IXI002/t1mri/default_acquisition/IXI002.nii",
    ],
)
def test_nii_file_is_selected_like_nii_gz(tmp_path, relative):
    _touch(tmp_path, relative)

    assert _t1_files(tmp_path) == [relative]


@pytest.mark.parametrize(
    "relative",
    [
        "derivatives/cortical_tiles-2026/IXI002.nii",
        "qc/IXI002.nii",
        ".snapshot/IXI002.nii",
        ".IXI002.nii",
        "sub-01/func/sub-01_task-rest_bold.nii",
    ],
)
def test_nii_file_is_left_out_like_nii_gz(tmp_path, relative):
    _touch(tmp_path, relative)

    assert _t1_files(tmp_path) == []


def test_lone_nii_file_reads_t1(tmp_path):
    _touch(tmp_path, "IXI002.nii")

    assert _kind(tmp_path) == "t1"


def test_t1_suffixes_constant_names_both_suffixes():
    assert set(_module().T1_SUFFIXES) == {".nii.gz", ".nii"}


def test_find_t1_files_reads_t1_suffixes(tmp_path, monkeypatch):
    _touch(tmp_path, "IXI002.nii.gz")
    _touch(tmp_path, "IXI012.nii")
    monkeypatch.setattr(_module(), "T1_SUFFIXES", (".nii",))

    assert _t1_files(tmp_path) == ["IXI012.nii"]


@pytest.mark.parametrize(
    "relative",
    [
        "sub-01/ses-V1/func/sub-01_ses-V1_task-rest_bold.nii.gz",
        "sub-01/ses-V1/dwi/sub-01_ses-V1_dwi.nii.gz",
        "sub-01/anat/sub-01_T2w.nii.gz",
        "sub-01/anat/sub-01_T1w_defaced.nii.gz",
        "sub-01/anat/sub-01.nii.gz",
    ],
)
def test_bids_file_without_t1w_is_no_t1(tmp_path, relative):
    _touch(tmp_path, relative)

    assert _t1_files(tmp_path) == []


def test_bids_tree_without_t1w_reads_neither(tmp_path):
    _touch(tmp_path, "sub-01/func/sub-01_task-rest_bold.nii.gz")

    assert _kind(tmp_path) == "neither"


@pytest.mark.parametrize(
    "relative",
    [
        "sub-01/anat/sub-01_T1w.nii.gz",
        "sub-01/ses-V1/anat/sub-01_ses-V1_acq-mprage_run-2_T1w.nii.gz",
        "sub-01/anat/sub-01_T1w.nii",
    ],
)
def test_bids_t1w_file_counts(tmp_path, relative):
    _touch(tmp_path, relative)

    assert _t1_files(tmp_path) == [relative]


@pytest.mark.parametrize("relative", ["sub-002.nii.gz", "sub-002.nii", "classic/sub-002.nii.gz"])
def test_flat_sub_file_counts_without_t1w(tmp_path, relative):
    _touch(tmp_path, relative)

    assert _t1_files(tmp_path) == [relative]


@pytest.mark.parametrize(
    "relative",
    ["IXI002-Guys-0828-T1.nii.gz", "site-a/scan_bold.nii.gz", "sub-01/anat/T1.nii.gz"],
)
def test_other_name_counts_without_t1w(tmp_path, relative):
    _touch(tmp_path, relative)

    assert _t1_files(tmp_path) == [relative]


def test_symlinked_folder_is_searched(tmp_path):
    _touch(tmp_path, "store/IXI002/IXI002.nii.gz")
    input_dir = tmp_path / "view"
    input_dir.mkdir()
    (input_dir / "cohort").symlink_to(tmp_path / "store", target_is_directory=True)

    assert _t1_files(input_dir) == ["cohort/IXI002/IXI002.nii.gz"]


def test_two_links_to_one_folder_both_count(tmp_path):
    _touch(tmp_path, "store/IXI002.nii.gz")
    input_dir = tmp_path / "view"
    input_dir.mkdir()
    (input_dir / "a").symlink_to(tmp_path / "store", target_is_directory=True)
    (input_dir / "b").symlink_to(tmp_path / "store", target_is_directory=True)

    assert _t1_files(input_dir) == ["a/IXI002.nii.gz", "b/IXI002.nii.gz"]


def test_symlinked_subject_folder_reads_morphologist(tmp_path):
    _morphologist_subject(tmp_path / "store", "IXI002")
    input_dir = tmp_path / "view"
    input_dir.mkdir()
    (input_dir / "IXI002").symlink_to(tmp_path / "store" / "IXI002", target_is_directory=True)

    assert _kind(input_dir) == "morphologist"


def test_symlink_cycle_lists_file_once(tmp_path):
    _touch(tmp_path, "a/IXI002.nii.gz")
    (tmp_path / "a" / "back").symlink_to(tmp_path, target_is_directory=True)

    assert _t1_files(tmp_path) == ["a/IXI002.nii.gz"]


def test_symlink_cycle_outside_input_lists_file_once(tmp_path):
    _touch(tmp_path, "store/IXI002.nii.gz")
    (tmp_path / "store" / "back").symlink_to(tmp_path / "store", target_is_directory=True)
    input_dir = tmp_path / "view"
    input_dir.mkdir()
    (input_dir / "cohort").symlink_to(tmp_path / "store", target_is_directory=True)

    assert _t1_files(input_dir) == ["cohort/IXI002.nii.gz"]


def test_symlink_cycle_does_not_stop_detection(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "back_one").symlink_to(tmp_path, target_is_directory=True)
    (tmp_path / "a" / "back_two").symlink_to(tmp_path, target_is_directory=True)

    assert _detect_in_new_process(tmp_path) == ["neither"]


@pytest.mark.parametrize(
    ("files", "first_t1"),
    [(["a/IXI002.nii.gz", "b.nii.gz"], "a/IXI002.nii.gz"), (["a.nii.gz", "b/IXI002.nii.gz"], "a.nii.gz")],
)
def test_walk_takes_files_and_folders_in_name_order(tmp_path, files, first_t1):
    for relative in files:
        _touch(tmp_path, relative)

    assert _scan(tmp_path).first_t1 == first_t1


def test_first_t1_is_first_in_name_order(tmp_path, monkeypatch):
    _touch(tmp_path, "IXI012.nii.gz")
    _touch(tmp_path, "IXI002.nii.gz")
    _reverse_listings(monkeypatch)

    assert _scan(tmp_path).first_t1 == "IXI002.nii.gz"


@pytest.mark.parametrize("relative", ["IXI002.nii.gz", "sub-01/anat/sub-01_T1w.nii.gz"])
def test_scan_names_first_t1(tmp_path, relative):
    assert _scan(tmp_path).first_t1 is None

    _touch(tmp_path, relative)

    assert _scan(tmp_path).first_t1 == relative


def test_scan_names_first_graph(tmp_path):
    assert _scan(tmp_path).first_graph is None

    _morphologist_subject(tmp_path, "IXI002", "subjects/")

    assert _scan(tmp_path).first_graph == f"subjects/IXI002/t1mri/default_acquisition/{GRAPH_FOLDER}/LIXI002.arg"


@pytest.mark.parametrize(
    "files",
    [[], ["IXI002.nii.gz"], ["IXI002/t1mri/a/LIXI002.arg"], ["IXI002/t1mri/a/LIXI002.arg", "z.nii.gz"]],
)
def test_scan_carries_detected_kind(tmp_path, files):
    for relative in files:
        _touch(tmp_path, relative)

    assert _scan(tmp_path).kind is _module().detect_input_kind(str(tmp_path))


def test_graph_before_first_t1_reads_both(tmp_path):
    _morphologist_subject(tmp_path, "IXI002")
    _touch(tmp_path, "z.nii.gz")

    assert _kind(tmp_path) == "both"


def test_no_t1mri_folder_is_read_after_first_graph(tmp_path, read_folders):  # noqa: F811 - imported fixture
    for subject in ("IXI002", "IXI012", "IXI013"):
        _morphologist_subject(tmp_path, subject)

    assert _kind(tmp_path) == "morphologist"

    read = _relative(read_folders, tmp_path)
    first_graph_folder = f"IXI002/t1mri/default_acquisition/{GRAPH_FOLDER}"
    assert read[read.index(first_graph_folder) + 1 :] == ["IXI012", "IXI013"]


@needs_permissions
@pytest.mark.parametrize("through_link", [False, True])
def test_unreadable_folder_warns_during_detection(tmp_path, pipeline_caplog, through_link):
    input_dir, target = _input_with_blocked_folder(tmp_path, through_link)

    with _unreadable(target):
        assert _kind(input_dir) == "neither"

    pipeline_caplog.assert_logged(logging.WARNING, "blocked")


@needs_permissions
@pytest.mark.parametrize("through_link", [False, True])
def test_detection_continues_past_unreadable_folder(tmp_path, through_link):
    input_dir, target = _input_with_blocked_folder(tmp_path, through_link)
    _touch(input_dir, "z/IXI002.nii.gz")

    with _unreadable(target):
        assert _kind(input_dir) == "t1"


@needs_permissions
@pytest.mark.parametrize("through_link", [False, True])
def test_unreadable_folder_warns_during_t1_search(tmp_path, pipeline_caplog, through_link):
    input_dir, target = _input_with_blocked_folder(tmp_path, through_link)

    with _unreadable(target):
        assert _t1_files(input_dir) == []

    pipeline_caplog.assert_logged(logging.WARNING, "blocked")


@needs_permissions
@pytest.mark.parametrize("through_link", [False, True])
def test_t1_search_continues_past_unreadable_folder(tmp_path, through_link):
    input_dir, target = _input_with_blocked_folder(tmp_path, through_link)
    _touch(input_dir, "z/IXI002.nii.gz")

    with _unreadable(target):
        assert _t1_files(input_dir) == ["z/IXI002.nii.gz"]


@needs_permissions
def test_unreadable_input_directory_reads_neither(tmp_path):
    input_dir = tmp_path / "input"
    _touch(input_dir, "IXI002.nii.gz")

    with _unreadable(input_dir):
        assert _kind(input_dir) == "neither"


def test_hidden_arg_is_no_graph(tmp_path):
    _touch(tmp_path, "IXI002/t1mri/default_acquisition/.LIXI002.arg")

    assert _kind(tmp_path) == "neither"


def test_broken_arg_link_is_no_graph(tmp_path):
    folder = tmp_path / "IXI002" / "t1mri" / "default_acquisition"
    folder.mkdir(parents=True)
    (folder / "LIXI002.arg").symlink_to(tmp_path / "missing.arg")

    assert _kind(tmp_path) == "neither"


def test_arg_link_to_file_is_a_graph(tmp_path):
    target = _touch(tmp_path, "store/LIXI002.arg")
    input_dir = tmp_path / "view"
    folder = input_dir / "IXI002" / "t1mri" / "default_acquisition"
    folder.mkdir(parents=True)
    (folder / "LIXI002.arg").symlink_to(target)

    assert _kind(input_dir) == "morphologist"


def test_t1_in_sourcedata_reads_t1(tmp_path):
    _touch(tmp_path, "sourcedata/IXI-T1/IXI002-Guys-0828-T1.nii.gz")

    assert _kind(tmp_path) == "t1"


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        (["anat/notes.txt", "derivatives/cat12/notes.txt"], [".", "derivatives", "derivatives/cat12", "anat"]),
        (["a/b/notes.txt", "a/derivatives/notes.txt"], [".", "a", "a/derivatives", "a/b"]),
    ],
)
def test_derivatives_is_entered_first(tmp_path, read_folders, files, expected):  # noqa: F811 - imported fixture
    for relative in files:
        _touch(tmp_path, relative)

    assert _kind(tmp_path) == "neither"
    assert _relative(read_folders, tmp_path) == expected


def test_derivatives_first_holds_in_any_listing_order(tmp_path, monkeypatch):
    _morphologist_subject(tmp_path, "IXI002", "derivatives/morphologist-6.0/")
    _touch(tmp_path, "anat/IXI012.nii.gz")
    _reverse_listings(monkeypatch)

    assert _kind(tmp_path) == "both"


@pytest.mark.parametrize(
    "t1",
    [
        "data/IXI012.nii.gz",
        "anat/IXI012.nii.gz",
        "BIDS/sub-012/anat/sub-012_T1w.nii.gz",
        "IXI012-Guys-0828-T1.nii.gz",
    ],
)
def test_t1_sorted_before_derivatives_reads_both(tmp_path, t1):
    _morphologist_subject(tmp_path, "IXI002", "derivatives/morphologist-6.0/")
    _touch(tmp_path, t1)

    assert _kind(tmp_path) == "both"


@pytest.mark.parametrize(
    ("with_graph", "raw_t1", "expected"),
    [
        (True, None, "morphologist"),
        (True, "rawdata/IXI012.nii.gz", "both"),
        (False, None, "neither"),
        (False, "rawdata/IXI012.nii.gz", "t1"),
    ],
)
def test_nifti_in_derivatives_is_no_t1_for_detection(tmp_path, with_graph, raw_t1, expected):
    _touch(tmp_path, "derivatives/cat12/IXI002/mri/wmIXI002.nii.gz")
    if with_graph:
        _morphologist_subject(tmp_path, "IXI002", "derivatives/morphologist-2021/subjects/")
    if raw_t1:
        _touch(tmp_path, raw_t1)

    assert _kind(tmp_path) == expected
    assert _scan(tmp_path).first_t1 == raw_t1
