"""Tests of the Morphologist subjects folder resolver and of where stage 1 writes the subjects.

The module under test is imported inside each test, through ``_module``, so a function that does not exist yet fails
each test on its own instead of stopping collection of the whole file.

A real tree has one of two shapes: stage 1 writes ``<root>/derivatives/morphologist-6.0/<sub>``, older trees carry a
``subjects`` level (``/neurospin/psy/ixi/derivatives/morphologist-2021/subjects``). The fixtures copy these shapes.
"""

import importlib
import logging
import os
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from champollion_pipeline.derivatives_layout import compute_run_log_derivatives_dir
from tests.test_input_kind import _unreadable

MODULE = "champollion_pipeline.utils.input_kind"
ACQUISITION = "default_acquisition"
GRAPH_FOLDER = f"t1mri/{ACQUISITION}/default_analysis/folds/3.1"
MORPHOLOGIST_6 = "derivatives/morphologist-6.0"
MORPHOLOGIST_2021 = "derivatives/morphologist-2021"
RUNNING_AS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0
needs_permissions = pytest.mark.skipif(RUNNING_AS_ROOT, reason="root ignores directory permissions")
# The functions built on the shared walk: an invariant of the walk holds for each.
WALKERS = ("detect_input_kind", "find_t1_files", "derived_volumes", "resolve_subjects_dir")
# Folder names every walker enters: find_t1_files rightly stays out of the folders that cannot hold a T1 file.
WALKED_NAMES = st.from_regex(r"[a-z][a-z0-9_-]{0,8}", fullmatch=True).filter(
    lambda name: name not in {"derivatives", "qc", "t1mri"}
)
walker_property = settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
AMBIGUOUS_TREES = [
    pytest.param((f"{MORPHOLOGIST_2021}/subjects", MORPHOLOGIST_6), id="two_folders"),
    pytest.param((MORPHOLOGIST_6, "archive/subjects"), id="sorted_order_is_not_walk_order"),
    pytest.param((MORPHOLOGIST_6, "archive/subjects", f"{MORPHOLOGIST_2021}/subjects"), id="three_folders"),
]
# What sits beside the input's own t1mri folder: a folder of subjects, then a subject directly in the input.
STRAYS = [
    pytest.param("backup", "s1", id="stray_folder_sorts_first"),
    pytest.param("zzz", "s1", id="stray_folder_sorts_last"),
    pytest.param(".", "aaa", id="direct_subject_sorts_first"),
    pytest.param(".", "zzz", id="direct_subject_sorts_last"),
]


def _module():
    """Return the module under test."""
    return importlib.import_module(MODULE)


def _resolve(path: Path) -> str | None:
    """Return the subjects folder resolved from ``path``."""
    return _module().resolve_subjects_dir(str(path))


def _stage1_dir(root: Path) -> str:
    """Return where stage 1 writes the subjects below the output root ``root``."""
    return _module().stage1_subjects_dir(str(root))


def _touch(root: Path, relative: str) -> Path:
    """Create the empty file ``relative`` below ``root``, with its folders."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _graph(subjects_dir: Path, subject: str) -> Path:
    """Create one hemisphere graph of ``subject`` in ``subjects_dir``."""
    return _touch(subjects_dir, f"{subject}/{GRAPH_FOLDER}/L{subject}.arg")


def _graphs_in(root: Path, folders: tuple[str, ...]) -> list[str]:
    """Create one subject with a graph in each of ``folders``, and return the folders as sorted paths."""
    for folder in folders:
        _graph(root / folder, "IXI002")
    return sorted(str(root / folder) for folder in folders)


def _outcome(path: Path) -> object:
    """Return what the resolver gives for ``path``, an ambiguity included; any other error fails the test."""
    resolve = _module().resolve_subjects_dir
    ambiguous = _module().AmbiguousSubjectsFolders
    try:
        return resolve(str(path))
    except ambiguous as error:
        return error


def _ambiguity(path: Path):
    """Return the ambiguity the resolver raises for ``path``."""
    with pytest.raises(_module().AmbiguousSubjectsFolders) as caught:
        _resolve(path)
    return caught.value


def _input_with_blocked_tree(root: Path, through_link: bool) -> tuple[Path, Path]:
    """Return an input holding a folder named ``blocked`` with a subjects folder inside, and the folder to block."""
    input_dir = root / "input"
    input_dir.mkdir()
    target = root / "store" if through_link else input_dir / "blocked"
    _graph(target / "tree", "IXI002")
    if through_link:
        (input_dir / "blocked").symlink_to(target, target_is_directory=True)
    return input_dir, target


@pytest.fixture
def read_folders(monkeypatch) -> list[str]:
    """Record, in read order, each folder passed to os.scandir."""
    folders: list[str] = []
    real_scandir = os.scandir

    def recording_scandir(path="."):
        if not isinstance(path, int):  # TemporaryDirectory cleanup reads by file descriptor
            folders.append(os.fspath(path))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", recording_scandir)
    return folders


def test_dataset_root_gives_morphologist_folder(tmp_path):
    subjects_dir = tmp_path / MORPHOLOGIST_6
    _graph(subjects_dir, "sub-01")
    _graph(subjects_dir, "sub-02")
    _touch(subjects_dir, "input_qc.tsv")
    _touch(subjects_dir, ".log/morphologist_20261106-120000.log")
    _touch(tmp_path, "derivatives/champollion_V1/canonical_25/embeddings/full.csv")
    _touch(tmp_path, "rawdata/sub-01/anat/sub-01_T1w.nii.gz")

    assert _resolve(tmp_path) == str(subjects_dir)


def test_resolved_folder_is_where_stage1_writes(tmp_path):
    subjects_dir = Path(_stage1_dir(tmp_path))
    _graph(subjects_dir, "sub-01")

    assert _resolve(tmp_path) == str(subjects_dir)


def test_dataset_root_gives_legacy_subjects_level(tmp_path):
    subjects_dir = tmp_path / MORPHOLOGIST_6 / "subjects"
    _graph(subjects_dir, "IXI002")

    assert _resolve(tmp_path) == str(subjects_dir)


def test_dataset_root_gives_morphologist_2021_subjects(tmp_path):
    subjects_dir = tmp_path / MORPHOLOGIST_2021 / "subjects"
    _graph(subjects_dir, "IXI002")
    _touch(tmp_path, "rawdata/IXI002-Guys-0828-T1.nii.gz")

    assert _resolve(tmp_path) == str(subjects_dir)


def test_morphologist_folder_gives_legacy_subjects_level(tmp_path):
    morphologist_dir = tmp_path / MORPHOLOGIST_6
    _graph(morphologist_dir / "subjects", "IXI002")

    assert _resolve(morphologist_dir) == str(morphologist_dir / "subjects")


@pytest.mark.parametrize(
    "folder",
    [MORPHOLOGIST_6, f"{MORPHOLOGIST_6}/subjects", f"{MORPHOLOGIST_2021}/subjects", "graphs"],
)
def test_folder_of_subjects_gives_itself(tmp_path, folder):
    subjects_dir = tmp_path / folder
    _touch(subjects_dir, f"IXI001/t1mri/{ACQUISITION}/IXI001.nii.gz")
    _graph(subjects_dir, "IXI002")
    _graph(subjects_dir, "IXI012")

    assert _resolve(subjects_dir) == str(subjects_dir)


def test_subject_named_subjects_is_a_subject(tmp_path):
    _graph(tmp_path, "subjects")

    assert _resolve(tmp_path) == str(tmp_path)


@pytest.mark.parametrize("input_below_root", ["", MORPHOLOGIST_6], ids=["dataset_root", "subjects_folder"])
def test_session_level_is_no_subject_level(tmp_path, input_below_root):
    subjects_dir = tmp_path / MORPHOLOGIST_6
    _touch(subjects_dir, f"sub-01/ses-1/{GRAPH_FOLDER}/Lsub-01.arg")

    assert _resolve(tmp_path / input_below_root) == str(subjects_dir)


@pytest.mark.parametrize(
    ("t1_file", "subjects_folder"),
    [
        ("a.nii.gz", "."),
        ("aaa/a.nii.gz", "."),
        ("aaa/a.nii.gz", "zzz/morphologist"),
    ],
)
def test_t1_sorted_first_does_not_hide_graph(tmp_path, t1_file, subjects_folder):
    subjects_dir = tmp_path / subjects_folder
    _touch(tmp_path, t1_file)
    graph = _graph(subjects_dir, "zzz-subject")

    # REQ-ARGDISC-BDRABCZUK-80E144705B28: the only graph sorts after the first T1 file, so the input reads both
    scan = _module().scan_input(str(tmp_path))
    assert scan.kind is _module().InputKind.BOTH
    assert scan.first_graph == os.path.relpath(graph, tmp_path)

    # REQ-INPUTKIND-BDRABCZUK-542640BE44B3: the resolver still finds the subjects folder of that graph
    assert _resolve(tmp_path) == str(subjects_dir)


def _no_input(root: Path) -> Path:
    """Return a path that does not exist."""
    return root / "missing"


def _file_input(root: Path) -> Path:
    """Return a path that is a regular file."""
    return _touch(root, "IXI002.arg")


def _empty_input(root: Path) -> Path:
    """Return an empty folder."""
    return root


def _t1_input(root: Path) -> Path:
    """Return a folder of T1 files."""
    _touch(root, "IXI002.nii.gz")
    _touch(root, "nested/IXI012.nii.gz")
    return root


def _no_folds_input(root: Path) -> Path:
    """Return a subjects folder whose t1mri folders hold no folds folder, as openbhb/ixi morphologist-6.0/subjects."""
    subjects_dir = root / MORPHOLOGIST_6 / "subjects"
    _touch(subjects_dir, f"IXI002/t1mri/{ACQUISITION}/IXI002.nii.gz")
    _touch(subjects_dir, f"IXI002/t1mri/{ACQUISITION}/default_analysis/segmentation/brain_IXI002.nii.gz")
    return subjects_dir


def _arg_outside_t1mri_input(root: Path) -> Path:
    """Return a folder whose only .arg file sits outside any t1mri folder."""
    _touch(root, "IXI002/other/LIXI002.arg")
    return root


@pytest.mark.parametrize(
    "build_input",
    [_no_input, _file_input, _empty_input, _t1_input, _no_folds_input, _arg_outside_t1mri_input],
)
def test_no_graph_gives_none(tmp_path, build_input: Callable[[Path], Path]):
    assert _resolve(build_input(tmp_path)) is None


@pytest.mark.parametrize("graph_folder", [GRAPH_FOLDER, f"ses-1/{GRAPH_FOLDER}"])
def test_subject_folder_gives_none(tmp_path, graph_folder):
    subject_dir = tmp_path / "subjects" / "IXI002"
    _touch(subject_dir, f"{graph_folder}/LIXI002.arg")

    assert _resolve(subject_dir) is None


def test_no_folder_is_read_after_first_graph(tmp_path, read_folders):
    for subject in ("IXI002", "IXI012", "IXI013"):
        _graph(tmp_path, subject)

    assert _resolve(tmp_path) == str(tmp_path)

    assert read_folders[-1] == str(tmp_path / "IXI002" / GRAPH_FOLDER)


def test_each_subjects_folder_is_left_at_its_first_graph(tmp_path, read_folders):
    older = tmp_path / MORPHOLOGIST_2021 / "subjects"
    newer = tmp_path / MORPHOLOGIST_6
    for subjects_dir, subjects in ((older, ("IXI002", "IXI012")), (newer, ("sub-01", "sub-02"))):
        for subject in subjects:
            _graph(subjects_dir, subject)

    _outcome(tmp_path)

    assert str(older / "IXI002" / GRAPH_FOLDER) in read_folders
    assert str(newer / "sub-01" / GRAPH_FOLDER) in read_folders
    assert str(older / "IXI012") not in read_folders
    assert str(newer / "sub-02") not in read_folders


@pytest.mark.parametrize(
    "folders",
    [
        pytest.param((f"{MORPHOLOGIST_2021}/subjects", MORPHOLOGIST_6), id="two_morphologist_versions"),
        pytest.param((f"{MORPHOLOGIST_6}/subjects", f"{MORPHOLOGIST_2021}/subjects"), id="two_legacy_levels"),
        pytest.param(("archive/subjects", MORPHOLOGIST_6), id="outside_derivatives"),
    ],
)
def test_two_subjects_folders_give_no_silent_pick(tmp_path, folders):
    _graphs_in(tmp_path, folders)

    _ambiguity(tmp_path)


def test_t1_sorted_first_does_not_hide_ambiguity(tmp_path):
    _touch(tmp_path, "a.nii.gz")
    _graphs_in(tmp_path, ("zzz/tree_a", "zzz/tree_b"))

    # REQ-ARGDISC-BDRABCZUK-80E144705B28: the graphs sort after the first T1 file, so the input reads both
    assert _module().scan_input(str(tmp_path)).kind is _module().InputKind.BOTH

    # REQ-INPUTKIND-BDRABCZUK-313AEEB1E572: the two subjects folders stay an ambiguity for the resolver
    _ambiguity(tmp_path)


@pytest.mark.parametrize("folders", AMBIGUOUS_TREES)
def test_ambiguity_names_candidates_in_sorted_order(tmp_path, folders):
    candidates = _graphs_in(tmp_path, folders)

    report = str(_ambiguity(tmp_path))

    assert all(candidate in report for candidate in candidates), report
    positions = [report.index(candidate) for candidate in candidates]
    assert positions == sorted(positions), report


def test_ambiguity_is_a_value_error():
    assert issubclass(_module().AmbiguousSubjectsFolders, ValueError)


def test_ambiguity_carries_the_input_path(tmp_path):
    _graphs_in(tmp_path, (f"{MORPHOLOGIST_2021}/subjects", MORPHOLOGIST_6))

    assert _ambiguity(tmp_path).path == str(tmp_path)


@pytest.mark.parametrize("folders", AMBIGUOUS_TREES)
def test_ambiguity_carries_sorted_candidates(tmp_path, folders):
    candidates = _graphs_in(tmp_path, folders)

    assert _ambiguity(tmp_path).candidates == tuple(candidates)


@pytest.mark.parametrize(
    ("link_name", "target_name", "level_below"),
    [
        pytest.param("morphologist-latest", "morphologist-6.0", "", id="link_to_the_subjects_folder"),
        pytest.param("latest", "morphologist-2021", "subjects", id="link_above_the_subjects_folder"),
    ],
)
def test_link_and_target_are_two_candidates(tmp_path, link_name, target_name, level_below):
    target = tmp_path / "derivatives" / target_name
    link = tmp_path / "derivatives" / link_name
    _graph(target / level_below, "IXI002")
    link.symlink_to(target, target_is_directory=True)

    expected = tuple(sorted((str(target / level_below), str(link / level_below))))

    assert _ambiguity(tmp_path).candidates == expected


@pytest.mark.parametrize(("stray_folder", "stray_subject"), STRAYS)
def test_own_t1mri_gives_none_whatever_sits_beside(tmp_path, stray_folder, stray_subject):
    input_dir = tmp_path / "subjects" / "IXI002"
    _touch(input_dir, f"{GRAPH_FOLDER}/LIXI002.arg")
    _graph(input_dir / stray_folder, stray_subject)

    assert _outcome(input_dir) is None


@pytest.mark.parametrize(("stray_folder", "stray_subject"), STRAYS)
def test_own_session_t1mri_gives_none_whatever_sits_beside(tmp_path, stray_folder, stray_subject):
    input_dir = tmp_path / "subjects" / "IXI002"
    _touch(input_dir, f"ses-1/{GRAPH_FOLDER}/LIXI002.arg")
    _graph(input_dir / stray_folder, stray_subject)

    assert _outcome(input_dir) is None


def test_resolver_follows_a_link_to_a_subject(tmp_path):
    store = tmp_path / "store"
    input_dir = tmp_path / "input"
    _graph(store, "IXI002")
    input_dir.mkdir()
    (input_dir / "IXI002").symlink_to(store / "IXI002", target_is_directory=True)

    assert _resolve(input_dir) == str(input_dir)


def test_resolver_follows_a_link_to_a_subjects_folder(tmp_path):
    store = tmp_path / "store"
    dataset = tmp_path / "dataset"
    _graph(store, "IXI002")
    (dataset / "derivatives").mkdir(parents=True)
    (dataset / MORPHOLOGIST_6).symlink_to(store, target_is_directory=True)

    assert _resolve(dataset) == str(dataset / MORPHOLOGIST_6)


def test_resolver_leaves_a_link_cycle_unfollowed(tmp_path):
    (tmp_path / "aaa").mkdir()
    (tmp_path / "aaa" / "back").symlink_to(tmp_path, target_is_directory=True)
    _graph(tmp_path / "zzz", "IXI002")

    assert _outcome(tmp_path) == str(tmp_path / "zzz")


def test_graph_below_a_dot_folder_is_not_counted(tmp_path):
    _graph(tmp_path / ".snapshot" / "old", "IXI002")

    assert _resolve(tmp_path) is None

    _graph(tmp_path / "tree", "IXI012")

    assert _outcome(tmp_path) == str(tmp_path / "tree")


def test_hidden_arg_file_is_not_counted(tmp_path):
    _touch(tmp_path, f"IXI002/{GRAPH_FOLDER}/.LIXI002.arg")

    assert _resolve(tmp_path) is None


@needs_permissions
@pytest.mark.parametrize("through_link", [False, True], ids=["folder", "link"])
def test_resolver_goes_on_past_an_unreadable_folder(tmp_path, through_link):
    input_dir, target = _input_with_blocked_tree(tmp_path, through_link)
    _graph(input_dir / "tree", "IXI012")

    with _unreadable(target):
        assert _outcome(input_dir) == str(input_dir / "tree")


@needs_permissions
@pytest.mark.parametrize("through_link", [False, True], ids=["folder", "link"])
def test_resolver_warns_about_an_unreadable_folder(tmp_path, pipeline_caplog, through_link):
    input_dir, target = _input_with_blocked_tree(tmp_path, through_link)

    with _unreadable(target):
        assert _resolve(input_dir) is None

    pipeline_caplog.assert_logged(logging.WARNING, "blocked")


@pytest.mark.parametrize("subject", ["IXI002", "zzz-subject"], ids=["sorts_first", "sorts_last"])
@pytest.mark.parametrize("session", ["", "ses-1/"], ids=["no_session", "session"])
def test_subjects_folder_input_is_not_ambiguous(tmp_path, subject, session):
    _touch(tmp_path, f"{subject}/{session}{GRAPH_FOLDER}/L{subject}.arg")
    _graph(tmp_path / "subjects", "IXI012")

    assert _outcome(tmp_path) == str(tmp_path)


@pytest.mark.parametrize("subject", ["IXI002", "zzz-subject"], ids=["sorts_first", "sorts_last"])
def test_nested_subjects_folder_is_no_candidate(tmp_path, subject):
    outer = tmp_path / MORPHOLOGIST_6
    _graph(outer, subject)
    _graph(outer / "subjects", "IXI012")

    assert _outcome(tmp_path) == str(outer)


@needs_permissions
@pytest.mark.parametrize("walker", WALKERS)
@walker_property
@given(name=WALKED_NAMES)
def test_each_walker_warns_about_an_unreadable_folder(pipeline_caplog, walker, name):
    with TemporaryDirectory() as input_dir:
        blocked = Path(input_dir) / name
        blocked.mkdir()

        with _unreadable(blocked):
            getattr(_module(), walker)(input_dir)

        pipeline_caplog.assert_logged(logging.WARNING, str(blocked))


@pytest.mark.parametrize("walker", WALKERS)
@walker_property
@given(name=WALKED_NAMES)
def test_each_walker_leaves_a_link_cycle_unfollowed(read_folders, walker, name):
    with TemporaryDirectory() as input_dir:
        link = Path(input_dir) / name / "back"
        link.parent.mkdir()
        link.symlink_to(input_dir, target_is_directory=True)

        getattr(_module(), walker)(input_dir)

        assert str(link.parent) in read_folders
        assert not [folder for folder in read_folders if folder.startswith(str(link))]


def test_stage1_subjects_dir_of_output_root(tmp_path):
    assert _stage1_dir(tmp_path) == str(tmp_path / MORPHOLOGIST_6)


def test_stage1_subjects_dir_is_the_run_log_parent(tmp_path):
    assert _stage1_dir(tmp_path) == compute_run_log_derivatives_dir("morphologist", str(tmp_path))


def test_stage1_subjects_dir_before_stage1_ran(tmp_path):
    output_root = tmp_path / "not_written_yet"

    assert _stage1_dir(output_root) == str(output_root / MORPHOLOGIST_6)
    assert not output_root.exists()
