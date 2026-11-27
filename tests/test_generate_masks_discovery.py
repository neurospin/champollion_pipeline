"""generate_masks finds each labelled subject's graph through graph discovery (TASK-ARGDISC-BDRABCZUK-0FAC91BE1C99).

Every check builds a synthetic folder tree (names only) with tests/morphologist_layout_fixtures.py in ``tmp_path``.
``aims`` is a fake reader that records the graph paths it is asked to read; the sulcus workers are fakes that
record the subjects they receive. No real graph is read.

Requirements (ids in the ledger, titles here):
    optional     the command line may omit --path_to_graph_supervised
    discovery    without --path_to_graph_supervised, the graphs of a subject and hemisphere come from graph discovery
    labelled     without --path_to_graph_supervised, a discovered graph without labelling set is left out
    several      several labelled discovered graphs of one subject and hemisphere: an error naming each
    override     --path_to_graph_supervised given: read below the subject folder, no graph discovery
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline import generate_masks as gm
from champollion_pipeline.file_indexer.pipeline_checks import FileIndexStageIndexer
from tests.morphologist_layout_fixtures import (
    build_classic_tree,
    build_morphologist_cli_tree,
    build_senior_tree,
)

CLASSIC_SUBJECTS = ("IXI001-Guys-0001", "IXI002-HH-0002")
CLASSIC_FOLDS = "t1mri/default_acquisition/default_analysis/folds/3.1"
SENIOR_ACQUISITIONS = ("3TV0", "3TV1", "3TV2", "3TV10")
SENIOR_SESSIONS = ("default_session_auto", "long_session_auto")
SENIOR_SUBJECTS = (("subjects_f", "sf0001"), ("subjects_h", "sh0001"))


class _FakeGraph:
    """A graph with a voxel size and no vertex."""

    def __getitem__(self, _key):
        return [2.0, 2.0, 2.0, 1.0]

    def vertices(self):
        return []


class _Recorder:
    """Fake aims that records each read path, and fake sulcus workers that record the subjects they receive."""

    def __init__(self):
        self.reads: list[str] = []
        self.subjects: set[str] = set()
        self.aims = MagicMock()
        self.aims.read.side_effect = self._read  # noqa: V101 - fake aims.read

    def _read(self, path):
        self.reads.append(str(Path(path).resolve()))
        return _FakeGraph()

    def compute_one_sulcus(self, _sf, per_subject_voxels, *_args):
        self.subjects.update(per_subject_voxels)
        return gm.RETURN_DICTIONARY["ok"]

    def compute_mask(self, **_kwargs):
        return None


def _config(tmp_path: Path, labeled_subjects_dir: Path, path_to_graph_supervised=None, sides=("L", "R")):
    return gm.RunConfig(
        sulci={"S.C."},
        sides=list(sides),
        mask_dir=str(tmp_path / "masks"),
        labeled_subjects_dir=str(labeled_subjects_dir),
        path_to_graph_supervised=path_to_graph_supervised,
        nb_subjects=-1,
        voxel_size=2.0,
        force=True,
    )


def _run(runner, config) -> _Recorder:
    """Drive runner over config with the fake reader and workers; return the recorder."""
    recorder = _Recorder()
    with (
        patch.object(gm, "aims", recorder.aims),
        patch.object(gm, "complete_sulci_name", lambda sulcus, side: f"{sulcus}_{side}"),
        patch.object(gm, "_compute_one_sulcus", recorder.compute_one_sulcus),
        patch.object(gm, "compute_mask", recorder.compute_mask),
    ):
        list(runner(config))
    return recorder


def _resolved(root: Path, *names: str) -> set[str]:
    return {str((root / name).resolve()) for name in names}


def _classic_labelled(root: Path) -> set[str]:
    return _resolved(
        root,
        *(
            f"{subject}/{CLASSIC_FOLDS}/default_session_auto/{hemi}{subject}_default_session_auto.arg"
            for subject in CLASSIC_SUBJECTS
            for hemi in ("L", "R")
        ),
    )


def _classic_unlabelled(root: Path) -> set[str]:
    return _resolved(
        root, *(f"{subject}/{CLASSIC_FOLDS}/{hemi}{subject}.arg" for subject in CLASSIC_SUBJECTS for hemi in ("L", "R"))
    )


def _senior_labelled_left(group: str, subject: str) -> list[str]:
    """Relative paths of the labelled left graphs of one SENIOR subject."""
    return [
        f"derivatives/{group}/{subject}/t1mri/{acquisition}/default_analysis/folds/3.1/{session}/"
        f"L{subject}_{session}.arg"
        for acquisition in SENIOR_ACQUISITIONS
        for session in SENIOR_SESSIONS
    ]


# --------------------------------------------------------------------------- optional


def test_command_line_without_path_to_graph_supervised_parses(tmp_path):
    script = gm.GenerateMasks()
    script.parse_args(["--labeled_subjects_dir", str(tmp_path / "db"), "--output_dir", str(tmp_path / "out")])
    assert script.args.path_to_graph_supervised is None


# --------------------------------------------------------------------------- discovery


def test_buffered_classic_layout_reads_discovered_graphs(tmp_path):
    root = build_classic_tree(tmp_path / "db")
    recorder = _run(gm.BufferedRunner(1), _config(tmp_path, root))
    assert set(recorder.reads) == _classic_labelled(root)


def test_buffered_classic_layout_names_subjects_by_discovery(tmp_path):
    root = build_classic_tree(tmp_path / "db")
    recorder = _run(gm.BufferedRunner(1), _config(tmp_path, root))
    assert recorder.subjects == set(CLASSIC_SUBJECTS)


def test_buffered_morphologist_cli_layout_reads_discovered_graph(tmp_path):
    root = build_morphologist_cli_tree(tmp_path / "db")
    recorder = _run(gm.BufferedRunner(1), _config(tmp_path, root))
    assert set(recorder.reads) == _resolved(
        root,
        "sub-01/ses-1/run-0/ana-0/folds/3.1/sul-0_auto/sub-01_ses-1_run-0_sul-0_hemi-L_auto.arg",
        "sub-01/ses-1/run-0/ana-0/folds/3.1/sul-0_auto/sub-01_ses-1_run-0_sul-0_hemi-R_auto.arg",
    )


def test_serial_classic_layout_reads_discovered_graphs(tmp_path):
    root = build_classic_tree(tmp_path / "db")
    recorder = _run(gm.SerialRunner(), _config(tmp_path, root))
    assert set(recorder.reads) == _classic_labelled(root)


# --------------------------------------------------------------------------- labelled


def test_unlabelled_classic_graphs_are_not_read(tmp_path):
    root = build_classic_tree(tmp_path / "db")
    recorder = _run(gm.BufferedRunner(1), _config(tmp_path, root))
    assert recorder.reads
    assert not set(recorder.reads) & _classic_unlabelled(root)


def test_subject_with_only_unlabelled_graphs_contributes_nothing(tmp_path):
    root = build_morphologist_cli_tree(tmp_path / "db")
    recorder = _run(gm.BufferedRunner(1), _config(tmp_path, root))
    assert recorder.subjects == {"sub-01"}
    assert not [path for path in recorder.reads if "/sub-02/" in path]


# --------------------------------------------------------------------------- several


def test_several_labelled_graphs_error_names_each(tmp_path):
    root = build_senior_tree(tmp_path / "db")
    with pytest.raises(Exception) as caught:
        _run(gm.BufferedRunner(1), _config(tmp_path, root, sides=("L",)))
    message = str(caught.value)
    assert any(
        all(path in message for path in _senior_labelled_left(group, subject)) for group, subject in SENIOR_SUBJECTS
    ), message


def test_several_labelled_graphs_are_not_read(tmp_path):
    root = build_senior_tree(tmp_path / "db")
    recorder = _Recorder()
    with (
        patch.object(gm, "aims", recorder.aims),
        patch.object(gm, "complete_sulci_name", lambda sulcus, side: f"{sulcus}_{side}"),
        patch.object(gm, "_compute_one_sulcus", recorder.compute_one_sulcus),
        pytest.raises(Exception) as caught,
    ):
        list(gm.BufferedRunner(1)(_config(tmp_path, root, sides=("L",))))
    assert any(subject in str(caught.value) for _group, subject in SENIOR_SUBJECTS), str(caught.value)
    assert recorder.reads == []


# --------------------------------------------------------------------------- override


def test_path_to_graph_supervised_given_overrides_discovery(tmp_path):
    root = build_classic_tree(tmp_path / "db")

    def no_discovery(*_args, **_kwargs):
        raise AssertionError("graph discovery used although --path_to_graph_supervised is given")

    with patch.object(FileIndexStageIndexer, "discover_graphs", no_discovery):
        recorder = _run(gm.BufferedRunner(1), _config(tmp_path, root, path_to_graph_supervised=CLASSIC_FOLDS))
    assert set(recorder.reads) == _classic_unlabelled(root)
