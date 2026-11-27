"""Proof that cortical_tiles 7b80c89 matches subject folders exactly (TASK-CORTILES-BDRABCZUK-96603E82E1F7).

DEF-CORTILES-BDRABCZUK-8A44DAD7F312: cortical_tiles once globbed ``{subject}*/`` and kept the first match, so
sub-1 could get the ICBM2009c transform of sub-10. The pipeline's temporary prefix guard is removed on this proof.
The tests run the folder-selection code of the cortical_tiles submodule on a synthetic sub-1 / sub-10 / sub-100
tree, with a fake ``aims`` that writes the path of the graph it was given into the transform file; no BrainVISA.

REQ-CORTILES-BDRABCZUK-A62DC3779FDB: each subject gets the graphs of its own folder only.
"""

import types
from pathlib import Path

import pytest
from cortical_tiles.brainvisa import generate_ICBM2009c_transforms as gt

pytestmark = pytest.mark.unit  # noqa: V107

SIDE = "R"
PATH_TO_GRAPH = "t1mri/folds"
SUBJECTS = ("sub-1", "sub-10", "sub-100")


def _graph(src: Path, folder: str, stem: str) -> Path:
    directory = src / folder / PATH_TO_GRAPH
    directory.mkdir(parents=True, exist_ok=True)
    graph = directory / f"{SIDE}{stem}.arg"
    graph.write_text("")
    return graph


@pytest.fixture  # noqa: V103 - used through usefixtures
def fake_aims(monkeypatch):
    """The transform file of a graph holds that graph's path."""

    def write(transform, path):
        Path(path).write_text(str(transform))

    manip = types.SimpleNamespace(getICBM2009cTemplateTransform=lambda graph: graph)
    monkeypatch.setattr(gt, "aims", types.SimpleNamespace(read=lambda path: path, write=write, GraphManip=manip))


def _step(tmp_path: Path, **kwargs):
    src = tmp_path / "graphs"
    src.mkdir(exist_ok=True)
    return gt.GraphGenerateTransform(
        src_dir=str(src),
        transform_dir=str(tmp_path / "transforms"),
        side=SIDE,
        parallel=False,
        bids=False,
        qc_path="",
        skip_dir=str(tmp_path / "skip"),
        **{"path_to_graph": None, **kwargs},
    ), src


def _transform_text(tmp_path: Path, subject: str) -> str:
    return (tmp_path / "transforms" / SIDE / f"{SIDE}transform_to_ICBM2009c_{subject}.trm").read_text()


class TestExactSubjectFolders:
    def test_subject_folders_hold_only_the_own_folder(self, tmp_path):
        """REQ-CORTILES-BDRABCZUK-A62DC3779FDB: _subject_folders of sub-1 never lists sub-10 or sub-100."""
        step, src = _step(tmp_path, path_to_graph=PATH_TO_GRAPH)
        for name in SUBJECTS:
            _graph(src, name, name)

        for name in SUBJECTS:
            assert step._subject_folders(name) == [str(src / name)]

    @pytest.mark.usefixtures("fake_aims")
    def test_path_to_graph_glob_gives_each_subject_its_own_transform(self, tmp_path):
        """REQ-CORTILES-BDRABCZUK-A62DC3779FDB: each transform comes from the graph of its own folder."""
        step, src = _step(tmp_path, path_to_graph=PATH_TO_GRAPH)
        graphs = {name: _graph(src, name, name) for name in SUBJECTS}

        for name in SUBJECTS:
            assert step.generate_one_transform(name) == (name, None)

        for name in SUBJECTS:
            assert _transform_text(tmp_path, name) == str(graphs[name])

    @pytest.mark.usefixtures("fake_aims")
    def test_graphs_map_gives_each_subject_its_own_transform(self, tmp_path):
        """REQ-CORTILES-BDRABCZUK-A62DC3779FDB: the --graphs path (a per-subject map) selects the own folder only."""
        src = tmp_path / "graphs"
        graphs = {name: _graph(src, name, name) for name in SUBJECTS}
        listing = {name: {SIDE: [graphs[name]]} for name in SUBJECTS}
        step, _ = _step(tmp_path, graphs=listing)

        for name in SUBJECTS:
            assert step.generate_one_transform(name) == (name, None)

        for name in SUBJECTS:
            assert _transform_text(tmp_path, name) == str(graphs[name])

    @pytest.mark.usefixtures("fake_aims")
    def test_subject_without_graph_never_takes_a_longer_id_graph(self, tmp_path):
        """REQ-CORTILES-BDRABCZUK-A62DC3779FDB: sub-1 has no graph, sub-10 has one: sub-1 gets none."""
        step, src = _step(tmp_path, path_to_graph=PATH_TO_GRAPH)
        _graph(src, "sub-10", "sub-10")

        subject, error = step.generate_one_transform("sub-1")

        assert subject == "sub-1"
        assert "No graph file" in error
