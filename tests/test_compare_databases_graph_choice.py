"""Graph choice in the databases-mode worker of ``src/compare.py`` (TASK-181).

``_get_subject_voxel_counts`` globs ``%(side)s%(subject)s*.arg`` and must not
depend on filesystem order when several graphs match (e.g. several labelling
sessions). ``glob.glob`` is patched to return an unsorted list so the
nondeterminism is observable, and ``soma.aims`` is replaced by a fake reader so
no real graph is needed.

Requirements: REQ-COMPARE-83 (sorted-first choice), REQ-COMPARE-84 (warning on
several matches), REQ-COMPARE-85 (single-match characterization).
"""

import glob
import re
import sys

import pytest

from compare import _get_subject_voxel_counts

SUBJECT = "subj-alpha"
GRAPH_DIR = "/fake/db/subj-alpha/t1mri/default_acquisition/default_analysis/folds/session"
PATH_A = f"{GRAPH_DIR}/Lsubj-alpha_sessA.arg"
PATH_B = f"{GRAPH_DIR}/Lsubj-alpha_sessB.arg"
PATH_C = f"{GRAPH_DIR}/Lsubj-alpha_sessC.arg"
# Deliberately not sorted, and the sorted-first path is not at index 0.
UNSORTED_MATCHES = [PATH_C, PATH_A, PATH_B]


class _FakeVertex:
    def __init__(self, name, n_ss):
        self._data = {"name": name, "aims_ss": [{f"v{i}": 1 for i in range(n_ss)}]}

    def get(self, key):
        return self._data.get(key)


class _FakeGraph:
    def __init__(self, vertices):
        self._vertices = vertices

    def vertices(self):
        return self._vertices


class _FakeAims:
    """Fake ``soma.aims``: returns a pre-registered graph per path, records reads."""

    def __init__(self, graphs):
        self.graphs = graphs
        self.read_paths = []

    def read(self, path):
        self.read_paths.append(str(path))
        return self.graphs[str(path)]


def _graph_with_count(n_ss):
    return _FakeGraph([_FakeVertex("S.C._left", n_ss)])


@pytest.fixture
def install(monkeypatch):
    """Patch glob results and ``soma.aims`` for one worker call."""

    def _install(matches):
        listed = list(matches)
        monkeypatch.setattr(glob, "glob", lambda *_a, **_k: list(listed))
        monkeypatch.setattr(glob, "iglob", lambda *_a, **_k: iter(list(listed)))
        graphs = {PATH_A: _graph_with_count(1), PATH_B: _graph_with_count(2), PATH_C: _graph_with_count(3)}
        fake = _FakeAims(graphs)
        monkeypatch.setattr(sys.modules["soma"], "aims", fake)
        return fake

    return _install


def _sub():
    return {"subject": SUBJECT, "dir": GRAPH_DIR, "graph_file": f"L{SUBJECT}*.arg"}


def _warning_lines(text):
    return [line for line in text.splitlines() if "WARNING" in line]


class TestMultipleGraphsSortedChoice:
    """REQ-COMPARE-83: several matches -> the sorted-first path is read."""

    def test_reads_sorted_first_graph_when_glob_order_is_unsorted(self, install):
        fake = install(UNSORTED_MATCHES)
        _get_subject_voxel_counts(_sub(), sys.path[0])
        assert fake.read_paths == [sorted(UNSORTED_MATCHES)[0]]

    def test_counts_come_from_sorted_first_graph(self, install):
        install(UNSORTED_MATCHES)
        name, counts = _get_subject_voxel_counts(_sub(), sys.path[0])
        assert name == SUBJECT
        assert counts == {"S.C._left": 1}  # PATH_A's graph, not PATH_C's (3)


class TestMultipleGraphsWarning:
    """REQ-COMPARE-84: several matches -> one WARNING line on stdout."""

    def test_one_warning_line_names_subject_selected_path_and_match_count(self, install, capsys):
        install(UNSORTED_MATCHES)
        _get_subject_voxel_counts(_sub(), sys.path[0])
        out = capsys.readouterr().out
        lines = _warning_lines(out)
        assert len(lines) == 1, f"expected one WARNING line on stdout, got: {out!r}"
        line = lines[0]
        assert SUBJECT in line
        assert PATH_A in line
        remainder = line
        for path in UNSORTED_MATCHES:
            remainder = remainder.replace(path, "")
        remainder = remainder.replace(SUBJECT, "")
        assert re.search(r"(?<!\d)3(?!\d)", remainder), f"match count 3 missing from: {line!r}"


class TestSingleGraphCharacterization:
    """REQ-COMPARE-85: exactly one match -> read it, no WARNING."""

    def test_single_match_reads_that_graph(self, install):
        fake = install([PATH_B])
        name, counts = _get_subject_voxel_counts(_sub(), sys.path[0])
        assert fake.read_paths == [PATH_B]
        assert (name, counts) == (SUBJECT, {"S.C._left": 2})

    def test_single_match_prints_no_warning(self, install, capsys):
        install([PATH_B])
        _get_subject_voxel_counts(_sub(), sys.path[0])
        captured = capsys.readouterr()
        assert _warning_lines(captured.out + captured.err) == []
