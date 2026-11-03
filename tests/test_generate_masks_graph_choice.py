"""Supervised-graph choice in generate_masks (TASK-PATTERNS-BDRABCZUK-3E53EFBFDCA8).

``_load_and_extract_subject`` must pick the graph of one subject and side by an
exact rule: the file name is the side and subject followed by '_' or '.', glob
characters and '%' in the directory, subject and graph path are literal, and
anything but exactly one match is an error (none: the subject is skipped).
Synthetic empty files on tmp_path stand in for graphs; ``aims`` is a fake reader.
"""

import types
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline import generate_masks as gm

GRAPH_PATH = "t1mri/t1/default_analysis/folds/3.3/base"
VT = (2.0, 2.0, 2.0)


class _FakeGraph:
    def __getitem__(self, _key):
        return [2.0, 2.0, 2.0, 1.0]

    def vertices(self):
        return []


def _fake_aims():
    aims = MagicMock()
    aims.read.side_effect = lambda _path: _FakeGraph()  # noqa: V101
    aims.GraphManip.getICBM2009cTemplateTransform.return_value = MagicMock()
    return aims


def _touch(directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text("")
    return str(directory / name)


def _run_runner(tmp_path, subject, db_dir, side="L"):
    """Drive BufferedRunner for one listed subject; return (aims fake, result of the load worker)."""
    seen = {}
    aims = _fake_aims()
    subjects_mod = types.ModuleType("subjects")
    subjects_mod.get_all_subjects_as_dictionary = lambda dirs, patterns, s: [
        {"subject": subject, "side": s, "dir": dirs[0], "graph_file": patterns[0] % {"subject": subject, "side": s}}
    ]
    subjects_mod.select_subjects_int_if_list_of_dict = lambda subs, _all, _nb: subs
    real_load = gm._load_and_extract_subject

    def load(sub, *args):
        seen["result"] = real_load(sub, *args)
        return seen["result"]

    config = gm.RunConfig(
        sulci={"S.C."},
        sides=[side],
        mask_dir=str(tmp_path / "masks"),
        labeled_subjects_dir=str(db_dir),
        path_to_graph_supervised=GRAPH_PATH,
        nb_subjects=-1,
        voxel_size=2.0,
        force=True,
    )
    with (
        patch.object(gm, "aims", aims),
        patch.object(gm, "get_all_subjects_as_dictionary", subjects_mod.get_all_subjects_as_dictionary),
        patch.object(gm, "select_subjects_int_if_list_of_dict", subjects_mod.select_subjects_int_if_list_of_dict),
        patch.object(gm, "complete_sulci_name", lambda s, side: f"{s}_left"),
        patch.object(gm, "_load_and_extract_subject", load),
        patch.object(gm, "_compute_one_sulcus", lambda *a: gm.RETURN_DICTIONARY["ok"]),
    ):
        list(gm.BufferedRunner(1)(config))
    return aims, seen["result"]


class TestSeveralMatches:
    def test_error_names_every_candidate(self, tmp_path):
        a = _touch(tmp_path, "Lsub01_sessA.arg")
        b = _touch(tmp_path, "Lsub01_sessB.arg")
        sub = {"subject": "sub01", "side": "L", "dir": str(tmp_path), "graph_file": "Lsub01*.arg"}
        with patch.object(gm, "aims", _fake_aims()), pytest.raises(RuntimeError) as err:
            gm._load_and_extract_subject(sub, set(), VT)
        assert a in str(err.value)
        assert b in str(err.value)

    def test_graph_is_not_read_when_ambiguous(self, tmp_path):
        _touch(tmp_path, "Lsub01_sessA.arg")
        _touch(tmp_path, "Lsub01_sessB.arg")
        sub = {"subject": "sub01", "side": "L", "dir": str(tmp_path), "graph_file": "Lsub01*.arg"}
        aims = _fake_aims()
        with patch.object(gm, "aims", aims), pytest.raises(RuntimeError):
            gm._load_and_extract_subject(sub, set(), VT)
        aims.read.assert_not_called()


class TestSingleMatch:
    def test_the_one_match_is_read(self, tmp_path):
        path = _touch(tmp_path, "Lsub01_sessA.arg")
        sub = {"subject": "sub01", "side": "L", "dir": str(tmp_path), "graph_file": "Lsub01*.arg"}
        aims = _fake_aims()
        with patch.object(gm, "aims", aims):
            name, data = gm._load_and_extract_subject(sub, set(), VT)
        assert (name, data) == ("sub01", {})
        aims.read.assert_called_once_with(path)

    def test_no_match_returns_none(self, tmp_path):
        sub = {"subject": "sub01", "side": "L", "dir": str(tmp_path), "graph_file": "Lsub01*.arg"}
        with patch.object(gm, "aims", _fake_aims()):
            assert gm._load_and_extract_subject(sub, set(), VT) == ("sub01", None)


class TestAnchoredFilename:
    def test_longer_subject_id_is_not_a_match(self, tmp_path):
        wanted = _touch(tmp_path, "Lsub-1_sessA.arg")
        _touch(tmp_path, "Lsub-10_sessA.arg")
        sub = {"subject": "sub-1", "side": "L", "dir": str(tmp_path), "graph_file": "Lsub-1*.arg"}
        aims = _fake_aims()
        with patch.object(gm, "aims", aims):
            gm._load_and_extract_subject(sub, set(), VT)
        aims.read.assert_called_once_with(wanted)

    def test_dot_after_subject_is_a_match(self, tmp_path):
        wanted = _touch(tmp_path, "Lsub-1.arg")
        sub = {"subject": "sub-1", "side": "L", "dir": str(tmp_path), "graph_file": "Lsub-1*.arg"}
        aims = _fake_aims()
        with patch.object(gm, "aims", aims):
            gm._load_and_extract_subject(sub, set(), VT)
        aims.read.assert_called_once_with(wanted)

    def test_other_side_is_not_a_match(self, tmp_path):
        _touch(tmp_path, "Rsub01_sessA.arg")
        sub = {"subject": "sub01", "side": "L", "dir": str(tmp_path), "graph_file": "Lsub01*.arg"}
        with patch.object(gm, "aims", _fake_aims()):
            assert gm._load_and_extract_subject(sub, set(), VT) == ("sub01", None)


class TestLiteralCharacters:
    def test_glob_characters_in_subject_and_directory_are_literal(self, tmp_path):
        db = tmp_path / "db[x]%d"
        graph_dir = db / "sub[1]" / GRAPH_PATH
        wanted = _touch(graph_dir, "Lsub[1]_sessA.arg")
        _touch(db / "sub1" / GRAPH_PATH, "Lsub1_sessA.arg")
        aims, result = _run_runner(tmp_path, "sub[1]", db)
        assert result == ("sub[1]", {})
        aims.read.assert_called_once_with(wanted)

    def test_percent_in_graph_path_is_not_formatted_twice(self, tmp_path):
        db = tmp_path / "db%(subject)s"
        wanted = _touch(db / "sub01" / GRAPH_PATH, "Lsub01_sessA.arg")
        aims, result = _run_runner(tmp_path, "sub01", db)
        assert result == ("sub01", {})
        aims.read.assert_called_once_with(wanted)
