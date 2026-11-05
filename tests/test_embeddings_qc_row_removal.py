#!/usr/bin/env python3
"""
Edge cases of embeddings_qc.remove_subject_rows on one embeddings CSV file. Synthetic fixtures only.

The rows dropped from an ordinary file are covered through the stage in tests/test_generate_embeddings_input_qc.py.
"""

import os
import stat
from unittest.mock import patch

import pytest

from champollion_pipeline.models.embeddings_qc import remove_subject_rows

_MODULE = "champollion_pipeline.models.embeddings_qc"
_SUBJECT = "sub-01"
_OTHER_SUBJECT = "sub-02"
_LAST_SUBJECT = "sub-03"
_ROW = f"{_SUBJECT},0.5\n"
_NO_ID_CELL = f"name,dim1\n{_ROW}"
_NO_SUBJECT = f"ID,dim1\n{_OTHER_SUBJECT},0.5\n"
_DROPPABLE = f"ID,dim1\n{_ROW}{_OTHER_SUBJECT},0.5\n"
_PAST_NS = 10**18  # 2001: a rewrite cannot land on it, however coarse the clock of the file system


def _lay(tmp_path, content):
    """An embeddings file holding ``content`` verbatim, last modified at _PAST_NS."""
    path = tmp_path / "full_embeddings.csv"
    path.write_bytes(content.encode())
    os.utime(path, ns=(_PAST_NS, _PAST_NS))
    return path


def _remove(path):
    return remove_subject_rows(str(path), frozenset({_SUBJECT}))


class TestNoIdCell:
    @pytest.mark.parametrize("content", ["", _NO_ID_CELL], ids=["empty_file", "header_without_id_cell"])
    def test_gives_0(self, tmp_path, content):
        assert _remove(_lay(tmp_path, content)) == 0


class TestNoRowDropped:
    @pytest.mark.parametrize(
        "content",
        ["", _NO_ID_CELL, _NO_SUBJECT],
        ids=["empty_file", "header_without_id_cell", "no_id_of_the_subjects"],
    )
    def test_file_stays_unwritten(self, tmp_path, content):
        path = _lay(tmp_path, content)

        _remove(path)

        assert path.stat().st_mtime_ns == _PAST_NS
        assert path.read_bytes() == content.encode()


class TestShortLine:
    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            (f"ID,dim1\n\n{_ROW}", "ID,dim1\n\n"),
            (f"dim1,ID\nonly\n0.5,{_SUBJECT}\n", "dim1,ID\nonly\n"),
        ],
        ids=["blank_line", "line_ending_before_the_id_cell"],
    )
    def test_line_too_short_to_hold_the_id_cell_is_kept(self, tmp_path, content, expected):
        path = _lay(tmp_path, content)

        assert _remove(path) == 1
        assert path.read_bytes() == expected.encode()


class TestFileMode:
    @pytest.mark.parametrize("mode", [0o644, 0o640, 0o664], ids=oct)
    def test_rewritten_file_holds_the_mode_of_the_original(self, tmp_path, mode):
        path = _lay(tmp_path, _DROPPABLE)
        path.chmod(mode)

        _remove(path)

        assert stat.S_IMODE(path.stat().st_mode) == mode


class TestLineEnds:
    @pytest.mark.parametrize("end", ["\r\n", "\r", "\n"], ids=["crlf", "cr", "lf"])
    def test_kept_lines_keep_their_line_ends(self, tmp_path, end):
        content = f"ID,dim1{end}{_SUBJECT},0.5{end}{_OTHER_SUBJECT},0.5{end}{_LAST_SUBJECT},0.5"
        path = _lay(tmp_path, content)

        assert _remove(path) == 1
        assert path.read_bytes() == content.replace(f"{_SUBJECT},0.5{end}", "").encode()


class TestFailure:
    @pytest.mark.parametrize("step", ["os.replace", "shutil.copymode"])
    def test_temporary_file_is_removed(self, tmp_path, step):
        path = _lay(tmp_path, _DROPPABLE)

        with patch(f"{_MODULE}.{step}", side_effect=OSError), pytest.raises(OSError):
            _remove(path)

        assert {child.name: child.read_bytes() for child in tmp_path.iterdir()} == {path.name: _DROPPABLE.encode()}
