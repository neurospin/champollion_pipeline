"""LOGGING convention in the compare_sulcal helper modules (TASK-LOGGING-BDRABCZUK-989CD9BB777F).

The compare tools' messages go through module loggers named after the module
(``compare_sulcal.<module>``) instead of print(); tests assert the records with
caplog, not capsys.

REQ-LOGGING-BDRABCZUK-FC02DEF3E711: a set A UMAP failure -> WARNING with region, side, exception text.
REQ-LOGGING-BDRABCZUK-EF76D3BE7CB7: a set B UMAP failure -> WARNING with region, side, exception text.
REQ-LOGGING-BDRABCZUK-F69F24ED5A82: a direct crop comparison without offset -> WARNING with key and reason.
REQ-LOGGING-BDRABCZUK-9F42816479CE: several matching graphs -> one WARNING with subject, path, count.
REQ-LOGGING-BDRABCZUK-07EA64A2D67D: a non-zero viewer exit -> ERROR with the return code.
REQ-LOGGING-BDRABCZUK-1B8E58FA45EB: no changed mask -> LOG "No changed masks to visualise".
REQ-LOGGING-BDRABCZUK-E6B52DB20D7A: no usable display -> WARNING containing "display".
REQ-LOGGING-BDRABCZUK-DC5BCAA4CA21: viewer start -> LOG with the number of masks shown.
REQ-LOGGING-BDRABCZUK-C96C8DAAF703 / -22ABEF2F2CBD (no print call, module logger) are checked by
tests/test_logging_convention.py through LOG_CONVERTED_MODULES.
"""

import glob
import logging
import re
import subprocess
import types

import numpy as np
import pytest

import compare_sulcal.brainvisa
from compare_sulcal import crops, databases, embeddings, viewer
from tests.log_assertions import LOG

UMAP_ERROR = "corrupt umap pickle"


@pytest.fixture
def compare_caplog(caplog):
    """caplog capturing every level of the compare_sulcal loggers."""
    caplog.set_level(1, logger="compare_sulcal")
    return caplog


def _records(caplog, logger_name, level):
    return [record for record in caplog.records if record.name == logger_name and record.levelno == level]


def _only_record(caplog, logger_name, level):
    records = _records(caplog, logger_name, level)
    captured = [(record.name, record.levelname, record.getMessage()) for record in caplog.records]
    assert len(records) == 1, f"expected one {logging.getLevelName(level)} record on {logger_name}; got {captured}"
    return records[0].getMessage()


# --------------------------------------------------------------------------- #
# compare_sulcal.embeddings
# --------------------------------------------------------------------------- #


class _FakeUmap:
    def transform(self, data):
        return np.zeros((len(data), 2))


@pytest.fixture
def umap_dirs(tmp_path):
    """Set A and set B UMAP reference dirs, each holding the left S.C.-sylv. pickle file name."""
    dirs = []
    for name in ("umap_a", "umap_b"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "umap_S.C.-sylv._left.pkl").touch()
        dirs.append(str(directory))
    return dirs


def _failing_loader(failing_dir):
    def _load(path):
        if path.startswith(failing_dir):
            raise ValueError(UMAP_ERROR)
        return _FakeUmap()

    return _load


class TestEmbeddingsLogging:
    def _check(self, umap_dirs, monkeypatch, compare_caplog, failing, reason):
        monkeypatch.setattr(embeddings, "load_umap_model", _failing_loader(umap_dirs[failing]))
        emb = np.zeros((3, 4))

        result = embeddings._compute_umap_projection(umap_dirs[0], umap_dirs[1], "S.C.-sylv.", "L", emb, emb)

        assert result == reason
        message = _only_record(compare_caplog, "compare_sulcal.embeddings", logging.WARNING)
        assert "S.C.-sylv." in message
        assert re.search(r"\bL\b", message), f"side L missing from: {message!r}"
        assert UMAP_ERROR in message

    def test_umap_a_failure_logged_at_warning(self, umap_dirs, monkeypatch, compare_caplog):
        self._check(umap_dirs, monkeypatch, compare_caplog, 0, "umap_transform_failed_a")

    def test_umap_b_failure_logged_at_warning(self, umap_dirs, monkeypatch, compare_caplog):
        self._check(umap_dirs, monkeypatch, compare_caplog, 1, "umap_transform_failed_b")


# --------------------------------------------------------------------------- #
# compare_sulcal.crops
# --------------------------------------------------------------------------- #


class TestCropsLogging:
    @pytest.mark.parametrize("reason", ["no_mask_cropped", "no_transformation"])
    def test_direct_comparison_logged_at_warning(self, compare_caplog, reason):
        assert crops._unaligned_skip_reason("S.C.-sylv./L", reason, (4, 5, 6), (4, 5, 6)) is None

        message = _only_record(compare_caplog, "compare_sulcal.crops", logging.WARNING)
        assert "S.C.-sylv./L" in message
        assert reason in message


# --------------------------------------------------------------------------- #
# compare_sulcal.databases
# --------------------------------------------------------------------------- #

SUBJECT = "subj-alpha"
GRAPH_DIR = "/fake/db/subj-alpha/folds"
GRAPHS = [f"{GRAPH_DIR}/Lsubj-alpha_sessC.arg", f"{GRAPH_DIR}/Lsubj-alpha_sessA.arg"]


class _FakeGraph:
    def vertices(self):
        return []


class TestDatabasesLogging:
    def test_several_graphs_logged_at_warning(self, monkeypatch, compare_caplog):
        monkeypatch.setattr(glob, "glob", lambda *_a, **_k: list(GRAPHS))
        monkeypatch.setattr(compare_sulcal.brainvisa, "aims", types.SimpleNamespace(read=lambda _p: _FakeGraph()))
        sub = {"subject": SUBJECT, "dir": GRAPH_DIR, "graph_file": f"L{SUBJECT}*.arg"}

        databases._get_subject_voxel_counts(sub, "/fake/brainvisa")

        message = _only_record(compare_caplog, "compare_sulcal.databases", logging.WARNING)
        assert SUBJECT in message
        assert sorted(GRAPHS)[0] in message
        assert "2" in message.replace(SUBJECT, "").replace(GRAPH_DIR, "")


# --------------------------------------------------------------------------- #
# compare_sulcal.viewer
# --------------------------------------------------------------------------- #

MASKS_A = {"L/a.nii.gz": "/a/L/a.nii.gz", "L/b.nii.gz": "/a/L/b.nii.gz"}
MASKS_B = {"L/a.nii.gz": "/b/L/a.nii.gz", "L/b.nii.gz": "/b/L/b.nii.gz"}


@pytest.fixture  # noqa: V103 - requested through usefixtures
def display_env(monkeypatch):
    """A usable X11 display, so the viewer is not skipped."""
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)


@pytest.fixture
def viewer_run(monkeypatch):
    """Replace subprocess.run with a recorder returning the given return code."""

    def _install(returncode):
        calls = []

        def _run(cmd, *_args, **_kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, returncode)

        monkeypatch.setattr(subprocess, "run", _run)
        return calls

    return _install


class TestViewerLogging:
    @pytest.mark.usefixtures("display_env")
    def test_viewer_failure_logged_at_error(self, viewer_run, compare_caplog, tmp_path):
        viewer_run(139)

        assert viewer.visualise_mask_diffs({"L/a.nii.gz": 1.0}, MASKS_A, MASKS_B, xor_dir=tmp_path) == 1

        assert "139" in _only_record(compare_caplog, "compare_sulcal.viewer", logging.ERROR)

    @pytest.mark.usefixtures("display_env")
    def test_no_changed_masks_logged_at_log(self, viewer_run, compare_caplog):
        calls = viewer_run(0)

        assert viewer.visualise_mask_diffs({"L/a.nii.gz": 0.0}, MASKS_A, MASKS_B) == 0

        assert calls == []
        assert "No changed masks to visualise" in _only_record(compare_caplog, "compare_sulcal.viewer", LOG)

    @pytest.mark.parametrize(
        "env",
        [{}, {"DISPLAY": ":0", "QT_QPA_PLATFORM": "offscreen"}],
        ids=["no_display", "offscreen"],
    )
    def test_missing_display_logged_at_warning(self, monkeypatch, viewer_run, compare_caplog, tmp_path, env):
        for var in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"):
            monkeypatch.delenv(var, raising=False)
        for var, value in env.items():
            monkeypatch.setenv(var, value)
        calls = viewer_run(0)

        assert viewer.visualise_mask_diffs({"L/a.nii.gz": 1.0}, MASKS_A, MASKS_B, xor_dir=tmp_path) == 0

        assert calls == []
        assert "display" in _only_record(compare_caplog, "compare_sulcal.viewer", logging.WARNING).lower()

    @pytest.mark.usefixtures("display_env")
    def test_viewer_start_logged_at_log(self, viewer_run, compare_caplog, tmp_path):
        calls = viewer_run(0)

        scores = {"L/a.nii.gz": 3.0, "L/b.nii.gz": 1.0}
        assert viewer.visualise_mask_diffs(scores, MASKS_A, MASKS_B, xor_dir=tmp_path) == 0

        assert len(calls) == 1
        assert "2" in _only_record(compare_caplog, "compare_sulcal.viewer", LOG)
