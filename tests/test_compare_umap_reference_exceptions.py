"""Broad exception handlers of compare_sulcal and generate_umap_reference (TASK-PYTHONIC-BDRABCZUK-2DA15DB02EDA).

Both handlers are per-item boundaries (one region/side, one model directory): the broad catch stays,
the item is still skipped, and the WARNING record keeps the traceback.

REQ-PYTHONIC-BDRABCZUK-B75E5AFD585A: a set A or set B UMAP model failure -> WARNING with the traceback.
REQ-PYTHONIC-BDRABCZUK-CFD046713C4C: a set A or set B UMAP model failure -> WARNING naming the model file.
REQ-PYTHONIC-BDRABCZUK-2BCAA8055D89: an unreadable reference CSV -> WARNING with the traceback.
"""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from compare_sulcal import embeddings

# The champollion_V1 helper module may be absent; stub it before importing the script.
sys.modules.setdefault("champollion.utils.put_together_embeddings_files", MagicMock())

import generate_umap_reference  # noqa: E402

REGION = "S.C.-sylv."
PKL_NAME = f"umap_{REGION}_left.pkl"
UMAP_ERROR = "corrupt umap pickle"
CSV_ERROR = "synthetic broken header"


class _FakeUmap:
    def transform(self, data):
        return np.zeros((len(data), 2))


@pytest.fixture
def umap_dirs(tmp_path):
    """Set A and set B UMAP reference dirs, each holding the left S.C.-sylv. pickle file name."""
    dirs = {}
    for label in ("a", "b"):
        directory = tmp_path / f"umap_{label}"
        directory.mkdir()
        (directory / PKL_NAME).touch()
        dirs[label] = str(directory)
    return dirs


def _failing_loader(failing_dir):
    def _load(path):
        if path.startswith(failing_dir):
            raise ValueError(UMAP_ERROR)
        return _FakeUmap()

    return _load


def _warnings(caplog, logger_name):
    return [record for record in caplog.records if record.name == logger_name and record.levelno == logging.WARNING]


def _only_warning(caplog, logger_name):
    records = _warnings(caplog, logger_name)
    captured = [(record.name, record.levelname, record.getMessage()) for record in caplog.records]
    assert len(records) == 1, f"expected one WARNING record on {logger_name}; got {captured}"
    return records[0]


def _assert_traceback(record, error_type, error_text):
    assert record.exc_info, f"WARNING record carries no traceback: {record.getMessage()!r}"
    exc_type, exc_value, exc_tb = record.exc_info
    assert exc_type is error_type
    assert error_text in str(exc_value)
    assert exc_tb is not None


# --------------------------------------------------------------------------- #
# compare_sulcal.embeddings._compute_umap_projection
# --------------------------------------------------------------------------- #


class TestUmapProjectionFailure:
    def _project(self, umap_dirs, monkeypatch, caplog, failing):
        caplog.set_level(1, logger="compare_sulcal")
        monkeypatch.setattr(embeddings, "load_umap_model", _failing_loader(umap_dirs[failing]))
        emb = np.zeros((3, 4))

        result = embeddings._compute_umap_projection(umap_dirs["a"], umap_dirs["b"], REGION, "L", emb, emb)

        assert result == f"umap_transform_failed_{failing}"
        return _only_warning(caplog, "compare_sulcal.embeddings")

    @pytest.mark.parametrize("failing", ["a", "b"])
    def test_warning_carries_traceback(self, umap_dirs, monkeypatch, caplog, failing):
        record = self._project(umap_dirs, monkeypatch, caplog, failing)

        _assert_traceback(record, ValueError, UMAP_ERROR)

    @pytest.mark.parametrize("failing", ["a", "b"])
    def test_warning_names_model_file(self, umap_dirs, monkeypatch, caplog, failing):
        record = self._project(umap_dirs, monkeypatch, caplog, failing)

        model_file = str(Path(umap_dirs[failing]) / PKL_NAME)
        assert model_file in record.getMessage()


# --------------------------------------------------------------------------- #
# generate_umap_reference.generate_umap_reference
# --------------------------------------------------------------------------- #


class TestReferenceCsvFailure:
    def test_warning_carries_traceback(self, monkeypatch, caplog, tmp_path):
        caplog.set_level(1, logger="generate_umap_reference")
        models_dir = tmp_path / "models"
        model_path = models_dir / "SC_left" / "run1"
        model_path.mkdir(parents=True)
        (model_path / "emb.csv").write_text("ID,d0\nsub-01,1.0\n")
        monkeypatch.setattr(generate_umap_reference, "get_model_paths", lambda _d: [str(model_path)])
        monkeypatch.setattr("pandas.read_csv", MagicMock(side_effect=ValueError(CSV_ERROR)))

        generated = generate_umap_reference.generate_umap_reference(str(models_dir), "emb.csv", str(tmp_path / "out"))

        assert generated == []
        csv_warnings = [r for r in _warnings(caplog, "generate_umap_reference") if CSV_ERROR in r.getMessage()]
        assert len(csv_warnings) == 1, [r.getMessage() for r in caplog.records]
        _assert_traceback(csv_warnings[0], ValueError, CSV_ERROR)
