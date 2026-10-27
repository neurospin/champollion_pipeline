"""LOGGING convention in src/generate_umap_reference.py (TASK-LOGGING-BDRABCZUK-B8825FF10EEB).

The UMAP reference script reports through the ``generate_umap_reference`` module logger instead
of print(); tests assert the records with caplog, not capsys. ``main()`` calls
``init_pipeline_process()`` and ``ScriptBuilder.main()``, whose root handlers send LOG and OUTPUT
records to stdout and WARNING and above to stderr.

Warnings (WARNING):
REQ-LOGGING-BDRABCZUK-57787FDB20D3: no model directory -> No model directories found and the path.
REQ-LOGGING-BDRABCZUK-D3C895E7A4A0: directory name without _left/_right suffix -> the name.
REQ-LOGGING-BDRABCZUK-2A6DE22C2B63: missing reference CSV -> Reference CSV not found and the path.
REQ-LOGGING-BDRABCZUK-B5659856951E: unreadable reference CSV -> the path and the exception message.
REQ-LOGGING-BDRABCZUK-6EFE5B483902: no reference embeddings -> No reference embeddings found.
Progress (LOG):
REQ-LOGGING-BDRABCZUK-D712FC798538: existing artefacts skipped -> region, hemisphere, already exists.
REQ-LOGGING-BDRABCZUK-ADE2C7673FDB: UMAP fit start -> region, hemisphere, reference point count.
REQ-LOGGING-BDRABCZUK-98DFE40B2113: run settings -> the six option values.
REQ-LOGGING-BDRABCZUK-42E72DFF2282: --overwrite -> Overwrite mode.
Results (OUTPUT):
REQ-LOGGING-BDRABCZUK-053D8848323E: saved artefacts -> both artefact file names.
REQ-LOGGING-BDRABCZUK-BD98B171FE3A: available UMAP reference model count.
Static:
REQ-LOGGING-BDRABCZUK-FA28F082DC1D: no static logging text matches the MCP fold regex.
(REQ-LOGGING-BDRABCZUK-C96C8DAAF703 and -22ABEF2F2CBD: tests/test_logging_convention.py, through
LOG_CONVERTED_MODULES.)

Synthetic model directories only; UMAP fitting and artefact writes are replaced by fakes.
"""

import ast
import logging
import os.path as osp
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# The champollion_V1 helper module may be absent; stub it before importing the script.
sys.modules.setdefault("put_together_embeddings_files", MagicMock())

import generate_umap_reference  # noqa: E402
from generate_umap_reference import GenerateUmapReference  # noqa: E402
from tests.log_assertions import LOG, OUTPUT  # noqa: E402

LOGGER = "generate_umap_reference"
SCRIPT_PY = Path(__file__).resolve().parents[1] / "src" / "generate_umap_reference.py"
# Copy of champollion_sulcal_mcp runner.FOLD_RE (it reads the stage output line by line).
MCP_FOLD_RE = re.compile(r"[Ff]old\s+(\d+)\s*(?:/|of)\s*(\d+)")

generate_umap_reference_fn = generate_umap_reference.generate_umap_reference


class _FakeReducer:
    """Stand-in for ``umap.UMAP`` that projects onto the first two columns."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, X):
        return np.asarray(X)[:, :2]


@pytest.fixture
def umap_caplog(caplog):
    """caplog capturing every level of the generate_umap_reference logger."""
    caplog.set_level(1, logger=LOGGER)
    return caplog


@pytest.fixture  # noqa: V103 - used through usefixtures
def fake_umap_no_writes(monkeypatch):
    """Replace the UMAP fit, joblib.dump and np.save so nothing is fitted or written."""
    module = MagicMock()
    module.UMAP = _FakeReducer
    monkeypatch.setattr(generate_umap_reference, "umap_lib", module)
    monkeypatch.setattr(generate_umap_reference.joblib, "dump", lambda obj, path: None)
    monkeypatch.setattr(generate_umap_reference.np, "save", lambda path, arr: None)


def _messages(caplog, level: int) -> list[str]:
    """generate_umap_reference messages captured at ``level``, in emission order."""
    return [record.getMessage() for record in caplog.records if record.name == LOGGER and record.levelno == level]


def _assert_one(caplog, level: int, *texts: str) -> str:
    """Return the single message at ``level`` containing every one of ``texts``; fail listing the records."""
    found = [message for message in _messages(caplog, level) if all(text in message for text in texts)]
    captured = [(record.name, record.levelname, record.getMessage()) for record in caplog.records]
    assert len(found) == 1, f"expected one {logging.getLevelName(level)} record with {texts}; captured: {captured}"
    return found[0]


def _make_model_dir(models_dir: Path, region_hemi: str, subpath: str, rows: int = 3) -> str:
    """Create ``models_dir/region_hemi/run1/subpath`` holding a small synthetic CSV."""
    model_path = models_dir / region_hemi / "run1"
    csv_path = model_path / subpath
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["ID,d0,d1,d2,d3"] + [f"sub-{r:02d},{r}.0,{r + 1}.0,{r + 2}.0,{r + 3}.0" for r in range(rows)]
    csv_path.write_text("\n".join(lines) + "\n")
    return str(model_path)


def _patch_model_paths(monkeypatch, paths: list) -> None:
    monkeypatch.setattr(generate_umap_reference, "get_model_paths", lambda d: list(paths))


def _make_script(argv: list) -> GenerateUmapReference:
    script = GenerateUmapReference()
    script.args = script.parse_args(argv)
    return script


# --------------------------------------------------------------------------- #
# Warnings
# --------------------------------------------------------------------------- #


class TestWarnings:
    def test_no_model_directory_logged_at_warning(self, monkeypatch, umap_caplog, tmp_path):
        models_dir = tmp_path / "models"
        _patch_model_paths(monkeypatch, [])

        assert generate_umap_reference_fn(str(models_dir), "emb.csv", str(tmp_path / "out")) == []

        _assert_one(umap_caplog, logging.WARNING, "No model directories found", osp.abspath(models_dir))

    def test_unrecognised_dir_name_logged_at_warning(self, monkeypatch, umap_caplog, tmp_path):
        models_dir = tmp_path / "models"
        model_path = models_dir / "models_cache" / "run1"
        model_path.mkdir(parents=True)
        _patch_model_paths(monkeypatch, [str(model_path)])

        generate_umap_reference_fn(str(models_dir), "emb.csv", str(tmp_path / "out"))

        _assert_one(umap_caplog, logging.WARNING, "models_cache")

    def test_missing_reference_csv_logged_at_warning(self, monkeypatch, umap_caplog, tmp_path):
        models_dir = tmp_path / "models"
        model_path = models_dir / "SC_left" / "run1"
        model_path.mkdir(parents=True)
        _patch_model_paths(monkeypatch, [str(model_path)])

        generate_umap_reference_fn(str(models_dir), "missing.csv", str(tmp_path / "out"))

        _assert_one(umap_caplog, logging.WARNING, "Reference CSV not found", str(model_path / "missing.csv"))

    def test_unreadable_reference_csv_logged_at_warning(self, monkeypatch, umap_caplog, tmp_path):
        models_dir = tmp_path / "models"
        model_path = _make_model_dir(models_dir, "SC_left", "emb.csv")
        _patch_model_paths(monkeypatch, [model_path])
        monkeypatch.setattr("pandas.read_csv", MagicMock(side_effect=ValueError("synthetic broken header")))

        assert generate_umap_reference_fn(str(models_dir), "emb.csv", str(tmp_path / "out")) == []

        _assert_one(umap_caplog, logging.WARNING, osp.join(model_path, "emb.csv"), "synthetic broken header")

    def test_no_reference_embeddings_logged_at_warning(self, monkeypatch, umap_caplog, tmp_path):
        models_dir = tmp_path / "models"
        model_path = models_dir / "SC_left" / "run1"
        model_path.mkdir(parents=True)
        _patch_model_paths(monkeypatch, [str(model_path)])

        assert generate_umap_reference_fn(str(models_dir), "missing.csv", str(tmp_path / "out")) == []

        _assert_one(umap_caplog, logging.WARNING, "No reference embeddings found")


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #


class TestProgress:
    def test_existing_artefacts_skip_logged_at_log(self, monkeypatch, umap_caplog, tmp_path):
        models_dir = tmp_path / "models"
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "umap_SC_left.pkl").write_bytes(b"")
        (out_dir / "umap_SC_left_coords.npy").write_bytes(b"")
        _patch_model_paths(monkeypatch, [_make_model_dir(models_dir, "SC_left", "emb.csv")])

        assert generate_umap_reference_fn(str(models_dir), "emb.csv", str(out_dir)) == [("SC", "left")]

        message = _assert_one(umap_caplog, LOG, "already exists")
        assert re.search(r"\bSC\b", message), message
        assert re.search(r"\bleft\b", message), message

    @pytest.mark.usefixtures("fake_umap_no_writes")
    def test_umap_fit_start_logged_at_log(self, monkeypatch, umap_caplog, tmp_path):
        models_dir = tmp_path / "models"
        _patch_model_paths(monkeypatch, [_make_model_dir(models_dir, "SC_left", "emb.csv", rows=37)])

        generate_umap_reference_fn(str(models_dir), "emb.csv", str(tmp_path / "out"))

        fit_records = [m for m in _messages(umap_caplog, LOG) if re.search(r"\b37\b", m)]
        captured = [(record.levelname, record.getMessage()) for record in umap_caplog.records]
        assert len(fit_records) == 1, f"expected one LOG record with the point count 37; captured: {captured}"
        assert re.search(r"\bSC\b", fit_records[0]), fit_records[0]
        assert re.search(r"\bleft\b", fit_records[0]), fit_records[0]


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


class TestResults:
    @pytest.mark.usefixtures("fake_umap_no_writes")
    def test_saved_artefacts_logged_at_output(self, monkeypatch, umap_caplog, tmp_path):
        models_dir = tmp_path / "models"
        _patch_model_paths(monkeypatch, [_make_model_dir(models_dir, "SC_left", "emb.csv")])

        generate_umap_reference_fn(str(models_dir), "emb.csv", str(tmp_path / "out"))

        _assert_one(umap_caplog, OUTPUT, "umap_SC_left.pkl", "umap_SC_left_coords.npy")


# --------------------------------------------------------------------------- #
# GenerateUmapReference.run
# --------------------------------------------------------------------------- #


class TestRun:
    def test_run_settings_logged_at_log(self, monkeypatch, umap_caplog, tmp_path):
        monkeypatch.setattr(generate_umap_reference, "generate_umap_reference", lambda **kwargs: [])
        values = {
            "--models_dir": str(tmp_path / "synthetic_models"),
            "--reference_subpath": "synthetic_reference/emb.csv",
            "--output_dir": str(tmp_path / "synthetic_out"),
            "--n_neighbors": "13",
            "--min_dist": "0.37",
            "--random_state": "4242",
        }
        script = _make_script([item for pair in values.items() for item in pair])

        assert script.run() == 0

        logged = "\n".join(_messages(umap_caplog, LOG))
        missing = [option for option, value in values.items() if value not in logged]
        assert missing == [], f"run settings not in a LOG record: {missing}; LOG records: {logged!r}"

    def test_overwrite_mode_logged_at_log(self, monkeypatch, umap_caplog):
        monkeypatch.setattr(generate_umap_reference, "generate_umap_reference", lambda **kwargs: [])
        script = _make_script(["--models_dir", "m", "--reference_subpath", "e.csv", "--overwrite"])

        assert script.run() == 0

        _assert_one(umap_caplog, LOG, "Overwrite mode")

    def test_available_model_count_logged_at_output(self, monkeypatch, umap_caplog):
        generated = [("SC", "left"), ("SC", "right"), ("FIP", "left")]
        monkeypatch.setattr(generate_umap_reference, "generate_umap_reference", lambda **kwargs: generated)
        script = _make_script(["--models_dir", "m", "--reference_subpath", "e.csv"])

        assert script.run() == 0

        counts = [m for m in _messages(umap_caplog, OUTPUT) if re.search(r"\b3\b", m) and "UMAP reference" in m]
        captured = [(record.levelname, record.getMessage()) for record in umap_caplog.records]
        assert len(counts) == 1, f"expected one OUTPUT record with the model count 3; captured: {captured}"


# --------------------------------------------------------------------------- #
# Static: no fold-progress text
# --------------------------------------------------------------------------- #


def _logging_call_texts(tree: ast.Module) -> list[str]:
    """Static string parts of the ``logger.<method>(...)`` calls in ``tree``."""
    texts = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "logger"
        ):
            for arg in node.args:
                texts.extend(
                    part.value
                    for part in ast.walk(arg)
                    if isinstance(part, ast.Constant) and isinstance(part.value, str)
                )
    return texts


def test_no_static_message_matches_fold_pattern():
    texts = _logging_call_texts(ast.parse(SCRIPT_PY.read_text(encoding="utf-8")))

    assert texts, "src/generate_umap_reference.py has no logger call"
    assert [text for text in texts if MCP_FOLD_RE.search(text)] == []
