"""LOGGING convention in src/upload_models_to_hf.py (TASK-LOGGING-BDRABCZUK-B8825FF10EEB).

The Hugging Face upload script reports through the ``upload_models_to_hf`` module logger instead
of print(); tests assert the records with caplog, not capsys. ``main()`` calls
``init_pipeline_process()`` and ``ScriptBuilder.main()``, whose root handlers send LOG and OUTPUT
records to stdout and WARNING and above to stderr.

Errors (ERROR):
REQ-LOGGING-BDRABCZUK-7A5F00D1A8E7: huggingface_hub not installed -> huggingface_hub is required.
REQ-LOGGING-BDRABCZUK-4D970123CEB9: models_dir not a directory -> models_dir not found and the path.
Progress (LOG):
REQ-LOGGING-BDRABCZUK-5B4184817394: upload start -> models directory, repository id, --masks-version.
Results (OUTPUT):
REQ-LOGGING-BDRABCZUK-C11E463A6C1F: browse URL https://huggingface.co/<repo_id>/tree/main/<masks_version>.
Static:
REQ-LOGGING-BDRABCZUK-6ED9BF8CF656: no static logging text matches the MCP fold regex.
(REQ-LOGGING-BDRABCZUK-C96C8DAAF703 and -22ABEF2F2CBD: tests/test_logging_convention.py, through
LOG_CONVERTED_MODULES.)

No network: every test replaces the module's ``huggingface_hub`` binding with a stub (or None).
"""

import ast
import logging
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import upload_models_to_hf
from tests.log_assertions import LOG, OUTPUT
from upload_models_to_hf import UploadModelsToHF

LOGGER = "upload_models_to_hf"
SCRIPT_PY = Path(__file__).resolve().parents[1] / "src" / "upload_models_to_hf.py"
# Copy of champollion_sulcal_mcp runner.FOLD_RE (it reads the stage output line by line).
MCP_FOLD_RE = re.compile(r"[Ff]old\s+(\d+)\s*(?:/|of)\s*(\d+)")

REPO_ID = "synthetic-lab/champollion-test"
MASKS_VERSION = "canonical_test"


@pytest.fixture
def upload_caplog(caplog):
    """caplog capturing every level of the upload_models_to_hf logger."""
    caplog.set_level(1, logger=LOGGER)
    return caplog


@pytest.fixture  # noqa: V103 - used through usefixtures
def hf_stub(monkeypatch):
    """Replace ``huggingface_hub`` with a stub whose ``HfApi`` is a mock: no network call."""
    module = MagicMock()
    module.HfApi = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(upload_models_to_hf, "huggingface_hub", module)
    return module


def _messages(caplog, level: int) -> list[str]:
    """upload_models_to_hf messages captured at ``level``, in emission order."""
    return [record.getMessage() for record in caplog.records if record.name == LOGGER and record.levelno == level]


def _assert_one(caplog, level: int, *texts: str) -> str:
    """Return the single message at ``level`` containing every one of ``texts``; fail listing the records."""
    found = [message for message in _messages(caplog, level) if all(text in message for text in texts)]
    captured = [(record.name, record.levelname, record.getMessage()) for record in caplog.records]
    assert len(found) == 1, f"expected one {logging.getLevelName(level)} record with {texts}; captured: {captured}"
    return found[0]


def _make_script(models_dir: Path) -> UploadModelsToHF:
    script = UploadModelsToHF()
    script.args = script.parse_args([str(models_dir), REPO_ID, "--masks-version", MASKS_VERSION])
    return script


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class TestErrors:
    def test_missing_huggingface_hub_logged_at_error(self, monkeypatch, upload_caplog, tmp_path):
        monkeypatch.setattr(upload_models_to_hf, "huggingface_hub", None)

        assert _make_script(tmp_path).run() == 1

        _assert_one(upload_caplog, logging.ERROR, "huggingface_hub is required")

    @pytest.mark.usefixtures("hf_stub")
    def test_missing_models_dir_logged_at_error(self, upload_caplog, tmp_path):
        missing = tmp_path / "no_models_here"

        assert _make_script(missing).run() == 1

        _assert_one(upload_caplog, logging.ERROR, "models_dir not found", str(missing.resolve()))


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #


class TestProgress:
    @pytest.mark.usefixtures("hf_stub")
    def test_upload_start_logged_at_log(self, upload_caplog, tmp_path):
        assert _make_script(tmp_path).run() == 0

        _assert_one(upload_caplog, LOG, str(tmp_path.resolve()), REPO_ID, MASKS_VERSION)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


class TestResults:
    @pytest.mark.usefixtures("hf_stub")
    def test_browse_url_logged_at_output(self, upload_caplog, tmp_path):
        assert _make_script(tmp_path).run() == 0

        _assert_one(upload_caplog, OUTPUT, f"https://huggingface.co/{REPO_ID}/tree/main/{MASKS_VERSION}")


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

    assert texts, "src/upload_models_to_hf.py has no logger call"
    assert [text for text in texts if MCP_FOLD_RE.search(text)] == []
