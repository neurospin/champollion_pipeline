"""LOGGING convention for models/fetch.py (TASK-LOGGING-BDRABCZUK-D04DAA358F54).

The model-fetch strategies report through the champollion_pipeline.models.fetch logger
instead of print(); the interactive prompts stay input() calls (a prompt is not a log record).

REQ-LOGGING-BDRABCZUK-192E2C6D29CC: a failed Hugging Face, remote archive or local path strategy
    is logged by fetch_model_set at WARNING with the exception message.
REQ-LOGGING-BDRABCZUK-30EA533B6978: a successful Hugging Face download is logged at OUTPUT with its path.
REQ-LOGGING-BDRABCZUK-8EC95D00C4C7: a written model_set.json marker is logged at OUTPUT with its path.
REQ-LOGGING-BDRABCZUK-35E2DB99D460: the Hugging Face download start is logged at LOG with the repo id.
REQ-LOGGING-BDRABCZUK-937B3160CBF1: the remote archive download start is logged at LOG with the URL.
REQ-LOGGING-BDRABCZUK-3B8AD0885242: a local models directory is logged at LOG with its path.
REQ-LOGGING-BDRABCZUK-A69D9067DAFB: a local archive extraction is logged at LOG with the archive path.
REQ-LOGGING-BDRABCZUK-1F8C482DB69F: a reused cached extraction logs the --no-cache hint at LOG.
REQ-LOGGING-BDRABCZUK-420650F12617: the interactive fallback is logged at WARNING with the models path.
REQ-LOGGING-BDRABCZUK-99264EF02912: a declined local copy is logged at WARNING before the exit.
REQ-LOGGING-BDRABCZUK-B1469DA4FF11: a models directory given at the prompt is logged at LOG.
(REQ-LOGGING-BDRABCZUK-C96C8DAAF703 and -22ABEF2F2CBD: tests/test_logging_convention.py, through
LOG_CONVERTED_MODULES.)

No network: huggingface_hub.snapshot_download and urllib.request.urlretrieve are replaced by fakes.
"""

import logging
import os
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.models import fetch as models_fetch
from champollion_pipeline.models.cache import MODEL_SET_FILE_NAME
from champollion_pipeline.models.fetch import (
    HuggingFaceStrategy,
    InteractiveFallbackStrategy,
    LocalPathStrategy,
    RemoteArchiveStrategy,
    fetch_model_set,
)
from tests.log_assertions import LOG, OUTPUT

pytestmark = pytest.mark.unit  # noqa: V107

REPO_ID = "neurospin/Champollion_V1"
VERSION = "canonical_corrected_26_1"
ARCHIVE_URL = "https://example.com/models.tar.gz"
LOCAL_COPY_PROMPT = "Do you have a local copy? (yes/no): "
MODELS_PATH_PROMPT = "Path to models directory or archive (tar.xz, tar.gz, .gz): "
STRATEGY_ERROR = "strategy broke: disk full"


def _make_tar(tmp_path: Path, name: str = "models.tar.gz", top: str = "models") -> Path:
    """Create a small tar archive holding top/weights.txt."""
    payload = tmp_path / "payload"
    payload.mkdir(exist_ok=True)
    (payload / "weights.txt").write_text("weights")
    archive = tmp_path / name
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload / "weights.txt", arcname=f"{top}/weights.txt")
    return archive


def _messages_at(capture, level: int) -> list[str]:
    """Messages of the records captured so far at ``level``."""
    return [record.getMessage() for record in capture.records if record.levelno == level]


def _fake_snapshot_download(on_call=None):
    """Return a snapshot_download stand-in that creates <local_dir>/<subfolder>/ and returns local_dir."""

    def fake(*_args, local_dir, allow_patterns=None, **_options):
        if on_call is not None:
            on_call()
        for pattern in allow_patterns or ["*"]:
            os.makedirs(os.path.join(local_dir, pattern.rstrip("*").rstrip("/")), exist_ok=True)
        return local_dir

    return fake


class _FakeInput:
    """input() stand-in answering from a list and keeping the prompts it was given."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str = "") -> str:
        self.prompts.append(prompt)
        return self._answers.pop(0)


def _interactive(fake_input: _FakeInput):
    """Patch stdin as a terminal and input() with ``fake_input``."""
    return (
        patch.object(models_fetch.sys.stdin, "isatty", return_value=True),
        patch("builtins.input", side_effect=fake_input),
    )


class TestFetchModelSetLogging:
    """REQ-LOGGING-BDRABCZUK-192E2C6D29CC."""

    @pytest.mark.parametrize("case", ["huggingface", "remote", "local"])
    def test_failed_strategy_logged_at_warning(self, tmp_path, pipeline_caplog, case):
        local_models = tmp_path / "models"
        local_models.mkdir()
        strategy, models_path = {
            "huggingface": (HuggingFaceStrategy, REPO_ID),
            "remote": (RemoteArchiveStrategy, ARCHIVE_URL),
            "local": (LocalPathStrategy, str(local_models)),
        }[case]
        with (
            patch.object(strategy, "fetch", side_effect=OSError(STRATEGY_ERROR)),
            patch.object(models_fetch.sys.stdin, "isatty", return_value=False),
            pytest.raises(RuntimeError, match="Cannot find models"),
        ):
            fetch_model_set(models_path, str(tmp_path / "cache"), None, False)

        record = pipeline_caplog.assert_logged(logging.WARNING, STRATEGY_ERROR)
        assert record.name == "champollion_pipeline.models.fetch"


class TestHuggingFaceLogging:
    """REQ-LOGGING-BDRABCZUK-30EA533B6978, -8EC95D00C4C7, -35E2DB99D460."""

    def test_download_result_logged_at_output(self, tmp_path, pipeline_caplog):
        with patch("huggingface_hub.snapshot_download", side_effect=_fake_snapshot_download()):
            downloaded = HuggingFaceStrategy(subfolder=VERSION).fetch(REPO_ID, str(tmp_path))

        pipeline_caplog.assert_logged(OUTPUT, downloaded)

    def test_marker_written_logged_at_output(self, tmp_path, pipeline_caplog):
        with patch("huggingface_hub.snapshot_download", side_effect=_fake_snapshot_download()):
            downloaded = HuggingFaceStrategy(subfolder=VERSION).fetch(REPO_ID, str(tmp_path))

        marker = os.path.join(downloaded, MODEL_SET_FILE_NAME)
        assert os.path.isfile(marker)
        pipeline_caplog.assert_logged(OUTPUT, marker)

    def test_download_start_logged_at_log(self, tmp_path, pipeline_caplog):
        logged_before_download: list[str] = []

        def snapshot_records() -> None:
            logged_before_download.extend(_messages_at(pipeline_caplog, LOG))

        with patch("huggingface_hub.snapshot_download", side_effect=_fake_snapshot_download(snapshot_records)):
            HuggingFaceStrategy(subfolder=VERSION).fetch(REPO_ID, str(tmp_path))

        assert any(REPO_ID in message for message in logged_before_download), (
            f"no LOG record naming {REPO_ID!r} before snapshot_download; captured: {logged_before_download}"
        )


class TestRemoteArchiveLogging:
    """REQ-LOGGING-BDRABCZUK-937B3160CBF1."""

    def test_download_start_logged_at_log(self, tmp_path, pipeline_caplog):
        source = _make_tar(tmp_path, name="source.tar.gz")
        cache = tmp_path / "cache"
        cache.mkdir()
        logged_before_download: list[str] = []

        def fake_urlretrieve(_url, target):
            logged_before_download.extend(_messages_at(pipeline_caplog, LOG))
            Path(target).write_bytes(source.read_bytes())

        with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            RemoteArchiveStrategy().fetch(ARCHIVE_URL, str(cache))

        assert any(ARCHIVE_URL in message for message in logged_before_download), (
            f"no LOG record naming {ARCHIVE_URL!r} before the download; captured: {logged_before_download}"
        )


class TestLocalPathLogging:
    """REQ-LOGGING-BDRABCZUK-3B8AD0885242, -A69D9067DAFB."""

    def test_local_directory_logged_at_log(self, tmp_path, pipeline_caplog):
        models = tmp_path / "models"
        models.mkdir()

        LocalPathStrategy().fetch(str(models), str(tmp_path / "cache"))

        pipeline_caplog.assert_logged(LOG, str(models))

    def test_archive_extraction_logged_at_log(self, tmp_path, pipeline_caplog):
        archive = _make_tar(tmp_path)
        logged_before_extraction: list[str] = []

        def fake_extract(_archive_path, extract_to):
            logged_before_extraction.extend(_messages_at(pipeline_caplog, LOG))
            return extract_to

        with patch.object(LocalPathStrategy, "_extract_archive", side_effect=fake_extract):
            LocalPathStrategy().fetch(str(archive), str(tmp_path / "cache"))

        assert any(str(archive) in message for message in logged_before_extraction), (
            f"no LOG record naming {archive} before the extraction; captured: {logged_before_extraction}"
        )


class TestCachedExtractionHint:
    """REQ-LOGGING-BDRABCZUK-1F8C482DB69F."""

    @pytest.mark.parametrize("case", ["local", "remote", "interactive"])
    def test_no_cache_hint_logged_at_log(self, tmp_path, pipeline_caplog, case):
        archive = _make_tar(tmp_path)
        cache = tmp_path / "cache"
        (cache / "models").mkdir(parents=True)
        (cache / "models" / "weights.txt").write_text("weights")

        if case == "local":
            result = LocalPathStrategy().fetch(str(archive), str(cache))
        elif case == "remote":
            with patch("urllib.request.urlretrieve") as retrieve:
                result = RemoteArchiveStrategy().fetch(ARCHIVE_URL, str(cache))
            retrieve.assert_not_called()
        else:
            stdin_patch, input_patch = _interactive(_FakeInput(["yes", str(archive)]))
            with stdin_patch, input_patch:
                result = InteractiveFallbackStrategy().fetch(str(archive), str(cache))

        assert result == str(cache / "models")
        pipeline_caplog.assert_logged(LOG, "--no-cache")


class TestInteractiveFallbackLogging:
    """REQ-LOGGING-BDRABCZUK-420650F12617, -99264EF02912, -B1469DA4FF11."""

    def test_retrieval_failure_logged_at_warning(self, tmp_path, pipeline_caplog):
        missing = str(tmp_path / "missing_models")
        with (
            patch.object(models_fetch.sys.stdin, "isatty", return_value=False),
            pytest.raises(RuntimeError, match="Cannot find models"),
        ):
            fetch_model_set(missing, str(tmp_path / "cache"), None, False)

        pipeline_caplog.assert_logged(logging.WARNING, missing)

    def test_declined_local_copy_logged_at_warning(self, tmp_path, pipeline_caplog):
        fake_input = _FakeInput(["no"])
        stdin_patch, input_patch = _interactive(fake_input)
        with stdin_patch, input_patch, pytest.raises(SystemExit) as exit_info:
            InteractiveFallbackStrategy().fetch(str(tmp_path / "missing_models"), str(tmp_path / "cache"))

        assert exit_info.value.code == 0
        pipeline_caplog.assert_logged(logging.WARNING, "Cannot proceed without models")
        # The question stays an input() prompt, not a log record.
        assert fake_input.prompts == [LOCAL_COPY_PROMPT]

    def test_given_directory_logged_at_log(self, tmp_path, pipeline_caplog):
        models = tmp_path / "models"
        models.mkdir()
        fake_input = _FakeInput(["yes", str(models)])
        stdin_patch, input_patch = _interactive(fake_input)
        with stdin_patch, input_patch:
            result = InteractiveFallbackStrategy().fetch(str(tmp_path / "missing_models"), str(tmp_path / "cache"))

        assert result == str(models)
        pipeline_caplog.assert_logged(LOG, str(models))
        assert fake_input.prompts == [LOCAL_COPY_PROMPT, MODELS_PATH_PROMPT]
