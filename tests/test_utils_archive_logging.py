"""LOGGING convention in champollion_pipeline.utils.archive (TASK-PYTHONIC-BDRABCZUK-17F097040EE7).

The archive helpers report their extraction and cache steps as progress (LOG) records on the
champollion_pipeline.utils.archive logger instead of print(); tests assert with caplog, not capsys.

REQ-LOGGING-BDRABCZUK-935AC8217F9B: a non-empty tar extraction -> LOG "Extracted to: <returned path>".
REQ-LOGGING-BDRABCZUK-D8372D08113A: a plain .gz decompression -> LOG "Extracted to: <returned file path>".
REQ-LOGGING-BDRABCZUK-35E35AED53FA: a reused cached extraction -> LOG "Using cached extraction: <cached_path>".
REQ-LOGGING-BDRABCZUK-2676AF2360F4: a deleted stale cache -> LOG "Removing existing cache: <cached_path>".
REQ-LOGGING-BDRABCZUK-C96C8DAAF703 / -22ABEF2F2CBD (no print call, module logger) are checked by
tests/test_logging_convention.py through LOG_CONVERTED_MODULES.
"""

import gzip
import tarfile

from champollion_pipeline.utils import archive
from tests.log_assertions import LOG

ARCHIVE_LOGGER = "champollion_pipeline.utils.archive"


def _assert_archive_record(pipeline_caplog, text):
    record = pipeline_caplog.assert_logged(LOG, text)
    assert record.name == ARCHIVE_LOGGER
    return record


class TestArchiveLogging:
    def test_tar_extraction_logged_at_log(self, tmp_path, pipeline_caplog):
        weights = tmp_path / "weights.txt"
        weights.write_text("weights")
        path = tmp_path / "models.tar.gz"
        with tarfile.open(path, "w:gz") as tar:
            tar.add(weights, arcname="models/weights.txt")

        result = archive.extract_archive(str(path), str(tmp_path / "out"), top_level=True)

        _assert_archive_record(pipeline_caplog, f"Extracted to: {result}")

    def test_gzip_decompression_logged_at_log(self, tmp_path, pipeline_caplog):
        path = tmp_path / "weights.bin.gz"
        with gzip.open(path, "wb") as f_out:
            f_out.write(b"weights")

        result = archive.extract_archive(str(path), str(tmp_path / "out"), top_level=False)

        _assert_archive_record(pipeline_caplog, f"Extracted to: {result}")

    def test_reused_cache_logged_at_log(self, tmp_path, pipeline_caplog):
        cached = tmp_path / "models"
        cached.mkdir()

        assert archive.reuse_cached_extraction(str(cached), False) is True

        _assert_archive_record(pipeline_caplog, f"Using cached extraction: {cached}")

    def test_cache_removal_logged_at_log(self, tmp_path, pipeline_caplog):
        cached = tmp_path / "models"
        cached.mkdir()

        archive.remove_stale_cache(str(cached), True)

        assert not cached.exists()
        _assert_archive_record(pipeline_caplog, f"Removing existing cache: {cached}")
