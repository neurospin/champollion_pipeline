"""LOGGING convention for put_together_embeddings.py (TASK-LOGGING-BDRABCZUK-1761ECDBE832).

PutTogetherEmbeddings reports through the champollion_pipeline.put_together_embeddings logger
instead of print().

REQ-LOGGING-BDRABCZUK-A0C1CBA4176F: run() logs the embeddings_source path at LOG.
REQ-LOGGING-BDRABCZUK-2755E39A8331: run() logs the --output_path value at LOG.
REQ-LOGGING-BDRABCZUK-C90FB2BAA2E3: a region without full_embeddings.csv is logged at WARNING with its name.
REQ-LOGGING-BDRABCZUK-853BB2689827: the copied file count and --output_path are logged at OUTPUT.
REQ-LOGGING-BDRABCZUK-DE70E7A05D35: zero copied embeddings are logged at WARNING with the source path.
(REQ-LOGGING-BDRABCZUK-C96C8DAAF703 and -22ABEF2F2CBD: tests/test_logging_convention.py, through
LOG_CONVERTED_MODULES.)
"""

import logging
import re
from pathlib import Path

import pytest

from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings
from tests.log_assertions import LOG, OUTPUT

pytestmark = pytest.mark.unit  # noqa: V107

LOGGER_NAME = "champollion_pipeline.put_together_embeddings"
REGION_WITH_CSV = "SC-sylv_right"
REGION_WITHOUT_CSV = "SC-sylv_left"


def _source(tmp_path: Path, regions_with_csv: tuple[str, ...], regions_without_csv: tuple[str, ...]) -> Path:
    """Per-region embeddings directory; only regions_with_csv hold a full_embeddings.csv."""
    source = tmp_path / "region_embeddings"
    source.mkdir()
    for region in regions_with_csv:
        (source / region).mkdir()
        (source / region / "full_embeddings.csv").write_text("ID,dim1\nsub-01,0.5\n")
    for region in regions_without_csv:
        (source / region).mkdir()
    return source


def _run(source: Path, output: Path) -> None:
    script = PutTogetherEmbeddings()
    script.parse_args([str(source), "--output_path", str(output)])
    assert script.run() == 0


def test_embeddings_source_logged_at_log(tmp_path, pipeline_caplog):
    source = _source(tmp_path, (REGION_WITH_CSV,), ())

    _run(source, tmp_path / "combined")

    record = pipeline_caplog.assert_logged(LOG, str(source))
    assert record.name == LOGGER_NAME


def test_output_path_logged_at_log(tmp_path, pipeline_caplog):
    source = _source(tmp_path, (REGION_WITH_CSV,), ())
    output = tmp_path / "combined"

    _run(source, output)

    record = pipeline_caplog.assert_logged(LOG, str(output))
    assert record.name == LOGGER_NAME


def test_region_without_csv_logged_at_warning(tmp_path, pipeline_caplog):
    source = _source(tmp_path, (REGION_WITH_CSV,), (REGION_WITHOUT_CSV,))

    _run(source, tmp_path / "combined")

    record = pipeline_caplog.assert_logged(logging.WARNING, REGION_WITHOUT_CSV)
    assert record.name == LOGGER_NAME


def test_copied_count_logged_at_output(tmp_path, pipeline_caplog):
    source = _source(tmp_path, (REGION_WITH_CSV, "STs_right"), (REGION_WITHOUT_CSV,))
    output = tmp_path / "combined"

    _run(source, output)

    record = pipeline_caplog.assert_logged(OUTPUT, str(output))
    message = record.getMessage()
    assert re.search(r"\b2\b", message[: message.index(str(output))]), message
    assert record.name == LOGGER_NAME


def test_zero_copied_logged_at_warning(tmp_path, pipeline_caplog):
    # No region directory at all, so no per-region skip warning can carry the source path.
    source = _source(tmp_path, (), ())

    _run(source, tmp_path / "combined")

    record = pipeline_caplog.assert_logged(logging.WARNING, str(source))
    assert record.name == LOGGER_NAME
