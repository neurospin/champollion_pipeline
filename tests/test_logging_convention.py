"""LOGGING convention: caplog helper, level table and module rules (TASK-LOGGING-BDRABCZUK-0CCC059DB54E).

REQ-LOGGING-BDRABCZUK-2F6A4132B811: pipeline_caplog captures a WARNING from a module logger
    while the champollion_pipeline logger does not propagate.
REQ-LOGGING-BDRABCZUK-601E49CD02DB: pipeline_caplog captures a LOG record while the
    champollion_pipeline logger level was WARNING before the fixture started.
REQ-LOGGING-BDRABCZUK-525A4234278D: the champollion_pipeline logger level, propagate flag and
    handler list are restored when the capture ends.
REQ-LOGGING-BDRABCZUK-ADB0E9F7C151: assert_logged raises AssertionError when no record matches.
REQ-LOGGING-BDRABCZUK-83EECE591396: LEVEL_BY_KIND maps progress, result, warning, error, failure
    to LOG, OUTPUT, WARNING, ERROR, FAIL.
REQ-LOGGING-BDRABCZUK-22ABEF2F2CBD: a module of LOG_CONVERTED_MODULES binds ``logger`` to the
    logger named after the module.
REQ-LOGGING-BDRABCZUK-C96C8DAAF703: a module of LOG_CONVERTED_MODULES calls print() nowhere.
REQ-LOGGING-BDRABCZUK-72C6D5EBD254: per-fold progress is logged below WARNING (stdout) with
    "Fold <n>/<m>" in the message, as champollion_sulcal_mcp parses it.
"""

import ast
import importlib
import inspect
import logging
import re
import textwrap
from types import ModuleType

import pytest
from champollion_utils.logger import FAIL, LOG, OUTPUT, setup_logging

from tests.log_assertions import (
    LEVEL_BY_KIND,
    LOG_CONVERTED_MODULES,
    PIPELINE_LOGGER_NAME,
    PipelineLogCapture,
    module_logger_problem,
    print_call_lines,
)

# Copy of champollion_sulcal_mcp runner.FOLD_RE (it reads the stage output line by line).
MCP_FOLD_RE = re.compile(r"[Ff]old\s+(\d+)\s*(?:/|of)\s*(\d+)")


@pytest.fixture  # noqa: V103 - requested through usefixtures
def pipeline_logger_at_warning():
    """Set the champollion_pipeline logger level to WARNING before pipeline_caplog starts."""
    pipeline_logger = logging.getLogger(PIPELINE_LOGGER_NAME)
    level = pipeline_logger.level
    pipeline_logger.setLevel(logging.WARNING)
    yield pipeline_logger
    pipeline_logger.setLevel(level)


def test_pipeline_caplog_captures_warning_when_pipeline_logger_does_not_propagate(monkeypatch, pipeline_caplog):
    # main.py's _setup_logger sets propagate=False when console logging is off.
    monkeypatch.setattr(logging.getLogger(PIPELINE_LOGGER_NAME), "propagate", False)

    logging.getLogger("champollion_pipeline.generate_masks").warning("mask volume missing for sub-01")

    record = pipeline_caplog.assert_logged(logging.WARNING, "mask volume missing for sub-01")
    assert record.name == "champollion_pipeline.generate_masks"


@pytest.mark.usefixtures("pipeline_logger_at_warning")
def test_pipeline_caplog_captures_log_level_below_pipeline_logger_level(pipeline_caplog):
    logging.getLogger("champollion_pipeline.generate_embeddings").log(LOG, "Fold 3/5 done")

    record = pipeline_caplog.assert_logged(LOG, "Fold 3/5 done")
    assert record.levelno == LOG


def test_pipeline_log_capture_restores_pipeline_logger_state(monkeypatch, caplog):
    pipeline_logger = logging.getLogger(PIPELINE_LOGGER_NAME)
    sentinel_handler = logging.NullHandler()
    monkeypatch.setattr(pipeline_logger, "handlers", [sentinel_handler])
    monkeypatch.setattr(pipeline_logger, "propagate", False)
    monkeypatch.setattr(pipeline_logger, "level", logging.ERROR)

    capture = PipelineLogCapture(caplog)
    capture.start()
    logging.getLogger("champollion_pipeline.config").warning("config key ignored")
    assert [r.getMessage() for r in capture.records] == ["config key ignored"]
    capture.stop()

    assert pipeline_logger.level == logging.ERROR
    assert pipeline_logger.propagate is False
    assert pipeline_logger.handlers == [sentinel_handler]


def test_assert_logged_raises_when_no_record_matches(pipeline_caplog):
    logging.getLogger("champollion_pipeline.purge_subject").warning("subject sub-02 not found")

    with pytest.raises(AssertionError):
        pipeline_caplog.assert_logged(logging.ERROR, "subject sub-02 not found")
    with pytest.raises(AssertionError):
        pipeline_caplog.assert_logged(logging.WARNING, "subject sub-03 not found")


def test_assert_logged_returns_matching_record(pipeline_caplog):
    logging.getLogger("champollion_pipeline.purge_subject").warning("subject sub-02 not found")

    record = pipeline_caplog.assert_logged(logging.WARNING, "sub-02 not found")

    assert record.levelno == logging.WARNING
    assert record.getMessage() == "subject sub-02 not found"


def test_level_by_kind_table():
    assert LEVEL_BY_KIND == {
        "progress": LOG,
        "result": OUTPUT,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "failure": FAIL,
    }


def _fake_module(name: str, logger_name: str | None) -> ModuleType:
    module = ModuleType(name)
    if logger_name is not None:
        module.logger = logging.getLogger(logger_name)
    return module


def test_module_logger_problem_accepts_logger_named_after_module():
    module = _fake_module("champollion_pipeline.fake_stage", "champollion_pipeline.fake_stage")

    assert module_logger_problem(module) is None


def test_module_logger_problem_reports_misnamed_logger():
    module = _fake_module("champollion_pipeline.fake_stage", PIPELINE_LOGGER_NAME)

    assert module_logger_problem(module)


def test_module_logger_problem_reports_missing_logger():
    module = _fake_module("champollion_pipeline.fake_stage", None)

    assert module_logger_problem(module)


def test_print_call_lines_finds_print_call():
    source = textwrap.dedent(
        """\
        import logging
        logger = logging.getLogger(__name__)
        print("progress")
        logger.info("progress")
        report.print()
        """
    )

    assert print_call_lines(ast.parse(source)) == [3]


@pytest.mark.parametrize("module_name", LOG_CONVERTED_MODULES)
def test_log_converted_module_binds_module_logger(module_name):
    module = importlib.import_module(module_name)

    assert module_logger_problem(module) is None


@pytest.mark.parametrize("module_name", LOG_CONVERTED_MODULES)
def test_log_converted_module_has_no_print_call(module_name):
    module = importlib.import_module(module_name)

    assert print_call_lines(ast.parse(inspect.getsource(module))) == []


def test_assert_fold_progress_returns_record_below_warning(pipeline_caplog):
    logging.getLogger("champollion_pipeline.generate_embeddings").log(LOG, "Fold 2/5 embedded: S.C.-sylv._left")

    record = pipeline_caplog.assert_fold_progress(2, 5)

    assert record.levelno < logging.WARNING


def test_assert_fold_progress_raises_for_warning_record(pipeline_caplog):
    logging.getLogger("champollion_pipeline.generate_embeddings").warning("Fold 2/5 embedded: S.C.-sylv._left")

    with pytest.raises(AssertionError):
        pipeline_caplog.assert_fold_progress(2, 5)


def test_fold_progress_line_reaches_stdout_for_mcp_parser(monkeypatch, capsys):
    # Root handlers are restored by the autouse restore_logging_and_excepthooks fixture.
    setup_logging()
    pipeline_logger = logging.getLogger(PIPELINE_LOGGER_NAME)
    monkeypatch.setattr(pipeline_logger, "propagate", True)
    monkeypatch.setattr(pipeline_logger, "level", logging.NOTSET)

    logging.getLogger("champollion_pipeline.generate_embeddings").log(
        LEVEL_BY_KIND.get("progress", logging.WARNING), "Fold 2/5 embedded: S.C.-sylv._left"
    )

    matches = [MCP_FOLD_RE.search(line) for line in capsys.readouterr().out.splitlines()]
    assert [m.groups() for m in matches if m] == [("2", "5")]
