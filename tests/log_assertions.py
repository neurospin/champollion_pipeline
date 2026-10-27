"""Log assertion helpers for the LOGGING convention (TASK-LOGGING-BDRABCZUK-0CCC059DB54E).

Stage messages go through module loggers (``logger = logging.getLogger(__name__)``)
instead of print(). These helpers let tests assert those records by level and
message, and check the modules that adopted the convention.
"""

import ast
import logging
from types import ModuleType

import pytest

# Logger every champollion_pipeline module logger descends from.
PIPELINE_LOGGER_NAME = "champollion_pipeline"

# Logger every compare_sulcal module logger descends from (not under champollion_pipeline).
COMPARE_LOGGER_NAME = "compare_sulcal"

# champollion_utils.logger levels, as numbers so conftest (which imports this module) still
# collects without champollion_utils; test_logging_convention pins them to that module.
LOG = 22
OUTPUT = 25
FAIL = 45

# Message kind -> log level (REQ-LOGGING-BDRABCZUK-83EECE591396).
LEVEL_BY_KIND: dict[str, int] = {
    "progress": LOG,
    "result": OUTPUT,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "failure": FAIL,
}

# Import names of the modules that follow the LOGGING convention; each later
# LOGGING task adds its modules here.
LOG_CONVERTED_MODULES: tuple[str, ...] = (
    "champollion_pipeline.generate_masks",
    "compare_sulcal.cli",
    "compare_sulcal.crops",
    "compare_sulcal.databases",
    "compare_sulcal.embeddings",
    "compare_sulcal.viewer",
    "champollion_pipeline.models.fetch",
)


class PipelineLogCapture:
    """Captures the records of the champollion_pipeline loggers through caplog."""

    def __init__(self, caplog: pytest.LogCaptureFixture) -> None:
        self._caplog = caplog
        self._saved: tuple[int, bool, list[logging.Handler]] | None = None

    def start(self) -> None:
        """Start capturing the champollion_pipeline records at every level.

        caplog's handler is also attached to the champollion_pipeline logger, so
        records are captured even if code under test turns propagation off.
        """
        pipeline = logging.getLogger(PIPELINE_LOGGER_NAME)
        self._saved = (pipeline.level, pipeline.propagate, list(pipeline.handlers))
        pipeline.propagate = True
        pipeline.setLevel(1)
        pipeline.addHandler(self._caplog.handler)

    def stop(self) -> None:
        """Stop capturing and restore the champollion_pipeline level, propagate flag and handlers."""
        if self._saved is None:
            return
        level, propagate, handlers = self._saved
        pipeline = logging.getLogger(PIPELINE_LOGGER_NAME)
        pipeline.setLevel(level)
        pipeline.propagate = propagate
        pipeline.handlers[:] = handlers
        self._saved = None

    @property
    def records(self) -> list[logging.LogRecord]:
        """Records of the champollion_pipeline loggers captured in the current test phase, each once."""
        prefix = f"{PIPELINE_LOGGER_NAME}."
        # A propagating record reaches caplog's handler twice: on the pipeline logger and on the root.
        unique = {
            id(record): record
            for record in self._caplog.records
            if record.name == PIPELINE_LOGGER_NAME or record.name.startswith(prefix)
        }
        return list(unique.values())

    def _first(self, predicate, wanted: str) -> logging.LogRecord:
        """Return the first record matching ``predicate``, else fail listing the captured records."""
        for record in self.records:
            if predicate(record):
                return record
        captured = [(record.levelname, record.getMessage()) for record in self.records]
        raise AssertionError(f"no {wanted} record; captured: {captured}")

    def assert_logged(self, level: int, text: str) -> logging.LogRecord:
        """Return a captured record at ``level`` whose message contains ``text``."""
        return self._first(
            lambda record: record.levelno == level and text in record.getMessage(),
            f"{logging.getLevelName(level)} {text!r}",
        )

    def assert_fold_progress(self, fold: int, folds: int) -> logging.LogRecord:
        """Return a captured record below WARNING whose message contains ``Fold <fold>/<folds>``."""
        text = f"Fold {fold}/{folds}"
        return self._first(
            lambda record: record.levelno < logging.WARNING and text in record.getMessage(),
            f"below-WARNING {text!r}",
        )


def logged_messages(caplog: pytest.LogCaptureFixture, logger_name: str, level: int) -> list[str]:
    """Messages of the records caplog captured on ``logger_name`` at ``level``, in emission order."""
    return [record.getMessage() for record in caplog.records if record.name == logger_name and record.levelno == level]


def module_logger_problem(module: ModuleType) -> str | None:
    """Describe why ``module`` does not bind ``logger`` to its own named logger, or None."""
    if not hasattr(module, "logger"):
        return f"{module.__name__} has no module-level logger"
    logger = module.logger
    if not isinstance(logger, logging.Logger):
        return f"{module.__name__}.logger is a {type(logger).__name__}, not a logging.Logger"
    if logger.name != module.__name__:
        return f"{module.__name__}.logger is named {logger.name!r}, expected {module.__name__!r}"
    return None


def print_call_lines(tree: ast.Module) -> list[int]:
    """Line numbers of the calls to the built-in print function in ``tree``."""
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
    )
