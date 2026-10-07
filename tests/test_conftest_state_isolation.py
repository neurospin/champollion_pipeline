"""Tests for REQ-TESTISOL-04 (TASK-231) — per-test isolation of process-wide logging state.

``tests/conftest.py`` must restore, after every test, four pieces of
process-wide state a test may change: the root logger's handler list, the
root logger's level, ``sys.excepthook`` and ``threading.excepthook``. Once
``ScriptBuilder.main()`` calls ``champollion_utils.init_process()``, tests that
run the real ``main()`` in-process would otherwise leave all four modified
for every later test.

Design: each test runs an inner pytest session (``pytester``) in a fresh
subprocess, with the *real* ``tests/conftest.py`` copied in as the inner
conftest and one inner module holding two tests, ``test_a_mutates`` then
``test_b_observes`` (pytest runs a module's tests in file order). A changes
one piece of state; B asserts it sees the pre-A value. The outer test asserts
the inner session passed.

Why this shape rather than two sibling tests in this file:
- Ordering is guaranteed: the inner session is a single process with no
  xdist, so A always runs before B. Two outer sibling tests could be split
  across workers under ``-n auto --dist loadscope`` only if moved into a
  class/other module, and any future reordering plugin would break them.
- No leakage into the outer session: ``runpytest_subprocess`` keeps A's
  mutation (which, while red, is never undone) out of this process, so the
  red state cannot pollute later tests in the real run.
- No dependency on ``champollion_utils.init_process()``: A mutates state
  directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]  # noqa: V107

REAL_CONFTEST = Path(__file__).resolve().parent / "conftest.py"

_ROOT_HANDLERS = """
import logging


class _LeakedHandler(logging.Handler):
    def emit(self, record):
        pass


def test_a_mutates():
    logging.getLogger().addHandler(_LeakedHandler())


def test_b_observes():
    leaked = [h for h in logging.getLogger().handlers if type(h).__name__ == "_LeakedHandler"]
    assert leaked == [], f"root logger handler leaked from previous test: {leaked!r}"
"""

_ROOT_LEVEL = """
import logging

BASELINE = logging.getLogger().level
MUTATED = logging.DEBUG if BASELINE != logging.DEBUG else logging.CRITICAL


def test_a_mutates():
    logging.getLogger().setLevel(MUTATED)


def test_b_observes():
    level = logging.getLogger().level
    assert level == BASELINE, (
        f"root logger level leaked from previous test: {logging.getLevelName(level)} "
        f"(expected {logging.getLevelName(BASELINE)})"
    )
"""

_SYS_EXCEPTHOOK = """
import sys

BASELINE = sys.excepthook


def _leaked_hook(exc_type, exc, tb):
    pass


def test_a_mutates():
    sys.excepthook = _leaked_hook


def test_b_observes():
    assert sys.excepthook is BASELINE, f"sys.excepthook leaked from previous test: {sys.excepthook!r}"
"""

_THREADING_EXCEPTHOOK = """
import threading

_STATE = {}


def _leaked_hook(args):
    pass


def test_a_mutates():
    _STATE["before"] = threading.excepthook
    threading.excepthook = _leaked_hook


def test_b_observes():
    assert threading.excepthook is not _leaked_hook, (
        f"threading.excepthook leaked from previous test: {threading.excepthook!r}"
    )
    assert threading.excepthook is _STATE["before"], (
        f"threading.excepthook not restored: {threading.excepthook!r} (expected {_STATE['before']!r})"
    )
"""


@pytest.mark.parametrize(
    "inner_module",
    [
        pytest.param(_ROOT_HANDLERS, id="root_logger_handlers"),
        pytest.param(_ROOT_LEVEL, id="root_logger_level"),
        pytest.param(_SYS_EXCEPTHOOK, id="sys_excepthook"),
        pytest.param(_THREADING_EXCEPTHOOK, id="threading_excepthook"),
    ],
)
def test_conftest_restores_process_wide_state_after_each_test(pytester, inner_module):
    """A test's change to one piece of process-wide state is undone before the next test runs."""
    pytester.makeconftest(REAL_CONFTEST.read_text(encoding="utf-8"))
    pytester.makepyfile(test_inner=inner_module)

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-p", "no:randomly", "-v")

    result.assert_outcomes(passed=2)
