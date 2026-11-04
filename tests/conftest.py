#!/usr/bin/env python3
"""
Pytest configuration and fixtures for champollion_pipeline tests.
"""

import logging
import os
import shutil
import sys
import tempfile
import threading
from unittest.mock import MagicMock

import pytest

from tests.log_assertions import PipelineLogCapture

# Keep src/ on path for non-package scripts (compare tools, file_indexer, etc.)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Stub all BrainVISA soma subpackages so any module that does top-level
# "from soma import aims" or "from soma.aimsalgo import ..." can be imported
# outside the brainvisa environment.  Each entry must be registered separately
# because Python's import machinery resolves submodules via sys.modules lookups,
# not via __getattr__ on the parent mock.
for _soma_mod in [
    "soma",
    "soma.aims",
    "soma.aimsalgo",
    "soma.aimsalgo.sulci",
    "soma.qt_gui",
    "soma.qt_gui.qt_backend",
]:
    sys.modules.setdefault(_soma_mod, MagicMock())


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(autouse=True)  # noqa: V103
def reset_cwd():
    """Reset current working directory after each test."""
    original_cwd = os.getcwd()
    yield
    os.chdir(original_cwd)


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage2_output_qc(monkeypatch):
    """Let a test that mocks the stage 2 crop step reach the end of run() with no crop on disk.

    tests/test_run_cortical_tiles_output_qc.py overrides this fixture to run the real check.
    """
    module = sys.modules.get("champollion_pipeline.run_cortical_tiles")
    if module is not None:
        monkeypatch.setattr(module.RunCorticalTiles, "_write_output_qc", MagicMock(return_value=("", True)))


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage4_input_qc(monkeypatch):
    """Let a test that mocks evaluate.py reach it with no crop on disk.

    GenerateEmbeddings._run_input_qc returns the input QC status: 0, 3 (partial) or 1 (no usable subject).
    tests/test_generate_embeddings_input_qc.py overrides this fixture to run the real check.
    """
    module = sys.modules.get("champollion_pipeline.generate_embeddings")
    if module is not None:
        monkeypatch.setattr(module.GenerateEmbeddings, "_run_input_qc", MagicMock(return_value=0), raising=False)


@pytest.fixture(autouse=True)  # noqa: V103
def restore_logging_and_excepthooks():
    """Restore the root logger handlers and level and the exception hooks after each test.

    Code under test may call champollion_utils.init_process() (e.g. through
    ScriptBuilder.main()), which configures the root logger and replaces
    sys.excepthook and threading.excepthook for the whole process.
    """
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    excepthook = sys.excepthook
    thread_excepthook = threading.excepthook
    yield
    for handler in root.handlers[:]:
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    for handler in handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)
    sys.excepthook = excepthook
    threading.excepthook = thread_excepthook


@pytest.fixture
def pipeline_caplog(caplog):
    """Capture the champollion_pipeline records, whatever its propagate flag and level.

    Returns a PipelineLogCapture: ``records`` and ``assert_logged(level, text)``.
    The champollion_pipeline logger is restored when the test finishes.
    """
    capture = PipelineLogCapture(caplog)
    capture.start()
    yield capture
    capture.stop()
