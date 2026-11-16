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
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests import pypi_dependency_check
from tests.log_assertions import PipelineLogCapture
from tests.stage2_skeletons import SIDES, SKELETON_FILE, skeleton_volume

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

# Hypothesis profile for the filesystem-bound property tests: a loaded machine
# must not turn timing into failures (DeadlineExceeded, FlakyFailure, too_slow).
# Skipped when hypothesis is not installed.
try:
    from hypothesis import HealthCheck
    from hypothesis import settings as _hyp_settings

    _hyp_settings.register_profile(
        "champollion_load_proof",
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    _hyp_settings.load_profile("champollion_load_proof")
except ImportError:  # pragma: no cover
    pass


def pytest_sessionstart(session):  # noqa: V103 - pytest hook
    """Stop the session with one message when the environment lacks a pixi PyPI dependency.

    Without it a merge run in such an environment fails with hundreds of collection errors.
    Under xdist only the controller checks; its workers inherit the same environment.
    """
    if hasattr(session.config, "workerinput"):
        return
    message = pypi_dependency_check.check_environment()
    if message:
        pytest.exit(message, returncode=3)


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
def _stub_stage2_skeletons(monkeypatch):
    """Let a test of stage 2 that is not about skeletons pass the skeleton check of the input QC.

    Lays both skeletons of each input subject before the input QC runs, and gives aims.read a usable volume meanwhile.
    tests/test_run_cortical_tiles_input_qc_skeleton.py overrides this fixture to run the real check.
    """
    module = sys.modules.get("champollion_pipeline.run_cortical_tiles")
    if module is None:
        return
    write_input_qc = module.RunCorticalTiles._write_input_qc

    def write_input_qc_on_skeletons(script, input_abs, *args, **kwargs):
        for subject in module.list_input_subjects(input_abs):
            folder = Path(input_abs, subject, script.args.path_sk_with_hull.replace("*", "acquisition"))
            folder.mkdir(parents=True, exist_ok=True)
            for side in SIDES:
                (folder / SKELETON_FILE.format(side=side, subject=subject)).touch()
        with patch.object(module.aims, "read", return_value=skeleton_volume()):
            return write_input_qc(script, input_abs, *args, **kwargs)

    monkeypatch.setattr(module.RunCorticalTiles, "_write_input_qc", write_input_qc_on_skeletons)


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
def _stub_stage4_output_qc(monkeypatch):
    """Let a test that mocks evaluate.py reach the end of run() with no valid embeddings CSV on disk.

    tests/test_generate_embeddings_output_qc.py overrides this fixture to run the real check.
    """
    module = sys.modules.get("champollion_pipeline.generate_embeddings")
    if module is not None:
        monkeypatch.setattr(module.GenerateEmbeddings, "_run_output_qc", MagicMock(return_value=0), raising=False)


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage3_input_qc(monkeypatch):
    """Let a test of generate_champollion_config.py reach the end of run() with no complete crop on disk.

    One passing subject, no report on disk. raising=False: both names join the module with the input QC.
    tests/test_generate_champollion_config_input_qc.py overrides this fixture to run the real check.
    """
    module = sys.modules.get("champollion_pipeline.generate_champollion_config")
    if module is not None:
        row = {"participant_id": "sub-stub", "qc": 1, "reason": ""}
        monkeypatch.setattr(module, "fetch_crop_qc_rows", MagicMock(return_value=[row]), raising=False)
        monkeypatch.setattr(module, "write_qc_tsv", MagicMock(), raising=False)


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage3_output_qc(monkeypatch):
    """Let a test of generate_champollion_config.py reach the end of run() with no usable config on disk.

    No report on disk, status 0. raising=False: the name joins the class with the output QC.
    tests/test_generate_champollion_config_output_qc.py overrides this fixture to run the real check.
    """
    module = sys.modules.get("champollion_pipeline.generate_champollion_config")
    if module is not None:
        stub = MagicMock(return_value=0)
        monkeypatch.setattr(module.GenerateChampollionConfig, "_run_output_qc", stub, raising=False)


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage5_input_qc(monkeypatch):
    """Stub the stage 5 input QC: the copy tests of put_together_embeddings write region files that are not embeddings.

    PutTogetherEmbeddings._run_input_qc returns the input QC status.
    tests/test_put_together_embeddings_input_qc.py overrides this fixture to run the real check.
    """
    module = sys.modules.get("champollion_pipeline.put_together_embeddings")
    if module is not None:
        monkeypatch.setattr(module.PutTogetherEmbeddings, "_run_input_qc", MagicMock(return_value=0), raising=False)


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage5_output_qc(monkeypatch):
    """Let a test of put_together_embeddings.py reach the end of run() with no valid copy on disk.

    No report on disk, status 0. raising=False: the name joins the class with the output QC.
    tests/test_put_together_embeddings_output_qc.py overrides this fixture to run the real check.
    """
    module = sys.modules.get("champollion_pipeline.put_together_embeddings")
    if module is not None:
        monkeypatch.setattr(module.PutTogetherEmbeddings, "_run_output_qc", MagicMock(return_value=0), raising=False)


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
