#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures for champollion_pipeline tests.
"""

import os
import shutil
import sys
import tempfile
from unittest.mock import MagicMock

import pytest

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
