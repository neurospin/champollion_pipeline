#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures for champollion_pipeline tests.
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_file_structure(temp_dir):
    """Create a mock file structure for testing."""
    structure = {
        'input': Path(temp_dir) / 'input',
        'output': Path(temp_dir) / 'output',
        'crops': Path(temp_dir) / 'crops',
        'graphs': Path(temp_dir) / 'graphs',
        'models': Path(temp_dir) / 'models',
    }

    for path in structure.values():
        path.mkdir(parents=True, exist_ok=True)

    # Create some mock files
    (structure['input'] / 'test_file.nii.gz').touch()
    (structure['input'] / 'test_file2.nii').touch()
    (structure['crops'] / 'crop1.nii').touch()

    return structure


@pytest.fixture
def mock_args():
    """Create mock argparse arguments."""
    class MockArgs:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return MockArgs


@pytest.fixture
def mock_subprocess_success():
    """Mock subprocess calls to return success."""
    with patch('subprocess.check_call', return_value=0):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            yield mock_run


@pytest.fixture
def mock_subprocess_failure():
    """Mock subprocess calls to return failure."""
    with patch('subprocess.check_call', side_effect=Exception("Command failed")):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            yield mock_run


@pytest.fixture
def sample_yaml_content():
    """Sample YAML content for testing."""
    return """
dataset_folder: /path/to/dataset
some_key: some_value
another_key: another_value
"""


@pytest.fixture
def sample_reference_yaml():
    """Sample reference.yaml content."""
    return """
dataset_path: TESTXX/some/path
config_key: config_value
"""


@pytest.fixture
def mock_validate_paths_success():
    """Mock validate_paths to always return True."""
    def mock_validate(paths):
        return True
    return mock_validate


@pytest.fixture
def mock_validate_paths_failure():
    """Mock validate_paths to always return False."""
    def mock_validate(paths):
        return False
    return mock_validate


@pytest.fixture
def script_builder_subclass():
    """Create a simple ScriptBuilder subclass for testing."""
    from champollion_utils.script_builder import ScriptBuilder

    class TestScript(ScriptBuilder):
        def __init__(self):
            super().__init__(
                script_name="test_script",
                description="Test script for unit tests"
            )
            self.add_argument("input", help="Input path")
            self.add_optional_argument("--output", "Output path", default="/tmp")

        def run(self):
            return 0

    return TestScript


@pytest.fixture(autouse=True)
def reset_cwd():
    """Reset current working directory after each test."""
    original_cwd = os.getcwd()
    yield
    os.chdir(original_cwd)


@pytest.fixture
def mock_file_list():
    """Mock list of input files."""
    return [
        "subject01.nii.gz",
        "subject02.nii.gz",
        "subject03.nii",
    ]


@pytest.fixture
def mock_cpu_count():
    """Mock CPU count."""
    with patch('joblib.cpu_count', return_value=8):
        yield 8


# Helper functions for tests

def create_test_file(path, content=""):
    """Create a test file with optional content."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)


def create_test_dir(path):
    """Create a test directory."""
    Path(path).mkdir(parents=True, exist_ok=True)
