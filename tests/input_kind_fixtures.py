"""Pytest fixtures shared by the input kind test files.

Import a fixture by name in the test module that requests it (``from tests.input_kind_fixtures import read_folders``);
pytest then finds it in that module's namespace.
"""

import os

import pytest

from tests.main_input_autodetect_harness import _main_module_under_test

HEADER_READERS = (
    "champollion_pipeline.utils.nifti_check.check_nifti_input",
    "champollion_pipeline.generate_morphologist_graphs.check_nifti_input",
)


@pytest.fixture
def read_folders(monkeypatch) -> list[str]:
    """Record, in read order, each folder passed to os.scandir."""
    folders: list[str] = []
    real_scandir = os.scandir

    def recording_scandir(path="."):
        folders.append(os.fspath(path))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", recording_scandir)
    return folders


@pytest.fixture
def main_module(monkeypatch, tmp_path):
    """main.py loaded as a module for one test, by the autodetect harness."""
    return _main_module_under_test(monkeypatch, tmp_path)


@pytest.fixture
def header_reads(monkeypatch) -> list[str]:
    """Replace the NIfTI header reader where main.py and stage 1 reach it; return the paths it was asked to read."""
    reads: list[str] = []

    def record_read(path, *_args, **_kwargs):
        reads.append(str(path))
        return ()

    for target in HEADER_READERS:
        monkeypatch.setattr(target, record_read)
    return reads
