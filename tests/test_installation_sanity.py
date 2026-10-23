#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Installation sanity checks.

Run with: pixi run test-smoke

These tests catch environment contamination and missing dependencies early,
before users hit cryptic errors deep in the pipeline.
"""

import importlib
import sys

import pytest


@pytest.mark.smoke
def test_gmpy2_has_valid_version():
    """gmpy2 must expose a version string.

    A None version means a system gmpy2 without pip metadata is leaking into
    the pixi environment (seen on shared Neurospin installs). This causes sympy
    to crash with TypeError when importing torch -> torchvision -> sympy.

    Fix: export SYMPY_GROUND_TYPES=python
    Or:  pip install --force-reinstall --no-deps gmpy2==2.2.1
    """
    gmpy2 = importlib.import_module("gmpy2")

    assert gmpy2.__version__ is not None, (
        "gmpy2.__version__ is None — a system gmpy2 without pip metadata is "
        "leaking into the pixi environment.\n"
        "Quick fix:  export SYMPY_GROUND_TYPES=python\n"
        "Proper fix: pip install --force-reinstall --no-deps gmpy2==2.2.1"
    )


@pytest.mark.smoke
def test_sympy_imports_cleanly():
    """sympy must import without crashing.

    A broken gmpy2 version causes sympy.external.gmpy.version_tuple() to
    receive None instead of a string, raising TypeError on import.
    """
    try:
        importlib.import_module("sympy")
    except TypeError as e:
        pytest.fail(
            f"sympy import raised TypeError — likely caused by gmpy2.__version__ "
            f"being None. Run test_gmpy2_has_valid_version for diagnosis.\n"
            f"Original error: {e}"
        )


@pytest.mark.smoke
def test_torch_imports():
    """torch must be importable.

    Catches broken PyTorch installs (missing CUDA libraries, wrong platform
    wheels, etc.). The pipeline can run on CPU, but torch must import cleanly.
    """
    try:
        importlib.import_module("torch")
    except ImportError as e:
        pytest.fail(
            f"torch is not importable. Re-run 'pixi run install-all' or check "
            f"that the linux-64 platform wheels are installed.\nError: {e}"
        )


@pytest.mark.smoke
def test_champollion_utils_importable():
    """champollion_utils must be installed (ScriptBuilder base class).

    Fails when the champollion_utils sibling repo was not cloned or its
    pip editable install was not run ('pixi run install-utils').
    """
    try:
        script_builder = importlib.import_module("champollion_utils.script_builder")
    except ImportError as e:
        pytest.fail(f"champollion_utils is not installed. Run: pixi run install-utils\nError: {e}")
    assert hasattr(script_builder, "ScriptBuilder"), "champollion_utils.script_builder defines no ScriptBuilder"


@pytest.mark.smoke
def test_cortical_tiles_importable():
    """cortical_tiles (the external/cortical_tiles submodule) must be installed.

    Upstream renamed its top-level package ``deep_folding`` -> ``cortical_tiles``
    in merge ``a194dc1``; importing the old name now always fails, so this check
    must use the current one to stay a test of the installation rather than of
    the rename.

    Fails when the cortical_tiles submodule was not initialised or its
    pip editable install was not run ('pixi run install-cortical-tiles').
    """
    try:
        importlib.import_module("cortical_tiles")
    except ImportError as e:
        pytest.fail(
            f"cortical_tiles is not installed.\n"
            f"Run: pixi run install-cortical-tiles\n"
            f"Or:  pixi run install-all\n"
            f"Error: {e}"
        )


@pytest.mark.smoke
def test_pandas_available():
    """pandas must be importable (used for QC TSV filtering throughout pipeline)."""
    try:
        importlib.import_module("pandas")
    except ImportError as e:
        pytest.fail(f"pandas is not importable: {e}")


@pytest.mark.smoke
def test_huggingface_hub_available():
    """huggingface_hub must be importable (used to download model weights)."""
    try:
        importlib.import_module("huggingface_hub")
    except ImportError as e:
        pytest.fail(f"huggingface_hub is not importable: {e}")


@pytest.mark.smoke
def test_pipeline_src_on_path():
    """src/ directory must be on sys.path so pipeline scripts are importable.

    Fails when tests are run from the wrong directory or conftest.py did not
    add src/ to sys.path.
    """
    try:
        lib = importlib.import_module("champollion_pipeline.utils.lib")
    except ImportError as e:
        pytest.fail(
            f"src/ is not on sys.path — run tests from the champollion_pipeline "
            f"root directory via 'pixi run test'.\nError: {e}"
        )
    assert hasattr(lib, "DERIVATIVES_FOLDER"), "champollion_pipeline.utils.lib defines no DERIVATIVES_FOLDER"


@pytest.mark.smoke
def test_python_version():
    """Python must be 3.10 or later (pixi env ships 3.11)."""
    major, minor = sys.version_info.major, sys.version_info.minor
    assert (major, minor) >= (3, 10), (
        f"Python {major}.{minor} detected — pipeline requires Python >=3.10. "
        f"Check that the pixi environment is activated."
    )
