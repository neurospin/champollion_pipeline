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
    import gmpy2
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
        import sympy  # noqa: F401
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
        import torch  # noqa: F401
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
        from champollion_utils.script_builder import ScriptBuilder  # noqa: F401
    except ImportError as e:
        pytest.fail(
            f"champollion_utils is not installed. Run: pixi run install-utils\n"
            f"Error: {e}"
        )


@pytest.mark.smoke
def test_cortical_tiles_importable():
    """deep_folding (cortical_tiles submodule) must be installed.

    Fails when the cortical_tiles submodule was not initialised or its
    pip editable install was not run ('pixi run install-cortical-tiles').
    """
    try:
        import deep_folding  # noqa: F401
    except ImportError as e:
        pytest.fail(
            f"deep_folding (cortical_tiles) is not installed.\n"
            f"Run: pixi run install-cortical-tiles\n"
            f"Or:  pixi run install-all\n"
            f"Error: {e}"
        )


@pytest.mark.smoke
def test_pandas_available():
    """pandas must be importable (used for QC TSV filtering throughout pipeline)."""
    try:
        import pandas  # noqa: F401
    except ImportError as e:
        pytest.fail(f"pandas is not importable: {e}")


@pytest.mark.smoke
def test_huggingface_hub_available():
    """huggingface_hub must be importable (used to download model weights)."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as e:
        pytest.fail(f"huggingface_hub is not importable: {e}")


@pytest.mark.smoke
def test_pipeline_src_on_path():
    """src/ directory must be on sys.path so pipeline scripts are importable.

    Fails when tests are run from the wrong directory or conftest.py did not
    add src/ to sys.path.
    """
    try:
        from utils.lib import DERIVATIVES_FOLDER  # noqa: F401
    except ImportError as e:
        pytest.fail(
            f"src/ is not on sys.path — run tests from the champollion_pipeline "
            f"root directory via 'pixi run test'.\nError: {e}"
        )


@pytest.mark.smoke
def test_python_version():
    """Python must be 3.10 or later (pixi env ships 3.11)."""
    major, minor = sys.version_info.major, sys.version_info.minor
    assert (major, minor) >= (3, 10), (
        f"Python {major}.{minor} detected — pipeline requires Python >=3.10. "
        f"Check that the pixi environment is activated."
    )
