#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Behaviour of src/compare.py when PyAIMS (``soma.aims``) is not installed.

conftest.py stubs ``soma`` in this process, so every check here runs in a
fresh Python subprocess where ``soma`` / ``soma.aims`` are blocked through
``sys.modules[...] = None`` (which makes ``from soma import aims`` raise
ImportError). The remote update check is disabled in the child so no test
touches the network.

Requirements: REQ-COMPARE-01 (import succeeds), REQ-COMPARE-02 (mask/database
subcommands exit non-zero), REQ-COMPARE-03 (error message on stderr),
REQ-COMPARE-22 (crops subcommand exits non-zero naming soma.aims on stderr).
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"

_CHILD_PRELUDE = textwrap.dedent(
    f"""
    import sys
    sys.path.insert(0, {str(_SRC_DIR)!r})
    sys.modules["soma"] = None
    sys.modules["soma.aims"] = None
    import champollion_utils.script_builder as _sb
    _sb.check_for_updates = lambda *a, **k: None
    """
)

_IMPORT_MARKER = "COMPARE_IMPORTED_OK"


def _run_child(code: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _CHILD_PRELUDE + textwrap.dedent(code)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=120,
    )


def _run_compare(argv: list, cwd: Path) -> subprocess.CompletedProcess:
    code = f"""
    sys.argv = ["compare.py"] + {argv!r}
    import compare
    sys.exit(compare.main())
    """
    return _run_child(code, cwd)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _masks_argv(tmp_path: Path) -> list:
    for side in ("set_a", "set_b"):
        _touch(tmp_path / side / "L" / "S.C.nii.gz")
    return [
        "masks",
        "--set_a",
        str(tmp_path / "set_a"),
        "--set_b",
        str(tmp_path / "set_b"),
        "--output",
        str(tmp_path / "report.json"),
    ]


def _cortical_tiles_argv(tmp_path: Path) -> list:
    for side in ("set_a", "set_b"):
        _touch(tmp_path / side / "S.C._left" / "mask" / "Lmask_skeleton.nii.gz")
    return [
        "cortical_tiles",
        "--set_a",
        str(tmp_path / "set_a"),
        "--set_b",
        str(tmp_path / "set_b"),
        "--output",
        str(tmp_path / "report.json"),
    ]


def _databases_argv(tmp_path: Path) -> list:
    graph_rel = "t1mri/t1/default_analysis/folds/3.3"
    for campaign in ("a", "b"):
        _touch(tmp_path / "subjects" / "sub-01" / graph_rel / campaign / "Lsub-01_a.arg")
    return [
        "databases",
        "--labeled_subjects_dir",
        str(tmp_path / "subjects"),
        "--path_to_graph_a",
        f"{graph_rel}/a",
        "--path_to_graph_b",
        f"{graph_rel}/b",
        "--output",
        str(tmp_path / "db.csv"),
        "--njobs",
        "1",
    ]


_MODE_ARGV_BUILDERS = {
    "masks": _masks_argv,
    "cortical_tiles": _cortical_tiles_argv,
    "databases": _databases_argv,
}


@pytest.mark.unit
class TestCompareImportWithoutAims:
    """REQ-COMPARE-01: importing compare.py without soma.aims succeeds."""

    def test_import_completes_without_exception(self, tmp_path):
        result = _run_child(f"import compare\nprint({_IMPORT_MARKER!r})\n", tmp_path)

        assert result.returncode == 0, (
            f"importing compare without soma.aims exited {result.returncode}\n"
            f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )
        assert _IMPORT_MARKER in result.stdout


@pytest.mark.unit
class TestCompareSubcommandsWithoutAims:
    """REQ-COMPARE-02 / REQ-COMPARE-03: aims-dependent subcommands fail loudly."""

    @pytest.mark.parametrize("mode", sorted(_MODE_ARGV_BUILDERS))
    def test_subcommand_exits_non_zero(self, tmp_path, mode):
        result = _run_compare(_MODE_ARGV_BUILDERS[mode](tmp_path), tmp_path)

        assert result.returncode != 0, (
            f"`compare.py {mode}` without soma.aims exited 0\nstdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )

    @pytest.mark.parametrize("mode", sorted(_MODE_ARGV_BUILDERS))
    def test_subcommand_reports_soma_aims_on_stderr(self, tmp_path, mode):
        result = _run_compare(_MODE_ARGV_BUILDERS[mode](tmp_path), tmp_path)

        assert "soma.aims" in result.stderr, (
            f"`compare.py {mode}` without soma.aims did not name soma.aims on stderr\n"
            f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )


def _crops_argv(tmp_path: Path) -> list:
    # A minimal, otherwise comparable crop set (numpy only), so the only reason
    # to fail is the missing PyAIMS.
    for name in ("set_a", "set_b"):
        mask_dir = tmp_path / name / "S.C.-sylv." / "mask"
        mask_dir.mkdir(parents=True, exist_ok=True)
        np.save(mask_dir / "Lskeleton.npy", np.zeros((1, 2, 2, 2, 1), dtype=np.int16))
        (mask_dir / "Lskeleton_subject.csv").write_text("Subject\ns1\n")
    return [
        "crops",
        "--set_a",
        str(tmp_path / "set_a"),
        "--set_b",
        str(tmp_path / "set_b"),
        "--output",
        str(tmp_path / "crops_out"),
        "--njobs",
        "1",
    ]


@pytest.mark.unit
class TestCropsWithoutAims:
    """REQ-COMPARE-22: `compare.py crops` fails loudly when soma.aims cannot be imported."""

    def test_crops_exits_non_zero_naming_soma_aims_on_stderr(self, tmp_path):
        result = _run_compare(_crops_argv(tmp_path), tmp_path)

        assert result.returncode != 0, (
            f"`compare.py crops` without soma.aims exited 0\nstdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )
        assert "soma.aims" in result.stderr, (
            "`compare.py crops` without soma.aims did not name soma.aims on stderr\n"
            f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )
