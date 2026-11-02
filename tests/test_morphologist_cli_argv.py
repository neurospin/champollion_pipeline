"""morphologist-cli runs from an argument list, not through a shell (TASK-PYTHONIC-BDRABCZUK-0AD195279EFB).

NTUI: input file names and the output path are user input. Joined with spaces and run by
bash, a name such as ``a b;$(touch x).nii.gz`` is split and executed. With shell=False each
value reaches morphologist-cli as one argv entry, so the pipeline_steps value must no longer
carry the double quotes bash used to strip.

REQ-PYTHONIC-BDRABCZUK-B5A0F717AB0D: execute_command gets shell=False.
REQ-PYTHONIC-BDRABCZUK-4B667191900B: each kept input file name is one literal argument.
REQ-PYTHONIC-BDRABCZUK-9AA14FAEA218: the output path is one literal argument.
REQ-PYTHONIC-BDRABCZUK-6AD5B1EBE6A4: last argument pipeline_steps=..., no quote character.
"""

import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.generate_morphologist_graphs import GenerateMorphologistGraphs
from tests.test_generate_morphologist_graphs import _write_nifti

pytestmark = pytest.mark.unit  # noqa: V107

PIPELINE_STEPS = "pipeline_steps=sulci_labelling=False, sulcal_morphometry=False"
NONOVERLAP = "morphologist-auto-nonoverlap-1.0"
METACHAR_NAME = "sub 01;$(touch pwned).nii.gz"
SENTINEL = "pwned"

# Fake morphologist-cli: writes its arguments, NUL-separated, to $ARGV_OUT.
_FAKE_CLI = '#!/bin/sh\nprintf \'%s\\0\' "$@" > "$ARGV_OUT"\n'


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    """Put a fake morphologist-cli first on PATH; return the file its argv is written to."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cli = bin_dir / "morphologist-cli"
    cli.write_text(_FAKE_CLI)
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    argv_out = tmp_path / "argv.bin"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("ARGV_OUT", str(argv_out))
    return argv_out


def _child_argv(argv_out: Path) -> list[str]:
    """Arguments the fake morphologist-cli received (without argv[0])."""
    assert argv_out.exists(), "morphologist-cli was not run"
    return argv_out.read_bytes().decode().split("\0")[:-1]


def _dirs(tmp_path, output_name="output"):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / output_name
    input_dir.mkdir()
    output_dir.mkdir()
    return input_dir, output_dir


def _run(input_dir, output_dir, *extra):
    script = GenerateMorphologistGraphs()
    script.parse_args([str(input_dir), str(output_dir), *extra])
    return script.run()


def test_morphologist_cli_runs_without_shell(tmp_path):
    """REQ-PYTHONIC-BDRABCZUK-B5A0F717AB0D: execute_command is called with shell=False."""
    input_dir, output_dir = _dirs(tmp_path)
    _write_nifti(input_dir / "subject.nii.gz")
    script = GenerateMorphologistGraphs()
    script.parse_args([str(input_dir), str(output_dir)])

    with patch.object(script, "execute_command", return_value=0) as mock_exec:
        script.run()

    mock_exec.assert_called_once()
    assert mock_exec.call_args.kwargs.get("shell", False) is False


def test_default_argv_is_exact(tmp_path):
    """REQ-PYTHONIC-BDRABCZUK-6AD5B1EBE6A4: the whole list handed to execute_command, quotes-free last entry."""
    input_dir, output_dir = _dirs(tmp_path)
    _write_nifti(input_dir / "subject.nii.gz")
    script = GenerateMorphologistGraphs()
    script.parse_args([str(input_dir), str(output_dir)])

    with patch.object(script, "execute_command", return_value=0) as mock_exec:
        script.run()

    assert mock_exec.call_args.args[0] == [
        "morphologist-cli",
        "subject.nii.gz",
        str(output_dir),
        "--",
        "--if",
        NONOVERLAP,
        "--of",
        NONOVERLAP,
        PIPELINE_STEPS,
    ]


def test_pipeline_steps_reaches_child_without_quotes(tmp_path, fake_cli):
    """REQ-PYTHONIC-BDRABCZUK-6AD5B1EBE6A4: the child's last argument is pipeline_steps=..., no quote."""
    input_dir, output_dir = _dirs(tmp_path)
    _write_nifti(input_dir / "subject.nii.gz")

    assert _run(input_dir, output_dir) == 0

    argv = _child_argv(fake_cli)
    assert argv[-1] == PIPELINE_STEPS
    assert not any('"' in arg or "'" in arg for arg in argv)


def test_metacharacter_file_name_reaches_child_as_one_literal_argument(tmp_path, fake_cli):
    """REQ-PYTHONIC-BDRABCZUK-4B667191900B: spaces, ';' and '$( )' pass through unexpanded, as one argument."""
    input_dir, output_dir = _dirs(tmp_path)
    _write_nifti(input_dir / METACHAR_NAME)

    assert _run(input_dir, output_dir) == 0

    argv = _child_argv(fake_cli)
    assert argv[0] == METACHAR_NAME
    assert argv.count(METACHAR_NAME) == 1
    assert not (input_dir / SENTINEL).exists(), "the $( ) in the file name was executed"
    assert not (tmp_path / SENTINEL).exists(), "the $( ) in the file name was executed"


def test_output_path_with_space_reaches_child_as_one_argument(tmp_path, fake_cli):
    """REQ-PYTHONIC-BDRABCZUK-9AA14FAEA218: an output path with a space is one argument, equal to the path."""
    input_dir, output_dir = _dirs(tmp_path, output_name="my output")
    _write_nifti(input_dir / "subject.nii.gz")

    assert _run(input_dir, output_dir) == 0

    argv = _child_argv(fake_cli)
    assert argv[:3] == ["subject.nii.gz", str(output_dir), "--"]
