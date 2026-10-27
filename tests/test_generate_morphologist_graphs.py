#!/usr/bin/env python3
"""
Unit tests for generate_morphologist_graphs.py
"""

import gzip
import logging
import math
import struct
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import call, patch

import pytest

from champollion_pipeline.generate_morphologist_graphs import GenerateMorphologistGraphs, main
from champollion_pipeline.utils.qc_report import read_qc_tsv

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
_NIFTI1_VOX_OFFSET = 352


def _nifti_bytes(dims=(4, 4, 4), zooms=None, *, sizeof_hdr=348, magic=b"n+1\0", missing_bytes=0):
    """Return the bytes of a single-file little-endian NIfTI-1 image of uint8 zeros.

    Standard library only, so a test can also craft a header the stage must refuse.
    """
    zooms = (1.0,) * len(dims) if zooms is None else tuple(zooms)
    header = bytearray(348)
    struct.pack_into("<i", header, 0, sizeof_hdr)
    struct.pack_into("<8h", header, 40, len(dims), *dims, *([1] * (7 - len(dims))))
    struct.pack_into("<hh", header, 70, 2, 8)  # datatype uint8, bitpix 8
    struct.pack_into("<8f", header, 76, 1.0, *zooms, *([1.0] * (7 - len(zooms))))
    struct.pack_into("<ff", header, 108, float(_NIFTI1_VOX_OFFSET), 1.0)  # vox_offset, scl_slope
    header[344:348] = magic
    n_voxels = math.prod(max(d, 0) for d in dims)
    return bytes(header) + bytes(4) + bytes(max(n_voxels - missing_bytes, 0))


def _write_nifti(path, content=None, **kwargs):
    """Write a NIfTI file (gzip-compressed when the name ends in .gz); return its path."""
    path = Path(path)
    data = _nifti_bytes(**kwargs) if content is None else content
    path.write_bytes(gzip.compress(data) if path.name.endswith(".gz") else data)
    return path


class TestGenerateMorphologistGraphsInit:
    """Test initialization of GenerateMorphologistGraphs."""

    def test_init_creates_script(self):
        """Test that script is initialized correctly."""
        script = GenerateMorphologistGraphs()
        assert script.script_name == "morphologist_graphs_generator"
        assert "morphologist" in script.description.lower()

    def test_init_configures_arguments(self):
        """Test that arguments are configured."""
        script = GenerateMorphologistGraphs()
        # Parse with help to see if arguments are set up
        with pytest.raises(SystemExit):
            script.parse_args([])  # Should fail due to missing required args


class TestGenerateMorphologistGraphsArguments:
    """Test argument parsing."""

    def test_parse_input_output_arguments(self):
        """Test parsing input and output arguments."""
        script = GenerateMorphologistGraphs()
        args = script.parse_args(["/path/to/input", "/path/to/output"])
        assert args.input == "/path/to/input"
        assert args.output == "/path/to/output"

    def test_missing_arguments_raises_error(self):
        """Test that missing arguments raise SystemExit."""
        script = GenerateMorphologistGraphs()
        with pytest.raises(SystemExit):
            script.parse_args([])

    def test_only_one_argument_raises_error(self):
        """Test that providing only one argument raises error."""
        script = GenerateMorphologistGraphs()
        with pytest.raises(SystemExit):
            script.parse_args(["/path/to/input"])

    def test_parallel_flag_default_false(self):
        """Test that --parallel defaults to False."""
        script = GenerateMorphologistGraphs()
        args = script.parse_args(["/input", "/output"])
        assert args.parallel is False

    def test_parallel_flag_can_be_set(self):
        """Test that --parallel can be set to True."""
        script = GenerateMorphologistGraphs()
        args = script.parse_args(["/input", "/output", "--parallel"])
        assert args.parallel is True

    def test_enable_sulcal_recognition_default_false(self):
        """Test that --enable-sulcal-recognition defaults to False."""
        script = GenerateMorphologistGraphs()
        args = script.parse_args(["/input", "/output"])
        assert args.enable_sulcal_recognition is False

    def test_enable_sulcal_recognition_can_be_set(self):
        """Test that --enable-sulcal-recognition can be set to True."""
        script = GenerateMorphologistGraphs()
        args = script.parse_args(["/input", "/output", "--enable-sulcal-recognition"])
        assert args.enable_sulcal_recognition is True

    def test_bids_default_false(self):
        """REQ-BIDS-03: --bids exists on the parser and defaults to False."""
        script = GenerateMorphologistGraphs()
        args = script.parse_args(["/input", "/output"])
        assert args.bids is False

    def test_bids_can_be_set(self):
        """REQ-BIDS-03: --bids is recognized and sets args.bids to True."""
        script = GenerateMorphologistGraphs()
        args = script.parse_args(["/input", "/output", "--bids"])
        assert args.bids is True


class TestGetInputFiles:
    """Test _get_input_files method."""

    def test_get_input_files_with_nii_gz(self, temp_dir):
        """Test getting input files with .nii.gz extension."""
        script = GenerateMorphologistGraphs()
        script.parse_args([temp_dir, temp_dir])

        # Create test files
        (Path(temp_dir) / "test1.nii.gz").touch()
        (Path(temp_dir) / "test2.nii.gz").touch()
        (Path(temp_dir) / "ignore.txt").touch()

        files = script._get_input_files()
        assert len(files) == 2
        assert all(f.endswith((".nii.gz", ".nii", ".gz")) for f in files)

    def test_get_input_files_with_nii(self, temp_dir):
        """Test getting input files with .nii extension."""
        script = GenerateMorphologistGraphs()
        script.parse_args([temp_dir, temp_dir])

        # Create test files
        (Path(temp_dir) / "test1.nii").touch()
        (Path(temp_dir) / "test2.nii").touch()

        files = script._get_input_files()
        assert len(files) == 2

    def test_get_input_files_mixed_extensions(self, temp_dir):
        """Test getting input files with mixed valid extensions."""
        script = GenerateMorphologistGraphs()
        script.parse_args([temp_dir, temp_dir])

        # Create test files
        (Path(temp_dir) / "test1.nii.gz").touch()
        (Path(temp_dir) / "test2.nii").touch()
        (Path(temp_dir) / "test3.gz").touch()
        (Path(temp_dir) / "ignore.txt").touch()
        (Path(temp_dir) / "ignore.json").touch()

        files = script._get_input_files()
        assert len(files) == 3
        assert "ignore.txt" not in files
        assert "ignore.json" not in files

    def test_get_input_files_no_valid_files(self, temp_dir):
        """Test getting input files when no valid files exist."""
        script = GenerateMorphologistGraphs()
        script.parse_args([temp_dir, temp_dir])

        # Create only invalid files
        (Path(temp_dir) / "test.txt").touch()
        (Path(temp_dir) / "test.json").touch()

        files = script._get_input_files()
        assert len(files) == 0

    def test_get_input_files_empty_directory(self, temp_dir):
        """Test getting input files from empty directory."""
        script = GenerateMorphologistGraphs()
        script.parse_args([temp_dir, temp_dir])

        files = script._get_input_files()
        assert len(files) == 0


class TestValidatePaths:
    """Test path validation in run method."""

    def test_validate_paths_success(self, temp_dir):
        """Test successful path validation."""
        script = GenerateMorphologistGraphs()
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        script.parse_args([str(input_dir), str(output_dir)])

        assert script.validate_paths([str(input_dir), str(output_dir)]) is True

    def test_validate_paths_failure_raises_error(self, temp_dir):
        """Test that invalid paths raise ValueError."""
        script = GenerateMorphologistGraphs()
        script.parse_args(["/nonexistent/input", "/nonexistent/output"])

        with (
            patch.object(script, "validate_paths", return_value=False),
            pytest.raises(ValueError, match="Please input valid paths"),
        ):
            script.run()


class TestRunMethod:
    """Test the run method."""

    @patch("champollion_pipeline.generate_morphologist_graphs.chdir")
    @patch("champollion_pipeline.generate_morphologist_graphs.getcwd", return_value="/original/dir")
    def test_run_changes_directory(self, _mock_getcwd, mock_chdir, temp_dir):
        """Test that run changes to input directory."""
        script = GenerateMorphologistGraphs()
        script.parse_args([temp_dir, temp_dir])
        _write_nifti(Path(temp_dir) / "test.nii.gz")

        with (
            patch.object(script, "validate_paths", return_value=True),
            patch.object(script, "_get_input_files", return_value=["test.nii.gz"]),
            patch.object(script, "execute_command", return_value=0),
        ):
            script.run()

            # Check that chdir was called to input and back
            assert call(temp_dir) in mock_chdir.call_args_list
            assert call("/original/dir") in mock_chdir.call_args_list

    def test_run_builds_correct_command(self, temp_dir):
        """Test that run builds correct morphologist-cli command."""
        script = GenerateMorphologistGraphs()
        input_dir = temp_dir
        output_dir = str(Path(temp_dir) / "output")
        Path(output_dir).mkdir()

        script.parse_args([input_dir, output_dir])

        test_files = ["subject1.nii.gz", "subject2.nii.gz"]
        for name in test_files:
            _write_nifti(Path(input_dir) / name)

        with (
            patch.object(script, "validate_paths", return_value=True),
            patch.object(script, "_get_input_files", return_value=test_files),
            patch.object(script, "execute_command", return_value=0) as mock_exec,
            patch("champollion_pipeline.generate_morphologist_graphs.chdir"),
            patch("champollion_pipeline.generate_morphologist_graphs.getcwd", return_value="/original"),
        ):
            script.run()

            # Check the command
            cmd = mock_exec.call_args[0][0]
            assert "morphologist-cli" in cmd
            assert "subject1.nii.gz" in cmd
            assert "subject2.nii.gz" in cmd
            assert output_dir in cmd
            assert "--" in cmd
            assert "--of" in cmd
            assert "morphologist-auto-nonoverlap-1.0" in cmd

    def test_run_with_shell_true(self, temp_dir):
        """Test that run executes command with shell=True."""
        script = GenerateMorphologistGraphs()
        script.parse_args([temp_dir, temp_dir])
        _write_nifti(Path(temp_dir) / "test.nii")

        with (
            patch.object(script, "validate_paths", return_value=True),
            patch.object(script, "_get_input_files", return_value=["test.nii"]),
            patch.object(script, "execute_command", return_value=0) as mock_exec,
            patch("champollion_pipeline.generate_morphologist_graphs.chdir"),
            patch("champollion_pipeline.generate_morphologist_graphs.getcwd", return_value="/original"),
        ):
            script.run()

            # Check that shell=True was passed
            assert mock_exec.call_args[1]["shell"] is True

    def test_run_returns_command_result(self, temp_dir):
        """Test that run returns the result from execute_command."""
        script = GenerateMorphologistGraphs()
        script.parse_args([temp_dir, temp_dir])
        _write_nifti(Path(temp_dir) / "test.nii")

        with (
            patch.object(script, "validate_paths", return_value=True),
            patch.object(script, "_get_input_files", return_value=["test.nii"]),
            patch.object(script, "execute_command", return_value=42),
            patch("champollion_pipeline.generate_morphologist_graphs.chdir"),
            patch("champollion_pipeline.generate_morphologist_graphs.getcwd", return_value="/original"),
        ):
            result = script.run()
            assert result == 42

    def test_run_restores_directory_on_success(self, temp_dir):
        """Test that original directory is restored after successful run."""
        script = GenerateMorphologistGraphs()
        script.parse_args([temp_dir, temp_dir])
        _write_nifti(Path(temp_dir) / "test.nii")
        original_dir = "/original/dir"

        with (
            patch("champollion_pipeline.generate_morphologist_graphs.getcwd", return_value=original_dir),
            patch("champollion_pipeline.generate_morphologist_graphs.chdir") as mock_chdir,
            patch.object(script, "validate_paths", return_value=True),
            patch.object(script, "_get_input_files", return_value=["test.nii"]),
            patch.object(script, "execute_command", return_value=0),
        ):
            script.run()

            # Verify we returned to original directory
            assert mock_chdir.call_args_list[-1] == call(original_dir)


class TestMainFunction:
    """Test the main entry point."""

    def test_main_inits_then_returns_script_main(self):
        """main() calls init_pipeline_process(), then returns the script's ScriptBuilder.main() result."""
        calls = []
        with patch(
            "champollion_pipeline.generate_morphologist_graphs.init_pipeline_process",
            side_effect=lambda: calls.append("init"),
        ):
            script_main = {"return_value.main.side_effect": lambda: calls.append("main") or 3}
            with patch(
                "champollion_pipeline.generate_morphologist_graphs.GenerateMorphologistGraphs", **script_main
            ) as MockScript:
                result = main()

        MockScript.assert_called_once_with()
        MockScript.return_value.main.assert_called_once_with()
        assert calls == ["init", "main"]
        assert result == 3


class TestCommandFlags:
    """Test that --parallel and --enable-sulcal-recognition affect the command."""

    def test_parallel_adds_swf_to_command(self, temp_dir):
        """Test that --parallel adds --swf to the command."""
        script = GenerateMorphologistGraphs()
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        _write_nifti(input_dir / "subject.nii.gz")

        script.parse_args([str(input_dir), str(output_dir), "--parallel"])

        with patch.object(script, "execute_command", return_value=0) as mock_exec:
            script.run()
            cmd = mock_exec.call_args[0][0]
            assert "--swf" in cmd

    def test_no_parallel_does_not_add_swf(self, temp_dir):
        """Test that without --parallel, --swf is not in the command."""
        script = GenerateMorphologistGraphs()
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        _write_nifti(input_dir / "subject.nii.gz")

        script.parse_args([str(input_dir), str(output_dir)])

        with patch.object(script, "execute_command", return_value=0) as mock_exec:
            script.run()
            cmd = mock_exec.call_args[0][0]
            assert "--swf" not in cmd

    def test_default_skips_sulcal_recognition(self, temp_dir):
        """Test that without --enable-sulcal-recognition, sulci_labelling=False is in cmd."""
        script = GenerateMorphologistGraphs()
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        _write_nifti(input_dir / "subject.nii.gz")

        script.parse_args([str(input_dir), str(output_dir)])

        with patch.object(script, "execute_command", return_value=0) as mock_exec:
            script.run()
            cmd = mock_exec.call_args[0][0]
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            assert "sulci_labelling=False" in cmd_str

    def test_enable_sulcal_recognition_removes_disable_step(self, temp_dir):
        """Test that --enable-sulcal-recognition removes the pipeline_steps disable."""
        script = GenerateMorphologistGraphs()
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        _write_nifti(input_dir / "subject.nii.gz")

        script.parse_args([str(input_dir), str(output_dir), "--enable-sulcal-recognition"])

        with patch.object(script, "execute_command", return_value=0) as mock_exec:
            script.run()
            cmd = mock_exec.call_args[0][0]
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            assert "sulci_labelling=False" not in cmd_str


class TestBidsFormat:
    """REQ-BIDS-03: --bids selects the morphologist-cli --if/--of formats."""

    @staticmethod
    def _run_and_get_cmd(temp_dir, extra_args):
        script = GenerateMorphologistGraphs()
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        # BIDS-named so that the file has a sub-<label> entity with and without --bids
        _write_nifti(input_dir / "sub-01_T1w.nii.gz")

        script.parse_args([str(input_dir), str(output_dir), *extra_args])

        with patch.object(script, "execute_command", return_value=0) as mock_exec:
            script.run()
        return mock_exec.call_args[0][0]

    def test_bids_flag_sets_bids_input_and_output_formats(self, temp_dir):
        """With --bids, both --if and --of are morphologist-bids-2.0."""
        cmd = self._run_and_get_cmd(temp_dir, ["--bids"])
        assert cmd[cmd.index("--if") + 1] == "morphologist-bids-2.0"
        assert cmd[cmd.index("--of") + 1] == "morphologist-bids-2.0"
        assert "morphologist-auto-nonoverlap-1.0" not in cmd

    def test_no_bids_flag_keeps_nonoverlap_formats(self, temp_dir):
        """Regression guard: without --bids, both --if and --of stay morphologist-auto-nonoverlap-1.0."""
        cmd = self._run_and_get_cmd(temp_dir, [])
        assert cmd[cmd.index("--if") + 1] == "morphologist-auto-nonoverlap-1.0"
        assert cmd[cmd.index("--of") + 1] == "morphologist-auto-nonoverlap-1.0"
        assert "morphologist-bids-2.0" not in cmd


@pytest.mark.integration
class TestGenerateMorphologistGraphsIntegration:
    """Integration tests for GenerateMorphologistGraphs."""

    def test_full_workflow_with_valid_files(self, temp_dir):
        """Test complete workflow with valid input files."""
        script = GenerateMorphologistGraphs()
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Create test files
        _write_nifti(input_dir / "subject1.nii.gz")
        _write_nifti(input_dir / "subject2.nii.gz")

        script.parse_args([str(input_dir), str(output_dir)])

        with patch.object(script, "execute_command", return_value=0) as mock_exec:
            result = script.run()

            assert result == 0
            mock_exec.assert_called_once()
            # Verify command structure
            cmd = mock_exec.call_args[0][0]
            assert len(cmd) > 0
            assert "morphologist-cli" in cmd


@pytest.mark.smoke
class TestGenerateMorphologistGraphsSmoke:
    """Smoke tests for basic functionality."""

    def test_script_can_be_instantiated(self):
        """Test that script can be created."""
        script = GenerateMorphologistGraphs()
        assert script is not None

    def test_script_has_required_methods(self):
        """Test that script has all required methods."""
        script = GenerateMorphologistGraphs()
        assert hasattr(script, "run")
        assert hasattr(script, "_get_input_files")
        assert callable(script.run)
        assert callable(script._get_input_files)


# ---------------------------------------------------------------------------
# NTUI stage 1 input QC (TASK-NTUI-BDRABCZUK-18A7A8483C9F)
# ---------------------------------------------------------------------------


def _input_qc_path(output_dir):
    """<output>/derivatives/morphologist-6.0/input_qc.tsv."""
    return Path(output_dir) / "derivatives" / "morphologist-6.0" / "input_qc.tsv"


def _qc_by_id(output_dir):
    """Read the stage 1 input QC file back as {participant_id: (qc, reason)}."""
    return {r["participant_id"]: (r["qc"], r["reason"]) for r in read_qc_tsv(_input_qc_path(output_dir))}


@pytest.fixture
def qc_dirs(tmp_path):
    """Return (input_dir, output_dir), both existing."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    return input_dir, output_dir


def _run_stage(input_dir, output_dir, status=0, extra_args=()):
    """Run the stage with morphologist-cli mocked; return (result, execute_command mock)."""
    script = GenerateMorphologistGraphs()
    script.parse_args([str(input_dir), str(output_dir), *extra_args])
    with patch.object(script, "execute_command", return_value=status) as mock_exec:
        result = script.run()
    return result, mock_exec


class TestInputQcFile:
    """The input QC file, one row per participant_id."""

    def test_one_row_per_participant_id(self, qc_dirs):
        """REQ-NTUI-BDRABCZUK-A474A941FE23, REQ-NTUI-BDRABCZUK-2A3110BE30EA: without --bids, name minus extension."""
        input_dir, output_dir = qc_dirs
        _write_nifti(input_dir / "sub-a.nii.gz")
        _write_nifti(input_dir / "sub-b.nii")
        _write_nifti(input_dir / "sub-c.gz")
        (input_dir / "notes.txt").write_text("not an image")
        qc_path = _input_qc_path(output_dir)
        seen_at_call = []

        script = GenerateMorphologistGraphs()
        script.parse_args([str(input_dir), str(output_dir)])
        with patch.object(
            script, "execute_command", side_effect=lambda *a, **k: seen_at_call.append(qc_path.exists()) or 0
        ):
            script.run()

        assert seen_at_call == [True], "input_qc.tsv must exist before morphologist-cli runs"
        header = qc_path.read_text(encoding="utf-8").splitlines()[0].split("\t")
        assert header[:2] == ["participant_id", "qc"] and header[-1] == "reason"
        assert _qc_by_id(output_dir) == {"sub-a": (1, ""), "sub-b": (1, ""), "sub-c": (1, "")}

    def test_bids_participant_id_is_subject_entity(self, qc_dirs):
        """REQ-NTUI-BDRABCZUK-5F5D65A6F044: with --bids, sub-<label>, plus _ses-<session> when that entity follows.

        Two sessions of one subject are two participant_ids, both kept.
        """
        input_dir, output_dir = qc_dirs
        _write_nifti(input_dir / "sub-01_ses-1_T1w.nii.gz")
        _write_nifti(input_dir / "sub-01_ses-2_T1w.nii.gz")
        _write_nifti(input_dir / "sub-ABC12_acq-mprage_T1w.nii")
        _write_nifti(input_dir / "sub-7.nii.gz")
        _write_nifti(input_dir / "sub-8_ses-pre.nii")
        _run_stage(input_dir, output_dir, extra_args=["--bids"])
        assert _qc_by_id(output_dir) == {
            "sub-01_ses-1": (1, ""),
            "sub-01_ses-2": (1, ""),
            "sub-ABC12": (1, ""),
            "sub-7": (1, ""),
            "sub-8_ses-pre": (1, ""),
        }


class TestInputQcReasons:
    """One reason per failed check; a sibling good file keeps qc 1."""

    @pytest.mark.parametrize(
        "name, content",
        [
            ("bad.nii.gz", b"this is not a NIfTI image"),
            ("bad.nii", b"plain text renamed .nii\n"),
            ("bad.nii", _nifti_bytes(magic=b"ni1\0")),  # two-file header, no data in the .nii
            ("bad.nii.gz", _nifti_bytes(sizeof_hdr=540)),  # NIfTI-2 size with a NIfTI-1 magic
            ("bad.nii.gz", b""),
        ],
        ids=["gz_garbage", "text", "magic_ni1", "size_magic_mismatch", "empty"],
    )
    def test_unreadable_header(self, qc_dirs, name, content):
        """REQ-NTUI-BDRABCZUK-5C5C231009D7."""
        input_dir, output_dir = qc_dirs
        _write_nifti(input_dir / "good.nii.gz")
        (input_dir / name).write_bytes(gzip.compress(content) if name.endswith(".gz") else content)
        _run_stage(input_dir, output_dir)
        assert _qc_by_id(output_dir) == {"good": (1, ""), "bad": (0, "unreadable_header")}

    def test_not_3d(self, qc_dirs):
        """REQ-NTUI-BDRABCZUK-390968084904: dim[0] below 3, or more than one volume."""
        input_dir, output_dir = qc_dirs
        _write_nifti(input_dir / "good.nii.gz")
        _write_nifti(input_dir / "single_volume_4d.nii.gz", dims=(4, 4, 4, 1))
        _write_nifti(input_dir / "flat.nii.gz", dims=(4, 4))
        _write_nifti(input_dir / "two_volumes.nii.gz", dims=(4, 4, 4, 2))
        _write_nifti(input_dir / "five_d.nii", dims=(4, 4, 4, 1, 3))
        _run_stage(input_dir, output_dir)
        assert _qc_by_id(output_dir) == {
            "good": (1, ""),
            "single_volume_4d": (1, ""),
            "flat": (0, "not_3d"),
            "two_volumes": (0, "not_3d"),
            "five_d": (0, "not_3d"),
        }

    def test_invalid_geometry(self, qc_dirs):
        """REQ-NTUI-BDRABCZUK-CB9D22C84BD3: a spatial dim below 1, or a spatial voxel size not finite above 0."""
        input_dir, output_dir = qc_dirs
        _write_nifti(input_dir / "good.nii.gz", zooms=(0.7, 1.0, 1.3))
        _write_nifti(input_dir / "zero_dim.nii.gz", dims=(4, 0, 4))
        _write_nifti(input_dir / "zero_voxel.nii.gz", zooms=(1.0, 0.0, 1.0))
        _write_nifti(input_dir / "negative_voxel.nii", zooms=(1.0, 1.0, -1.0))
        _write_nifti(input_dir / "nan_voxel.nii.gz", zooms=(math.nan, 1.0, 1.0))
        _write_nifti(input_dir / "inf_voxel.nii", zooms=(1.0, math.inf, 1.0))
        _run_stage(input_dir, output_dir)
        assert _qc_by_id(output_dir) == {
            "good": (1, ""),
            "zero_dim": (0, "invalid_geometry"),
            "zero_voxel": (0, "invalid_geometry"),
            "negative_voxel": (0, "invalid_geometry"),
            "nan_voxel": (0, "invalid_geometry"),
            "inf_voxel": (0, "invalid_geometry"),
        }

    def test_truncated(self, qc_dirs):
        """REQ-NTUI-BDRABCZUK-BED9B1143D07: fewer bytes than vox_offset + n_voxels * bitpix / 8."""
        input_dir, output_dir = qc_dirs
        _write_nifti(input_dir / "good.nii.gz")
        _write_nifti(input_dir / "good_plain.nii")
        _write_nifti(input_dir / "short_gz.nii.gz", missing_bytes=1)
        _write_nifti(input_dir / "short_plain.nii", missing_bytes=1)
        _write_nifti(input_dir / "header_only.nii.gz", missing_bytes=64)
        _run_stage(input_dir, output_dir)
        assert _qc_by_id(output_dir) == {
            "good": (1, ""),
            "good_plain": (1, ""),
            "short_gz": (0, "truncated"),
            "short_plain": (0, "truncated"),
            "header_only": (0, "truncated"),
        }

    def test_duplicate_subject(self, qc_dirs):
        """REQ-NTUI-BDRABCZUK-6B48AF1E3A51: two files with one participant_id."""
        input_dir, output_dir = qc_dirs
        _write_nifti(input_dir / "good.nii.gz")
        _write_nifti(input_dir / "twice.nii.gz")
        _write_nifti(input_dir / "twice.nii")
        _run_stage(input_dir, output_dir)
        assert _qc_by_id(output_dir) == {"good": (1, ""), "twice": (0, "duplicate_subject")}

    def test_bids_duplicate_subject(self, qc_dirs):
        """REQ-NTUI-BDRABCZUK-6B48AF1E3A51 with REQ-NTUI-BDRABCZUK-5F5D65A6F044: two runs of one sub+ses."""
        input_dir, output_dir = qc_dirs
        _write_nifti(input_dir / "sub-01_ses-1_T1w.nii.gz")
        _write_nifti(input_dir / "sub-02_ses-1_run-1_T1w.nii.gz")
        _write_nifti(input_dir / "sub-02_ses-1_run-2_T1w.nii.gz")
        _run_stage(input_dir, output_dir, extra_args=["--bids"])
        assert _qc_by_id(output_dir) == {"sub-01_ses-1": (1, ""), "sub-02_ses-1": (0, "duplicate_subject")}

    def test_no_bids_subject(self, qc_dirs):
        """REQ-NTUI-BDRABCZUK-68925CFAE769: with --bids, a name not starting with sub-<label> then _ or ."""
        input_dir, output_dir = qc_dirs
        _write_nifti(input_dir / "sub-01_T1w.nii.gz")
        _write_nifti(input_dir / "T1w.nii.gz")
        _write_nifti(input_dir / "subject01.nii")
        _write_nifti(input_dir / "sub-_T1w.nii.gz")
        _write_nifti(input_dir / "anat_sub-02_T1w.nii.gz")
        _run_stage(input_dir, output_dir, extra_args=["--bids"])
        assert _qc_by_id(output_dir) == {
            "sub-01": (1, ""),
            "T1w": (0, "no_bids_subject"),
            "subject01": (0, "no_bids_subject"),
            "sub-_T1w": (0, "no_bids_subject"),
            "anat_sub-02_T1w": (0, "no_bids_subject"),
        }


class TestInputQcGating:
    """What the stage does with the QC verdicts (user decision 2026-10-27: QC never fails the stage)."""

    def test_only_qc1_files_in_command(self, qc_dirs):
        """REQ-NTUI-BDRABCZUK-31C5B86C78D0."""
        input_dir, output_dir = qc_dirs
        _write_nifti(input_dir / "good1.nii.gz")
        _write_nifti(input_dir / "good2.nii")
        (input_dir / "bad.nii.gz").write_bytes(gzip.compress(b"garbage"))
        _write_nifti(input_dir / "twice.nii.gz")
        _write_nifti(input_dir / "twice.nii")
        _, mock_exec = _run_stage(input_dir, output_dir)
        cmd = mock_exec.call_args[0][0]
        image_args = [a for a in cmd if a.endswith((".nii.gz", ".nii", ".gz"))]
        assert sorted(image_args) == ["good1.nii.gz", "good2.nii"]
        assert _qc_by_id(output_dir)["bad"][0] == 0

    def test_no_qc1_returns_0_without_morphologist(self, qc_dirs):
        """REQ-NTUI-BDRABCZUK-E1FE4637F72D."""
        input_dir, output_dir = qc_dirs
        (input_dir / "bad.nii.gz").write_bytes(gzip.compress(b"garbage"))
        _write_nifti(input_dir / "flat.nii", dims=(4, 4))
        result, mock_exec = _run_stage(input_dir, output_dir, status=9)
        assert _qc_by_id(output_dir) == {"bad": (0, "unreadable_header"), "flat": (0, "not_3d")}
        mock_exec.assert_not_called()
        assert result == 0

    def test_bids_command_passes_qc1_file_names(self, qc_dirs):
        """REQ-NTUI-BDRABCZUK-31C5B86C78D0 under --bids: qc 1 file names only; a no_bids_subject file is left out."""
        input_dir, output_dir = qc_dirs
        _write_nifti(input_dir / "sub-01_ses-1_T1w.nii.gz")
        _write_nifti(input_dir / "sub-01_ses-2_T1w.nii.gz")
        _write_nifti(input_dir / "sub-02_T1w.nii")
        (input_dir / "sub-03_T1w.nii.gz").write_bytes(gzip.compress(b"garbage"))
        _write_nifti(input_dir / "T1w.nii.gz")
        _, mock_exec = _run_stage(input_dir, output_dir, extra_args=["--bids"])
        cmd = mock_exec.call_args[0][0]
        image_args = [a for a in cmd if a.endswith((".nii.gz", ".nii", ".gz"))]
        assert sorted(image_args) == ["sub-01_ses-1_T1w.nii.gz", "sub-01_ses-2_T1w.nii.gz", "sub-02_T1w.nii"]

    def test_one_warning_record_names_excluded(self, qc_dirs, caplog):
        """REQ-NTUI-BDRABCZUK-A4354BA1930C: one WARNING record through the module logger."""
        input_dir, output_dir = qc_dirs
        _write_nifti(input_dir / "good.nii.gz")
        (input_dir / "bad.nii.gz").write_bytes(gzip.compress(b"garbage"))
        _write_nifti(input_dir / "flat.nii", dims=(4, 4))
        logger_name = "champollion_pipeline.generate_morphologist_graphs"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            _run_stage(input_dir, output_dir)
        records = [r for r in caplog.records if r.name == logger_name and r.levelno == logging.WARNING]
        assert len(records) == 1, [r.getMessage() for r in caplog.records]
        prefix = f"input QC excluded 2 subject(s), see {_input_qc_path(output_dir)}: "
        message = records[0].getMessage()
        assert message.startswith(prefix), message
        assert sorted(message[len(prefix) :].split(", ")) == ["bad", "flat"]

    def test_no_warning_record_when_all_pass(self, qc_dirs, caplog):
        """REQ-NTUI-BDRABCZUK-A4354BA1930C: no qc 0 row, no WARNING record."""
        input_dir, output_dir = qc_dirs
        _write_nifti(input_dir / "good.nii.gz")
        logger_name = "champollion_pipeline.generate_morphologist_graphs"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            _run_stage(input_dir, output_dir)
        assert [r for r in caplog.records if r.name == logger_name and r.levelno >= logging.WARNING] == []

    @pytest.mark.parametrize("status", [0, 7])
    def test_partial_exclusion_returns_morphologist_status(self, qc_dirs, status):
        """REQ-NTUI-BDRABCZUK-2450607E5A2F."""
        input_dir, output_dir = qc_dirs
        _write_nifti(input_dir / "good.nii.gz")
        (input_dir / "bad.nii.gz").write_bytes(gzip.compress(b"garbage"))
        result, mock_exec = _run_stage(input_dir, output_dir, status=status)
        assert sorted(qc for qc, _ in _qc_by_id(output_dir).values()) == [0, 1]
        mock_exec.assert_called_once()
        assert result == status


class TestInputQcWithoutSoma:
    """REQ-NTUI-BDRABCZUK-4B840E22360B: the input QC needs no BrainVISA Python."""

    def test_qc_file_written_when_soma_unavailable(self, qc_dirs, tmp_path):
        """conftest stubs soma in-process, so this runs in a child where soma is blocked."""
        input_dir, output_dir = qc_dirs
        _write_nifti(input_dir / "good.nii.gz")
        _write_nifti(input_dir / "flat.nii", dims=(4, 4))
        code = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(_SRC_DIR)!r})
            sys.modules["soma"] = None
            sys.modules["soma.aims"] = None
            import champollion_utils.script_builder as _sb
            _sb.check_for_updates = lambda *a, **k: None
            from champollion_pipeline.generate_morphologist_graphs import GenerateMorphologistGraphs
            script = GenerateMorphologistGraphs()
            script.parse_args([{str(input_dir)!r}, {str(output_dir)!r}])
            script.execute_command = lambda *a, **k: 0
            sys.exit(script.run())
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], cwd=str(tmp_path), capture_output=True, text=True, timeout=120
        )
        assert proc.returncode == 0, proc.stderr
        assert _qc_by_id(output_dir) == {"good": (1, ""), "flat": (0, "not_3d")}
