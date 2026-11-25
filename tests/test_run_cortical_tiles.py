#!/usr/bin/env python3
"""
Unit tests for run_cortical_tiles.py: init, command, configuration, run(), main() and mask npys.

The whole-brain step tests are in test_run_cortical_tiles_whole_brain.py; their shared helpers in whole_brain_step.py.
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from champollion_pipeline.run_cortical_tiles import RunCorticalTiles, main
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from tests.log_assertions import LOG
from tests.stage2_layout import (  # noqa: F401
    _stub_whole_brain_functions,  # noqa: V103 - autouse fixture
    add_graphs,
    run_on_config,
    stage_dirs,
)
from tests.whole_brain_step import (
    _GRAPH_SUBPATH,
    _LOGGER_NAME,
)


class TestRunCorticalTilesInit:
    """Test initialization of RunCorticalTiles."""

    def test_init_creates_script(self):
        """Test that script is initialized correctly."""
        script = RunCorticalTiles()
        assert script.script_name == "run_cortical_tiles"
        assert "cortical_tiles" in script.description.lower() or "sulcal" in script.description.lower()

    def test_init_configures_arguments(self):
        """Test that arguments are configured."""
        script = RunCorticalTiles()
        with pytest.raises(SystemExit):
            script.parse_args([])  # Should fail due to missing required args


class TestBuildCommand:
    """Test command building."""

    def test_command_includes_required_parameters(self, temp_dir):
        """Test that command includes all required parameters."""
        add_graphs(temp_dir, graph_dir="custom/graph/path")
        script = RunCorticalTiles()
        script.parse_args(
            [temp_dir, temp_dir, "--path_to_graph", "custom/graph/path", "--path_sk_with_hull", "custom/skeleton/path"]
        )

        with (
            patch.object(script, "validate_paths", return_value=True),
            patch.object(script, "execute_command", return_value=0) as mock_exec,
        ):
            script.run()

            # Check execute_command was called with proper command
            mock_exec.assert_called()
            cmd = mock_exec.call_args[0][0]

            # Verify command structure
            assert "-d" in cmd
            assert "--path_to_graph" in cmd
            assert "--path_sk_with_hull" in cmd
            assert "--njobs" in cmd

    def test_command_includes_input_types_when_set(self, temp_dir):
        """Test that --input-types adds -y flag to command."""
        add_graphs(temp_dir)
        script = RunCorticalTiles()
        script.parse_args(
            [
                temp_dir,
                temp_dir,
                "--path_to_graph",
                "graphs",
                "--path_sk_with_hull",
                "skeleton",
                "--input-types",
                "skeleton",
                "foldlabel",
            ]
        )

        with (
            patch.object(script, "validate_paths", return_value=True),
            patch.object(script, "execute_command", return_value=0) as mock_exec,
        ):
            script.run()

            cmd = mock_exec.call_args[0][0]
            assert "-y" in cmd
            assert "skeleton" in cmd
            assert "foldlabel" in cmd

    def test_command_includes_sk_qc_path_when_set(self, temp_dir):
        """Test that --sk_qc_path is included when set."""
        add_graphs(temp_dir)
        qc_file = Path(temp_dir) / "qc.tsv"
        qc_file.write_text("participant_id\tqc\nsubj01\t1\n")
        script = RunCorticalTiles()
        script.parse_args(
            [
                temp_dir,
                temp_dir,
                "--path_to_graph",
                "graphs",
                "--path_sk_with_hull",
                "skeleton",
                "--sk_qc_path",
                str(qc_file),
            ]
        )

        with (
            patch.object(script, "validate_paths", return_value=True),
            patch.object(script, "execute_command", return_value=0) as mock_exec,
        ):
            script.run()

            cmd = mock_exec.call_args[0][0]
            assert "--sk_qc_path" in cmd
            assert str(qc_file) in cmd


class TestSkipDistbottom:
    """Test the rewrite of pipeline_loop_2mm.json (distbottom flags: test_run_cortical_tiles_no_distbottom.py)."""

    def test_graphs_dir_set_to_input(self, temp_dir):
        """Test that graphs_dir in config is set to input path."""
        input_dir, output_dir = stage_dirs(temp_dir)

        updated = run_on_config(input_dir, output_dir, {"graphs_dir": "$local", "other_key": "value"})
        assert updated["graphs_dir"] == str(input_dir.resolve())

    def test_output_dir_set_from_output_arg(self, temp_dir):
        """Test that output_dir in config is set to output/DERIVATIVES_FOLDER."""

        add_graphs(temp_dir)
        output_dir = Path(temp_dir) / "derivatives"
        output_dir.mkdir()

        # Config is placed inside the output (derivatives) directory, not the subjects dir
        updated = run_on_config(temp_dir, output_dir, {"graphs_dir": "$local", "output_dir": "$local"})
        expected = str(output_dir.resolve() / DERIVATIVES_FOLDER)
        assert updated["output_dir"] == expected

    def test_config_with_null_values_overwritten_correctly(self, temp_dir):
        """JSON null (None) for graphs_dir/output_dir is overwritten with correct paths."""

        input_dir, output_dir = stage_dirs(temp_dir)

        updated = run_on_config(input_dir, output_dir, {"graphs_dir": None, "output_dir": None})
        assert updated["graphs_dir"] == str(input_dir.resolve())
        assert updated["output_dir"] == str(output_dir.resolve() / DERIVATIVES_FOLDER)


class TestBidsConfig:
    """REQ-BIDS-01: --bids sets the top-level "bids" key in pipeline_loop_2mm.json.

    generate_one_sulcal_region.py reads params['bids'] to drive BIDS-aware
    skeleton/crop generation; run() must write it when --bids is given and
    leave the existing value untouched otherwise.
    """

    def _run(self, temp_dir, config_data, extra_args=None):
        input_dir, output_dir = stage_dirs(temp_dir)
        return run_on_config(input_dir, output_dir, config_data, *(extra_args or []))

    def test_bids_flag_sets_bids_true_in_config(self, temp_dir):
        """With --bids, the written pipeline_loop_2mm.json has "bids": true, overriding the template's false."""
        updated_config = self._run(temp_dir, {"graphs_dir": "", "bids": False}, extra_args=["--bids"])
        assert updated_config.get("bids") is True

    def test_no_bids_flag_leaves_bids_unchanged_in_config(self, temp_dir):
        """Without --bids, the config's existing "bids" value is left as-is (regression guard)."""
        updated_config = self._run(temp_dir, {"graphs_dir": "", "bids": False})
        assert updated_config.get("bids") is False


class TestGraphPathConfigOverride:
    """Test that path_to_graph and path_sk_with_hull are written into the JSON config."""

    def _run_with_config(self, temp_dir, config_data, graph_path, skel_path):

        input_dir, output_dir = stage_dirs(temp_dir, graph_path)
        config_path = output_dir / "pipeline_loop_2mm.json"
        config_path.write_text(json.dumps(config_data))
        script = RunCorticalTiles()
        script.parse_args(
            [
                str(input_dir),
                str(output_dir),
                "--path_to_graph",
                graph_path,
                "--path_sk_with_hull",
                skel_path,
            ]
        )
        with (
            patch.object(script, "validate_paths", return_value=True),
            patch.object(script, "execute_command", return_value=0),
        ):
            script.run()
        return json.loads(config_path.read_text())

    def test_path_to_graph_written_to_config(self, temp_dir):
        """path_to_graph CLI arg is written into pipeline_loop_2mm.json."""
        updated = self._run_with_config(
            temp_dir,
            {"graphs_dir": "", "output_dir": ""},
            "t1mri/M12/default_analysis/folds/3.1",
            "t1mri/M12/default_analysis/segmentation",
        )
        assert updated["path_to_graph"] == "t1mri/M12/default_analysis/folds/3.1"

    def test_path_sk_with_hull_written_to_config(self, temp_dir):
        """path_sk_with_hull CLI arg is written into pipeline_loop_2mm.json."""
        updated = self._run_with_config(
            temp_dir,
            {"graphs_dir": "", "output_dir": ""},
            "t1mri/M12/default_analysis/folds/3.1",
            "t1mri/M12/default_analysis/segmentation",
        )
        assert updated["path_to_skeleton_with_hull"] == "t1mri/M12/default_analysis/segmentation"

    def test_path_to_graph_overrides_hardcoded_default(self, temp_dir):
        """CLI arg overrides the old hardcoded default value in the JSON."""
        updated = self._run_with_config(
            temp_dir,
            {"graphs_dir": "", "output_dir": "", "path_to_graph": "t1mri/default_acquisition/0/folds/3.1"},
            "t1mri/M12/custom/folds/3.1",
            "t1mri/M12/custom/segmentation",
        )
        assert updated["path_to_graph"] == "t1mri/M12/custom/folds/3.1"

    def test_path_sk_with_hull_overrides_hardcoded_default(self, temp_dir):
        """CLI arg overrides the old hardcoded skeleton default value in the JSON."""
        updated = self._run_with_config(
            temp_dir,
            {
                "graphs_dir": "",
                "output_dir": "",
                "path_to_skeleton_with_hull": "t1mri/default_acquisition/0/segmentation",
            },
            "t1mri/M12/custom/folds/3.1",
            "t1mri/M12/custom/segmentation",
        )
        assert updated["path_to_skeleton_with_hull"] == "t1mri/M12/custom/segmentation"


class TestConfigLocationInvariants:
    """Tests that config is always written to output dir, never to input dir."""

    def test_config_copied_to_output_not_input(self, temp_dir):
        """pipeline_loop_2mm.json is copied to output dir, never to input dir, in Python: `cp` keeps read-only."""
        input_dir, output_dir = stage_dirs(temp_dir)
        # No pre-existing config: forces the copy branch

        script = RunCorticalTiles()
        script.parse_args(
            [
                str(input_dir),
                str(output_dir),
                "--path_to_graph",
                "graphs",
                "--path_sk_with_hull",
                "skeleton",
            ]
        )

        with patch.object(script, "execute_command", return_value=0) as mock_exec:
            script.run()

        assert [call.args[0] for call in mock_exec.call_args_list if call.args[0][0] == "cp"] == []
        assert (output_dir / "pipeline_loop_2mm.json").is_file()
        assert not (input_dir / "pipeline_loop_2mm.json").exists()

    def test_d_arg_points_to_output_not_input(self, temp_dir):
        """-d in the generate_sulcal_regions command is output dir, not input dir."""
        input_dir, output_dir = stage_dirs(temp_dir)
        config_path = output_dir / "pipeline_loop_2mm.json"
        config_path.write_text(json.dumps({"graphs_dir": "", "output_dir": ""}))

        script = RunCorticalTiles()
        script.parse_args(
            [
                str(input_dir),
                str(output_dir),
                "--path_to_graph",
                "graphs",
                "--path_sk_with_hull",
                "skeleton",
            ]
        )

        with (
            patch.object(script, "validate_paths", return_value=True),
            patch.object(script, "execute_command", return_value=0) as mock_exec,
        ):
            script.run()

            final_cmd = mock_exec.call_args_list[-1][0][0]
            d_idx = final_cmd.index("-d")
            assert final_cmd[d_idx + 1] == str(output_dir.resolve())
            assert final_cmd[d_idx + 1] != str(input_dir.resolve())


class TestHypothesisConfigInvariants:
    """Property-based tests: config fields always written correctly for any input."""

    def _run_in_temp(self, config_data):
        """Create fresh input/output dirs, write config in output, run script."""

        tmp = tempfile.mkdtemp()
        try:
            input_dir, output_dir = stage_dirs(tmp, "g")
            config_path = output_dir / "pipeline_loop_2mm.json"
            config_path.write_text(json.dumps(config_data))
            script = RunCorticalTiles()
            script.parse_args(
                [
                    str(input_dir),
                    str(output_dir),
                    "--path_to_graph",
                    "g",
                    "--path_sk_with_hull",
                    "s",
                ]
            )
            with (
                patch.object(script, "validate_paths", return_value=True),
                patch.object(script, "execute_command", return_value=0) as mock_exec,
            ):
                script.run()
                updated = json.loads(config_path.read_text())
                return updated, input_dir, output_dir, mock_exec
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.hypothesis
    def test_graphs_dir_always_overwritten_to_input(self):
        """graphs_dir is always set to abspath(input) regardless of initial value."""

        json_val = st.one_of(
            st.none(),
            st.booleans(),
            st.integers(-10, 10),
            st.text(max_size=30),
            st.just("$local"),
            st.just(""),
        )

        @given(initial=json_val)
        @settings(max_examples=50)
        def inner(initial):
            updated, input_dir, _, _ = self._run_in_temp({"graphs_dir": initial, "output_dir": ""})
            assert updated["graphs_dir"] == str(input_dir.resolve())

        inner()

    @pytest.mark.hypothesis
    def test_output_dir_always_output_plus_derivatives(self):
        """output_dir is always output/DERIVATIVES_FOLDER regardless of initial value."""

        json_val = st.one_of(
            st.none(),
            st.booleans(),
            st.integers(-10, 10),
            st.text(max_size=30),
            st.just("$local"),
            st.just(""),
        )

        @given(initial=json_val)
        @settings(max_examples=50)
        def inner(initial):
            updated, _, output_dir, _ = self._run_in_temp({"graphs_dir": "", "output_dir": initial})
            assert updated["output_dir"] == str(output_dir.resolve() / DERIVATIVES_FOLDER)

        inner()

    @pytest.mark.hypothesis
    def test_extra_config_keys_are_preserved(self):
        """Arbitrary extra keys in the config JSON survive the JSON rewrite."""

        reserved = {"graphs_dir", "output_dir", "skip_distbottom"}
        key_st = st.from_regex(r"[a-z][a-z0-9_]{0,15}", fullmatch=True).filter(lambda k: k not in reserved)
        json_val = st.one_of(
            st.none(),
            st.booleans(),
            st.integers(-10, 10),
            st.text(max_size=30),
            st.just("$local"),
            st.just(""),
        )

        @given(key=key_st, value=json_val)
        @settings(max_examples=30)
        def inner(key, value):
            updated, _, _, _ = self._run_in_temp({"graphs_dir": "", "output_dir": "", key: value})
            assert key in updated
            assert updated[key] == value

        inner()

    @pytest.mark.hypothesis
    def test_d_arg_always_output_for_any_path_names(self):
        """-d is always abspath(output), regardless of directory names."""

        path_part = st.from_regex(r"[a-zA-Z0-9_-]{1,15}", fullmatch=True)

        @given(in_name=path_part, out_name=path_part)
        @settings(max_examples=30)
        def inner(in_name, out_name):
            assume(in_name != out_name)
            tmp = tempfile.mkdtemp()
            try:
                input_dir = Path(tmp) / in_name
                output_dir = Path(tmp) / out_name
                add_graphs(input_dir, graph_dir="g")
                output_dir.mkdir(exist_ok=True)
                (output_dir / "pipeline_loop_2mm.json").write_text(json.dumps({"graphs_dir": "", "output_dir": ""}))
                script = RunCorticalTiles()
                script.parse_args(
                    [
                        str(input_dir),
                        str(output_dir),
                        "--path_to_graph",
                        "g",
                        "--path_sk_with_hull",
                        "s",
                    ]
                )
                with (
                    patch.object(script, "validate_paths", return_value=True),
                    patch.object(script, "execute_command", return_value=0) as mock_exec,
                ):
                    script.run()
                    final_cmd = mock_exec.call_args_list[-1][0][0]
                    d_idx = final_cmd.index("-d")
                    assert final_cmd[d_idx + 1] == str(output_dir.resolve())
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        inner()


class TestRunMethod:
    """Test the run method."""

    def test_run_executes_with_shell_false(self, temp_dir):
        """Test that command is executed with shell=False."""
        add_graphs(temp_dir)
        script = RunCorticalTiles()
        script.parse_args([temp_dir, temp_dir, "--path_to_graph", "graphs", "--path_sk_with_hull", "skeleton"])

        with (
            patch.object(script, "validate_paths", return_value=True),
            patch.object(script, "execute_command", return_value=0) as mock_exec,
        ):
            script.run()

            # Check shell=False
            assert mock_exec.call_args[1]["shell"] is False

    def test_run_returns_command_result(self, temp_dir):
        """Test that run returns the result from execute_command."""
        add_graphs(temp_dir)
        script = RunCorticalTiles()
        script.parse_args([temp_dir, temp_dir, "--path_to_graph", "graphs", "--path_sk_with_hull", "skeleton"])

        with (
            patch.object(script, "validate_paths", return_value=True),
            patch.object(script, "execute_command", return_value=42),
        ):
            result = script.run()
            assert result == 42

    def test_run_logs_input_output(self, temp_dir, pipeline_caplog):
        """run logs the input and output paths at LOG (REQ-LOGGING-BDRABCZUK-72B7182FAE4D, -3CD073B72CB8)."""
        add_graphs(temp_dir)
        script = RunCorticalTiles()
        script.parse_args([temp_dir, temp_dir, "--path_to_graph", "graphs", "--path_sk_with_hull", "skeleton"])

        with (
            patch.object(script, "validate_paths", return_value=True),
            patch.object(script, "execute_command", return_value=0),
        ):
            script.run()

        logged = [
            r.getMessage().lower() for r in pipeline_caplog.records if r.name == _LOGGER_NAME and r.levelno == LOG
        ]
        assert any("input" in message for message in logged), logged
        assert any("output" in message for message in logged), logged


class TestMainFunction:
    """Test the main entry point."""

    def test_main_inits_then_returns_script_main(self):
        """main() calls init_pipeline_process(), then returns the script's ScriptBuilder.main() result."""
        calls = []
        with patch(
            "champollion_pipeline.run_cortical_tiles.init_pipeline_process", side_effect=lambda: calls.append("init")
        ):
            script_main = {"return_value.main.side_effect": lambda: calls.append("main") or 3}
            with patch("champollion_pipeline.run_cortical_tiles.RunCorticalTiles", **script_main) as MockScript:
                result = main()

        MockScript.assert_called_once_with()
        MockScript.return_value.main.assert_called_once_with()
        assert calls == ["init", "main"]
        assert result == 3


@pytest.mark.integration
class TestRunCorticalTilesIntegration:
    """Integration tests for RunCorticalTiles."""

    def test_full_workflow(self, temp_dir):
        """Test complete workflow."""
        script = RunCorticalTiles()
        input_dir, output_dir = stage_dirs(temp_dir, _GRAPH_SUBPATH)

        script.parse_args(
            [
                str(input_dir),
                str(output_dir),
                "--path_to_graph",
                "graphs/path",
                "--path_sk_with_hull",
                "skeleton/path",
                "--njobs",
                "4",
            ]
        )

        with (
            patch.object(script, "execute_command", return_value=0) as mock_exec,
        ):
            result = script.run()

            assert result == 0
            mock_exec.assert_called()


class TestGenerateMaskNpys:
    """Tests for _generate_mask_npys post-processing step."""

    def test_generates_npy_from_nii_gz(self, tmp_path):
        """Converts mask_cropped.nii.gz to mask.npy for each side found."""
        mask_dir = tmp_path / "cortical_tiles-2026" / "crops" / "canonical_corrected_26_1" / "SC-sylv" / "mask"
        mask_dir.mkdir(parents=True)
        (mask_dir / "Lmask_cropped.nii.gz").write_bytes(b"fake")
        (mask_dir / "Rmask_cropped.nii.gz").write_bytes(b"fake")

        script = RunCorticalTiles()

        fake_arr = np.ones((10, 10, 10, 1))
        mock_vol = MagicMock()
        mock_vol.np = fake_arr

        with (
            patch("champollion_pipeline.run_cortical_tiles.aims.read", return_value=mock_vol),
            patch("champollion_pipeline.run_cortical_tiles.np.save") as mock_save,
        ):
            script._generate_mask_npys(str(tmp_path))

        assert mock_save.call_count == 2
        saved_paths = [str(c.args[0]) for c in mock_save.call_args_list]
        assert any("Lmask.npy" in p for p in saved_paths)
        assert any("Rmask.npy" in p for p in saved_paths)

    def test_skips_existing_npy(self, tmp_path):
        """Does not regenerate mask.npy if it already exists."""
        mask_dir = tmp_path / "cortical_tiles-2026" / "crops" / "canonical_corrected_26_1" / "SC-sylv" / "mask"
        mask_dir.mkdir(parents=True)
        (mask_dir / "Lmask_cropped.nii.gz").write_bytes(b"fake")
        (mask_dir / "Lmask.npy").write_bytes(b"existing")

        script = RunCorticalTiles()

        with (
            patch("champollion_pipeline.run_cortical_tiles.aims.read") as mock_read,
            patch("champollion_pipeline.run_cortical_tiles.np.save") as mock_save,
        ):
            script._generate_mask_npys(str(tmp_path))

        mock_read.assert_not_called()
        mock_save.assert_not_called()

    def test_no_nii_gz_files_is_silent(self, tmp_path):
        """Returns without error when no nii.gz mask files exist."""
        script = RunCorticalTiles()
        with patch("champollion_pipeline.run_cortical_tiles.aims.read") as mock_read:
            script._generate_mask_npys(str(tmp_path))
        mock_read.assert_not_called()


@pytest.mark.smoke
class TestRunCorticalTilesSmoke:
    """Smoke tests for basic functionality."""

    def test_script_can_be_instantiated(self):
        """Test that script can be created."""
        script = RunCorticalTiles()
        assert script is not None

    def test_script_has_required_methods(self):
        """Test that script has all required methods."""
        script = RunCorticalTiles()
        assert hasattr(script, "run")
        assert callable(script.run)
