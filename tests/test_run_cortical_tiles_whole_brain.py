#!/usr/bin/env python3
"""
Unit tests for the whole-brain step of run_cortical_tiles.py
(moved out of test_run_cortical_tiles.py; the shared helpers are imported from whole_brain_step.py).
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from tests.stage2_layout import (  # noqa: F401
    _stub_whole_brain_functions,  # noqa: V103 - autouse fixture
    write_whole_brain_volumes,
)
from tests.whole_brain_step import (
    _WILDCARD_SESSION,
    _add_fixture_subject,
    _parsed_fixture_script,
    _resolved_graph_path,
    _run_whole_brain_step,
    _run_wildcard_step,
    _warnings_mentioning,
)


class TestWholeBrainGeneration:
    """Tests for REQ-WHOLEBRAIN-01/-02: unconditional whole-brain volume
    generation, reading skeletons from their real location.

    ``run()`` must, on every invocation (no flag gating), additionally call
    ``add_left_and_right_volumes`` then ``remove_ventricle`` from the
    ``cortical_tiles`` submodule after the existing per-region crop step,
    without disturbing that existing crop step. Both calls must read the
    L/R resampled skeletons from
    ``{output}/cortical_tiles-{YEAR}/skeletons/2mm/`` (where
    ``generate_sulcal_regions.py`` actually writes them), not from the
    Morphologist input directory.
    """

    def _run(self, temp_dir, extra_args=None):
        script, input_dir, output_dir = _parsed_fixture_script(temp_dir, extra_args)

        with (
            patch.object(script, "validate_paths", return_value=True),
            patch.object(script, "execute_command", return_value=0) as mock_exec,
            patch.object(script, "_generate_mask_npys") as mock_gen_npys,
            patch(
                "champollion_pipeline.run_cortical_tiles.add_left_and_right_volumes.add_left_and_right_volumes"
            ) as mock_add,
            patch("champollion_pipeline.run_cortical_tiles.remove_ventricle.remove_ventricle") as mock_remove,
        ):
            script.run()

        return script, input_dir, output_dir, mock_exec, mock_gen_npys, mock_add, mock_remove

    def test_whole_brain_functions_called_unconditionally(self, temp_dir):
        """add_left_and_right_volumes and remove_ventricle both run with no flag needed."""
        _, _, _, _, _, mock_add, mock_remove = self._run(temp_dir)

        mock_add.assert_called_once()
        mock_remove.assert_called_once()

    def test_add_left_and_right_volumes_called_with_skeleton_dir(self, temp_dir):
        """add_left_and_right_volumes receives src_dir=<output>/cortical_tiles-YEAR/skeletons/2mm, parallel=True.

        NOT the Morphologist input dir -- that directory has no L/ or R/
        subdirs and is read-only (REQ-WHOLEBRAIN-02).
        """
        _, _, output_dir, _, _, mock_add, _ = self._run(temp_dir)

        _, kwargs = mock_add.call_args
        expected_skeleton_dir = str(output_dir.resolve() / "cortical_tiles-2026" / "skeletons" / "2mm")
        assert kwargs["src_dir"] == expected_skeleton_dir
        assert kwargs["parallel"] is True

    def test_remove_ventricle_called_with_side_f_and_path_to_graph(self, temp_dir):
        """remove_ventricle receives side='F', src_dir=<skeleton dir>, morpho_dir=<input dir>, path_to_graph.

        src_dir and morpho_dir are intentionally different: src_dir is where
        the L/R resampled skeletons (and the fused F/ output written by
        add_left_and_right_volumes) actually live, while morpho_dir is the
        Morphologist directory holding the labelled graphs used to strip the
        ventricle (REQ-WHOLEBRAIN-02).
        """
        script, input_dir, output_dir, _, _, _, mock_remove = self._run(temp_dir)

        _, kwargs = mock_remove.call_args
        expected_skeleton_dir = str(output_dir.resolve() / "cortical_tiles-2026" / "skeletons" / "2mm")
        assert kwargs["side"] == "F"
        assert kwargs["src_dir"] == expected_skeleton_dir
        assert kwargs["morpho_dir"] == str(input_dir.resolve())
        assert kwargs["path_to_graph"] == script.args.path_to_graph
        assert kwargs["parallel"] is True

    def test_add_and_remove_ventricle_use_the_same_skeleton_src_dir(self, temp_dir):
        """The fused F/ tree add_left_and_right_volumes writes must be exactly
        what remove_ventricle(side='F') reads back -- both calls must agree
        on src_dir (REQ-WHOLEBRAIN-02)."""
        _, _, _, _, _, mock_add, mock_remove = self._run(temp_dir)

        assert mock_add.call_args.kwargs["src_dir"] == mock_remove.call_args.kwargs["src_dir"]

    def test_remove_ventricle_called_with_transform_dir(self, temp_dir):
        """remove_ventricle receives transform_dir=<output>/cortical_tiles-YEAR/transforms (REQ-WHOLEBRAIN-03).

        That is where generate_sulcal_regions.py (via
        generate_ICBM2009c_transforms) writes the per-subject L/R
        ``{side}transform_to_ICBM2009c_{subject}.trm`` files. Without it,
        remove_ventricle(side='F') applies native-grid ventricle indices to
        the resampled 2mm skeleton instead of the REQ-FULLBRAIN-05
        resampled-grid mapping.
        """
        _, _, output_dir, _, _, _, mock_remove = self._run(temp_dir)

        expected_transform_dir = str(output_dir.resolve() / "cortical_tiles-2026" / "transforms")
        assert mock_remove.call_args.kwargs.get("transform_dir") == expected_transform_dir

    def test_remove_ventricle_called_with_default_labelling_session(self, temp_dir):
        """Without --labelling_session, remove_ventricle receives labelling_session='deepcnn_session_auto'
        for the fixture subject, whose labelled graphs exist under that session
        (REQ-LABELSESSION-01, superseded by the session auto-detection requirement;
        see TestLabellingSessionAutoDetection).

        Same value as cortical_tiles.brainvisa.remove_ventricle._LABELLING_SESSION_DEFAULT,
        so existing runs keep their current behaviour -- but it must be passed
        explicitly, not left to the submodule's own default.
        """
        _, _, _, _, _, _, mock_remove = self._run(temp_dir)

        assert mock_remove.call_args.kwargs.get("labelling_session") == "deepcnn_session_auto"

    def test_remove_ventricle_called_with_cli_labelling_session(self, temp_dir):
        """--labelling_session <S> is forwarded as remove_ventricle(labelling_session=<S>) (REQ-LABELSESSION-01).

        Real datasets (e.g. DEMO01) hold their labelled graphs under session
        '0_auto'; without this override remove_ventricle looks for
        'deepcnn_session_auto' graphs, hits FileNotFoundError per subject, and
        its broad exception handling silently produces no whole-brain output.
        """
        _, _, _, _, _, _, mock_remove = self._run(temp_dir, extra_args=["--labelling_session", "0_auto"])

        assert mock_remove.call_args.kwargs.get("labelling_session") == "0_auto"

    def test_remove_ventricle_called_with_bids_false_by_default(self, temp_dir):
        """Without --bids, remove_ventricle receives bids=False explicitly (REQ-BIDS-02)."""
        _, _, _, _, _, _, mock_remove = self._run(temp_dir)

        assert mock_remove.call_args.kwargs.get("bids") is False

    def test_remove_ventricle_called_with_bids_true_when_flag_set(self, temp_dir):
        """--bids is forwarded as remove_ventricle(bids=True) (REQ-BIDS-02)."""
        _, _, _, _, _, _, mock_remove = self._run(temp_dir, extra_args=["--bids"])

        assert mock_remove.call_args.kwargs.get("bids") is True

    def test_add_left_and_right_volumes_called_before_remove_ventricle(self, temp_dir):
        """Fusion must happen before ventricle removal reads the fused F/ directory."""
        call_order = []

        # Re-implement _run inline to track call order via side_effect.

        tmp = tempfile.mkdtemp()
        try:
            input_dir_path = Path(tmp) / "input"
            output_dir_path = Path(tmp) / "output"
            input_dir_path.mkdir()
            output_dir_path.mkdir()
            _add_fixture_subject(input_dir_path, output_dir_path)

            script = RunCorticalTiles()
            script.parse_args(
                [
                    str(input_dir_path),
                    str(output_dir_path),
                    "--path_to_graph",
                    "graphs/path",
                    "--path_sk_with_hull",
                    "skeleton/path",
                ]
            )

            with (
                patch.object(script, "validate_paths", return_value=True),
                patch.object(script, "execute_command", return_value=0),
                patch.object(script, "_generate_mask_npys"),
                patch(
                    "champollion_pipeline.run_cortical_tiles.add_left_and_right_volumes.add_left_and_right_volumes",
                    side_effect=lambda **kw: call_order.append("add_left_and_right_volumes"),
                ),
                patch(
                    "champollion_pipeline.run_cortical_tiles.remove_ventricle.remove_ventricle",
                    side_effect=lambda **kw: call_order.append("remove_ventricle"),
                ),
            ):
                script.run()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        assert call_order == ["add_left_and_right_volumes", "remove_ventricle"]

    def test_crop_subprocess_still_runs_alongside_whole_brain_step(self, temp_dir):
        """The existing generate_sulcal_regions.py crop subprocess is unaffected."""
        _, _, _, mock_exec, _, _, _ = self._run(temp_dir)

        crop_calls = [
            c for c in mock_exec.call_args_list if any("generate_sulcal_regions.py" in str(arg) for arg in c[0][0])
        ]
        assert len(crop_calls) == 1

    def test_generate_mask_npys_still_runs_for_crop_step(self, temp_dir):
        """_generate_mask_npys (crop-mask-specific) is untouched by the whole-brain addition."""
        _, _, _, _, mock_gen_npys, _, _ = self._run(temp_dir)

        mock_gen_npys.assert_called_once()

    def test_whole_brain_step_runs_regardless_of_crop_only_args(self, temp_dir):
        """Crop-only args (e.g. --input-types) don't gate or alter the whole-brain step."""
        _, _, _, _, _, mock_add, mock_remove = self._run(
            temp_dir, extra_args=["--input-types", "skeleton", "foldlabel"]
        )

        mock_add.assert_called_once()
        mock_remove.assert_called_once()


class TestCropFailureStopsDownstreamSteps:
    """Tests for REQ-WHOLEBRAIN-04: a failed crop subprocess stops run().

    ``ScriptBuilder.execute_command`` swallows subprocess exceptions and
    returns 1 (or the shell return code) instead of raising. When the
    generate_sulcal_regions.py crop call reports failure, run() must return
    that value without going on to ``_generate_mask_npys`` or the whole-brain
    fusion (``add_left_and_right_volumes`` / ``remove_ventricle``) -- otherwise
    fusion runs on incomplete L/R data (e.g. a left-only whole-brain volume
    when the right hemisphere's skeletons were never generated).
    """

    def _run(self, temp_dir, crop_return_code):
        script, _, _ = _parsed_fixture_script(temp_dir)

        with (
            patch.object(script, "validate_paths", return_value=True),
            patch.object(script, "execute_command", return_value=crop_return_code) as mock_exec,
            patch.object(script, "_generate_mask_npys") as mock_gen_npys,
            patch(
                "champollion_pipeline.run_cortical_tiles.add_left_and_right_volumes.add_left_and_right_volumes"
            ) as mock_add,
            patch(
                "champollion_pipeline.run_cortical_tiles.remove_ventricle.remove_ventricle",
                side_effect=write_whole_brain_volumes,
            ) as mock_remove,
        ):
            result = script.run()

        return result, mock_exec, mock_gen_npys, mock_add, mock_remove

    @pytest.mark.parametrize("crop_return_code", [1, 2])
    def test_crop_failure_skips_generate_mask_npys(self, temp_dir, crop_return_code):
        """Non-zero crop return code: _generate_mask_npys is not called."""
        _, _, mock_gen_npys, _, _ = self._run(temp_dir, crop_return_code)

        mock_gen_npys.assert_not_called()

    @pytest.mark.parametrize("crop_return_code", [1, 2])
    def test_crop_failure_skips_add_left_and_right_volumes(self, temp_dir, crop_return_code):
        """Non-zero crop return code: L+R whole-brain fusion is not called."""
        _, _, _, mock_add, _ = self._run(temp_dir, crop_return_code)

        mock_add.assert_not_called()

    @pytest.mark.parametrize("crop_return_code", [1, 2])
    def test_crop_failure_skips_remove_ventricle(self, temp_dir, crop_return_code):
        """Non-zero crop return code: whole-brain ventricle removal is not called."""
        _, _, _, _, mock_remove = self._run(temp_dir, crop_return_code)

        mock_remove.assert_not_called()

    @pytest.mark.parametrize("crop_return_code", [1, 2])
    def test_crop_failure_returns_crop_return_code(self, temp_dir, crop_return_code):
        """Non-zero crop return code is returned by run() unchanged (becomes the process exit code via main())."""
        result, mock_exec, _, _, _ = self._run(temp_dir, crop_return_code)

        mock_exec.assert_called_once()
        assert result == crop_return_code

    def test_crop_success_still_runs_all_downstream_steps(self, temp_dir):
        """Regression guard: zero crop return code still runs mask npys, fusion and ventricle removal, returns 0."""
        result, _, mock_gen_npys, mock_add, mock_remove = self._run(temp_dir, 0)

        mock_gen_npys.assert_called_once()
        mock_add.assert_called_once()
        mock_remove.assert_called_once()
        assert result == 0


class TestLabellingSessionAutoDetection:
    """Labelling session passed to remove_ventricle, per subject.

    Without --labelling_session, each subject gets the first of
    deepcnn_session_auto and default_session_auto whose labelled graph
    directory exists under ``<input>/<subject>/<path_to_graph>/``; with the
    option, the option's value. Datasets labelled only under
    default_session_auto (FB-LABELSESSION-BDRABCZUK-C507316C48E3) otherwise
    fail with FileNotFoundError per subject.
    """

    def test_auto_session_prefers_deepcnn_session_when_both_exist(self, temp_dir):
        recorder, _, _ = _run_whole_brain_step(temp_dir, {"subj01": ["deepcnn_session_auto", "default_session_auto"]})

        assert recorder.sessions_by_subject == {"subj01": ["deepcnn_session_auto"]}

    def test_auto_session_falls_back_to_default_session_auto(self, temp_dir):
        recorder, _, _ = _run_whole_brain_step(temp_dir, {"subj02": ["default_session_auto"]})

        assert recorder.sessions_by_subject == {"subj02": ["default_session_auto"]}

    def test_auto_session_is_resolved_per_subject(self, temp_dir):
        recorder, _, _ = _run_whole_brain_step(
            temp_dir, {"subj01": ["deepcnn_session_auto"], "subj02": ["default_session_auto"]}
        )

        assert recorder.sessions_by_subject == {
            "subj01": ["deepcnn_session_auto"],
            "subj02": ["default_session_auto"],
        }

    def test_auto_session_ignores_non_candidate_sessions(self, temp_dir):
        """Only deepcnn_session_auto and default_session_auto are candidates; '0_auto' is not."""
        recorder, _, _ = _run_whole_brain_step(temp_dir, {"subj03": ["0_auto", "default_session_auto"]})

        assert recorder.sessions_by_subject == {"subj03": ["default_session_auto"]}

    def test_explicit_labelling_session_is_passed_for_each_processed_subject(self, temp_dir):
        recorder, _, _ = _run_whole_brain_step(
            temp_dir,
            {"subj01": ["0_auto", "deepcnn_session_auto"], "subj02": ["0_auto"]},
            extra_args=["--labelling_session", "0_auto"],
        )

        assert recorder.sessions_by_subject == {"subj01": ["0_auto"], "subj02": ["0_auto"]}


class TestSubjectsWithoutLabelledGraph:
    """Subjects with no labelled graph under deepcnn_session_auto or
    default_session_auto (no sulcal recognition run) are left out of
    whole-brain ventricle removal, with one warning naming them.
    """

    def test_subject_without_labelled_graph_is_excluded_from_ventricle_removal(self, temp_dir):
        recorder, _, _ = _run_whole_brain_step(temp_dir, {"subj01": ["deepcnn_session_auto"], "subj09": []})

        assert recorder.sessions_by_subject == {"subj01": ["deepcnn_session_auto"]}

    def test_subject_with_only_non_candidate_session_is_excluded(self, temp_dir):
        recorder, _, _ = _run_whole_brain_step(temp_dir, {"subj01": ["deepcnn_session_auto"], "subj07": ["0_auto"]})

        assert "subj07" not in recorder.sessions_by_subject

    def test_no_subject_processed_when_none_has_labelled_graph(self, temp_dir):
        recorder, _, _ = _run_whole_brain_step(temp_dir, {"subj08": [], "subj09": []})

        assert recorder.sessions_by_subject == {}

    def test_one_warning_lists_excluded_subjects_and_requires_sulcal_recognition(self, temp_dir, pipeline_caplog):
        _run_whole_brain_step(temp_dir, {"subj01": ["deepcnn_session_auto"], "subj08": [], "subj09": []})

        matching = _warnings_mentioning("sulcal recognition", pipeline_caplog)
        assert len(matching) == 1
        assert "subj08" in matching[0]
        assert "subj09" in matching[0]
        assert "subj01" not in matching[0]

    def test_no_sulcal_recognition_warning_when_no_subject_is_excluded(self, temp_dir, pipeline_caplog):
        _run_whole_brain_step(temp_dir, {"subj01": ["deepcnn_session_auto"]})

        assert _warnings_mentioning("sulcal recognition", pipeline_caplog) == []

    def test_run_returns_zero_when_subjects_are_excluded_after_successful_crop(self, temp_dir):
        """Exit status stays 0 when only whole-brain ventricle removal is skipped:
        the crops succeeded and the skip is reported by the warning (user decision 2026-10-27)."""
        _, _, result = _run_whole_brain_step(temp_dir, {"subj01": ["deepcnn_session_auto"], "subj08": [], "subj09": []})

        assert result == 0

    def test_run_returns_zero_when_every_subject_is_excluded_after_successful_crop(self, temp_dir):
        _, _, result = _run_whole_brain_step(temp_dir, {"subj08": [], "subj09": []})

        assert result == 0


class TestPathToGraphWildcardResolution:
    """Non-BIDS --path_to_graph holding ``*`` (as the README advises), resolved per subject.

    cortical_tiles' remove_ventricle joins path_to_graph literally without --bids,
    so run() passes it, per subject, the one folder under ``<input>/<subject>``
    that path_to_graph matches with ``*`` as a wildcard; a subject matching no
    folder, or two or more, is excluded from whole-brain ventricle removal
    (coupling report risk 2).
    """

    def test_wildcard_resolved_to_the_matched_folder(self, temp_dir):
        """REQ-WHOLEBRAIN-BDRABCZUK-8028CE6C4890."""
        recorder, _, _ = _run_wildcard_step(temp_dir, {"subj01": ["default_acquisition"]})

        assert recorder.path_to_graph_by_subject == {"subj01": [_resolved_graph_path("default_acquisition")]}

    def test_wildcard_resolved_per_subject(self, temp_dir):
        """REQ-WHOLEBRAIN-BDRABCZUK-8028CE6C4890: subjects matching different folders
        each get their own folder."""
        recorder, _, _ = _run_wildcard_step(temp_dir, {"subj01": ["acq_a"], "subj02": ["acq_b"]})

        assert recorder.path_to_graph_by_subject == {
            "subj01": [_resolved_graph_path("acq_a")],
            "subj02": [_resolved_graph_path("acq_b")],
        }
        assert recorder.sessions_by_subject == {"subj01": [_WILDCARD_SESSION], "subj02": [_WILDCARD_SESSION]}

    def test_subject_with_no_matching_folder_is_excluded(self, temp_dir):
        """REQ-WHOLEBRAIN-BDRABCZUK-D1E16A7827A2."""
        recorder, _, _ = _run_wildcard_step(temp_dir, {"subj01": ["default_acquisition"], "subj05": []})

        assert recorder.path_to_graph_by_subject == {"subj01": [_resolved_graph_path("default_acquisition")]}

    def test_subject_with_several_matching_folders_is_excluded(self, temp_dir):
        """REQ-WHOLEBRAIN-BDRABCZUK-3AE09F03A6E7: an ambiguous match is not guessed."""
        recorder, _, _ = _run_wildcard_step(temp_dir, {"subj01": ["default_acquisition"], "subj06": ["acq_a", "acq_b"]})

        assert recorder.path_to_graph_by_subject == {"subj01": [_resolved_graph_path("default_acquisition")]}

    def test_unresolved_subjects_are_listed_in_the_one_exclusion_warning(self, temp_dir, pipeline_caplog):
        """REQ-WHOLEBRAIN-BDRABCZUK-D1E16A7827A2 and -3AE09F03A6E7 exclusions go through
        the existing one-line warning (REQ-LABELSESSION-BDRABCZUK-56365C1DEE1F)."""
        _run_wildcard_step(temp_dir, {"subj01": ["default_acquisition"], "subj05": [], "subj06": ["acq_a", "acq_b"]})

        matching = _warnings_mentioning("sulcal recognition", pipeline_caplog)
        assert len(matching) == 1
        assert "subj05" in matching[0]
        assert "subj06" in matching[0]
        assert "subj01" not in matching[0]
