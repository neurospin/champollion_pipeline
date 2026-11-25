"""
Guard for the split of tests/test_run_cortical_tiles.py into topic files (TASK-CODE-BDRABCZUK-5F39ABE65D2D, refactor,
behaviour-preserving).

The reference is tests/test_run_cortical_tiles.py at WIP commit 5c291de: its 65 class-qualified test function names are
frozen in REFERENCE_TEST_NAMES below. The split keeps each of them defined in exactly one file matching
tests/test_run_cortical_tiles*.py, and keeps each of those files within the project's max_file_lines (1000).

Requirements:
REQ-CTSPLIT-BDRABCZUK-5EFA5CDE7D87 (each reference test name defined in exactly one test_run_cortical_tiles*.py file),
REQ-CTSPLIT-BDRABCZUK-E1344F241964 (each test_run_cortical_tiles*.py file at most 1000 lines; budget
  run_cortical_tiles_test_file_lines_max).
"""

from tests.split_guard import assert_each_name_defined_once, assert_files_within_limit

SPLIT_GLOB = "test_run_cortical_tiles*.py"
MEASURE = "run_cortical_tiles_test_file_lines_max"

# Class-qualified test function names of tests/test_run_cortical_tiles.py at commit 5c291de, in file order.
REFERENCE_TEST_NAMES = (
    "TestRunCorticalTilesInit::test_init_creates_script",
    "TestRunCorticalTilesInit::test_init_configures_arguments",
    "TestBuildCommand::test_command_includes_required_parameters",
    "TestBuildCommand::test_command_includes_input_types_when_set",
    "TestBuildCommand::test_command_includes_sk_qc_path_when_set",
    "TestSkipDistbottom::test_graphs_dir_set_to_input",
    "TestSkipDistbottom::test_output_dir_set_from_output_arg",
    "TestSkipDistbottom::test_config_with_null_values_overwritten_correctly",
    "TestBidsConfig::test_bids_flag_sets_bids_true_in_config",
    "TestBidsConfig::test_no_bids_flag_leaves_bids_unchanged_in_config",
    "TestGraphPathConfigOverride::test_path_to_graph_written_to_config",
    "TestGraphPathConfigOverride::test_path_sk_with_hull_written_to_config",
    "TestGraphPathConfigOverride::test_path_to_graph_overrides_hardcoded_default",
    "TestGraphPathConfigOverride::test_path_sk_with_hull_overrides_hardcoded_default",
    "TestConfigLocationInvariants::test_config_copied_to_output_not_input",
    "TestConfigLocationInvariants::test_d_arg_points_to_output_not_input",
    "TestHypothesisConfigInvariants::test_graphs_dir_always_overwritten_to_input",
    "TestHypothesisConfigInvariants::test_output_dir_always_output_plus_derivatives",
    "TestHypothesisConfigInvariants::test_extra_config_keys_are_preserved",
    "TestHypothesisConfigInvariants::test_d_arg_always_output_for_any_path_names",
    "TestRunMethod::test_run_executes_with_shell_false",
    "TestRunMethod::test_run_returns_command_result",
    "TestRunMethod::test_run_logs_input_output",
    "TestMainFunction::test_main_inits_then_returns_script_main",
    "TestRunCorticalTilesIntegration::test_full_workflow",
    "TestGenerateMaskNpys::test_generates_npy_from_nii_gz",
    "TestGenerateMaskNpys::test_skips_existing_npy",
    "TestGenerateMaskNpys::test_no_nii_gz_files_is_silent",
    "TestWholeBrainGeneration::test_whole_brain_functions_called_unconditionally",
    "TestWholeBrainGeneration::test_add_left_and_right_volumes_called_with_skeleton_dir",
    "TestWholeBrainGeneration::test_remove_ventricle_called_with_side_f_and_path_to_graph",
    "TestWholeBrainGeneration::test_add_and_remove_ventricle_use_the_same_skeleton_src_dir",
    "TestWholeBrainGeneration::test_remove_ventricle_called_with_transform_dir",
    "TestWholeBrainGeneration::test_remove_ventricle_called_with_default_labelling_session",
    "TestWholeBrainGeneration::test_remove_ventricle_called_with_cli_labelling_session",
    "TestWholeBrainGeneration::test_remove_ventricle_called_with_bids_false_by_default",
    "TestWholeBrainGeneration::test_remove_ventricle_called_with_bids_true_when_flag_set",
    "TestWholeBrainGeneration::test_add_left_and_right_volumes_called_before_remove_ventricle",
    "TestWholeBrainGeneration::test_crop_subprocess_still_runs_alongside_whole_brain_step",
    "TestWholeBrainGeneration::test_generate_mask_npys_still_runs_for_crop_step",
    "TestWholeBrainGeneration::test_whole_brain_step_runs_regardless_of_crop_only_args",
    "TestCropFailureStopsDownstreamSteps::test_crop_failure_skips_generate_mask_npys",
    "TestCropFailureStopsDownstreamSteps::test_crop_failure_skips_add_left_and_right_volumes",
    "TestCropFailureStopsDownstreamSteps::test_crop_failure_skips_remove_ventricle",
    "TestCropFailureStopsDownstreamSteps::test_crop_failure_returns_crop_return_code",
    "TestCropFailureStopsDownstreamSteps::test_crop_success_still_runs_all_downstream_steps",
    "TestLabellingSessionAutoDetection::test_auto_session_prefers_deepcnn_session_when_both_exist",
    "TestLabellingSessionAutoDetection::test_auto_session_falls_back_to_default_session_auto",
    "TestLabellingSessionAutoDetection::test_auto_session_is_resolved_per_subject",
    "TestLabellingSessionAutoDetection::test_auto_session_ignores_non_candidate_sessions",
    "TestLabellingSessionAutoDetection::test_explicit_labelling_session_is_passed_for_each_processed_subject",
    "TestSubjectsWithoutLabelledGraph::test_subject_without_labelled_graph_is_excluded_from_ventricle_removal",
    "TestSubjectsWithoutLabelledGraph::test_subject_with_only_non_candidate_session_is_excluded",
    "TestSubjectsWithoutLabelledGraph::test_no_subject_processed_when_none_has_labelled_graph",
    "TestSubjectsWithoutLabelledGraph::test_one_warning_lists_excluded_subjects_and_requires_sulcal_recognition",
    "TestSubjectsWithoutLabelledGraph::test_no_sulcal_recognition_warning_when_no_subject_is_excluded",
    "TestSubjectsWithoutLabelledGraph::test_run_returns_zero_when_subjects_are_excluded_after_successful_crop",
    "TestSubjectsWithoutLabelledGraph::test_run_returns_zero_when_every_subject_is_excluded_after_successful_crop",
    "TestPathToGraphWildcardResolution::test_wildcard_resolved_to_the_matched_folder",
    "TestPathToGraphWildcardResolution::test_wildcard_resolved_per_subject",
    "TestPathToGraphWildcardResolution::test_subject_with_no_matching_folder_is_excluded",
    "TestPathToGraphWildcardResolution::test_subject_with_several_matching_folders_is_excluded",
    "TestPathToGraphWildcardResolution::test_unresolved_subjects_are_listed_in_the_one_exclusion_warning",
    "TestRunCorticalTilesSmoke::test_script_can_be_instantiated",
    "TestRunCorticalTilesSmoke::test_script_has_required_methods",
)


def test_reference_test_names_each_defined_once():
    """REQ-CTSPLIT-BDRABCZUK-5EFA5CDE7D87: each 5c291de test name is defined in exactly one split file."""
    assert_each_name_defined_once(REFERENCE_TEST_NAMES, 65, SPLIT_GLOB)


def test_run_cortical_tiles_test_files_at_most_1000_lines():
    """REQ-CTSPLIT-BDRABCZUK-E1344F241964: each test_run_cortical_tiles*.py file has at most 1000 lines."""
    assert_files_within_limit(SPLIT_GLOB, MEASURE)
