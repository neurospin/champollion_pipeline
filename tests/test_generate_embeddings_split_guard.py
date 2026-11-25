"""
Guard for the split of tests/test_generate_embeddings.py into topic files (TASK-CODE-BDRABCZUK-73BD87657EA2, refactor,
behaviour-preserving).

The reference is tests/test_generate_embeddings.py at WIP commit 52d7e16: its 70 class-qualified test function names are
frozen in REFERENCE_TEST_NAMES below. The split keeps each of them defined in exactly one file matching
tests/test_generate_embeddings*.py, and keeps each of those files within the project's max_file_lines (1000).

Requirements:
REQ-GESPLIT-BDRABCZUK-F5CEB05BC37D (each reference test name defined in exactly one test_generate_embeddings*.py file),
REQ-GESPLIT-BDRABCZUK-65BC0E127058 (each test_generate_embeddings*.py file at most 1000 lines; budget
  generate_embeddings_test_file_lines_max).
"""

from tests.split_guard import assert_each_name_defined_once, assert_files_within_limit

SPLIT_GLOB = "test_generate_embeddings*.py"
MEASURE = "generate_embeddings_test_file_lines_max"

# Class-qualified test function names of tests/test_generate_embeddings.py at commit 52d7e16, in file order.
REFERENCE_TEST_NAMES = (
    "TestGenerateEmbeddingsInit::test_init_creates_script",
    "TestGenerateEmbeddingsArguments::test_parse_required_arguments",
    "TestGenerateEmbeddingsArguments::test_flags_default_to_false",
    "TestGenerateEmbeddingsArguments::test_flags_can_be_set",
    "TestPerRegionInvocation::test_run_per_region_calls_execute_for_each_region",
    "TestPerRegionInvocation::test_run_per_region_skips_existing_output_without_overwrite",
    "TestPerRegionInvocation::test_run_per_region_runs_when_overwrite_set",
    "TestModelDiscovery::test_discovery_descends_into_mask_version_parent_directory",
    "TestModelDiscovery::test_discovery_ignores_directory_without_logs_marker",
    "TestModelDiscovery::test_discovery_skips_dot_directory_subtree",
    "TestRegionNameResolution::test_nested_hf_layout_skeleton_path_uses_region_side_folder",
    "TestRegionNameResolution::test_nested_hf_layout_subjects_path_uses_region_side_folder",
    "TestRegionNameResolution::test_nested_hf_layout_saving_path_uses_region_side_folder",
    "TestRegionNameResolution::test_nested_hf_layout_model_path_is_fold_directory",
    "TestRegionNameResolution::test_flat_layout_paths_unchanged",
    "TestRunMethod::test_run_calls_run_per_region",
    "TestRunMethod::test_run_returns_result",
    "TestLegacyCropsPath::test_legacy_crops_dir_has_no_mask_version_segment",
    "TestLegacyCropsPath::test_legacy_crops_dir_ignores_explicit_masks_option",
    "TestLegacyCropsPath::test_non_legacy_crops_dir_keeps_mask_version_segment",
    "TestGenerateEmbeddingsIntegration::test_full_workflow",
    "TestCorticalVersionFlags::test_cortical_version_default",
    "TestCorticalVersionFlags::test_cortical_version_can_be_overridden",
    "TestCorticalVersionFlags::test_legacy_flag_defaults_false",
    "TestCorticalVersionFlags::test_legacy_flag_can_be_set",
    "TestCorticalVersionFlags::test_get_derivatives_folder_returns_cortical_version_by_default",
    "TestCorticalVersionFlags::test_get_derivatives_folder_legacy_overrides",
    "TestCorticalVersionFlags::test_get_derivatives_folder_cortical_version_overrides",
    "TestCorticalVersionFlags::test_patch_config_paths_rewrites_derivatives_folder",
    "TestCorticalVersionFlags::test_patch_config_paths_leaves_unrelated_lines_intact",
    "TestCorticalVersionFlags::test_patch_config_paths_skips_already_correct_files",
    "TestCorticalVersionFlags::test_patch_config_paths_recurses_into_subdirs",
    "TestRegionsFilter::test_regions_default_none",
    "TestRegionsFilter::test_regions_single",
    "TestRegionsFilter::test_regions_multiple",
    "TestRegionsFilter::test_make_regions_tmpdir_creates_symlinks",
    "TestRegionsFilter::test_make_regions_tmpdir_raises_on_missing_region",
    "TestRegionsFilter::test_run_pipeline_uses_tmpdir_when_regions_set",
    "TestRegionsFilter::test_run_pipeline_cleans_up_tmpdir_on_error",
    "TestRegionsFilter::test_run_pipeline_no_tmpdir_without_regions",
    "TestProfiling::test_run_dispatches_to_profiling_when_flag_set",
    "TestProfiling::test_run_dispatches_to_normal_without_profiling_flag",
    "TestProfiling::test_run_with_profiling_calls_run_normal",
    "TestProfiling::test_run_with_profiling_dumps_profile_file",
    "TestProfiling::test_run_with_profiling_dumps_on_error",
    "TestGenerateEmbeddingsSmoke::test_script_can_be_instantiated",
    "TestGenerateEmbeddingsSmoke::test_script_has_run_method",
    "TestHuggingFaceStrategy::test_fetch_without_subfolder_calls_snapshot_download_no_allow_patterns",
    "TestHuggingFaceStrategy::test_fetch_with_subfolder_uses_allow_patterns",
    "TestHuggingFaceStrategy::test_fetch_never_passes_subfolder_kwarg",
    "TestHuggingFaceStrategy::test_fetch_with_subfolder_returns_path_including_subfolder",
    "TestHuggingFaceStrategy::test_fetch_without_subfolder_returns_snapshot_root_unchanged",
    "TestPixiTaskPaths::test_all_pixi_task_scripts_exist",
    "TestPerRegionSubjectsFile::test_left_region_receives_its_own_lskeleton_subject_csv",
    "TestPerRegionSubjectsFile::test_right_region_receives_its_own_rskeleton_subject_csv",
    "TestEmbeddingsLogging::test_fold_progress_logged_per_region_model",
    "TestEmbeddingsLogging::test_fold_progress_logged_for_skipped_region",
    "TestEmbeddingsLogging::test_run_start_logged_with_output_directory",
    "TestEmbeddingsLogging::test_skipped_region_logged",
    "TestEmbeddingsLogging::test_region_weights_logged",
    "TestEmbeddingsLogging::test_best_weights_stand_in_logged_as_warning",
    "TestEmbeddingsLogging::test_failed_region_logged_at_fail",
    "TestEmbeddingsLogging::test_provenance_path_logged_at_output",
    "TestEmbeddingsLogging::test_cpu_mode_logged",
    "TestEmbeddingsLogging::test_profile_file_logged_at_output",
    "TestEmbeddingsLogging::test_profile_statistics_logged_at_output",
    "TestEmbeddingsLogging::test_cka_output_directory_logged",
    "TestEmbeddingsLogging::test_cka_completion_logged_at_output",
    "TestEmbeddingsLogging::test_cka_failure_logged_at_warning",
    "TestEmbeddingsLogging::test_module_has_no_print_call",
)


def test_reference_test_names_each_defined_once():
    """REQ-GESPLIT-BDRABCZUK-F5CEB05BC37D: each 52d7e16 test name is defined in exactly one split file."""
    assert_each_name_defined_once(REFERENCE_TEST_NAMES, 70, SPLIT_GLOB)


def test_generate_embeddings_test_files_at_most_1000_lines():
    """REQ-GESPLIT-BDRABCZUK-65BC0E127058: each test_generate_embeddings*.py file has at most 1000 lines."""
    assert_files_within_limit(SPLIT_GLOB, MEASURE)
