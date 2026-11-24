"""main.py stops before any stage when a local model set and the Mask version of the run disagree
(TASK-MODELVER-BDRABCZUK-3B2E4833B088, Defect DEF-MODELVER-BDRABCZUK-291232AA949D, user decisions 2026-11-17 09:49).

The per-region Mask version resolver (models.version_options.resolve_region_mask_versions) gives each region model
name of a local model set its Mask version (model_set.json, else PROVENANCE.csv, else the folder name), or none
(unresolved) or two (conflict). In sequential mode (the default; streaming mode keeps today's behaviour, user
decision 2026-11-24 10:27) and given no per-region Mask version map, main.py checks the run's region model names
before stage 1:

- no --masks-version, a local model set covering each processed region model name (a partial set is refused earlier,
  REQ-MODELVER-BDRABCZUK-B463E4170F9B) and a region model name without one Mask version: exit 2 before any stage,
  with one ERROR record naming models_path and asking for --masks-version (the resolver's own ERROR record on a
  broken PROVENANCE.csv is the cause, REQ-MODELVER-BDRABCZUK-D1508E867131 / 844D71D1C93E; this one is the stop);
- --masks-version <V> differing from a region model name's resolved Mask version <K>: exit 2 before any stage, with
  one ERROR record naming <V>, each such region model name and its <K> (as stage 4 does,
  REQ-MODELVER-BDRABCZUK-440EB622BC61, 524734E7EFE7);
- the same with --allow-version-mismatch: one WARNING record per region model name naming <R>, <V> and <K> before any
  stage (as stage 4, REQ-MODELVER-BDRABCZUK-2C7538CAE27C), no Python UserWarning, and the run goes on.

Requirements (wording in the ledger, all derived from REQ-MODELVER-BDRABCZUK-1A70BB31226B):
    test_unresolved_region_without_masks_version_exits_2_before_any_stage
    test_unresolved_region_without_masks_version_logs_one_error_record
    test_masks_version_mismatch_exits_2_before_any_stage
    test_masks_version_mismatch_logs_one_error_record_naming_each_region
    test_allowed_masks_version_mismatch_logs_one_warning_record_per_region
    test_allowed_masks_version_mismatch_runs_the_enabled_stages
    test_allowed_masks_version_mismatch_emits_no_user_warning
    test_allowed_masks_version_mismatch_passes_masks_version_to_stages_2_to_4
    and the test_*config_file* twin of each (a Config file masks_version instead of the option), plus
    test_region_masks_differing_from_provenance_runs_the_enabled_stages
    test_region_map_on_unresolved_set_runs_the_enabled_stages
    test_config_file_map_beside_differing_masks_version_runs_the_enabled_stages
    and the test_*masks_version_beside_config_file_map* tests (--masks-version beside a Config file map), plus
    (user answers 2026-11-24 11:10) test_*conflicting_region*, test_*map_mismatch*, the dataset.allow_version_mismatch
    tests (Config file field, replay, --no-allow-version-mismatch) and the stage 4 --allow-version-mismatch tests

A Config file giving masks_version <V> counts as given, as --masks-version <V> does (--masks-version overrides it);
a Config file without a masks_version key gives none (REQ-MODELVER-BDRABCZUK-4F4D037D220A, user 2026-11-24 09:33).
A per-region Mask version map (--region-masks, REQ-MODELVER-BDRABCZUK-2D0C58CCD306, or held by a Config file,
EDF6D2E377E5) is used as is, so the unresolved stop does not apply to it (REQ-MODELVER-BDRABCZUK-63F4A3AA0589,
D3973A36209B, E504C9F416D9; user answer 2026-11-24 09:33); the mismatch stop does (user answer 2026-11-24 11:10).
--masks-version <V> beside a Config file holding a map is the exception (user decision 2026-11-24 10:50, CLI-first
rule): the command line wins, the Config file map is dropped, <V> applies to every region model name, the mismatch
rules above then apply, and one INFO record names --masks-version and the Config file path.

User answers 2026-11-24 11:10 (red of 20261124T101042):
- --allow-version-mismatch is also the Config file field dataset.allow_version_mismatch (default false): the option
  registry gives --allow-version-mismatch and --no-allow-version-mismatch, the replay file keeps the field (replaying
  an accepted run does not exit 2), and main.py passes --allow-version-mismatch to stage 4 when the setting is true;
- a given Mask version beside a region model name the resolver reports as conflicting is a mismatch (as stage 4,
  REQ-MODELVER-BDRABCZUK-D99E7930528B, EBCF851E3F94), whatever gives the Mask version;
- a per-region Mask version map differing from the local model set's own Mask versions follows the same mismatch rule
  (exit 2 with one ERROR record, or with --allow-version-mismatch one WARNING record per region model name). The map
  is still used as is otherwise; only a map PROVENANCE.csv names no Mask version for stops nothing.

With --allow-version-mismatch the explicit --masks-version wins: stages 2, 3 and 4 get --masks <V> for every region
(user decision 2026-11-24 09:28). That test runs main() with the orchestrator replaced by a recorder
(main_stage_helpers.main_runner) and reads each stage's argv from its build_args; nothing runs.

main() runs config-less on a tmp_path Morphologist input (stages 2 to 6) in a temporary working directory, with the
update check, the process setup and profiling stubbed. The real PipelineOrchestrator orders the stages; its
_run_stage is replaced by a recorder, so no stage executes. The local model set is a tmp_path folder named after no
Mask version, with one <region>/logs/ folder per region model name and a PROVENANCE.csv. No network, no model
weights, no real dataset.
"""

import dataclasses
import logging
import warnings
from pathlib import Path

import pytest
import yaml

from tests.main_stage_helpers import (
    enter_work_dir,
    load_main_module,
    main_runner,
    make_morphologist_input,
    option_values,
    ran_config,
    stub_main_setup,
)

pytestmark = pytest.mark.unit  # noqa: V107

MAIN = load_main_module("_champollion_main_masks_models_mismatch_under_test")

STAGES_2_TO_6 = [
    "run_cortical_tiles",
    "generate_champollion_config",
    "generate_embeddings",
    "put_together_embeddings",
    "generate_snapshots",
]
# Stage class name in cli.py -> stage name, for the stages that take --masks.
MASKS_STAGES = {
    "RunCorticalTilesStage": "run_cortical_tiles",
    "GenerateChampollionConfigStage": "generate_champollion_config",
    "GenerateEmbeddingsStage": "generate_embeddings",
}
C25 = "canonical_25"
C26 = "canonical_corrected_26_1"
R1 = "SC-sylv_left"
R2 = "SC-sylv_right"
MASKS_VERSION_OPTION = "--masks-version"
ALLOW_OPTION = "--allow-version-mismatch"
REGION_MASKS_OPTION = "--region-masks"
PROVENANCE_HEADER = "region,mask_version_trained_on\n"
# The Sulcal region of R1 and R2. The local model sets below hold 2 of the 56 region model names, so a run given no
# Mask version processes only this region (--regions, or the Config file's regions): a set lacking a model for a
# processed region model name is refused earlier by REQ-MODELVER-BDRABCZUK-B463E4170F9B / 4A87AEB9B7F5 (sibling
# TASK-MODELVER-BDRABCZUK-6C6C79B0F6DE), so the unresolved stop applies to a set covering the processed names only.
PROCESSED_REGION = "S.C.-sylv."
REGIONS_OPTION = "--regions"

# PROVENANCE.csv texts leaving R2 without one Mask version (R1 resolves to canonical_25 where a row names it).
UNRESOLVED_CASES = {
    "no_row_for_region": PROVENANCE_HEADER + f"{R1},{C25}\n",
    "conflicting_rows": PROVENANCE_HEADER + f"{R1},{C25}\n{R2},{C25}\n{R2},{C26}\n",
    "header_lacks_column": "region,source_run\n" + f"{R1},run1\n{R2},run2\n",
    "no_provenance_file": None,
}
# Both region model names trained on canonical_25.
ALL_C25_PROVENANCE = PROVENANCE_HEADER + f"{R1},{C25}\n{R2},{C25}\n"


@dataclasses.dataclass
class RunResult:
    """One main() run: its exit status, the stages it ran, and how many records were logged when the first ran."""

    status: int | str | None
    stages: list = dataclasses.field(default_factory=list)
    records_before_first_stage: int | None = None


def _model_set(tmp_path: Path, provenance: str | None, name: str = "models") -> Path:
    """A local model set tmp_path/<name> (named after no Mask version) with R1/logs, R2/logs and PROVENANCE.csv."""
    models = tmp_path / name
    for region in (R1, R2):
        (models / region / "logs").mkdir(parents=True)
    if provenance is not None:
        (models / "PROVENANCE.csv").write_text(provenance)
    return models


def _write_config_file(
    tmp_path: Path, models: Path, input_path: Path, masks_version: str | None, region_masks: list[str] | None = None
) -> Path:
    """A hand-written --config YAML on input_path and the local model set models.

    The dataset section holds masks_version only when one is given: without the key the Config file gives no Mask
    version (REQ-MODELVER-BDRABCZUK-4F4D037D220A). It holds a per-region Mask version map (region_masks, a list of
    NAME=VERSION entries, the sibling Task's convention) only when region_masks is given.
    """
    dataset = {"name": input_path.name, "input_path": str(input_path), "datasets_root": str(input_path)}
    if masks_version is not None:
        dataset["masks_version"] = masks_version
    if region_masks is not None:
        dataset["region_masks"] = region_masks
    values = {
        "models_path": str(models),
        "outputs_path": str(tmp_path / "out"),
        "log_dir": str(tmp_path / "logs"),
        "dataset": dataset,
    }
    path = tmp_path / f"config_{models.name}.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False))
    return path


def _set_regions(config_file: Path, regions: list[str]) -> Path:
    """Set dataset.regions (the Sulcal regions the run processes) to regions in the Config file; return config_file."""
    values = yaml.safe_load(config_file.read_text())
    values["dataset"]["regions"] = regions
    config_file.write_text(yaml.safe_dump(values, sort_keys=False))
    return config_file


def _set_allow_version_mismatch(config_file: Path, value: bool) -> Path:
    """Set dataset.allow_version_mismatch to value in the Config file config_file; return config_file."""
    values = yaml.safe_load(config_file.read_text())
    values["dataset"]["allow_version_mismatch"] = value
    config_file.write_text(yaml.safe_dump(values, sort_keys=False))
    return config_file


@pytest.fixture
def work_dir(monkeypatch, tmp_path) -> Path:
    return enter_work_dir(monkeypatch, tmp_path)


@pytest.fixture
def input_path(tmp_path) -> Path:
    return make_morphologist_input(tmp_path)


@pytest.fixture
def run_argv(monkeypatch, work_dir, pipeline_caplog):
    """Run main() with the given command-line arguments; the real orchestrator records its stages, runs none."""
    stub_main_setup(MAIN, monkeypatch)
    monkeypatch.setattr(MAIN, "start_profiling", lambda *a, **k: None)
    monkeypatch.setattr("champollion_utils.script_builder.start_profiling", lambda *a, **k: None)
    orchestrator = MAIN.PipelineOrchestrator

    def _run(*cli_args: str) -> RunResult:
        result = RunResult(status=None)

        def _run_stage(self, stage_name, *_lists):
            if result.records_before_first_stage is None:
                result.records_before_first_stage = len(pipeline_caplog.records)
            result.stages.append(stage_name)
            return True

        monkeypatch.setattr(orchestrator, "_setup_logger", lambda self: logging.getLogger("test_main_masks_mismatch"))
        monkeypatch.setattr(orchestrator, "_run_stage", _run_stage)
        monkeypatch.setattr("sys.argv", ["main.py", *cli_args])
        try:
            result.status = MAIN.main()
        except SystemExit as exc:
            result.status = exc.code
        return result

    return _run


@pytest.fixture
def run_main(run_argv, input_path):
    """Run main() config-less on a Morphologist input with --models-path <models> and the given arguments."""

    def _run(models: Path, *cli_args: str) -> RunResult:
        return run_argv(str(input_path), "--models-path", str(models), *cli_args)

    return _run


@pytest.fixture
def run_config(run_argv, tmp_path, input_path):
    """Run main() with --config on a Config file over a Morphologist input and models, then the given arguments."""

    def _run(
        models: Path,
        masks_version: str | None,
        *cli_args: str,
        region_masks: list[str] | None = None,
        allow_version_mismatch: bool | None = None,
        regions: list[str] | None = None,
    ) -> RunResult:
        config_file = _write_config_file(tmp_path, models, input_path, masks_version, region_masks)
        if regions is not None:
            _set_regions(config_file, regions)
        if allow_version_mismatch is not None:
            _set_allow_version_mismatch(config_file, allow_version_mismatch)
        return run_argv("--config", str(config_file), *cli_args)

    return _run


def _messages(pipeline_caplog, level: int, records=None) -> list[str]:
    chosen = pipeline_caplog.records if records is None else records
    return [record.getMessage() for record in chosen if record.levelno == level]


def _naming(messages: list[str], *words: str) -> list[str]:
    return [message for message in messages if all(word in message for word in words)]


# ---- no --masks-version, a region model name without one Mask version ----


@pytest.mark.parametrize("provenance", list(UNRESOLVED_CASES.values()), ids=list(UNRESOLVED_CASES))
def test_unresolved_region_without_masks_version_exits_2_before_any_stage(run_main, tmp_path, provenance):
    result = run_main(_model_set(tmp_path, provenance), REGIONS_OPTION, PROCESSED_REGION)
    assert result.stages == [], f"stages ran though a region model name has no single Mask version: {result.stages}"
    assert result.status == 2, f"main.py exit status {result.status}, expected 2"


@pytest.mark.parametrize("provenance", list(UNRESOLVED_CASES.values()), ids=list(UNRESOLVED_CASES))
def test_unresolved_region_without_masks_version_logs_one_error_record(run_main, tmp_path, pipeline_caplog, provenance):
    models = _model_set(tmp_path, provenance)
    run_main(models, REGIONS_OPTION, PROCESSED_REGION)
    errors = _messages(pipeline_caplog, logging.ERROR)
    stop_records = _naming(errors, str(models), MASKS_VERSION_OPTION)
    assert len(stop_records) == 1, (
        f"expected exactly one ERROR record naming {models} and {MASKS_VERSION_OPTION}; ERROR records: {errors}"
    )


# ---- --masks-version <V> differing from the resolved Mask version, without --allow-version-mismatch ----


def test_masks_version_mismatch_exits_2_before_any_stage(run_main, tmp_path):
    result = run_main(_model_set(tmp_path, ALL_C25_PROVENANCE), MASKS_VERSION_OPTION, C26)
    assert result.stages == [], f"stages ran though --masks-version {C26} differs from {C25}: {result.stages}"
    assert result.status == 2, f"main.py exit status {result.status}, expected 2"


def test_masks_version_mismatch_logs_one_error_record_naming_each_region(run_main, tmp_path, pipeline_caplog):
    run_main(_model_set(tmp_path, ALL_C25_PROVENANCE), MASKS_VERSION_OPTION, C26)
    errors = _messages(pipeline_caplog, logging.ERROR)
    stop_records = _naming(errors, C26, R1, R2, C25)
    assert len(stop_records) == 1, (
        f"expected exactly one ERROR record naming {C26}, {R1}, {R2} and {C25}; ERROR records: {errors}"
    )


# ---- --masks-version <V> differing from the resolved Mask version, with --allow-version-mismatch ----


def test_allowed_masks_version_mismatch_logs_one_warning_record_per_region(run_main, tmp_path, pipeline_caplog):
    result = run_main(_model_set(tmp_path, ALL_C25_PROVENANCE), MASKS_VERSION_OPTION, C26, ALLOW_OPTION)
    assert result.status == 0, f"main.py refused or failed the accepted mismatch run: exit status {result.status}"
    before = pipeline_caplog.records[: result.records_before_first_stage]
    warnings_before = _messages(pipeline_caplog, logging.WARNING, before)
    for region in (R1, R2):
        named = _naming(warnings_before, region, C26, C25)
        assert len(named) == 1, (
            f"expected exactly one WARNING record naming {region}, {C26} and {C25} before the first stage; "
            f"WARNING records then: {warnings_before}"
        )


def test_allowed_masks_version_mismatch_runs_the_enabled_stages(run_main, tmp_path):
    result = run_main(_model_set(tmp_path, ALL_C25_PROVENANCE), MASKS_VERSION_OPTION, C26, ALLOW_OPTION)
    assert result.status == 0, f"main.py refused or failed the accepted mismatch run: exit status {result.status}"
    assert result.stages == STAGES_2_TO_6


def test_allowed_masks_version_mismatch_emits_no_user_warning(run_main, tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_main(_model_set(tmp_path, ALL_C25_PROVENANCE), MASKS_VERSION_OPTION, C26, ALLOW_OPTION)
    assert result.status == 0, f"main.py refused or failed the accepted mismatch run: exit status {result.status}"
    user_warnings = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings == [], f"Python UserWarning emitted: {user_warnings}"


@pytest.mark.parametrize("stage_class_name", list(MASKS_STAGES))
def test_allowed_masks_version_mismatch_passes_masks_version_to_stages_2_to_4(
    monkeypatch, work_dir, tmp_path, stage_class_name
):
    """The explicit --masks-version wins: stages 2, 3 and 4 get --masks <V> for every region (user 2026-11-24 09:28)."""
    models = _model_set(tmp_path, ALL_C25_PROVENANCE)
    cli_args = (str(make_morphologist_input(tmp_path)), "--models-path", str(models), MASKS_VERSION_OPTION, C26)
    cli_args += (ALLOW_OPTION,)
    config = ran_config(main_runner(MAIN, monkeypatch)(*cli_args), cli_args)
    stage = getattr(MAIN, stage_class_name)(MASKS_STAGES[stage_class_name], config, logging.getLogger(__name__))
    argv = list(stage.build_args())
    assert option_values(argv, "--masks") == [C26], f"{stage_class_name} argv {argv!r}"


# ---- a Config file giving masks_version <V> counts as given (REQ-MODELVER-BDRABCZUK-4F4D037D220A) ----

C26_PROVENANCE = PROVENANCE_HEADER + f"{R1},{C26}\n{R2},{C26}\n"


@pytest.mark.parametrize("provenance", list(UNRESOLVED_CASES.values()), ids=list(UNRESOLVED_CASES))
def test_unresolved_region_with_config_file_lacking_masks_version_exits_2_before_any_stage(
    run_config, tmp_path, provenance
):
    result = run_config(_model_set(tmp_path, provenance), None, regions=[PROCESSED_REGION])
    assert result.stages == [], f"stages ran though a region model name has no single Mask version: {result.stages}"
    assert result.status == 2, f"main.py exit status {result.status}, expected 2"


@pytest.mark.parametrize("provenance", list(UNRESOLVED_CASES.values()), ids=list(UNRESOLVED_CASES))
def test_unresolved_region_with_config_file_lacking_masks_version_logs_one_error_record(
    run_config, tmp_path, pipeline_caplog, provenance
):
    models = _model_set(tmp_path, provenance)
    run_config(models, None, regions=[PROCESSED_REGION])
    errors = _messages(pipeline_caplog, logging.ERROR)
    stop_records = _naming(errors, str(models), MASKS_VERSION_OPTION)
    assert len(stop_records) == 1, (
        f"expected exactly one ERROR record naming {models} and {MASKS_VERSION_OPTION}; ERROR records: {errors}"
    )


def test_config_file_masks_version_mismatch_exits_2_before_any_stage(run_config, tmp_path):
    result = run_config(_model_set(tmp_path, ALL_C25_PROVENANCE), C26)
    assert result.stages == [], f"stages ran though the Config file's {C26} differs from {C25}: {result.stages}"
    assert result.status == 2, f"main.py exit status {result.status}, expected 2"


def test_config_file_masks_version_mismatch_logs_one_error_record_naming_each_region(
    run_config, tmp_path, pipeline_caplog
):
    run_config(_model_set(tmp_path, ALL_C25_PROVENANCE), C26)
    errors = _messages(pipeline_caplog, logging.ERROR)
    stop_records = _naming(errors, C26, R1, R2, C25)
    assert len(stop_records) == 1, (
        f"expected exactly one ERROR record naming {C26}, {R1}, {R2} and {C25}; ERROR records: {errors}"
    )


def test_allowed_config_file_masks_version_mismatch_logs_one_warning_record_per_region(
    run_config, tmp_path, pipeline_caplog
):
    result = run_config(_model_set(tmp_path, ALL_C25_PROVENANCE), C26, ALLOW_OPTION)
    assert result.status == 0, f"main.py refused or failed the accepted mismatch run: exit status {result.status}"
    before = pipeline_caplog.records[: result.records_before_first_stage]
    warnings_before = _messages(pipeline_caplog, logging.WARNING, before)
    for region in (R1, R2):
        named = _naming(warnings_before, region, C26, C25)
        assert len(named) == 1, (
            f"expected exactly one WARNING record naming {region}, {C26} and {C25} before the first stage; "
            f"WARNING records then: {warnings_before}"
        )


def test_allowed_config_file_masks_version_mismatch_runs_the_enabled_stages(run_config, tmp_path):
    """The accepted mismatch run runs the stages the same Config file runs on a set trained on its masks_version."""
    matching = run_config(_model_set(tmp_path, C26_PROVENANCE, name="models_c26"), C26)
    assert matching.status == 0 and matching.stages, f"the matching Config file run did not run: {matching}"
    result = run_config(_model_set(tmp_path, ALL_C25_PROVENANCE), C26, ALLOW_OPTION)
    assert result.status == 0, f"main.py refused or failed the accepted mismatch run: exit status {result.status}"
    assert result.stages == matching.stages


def test_allowed_config_file_masks_version_mismatch_emits_no_user_warning(run_config, tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_config(_model_set(tmp_path, ALL_C25_PROVENANCE), C26, ALLOW_OPTION)
    assert result.status == 0, f"main.py refused or failed the accepted mismatch run: exit status {result.status}"
    user_warnings = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings == [], f"Python UserWarning emitted: {user_warnings}"


@pytest.mark.parametrize("stage_class_name", list(MASKS_STAGES))
def test_allowed_config_file_masks_version_mismatch_passes_masks_version_to_stages_2_to_4(
    monkeypatch, work_dir, tmp_path, input_path, stage_class_name
):
    """The given masks_version wins: stages 2, 3 and 4 get --masks <V> for every region (user 2026-11-24 09:28)."""
    config_file = _write_config_file(tmp_path, _model_set(tmp_path, ALL_C25_PROVENANCE), input_path, C26)
    cli_args = ("--config", str(config_file), ALLOW_OPTION)
    config = ran_config(main_runner(MAIN, monkeypatch)(*cli_args), cli_args)
    stage = getattr(MAIN, stage_class_name)(MASKS_STAGES[stage_class_name], config, logging.getLogger(__name__))
    argv = list(stage.build_args())
    assert option_values(argv, "--masks") == [C26], f"{stage_class_name} argv {argv!r}"


# ---- an accepted --region-masks map mismatch runs (REQ-MODELVER-BDRABCZUK-63F4A3AA0589, user 2026-11-24 11:10) ----


def test_region_masks_differing_from_provenance_runs_the_enabled_stages(run_main, tmp_path):
    """A map differing from PROVENANCE.csv is a mismatch (user 2026-11-24 11:10): --allow-version-mismatch runs it."""
    models = _model_set(tmp_path, ALL_C25_PROVENANCE)
    result = run_main(models, ALLOW_OPTION, REGION_MASKS_OPTION, f"{R1}={C26}", f"{R2}={C26}")
    assert result.status == 0, f"main.py exit status {result.status} for a --region-masks run, expected 0"
    assert result.stages == STAGES_2_TO_6


# ---- a per-region Mask version map given by --region-masks or a Config file: no stop (D3973A36209B, E504C9F416D9) ----

# The unresolved cases where a PROVENANCE.csv exists and names no Mask version for R2 (a PROVENANCE.csv naming two
# is a mismatch beside a map since the user answer of 2026-11-24 11:10, REQ-MODELVER-BDRABCZUK-F4A7EE228B1A head).
UNRESOLVED_PROVENANCE_FILES = {
    name: text for name, text in UNRESOLVED_CASES.items() if text is not None and name != "conflicting_rows"
}
MAP_ENTRIES = [f"{R1}={C25}", f"{R2}={C26}"]


def _matching_config_stages(run_config, tmp_path) -> list:
    """The stages the same Config file runs on a set trained on its masks_version (no map, no mismatch)."""
    matching = run_config(_model_set(tmp_path, C26_PROVENANCE, name="models_c26"), C26)
    assert matching.status == 0 and matching.stages, f"the matching Config file run did not run: {matching}"
    return matching.stages


@pytest.mark.parametrize("source", ["option", "config_file"])
@pytest.mark.parametrize(
    "provenance", list(UNRESOLVED_PROVENANCE_FILES.values()), ids=list(UNRESOLVED_PROVENANCE_FILES)
)
def test_region_map_on_unresolved_set_runs_the_enabled_stages(run_main, run_config, tmp_path, provenance, source):
    """A given map is used as is (user 2026-11-24 09:33): PROVENANCE.csv naming no Mask version for R2 stops nothing."""
    models = _model_set(tmp_path, provenance)
    if source == "option":
        result = run_main(models, REGION_MASKS_OPTION, *MAP_ENTRIES)
        expected = STAGES_2_TO_6
    else:
        expected = _matching_config_stages(run_config, tmp_path)
        result = run_config(models, None, region_masks=MAP_ENTRIES)
    assert result.status == 0, f"main.py exit status {result.status} for a {source} map run, expected 0"
    assert result.stages == expected


def test_config_file_map_beside_differing_masks_version_runs_the_enabled_stages(run_config, tmp_path):
    """A Config file holding the map is used as recorded, whatever masks_version it also holds (EDF6D2E377E5)."""
    expected = _matching_config_stages(run_config, tmp_path)
    models = _model_set(tmp_path, ALL_C25_PROVENANCE)
    result = run_config(models, C26, region_masks=[f"{R1}={C25}", f"{R2}={C25}"])
    assert result.status == 0, f"main.py exit status {result.status} for a Config file map run, expected 0"
    assert result.stages == expected


# ---- --masks-version <V> beside a Config file's per-region Mask version map: the command line wins ----
# User decision 2026-11-24 10:50 (CLI-first rule): the Config file map is dropped, <V> applies to every region model
# name, the mismatch rules above then apply, and one INFO record says the map was overridden.

# A Config file map naming the set's own Mask version: only the dropped map would let a --masks-version C26 run go on.
MATCHING_MAP_ENTRIES = [f"{R1}={C25}", f"{R2}={C25}"]
# A Config file map naming another Mask version than the set and the option: the option C25 matches the set.
OTHER_MAP_ENTRIES = [f"{R1}={C26}", f"{R2}={C26}"]


def _stage_argv_beside_config_map(monkeypatch, tmp_path, input_path, stage_class_name, map_entries, *cli_args):
    """argv of one stage of a run on a Config file holding map_entries (no masks_version) with the given options."""
    models = _model_set(tmp_path, ALL_C25_PROVENANCE)
    config_file = _write_config_file(tmp_path, models, input_path, None, map_entries)
    full_args = ("--config", str(config_file), *cli_args)
    config = ran_config(main_runner(MAIN, monkeypatch)(*full_args), full_args)
    stage = getattr(MAIN, stage_class_name)(MASKS_STAGES[stage_class_name], config, logging.getLogger(__name__))
    return list(stage.build_args())


@pytest.mark.parametrize("stage_class_name", list(MASKS_STAGES))
def test_masks_version_beside_config_file_map_passes_masks_version_to_stages_2_to_4(
    monkeypatch, work_dir, tmp_path, input_path, stage_class_name
):
    """--masks-version C25 on a set trained on C25 beside a Config file map naming C26: --masks C25, no map."""
    argv = _stage_argv_beside_config_map(
        monkeypatch, tmp_path, input_path, stage_class_name, OTHER_MAP_ENTRIES, MASKS_VERSION_OPTION, C25
    )
    assert option_values(argv, "--masks") == [C25], f"{stage_class_name} argv {argv!r}"
    assert option_values(argv, REGION_MASKS_OPTION) == [], f"{stage_class_name} got the dropped map: {argv!r}"


def test_masks_version_beside_config_file_map_mismatch_exits_2_before_any_stage(run_config, tmp_path):
    result = run_config(
        _model_set(tmp_path, ALL_C25_PROVENANCE), None, MASKS_VERSION_OPTION, C26, region_masks=MATCHING_MAP_ENTRIES
    )
    assert result.stages == [], f"stages ran though --masks-version {C26} differs from {C25}: {result.stages}"
    assert result.status == 2, f"main.py exit status {result.status}, expected 2"


def test_masks_version_beside_config_file_map_mismatch_logs_one_error_record_naming_each_region(
    run_config, tmp_path, pipeline_caplog
):
    run_config(
        _model_set(tmp_path, ALL_C25_PROVENANCE), None, MASKS_VERSION_OPTION, C26, region_masks=MATCHING_MAP_ENTRIES
    )
    errors = _messages(pipeline_caplog, logging.ERROR)
    stop_records = _naming(errors, C26, R1, R2, C25)
    assert len(stop_records) == 1, (
        f"expected exactly one ERROR record naming {C26}, {R1}, {R2} and {C25}; ERROR records: {errors}"
    )


def test_allowed_masks_version_beside_config_file_map_mismatch_logs_one_warning_record_per_region(
    run_config, tmp_path, pipeline_caplog
):
    models = _model_set(tmp_path, ALL_C25_PROVENANCE)
    result = run_config(models, None, MASKS_VERSION_OPTION, C26, ALLOW_OPTION, region_masks=MATCHING_MAP_ENTRIES)
    assert result.status == 0, f"main.py refused or failed the accepted mismatch run: exit status {result.status}"
    before = pipeline_caplog.records[: result.records_before_first_stage]
    warnings_before = _messages(pipeline_caplog, logging.WARNING, before)
    for region in (R1, R2):
        named = _naming(warnings_before, region, C26, C25)
        assert len(named) == 1, (
            f"expected exactly one WARNING record naming {region}, {C26} and {C25} before the first stage; "
            f"WARNING records then: {warnings_before}"
        )


@pytest.mark.parametrize("stage_class_name", list(MASKS_STAGES))
def test_allowed_masks_version_beside_config_file_map_mismatch_passes_masks_version_to_stages_2_to_4(
    monkeypatch, work_dir, tmp_path, input_path, stage_class_name
):
    """The command line wins over the Config file map: stages 2, 3 and 4 get --masks C26 and no map."""
    options = (MASKS_VERSION_OPTION, C26, ALLOW_OPTION)
    argv = _stage_argv_beside_config_map(
        monkeypatch, tmp_path, input_path, stage_class_name, MATCHING_MAP_ENTRIES, *options
    )
    assert option_values(argv, "--masks") == [C26], f"{stage_class_name} argv {argv!r}"
    assert option_values(argv, REGION_MASKS_OPTION) == [], f"{stage_class_name} got the dropped map: {argv!r}"


# (--masks-version value, Config file map): a run going on, and a run stopped on a mismatch.
OVERRIDE_CASES = {
    "run_goes_on": (C25, OTHER_MAP_ENTRIES),
    "run_stopped_on_mismatch": (C26, MATCHING_MAP_ENTRIES),
}


@pytest.mark.parametrize("override_case", list(OVERRIDE_CASES.values()), ids=list(OVERRIDE_CASES))
def test_masks_version_beside_config_file_map_logs_one_info_record_naming_the_override(
    run_argv, tmp_path, input_path, pipeline_caplog, override_case
):
    masks_version, map_entries = override_case
    models = _model_set(tmp_path, ALL_C25_PROVENANCE)
    config_file = _write_config_file(tmp_path, models, input_path, None, map_entries)
    run_argv("--config", str(config_file), MASKS_VERSION_OPTION, masks_version)
    infos = _messages(pipeline_caplog, logging.INFO)
    override_records = _naming(infos, MASKS_VERSION_OPTION, str(config_file))
    assert len(override_records) == 1, (
        f"expected exactly one INFO record naming {MASKS_VERSION_OPTION} and {config_file}; INFO records: {infos}"
    )


# ---- a given Mask version beside a conflicting region model name is a mismatch (user 2026-11-24 11:10, answer 2) ----

# R2 has two PROVENANCE.csv rows, canonical_25 and canonical_corrected_26_1; R1 has one, canonical_25.
CONFLICT_PROVENANCE = PROVENANCE_HEADER + f"{R1},{C25}\n{R2},{C25}\n{R2},{C26}\n"
# What gives the Mask version C25 for R1 and R2.
GIVEN_SOURCES = ("masks_version_option", "config_file_masks_version", "region_masks_option", "config_file_map")
# What gives the per-region Mask version map.
MAP_SOURCES = ("region_masks_option", "config_file_map")


def _run_given(run_main, run_config, models: Path, source: str, version: str, *cli_args: str) -> RunResult:
    """Run main() on models given Mask version version for R1 and R2 by source, with the extra cli_args."""
    entries = [f"{R1}={version}", f"{R2}={version}"]
    if source == "masks_version_option":
        return run_main(models, *cli_args, MASKS_VERSION_OPTION, version)
    if source == "config_file_masks_version":
        return run_config(models, version, *cli_args)
    if source == "region_masks_option":
        return run_main(models, *cli_args, REGION_MASKS_OPTION, *entries)
    return run_config(models, None, *cli_args, region_masks=entries)


def _assert_command_line_accepted(capsys) -> None:
    """The exit 2 under test is the mismatch stop, not argparse refusing an option the run gives."""
    err = capsys.readouterr().err
    assert "unrecognized arguments" not in err, f"argparse refused the command line: {err}"


def _expected_stages(run_config, tmp_path, source: str) -> list:
    """The stages a run of source goes on with: 2 to 6 config-less, those of a matching Config file run otherwise."""
    return _matching_config_stages(run_config, tmp_path) if source.startswith("config_file") else STAGES_2_TO_6


@pytest.mark.parametrize("source", GIVEN_SOURCES)
def test_conflicting_region_with_given_mask_version_exits_2_before_any_stage(
    run_main, run_config, tmp_path, capsys, source
):
    result = _run_given(run_main, run_config, _model_set(tmp_path, CONFLICT_PROVENANCE), source, C25)
    _assert_command_line_accepted(capsys)
    assert result.stages == [], f"stages ran though {R2} conflicts between {C25} and {C26}: {result.stages}"
    assert result.status == 2, f"main.py exit status {result.status}, expected 2"


@pytest.mark.parametrize("source", GIVEN_SOURCES)
def test_conflicting_region_with_given_mask_version_logs_one_error_record(
    run_main, run_config, tmp_path, pipeline_caplog, source
):
    _run_given(run_main, run_config, _model_set(tmp_path, CONFLICT_PROVENANCE), source, C25)
    errors = _messages(pipeline_caplog, logging.ERROR)
    stop_records = _naming(errors, R2, C25, C26)
    assert len(stop_records) == 1, (
        f"expected exactly one ERROR record naming {R2}, {C25} and {C26}; ERROR records: {errors}"
    )


@pytest.mark.parametrize("source", GIVEN_SOURCES)
def test_allowed_conflicting_region_logs_one_warning_record(run_main, run_config, tmp_path, pipeline_caplog, source):
    models = _model_set(tmp_path, CONFLICT_PROVENANCE)
    result = _run_given(run_main, run_config, models, source, C25, ALLOW_OPTION)
    assert result.status == 0, f"main.py refused or failed the accepted conflict run: exit status {result.status}"
    before = pipeline_caplog.records[: result.records_before_first_stage]
    warnings_before = _messages(pipeline_caplog, logging.WARNING, before)
    named = _naming(warnings_before, R2, C25, C26)
    assert len(named) == 1, (
        f"expected exactly one WARNING record naming {R2}, {C25} and {C26} before the first stage; "
        f"WARNING records then: {warnings_before}"
    )


@pytest.mark.parametrize("source", GIVEN_SOURCES)
def test_allowed_conflicting_region_runs_the_enabled_stages(run_main, run_config, tmp_path, pipeline_caplog, source):
    expected = _expected_stages(run_config, tmp_path, source)
    models = _model_set(tmp_path, CONFLICT_PROVENANCE)
    result = _run_given(run_main, run_config, models, source, C25, ALLOW_OPTION)
    assert result.status == 0, f"main.py refused or failed the accepted conflict run: exit status {result.status}"
    assert result.stages == expected
    before = pipeline_caplog.records[: result.records_before_first_stage]
    accepted = _naming(_messages(pipeline_caplog, logging.WARNING, before), R2, C26)
    assert accepted, f"the run went on without accepting the {R2} conflict: no WARNING record naming {R2} and {C26}"


# ---- a per-region Mask version map differing from the set is a mismatch (user 2026-11-24 11:10, answer 3) ----


@pytest.mark.parametrize("source", MAP_SOURCES)
def test_map_mismatch_exits_2_before_any_stage(run_main, run_config, tmp_path, capsys, source):
    result = _run_given(run_main, run_config, _model_set(tmp_path, ALL_C25_PROVENANCE), source, C26)
    _assert_command_line_accepted(capsys)
    assert result.stages == [], f"stages ran though the map's {C26} differs from {C25}: {result.stages}"
    assert result.status == 2, f"main.py exit status {result.status}, expected 2"


@pytest.mark.parametrize("source", MAP_SOURCES)
def test_map_mismatch_logs_one_error_record_naming_each_region(run_main, run_config, tmp_path, pipeline_caplog, source):
    _run_given(run_main, run_config, _model_set(tmp_path, ALL_C25_PROVENANCE), source, C26)
    errors = _messages(pipeline_caplog, logging.ERROR)
    stop_records = _naming(errors, C26, R1, R2, C25)
    assert len(stop_records) == 1, (
        f"expected exactly one ERROR record naming {C26}, {R1}, {R2} and {C25}; ERROR records: {errors}"
    )


@pytest.mark.parametrize("source", MAP_SOURCES)
def test_allowed_map_mismatch_logs_one_warning_record_per_region(
    run_main, run_config, tmp_path, pipeline_caplog, source
):
    result = _run_given(run_main, run_config, _model_set(tmp_path, ALL_C25_PROVENANCE), source, C26, ALLOW_OPTION)
    assert result.status == 0, f"main.py refused or failed the accepted map mismatch run: exit status {result.status}"
    before = pipeline_caplog.records[: result.records_before_first_stage]
    warnings_before = _messages(pipeline_caplog, logging.WARNING, before)
    for region in (R1, R2):
        named = _naming(warnings_before, region, C26, C25)
        assert len(named) == 1, (
            f"expected exactly one WARNING record naming {region}, {C26} and {C25} before the first stage; "
            f"WARNING records then: {warnings_before}"
        )


# ---- dataset.allow_version_mismatch: Config file field, replay, stage 4 (user 2026-11-24 11:10, answer 1) ----

ALLOW_FLAG_STAGE_4 = "--allow-version-mismatch"
NO_ALLOW_OPTION = "--no-allow-version-mismatch"


def test_replay_of_allowed_mismatch_run_runs_the_enabled_stages(monkeypatch, run_main, run_argv, tmp_path):
    written: list[str] = []
    write_replay_config = MAIN.write_replay_config

    def _recording_write(config) -> str:
        written.append(write_replay_config(config))
        return written[-1]

    monkeypatch.setattr(MAIN, "write_replay_config", _recording_write)
    models = _model_set(tmp_path, ALL_C25_PROVENANCE)
    first = run_main(models, MASKS_VERSION_OPTION, C26, ALLOW_OPTION)
    assert first.status == 0 and first.stages, f"the accepted mismatch run did not run: {first}"
    assert len(written) == 1, f"expected one replay file written by the accepted run, got {written}"
    replay_file = Path(written[0]).rename(tmp_path / "replay_of_allowed_run.yaml")
    replay = run_argv("--config", str(replay_file))
    assert replay.status == 0, f"main.py exit status {replay.status} replaying {replay_file}, expected 0"
    assert replay.stages == first.stages


def test_config_file_allowing_mismatch_runs_the_enabled_stages(run_config, tmp_path, pipeline_caplog):
    expected = _matching_config_stages(run_config, tmp_path)
    result = run_config(_model_set(tmp_path, ALL_C25_PROVENANCE), C26, allow_version_mismatch=True)
    assert result.status == 0, f"main.py refused or failed the allowed mismatch run: exit status {result.status}"
    assert result.stages == expected
    before = pipeline_caplog.records[: result.records_before_first_stage]
    accepted = _naming(_messages(pipeline_caplog, logging.WARNING, before), R1, C26, C25)
    assert accepted, f"the run went on without accepting the mismatch: no WARNING naming {R1}, {C26} and {C25}"


def test_no_allow_option_beside_config_file_allowing_mismatch_exits_2(run_config, tmp_path, pipeline_caplog):
    result = run_config(_model_set(tmp_path, ALL_C25_PROVENANCE), C26, NO_ALLOW_OPTION, allow_version_mismatch=True)
    assert result.stages == [], f"stages ran though {NO_ALLOW_OPTION} was given: {result.stages}"
    assert result.status == 2, f"main.py exit status {result.status}, expected 2"
    errors = _messages(pipeline_caplog, logging.ERROR)
    stop_records = _naming(errors, C26, R1, R2, C25)
    assert len(stop_records) == 1, (
        f"expected the mismatch stop: exactly one ERROR record naming {C26}, {R1}, {R2} and {C25}; ERROR: {errors}"
    )


def _stage_4_argv(monkeypatch, cli_args: tuple[str, ...]) -> tuple[list[str], object]:
    """(stage 4 argv, ran PipelineConfig) of a main() run with cli_args and the orchestrator replaced by a recorder.

    The callers give the set's own Mask version (canonical_25), so the partial local model set is not refused before
    the orchestrator (REQ-MODELVER-BDRABCZUK-B463E4170F9B applies only given no Mask version).
    """
    config = ran_config(main_runner(MAIN, monkeypatch)(*cli_args), cli_args)
    stage = MAIN.GenerateEmbeddingsStage("generate_embeddings", config, logging.getLogger(__name__))
    return list(stage.build_args()), config


@pytest.mark.parametrize("source", ["option", "config_file"])
def test_allow_setting_passes_allow_option_to_stage_4(monkeypatch, work_dir, tmp_path, input_path, source):
    models = _model_set(tmp_path, ALL_C25_PROVENANCE)
    if source == "option":
        cli_args = (str(input_path), "--models-path", str(models), MASKS_VERSION_OPTION, C25, ALLOW_OPTION)
    else:
        config_file = _set_allow_version_mismatch(_write_config_file(tmp_path, models, input_path, C25), True)
        cli_args = ("--config", str(config_file))
    argv, _ = _stage_4_argv(monkeypatch, cli_args)
    assert ALLOW_FLAG_STAGE_4 in argv, f"stage 4 argv lacks {ALLOW_FLAG_STAGE_4}: {argv!r}"


def test_without_allow_setting_stage_4_gets_no_allow_option(monkeypatch, work_dir, tmp_path, input_path):
    models = _model_set(tmp_path, ALL_C25_PROVENANCE)
    cli_args = (str(input_path), "--models-path", str(models), MASKS_VERSION_OPTION, C25)
    argv, config = _stage_4_argv(monkeypatch, cli_args)
    setting = getattr(config.dataset, "allow_version_mismatch", "absent")
    assert setting is False, f"dataset.allow_version_mismatch is {setting!r}, expected the default False"
    assert ALLOW_FLAG_STAGE_4 not in argv, f"stage 4 argv holds {ALLOW_FLAG_STAGE_4}: {argv!r}"
