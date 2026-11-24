"""champollion-pipeline resolves the per-region Mask version map of a local model set before stage 1 when
--masks-version is not given, and forwards it to stages 2, 3 and 4 (TASK-MODELVER-BDRABCZUK-6C6C79B0F6DE).

User decision 2026-11-17 (Defect DEF-MODELVER-BDRABCZUK-291232AA949D): a nominal run needs no global flag; a mixed
model set is used as shipped. The map is the one resolve_region_mask_versions returns for the local model set; the
stages get it as --region-masks NAME=VERSION ... and no --masks (the stages refuse --masks beside --region-masks).

Requirements pinned here:

* REQ-MODELVER-BDRABCZUK-1621C738EAB8 (system) - no Mask version given by option or Config file, a set resolving each
  region model name to one Mask version: stages 2, 3 and 4 run with the resolver's map.
* REQ-MODELVER-BDRABCZUK-A4B287DB9F0C (system) - a map run passes no --masks to stages 2, 3 and 4.
* REQ-MODELVER-BDRABCZUK-9DD11041D773 (system) - each stage's own parser accepts the arguments of a map run.
* REQ-MODELVER-BDRABCZUK-BCE6FBB7CDAF (system) - a uniform map keys the stage 4-6 folders by its Mask version.
* REQ-MODELVER-BDRABCZUK-1DFD052441D2 (system) - a mixed map puts the stage 4-6 folders under champollion_V1/L, L the
  map's Mask versions in MASK_VERSIONS order joined by "_and_" (user answer 2026-11-24 09:33).
* REQ-MODELVER-BDRABCZUK-4FBC179FC50A (system) - an explicit --masks-version V, with no region model name of the set
  resolved to another Mask version, gives stages 2, 3 and 4 --masks V.
* REQ-MODELVER-BDRABCZUK-B2B453643BD4 (system) - the replay file of a map run gives the stages the same map.
* REQ-MODELVER-BDRABCZUK-4F4D037D220A (system) - a Config file masks_version V without a map counts as given: --masks V.
* REQ-MODELVER-BDRABCZUK-196F6229C754 (system) - a Config file map (a replay file) is passed to the stages as recorded.
* REQ-MODELVER-BDRABCZUK-872193375DB3 (system) - a champollion-pipeline --region-masks map reaches the stages as is.
* REQ-MODELVER-BDRABCZUK-0DB34F4CBB0F (system) - --region-masks beside --masks-version: exit 2 before stage 1, the
  error naming both options.
* REQ-MODELVER-BDRABCZUK-9273939CFE5A (system) - a map entry that a stage's --region-masks option refuses, given by
  --region-masks or a Config file: exit 2 before any stage, exactly one ERROR record naming the value the option names
  (user answer 2026-11-24 10:27, Q2).
* REQ-MODELVER-BDRABCZUK-BFB170076CAB (system) - no Mask version given: the stage 5 and 6 folders do not depend on the
  enabled stages, so a stage 5 and 6 only run reads the folders of the map run (user answer 2026-11-24 10:27, Q3).
* Given mixed map on a Hugging Face stage 4 models source (user decision 2026-11-24 11:01): exit 2 before stage 1
  (TestMixedMapOnHuggingFaceExits2) and exactly one ERROR record naming a local model set
  (TestMixedMapOnHuggingFaceOneErrorRecord).
* Partial local model set (user decision 2026-11-24 11:27: partial-model-set support deferred past 1.0.0): a set
  holding no model for a region model name the run processes (the 56, or the two of each Sulcal region of --regions),
  one-hemisphere sets included, given no Mask version:
  - REQ-MODELVER-BDRABCZUK-B463E4170F9B (system) - exit 2 before stage 1 (TestPartialSetExits2); with --regions R
    only the models of R must exist (TestRegionsFilterNeedsOnlyItsModels).
  - REQ-MODELVER-BDRABCZUK-4A87AEB9B7F5 (system) - exactly one ERROR record naming those region model names and a
    full model set (TestPartialSetOneErrorRecord).

The map Requirements hold in sequential mode only (user answer 2026-11-24 10:27, Q1): every run here is sequential,
the default mode; streaming mode is TASK-MODELVER-BDRABCZUK-EDC9DF99633C.

Boundary with TASK-MODELVER-BDRABCZUK-3B2E4833B088 (tests/test_main_masks_models_mismatch.py): the exit-2 stops
(unresolved or conflicting region, explicit --masks-version differing from a region's version) are tested there; every
model set here but the partial ones resolves each region model name, and every explicit --masks-version here matches
the set. A partial set's region model names without a model take no part in those stops: its own stop is tested here.

The model sets are passed with an explicit --models-path, the first step of the stage 4 models source cascade
(TASK-MODELSRC-BDRABCZUK-EFFEE1006BFE): the tests hold with and without that cascade. No network, no model, no
BrainVISA: main() runs with the orchestrator replaced by a recorder, and each stage's argv comes from its build_args.
Synthetic subject IDs only, every folder under tmp_path.
"""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from champollion_pipeline.config import ConfigLoader
from champollion_pipeline.derivatives_layout import MASK_VERSIONS
from champollion_pipeline.generate_champollion_config import GenerateChampollionConfig
from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from champollion_pipeline.models.checkpoint import find_region_models
from champollion_pipeline.models.version_options import resolve_region_mask_versions
from champollion_pipeline.replay_config import logs_folder
from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from champollion_pipeline.utils.region_masks import (
    REGION_MASKS_OPTION,
    REGION_MODEL_NAMES,
    split_region_model_name,
)
from tests.main_stage_helpers import load_main_module, main_runner, make_t1_input, option_values, ran_config
from tests.test_models_source_resolver import make_checkout, refuse_hub

pytestmark = pytest.mark.unit  # noqa: V107

C25 = "canonical_25"
C26 = "canonical_corrected_26_1"
# Region model names the mixed set trains on canonical_25; every other one is canonical_corrected_26_1.
C25_NAMES = ("SC-sylv_left", "SC-sylv_right")
# Region model names of the partial local model set: both hemispheres of S.C.-sylv., the left one only of S.Or.
PARTIAL_NAMES = ("SC-sylv_left", "SC-sylv_right", "SOr_left")
PROVENANCE_HEADER = "region,mask_version_trained_on\n"
LOGGER_NAME = "test_main_masks_inferred_from_models"
# Separator of the run label of a mixed map (user answer 2026-11-24 09:33; "_and_" picked by red).
LABEL_SEPARATOR = "_and_"

# Stage class name in cli.py -> stage name.
MAP_STAGES = {
    "RunCorticalTilesStage": "run_cortical_tiles",
    "GenerateChampollionConfigStage": "generate_champollion_config",
    "GenerateEmbeddingsStage": "generate_embeddings",
}
FOLDER_STAGES = {
    "GenerateEmbeddingsStage": "generate_embeddings",
    "PutTogetherEmbeddingsStage": "put_together_embeddings",
    "GenerateSnapshotsStage": "generate_snapshots",
}
# (stage class name, option) of each stage 4-6 folder option; stage 5's source folder is its first argument.
FOLDER_OPTIONS = (
    ("GenerateEmbeddingsStage", "--output"),
    ("PutTogetherEmbeddingsStage", "--output_path"),
    ("GenerateSnapshotsStage", "--output_dir"),
    ("GenerateSnapshotsStage", "--embeddings_dir"),
)
STAGE_PARSERS = {
    "RunCorticalTilesStage": RunCorticalTiles,
    "GenerateChampollionConfigStage": GenerateChampollionConfig,
    "GenerateEmbeddingsStage": GenerateEmbeddings,
}


@pytest.fixture(scope="module")
def main_module():
    return load_main_module("_champollion_main_masks_inferred_from_models_under_test")


def _model_set(tmp_path: Path, folder: str, versions: dict[str, str]) -> Path:
    """Local model set tmp_path/<folder>: a model folder (with logs/) and a PROVENANCE.csv row per region model name
    of versions, so the set's models are exactly the names of versions."""
    models = tmp_path / folder
    models.mkdir()
    for name in versions:
        (models / name / "logs").mkdir(parents=True)
    rows = "".join(f"{name},{version}\n" for name, version in versions.items())
    (models / "PROVENANCE.csv").write_text(PROVENANCE_HEADER + rows)
    return models


def _mixed_versions() -> dict[str, str]:
    return {name: (C25 if name in C25_NAMES else C26) for name in REGION_MODEL_NAMES}


def _uniform_versions(version: str) -> dict[str, str]:
    return dict.fromkeys(REGION_MODEL_NAMES, version)


def _resolved_map(models: Path) -> dict[str, str]:
    """The per-region Mask version map the resolver returns for the model set (every entry resolved here)."""
    entries = resolve_region_mask_versions(str(models))
    assert all(entry.version is not None for entry in entries.values()), "fixture set leaves a name unresolved"
    return {name: entry.version for name, entry in entries.items()}


def _region_masks_of(argv: list[str]) -> dict[str, str] | None:
    """The NAME=VERSION pairs given to --region-masks in argv as a dict; None when the option is absent."""
    pairs: list[str] = []
    found = False
    for i, token in enumerate(argv):
        if token.startswith(REGION_MASKS_OPTION + "="):
            found = True
            pairs.append(token.split("=", 1)[1])
        elif token == REGION_MASKS_OPTION:
            found = True
            for value in argv[i + 1 :]:
                if value.startswith("--"):
                    break
                pairs.append(value)
    if not found:
        return None
    return dict(pair.split("=", 1) for pair in pairs)


def _stage_argv(main_module, config, stage_class_name: str) -> list[str]:
    """The argv the stage class of cli.py builds for config (nothing runs)."""
    names = {**MAP_STAGES, **FOLDER_STAGES}
    stage = getattr(main_module, stage_class_name)(names[stage_class_name], config, logging.getLogger(LOGGER_NAME))
    return list(stage.build_args())


def _run_main(main_module, monkeypatch, tmp_path: Path, models: Path, *extra: str):
    """Run main() config-less on a synthetic T1 input with --models-path models; return (config ran, input root)."""
    raw = make_t1_input(tmp_path)
    cli_args = (str(raw), "--models-path", str(models), *extra)
    config = ran_config(main_runner(main_module, monkeypatch)(*cli_args), cli_args)
    return config, raw


def _stage_folders(main_module, config) -> dict[str, str]:
    """The region embeddings, combined embeddings and snapshots folders the configuration gives stages 4 to 6."""
    argvs = {name: _stage_argv(main_module, config, name) for name in FOLDER_STAGES}
    folders = {f"{stage} {option}": option_values(argvs[stage], option) for stage, option in FOLDER_OPTIONS}
    folders["PutTogetherEmbeddingsStage source"] = argvs["PutTogetherEmbeddingsStage"][:1]
    for name, values in folders.items():
        assert len(values) == 1, f"{name}: {values!r}"
    return {name: values[0] for name, values in folders.items()}


def _run_label(versions: dict[str, str]) -> str:
    """The run label of a map: its distinct Mask versions in MASK_VERSIONS order joined by LABEL_SEPARATOR."""
    return LABEL_SEPARATOR.join(version for version in MASK_VERSIONS if version in set(versions.values()))


def _region_masks_tokens(versions: dict[str, str]) -> tuple[str, ...]:
    """--region-masks followed by one NAME=VERSION token per region model name."""
    return (REGION_MASKS_OPTION, *(f"{name}={version}" for name, version in versions.items()))


def _write_config_file(
    tmp_path: Path, models: Path | str, raw: Path, *, root_path: Path | None = None, **dataset: object
) -> Path:
    """A hand-written --config YAML on the T1 input raw and the local model set models ("": no models_path).

    The dataset section holds only the given keys beside name, input_path and datasets_root: without a masks_version
    key the YAML gives no Mask version (a YAML saved from a PipelineConfig always holds one). root_path, when given,
    replaces the checkout root (the models source cascade's <root_path>/models step).
    """
    values = {
        "models_path": str(models),
        "outputs_path": str(tmp_path / "out"),
        "log_dir": str(tmp_path / "logs"),
        "dataset": {"name": raw.name, "input_path": str(raw), "datasets_root": str(raw), **dataset},
    }
    if root_path is not None:
        values = {"root_path": str(root_path), **values}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False))
    return path


def _run_main_with_config(main_module, monkeypatch, config_file: Path, *extra: str):
    """Run main() with --config config_file and the extra arguments; return the configuration it ran."""
    cli_args = ("--config", str(config_file), *extra)
    return ran_config(main_runner(main_module, monkeypatch)(*cli_args), cli_args)


def _replay_of_map_run(main_module, monkeypatch, tmp_path: Path, models: Path) -> Path:
    """Run main() config-less on models, then rewrite its PROVENANCE.csv to canonical_25; return the replay file.

    After the rewrite a new resolution gives another map, so a map the stages get can only come from the replay file.
    """
    config, _ = _run_main(main_module, monkeypatch, tmp_path, models)
    replays = sorted(logs_folder(config).glob("replay_*.yaml"))
    assert len(replays) == 1, f"replay files in {logs_folder(config)}: {replays!r}"
    rows = "".join(f"{name},{C25}\n" for name in REGION_MODEL_NAMES)
    (models / "PROVENANCE.csv").write_text(PROVENANCE_HEADER + rows)
    return replays[0]


class TestMapForwardedToStages2To4:
    """REQ-MODELVER-BDRABCZUK-1621C738EAB8: stages 2, 3 and 4 get the resolver's map of a fully resolved local set."""

    @pytest.mark.parametrize("stage_class_name", list(MAP_STAGES))
    def test_stage_region_masks_is_resolved_map(self, main_module, monkeypatch, tmp_path, stage_class_name):
        models = _model_set(tmp_path, "mixed_set", _mixed_versions())
        config, _ = _run_main(main_module, monkeypatch, tmp_path, models)
        argv = _stage_argv(main_module, config, stage_class_name)
        assert _region_masks_of(argv) == _resolved_map(models), f"{stage_class_name} argv {argv!r}"


class TestMapRunPassesNoMasks:
    """REQ-MODELVER-BDRABCZUK-A4B287DB9F0C: a map run passes no --masks to stages 2, 3 and 4."""

    @pytest.mark.parametrize("stage_class_name", list(MAP_STAGES))
    def test_no_masks_option(self, main_module, monkeypatch, tmp_path, stage_class_name):
        models = _model_set(tmp_path, "mixed_set", _mixed_versions())
        config, _ = _run_main(main_module, monkeypatch, tmp_path, models)
        argv = _stage_argv(main_module, config, stage_class_name)
        assert option_values(argv, "--masks") == [], f"{stage_class_name} argv {argv!r}"


class TestMapRunArgvAccepted:
    """REQ-MODELVER-BDRABCZUK-9DD11041D773: each stage's own parser accepts the argv of a map run."""

    @pytest.mark.parametrize("stage_class_name", list(MAP_STAGES))
    def test_stage_parser_accepts_argv(self, main_module, monkeypatch, tmp_path, stage_class_name):
        models = _model_set(tmp_path, "mixed_set", _mixed_versions())
        config, raw = _run_main(main_module, monkeypatch, tmp_path, models)
        (raw / "derivatives" / "morphologist-6.0").mkdir(parents=True, exist_ok=True)
        argv = _stage_argv(main_module, config, stage_class_name)
        script = STAGE_PARSERS[stage_class_name]()
        with patch("champollion_utils.script_builder.check_for_updates", lambda *a, **k: None):
            try:
                script.parse_args(argv)
            except SystemExit as stop:
                pytest.fail(f"{stage_class_name} parser rejected {argv!r} (SystemExit {stop.code})")


class TestUniformSetFolders:
    """REQ-MODELVER-BDRABCZUK-BCE6FBB7CDAF: a set naming V for each region model name keys stage 4-6 folders by V."""

    def test_stage_4_to_6_folders_under_set_version(self, main_module, monkeypatch, tmp_path):
        models = _model_set(tmp_path, "uniform_set", _uniform_versions(C25))
        config, raw = _run_main(main_module, monkeypatch, tmp_path, models)
        tree = (raw / "derivatives" / "champollion_V1" / C25).resolve()
        for name, folder in _stage_folders(main_module, config).items():
            assert Path(folder).resolve().is_relative_to(tree), f"{name} {folder!r} not under {tree}"


class TestMixedSetFolders:
    """REQ-MODELVER-BDRABCZUK-1DFD052441D2: a mixed set puts stage 4-6 folders under one champollion_V1/L folder."""

    def test_stage_4_to_6_folders_under_one_label(self, main_module, monkeypatch, tmp_path):
        models = _model_set(tmp_path, "mixed_set", _mixed_versions())
        config, raw = _run_main(main_module, monkeypatch, tmp_path, models)
        tree = (raw / "derivatives" / "champollion_V1").resolve()
        labels = {}
        for name, folder in _stage_folders(main_module, config).items():
            path = Path(folder).resolve()
            assert path.is_relative_to(tree), f"{name} {folder!r} not under {tree}"
            labels[name] = path.relative_to(tree).parts[0]
        assert len(set(labels.values())) == 1, f"stage 4-6 folders under several labels: {labels!r}"


class TestExplicitMasksVersionOverrides:
    """REQ-MODELVER-BDRABCZUK-4FBC179FC50A: --masks-version V gives stages 2, 3 and 4 --masks V on a set naming only V.

    The set names V for each region model name, so no mismatch stop applies (TASK-MODELVER-BDRABCZUK-3B2E4833B088);
    V equal to the DatasetConfig default checks an explicit default is told from no option at all.
    """

    @pytest.mark.parametrize("version", [C25, C26])
    @pytest.mark.parametrize("stage_class_name", list(MAP_STAGES))
    def test_stage_masks_is_explicit_version(self, main_module, monkeypatch, tmp_path, stage_class_name, version):
        models = _model_set(tmp_path, "uniform_set", _uniform_versions(version))
        config, _ = _run_main(main_module, monkeypatch, tmp_path, models, "--masks-version", version)
        argv = _stage_argv(main_module, config, stage_class_name)
        assert option_values(argv, "--masks") == [version], f"{stage_class_name} argv {argv!r}"


class TestReplayKeepsMap:
    """REQ-MODELVER-BDRABCZUK-B2B453643BD4: the replay file of a map run gives stages 2-4 the map of the run.

    The set's PROVENANCE.csv is rewritten after the run, so the map can only come from the replay file.
    """

    @pytest.mark.parametrize("stage_class_name", list(MAP_STAGES))
    def test_replay_config_gives_run_map(self, main_module, monkeypatch, tmp_path, stage_class_name):
        models = _model_set(tmp_path, "mixed_set", _mixed_versions())
        expected = _resolved_map(models)
        replay = _replay_of_map_run(main_module, monkeypatch, tmp_path, models)
        argv = _stage_argv(main_module, ConfigLoader.load_from_yaml(str(replay)), stage_class_name)
        assert _region_masks_of(argv) == expected, f"{stage_class_name} replay argv {argv!r}"


class TestConfigFileWithoutMasksVersionRunsMap:
    """REQ-MODELVER-BDRABCZUK-1621C738EAB8: a Config file naming no masks_version and no map gives no Mask version.

    So a --config run on a fully resolved local set runs stages 2, 3 and 4 with the resolver's map, like a
    config-less run.
    """

    @pytest.mark.parametrize("stage_class_name", list(MAP_STAGES))
    def test_stage_region_masks_is_resolved_map(self, main_module, monkeypatch, tmp_path, stage_class_name):
        models = _model_set(tmp_path, "mixed_set", _mixed_versions())
        config_file = _write_config_file(tmp_path, models, make_t1_input(tmp_path))
        config = _run_main_with_config(main_module, monkeypatch, config_file)
        argv = _stage_argv(main_module, config, stage_class_name)
        assert _region_masks_of(argv) == _resolved_map(models), f"{stage_class_name} argv {argv!r}"


class TestMixedSetRunLabel:
    """REQ-MODELVER-BDRABCZUK-1DFD052441D2: a mixed map keys the stage 4-6 folders by its joined Mask versions.

    The set names canonical_25 for two region model names and canonical_corrected_26_1 for the others, so the label
    is canonical_25_and_canonical_corrected_26_1 (MASK_VERSIONS order, not the majority version).
    """

    def test_stage_4_to_6_folders_under_joined_label(self, main_module, monkeypatch, tmp_path):
        models = _model_set(tmp_path, "mixed_set", _mixed_versions())
        config, raw = _run_main(main_module, monkeypatch, tmp_path, models)
        label = _run_label(_resolved_map(models))
        assert label == f"{C25}{LABEL_SEPARATOR}{C26}", f"fixture map gives the label {label!r}"
        tree = (raw / "derivatives" / "champollion_V1" / label).resolve()
        for name, folder in _stage_folders(main_module, config).items():
            assert Path(folder).resolve().is_relative_to(tree), f"{name} {folder!r} not under {tree}"


class TestConfigFileMasksVersionGiven:
    """REQ-MODELVER-BDRABCZUK-4F4D037D220A: a Config file masks_version V without a map counts as given.

    The set names only V, so no mismatch stop applies (TASK-MODELVER-BDRABCZUK-3B2E4833B088); V equal to the
    DatasetConfig default checks a YAML value equal to the default still counts as given. This behaviour holds before
    the map work (it pins what the map resolution must keep).
    """

    @pytest.mark.parametrize("version", [C25, C26])
    @pytest.mark.parametrize("stage_class_name", list(MAP_STAGES))
    def test_stage_masks_is_config_version(self, main_module, monkeypatch, tmp_path, stage_class_name, version):
        models = _model_set(tmp_path, "uniform_set", _uniform_versions(version))
        config_file = _write_config_file(tmp_path, models, make_t1_input(tmp_path), masks_version=version)
        config = _run_main_with_config(main_module, monkeypatch, config_file)
        argv = _stage_argv(main_module, config, stage_class_name)
        assert option_values(argv, "--masks") == [version], f"{stage_class_name} argv {argv!r}"
        assert _region_masks_of(argv) is None, f"{stage_class_name} argv {argv!r}"


def _reversed_versions(versions: dict[str, str]) -> dict[str, str]:
    """versions with its region model names in reverse order (not the resolver's order)."""
    return dict(reversed(list(versions.items())))


class TestConfigFileMapUsedAsIs:
    """REQ-MODELVER-BDRABCZUK-196F6229C754: a Config file holding the map gives the stages that map, as recorded.

    The Config file is the replay file of a config-less map run on the mixed set, so it also holds a masks_version (the
    run label), with its region_masks list rewritten in reverse order: the map names the set's own Mask versions (no
    mismatch stop of REQ-MODELVER-BDRABCZUK-8F159BD7ED51, TASK-MODELVER-BDRABCZUK-3B2E4833B088), and the stages get
    its entries in the Config file's order, which a new resolution would not give.
    """

    @pytest.mark.parametrize("stage_class_name", list(MAP_STAGES))
    def test_config_file_map_reaches_stages(self, main_module, monkeypatch, tmp_path, stage_class_name):
        models = _model_set(tmp_path, "mixed_set", _mixed_versions())
        config, _ = _run_main(main_module, monkeypatch, tmp_path, models)
        replays = sorted(logs_folder(config).glob("replay_*.yaml"))
        assert len(replays) == 1, f"replay files in {logs_folder(config)}: {replays!r}"
        values = yaml.safe_load(replays[0].read_text())
        expected = list(reversed(values["dataset"]["region_masks"]))
        assert len(expected) == len(REGION_MODEL_NAMES), f"replay region_masks {expected!r}"
        values["dataset"]["region_masks"] = expected
        replays[0].write_text(yaml.safe_dump(values, sort_keys=False))
        config = _run_main_with_config(main_module, monkeypatch, replays[0])
        argv = _stage_argv(main_module, config, stage_class_name)
        assert _option_list(argv, REGION_MASKS_OPTION) == expected, f"{stage_class_name} argv {argv!r}"
        assert option_values(argv, "--masks") == [], f"{stage_class_name} argv {argv!r}"


class TestRegionMasksOptionUsedAsIs:
    """REQ-MODELVER-BDRABCZUK-872193375DB3: champollion-pipeline --region-masks M gives stages 2, 3 and 4 the map M.

    The given map is the mixed set's own map, so no mismatch stop of REQ-MODELVER-BDRABCZUK-8F159BD7ED51
    (TASK-MODELVER-BDRABCZUK-3B2E4833B088) applies, given in reverse order: the stages get its NAME=VERSION tokens in
    the given order, which the resolver's order is not, so the map comes from the option, used as is.
    """

    @pytest.mark.parametrize("stage_class_name", list(MAP_STAGES))
    def test_stage_region_masks_is_given_map(self, main_module, monkeypatch, tmp_path, stage_class_name):
        models = _model_set(tmp_path, "mixed_set", _mixed_versions())
        tokens = _region_masks_tokens(_reversed_versions(_resolved_map(models)))
        config, _ = _run_main(main_module, monkeypatch, tmp_path, models, *tokens)
        argv = _stage_argv(main_module, config, stage_class_name)
        assert _option_list(argv, REGION_MASKS_OPTION) == list(tokens[1:]), f"{stage_class_name} argv {argv!r}"


class TestRegionMasksBesideMasksVersionRefused:
    """REQ-MODELVER-BDRABCZUK-0DB34F4CBB0F: --region-masks beside --masks-version exits 2 before stage 1.

    The error message (the argparse error line or an ERROR record) names both options; the usage text, which lists
    every option, is not read.
    """

    def test_exit_2_naming_both_options(self, main_module, monkeypatch, tmp_path, capsys, caplog):
        models = _model_set(tmp_path, "uniform_set", _uniform_versions(C25))
        raw = make_t1_input(tmp_path)
        cli_args = (str(raw), "--models-path", str(models), "--masks-version", C25)
        cli_args += _region_masks_tokens(_uniform_versions(C25))
        with caplog.at_level(logging.ERROR):
            outcome = main_runner(main_module, monkeypatch)(*cli_args)
        assert outcome.status == 2, f"main() exit status {outcome.status!r}"
        assert outcome.configs == [], "main() handed a configuration to the orchestrator"
        captured = capsys.readouterr()
        messages = [line.split("error:", 1)[1] for line in captured.err.splitlines() if "error:" in line]
        messages += [line for line in (captured.out + captured.err).splitlines() if "ERROR" in line]
        messages += [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR]
        assert any(REGION_MASKS_OPTION in text and "--masks-version" in text for text in messages), (
            f"no error message names both options: {messages!r}"
        )


# (map entries, the value the --region-masks option of a stage names on refusing them): no equals sign, an empty side,
# an unknown region model name, an unknown Mask version, a region model name given twice.
MALFORMED_MAPS = {
    "no_equals_sign": (("SC-sylv_left",), "SC-sylv_left"),
    "empty_side": (("SC-sylv_left=",), "SC-sylv_left="),
    "unknown_region_model_name": (("XX-unknown_left=canonical_25",), "XX-unknown_left"),
    "unknown_mask_version": (("SC-sylv_left=canonical_99",), "canonical_99"),
    "name_given_twice": (("SC-sylv_left=canonical_25", "SC-sylv_left=canonical_25"), "SC-sylv_left"),
}
MAP_SOURCES = ("option", "config_file")
# (case, source) pairs of the refusal test, one per malformed map and source.
MALFORMED_CASES = [(case, source) for case in MALFORMED_MAPS for source in MAP_SOURCES]


class TestMalformedRegionMasksRefused:
    """REQ-MODELVER-BDRABCZUK-9273939CFE5A: a map entry a stage refuses stops the run before any stage.

    The entry comes from champollion-pipeline's --region-masks option or from the region_masks list of a Config file;
    the run exits with status 2, hands no configuration to the orchestrator and logs exactly one ERROR record, which
    names the value the stage option names (parse_region_masks' message). The set names canonical_25 for each region
    model name, so no mismatch stop of TASK-MODELVER-BDRABCZUK-3B2E4833B088 applies.
    """

    @pytest.mark.parametrize("case_source", MALFORMED_CASES, ids=["-".join(pair) for pair in MALFORMED_CASES])
    def test_exit_2_with_one_error_record(self, main_module, monkeypatch, tmp_path, pipeline_caplog, case_source):
        case, source = case_source
        entries, named = MALFORMED_MAPS[case]
        models = _model_set(tmp_path, "uniform_set", _uniform_versions(C25))
        raw = make_t1_input(tmp_path)
        if source == "option":
            cli_args = (str(raw), "--models-path", str(models), REGION_MASKS_OPTION, *entries)
        else:
            config_file = _write_config_file(tmp_path, models, raw, region_masks=list(entries))
            cli_args = ("--config", str(config_file))
        outcome = main_runner(main_module, monkeypatch)(*cli_args)
        assert outcome.status == 2, f"main() exit status {outcome.status!r} for {list(cli_args)}"
        assert outcome.configs == [], "main() handed a configuration to the orchestrator"
        errors = [record.getMessage() for record in pipeline_caplog.records if record.levelno >= logging.ERROR]
        assert len(errors) == 1, f"ERROR records: {errors!r}"
        assert named in errors[0], f"ERROR record does not name {named!r}: {errors[0]!r}"


# Stages a stage 5 and 6 only run enables.
STAGES_5_AND_6 = ("put_together_embeddings", "generate_snapshots")


class TestStageSelectionKeepsFolders:
    """REQ-MODELVER-BDRABCZUK-BFB170076CAB: the stage 5 and 6 folders do not depend on the enabled stages.

    Two sequential runs on one input and one local model set, given no Mask version: the default stage selection and
    --stages put_together_embeddings generate_snapshots. Both give stages 5 and 6 the same folders, under
    champollion_V1/L, L the run label of the resolver's map (REQ-MODELVER-BDRABCZUK-804574E0EC1E, 620EADAAF019).
    """

    @pytest.mark.parametrize("set_kind", ["mixed", "uniform"])
    def test_stage_5_and_6_only_run_reads_map_run_folders(self, main_module, monkeypatch, tmp_path, set_kind):
        versions = _mixed_versions() if set_kind == "mixed" else _uniform_versions(C25)
        models = _model_set(tmp_path, f"{set_kind}_set", versions)
        raw = make_t1_input(tmp_path)
        runner = main_runner(main_module, monkeypatch)
        default_args = (str(raw), "--models-path", str(models))
        selected_args = (*default_args, "--stages", *STAGES_5_AND_6)
        default_run = ran_config(runner(*default_args), default_args)
        selected_run = ran_config(runner(*selected_args), selected_args)
        assert [name for name, on in selected_run.stages.items() if on] == list(STAGES_5_AND_6)
        selected = _stages_5_and_6_folders(main_module, selected_run)
        tree = (raw / "derivatives" / "champollion_V1" / _run_label(_resolved_map(models))).resolve()
        for name, folder in selected.items():
            assert Path(folder).resolve().is_relative_to(tree), f"{name} {folder!r} not under {tree}"
        assert selected == _stages_5_and_6_folders(main_module, default_run)


def _stages_5_and_6_folders(main_module, config) -> dict[str, str]:
    """The folders the configuration gives stages 5 and 6 (stage 4's own --output left out)."""
    folders = _stage_folders(main_module, config)
    return {name: folder for name, folder in folders.items() if not name.startswith("GenerateEmbeddingsStage")}


# Hugging Face stage 4 models source of a run: "cascade" leaves the models source cascade nothing local (no
# --models-path, an empty checkout root, no models_cache entry), so it falls back to Hugging Face; "hf_enabled" names a
# Hugging Face repo id outright.
HF_SOURCES = ("cascade", "hf_enabled")
HF_REPO_ID = "neurospin/Champollion_V1"
LOCAL_MODEL_SET = "local model set"
HF_CASES = [(source, hf_source) for source in ("option", "config") for hf_source in HF_SOURCES]


def _run_mixed_map_on_hugging_face(main_module, monkeypatch, tmp_path: Path, source: str, hf_source: str):
    """Run main() with the mixed map given by --region-masks ("option") or the Config file ("config") on a Hugging
    Face models source; return (outcome, argv). No hub call reaches the network."""
    refuse_hub(monkeypatch)
    raw = make_t1_input(tmp_path)
    entries = [f"{name}={version}" for name, version in _mixed_versions().items()]
    dataset: dict[str, object] = {"models_cache": str(tmp_path / "models_cache")}
    if hf_source == "hf_enabled":
        dataset.update(hf_enabled=True, hf_repo_id=HF_REPO_ID)
    if source == "config":
        dataset["region_masks"] = entries
    config_file = _write_config_file(tmp_path, "", raw, root_path=make_checkout(tmp_path), **dataset)
    cli_args = ("--config", str(config_file), *((REGION_MASKS_OPTION, *entries) if source == "option" else ()))
    return main_runner(main_module, monkeypatch)(*cli_args), cli_args


class TestMixedMapOnHuggingFaceExits2:
    """A given map naming two Mask versions on a Hugging Face stage 4 models source exits 2 before stage 1.

    User decision 2026-11-24 11:01: stage 4 would otherwise get the run label as --model-version, a name no Hugging
    Face subfolder carries; a mixed map needs a local model set (--models-path or the local cascade steps).
    """

    @pytest.mark.parametrize("case", HF_CASES, ids=["-".join(case) for case in HF_CASES])
    def test_exit_2_before_stage_1(self, main_module, monkeypatch, tmp_path, case):
        outcome, cli_args = _run_mixed_map_on_hugging_face(main_module, monkeypatch, tmp_path, *case)
        assert outcome.status == 2, f"main() exit status {outcome.status!r} for {list(cli_args)}"
        assert outcome.configs == [], "main() handed a configuration to the orchestrator"


class TestMixedMapOnHuggingFaceOneErrorRecord:
    """A given map naming two Mask versions on a Hugging Face models source logs one ERROR record naming a local
    model set (user decision 2026-11-24 11:01)."""

    @pytest.mark.parametrize("case", HF_CASES, ids=["-".join(case) for case in HF_CASES])
    def test_one_error_record_naming_local_model_set(self, main_module, monkeypatch, tmp_path, pipeline_caplog, case):
        _run_mixed_map_on_hugging_face(main_module, monkeypatch, tmp_path, *case)
        errors = [record.getMessage() for record in pipeline_caplog.records if record.levelno >= logging.ERROR]
        assert len(errors) == 1, f"ERROR records: {errors!r}"
        assert LOCAL_MODEL_SET in errors[0], f"ERROR record does not name a {LOCAL_MODEL_SET}: {errors[0]!r}"


# Partial local model sets (user decision 2026-11-24 11:27: partial-model-set support deferred past 1.0.0): each case
# is (set folder name, region model name -> Mask version of the set's models, with a PROVENANCE.csv row each).
# "unresolved_rest": models for PARTIAL_NAMES only, in a folder whose name is no Mask version, so the resolver leaves
# every other region model name unresolved; "folder_name_rest": the same models in a folder named
# canonical_corrected_26_1, so the resolver resolves every other region model name to that folder name although the set
# has no model for it; "one_hemisphere": the mixed set without SOr_right (S.Or. has its left hemisphere only).
PARTIAL_SETS = {
    "unresolved_rest": ("partial_set", {"SC-sylv_left": C25, "SC-sylv_right": C25, "SOr_left": C26}),
    "folder_name_rest": (C26, dict.fromkeys(PARTIAL_NAMES, C25)),
    "one_hemisphere": (
        "one_hemisphere_set",
        {name: version for name, version in _mixed_versions().items() if name != "SOr_right"},
    ),
}
# The Mask version source of a partial set run: none at all ("cli"), or a Config file naming no masks_version and no
# map ("config_file").
PARTIAL_SOURCES = ("cli", "config_file")
PARTIAL_CASES = [(case, source) for case in PARTIAL_SETS for source in PARTIAL_SOURCES]
PARTIAL_IDS = ["-".join(pair) for pair in PARTIAL_CASES]
FULL_MODEL_SET = "full model set"


def _option_list(argv: list[str], option: str) -> list[str] | None:
    """The values of a multi-value option in argv ('--opt V ...' or '--opt=V'); None when the option is absent."""
    for i, token in enumerate(argv):
        if token.startswith(option + "="):
            return [token.split("=", 1)[1]]
        if token == option:
            values = []
            for value in argv[i + 1 :]:
                if value.startswith("--"):
                    break
                values.append(value)
            return values
    return None


def _run_partial_set(main_module, monkeypatch, tmp_path: Path, case: str, source: str, *extra: str):
    """Run main() on the partial set of case, given no Mask version; return (outcome, cli args, missing names).

    missing names: the region model names the set holds no model for, in REGION_MODEL_NAMES order.
    """
    folder, versions = PARTIAL_SETS[case]
    models = _model_set(tmp_path, folder, versions)
    assert sorted(entry.region for entry in find_region_models(str(models))) == sorted(versions)
    raw = make_t1_input(tmp_path)
    if source == "cli":
        cli_args = (str(raw), "--models-path", str(models), *extra)
    else:
        cli_args = ("--config", str(_write_config_file(tmp_path, models, raw)), *extra)
    missing = [name for name in REGION_MODEL_NAMES if name not in versions]
    return main_runner(main_module, monkeypatch)(*cli_args), cli_args, missing


class TestPartialSetExits2:
    """REQ-MODELVER-BDRABCZUK-B463E4170F9B: a local model set holding no model for one of the 56 region model names,
    given no Mask version, stops the run with exit status 2 before stage 1 (no configuration reaches the
    orchestrator), whether the names without a model are unresolved, resolved through the folder name only, or one
    hemisphere of a Sulcal region."""

    @pytest.mark.parametrize("case_source", PARTIAL_CASES, ids=PARTIAL_IDS)
    def test_exit_2_before_stage_1(self, main_module, monkeypatch, tmp_path, case_source):
        outcome, cli_args, _ = _run_partial_set(main_module, monkeypatch, tmp_path, *case_source)
        assert outcome.status == 2, f"main() exit status {outcome.status!r} for {list(cli_args)}"
        assert outcome.configs == [], "main() handed a configuration to the orchestrator"


class TestPartialSetOneErrorRecord:
    """REQ-MODELVER-BDRABCZUK-4A87AEB9B7F5: the stop logs exactly one ERROR record, naming each region model name the
    run processes that the set holds no model for, and a full model set."""

    @pytest.mark.parametrize("case_source", PARTIAL_CASES, ids=PARTIAL_IDS)
    def test_one_error_naming_missing(self, main_module, monkeypatch, tmp_path, pipeline_caplog, case_source):
        _, _, missing = _run_partial_set(main_module, monkeypatch, tmp_path, *case_source)
        errors = [record.getMessage() for record in pipeline_caplog.records if record.levelno >= logging.ERROR]
        assert len(errors) == 1, f"ERROR records: {errors!r}"
        unnamed = [name for name in missing if name not in errors[0]]
        assert unnamed == [], f"ERROR record does not name {unnamed!r}: {errors[0]!r}"
        assert FULL_MODEL_SET in errors[0].lower(), f"ERROR record does not name a {FULL_MODEL_SET}: {errors[0]!r}"

    def test_regions_filter_names_its_names_only(self, main_module, monkeypatch, tmp_path, pipeline_caplog):
        """--regions S.Or. on a set holding SOr_left only: the record names SOr_right, not a name outside S.Or."""
        _, _, missing = _run_partial_set(
            main_module, monkeypatch, tmp_path, "unresolved_rest", "cli", "--regions", "S.Or."
        )
        errors = [record.getMessage() for record in pipeline_caplog.records if record.levelno >= logging.ERROR]
        assert len(errors) == 1, f"ERROR records: {errors!r}"
        assert "SOr_right" in errors[0], f"ERROR record does not name SOr_right: {errors[0]!r}"
        outside = [name for name in missing if split_region_model_name(name)[0] != "S.Or." and name in errors[0]]
        assert outside == [], f"ERROR record names region model names outside --regions: {outside!r}"


class TestRegionsFilterNeedsOnlyItsModels:
    """REQ-MODELVER-BDRABCZUK-B463E4170F9B with --regions R: the region model names the run processes are the two of
    each Sulcal region of R, so only the models of R must exist (caller brief 2026-11-24)."""

    def test_set_holding_models_of_regions_runs(self, main_module, monkeypatch, tmp_path):
        """--regions S.C.-sylv. on a set holding both S.C.-sylv. models only: the run goes on."""
        outcome, cli_args, _ = _run_partial_set(
            main_module, monkeypatch, tmp_path, "folder_name_rest", "cli", "--regions", "S.C.-sylv."
        )
        ran_config(outcome, cli_args)

    def test_region_without_hemisphere_model_exits_2(self, main_module, monkeypatch, tmp_path):
        """--regions S.Or. on a set holding SOr_left only: exit 2 before stage 1."""
        outcome, cli_args, _ = _run_partial_set(
            main_module, monkeypatch, tmp_path, "folder_name_rest", "cli", "--regions", "S.Or."
        )
        assert outcome.status == 2, f"main() exit status {outcome.status!r} for {list(cli_args)}"
        assert outcome.configs == [], "main() handed a configuration to the orchestrator"
