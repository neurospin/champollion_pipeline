#!/usr/bin/env python3
"""
canonical_corrected_26_1 is the default Mask version (TASK-MODELVER-BDRABCZUK-39D727185A86, BREAKING;
user decision 2026-11-17): DEFAULT_MASKS_VERSION, main.py, generate_embeddings, purge_subject and
prune_failed_subjects follow it, and a rerun of a canonical_25 dataset leaves the canonical_25 trees alone.

- REQ-MODELVER-BDRABCZUK-55A89AA59006 (stakeholder): a run without a Mask version embeds
  canonical_corrected_26_1 crops with the canonical_corrected_26_1 model set.
- REQ-MODELVER-BDRABCZUK-DB6383571271: main.py passes --masks canonical_corrected_26_1 to stages 2-4.
- REQ-MODELVER-BDRABCZUK-F61A76A06BEB: main.py passes --model-version canonical_corrected_26_1 to stage 4.
- REQ-MODELVER-BDRABCZUK-0B63094F3C07: main.py's stage 4-6 folders lie under the canonical_corrected_26_1 tree.
- REQ-MODELVER-BDRABCZUK-866B9C952564: a main.py rerun of a canonical_25 dataset names no canonical_25 path.
- REQ-MODELVER-BDRABCZUK-5400A842B160: a Hugging Face models_path with neither --masks nor --model-version
  downloads the canonical_corrected_26_1 model set.
- REQ-MODELVER-BDRABCZUK-9CC58FEABB1D / REQ-MODELVER-BDRABCZUK-5C088F1A859B: purge_subject and
  prune_failed_subjects without --masks leave crops/canonical_25 unchanged.
- REQ-MODELVER-BDRABCZUK-D793BCAF17DC / REQ-MODELVER-BDRABCZUK-CA56E8032243: a purge that finds nothing and
  a prune with nothing to prune name the crops/<masks>/2mm directory they searched.
- REQ-MODELVER-BDRABCZUK-E75786C44C5A: the generate_embeddings --masks help names the new fallback.
- REQ-MODELVER-BDRABCZUK-371406F051B8: DEFAULT_MASKS_VERSION is canonical_corrected_26_1.
- Superseding revisions of the canonical_25 defaults (class TestSupersededDefaults):
  REQ-MODELVER-BDRABCZUK-82AC4A152D01, REQ-EMBVER-BDRABCZUK-3219994C034F, REQ-EMBVER-BDRABCZUK-BF32F3E9A1C3,
  REQ-PATTERNS-BDRABCZUK-28A325628A4D, REQ-PATTERNS-BDRABCZUK-8CC8EFCBDB20, REQ-MCPCLI-BDRABCZUK-C840CC1D1DFE,
  REQ-MCPCLI-BDRABCZUK-A6F28CB13103, REQ-MODELVER-BDRABCZUK-6A7EC2D5389F.

No model, no network, no BrainVISA: stage argv come from each stage's build_args, scripts run on
synthetic trees under tmp_path, the Hugging Face strategy is a mock. Synthetic subject IDs only.
"""

import dataclasses
import logging
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from champollion_pipeline.derivatives_layout import DEFAULT_MASKS_VERSION
from champollion_pipeline.generate_champollion_config import GenerateChampollionConfig
from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from champollion_pipeline.models import fetch as models_fetch
from champollion_pipeline.path_inference import infer_paths
from champollion_pipeline.prune_failed_subjects import PruneFailedSubjects
from champollion_pipeline.purge_subject import PurgeSubject
from champollion_pipeline.utils.input_kind import InputKind
from tests.main_stage_helpers import load_main_module, option_values
from tests.prune_layout import crops_2mm, make_crop_files, run_prune

pytestmark = pytest.mark.unit  # noqa: V107

NEW_DEFAULT = "canonical_corrected_26_1"
OLD_DEFAULT = "canonical_25"
HF_REPO = "neurospin/Champollion_V1"
REPO_ROOT = Path(__file__).resolve().parent.parent
PURGE_LOGGER = "champollion_pipeline.purge_subject"
PRUNE_LOGGER = "champollion_pipeline.prune_failed_subjects"
OUTPUT_LEVEL = 25

# Stage class name in main.py -> stage name, for stages 2 to 6.
STAGES_2_TO_6 = {
    "RunCorticalTilesStage": "run_cortical_tiles",
    "GenerateChampollionConfigStage": "generate_champollion_config",
    "GenerateEmbeddingsStage": "generate_embeddings",
    "PutTogetherEmbeddingsStage": "put_together_embeddings",
    "GenerateSnapshotsStage": "generate_snapshots",
}
MASKS_STAGES = ("RunCorticalTilesStage", "GenerateChampollionConfigStage", "GenerateEmbeddingsStage")


@pytest.fixture(scope="module")
def main_module():
    return load_main_module("_champollion_main_masks_default_corrected_under_test")


def _default_config(main_module, dataset_root: Path, hf_enabled: bool = False):
    """PipelineConfig main.py builds for a T1 input at dataset_root with no Mask version given."""
    config = main_module.PipelineConfig(models_path="/models")
    config.dataset.hf_enabled = hf_enabled  # noqa: V101
    if hf_enabled:
        config.dataset.hf_repo_id = HF_REPO  # noqa: V101
    return infer_paths(config, str(dataset_root), InputKind.T1)


def _stage_argv(main_module, config, stage_class_name: str) -> list[str]:
    """The argv main.py's stage ``stage_class_name`` builds for config (nothing runs)."""
    stage_cls = getattr(main_module, stage_class_name)
    stage = stage_cls(STAGES_2_TO_6[stage_class_name], config, logging.getLogger("test_masks_default_corrected"))
    return list(stage.build_args())


def _canonical_25_dataset(root: Path) -> Path:
    """A dataset processed under canonical_25: crops, configs, embeddings and snapshots trees on disk."""
    derivatives = root / "derivatives"
    champollion = derivatives / "champollion_V1" / OLD_DEFAULT
    for folder in (
        derivatives / "cortical_tiles-2026" / "crops" / OLD_DEFAULT / "2mm" / "S.Or." / "mask",
        champollion / "region_embeddings" / "SOr_left",
        champollion / "embeddings",
        champollion / "snapshots",
        derivatives / "champollion_V1" / "configs",
    ):
        folder.mkdir(parents=True)
    (champollion / "embeddings" / "SOr_left_embeddings.csv").write_text("ID,dim1\nsub-01,0.1\n")
    return root


def _parse_embeddings(argv: list[str]) -> GenerateEmbeddings:
    """A GenerateEmbeddings whose args are argv parsed (and resolved) by the real parser; fail on rejection."""
    script = GenerateEmbeddings()
    try:
        script.parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"GenerateEmbeddings parser rejected {argv!r} (SystemExit {exc.code})")
    return script


def _hf_subfolder(script: GenerateEmbeddings, models_path: str):
    """The subfolder fetch_models hands to HuggingFaceStrategy for models_path (the strategy is a mock)."""
    strategy = MagicMock(can_handle=MagicMock(return_value=True), fetch=MagicMock(return_value="/hf"))
    with patch.object(models_fetch, "HuggingFaceStrategy", return_value=strategy) as strategy_cls:
        script.fetch_models(models_path)
    assert strategy_cls.call_args is not None, "HuggingFaceStrategy was not constructed"
    return strategy_cls.call_args.kwargs.get("subfolder")


def _tree_state(root: Path) -> dict[str, bytes]:
    """Every file and directory under root, with the file contents."""
    return {
        str(path.relative_to(root)): (b"<dir>" if path.is_dir() else path.read_bytes())
        for path in sorted(root.rglob("*"))
    }


def _purge_tree(derivatives: Path, subject: str) -> None:
    """subject in skeletons/, and in crops/<version>/2mm (crop file and aggregated array) for both versions."""
    (derivatives / "skeletons" / subject).mkdir(parents=True)
    (derivatives / "skeletons" / subject / "data.nii.gz").touch()
    for version in (OLD_DEFAULT, NEW_DEFAULT):
        mask_dir = derivatives / "crops" / version / "2mm" / "region_a" / "mask"
        (mask_dir / "Rcrops").mkdir(parents=True)
        (mask_dir / "Rcrops" / f"{subject}.nii.gz").write_bytes(b"crop")
        np.save(mask_dir / "Rskeleton.npy", np.ones((2, 1, 2, 2, 2), dtype=np.float32))
        pd.DataFrame({"participant_id": [subject, "sub-keep"]}).to_csv(mask_dir / "Rskeleton_subject.csv", index=False)


def _run_purge(derivatives: Path, subject: str, masks: str | None = None) -> int:
    """Run purge_subject on derivatives; masks=None passes no --masks. Return run()'s status (1 on ValueError)."""
    script = PurgeSubject()
    argv = [str(derivatives), "--subject", subject] + (["--masks", masks] if masks is not None else [])
    script.args = script.parse_args(argv)
    try:
        return script.run()
    except ValueError:
        return 1


def _messages(capture, level: int, logger_name: str) -> list[str]:
    return [r.getMessage() for r in capture.records if r.levelno == level and r.name == logger_name]


def _readme_embeddings_masks_row() -> str:
    """The --masks row of the options table of the Stage 4 section of docs/usage.md (moved from README section 5)."""
    text = (REPO_ROOT / "docs" / "usage.md").read_text(encoding="utf-8")
    section = re.search(r"^##\s+Stage 4\b[^\n]*$(.*?)(?=^##\s|\Z)", text, re.MULTILINE | re.DOTALL)
    assert section, "docs/usage.md has no '## Stage 4' section"
    rows = [line for line in section.group(1).splitlines() if re.match(r"^\|\s*`--masks`\s*\|", line.strip())]
    assert rows, "docs/usage.md Stage 4 has no `--masks` table row"
    return rows[-1]


class TestDefaultRunEmbedsMatchingModelSet:
    """REQ-MODELVER-BDRABCZUK-55A89AA59006: no Mask version -> c26.1 crops embedded with the c26.1 model set."""

    def test_stage4_reads_c261_crops_and_fetches_c261_models(self, main_module, tmp_path):
        config = _default_config(main_module, tmp_path / "MYDATASET", hf_enabled=True)
        argv = _stage_argv(main_module, config, "GenerateEmbeddingsStage")
        script = _parse_embeddings(argv)
        assert script.args.masks == NEW_DEFAULT, f"stage 4 reads crops of {script.args.masks!r}; argv {argv!r}"
        subfolder = _hf_subfolder(script, script.args.models_path)
        assert subfolder == NEW_DEFAULT, f"stage 4 fetches model set {subfolder!r}; argv {argv!r}"


class TestMainForwardsDefaultMasks:
    """REQ-MODELVER-BDRABCZUK-DB6383571271: no Mask version -> --masks canonical_corrected_26_1 for stages 2-4."""

    @pytest.mark.parametrize("stage_class_name", MASKS_STAGES)
    def test_stage_masks_option(self, main_module, tmp_path, stage_class_name):
        argv = _stage_argv(main_module, _default_config(main_module, tmp_path / "MYDATASET"), stage_class_name)
        assert option_values(argv, "--masks") == [NEW_DEFAULT], f"{stage_class_name} argv {argv!r}"


class TestMainForwardsDefaultModelVersion:
    """REQ-MODELVER-BDRABCZUK-F61A76A06BEB: no Mask version -> --model-version canonical_corrected_26_1."""

    def test_stage4_model_version_option(self, main_module, tmp_path):
        config = _default_config(main_module, tmp_path / "MYDATASET")
        argv = _stage_argv(main_module, config, "GenerateEmbeddingsStage")
        assert option_values(argv, "--model-version") == [NEW_DEFAULT], f"generate_embeddings argv {argv!r}"


class TestMainVersionedFolders:
    """REQ-MODELVER-BDRABCZUK-0B63094F3C07: stage 4-6 folders lie under D/derivatives/champollion_V1/c26.1."""

    def test_stage_4_to_6_folders(self, main_module, tmp_path):
        root = tmp_path / "MYDATASET"
        config = _default_config(main_module, root)
        tree = root / "derivatives" / "champollion_V1" / NEW_DEFAULT
        embeddings_argv = _stage_argv(main_module, config, "GenerateEmbeddingsStage")
        combine_argv = _stage_argv(main_module, config, "PutTogetherEmbeddingsStage")
        snapshots_argv = _stage_argv(main_module, config, "GenerateSnapshotsStage")
        folders = {
            "stage 4 --output": option_values(embeddings_argv, "--output"),
            "stage 5 source": [combine_argv[0]],
            "stage 5 --output_path": option_values(combine_argv, "--output_path"),
            "stage 6 --output_dir": option_values(snapshots_argv, "--output_dir"),
            "stage 6 --embeddings_dir": option_values(snapshots_argv, "--embeddings_dir"),
        }
        for name, values in folders.items():
            assert len(values) == 1, f"{name}: {values!r}"
            assert Path(values[0]).resolve().is_relative_to(tree.resolve()), f"{name} {values[0]!r} not under {tree}"


class TestMainRerunLeavesCanonical25:
    """REQ-MODELVER-BDRABCZUK-866B9C952564: a rerun of a canonical_25 dataset names no canonical_25 path."""

    @pytest.mark.parametrize("stage_class_name", list(STAGES_2_TO_6))
    def test_rerun_argv_has_no_canonical_25(self, main_module, tmp_path, stage_class_name):
        root = _canonical_25_dataset(tmp_path / "MYDATASET")
        argv = _stage_argv(main_module, _default_config(main_module, root), stage_class_name)
        assert not [token for token in argv if OLD_DEFAULT in token], f"{stage_class_name} argv {argv!r}"


class TestHfDefaultModelSet:
    """REQ-MODELVER-BDRABCZUK-5400A842B160: HF models_path, neither option -> c26.1 model set downloaded."""

    def test_hf_subfolder_is_new_default(self, tmp_path):
        script = _parse_embeddings([HF_REPO, str(tmp_path / "MYDATASET")])
        subfolder = _hf_subfolder(script, HF_REPO)
        assert subfolder == NEW_DEFAULT, f"Hugging Face subfolder {subfolder!r}, expected {NEW_DEFAULT!r}"


class TestPurgeDefaultLeavesCanonical25:
    """REQ-MODELVER-BDRABCZUK-9CC58FEABB1D: purge_subject without --masks changes nothing under crops/canonical_25."""

    def test_canonical_25_crops_unchanged(self, tmp_path):
        derivatives = tmp_path / "cortical_tiles-2026"
        _purge_tree(derivatives, "sub-01")
        before = _tree_state(derivatives / "crops" / OLD_DEFAULT)
        _run_purge(derivatives, "sub-01")
        assert _tree_state(derivatives / "crops" / OLD_DEFAULT) == before


class TestPruneDefaultLeavesCanonical25:
    """REQ-MODELVER-BDRABCZUK-5C088F1A859B: prune without --masks deletes nothing under crops/canonical_25."""

    def test_canonical_25_crops_unchanged(self, tmp_path):
        for version in (OLD_DEFAULT, NEW_DEFAULT):
            make_crop_files(tmp_path, "S.Or.", "R", ["sub-01", "sub-02"], version)
        old_tree = crops_2mm(tmp_path, OLD_DEFAULT).parent
        before = _tree_state(old_tree)
        run_prune(tmp_path, [("sub-01", 0), ("sub-02", 1)])
        assert _tree_state(old_tree) == before


class TestPurgeNamesSearchedCrops:
    """REQ-MODELVER-BDRABCZUK-D793BCAF17DC: a purge refusal for an unknown subject names the crops dir searched."""

    @pytest.mark.parametrize("masks", [OLD_DEFAULT, NEW_DEFAULT])
    def test_error_names_crops_dir(self, tmp_path, pipeline_caplog, masks):
        derivatives = tmp_path / "cortical_tiles-2026"
        _purge_tree(derivatives, "sub-01")
        assert _run_purge(derivatives, "sub-99", masks) != 0
        searched = str(derivatives / "crops" / masks / "2mm")
        errors = _messages(pipeline_caplog, logging.ERROR, PURGE_LOGGER)
        assert any(searched in message for message in errors), f"no ERROR naming {searched}: {errors!r}"


class TestPruneNamesSearchedCrops:
    """REQ-MODELVER-BDRABCZUK-CA56E8032243: Nothing to prune names the crops/<masks>/2mm directory searched."""

    @pytest.mark.parametrize("masks", [OLD_DEFAULT, NEW_DEFAULT])
    def test_nothing_to_prune_names_crops_dir(self, tmp_path, pipeline_caplog, masks):
        make_crop_files(tmp_path, "S.Or.", "R", ["sub-01"], masks)
        assert run_prune(tmp_path, [("sub-01", 1)], masks=masks) == 0
        searched = str(crops_2mm(tmp_path, masks))
        outputs = _messages(pipeline_caplog, OUTPUT_LEVEL, PRUNE_LOGGER)
        assert any("Nothing to prune" in m and searched in m for m in outputs), outputs


class TestEmbeddingsMasksHelp:
    """REQ-MODELVER-BDRABCZUK-E75786C44C5A: the --masks help names canonical_corrected_26_1 as non-local default."""

    def test_masks_help_names_new_default(self):
        actions = [a for a in GenerateEmbeddings().parser._actions if "--masks" in a.option_strings]
        assert len(actions) == 1, actions
        help_text = " ".join(actions[0].help.split())
        assert NEW_DEFAULT in help_text, help_text
        assert "not a local directory" in help_text, help_text


class TestDefaultConstant:
    """REQ-MODELVER-BDRABCZUK-371406F051B8: DEFAULT_MASKS_VERSION equals canonical_corrected_26_1."""

    def test_constant(self):
        assert DEFAULT_MASKS_VERSION == NEW_DEFAULT


class TestSupersededDefaults:
    """Superseding revisions of the canonical_25 default Requirements (old ids in each docstring)."""

    def test_embeddings_non_local_masks(self, tmp_path):
        """REQ-MODELVER-BDRABCZUK-82AC4A152D01 (supersedes 860AD9250275)."""
        script = _parse_embeddings([HF_REPO, str(tmp_path / "MYDATASET")])
        assert script.args.masks == NEW_DEFAULT

    def test_dataset_config_default(self, main_module):
        """REQ-EMBVER-BDRABCZUK-3219994C034F (supersedes C56942772AB4)."""
        fields = {field.name: field for field in dataclasses.fields(main_module.DatasetConfig)}
        assert fields["masks_version"].default == NEW_DEFAULT

    def test_yaml_without_key(self, main_module, tmp_path):
        """REQ-EMBVER-BDRABCZUK-BF32F3E9A1C3 (supersedes 28951C1124C9)."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.safe_dump({"dataset": {"name": "example_dataset"}}))
        loaded = main_module.ConfigLoader.load_from_yaml(str(config_file))
        assert loaded.dataset.masks_version == NEW_DEFAULT

    def test_prune_parser_default(self, tmp_path):
        """REQ-PATTERNS-BDRABCZUK-28A325628A4D (supersedes CF0DAF759A12)."""
        args = PruneFailedSubjects().parse_args([str(tmp_path), "--qc", str(tmp_path / "qc.tsv")])
        assert args.masks == NEW_DEFAULT

    def test_purge_parser_default(self, tmp_path):
        """REQ-PATTERNS-BDRABCZUK-8CC8EFCBDB20 (supersedes 7A4C1CB45061)."""
        args = PurgeSubject().parse_args([str(tmp_path), "--subject", "sub-01"])
        assert args.masks == NEW_DEFAULT

    def test_config_cli_defaults(self):
        """REQ-MCPCLI-BDRABCZUK-C840CC1D1DFE (supersedes 5FC9DCCFB22C)."""
        namespace = vars(GenerateChampollionConfig().parser.parse_args(["/crops", "--dataset", "MYDATASET"]))
        champollion_loc = Path(namespace.pop("champollion_loc")).resolve()
        assert champollion_loc == (REPO_ROOT / "external" / "champollion_V1").resolve()
        expected = {
            "masks": NEW_DEFAULT,
            "localization": "local",
            "output": None,
            "external_config": None,
            "external_crops": False,
        }
        assert {key: namespace[key] for key in expected} == expected

    def test_embeddings_cli_defaults(self):
        """REQ-MCPCLI-BDRABCZUK-A6F28CB13103 (supersedes D543F0EAEB4B)."""
        namespace = vars(_parse_embeddings(["/models", "/data/MYDATASET"]).args)
        expected = {
            "masks": NEW_DEFAULT,
            "cortical_version": "cortical_tiles-2026",
            "checkpoint_epoch": 80,
            "masks_version": None,
            "output": None,
            "subjects": None,
            "regions": None,
            "models_cache": None,
            "legacy": False,
        }
        assert {key: namespace[key] for key in expected} == expected

    def test_readme_masks_row(self):
        """REQ-MODELVER-BDRABCZUK-6A7EC2D5389F (supersedes 37B69C8D1B88)."""
        row = _readme_embeddings_masks_row()
        assert "--model-version" in row, row
        assert NEW_DEFAULT in row, row
