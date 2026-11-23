#!/usr/bin/env python3
"""
Stage 4 reads each region model's crops from the crops folder of its Mask version
(TASK-MODELVER-BDRABCZUK-EAB7E63E7F08).

Every run test drives GenerateEmbeddings through parse_args, then run(), on a fixture models tree; fetch_models
returns that tree and execute_command records each evaluate.py command. The crops a model reads are the -sk skeleton
file and the -i subject file of its command.

The per-region Mask version map reaches the stage on the command line, as the --region-masks NAME=VERSION pairs of
the shared option (champollion_pipeline.utils.region_masks): run_stage() formats it with format_region_masks.

Each class names its Requirement by title (``alm get <id>``).

The models folder is named after FOLDER_VERSION, which the per-region Mask version resolver reads as the Mask version of
every region model it finds no PROVENANCE.csv row for. A run whose map (or --masks) names another version for a region
model writes a PROVENANCE.csv row naming that version (write_models_provenance), so the stage does not refuse the run
as a mismatch between the crops and the model (TASK-MODELVER-BDRABCZUK-86405DAD28CE).
"""

import json
import logging
import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.generate_embeddings import PROVENANCE_FILE_NAME, GenerateEmbeddings
from champollion_pipeline.models.version_options import PROVENANCE_FILE_NAME as MODELS_PROVENANCE_FILE_NAME
from champollion_pipeline.models.version_options import PROVENANCE_REGION_COLUMN, PROVENANCE_VERSION_COLUMN
from champollion_pipeline.utils.lib import CORTICAL_TILES_VERSION
from champollion_pipeline.utils.qc_report import read_qc_tsv
from champollion_pipeline.utils.region_masks import REGION_MASKS_OPTION, format_region_masks
from tests.stage2_layout import add_crops
from tests.test_generate_embeddings_multi_model_output import make_model, model_of_command

pytestmark = pytest.mark.unit  # noqa: V107

# The models folder is named after a Mask version, so the parser takes --masks from it when --masks is not given.
FOLDER_VERSION = "canonical_25"
OTHER_VERSION = "canonical_corrected_26_1"
UNNAMED_FOLDER = "my_models"
SC_LEFT = "SC-sylv_left"
STS_LEFT = "STs_left"
STS_RIGHT = "STs_right"
FOREIGN_REGION = "NOTAREGION_left"
CROP_OF = {SC_LEFT: "S.C.-sylv.", STS_LEFT: "S.T.s.", STS_RIGHT: "S.T.s."}
SIDE_OF = {SC_LEFT: "L", STS_LEFT: "L", STS_RIGHT: "R"}
COHORT = ("sub-01", "sub-02")
EMBEDDINGS = "ID,dim1\nsub-01,0.1\nsub-02,0.2\n"
INPUT_QC_FILE = "input_qc.tsv"
USAGE_ERROR = 2
MAP_KEY = "region_mask_versions"


def map_args(versions: dict[str, str] | None) -> list[str]:
    """The --region-masks arguments giving ``versions``; none for None."""
    if versions is None:
        return []
    return [REGION_MASKS_OPTION, *format_region_masks(versions)]


def crops_dir(root: Path, version: str) -> Path:
    """<datasets_root>/derivatives/<cortical_version>/crops/<version>/2mm of the fixture dataset under ``root``."""
    return root / "dataset" / "derivatives" / f"cortical_tiles-{CORTICAL_TILES_VERSION}" / "crops" / version / "2mm"


def lay_crops(root: Path, version: str, regions) -> None:
    """Skeleton crops of COHORT for the sulcal region of each region model of ``regions``, in ``version``'s folder."""
    add_crops(crops_dir(root, version), COHORT, sorted({CROP_OF[region] for region in regions}))


def expected_inputs(root: Path, version: str, region: str) -> tuple[Path, Path]:
    """The skeleton and subject files of ``region`` in the crops folder of ``version``."""
    mask = crops_dir(root, version) / CROP_OF[region] / "mask"
    side = SIDE_OF[region]
    return mask / f"{side}skeleton.npy", mask / f"{side}skeleton_subject.csv"


def error_line(stderr: str) -> str:
    """The argparse error line (``<prog>: error: <message>``) of ``stderr``; fails when there is none."""
    lines = [line for line in stderr.splitlines() if ": error: " in line]
    assert lines, f"no argparse error line in stderr: {stderr!r}"
    return lines[-1]


def names_option(text: str, option: str) -> bool:
    """Whether ``text`` names ``option`` as a whole option (--masks is not named by --region-masks)."""
    return re.search(rf"(?<![\w-]){re.escape(option)}(?![\w-])", text) is not None


class RecordingEvaluate:
    """Stand-in for execute_command: records each evaluate.py command and writes an embeddings table to its -s path.

    The -m folder is a temporary copy removed after the command, so the model it holds is resolved at call time."""

    def __init__(self):
        self.commands: list[list[str]] = []
        self.models: list[str] = []

    def __call__(self, command, **_kwargs):
        self.commands.append(list(command))
        self.models.append(model_of_command(command))
        saving = Path(command[command.index("-s") + 1])
        saving.parent.mkdir(parents=True, exist_ok=True)
        saving.write_text(EMBEDDINGS)
        return 0

    def inputs_of(self, model: Path) -> tuple[Path, Path]:
        """The -sk and -i paths of the one command evaluating ``model``."""
        wanted = os.path.realpath(model)
        commands = [command for command, held in zip(self.commands, self.models, strict=True) if held == wanted]
        assert len(commands) == 1, f"expected one evaluate.py command for {model}, got {self.commands}"
        (command,) = commands
        return Path(command[command.index("-sk") + 1]), Path(command[command.index("-i") + 1])


def run_stage(root: Path, *extra: str, versions: dict[str, str] | None = None) -> tuple[object, RecordingEvaluate]:
    """Parse, then run the stage on the models tree root/FOLDER_VERSION and the dataset root/dataset.

    ``versions`` is given as --region-masks. Returns the status (the exit code when parsing or running exits) and
    the recorded evaluate.py commands."""
    models = root / FOLDER_VERSION
    evaluate = RecordingEvaluate()
    script = GenerateEmbeddings()
    try:
        script.parse_args(
            [str(models), str(root / "dataset"), "--output", str(root / "out"), *extra, *map_args(versions)]
        )
        script.execute_command = evaluate
        with patch.object(GenerateEmbeddings, "fetch_models", return_value=str(models)):
            status = script.run()
    except SystemExit as stop:
        status = stop.code
    return status, evaluate


def parse_only(argv: list[str], capsys) -> tuple[object, str]:
    """Parse ``argv`` with a fresh stage parser; return the exit code (None when parsing ends) and the stderr."""
    try:
        GenerateEmbeddings().parse_args(argv)
        status = None
    except SystemExit as stop:
        status = stop.code
    return status, capsys.readouterr().err


def lay_flat_models(root: Path, regions) -> dict[str, Path]:
    """One model with an empty model id per region model name of ``regions``."""
    return {region: make_model(root / FOLDER_VERSION, region) for region in regions}


def write_models_provenance(root: Path, versions: dict[str, str]) -> None:
    """Write root/FOLDER_VERSION/PROVENANCE.csv with one row per region model name of ``versions``, naming its Mask
    version, so the per-region Mask version resolver gives each region model the version the run reads crops of."""
    rows = "".join(f"{region},{version}\n" for region, version in sorted(versions.items()))
    header = f"{PROVENANCE_REGION_COLUMN},{PROVENANCE_VERSION_COLUMN}\n"
    (root / FOLDER_VERSION / MODELS_PROVENANCE_FILE_NAME).write_text(header + rows)


def run_mapped(root: Path, versions: dict[str, str]) -> tuple[dict[str, Path], RecordingEvaluate]:
    """Run the stage with map ``versions`` over flat models of its region model names, crops in both versions; the
    models' PROVENANCE.csv names the mapped versions."""
    models = lay_flat_models(root, versions)
    write_models_provenance(root, versions)
    lay_crops(root, FOLDER_VERSION, models)
    lay_crops(root, OTHER_VERSION, models)
    status, evaluate = run_stage(root, versions=versions)
    assert status == 0, status
    return models, evaluate


def refusal_line(root: Path, folder: str, extra: list[str], capsys) -> str:
    """Parse models folder root/``folder``, ``extra`` and a one-pair map; assert exit 2; return the error line."""
    (root / folder).mkdir()
    argv = [str(root / folder), str(root / "dataset"), *extra, *map_args({SC_LEFT: FOLDER_VERSION})]
    status, stderr = parse_only(argv, capsys)
    assert status == USAGE_ERROR, (status, stderr)
    return error_line(stderr)


def read_provenance(root: Path) -> dict:
    """The provenance.json the stage wrote into root/out."""
    path = root / "out" / PROVENANCE_FILE_NAME
    assert path.is_file(), f"{path} was not written"
    return json.loads(path.read_text())


class TestEvaluateReadsTheMappedVersionCrops:
    """Stage 4 evaluates each region model on the crops of its mapped Mask version."""

    @pytest.mark.parametrize(
        ("sc_version", "sts_version"),
        [(FOLDER_VERSION, OTHER_VERSION), (OTHER_VERSION, OTHER_VERSION)],
        ids=["one_region_mapped_to_another_version", "every_region_mapped_to_another_version"],
    )
    def test_a_region_mapped_to_another_version_reads_that_version_folder(self, tmp_path, sc_version, sts_version):
        models, evaluate = run_mapped(tmp_path, {SC_LEFT: sc_version, STS_RIGHT: sts_version})

        assert evaluate.inputs_of(models[STS_RIGHT]) == expected_inputs(tmp_path, sts_version, STS_RIGHT)
        assert evaluate.inputs_of(models[SC_LEFT]) == expected_inputs(tmp_path, sc_version, SC_LEFT)

    def test_the_two_hemispheres_of_one_region_read_their_own_version_folders(self, tmp_path):
        models, evaluate = run_mapped(tmp_path, {STS_LEFT: OTHER_VERSION, STS_RIGHT: FOLDER_VERSION})

        assert evaluate.inputs_of(models[STS_LEFT]) == expected_inputs(tmp_path, OTHER_VERSION, STS_LEFT)
        assert evaluate.inputs_of(models[STS_RIGHT]) == expected_inputs(tmp_path, FOLDER_VERSION, STS_RIGHT)

    def test_each_model_id_of_a_region_reads_the_region_version_folder(self, tmp_path):
        first = make_model(tmp_path / FOLDER_VERSION, STS_RIGHT, "run_1")
        second = make_model(tmp_path / FOLDER_VERSION, STS_RIGHT, "run_2")
        write_models_provenance(tmp_path, {STS_RIGHT: OTHER_VERSION})
        lay_crops(tmp_path, FOLDER_VERSION, (STS_RIGHT,))
        lay_crops(tmp_path, OTHER_VERSION, (STS_RIGHT,))

        status, evaluate = run_stage(tmp_path, versions={STS_RIGHT: OTHER_VERSION})

        assert status == 0, status
        for model in (first, second):
            assert evaluate.inputs_of(model) == expected_inputs(tmp_path, OTHER_VERSION, STS_RIGHT)


class TestInputQcChecksTheMappedVersionCrops:
    """Stage 4 input QC checks each region model's crops in its mapped Mask version folder."""

    @pytest.fixture(autouse=True)  # noqa: V105 - overrides the conftest fixture
    def _stub_stage4_input_qc(self):
        """Run the real input QC: overrides the conftest fixture of the same name."""

    @staticmethod
    def qc_values(root: Path) -> dict[str, int]:
        path = root / "out" / INPUT_QC_FILE
        assert path.is_file(), f"{path} was not written"
        return {row["participant_id"]: row["qc"] for row in read_qc_tsv(path)}

    def test_crops_only_in_the_mapped_version_folder_pass_the_input_qc(self, tmp_path):
        versions = {SC_LEFT: FOLDER_VERSION, STS_RIGHT: OTHER_VERSION}
        lay_flat_models(tmp_path, versions)
        write_models_provenance(tmp_path, versions)
        lay_crops(tmp_path, FOLDER_VERSION, (SC_LEFT,))
        lay_crops(tmp_path, OTHER_VERSION, (STS_RIGHT,))

        status, _evaluate = run_stage(tmp_path, versions=versions)

        assert self.qc_values(tmp_path) == dict.fromkeys(COHORT, 1)
        assert status == 0, status

    def test_crops_only_in_the_masks_folder_fail_the_input_qc_of_a_region_mapped_elsewhere(self, tmp_path):
        versions = {SC_LEFT: FOLDER_VERSION, STS_RIGHT: OTHER_VERSION}
        lay_flat_models(tmp_path, versions)
        write_models_provenance(tmp_path, versions)
        lay_crops(tmp_path, FOLDER_VERSION, (SC_LEFT, STS_RIGHT))

        status, evaluate = run_stage(tmp_path, versions=versions)

        assert self.qc_values(tmp_path) == dict.fromkeys(COHORT, 0)
        assert evaluate.commands == []
        assert status != 0, status


class TestNoMapReadsTheMasksFolder:
    """Stage 4 without a per-region Mask version map reads every region model's crops from the --masks folder."""

    @pytest.mark.parametrize(
        ("masks_args", "masks"),
        [((), FOLDER_VERSION), (("--masks", OTHER_VERSION), OTHER_VERSION)],
        ids=["default_masks", "given_masks"],
    )
    def test_each_region_model_reads_the_masks_folder(self, tmp_path, masks_args, masks):
        models = lay_flat_models(tmp_path, (SC_LEFT, STS_LEFT, STS_RIGHT))
        write_models_provenance(tmp_path, dict.fromkeys(models, masks))
        lay_crops(tmp_path, FOLDER_VERSION, models)
        lay_crops(tmp_path, OTHER_VERSION, models)

        status, evaluate = run_stage(tmp_path, *masks_args)

        assert status == 0, status
        for region, model in models.items():
            assert evaluate.inputs_of(model) == expected_inputs(tmp_path, masks, region)


class TestUnmappedRegionModelExits2:
    """Stage 4 refuses a region model that the per-region Mask version map does not name."""

    @pytest.fixture(autouse=True)  # noqa: V105 - overrides the conftest fixture
    def _stub_stage4_input_qc(self):
        """Run the real input QC, so a run that reaches it writes input_qc.tsv: overrides the conftest fixture."""

    def test_a_region_model_missing_from_the_map_exits_2_naming_it_before_the_input_qc(self, tmp_path, capsys):
        lay_flat_models(tmp_path, (SC_LEFT, STS_RIGHT))
        lay_crops(tmp_path, FOLDER_VERSION, (SC_LEFT, STS_RIGHT))

        status, evaluate = run_stage(tmp_path, versions={SC_LEFT: FOLDER_VERSION})

        assert status == USAGE_ERROR, status
        assert STS_RIGHT in error_line(capsys.readouterr().err)
        assert not (tmp_path / "out" / INPUT_QC_FILE).exists()
        assert evaluate.commands == []

    def test_a_foreign_region_folder_warns_at_parse_then_exits_2_naming_it_before_the_input_qc(
        self, tmp_path, capsys, pipeline_caplog
    ):
        """REQ-MODELVER-BDRABCZUK-3AE63A554F2A then REQ-MODELVER-BDRABCZUK-386C6BFFA544 on one run (user answer
        2026-11-23 ~22:47): a region folder named none of the 56 region model names is reported once as unresolved at
        parse time, then the map, which cannot name it, makes the stage exit 2 naming it before the input QC."""
        lay_flat_models(tmp_path, (SC_LEFT, FOREIGN_REGION))
        lay_crops(tmp_path, FOLDER_VERSION, (SC_LEFT,))

        status, evaluate = run_stage(tmp_path, versions={SC_LEFT: FOLDER_VERSION})

        warnings_naming = [
            record.getMessage()
            for record in pipeline_caplog.records
            if record.levelno == logging.WARNING and FOREIGN_REGION in record.getMessage()
        ]
        assert len(warnings_naming) == 1, warnings_naming
        assert status == USAGE_ERROR, status
        assert FOREIGN_REGION in error_line(capsys.readouterr().err)
        assert not (tmp_path / "out" / INPUT_QC_FILE).exists()
        assert evaluate.commands == []


class TestLegacyWithMapExits2:
    """Stage 4 refuses --legacy beside --region-masks."""

    def test_legacy_with_region_masks_exits_2_naming_both(self, tmp_path, capsys):
        line = refusal_line(tmp_path, FOLDER_VERSION, ["--legacy"], capsys)

        assert names_option(line, "--legacy"), line
        assert names_option(line, REGION_MASKS_OPTION), line


class TestMasksWithMapExits2:
    """Stage 4 refuses --masks beside --region-masks."""

    def test_masks_with_region_masks_exits_2_naming_both(self, tmp_path, capsys):
        line = refusal_line(tmp_path, FOLDER_VERSION, ["--masks", FOLDER_VERSION], capsys)

        assert names_option(line, "--masks"), line
        assert names_option(line, REGION_MASKS_OPTION), line


class TestMapWithUnnamedModelsFolderExits2:
    """Stage 4 with --region-masks and an unnamed local models folder asks for --model-version."""

    def test_unnamed_folder_with_region_masks_exits_2_advising_model_version_or_renaming(self, tmp_path, capsys):
        line = refusal_line(tmp_path, UNNAMED_FOLDER, [], capsys)

        assert names_option(line, "--model-version"), line
        assert "rename" in line.lower(), line

    def test_model_version_with_region_masks_is_accepted(self, tmp_path, capsys):
        (tmp_path / UNNAMED_FOLDER).mkdir()
        argv = [str(tmp_path / UNNAMED_FOLDER), str(tmp_path / "dataset"), "--model-version", FOLDER_VERSION]

        status, stderr = parse_only([*argv, *map_args({SC_LEFT: FOLDER_VERSION})], capsys)

        assert status is None, stderr


class TestProvenanceHoldsTheMap:
    """Stage 4 provenance.json holds the per-region Mask version map."""

    def test_provenance_holds_the_given_map(self, tmp_path):
        versions = {SC_LEFT: OTHER_VERSION, STS_RIGHT: FOLDER_VERSION}

        run_mapped(tmp_path, versions)

        assert read_provenance(tmp_path)[MAP_KEY] == versions


class TestNoMapProvenanceHasNoMapKey:
    """Stage 4 provenance.json without a map has no region_mask_versions key."""

    def test_provenance_without_map_has_no_map_key(self, tmp_path):
        models = lay_flat_models(tmp_path, (SC_LEFT, STS_RIGHT))
        lay_crops(tmp_path, FOLDER_VERSION, models)

        status, _evaluate = run_stage(tmp_path)

        assert status == 0, status
        assert MAP_KEY not in read_provenance(tmp_path)
