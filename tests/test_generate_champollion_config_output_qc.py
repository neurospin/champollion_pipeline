#!/usr/bin/env python3
"""
Stage 3 output QC tests of generate_champollion_config.py: output_qc.tsv, one row per region dataset YAML file.

The report is a per-run report (check, qc, reason), as the snapshots output QC: a config belongs to a region and
a side, not to a subject. Synthetic fixtures only. Each test passes an explicit --champollion_loc under the test
folder, so nothing reaches the real checkout.
"""

import csv
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from champollion_pipeline.derivatives_layout import DEFAULT_MASKS_VERSION
from champollion_pipeline.generate_champollion_config import GenerateChampollionConfig
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from tests.log_assertions import FAIL
from tests.stage2_layout import add_crops, crop_array
from tests.test_generate_champollion_config_internals import fake_execute_command, make_crop

_MODULE = "champollion_pipeline.generate_champollion_config"
_QC_FILE = "output_qc.tsv"
_DATASET = "TEST01"
_REGIONS = ("S.C.-sylv.", "S.T.s.")
_KINDS = ("skeleton", "foldlabel", "mask")
_COHORT = ("sub-01", "sub-02")
_CONFIG = "SC-sylv_left.yaml"
_CONFIGS = ("STs_left.yaml", "STs_right.yaml", _CONFIG, "SC-sylv_right.yaml")
_PATH_KEYS = (
    "numpy_all",
    "subjects_all",
    "foldlabel_all",
    "subjects_foldlabel_all",
    "train_val_csv_file",
    "cutin_mask_path",
)
_REQUIRED_KEYS = ("dataset_name", *_PATH_KEYS, "input_size")
_ABSENT_FILE = "${dataset_folder}/no/such/file.npy"
_BROKEN_YAML = "a: [unclosed\n"
_PARTIAL_STATUS = 3


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage3_input_qc():
    """Run the real input QC: overrides the conftest fixture of the same name."""


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage3_output_qc():
    """Run the real output QC: overrides the conftest fixture of the same name."""


@pytest.fixture
def crops(temp_dir):
    """Crops folder of the dataset, holding the complete crops of the cohort in ``_REGIONS``."""
    return _add_cohort(_crops_dir(temp_dir))


@pytest.fixture
def partial_crops(crops):
    """Crops folder where ``_COHORT[1]`` is left out of the skeleton CSV of the left side of ``_REGIONS[0]``."""
    add_crops(crops, _COHORT[:1], _REGIONS[:1], sides="L")
    make_crop(crops, _REGIONS[0], side="L")
    return crops


@pytest.fixture
def lost_crops(crops):
    """Crops folder where no subject passes the input QC: the left skeleton array of ``_REGIONS[0]`` is gone."""
    crop_array(crops, _REGIONS[0], "L", "skeleton").unlink()
    return crops


@dataclass
class _StageRun:
    """What run() did: its result and the folders it wrote to."""

    result: int
    configs_root: Path
    champollion_loc: Path

    @property
    def qc_path(self):
        """Path of output_qc.tsv in the configs root."""
        return self.configs_root / _QC_FILE

    def rows(self):
        """Rows of output_qc.tsv, keyed by check."""
        assert self.qc_path.is_file(), f"{self.qc_path} was not written"
        with self.qc_path.open(newline="", encoding="utf-8") as report:
            return {row["check"]: row for row in csv.DictReader(report, delimiter="\t")}

    def outcome(self, check=_CONFIG):
        """The (qc, reason) cells of the row of ``check``."""
        row = self.rows()[check]
        return row["qc"], row["reason"]

    def qc_values(self):
        """The set of qc cells of the report."""
        return {row["qc"] for row in self.rows().values()}


def _dataset_parent(crops_dir):
    """<D>: the parent of the <dataset> directory in ``crops_dir``."""
    return next(parent.parent for parent in Path(crops_dir).parents if parent.name == _DATASET)


def _crops_dir(root):
    """Crops folder of the dataset under ``root``, for the default Mask version."""
    crops = Path(root) / "data" / _DATASET / "derivatives" / DERIVATIVES_FOLDER / "crops"
    return crops / DEFAULT_MASKS_VERSION / "2mm"


def _add_cohort(crops_dir):
    """Lay complete crops of the cohort in ``_REGIONS``, with skeleton arrays stage 3 reads a size from."""
    add_crops(crops_dir, _COHORT, _REGIONS, _KINDS)
    for region in _REGIONS:
        for side in "LR":
            make_crop(crops_dir, region, side=side)
    return Path(crops_dir)


def _run_stage(crops_dir, *extra_args, output=None, after_configs=None, write_localization=None):
    """Run run() on ``crops_dir``; ``output``, when given, is --output and the configs root of the returned run.

    ``after_configs`` gets the folder of the region YAMLs once they are written; ``write_localization``, called
    with the path and the dataset folder, stands in for the writing of the localization YAML.
    """
    crops_dir = Path(crops_dir)
    dataset_parent = _dataset_parent(crops_dir)
    default_root = dataset_parent / _DATASET / "derivatives" / "champollion_V1" / "configs"
    configs_root = Path(output) if output else default_root
    champollion_loc = dataset_parent / "champollion_V1_checkout"

    script = GenerateChampollionConfig()
    script.parse_args(
        [
            str(crops_dir),
            "--dataset",
            _DATASET,
            "--champollion_loc",
            str(champollion_loc),
            *(["--output", str(output)] if output else []),
            *extra_args,
        ]
    )
    script.execute_command = fake_execute_command
    if after_configs:
        create_configs = script._create_dataset_configs

        def create_then_change(*args):
            create_configs(*args)
            after_configs(configs_root / "dataset" / _DATASET)

        script._create_dataset_configs = create_then_change
    if write_localization:
        script._write_localization_yaml = write_localization
    return _StageRun(script.run(), configs_root, champollion_loc)


def _edit_mapping(edit, name=_CONFIG):
    """after_configs hook: apply ``edit`` to the parsed mapping of the region YAML ``name``."""

    def rewrite(dataset_loc):
        path = dataset_loc / name
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        edit(document)
        path.write_text(yaml.safe_dump(document), encoding="utf-8")

    return rewrite


def _write_text(text, name=_CONFIG):
    """after_configs hook: replace the region YAML ``name`` by ``text``."""

    def rewrite(dataset_loc):
        (dataset_loc / name).write_text(text, encoding="utf-8")

    return rewrite


def _remove(*names):
    """after_configs hook: delete the region YAMLs ``names``."""

    def delete(dataset_loc):
        for name in names:
            (dataset_loc / name).unlink()

    return delete


def _write_localization_text(text):
    """write_localization hook: write ``text`` as the localization YAML."""

    def write(dest_path, _dataset_folder):
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(dest_path).write_text(text, encoding="utf-8")

    return write


def _skip_localization(_dest_path, _dataset_folder):
    """write_localization hook: leave no localization YAML behind."""


def _messages_at(level, pipeline_caplog):
    """Messages of the generate_champollion_config logger records at exactly ``level``."""
    return [r.getMessage() for r in pipeline_caplog.records if r.name == _MODULE and r.levelno == level]


class TestReport:
    def test_report_lands_in_the_derivatives_configs_by_default(self, crops):
        assert _run_stage(crops).qc_path.is_file()

    def test_report_lands_in_the_output_folder(self, crops, temp_dir):
        output = Path(temp_dir) / "chosen_configs"

        _run_stage(crops, output=output)

        assert (output / _QC_FILE).is_file()

    def test_external_config_leaves_the_report_in_the_configs_root(self, crops, temp_dir):
        external = Path(temp_dir) / "elsewhere" / "local.yaml"

        run = _run_stage(crops, "--external-config", str(external))

        assert run.qc_path.is_file()
        assert not (external.parent / _QC_FILE).exists()

    def test_report_header_is_check_qc_reason(self, crops):
        run = _run_stage(crops)

        assert run.qc_path.is_file()
        assert run.qc_path.read_text(encoding="utf-8").splitlines()[0].split("\t") == ["check", "qc", "reason"]

    def test_report_stays_out_of_champollion_loc(self, crops):
        run = _run_stage(crops)

        assert run.qc_path.is_file()
        assert not run.champollion_loc.exists()

    def test_no_report_without_a_passed_subject(self, lost_crops):
        run = _run_stage(lost_crops)

        assert (run.configs_root / "input_qc.tsv").is_file()
        assert not run.qc_path.exists()


class TestRows:
    def test_one_row_per_region_and_side(self, crops):
        assert sorted(_run_stage(crops).rows()) == sorted(_CONFIGS)

    def test_reference_yaml_gets_no_row(self, crops):
        run = _run_stage(crops)

        assert (run.configs_root / "dataset" / _DATASET / "reference.yaml").is_file()
        assert "reference.yaml" not in run.rows()

    def test_file_under_crop_path_gets_no_row(self, crops):
        (crops / "notes.txt").write_text("not a region\n")

        assert sorted(_run_stage(crops).rows()) == sorted(_CONFIGS)


class TestUsableConfig:
    def test_generated_config_has_qc_1_and_no_reason(self, crops):
        rows = _run_stage(crops).rows()

        assert {(row["qc"], row["reason"]) for row in rows.values()} == {("1", "")}

    def test_external_crops_config_has_qc_1(self, temp_dir):
        crops = _add_cohort(Path(temp_dir) / "shared" / _DATASET / "some" / "crops" / "2mm")

        assert _run_stage(crops, "--external_crops").qc_values() == {"1"}

    def test_absent_distbottom_and_extremities_keep_qc_1(self, crops):
        assert not crop_array(crops, _REGIONS[0], "L", "distbottom").exists()
        assert not crop_array(crops, _REGIONS[0], "L", "extremities").exists()

        assert _run_stage(crops).outcome() == ("1", "")

    def test_absent_optional_key_keeps_qc_1(self, crops):
        run = _run_stage(crops, after_configs=_edit_mapping(lambda document: document.pop("pickle_normal")))

        assert run.outcome() == ("1", "")


class TestAbsentConfig:
    def test_absent_config_has_the_missing_config_reason(self, crops):
        assert _run_stage(crops, after_configs=_remove(_CONFIG)).outcome() == ("0", "missing_config")

    def test_other_rows_keep_qc_1(self, crops):
        run = _run_stage(crops, after_configs=_remove(_CONFIG))

        assert {run.outcome(check) for check in _CONFIGS if check != _CONFIG} == {("1", "")}

    def test_checkout_copy_does_not_stand_in_for_an_absent_config(self, crops):
        checkout = _dataset_parent(crops) / "champollion_V1_checkout" / "champollion" / "configs" / "dataset"

        def move_to_the_checkout(dataset_loc):
            (checkout / _DATASET).mkdir(parents=True)
            shutil.move(dataset_loc / _CONFIG, checkout / _DATASET / _CONFIG)

        assert _run_stage(crops, after_configs=move_to_the_checkout).outcome() == ("0", "missing_config")


class TestUnparsableConfig:
    def test_broken_yaml_has_the_unparsable_config_reason(self, crops):
        run = _run_stage(crops, after_configs=_write_text(_BROKEN_YAML))

        assert run.outcome() == ("0", "unparsable_config")

    def test_yaml_list_has_the_unparsable_config_reason(self, crops):
        run = _run_stage(crops, after_configs=_write_text("- numpy_all\n- subjects_all\n"))

        assert run.outcome() == ("0", "unparsable_config")


class TestRequiredKeys:
    @pytest.mark.parametrize("key", _REQUIRED_KEYS)
    def test_absent_required_key_has_the_missing_key_reason(self, crops, key):
        run = _run_stage(crops, after_configs=_edit_mapping(lambda document: document.pop(key)))

        assert run.outcome() == ("0", f"missing_key:{key}")

    @pytest.mark.parametrize("empty", [None, ""])
    def test_required_key_without_a_value_has_the_missing_key_reason(self, crops, empty):
        run = _run_stage(crops, after_configs=_edit_mapping(lambda document: document.update(numpy_all=empty)))

        assert run.outcome() == ("0", "missing_key:numpy_all")


class TestReferencedFiles:
    @pytest.mark.parametrize("key", _PATH_KEYS)
    def test_absent_referenced_file_has_the_missing_path_reason(self, crops, key):
        run = _run_stage(crops, after_configs=_edit_mapping(lambda document: document.update({key: _ABSENT_FILE})))

        assert run.outcome() == ("0", f"missing_path:{key}")

    def test_directory_is_not_a_referenced_file(self, crops):
        directory = "${dataset_folder}"

        run = _run_stage(crops, after_configs=_edit_mapping(lambda document: document.update(numpy_all=directory)))

        assert run.outcome() == ("0", "missing_path:numpy_all")

    def test_masks_other_than_the_crops_give_the_missing_path_reason(self, crops):
        rows = _run_stage(crops, "--masks", "another_mask_set").rows()

        assert {row["qc"] for row in rows.values()} == {"0"}
        assert all("missing_path:numpy_all" in row["reason"].split(";") for row in rows.values())


class TestLocalization:
    def test_external_config_file_is_read(self, crops, temp_dir):
        external = Path(temp_dir) / "elsewhere" / "local.yaml"

        assert _run_stage(crops, "--external-config", str(external)).qc_values() == {"1"}

    def test_external_config_directory_is_read(self, crops, temp_dir):
        external = Path(temp_dir) / "elsewhere"
        external.mkdir()

        assert _run_stage(crops, "--external-config", str(external)).qc_values() == {"1"}

    def test_localization_under_the_output_folder_is_read(self, crops, temp_dir):
        assert _run_stage(crops, output=Path(temp_dir) / "chosen_configs").qc_values() == {"1"}

    def test_localization_name_is_read(self, crops):
        assert _run_stage(crops, "--localization", "jean-zay").qc_values() == {"1"}

    def test_value_of_the_written_localization_is_used(self, crops):
        run = _run_stage(crops, write_localization=_write_localization_text("dataset_folder: /no/such/folder\n"))

        assert "missing_path:numpy_all" in run.outcome()[1].split(";")

    def test_configs_root_localization_is_ignored_with_external_config(self, crops, temp_dir):
        output = Path(temp_dir) / "chosen_configs"
        stale = output / "dataset_localization" / "local.yaml"
        stale.parent.mkdir(parents=True)
        stale.write_text("dataset_folder: /no/such/folder\n", encoding="utf-8")
        external = Path(temp_dir) / "elsewhere" / "local.yaml"

        assert _run_stage(crops, "--external-config", str(external), output=output).qc_values() == {"1"}

    @pytest.mark.parametrize(
        "write_localization",
        [_skip_localization, _write_localization_text(_BROKEN_YAML), _write_localization_text("other_key: 1\n")],
        ids=["absent", "unparsable", "no_dataset_folder_key"],
    )
    def test_unusable_localization_has_the_missing_dataset_folder_reason(self, crops, write_localization):
        rows = _run_stage(crops, write_localization=write_localization).rows()

        assert sorted(rows) == sorted(_CONFIGS)
        assert {row["qc"] for row in rows.values()} == {"0"}
        assert all("missing_dataset_folder" in row["reason"].split(";") for row in rows.values())


class TestSeveralReasons:
    def test_reasons_are_joined_by_semicolons(self, crops):
        def spoil(document):
            document.pop("numpy_all")
            document["foldlabel_all"] = _ABSENT_FILE

        _, reason = _run_stage(crops, after_configs=_edit_mapping(spoil)).outcome()

        assert set(reason.split(";")) == {"missing_key:numpy_all", "missing_path:foldlabel_all"}


class TestLog:
    def test_one_error_record_per_qc0_row(self, crops, pipeline_caplog):
        run = _run_stage(crops, after_configs=_remove(_CONFIG))

        errors = _messages_at(logging.ERROR, pipeline_caplog)
        assert run.outcome() == ("0", "missing_config")
        assert len(errors) == 1
        assert _CONFIG in errors[0]
        assert "missing_config" in errors[0]

    def test_no_error_record_on_a_complete_run(self, crops, pipeline_caplog):
        run = _run_stage(crops)

        assert run.qc_path.is_file()
        assert _messages_at(logging.ERROR, pipeline_caplog) == []

    def test_no_qc1_row_emits_one_fail_record(self, crops, pipeline_caplog):
        run = _run_stage(crops, after_configs=_remove(*_CONFIGS))

        fails = _messages_at(FAIL, pipeline_caplog)
        assert run.qc_values() == {"0"}
        assert len(fails) == 1
        assert str(run.qc_path) in fails[0]

    def test_qc1_row_emits_no_fail_record(self, crops, pipeline_caplog):
        run = _run_stage(crops, after_configs=_remove(_CONFIG))

        assert run.qc_values() == {"0", "1"}
        assert _messages_at(FAIL, pipeline_caplog) == []


class TestExitPolicy:
    def test_no_qc1_row_returns_1(self, crops):
        run = _run_stage(crops, after_configs=_remove(*_CONFIGS))

        assert run.qc_values() == {"0"}
        assert run.result == 1

    def test_partial_exclusion_with_no_qc1_row_returns_1(self, partial_crops):
        run = _run_stage(partial_crops, after_configs=_remove(*_CONFIGS))

        assert run.qc_values() == {"0"}
        assert run.result == 1

    def test_some_configs_invalid_returns_3(self, crops):
        run = _run_stage(crops, after_configs=_remove(_CONFIG))

        assert run.qc_values() == {"0", "1"}
        assert run.result == _PARTIAL_STATUS

    def test_partial_exclusion_with_some_configs_invalid_returns_3(self, partial_crops):
        run = _run_stage(partial_crops, after_configs=_remove(_CONFIG))

        assert run.qc_values() == {"0", "1"}
        assert run.result == _PARTIAL_STATUS

    def test_partial_exclusion_with_usable_configs_returns_3(self, partial_crops):
        run = _run_stage(partial_crops)

        assert run.qc_values() == {"1"}
        assert run.result == _PARTIAL_STATUS

    def test_complete_run_returns_0(self, crops):
        run = _run_stage(crops)

        assert run.qc_values() == {"1"}
        assert run.result == 0
