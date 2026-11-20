#!/usr/bin/env python3
"""
Stage 5 over several models (put_together_embeddings.py, main.py), epic TASK-MULTIMODEL-BDRABCZUK-A02369CED8AE, Task
TASK-MULTIMODEL-BDRABCZUK-55101BF650FA.

Stage 4 writes <source>/<region>/<model id>/full_embeddings.csv for a region with several models (or under
--per-model-folders), else <source>/<region>/full_embeddings.csv. A model id is the path below the region folder: any
name, possibly holding '/'. Stage 5 writes one combined folder per model id,
<output>/<model id>/<region>_embeddings.csv, copies a single-model (flat) region into each model id folder, keeps
today's flat output for a source with no model id, refuses a region folder holding both forms, and runs its input and
output QC per model id. --model <id> restricts it to one model id; main.py forwards dataset.model as --model.

The sources are fixture folders laid out as stage 4 leaves them: no stage 4 code runs. Each run names two regions with
--regions, so the input QC checks those two only, and leaves out the stage 4 output_qc.tsv, so the expected subjects
are the IDs of the region files. The real input and output QC run: the conftest stubs are overridden below.
"""

import argparse
import csv
import json
import logging
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

from champollion_pipeline.cli_options import add_config_arguments, apply_config_arguments
from champollion_pipeline.config import DatasetConfig, PipelineConfig
from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings
from champollion_pipeline.utils.qc_report import ID_COLUMN, QC_COLUMN, REASON_COLUMN, REASON_SEPARATOR, read_qc_tsv
from tests.log_assertions import FAIL
from tests.main_stage_helpers import load_main_module, option_values, stage_paths_config
from tests.test_put_together_embeddings_input_qc import (
    _COHORT,
    _OTHER_SUBJECT,
    _REGIONS,
    _SUBJECT,
    _messages,
    _shared_check,
)

pytestmark = pytest.mark.unit  # noqa: V107

_EMBEDDINGS = "full_embeddings.csv"
_SIDECAR = "embedding_width.json"
_INPUT_REPORT = "input_qc.tsv"
_OUTPUT_REPORT = "output_qc.tsv"
_REGION = _REGIONS[0]  # the multi-model region of the fixtures
_FLAT_REGION = _REGIONS[1]  # the single-model region of the fixtures
_ID = "v1"
_NESTED_ID = "v2/run2"  # a model id holding '/'
_MODEL_IDS = (_ID, _NESTED_ID)
_UNKNOWN_ID = "v9"
_WIDTH = 3
_NESTED_WIDTH = 5
_FLAT = ""  # the model id key of a flat region file in a layout
_MODEL_OPTION = "--model"

Layout = Mapping[str, Mapping[str, str]]  # region -> model id (_FLAT: the region folder itself) -> table text


def _table(value: str, subjects: tuple[str, ...] = _COHORT) -> str:
    """An embeddings table of ``subjects``, each row _WIDTH cells of ``value``."""
    rows = "".join(",".join((subject, *(value,) * _WIDTH)) + "\n" for subject in subjects)
    return "ID," + ",".join(f"dim{i}" for i in range(1, _WIDTH + 1)) + "\n" + rows


_ID_TABLE = _table("0.5")
_NESTED_TABLE = _table("0.25")
_FLAT_TABLE = _table("0.75")
_TABLE_OF = {_ID: _ID_TABLE, _NESTED_ID: _NESTED_TABLE}
_MULTI: Layout = {_REGION: {_ID: _ID_TABLE, _NESTED_ID: _NESTED_TABLE}, _FLAT_REGION: {_FLAT: _FLAT_TABLE}}
_NO_MODEL_ID: Layout = {_REGION: {_FLAT: _ID_TABLE}, _FLAT_REGION: {_FLAT: _FLAT_TABLE}}
_MIXED: Layout = {_REGION: {_FLAT: _ID_TABLE, _ID: _ID_TABLE}, _FLAT_REGION: {_FLAT: _FLAT_TABLE}}


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage5_input_qc():
    """Run the real stage 5 input QC: overrides the conftest fixture of the same name."""


@pytest.fixture(autouse=True)  # noqa: V103
def _stub_stage5_output_qc():
    """Run the real stage 5 output QC: overrides the conftest fixture of the same name."""


def _make_source(tmp_path: Path, layout: Layout, widths: Mapping[str, int] | None = None) -> Path:
    """The stage 4 region embeddings folder of ``layout``, an embedding_width.json beside each table.

    ``widths`` gives the sidecar width of a model id; the others get _WIDTH.
    """
    source = tmp_path / "region_embeddings"
    for region, models in layout.items():
        for model_id, table in models.items():
            folder = source / region / model_id if model_id else source / region
            folder.mkdir(parents=True, exist_ok=True)
            (folder / _EMBEDDINGS).write_text(table, encoding="utf-8")
            width = (widths or {}).get(model_id, _WIDTH)
            (folder / _SIDECAR).write_text(json.dumps({"width": width}), encoding="utf-8")
    return source


@dataclass(frozen=True)
class _Run:
    status: int
    source: Path
    output: Path

    def copies(self) -> list[str]:
        """POSIX paths, relative to the output folder, of every region copy below it."""
        return sorted(path.relative_to(self.output).as_posix() for path in self.output.rglob("*_embeddings.csv"))

    def copy_folders(self) -> set[str]:
        """The folders, relative to the output folder, holding a region copy."""
        return {path.rpartition("/")[0] for path in self.copies()}

    def copy_of(self, model_id: str, region: str) -> Path:
        return self.output / model_id / f"{region}_embeddings.csv"

    def report_rows(self, model_id: str, name: str) -> dict[str, tuple[int, str]]:
        """(qc, reason) by participant_id of the report ``name`` of ``model_id``."""
        report = self.output / model_id / name
        assert report.is_file(), f"{report} was not written"
        return {row[ID_COLUMN]: (row[QC_COLUMN], row[REASON_COLUMN]) for row in read_qc_tsv(report)}


def _run(tmp_path: Path, source: Path, *options: str) -> _Run:
    output = tmp_path / "combined"
    script = PutTogetherEmbeddings()
    script.parse_args([str(source), "--output_path", str(output), "--regions", _REGION, _FLAT_REGION, *options])
    return _Run(script.run(), source, output)


def _run_layout(tmp_path: Path, layout: Layout, *options: str) -> _Run:
    return _run(tmp_path, _make_source(tmp_path, layout), *options)


def _row_ids(csv_path: Path) -> list[str]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        return [row["ID"] for row in csv.DictReader(handle)]


def _error_messages(capture) -> list[str]:
    return [text for level in (logging.ERROR, FAIL) for text in _messages(capture, level)]


class TestCombinePerModelId:
    def test_one_folder_per_model_id(self, tmp_path):
        run = _run_layout(tmp_path, _MULTI)

        assert run.copy_folders() == set(_MODEL_IDS)

    @pytest.mark.parametrize("model_id", _MODEL_IDS)
    def test_model_file_copied_to_its_id_folder(self, tmp_path, model_id):
        run = _run_layout(tmp_path, _MULTI)

        copy = run.copy_of(model_id, _REGION)
        assert copy.is_file(), f"{copy} was not written; copies: {run.copies()}"
        assert copy.read_text(encoding="utf-8") == _TABLE_OF[model_id]


class TestFlatRegionInEachIdFolder:
    @pytest.mark.parametrize("model_id", _MODEL_IDS)
    def test_flat_region_copied_into_each_id_folder(self, tmp_path, model_id):
        run = _run_layout(tmp_path, _MULTI)

        copy = run.copy_of(model_id, _FLAT_REGION)
        assert copy.is_file(), f"{copy} was not written; copies: {run.copies()}"
        assert copy.read_text(encoding="utf-8") == _FLAT_TABLE


class TestSourceWithoutModelId:
    def test_region_copies_directly_in_output(self, tmp_path):
        run = _run_layout(tmp_path, _NO_MODEL_ID)

        assert run.copies() == sorted(f"{region}_embeddings.csv" for region in (_REGION, _FLAT_REGION))
        assert (run.output / f"{_FLAT_REGION}_embeddings.csv").read_text(encoding="utf-8") == _FLAT_TABLE
        assert [path.name for path in run.output.iterdir() if path.is_dir()] == []


class TestMixedRegionRefused:
    def test_returns_1(self, tmp_path):
        assert _run_layout(tmp_path, _MIXED).status == 1

    def test_error_names_the_region(self, tmp_path, pipeline_caplog):
        _run_layout(tmp_path, _MIXED)

        assert [text for text in _error_messages(pipeline_caplog) if _REGION in text] != []


class TestModelOption:
    @pytest.mark.parametrize("model_id", _MODEL_IDS)
    def test_only_the_chosen_id_folder(self, tmp_path, model_id):
        run = _run_layout(tmp_path, _MULTI, _MODEL_OPTION, model_id)

        assert run.copies() != []
        assert run.copy_folders() == {model_id}

    def test_chosen_model_file_copied(self, tmp_path):
        run = _run_layout(tmp_path, _MULTI, _MODEL_OPTION, _NESTED_ID)

        copy = run.copy_of(_NESTED_ID, _REGION)
        assert copy.is_file(), f"{copy} was not written; copies: {run.copies()}"
        assert copy.read_text(encoding="utf-8") == _NESTED_TABLE

    def test_flat_region_copied_into_the_chosen_folder(self, tmp_path):
        run = _run_layout(tmp_path, _MULTI, _MODEL_OPTION, _ID)

        copy = run.copy_of(_ID, _FLAT_REGION)
        assert copy.is_file(), f"{copy} was not written; copies: {run.copies()}"
        assert copy.read_text(encoding="utf-8") == _FLAT_TABLE

    def test_unknown_id_returns_1(self, tmp_path):
        assert _run_layout(tmp_path, _MULTI, _MODEL_OPTION, _UNKNOWN_ID).status == 1

    def test_unknown_id_error_lists_the_ids(self, tmp_path, pipeline_caplog):
        _run_layout(tmp_path, _MULTI, _MODEL_OPTION, _UNKNOWN_ID)

        errors = _error_messages(pipeline_caplog)
        assert [text for text in errors if all(model_id in text for model_id in _MODEL_IDS)] != [], errors


class TestQcPerModelId:
    @pytest.mark.parametrize("model_id", _MODEL_IDS)
    def test_input_qc_report_per_id(self, tmp_path, model_id):
        run = _run_layout(tmp_path, _MULTI)

        assert run.report_rows(model_id, _INPUT_REPORT) == {subject: (1, "") for subject in _COHORT}

    @pytest.mark.parametrize("model_id", _MODEL_IDS)
    def test_output_qc_report_per_id(self, tmp_path, model_id):
        run = _run_layout(tmp_path, _MULTI)

        assert run.report_rows(model_id, _OUTPUT_REPORT) == {subject: (1, "") for subject in _COHORT}

    def test_region_lacking_the_id_is_missing(self, tmp_path):
        layout = {_REGION: {_ID: _ID_TABLE, _NESTED_ID: _NESTED_TABLE}, _FLAT_REGION: {_ID: _FLAT_TABLE}}

        rows = _run_layout(tmp_path, layout).report_rows(_NESTED_ID, _INPUT_REPORT)

        assert set(rows) == set(_COHORT)
        for qc, reasons in rows.values():
            assert qc == 0
            assert f"missing_embeddings:{_FLAT_REGION}" in reasons.split(REASON_SEPARATOR)

    def test_width_from_the_sidecar_beside_the_model_file(self, tmp_path):
        source = _make_source(tmp_path, _MULTI, widths={_NESTED_ID: _NESTED_WIDTH})
        widths: dict[str, int | None] = {}

        def fake_check(csv_path, region, expected_subjects, width):
            path = Path(csv_path)
            if path.is_relative_to(source):
                widths[path.relative_to(source).as_posix()] = width
            return {}

        with _shared_check(fake_check):
            _run(tmp_path, source)

        nested_file = f"{_REGION}/{_NESTED_ID}/{_EMBEDDINGS}"
        assert widths.get(nested_file) == _NESTED_WIDTH, widths


_ONE_SUBJECT_LOST_FOR_ID: Layout = {
    _REGION: {_ID: _table("0.5", (_SUBJECT,)), _NESTED_ID: _NESTED_TABLE},
    _FLAT_REGION: {_FLAT: _FLAT_TABLE},
}


class TestExcludedSubjectPerModelId:
    def test_rows_removed_in_its_id_folder(self, tmp_path):
        run = _run_layout(tmp_path, _ONE_SUBJECT_LOST_FOR_ID)

        copy = run.copy_of(_ID, _FLAT_REGION)
        assert copy.is_file(), f"{copy} was not written; copies: {run.copies()}"
        assert _row_ids(copy) == [_SUBJECT]

    def test_rows_kept_in_another_id_folder(self, tmp_path):
        run = _run_layout(tmp_path, _ONE_SUBJECT_LOST_FOR_ID)

        copy = run.copy_of(_NESTED_ID, _FLAT_REGION)
        assert copy.is_file(), f"{copy} was not written; copies: {run.copies()}"
        assert _OTHER_SUBJECT in _row_ids(copy)


@pytest.fixture(scope="module")
def main_module():
    return load_main_module("_champollion_main_multi_model_combine_under_test")


def _combine_stage(main_module, config):
    return main_module.PutTogetherEmbeddingsStage(
        "put_together_embeddings", config, logging.getLogger("test_put_together_embeddings_multi_model")
    )


class TestMainModelSetting:
    def test_dataset_model_default_empty(self):
        assert typing.get_type_hints(DatasetConfig).get("model") is str
        assert DatasetConfig().model == ""

    def test_model_option_sets_dataset_model(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        add_config_arguments(parser)
        config = PipelineConfig()

        apply_config_arguments(config, parser.parse_args([_MODEL_OPTION, _NESTED_ID]))

        assert config.dataset.model == _NESTED_ID

    def test_model_forwarded(self, main_module):
        config = stage_paths_config(main_module)
        config.dataset.model = _NESTED_ID  # noqa: V101

        argv = _combine_stage(main_module, config).build_args()

        assert option_values(argv, _MODEL_OPTION) == [_NESTED_ID]

    def test_no_model_option_when_empty(self, main_module):
        argv = _combine_stage(main_module, stage_paths_config(main_module)).build_args()

        assert option_values(argv, _MODEL_OPTION) == []
