#!/usr/bin/env python3
"""
Per-region Mask version of a local model set, part 2: PROVENANCE.csv (TASK-MODELVER-BDRABCZUK-5F2E9945DE10,
DEF-MODELVER-BDRABCZUK-291232AA949D).

A local model set without model_set.json may carry a PROVENANCE.csv (header region, mask_version_trained_on,
source_run, training_ckpt, original_lightning_version; one row per region model name such as
FPO-SCu-ScCal_left). resolve_region_mask_versions(models_path) reads the row of each region model name:
model_set.json masks > PROVENANCE.csv row > folder name. A region model name without a row, or with an empty
value, falls to the folder name; a value outside MASK_VERSIONS is unresolved; two rows naming different known
Mask versions conflict; a PROVENANCE.csv lacking a required column, or unreadable, leaves every name unresolved,
and the resolver's module logger then emits one ERROR record naming the path of that PROVENANCE.csv.

The fixture shape mirrors the rosette canonical_corrected_26_1 set (47 region model names trained on
canonical_25, 9 on canonical_corrected_26_1), written under tmp_path only.

Pure function: no model, no network; only tmp_path model set folders.
"""

import csv
import json
import logging

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from champollion_pipeline.derivatives_layout import DEFAULT_REGIONS, compute_region_model_names
from champollion_pipeline.models import version_options
from champollion_pipeline.models.version_options import (
    MASK_CONFLICT,
    MASK_RESOLVED,
    MASK_SOURCE_FOLDER_NAME,
    MASK_SOURCE_MODEL_SET,
    MASK_UNRESOLVED,
    MASK_VERSIONS,
    resolve_region_mask_versions,
)

pytestmark = pytest.mark.unit  # noqa: V107

MODEL_NAMES = compute_region_model_names(DEFAULT_REGIONS)
SOURCE_PROVENANCE = "PROVENANCE.csv"
PROVENANCE_HEADER = (
    "region",
    "mask_version_trained_on",
    "source_run",
    "training_ckpt",
    "original_lightning_version",
)
UNKNOWN_FOLDER_NAME = "my_models"
UNKNOWN_VERSION = "canonical_99"
CANONICAL_25 = "canonical_25"
CORRECTED_26_1 = "canonical_corrected_26_1"
REGION_MODEL_NAME = "FPO-SCu-ScCal_left"
# The 9 region model names of the rosette canonical_corrected_26_1 set trained on canonical_corrected_26_1.
ROSETTE_CORRECTED_NAMES = (
    "FCLp-subsc-FCLa-INSULA_left",
    "FCLp-subsc-FCLa-INSULA_right",
    "FPO-SCu-ScCal_left",
    "SCall-SsP-SintraCing_left",
    "ScCal-SLi_left",
    "SFinter-SFsup_right",
    "SFint-FCMant_right",
    "SFint-SR_right",
    "SsP-SPaint_left",
)


def _other_version(version: str) -> str:
    """The known Mask version that is not <version>."""
    return CANONICAL_25 if version == CORRECTED_26_1 else CORRECTED_26_1


def _model_set(base, folder_name: str, marker_masks: str | None = None):
    """A local model set folder <base>/models/<folder_name>; with model_set.json naming <marker_masks> when given."""
    path = base / "models" / folder_name
    path.mkdir(parents=True)
    if marker_masks is not None:
        (path / "model_set.json").write_text(
            json.dumps({"model_version": marker_masks, "masks": marker_masks}), encoding="utf-8"
        )
    return path


def _write_provenance(model_set_dir, rows, header=PROVENANCE_HEADER) -> None:
    """Write <model_set_dir>/PROVENANCE.csv with <header>; each (region, version) row fills the other columns."""
    with open(model_set_dir / SOURCE_PROVENANCE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for region, version in rows:
            values = {
                "region": region,
                "mask_version_trained_on": version,
                "source_run": f"run_{region}",
                "training_ckpt": "epoch=80-step=1000.ckpt",
                "original_lightning_version": "version_0",
            }
            writer.writerow([values[column] for column in header])


def _rows_for(names, version: str) -> list[tuple[str, str]]:
    """One (name, version) row per name."""
    return [(name, version) for name in names]


def _rosette_rows() -> list[tuple[str, str]]:
    """56 rows: canonical_corrected_26_1 for the 9 rosette names, canonical_25 for the 47 others."""
    return [(name, CORRECTED_26_1 if name in ROSETTE_CORRECTED_NAMES else CANONICAL_25) for name in MODEL_NAMES]


def _resolve(model_set_dir) -> dict:
    return resolve_region_mask_versions(str(model_set_dir))


class TestProvenanceRow:
    """REQ PROVENANCE row: no model_set.json; the row of N names known Mask version V -> N maps to V, PROVENANCE.csv."""

    @pytest.mark.parametrize("version", MASK_VERSIONS)
    def test_every_named_region_maps_to_row_version(self, tmp_path, version):
        model_set_dir = _model_set(tmp_path, UNKNOWN_FOLDER_NAME)
        _write_provenance(model_set_dir, _rows_for(MODEL_NAMES, version))
        result = _resolve(model_set_dir)
        assert set(result) == set(MODEL_NAMES)
        assert {(entry.status, entry.version, entry.source) for entry in result.values()} == {
            (MASK_RESOLVED, version, SOURCE_PROVENANCE)
        }

    @pytest.mark.parametrize("version", MASK_VERSIONS)
    def test_single_row_maps_its_region_model_name(self, tmp_path, version):
        model_set_dir = _model_set(tmp_path, UNKNOWN_FOLDER_NAME)
        _write_provenance(model_set_dir, [(REGION_MODEL_NAME, version)])
        entry = _resolve(model_set_dir)[REGION_MODEL_NAME]
        assert (entry.status, entry.version, entry.source) == (MASK_RESOLVED, version, SOURCE_PROVENANCE)

    def test_repeated_identical_row_maps_once(self, tmp_path):
        model_set_dir = _model_set(tmp_path, UNKNOWN_FOLDER_NAME)
        _write_provenance(model_set_dir, [(REGION_MODEL_NAME, CANONICAL_25), (REGION_MODEL_NAME, CANONICAL_25)])
        entry = _resolve(model_set_dir)[REGION_MODEL_NAME]
        assert (entry.status, entry.version, entry.source) == (MASK_RESOLVED, CANONICAL_25, SOURCE_PROVENANCE)


class TestProvenanceOutranksFolderName:
    """REQ PROVENANCE over folder name: folder named after known F, row of N names another known V -> N maps to V."""

    def test_rosette_mixed_set(self, tmp_path):
        """Regression of DEF-MODELVER-BDRABCZUK-291232AA949D on the rosette shape: the folder name
        canonical_corrected_26_1 is not the Mask version of 47 of its 56 region model names."""
        model_set_dir = _model_set(tmp_path, CORRECTED_26_1)
        _write_provenance(model_set_dir, _rosette_rows())
        result = _resolve(model_set_dir)
        assert {name: (entry.status, entry.version, entry.source) for name, entry in result.items()} == {
            name: (MASK_RESOLVED, version, SOURCE_PROVENANCE) for name, version in _rosette_rows()
        }

    @pytest.mark.parametrize("folder_version", MASK_VERSIONS)
    def test_row_version_wins_over_folder_name(self, tmp_path, folder_version):
        row_version = _other_version(folder_version)
        model_set_dir = _model_set(tmp_path, folder_version)
        _write_provenance(model_set_dir, [(REGION_MODEL_NAME, row_version)])
        entry = _resolve(model_set_dir)[REGION_MODEL_NAME]
        assert (entry.status, entry.version, entry.source) == (MASK_RESOLVED, row_version, SOURCE_PROVENANCE)


class TestMissingRowTakesFolderName:
    """REQ no row: no model_set.json, no row for N, folder name known F -> N maps to F with source folder name."""

    @pytest.mark.parametrize("folder_version", MASK_VERSIONS)
    def test_region_without_row_takes_folder_name(self, tmp_path, folder_version):
        model_set_dir = _model_set(tmp_path, folder_version)
        other_names = [name for name in MODEL_NAMES if name != REGION_MODEL_NAME]
        _write_provenance(model_set_dir, _rows_for(other_names, _other_version(folder_version)))
        entry = _resolve(model_set_dir)[REGION_MODEL_NAME]
        assert (entry.status, entry.version, entry.source) == (MASK_RESOLVED, folder_version, MASK_SOURCE_FOLDER_NAME)

    def test_region_value_in_other_case_is_no_row(self, tmp_path):
        model_set_dir = _model_set(tmp_path, CANONICAL_25)
        _write_provenance(model_set_dir, [(REGION_MODEL_NAME.lower(), CORRECTED_26_1)])
        entry = _resolve(model_set_dir)[REGION_MODEL_NAME]
        assert (entry.status, entry.version, entry.source) == (MASK_RESOLVED, CANONICAL_25, MASK_SOURCE_FOLDER_NAME)


class TestEmptyValueTakesFolderName:
    """REQ empty value: no model_set.json, empty value in the row of N, folder name known F -> N maps to F."""

    @pytest.mark.parametrize("folder_version", MASK_VERSIONS)
    def test_empty_value_takes_folder_name(self, tmp_path, folder_version):
        model_set_dir = _model_set(tmp_path, folder_version)
        _write_provenance(model_set_dir, [(REGION_MODEL_NAME, "")])
        entry = _resolve(model_set_dir)[REGION_MODEL_NAME]
        assert (entry.status, entry.version, entry.source) == (MASK_RESOLVED, folder_version, MASK_SOURCE_FOLDER_NAME)


class TestMissingRowUnknownFolderName:
    """REQ no row, no known folder name: no model_set.json, no row for N, folder name not known -> N unresolved."""

    def test_region_without_row_unresolved(self, tmp_path):
        model_set_dir = _model_set(tmp_path, UNKNOWN_FOLDER_NAME)
        other_names = [name for name in MODEL_NAMES if name != REGION_MODEL_NAME]
        _write_provenance(model_set_dir, _rows_for(other_names, CANONICAL_25))
        entry = _resolve(model_set_dir)[REGION_MODEL_NAME]
        assert (entry.status, entry.version, entry.source) == (MASK_UNRESOLVED, None, None)


class TestUnknownProvenanceValue:
    """REQ unknown value: no model_set.json, the row of N names a value outside MASK_VERSIONS -> N unresolved."""

    @pytest.mark.parametrize("folder_name", (UNKNOWN_FOLDER_NAME, *MASK_VERSIONS))
    def test_unknown_value_unresolved_naming_it(self, tmp_path, folder_name):
        model_set_dir = _model_set(tmp_path, folder_name)
        _write_provenance(model_set_dir, [(REGION_MODEL_NAME, UNKNOWN_VERSION)])
        entry = _resolve(model_set_dir)[REGION_MODEL_NAME]
        assert (entry.status, entry.version, entry.source) == (MASK_UNRESOLVED, None, None)
        assert UNKNOWN_VERSION in entry.named


class TestMissingColumn:
    """REQ missing column: no model_set.json, header lacks region or mask_version_trained_on -> every name
    unresolved with a reason naming the missing column."""

    @pytest.mark.parametrize("missing", ("region", "mask_version_trained_on"))
    def test_missing_column_leaves_every_name_unresolved(self, tmp_path, missing):
        model_set_dir = _model_set(tmp_path, CORRECTED_26_1)
        header = tuple(column for column in PROVENANCE_HEADER if column != missing)
        _write_provenance(model_set_dir, _rosette_rows(), header=header)
        result = _resolve(model_set_dir)
        assert set(result) == set(MODEL_NAMES)
        assert {entry.status for entry in result.values()} == {MASK_UNRESOLVED}
        assert all(entry.reason is not None and missing in entry.reason for entry in result.values())

    def test_empty_file_names_both_columns(self, tmp_path):
        model_set_dir = _model_set(tmp_path, CORRECTED_26_1)
        (model_set_dir / SOURCE_PROVENANCE).write_text("", encoding="utf-8")
        result = _resolve(model_set_dir)
        assert {entry.status for entry in result.values()} == {MASK_UNRESOLVED}
        for entry in result.values():
            assert entry.reason is not None
            assert "region" in entry.reason
            assert "mask_version_trained_on" in entry.reason


class TestDuplicateRows:
    """REQ duplicate rows: no model_set.json, two rows of N name different known Mask versions -> N conflicting."""

    @pytest.mark.parametrize("folder_name", (UNKNOWN_FOLDER_NAME, *MASK_VERSIONS))
    def test_two_rows_with_different_versions_conflict(self, tmp_path, folder_name):
        model_set_dir = _model_set(tmp_path, folder_name)
        _write_provenance(model_set_dir, [(REGION_MODEL_NAME, CANONICAL_25), (REGION_MODEL_NAME, CORRECTED_26_1)])
        entry = _resolve(model_set_dir)[REGION_MODEL_NAME]
        assert (entry.status, entry.version, entry.source) == (MASK_CONFLICT, None, None)
        assert {CANONICAL_25, CORRECTED_26_1} <= set(entry.named)


class TestModelSetOutranksProvenance:
    """REQ model_set.json over PROVENANCE.csv: folder name not known, marker masks known V -> every name maps to V
    with source model_set.json, whatever PROVENANCE.csv holds."""

    @pytest.mark.parametrize("version", MASK_VERSIONS)
    def test_marker_wins_over_provenance_rows(self, tmp_path, version):
        model_set_dir = _model_set(tmp_path, UNKNOWN_FOLDER_NAME, marker_masks=version)
        _write_provenance(model_set_dir, _rows_for(MODEL_NAMES, _other_version(version)))
        result = _resolve(model_set_dir)
        assert set(result) == set(MODEL_NAMES)
        assert {(entry.status, entry.version, entry.source) for entry in result.values()} == {
            (MASK_RESOLVED, version, MASK_SOURCE_MODEL_SET)
        }


class TestUnreadableProvenance:
    """REQ read failure: no model_set.json, reading PROVENANCE.csv fails -> every name unresolved, reason naming it."""

    def test_invalid_utf8_leaves_every_name_unresolved(self, tmp_path):
        model_set_dir = _model_set(tmp_path, CORRECTED_26_1)
        (model_set_dir / SOURCE_PROVENANCE).write_bytes(b"region,mask_version_trained_on\n\xff\xfe\x80,\x81\n")
        result = _resolve(model_set_dir)
        assert {entry.status for entry in result.values()} == {MASK_UNRESOLVED}
        assert all(entry.reason is not None and SOURCE_PROVENANCE in entry.reason for entry in result.values())

    def test_directory_named_provenance_leaves_every_name_unresolved(self, tmp_path):
        model_set_dir = _model_set(tmp_path, CORRECTED_26_1)
        (model_set_dir / SOURCE_PROVENANCE).mkdir()
        result = _resolve(model_set_dir)
        assert {entry.status for entry in result.values()} == {MASK_UNRESOLVED}
        assert all(entry.reason is not None and SOURCE_PROVENANCE in entry.reason for entry in result.values())


@pytest.fixture
def error_records(caplog):
    """caplog's handler put on version_options.logger itself (propagation off meanwhile, so no record twice)."""
    logger = version_options.logger
    level, propagate = logger.level, logger.propagate
    logger.addHandler(caplog.handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    yield caplog
    logger.removeHandler(caplog.handler)
    logger.setLevel(level)
    logger.propagate = propagate


def _assert_one_error_naming(capture, model_set_dir) -> None:
    """Exactly one ERROR record, whose message names the path of <model_set_dir>/PROVENANCE.csv."""
    errors = [record for record in capture.records if record.levelno == logging.ERROR]
    assert len(errors) == 1, [record.getMessage() for record in errors]
    assert str(model_set_dir / SOURCE_PROVENANCE) in errors[0].getMessage()


class TestUnreadableProvenanceErrorRecord:
    """REQ read failure ERROR record: no model_set.json, reading PROVENANCE.csv fails -> one ERROR record naming the
    path of that PROVENANCE.csv."""

    def test_invalid_utf8_gives_one_error_record_naming_path(self, tmp_path, error_records):
        model_set_dir = _model_set(tmp_path, CORRECTED_26_1)
        (model_set_dir / SOURCE_PROVENANCE).write_bytes(b"region,mask_version_trained_on\n\xff\xfe\x80,\x81\n")
        _resolve(model_set_dir)
        _assert_one_error_naming(error_records, model_set_dir)

    def test_directory_named_provenance_gives_one_error_record_naming_path(self, tmp_path, error_records):
        model_set_dir = _model_set(tmp_path, UNKNOWN_FOLDER_NAME)
        (model_set_dir / SOURCE_PROVENANCE).mkdir()
        _resolve(model_set_dir)
        _assert_one_error_naming(error_records, model_set_dir)


class TestMissingColumnErrorRecord:
    """REQ missing column ERROR record: no model_set.json, header lacks region or mask_version_trained_on -> one
    ERROR record naming the path of that PROVENANCE.csv."""

    @pytest.mark.parametrize("missing", ("region", "mask_version_trained_on"))
    def test_missing_column_gives_one_error_record_naming_path(self, tmp_path, error_records, missing):
        model_set_dir = _model_set(tmp_path, CORRECTED_26_1)
        header = tuple(column for column in PROVENANCE_HEADER if column != missing)
        _write_provenance(model_set_dir, _rosette_rows(), header=header)
        _resolve(model_set_dir)
        _assert_one_error_naming(error_records, model_set_dir)

    def test_empty_file_gives_one_error_record_naming_path(self, tmp_path, error_records):
        model_set_dir = _model_set(tmp_path, UNKNOWN_FOLDER_NAME)
        (model_set_dir / SOURCE_PROVENANCE).write_text("", encoding="utf-8")
        _resolve(model_set_dir)
        _assert_one_error_naming(error_records, model_set_dir)


ROW_VALUES = st.sampled_from((*MASK_VERSIONS, "", UNKNOWN_VERSION))
ROWS = st.lists(st.tuples(st.sampled_from(MODEL_NAMES), ROW_VALUES), min_size=1, max_size=12)


class TestRowOrder:
    """REQ row order: two model sets without model_set.json differing only in PROVENANCE.csv row order -> equal
    results (fixed read order)."""

    @given(data=st.data(), rows=ROWS)
    @settings(max_examples=30)
    def test_row_order_does_not_change_the_map(self, tmp_path_factory, data, rows):
        shuffled = data.draw(st.permutations(rows))
        first = _model_set(tmp_path_factory.mktemp("first"), UNKNOWN_FOLDER_NAME)
        second = _model_set(tmp_path_factory.mktemp("second"), UNKNOWN_FOLDER_NAME)
        _write_provenance(first, rows)
        _write_provenance(second, shuffled)
        assert _resolve(first) == _resolve(second)

    def test_reversed_rosette_rows_give_equal_maps(self, tmp_path):
        first = _model_set(tmp_path / "a", CORRECTED_26_1)
        second = _model_set(tmp_path / "b", CORRECTED_26_1)
        _write_provenance(first, _rosette_rows())
        _write_provenance(second, list(reversed(_rosette_rows())))
        assert _resolve(first) == _resolve(second)
