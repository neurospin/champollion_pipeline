#!/usr/bin/env python3
"""
Per-region Mask version of a local model set, part 1 (TASK-MODELVER-BDRABCZUK-B9820AF55783,
DEF-MODELVER-BDRABCZUK-291232AA949D).

resolve_region_mask_versions(models_path) gives one entry per region model name (the 56 names of
compute_region_model_names(DEFAULT_REGIONS)): the Mask version and its source, read from the
model_set.json masks value, else from the folder name when it is a known Mask version (a value of
MASK_VERSIONS). A version outside MASK_VERSIONS, or no source, is unresolved; model_set.json and the
folder name naming two different known Mask versions is a conflict naming both.

Pure function: no model, no network; only tmp_path model set folders.
"""

import json

import pytest

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
UNKNOWN_FOLDER_NAME = "my_models"
UNKNOWN_VERSION = "canonical_99"
CANONICAL_25 = "canonical_25"
CORRECTED_26_1 = "canonical_corrected_26_1"


def _model_set(tmp_path, folder_name: str, marker_masks: str | None = None) -> str:
    """A local model set folder named <folder_name>; with model_set.json naming <marker_masks> when given."""
    path = tmp_path / "models" / folder_name
    path.mkdir(parents=True)
    if marker_masks is not None:
        (path / "model_set.json").write_text(
            json.dumps({"model_version": marker_masks, "masks": marker_masks}), encoding="utf-8"
        )
    return str(path)


def _assert_every_name(result: dict, status: str) -> None:
    """The result has one entry per region model name (56), each with <status>."""
    assert len(MODEL_NAMES) == 56
    assert set(result) == set(MODEL_NAMES)
    assert {name: entry.status for name, entry in result.items()} == {name: status for name in MODEL_NAMES}


class TestModelSetSource:
    """REQ-MODELVER-BDRABCZUK-0935BB3FC68F: model_set.json masks V (known), folder name not another known version."""

    @pytest.mark.parametrize("version", MASK_VERSIONS)
    def test_maps_every_model_name_to_marker_masks(self, tmp_path, version):
        result = resolve_region_mask_versions(_model_set(tmp_path, UNKNOWN_FOLDER_NAME, marker_masks=version))
        _assert_every_name(result, MASK_RESOLVED)
        assert {(entry.version, entry.source) for entry in result.values()} == {(version, MASK_SOURCE_MODEL_SET)}

    @pytest.mark.parametrize("version", MASK_VERSIONS)
    def test_marker_wins_when_folder_names_same_version(self, tmp_path, version):
        result = resolve_region_mask_versions(_model_set(tmp_path, version, marker_masks=version))
        _assert_every_name(result, MASK_RESOLVED)
        assert {(entry.version, entry.source) for entry in result.values()} == {(version, MASK_SOURCE_MODEL_SET)}


class TestFolderNameSource:
    """REQ-MODELVER-BDRABCZUK-FCAF0646178E: no model_set.json, folder name a known Mask version V."""

    @pytest.mark.parametrize("version", MASK_VERSIONS)
    def test_maps_every_model_name_to_folder_name(self, tmp_path, version):
        result = resolve_region_mask_versions(_model_set(tmp_path, version))
        _assert_every_name(result, MASK_RESOLVED)
        assert {(entry.version, entry.source) for entry in result.values()} == {(version, MASK_SOURCE_FOLDER_NAME)}

    def test_trailing_slash_folder_name(self, tmp_path):
        result = resolve_region_mask_versions(_model_set(tmp_path, CORRECTED_26_1) + "/")
        _assert_every_name(result, MASK_RESOLVED)
        assert {entry.version for entry in result.values()} == {CORRECTED_26_1}

    def test_defect_folder_maps_no_model_name_to_canonical_25(self, tmp_path):
        """Regression of DEF-MODELVER-BDRABCZUK-291232AA949D: the rosette canonical_corrected_26_1 set had no
        model_set.json and was cropped with canonical_25."""
        result = resolve_region_mask_versions(_model_set(tmp_path, CORRECTED_26_1))
        assert [name for name, entry in result.items() if entry.version == CANONICAL_25] == []
        assert len(result) == 56


class TestNoSource:
    """REQ-MODELVER-BDRABCZUK-31EE133BE952: no model_set.json, folder name not a known Mask version."""

    def test_every_model_name_unresolved(self, tmp_path):
        result = resolve_region_mask_versions(_model_set(tmp_path, UNKNOWN_FOLDER_NAME))
        _assert_every_name(result, MASK_UNRESOLVED)

    def test_unresolved_entry_names_no_version(self, tmp_path):
        result = resolve_region_mask_versions(_model_set(tmp_path, UNKNOWN_FOLDER_NAME))
        assert {(entry.version, entry.source, entry.named) for entry in result.values()} == {(None, None, ())}


class TestUnknownMarkerMasks:
    """REQ-MODELVER-BDRABCZUK-8B625B8917DB: model_set.json masks value not a known Mask version."""

    def test_unknown_marker_masks_unresolved(self, tmp_path):
        result = resolve_region_mask_versions(_model_set(tmp_path, UNKNOWN_FOLDER_NAME, marker_masks=UNKNOWN_VERSION))
        _assert_every_name(result, MASK_UNRESOLVED)
        assert {entry.version for entry in result.values()} == {None}
        assert all(UNKNOWN_VERSION in entry.named for entry in result.values())

    @pytest.mark.parametrize("folder_version", MASK_VERSIONS)
    def test_unknown_marker_masks_unresolved_beside_known_folder_name(self, tmp_path, folder_version):
        result = resolve_region_mask_versions(_model_set(tmp_path, folder_version, marker_masks=UNKNOWN_VERSION))
        _assert_every_name(result, MASK_UNRESOLVED)
        assert {entry.version for entry in result.values()} == {None}
        assert all(UNKNOWN_VERSION in entry.named for entry in result.values())


class TestSourceConflict:
    """REQ-MODELVER-BDRABCZUK-1E597A87208D: model_set.json masks and folder name two different known versions."""

    @pytest.mark.parametrize(("marker", "folder"), [(CANONICAL_25, CORRECTED_26_1), (CORRECTED_26_1, CANONICAL_25)])
    def test_every_model_name_conflicting(self, tmp_path, marker, folder):
        result = resolve_region_mask_versions(_model_set(tmp_path, folder, marker_masks=marker))
        _assert_every_name(result, MASK_CONFLICT)
        assert {entry.version for entry in result.values()} == {None}

    def test_conflict_names_both_versions(self, tmp_path):
        result = resolve_region_mask_versions(_model_set(tmp_path, CORRECTED_26_1, marker_masks=CANONICAL_25))
        assert {frozenset(entry.named) for entry in result.values()} == {frozenset({CANONICAL_25, CORRECTED_26_1})}


class TestKnownVersionsFromMaskVersions:
    """REQ-MODELVER-BDRABCZUK-D09D0B5FC25B: known Mask versions are exactly MASK_VERSIONS."""

    def test_added_mask_version_resolves(self, tmp_path, monkeypatch):
        monkeypatch.setattr(version_options, "MASK_VERSIONS", (*MASK_VERSIONS, UNKNOWN_VERSION))
        result = resolve_region_mask_versions(_model_set(tmp_path, UNKNOWN_VERSION))
        _assert_every_name(result, MASK_RESOLVED)
        assert {entry.version for entry in result.values()} == {UNKNOWN_VERSION}

    def test_removed_mask_version_unresolved(self, tmp_path, monkeypatch):
        monkeypatch.setattr(version_options, "MASK_VERSIONS", (CORRECTED_26_1,))
        result = resolve_region_mask_versions(_model_set(tmp_path, CANONICAL_25))
        _assert_every_name(result, MASK_UNRESOLVED)
