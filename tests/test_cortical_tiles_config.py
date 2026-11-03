import json
import os

import pytest

from champollion_pipeline.utils.cortical_tiles_config import (
    CorticalTilesConfigFactory,
    compute_versioned_crops_dir,
    versioned_crops_exist,
)


class FakeArgs:
    def __init__(self, masks=None):
        self.masks = masks


class TestCorticalTilesConfigFactory:
    def test_from_args_explicit_version(self):
        args = FakeArgs(masks="canonical_corrected_26_1")
        cfg = CorticalTilesConfigFactory.from_args(args)
        assert cfg.masks_version == "canonical_corrected_26_1"
        assert cfg.out_voxel_size == 2.0

    def test_from_args_defaults_to_canonical_25(self):
        args = FakeArgs(masks=None)
        cfg = CorticalTilesConfigFactory.from_args(args)
        assert cfg.masks_version == "canonical_25"

    def test_from_pipeline_json_reads_version(self, tmp_path):
        cfg_file = tmp_path / "pipeline_loop_2mm.json"
        cfg_file.write_text(json.dumps({"masks_version": "canonical_corrected_26_1"}))
        cfg = CorticalTilesConfigFactory.from_pipeline_json(str(cfg_file))
        assert cfg is not None
        assert cfg.masks_version == "canonical_corrected_26_1"

    def test_from_pipeline_json_missing_field_defaults(self, tmp_path):
        cfg_file = tmp_path / "pipeline_loop_2mm.json"
        cfg_file.write_text(json.dumps({}))
        cfg = CorticalTilesConfigFactory.from_pipeline_json(str(cfg_file))
        assert cfg is not None
        assert cfg.masks_version == "canonical_25"

    def test_from_pipeline_json_absent_file_returns_none(self, tmp_path):
        cfg = CorticalTilesConfigFactory.from_pipeline_json(str(tmp_path / "missing.json"))
        assert cfg is None


class TestVersionedCropsExist:
    def test_returns_false_when_dir_absent(self, tmp_path):
        assert not versioned_crops_exist(str(tmp_path), "canonical_25", 2.0)

    def test_returns_false_when_dir_empty(self, tmp_path):
        crops_dir = tmp_path / "crops" / "canonical_25" / "2mm"
        crops_dir.mkdir(parents=True)
        assert not versioned_crops_exist(str(tmp_path), "canonical_25", 2.0)

    def test_returns_true_when_dir_has_content(self, tmp_path):
        crops_dir = tmp_path / "crops" / "canonical_25" / "2mm"
        crops_dir.mkdir(parents=True)
        (crops_dir / "SC-sylv_left").mkdir()
        assert versioned_crops_exist(str(tmp_path), "canonical_25", 2.0)

    def test_different_version_not_detected(self, tmp_path):
        crops_dir = tmp_path / "crops" / "canonical_25" / "2mm"
        crops_dir.mkdir(parents=True)
        (crops_dir / "SC-sylv_left").mkdir()
        assert not versioned_crops_exist(str(tmp_path), "canonical_corrected_26_1", 2.0)

    def test_voxel_size_float_formatting(self, tmp_path):
        crops_dir = tmp_path / "crops" / "canonical_25" / "2mm"
        crops_dir.mkdir(parents=True)
        (crops_dir / "SC-sylv_left").mkdir()
        assert versioned_crops_exist(str(tmp_path), "canonical_25", 2.0)
        assert not versioned_crops_exist(str(tmp_path), "canonical_25", 1.0)


class TestComputeVersionedCropsDir:
    """REQ-PATTERNS-BDRABCZUK-66BCD9D856F6 and -2ED8EA782B0F: <D>/crops/<M>/<S>mm, pure."""

    @pytest.mark.parametrize(("voxel_size", "vox_dir"), [(2.0, "2mm"), (1.0, "1mm"), (3, "3mm")])
    def test_returns_versioned_crops_path(self, tmp_path, voxel_size, vox_dir):
        derivatives = str(tmp_path / "derivatives")
        expected = os.path.join(derivatives, "crops", "masks_v_test", vox_dir)

        assert compute_versioned_crops_dir(derivatives, "masks_v_test", voxel_size) == expected
        assert not os.path.exists(expected)

    def test_default_voxel_size_is_2mm(self, tmp_path):
        derivatives = str(tmp_path / "derivatives")

        assert compute_versioned_crops_dir(derivatives, "canonical_25") == os.path.join(
            derivatives, "crops", "canonical_25", "2mm"
        )
