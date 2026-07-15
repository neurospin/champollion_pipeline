import json

from champollion_pipeline.utils.cortical_tiles_config import (
    CorticalTilesConfigFactory,
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
