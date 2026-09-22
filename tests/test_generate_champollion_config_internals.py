#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the internal helpers and run() branches of
generate_champollion_config.py.

Crop sizes are read from small synthetic .npy arrays and .minf files; no real
cortical-tiles output is needed and no subprocess is ever spawned (the
ScriptBuilder ``execute_command`` shell-outs are replaced by in-process
equivalents).
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from champollion_pipeline import generate_champollion_config as gcc
from champollion_pipeline.generate_champollion_config import GenerateChampollionConfig, main
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER


def make_script(argv):
    """Return a GenerateChampollionConfig with parsed arguments."""
    script = GenerateChampollionConfig()
    script.parse_args(argv)
    return script


def fake_execute_command(cmd, shell=False):
    """In-process stand-in for the `mkdir -p` / `cp` shell-outs in run()."""
    if cmd[0] == "mkdir":
        os.makedirs(cmd[-1], exist_ok=True)
    elif cmd[0] == "cp":
        shutil.copy(cmd[1], cmd[2])
    return 0


def make_crop(crop_root, crop_name, side="L", shape=(1, 4, 5, 6), suffix="skeleton.npy"):
    """Create {crop_root}/{crop_name}/mask/{side}{suffix} holding an array."""
    mask_dir = Path(crop_root) / crop_name / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    np.save(mask_dir / f"{side}{suffix}", np.zeros(shape, dtype=np.int16))
    return mask_dir


class TestGetCropSize:
    """Test _get_crop_size."""

    def test_reads_4d_npy_shape(self, temp_dir):
        make_crop(temp_dir, "S.C.-sylv.", shape=(1, 11, 12, 13))
        script = make_script([temp_dir, "--dataset", "TEST01"])
        assert script._get_crop_size(os.path.join(temp_dir, "S.C.-sylv."), "L") == (11, 12, 13)

    def test_reads_5d_npy_shape(self, temp_dir):
        make_crop(temp_dir, "S.Or.", shape=(1, 7, 8, 9, 1))
        script = make_script([temp_dir, "--dataset", "TEST01"])
        assert script._get_crop_size(os.path.join(temp_dir, "S.Or."), "L") == (7, 8, 9)

    def test_falls_back_to_label_npy(self, temp_dir):
        make_crop(temp_dir, "S.Or.", shape=(1, 2, 3, 4), suffix="label.npy")
        script = make_script([temp_dir, "--dataset", "TEST01"])
        assert script._get_crop_size(os.path.join(temp_dir, "S.Or."), "L") == (2, 3, 4)

    def test_falls_back_to_distbottom_npy(self, temp_dir):
        make_crop(temp_dir, "S.Or.", shape=(1, 5, 5, 5), suffix="distbottom.npy")
        script = make_script([temp_dir, "--dataset", "TEST01"])
        assert script._get_crop_size(os.path.join(temp_dir, "S.Or."), "L") == (5, 5, 5)

    def test_skeleton_npy_takes_precedence_over_label_npy(self, temp_dir):
        make_crop(temp_dir, "S.Or.", shape=(1, 1, 1, 1), suffix="skeleton.npy")
        make_crop(temp_dir, "S.Or.", shape=(1, 9, 9, 9), suffix="label.npy")
        script = make_script([temp_dir, "--dataset", "TEST01"])
        assert script._get_crop_size(os.path.join(temp_dir, "S.Or."), "L") == (1, 1, 1)

    def test_reads_minf_when_no_npy_present(self, temp_dir):
        mask_dir = Path(temp_dir) / "S.T.s." / "mask"
        mask_dir.mkdir(parents=True)
        (mask_dir / "Lmask_cropped.nii.gz.minf").write_text(
            "attributes = {'sizeX': 21, 'sizeY': 22, 'sizeZ': 23}"
        )
        script = make_script([temp_dir, "--dataset", "TEST01"])
        assert script._get_crop_size(os.path.join(temp_dir, "S.T.s."), "L") == (21, 22, 23)

    def test_returns_none_when_nothing_found(self, temp_dir):
        (Path(temp_dir) / "empty" / "mask").mkdir(parents=True)
        script = make_script([temp_dir, "--dataset", "TEST01"])
        assert script._get_crop_size(os.path.join(temp_dir, "empty"), "L") is None

    def test_side_is_respected(self, temp_dir):
        make_crop(temp_dir, "S.Or.", side="R", shape=(1, 3, 3, 3))
        script = make_script([temp_dir, "--dataset", "TEST01"])
        crop_dir = os.path.join(temp_dir, "S.Or.")
        assert script._get_crop_size(crop_dir, "R") == (3, 3, 3)
        assert script._get_crop_size(crop_dir, "L") is None


class TestCreateDatasetConfigs:
    """Test _create_dataset_configs."""

    REF = (
        "dataset_name: REPLACE_DATASET\n"
        "crop: REPLACE_CROP_NAME\n"
        "side: REPLACE_SIDE\n"
        "input_size: (1, REPLACE_SIZEX, REPLACE_SIZEY, REPLACE_SIZEZ)\n"
    )

    def test_writes_one_yaml_per_side(self, temp_dir):
        crops = Path(temp_dir) / "crops"
        out = Path(temp_dir) / "out"
        out.mkdir()
        make_crop(crops, "S.C.-sylv.", side="L", shape=(1, 4, 5, 6))
        make_crop(crops, "S.C.-sylv.", side="R", shape=(1, 4, 5, 6))

        script = make_script([str(crops), "--dataset", "TEST01"])
        script._create_dataset_configs(str(crops), str(out), self.REF)

        assert (out / "SC-sylv_left.yaml").exists()
        assert (out / "SC-sylv_right.yaml").exists()

    def test_placeholders_are_substituted(self, temp_dir):
        crops = Path(temp_dir) / "crops"
        out = Path(temp_dir) / "out"
        out.mkdir()
        make_crop(crops, "S.Or.", side="L", shape=(1, 4, 5, 6))

        script = make_script([str(crops), "--dataset", "TEST01"])
        script._create_dataset_configs(str(crops), str(out), self.REF)

        content = (out / "SOr_left.yaml").read_text()
        assert "dataset_name: SOr_left" in content
        assert "crop: S.Or." in content
        assert "side: L" in content
        assert "input_size: (1, 4, 5, 6)" in content

    def test_non_directory_entries_are_ignored(self, temp_dir):
        crops = Path(temp_dir) / "crops"
        crops.mkdir()
        (crops / "stray_file.txt").write_text("not a crop")
        out = Path(temp_dir) / "out"
        out.mkdir()

        script = make_script([str(crops), "--dataset", "TEST01"])
        script._create_dataset_configs(str(crops), str(out), self.REF)
        assert list(out.iterdir()) == []

    def test_sulci_without_size_information_are_reported_as_skipped(self, temp_dir, capsys):
        crops = Path(temp_dir) / "crops"
        (crops / "S.Or." / "mask").mkdir(parents=True)
        out = Path(temp_dir) / "out"
        out.mkdir()

        script = make_script([str(crops), "--dataset", "TEST01"])
        script._create_dataset_configs(str(crops), str(out), self.REF)

        captured = capsys.readouterr().out
        assert "Skipped 2 sulci" in captured
        assert list(out.iterdir()) == []


class TestValidateInputs:
    """Test _validate_inputs."""

    def test_accepts_existing_crop_path(self, temp_dir):
        make_script([temp_dir, "--dataset", "TEST01"])._validate_inputs()

    def test_rejects_missing_crop_path(self, temp_dir):
        missing = os.path.join(temp_dir, "nope")
        with pytest.raises(ValueError, match="does not exist"):
            make_script([missing, "--dataset", "TEST01"])._validate_inputs()


class TestWriteLocalizationYaml:
    """Test _write_localization_yaml."""

    def test_creates_parent_directories(self, temp_dir):
        dest = os.path.join(temp_dir, "a", "b", "local.yaml")
        make_script([temp_dir, "--dataset", "TEST01"])._write_localization_yaml(dest, "/data")
        assert os.path.exists(dest)

    def test_content_uses_global_package_header(self, temp_dir):
        dest = os.path.join(temp_dir, "local.yaml")
        make_script([temp_dir, "--dataset", "TEST01"])._write_localization_yaml(dest, "/data/root")
        content = Path(dest).read_text()
        assert content.startswith("# @package _global_")
        assert "dataset_folder: /data/root" in content


@pytest.fixture
def crop_tree(temp_dir):
    """Build a pipeline-shaped crops tree containing one region, both sides."""
    crops = (
        Path(temp_dir)
        / "data"
        / "TEST01"
        / "derivatives"
        / DERIVATIVES_FOLDER
        / "crops"
        / "canonical_25"
        / "2mm"
    )
    make_crop(crops, "S.C.-sylv.", side="L", shape=(1, 4, 5, 6))
    make_crop(crops, "S.C.-sylv.", side="R", shape=(1, 4, 5, 6))
    return crops


class TestRun:
    """Test run() end-to-end with shell-outs replaced in process."""

    def _run(self, argv, output_root):
        script = make_script(argv)
        script.execute_command = fake_execute_command
        result = script.run()
        return script, result

    def test_returns_zero_and_writes_region_configs(self, crop_tree, temp_dir):
        out = Path(temp_dir) / "configs"
        _, result = self._run(
            [str(crop_tree), "--dataset", "TEST01", "--output", str(out)], out
        )
        assert result == 0
        dataset_dir = out / "dataset" / "TEST01"
        assert (dataset_dir / "reference.yaml").exists()
        assert (dataset_dir / "SC-sylv_left.yaml").exists()
        assert (dataset_dir / "SC-sylv_right.yaml").exists()

    def test_reference_yaml_gets_derivatives_path(self, crop_tree, temp_dir):
        out = Path(temp_dir) / "configs"
        self._run([str(crop_tree), "--dataset", "TEST01", "--output", str(out)], out)
        ref = (out / "dataset" / "TEST01" / "reference.yaml").read_text()
        assert f"TEST01/derivatives/{DERIVATIVES_FOLDER}/crops/canonical_25/2mm" in ref
        assert "TESTXX/crops/2mm" not in ref

    def test_masks_version_override_is_used(self, crop_tree, temp_dir):
        out = Path(temp_dir) / "configs"
        self._run(
            [str(crop_tree), "--dataset", "TEST01", "--output", str(out), "--masks", "canonical_99"],
            out,
        )
        ref = (out / "dataset" / "TEST01" / "reference.yaml").read_text()
        assert "crops/canonical_99/2mm" in ref

    def test_external_crops_uses_relative_crop_path(self, crop_tree, temp_dir):
        out = Path(temp_dir) / "configs"
        self._run(
            [str(crop_tree), "--dataset", "TEST01", "--output", str(out), "--external_crops"],
            out,
        )
        ref = (out / "dataset" / "TEST01" / "reference.yaml").read_text()
        expected = os.path.relpath(str(crop_tree), str(Path(temp_dir) / "data"))
        assert expected in ref

    def test_default_output_lands_in_champollion_loc(self, crop_tree, temp_dir):
        champollion_loc = Path(temp_dir) / "champollion_V1"
        self._run(
            [str(crop_tree), "--dataset", "TEST01", "--champollion_loc", str(champollion_loc)],
            champollion_loc,
        )
        assert (champollion_loc / "champollion" / "configs" / "dataset" / "TEST01" / "reference.yaml").exists()

    def test_localization_yaml_written_into_champollion_loc_by_default(self, crop_tree, temp_dir):
        champollion_loc = Path(temp_dir) / "champollion_V1"
        self._run(
            [str(crop_tree), "--dataset", "TEST01", "--champollion_loc", str(champollion_loc)],
            champollion_loc,
        )
        loc = champollion_loc / "champollion" / "configs" / "dataset_localization" / "local.yaml"
        assert loc.exists()
        assert f"dataset_folder: {Path(temp_dir) / 'data'}" in loc.read_text()

    def test_localization_name_is_configurable(self, crop_tree, temp_dir):
        champollion_loc = Path(temp_dir) / "champollion_V1"
        self._run(
            [
                str(crop_tree),
                "--dataset",
                "TEST01",
                "--champollion_loc",
                str(champollion_loc),
                "--localization",
                "jean-zay",
            ],
            champollion_loc,
        )
        loc_dir = champollion_loc / "champollion" / "configs" / "dataset_localization"
        assert (loc_dir / "jean-zay.yaml").exists()
        assert not (loc_dir / "local.yaml").exists()

    def test_external_config_file_path_is_used_verbatim(self, crop_tree, temp_dir):
        out = Path(temp_dir) / "configs"
        external = Path(temp_dir) / "external" / "my_local.yaml"
        self._run(
            [
                str(crop_tree),
                "--dataset",
                "TEST01",
                "--output",
                str(out),
                "--external-config",
                str(external),
            ],
            out,
        )
        assert external.exists()

    def test_external_config_directory_gets_standard_layout(self, crop_tree, temp_dir):
        out = Path(temp_dir) / "configs"
        external_dir = Path(temp_dir) / "external"
        external_dir.mkdir()
        self._run(
            [
                str(crop_tree),
                "--dataset",
                "TEST01",
                "--output",
                str(out),
                "--external-config",
                str(external_dir),
            ],
            out,
        )
        assert (external_dir / "dataset_localization" / "local.yaml").exists()

    def test_run_is_idempotent(self, crop_tree, temp_dir):
        out = Path(temp_dir) / "configs"
        argv = [str(crop_tree), "--dataset", "TEST01", "--output", str(out)]
        self._run(argv, out)
        first = (out / "dataset" / "TEST01" / "reference.yaml").read_text()
        self._run(argv, out)
        assert (out / "dataset" / "TEST01" / "reference.yaml").read_text() == first


class TestMain:
    """Test the main() entry point."""

    def test_main_builds_prints_and_runs(self, monkeypatch):
        calls = []

        class FakeScript:
            def build(self):
                calls.append("build")
                return self

            def print_args(self):
                calls.append("print_args")
                return self

            def run(self):
                calls.append("run")
                return 0

        monkeypatch.setattr(gcc, "GenerateChampollionConfig", FakeScript)
        assert main() == 0
        assert calls == ["build", "print_args", "run"]
