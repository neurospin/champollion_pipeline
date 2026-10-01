#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Default output location of generate_champollion_config.py (REQ-CFGLOC-01, -03, -05).

Without --output / --external-config, stage 3 must write its YAMLs under the
dataset's own derivatives tree,
``<D>/<dataset>/derivatives/champollion_V1/configs``, where ``<D>`` is the
parent of the crop path's ``<dataset>`` component, and must leave the
``--champollion_loc`` checkout untouched. With --output but no
--external-config, the localization YAML follows the --output configs root
(REQ-CFGLOC-05, superseding REQ-CFGLOC-02).

Every test passes an explicit tmp_path ``--champollion_loc`` and the autouse
fixture also redirects ``_DEFAULT_CHAMPOLLION_LOC``, so nothing is ever written
into the real external/champollion_V1 checkout (REQ-TESTISOL-01/02).
"""

from pathlib import Path

import pytest

from champollion_pipeline import generate_champollion_config as gcc
from champollion_pipeline.generate_champollion_config import GenerateChampollionConfig
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from tests.test_generate_champollion_config_internals import fake_execute_command, make_crop

DATASET = "TEST01"


@pytest.fixture(autouse=True)  # noqa: V103
def isolated_default_champollion_loc(tmp_path, monkeypatch):
    """Keep run() from ever touching the real champollion_V1 checkout."""
    monkeypatch.setattr(gcc, "_DEFAULT_CHAMPOLLION_LOC", str(tmp_path / "default_champollion_V1"))


@pytest.fixture
def layout(tmp_path):
    """Build <tmp>/data/TEST01/derivatives/.../2mm with one region (L+R) and return key paths."""
    dataset_parent = tmp_path / "data"
    crops = dataset_parent / DATASET / "derivatives" / DERIVATIVES_FOLDER / "crops" / "canonical_25" / "2mm"
    make_crop(crops, "S.C.-sylv.", side="L", shape=(1, 4, 5, 6))
    make_crop(crops, "S.C.-sylv.", side="R", shape=(1, 4, 5, 6))
    return {
        "crops": crops,
        "dataset_parent": dataset_parent,
        "configs_root": dataset_parent / DATASET / "derivatives" / "champollion_V1" / "configs",
        "champollion_loc": tmp_path / "champollion_V1",
    }


def _run_default(layout, *extra):
    """Run stage 3 without --external-config (and without --output unless passed in ``extra``)."""
    script = GenerateChampollionConfig()
    script.parse_args(
        [str(layout["crops"]), "--dataset", DATASET, "--champollion_loc", str(layout["champollion_loc"]), *extra]
    )
    script.execute_command = fake_execute_command
    return script.run()


@pytest.mark.unit
class TestDefaultDatasetYamlLocation:
    """REQ-CFGLOC-01: dataset YAMLs default to <D>/<dataset>/derivatives/champollion_V1/configs/dataset/<dataset>/."""

    def test_reference_yaml_written_under_derivatives_configs(self, layout):
        _run_default(layout)
        expected = layout["configs_root"] / "dataset" / DATASET / "reference.yaml"
        assert expected.is_file(), f"reference.yaml not written at default {expected}"

    def test_region_yamls_written_under_derivatives_configs(self, layout):
        _run_default(layout)
        dataset_dir = layout["configs_root"] / "dataset" / DATASET
        missing = [n for n in ("SC-sylv_left.yaml", "SC-sylv_right.yaml") if not (dataset_dir / n).is_file()]
        assert not missing, f"region YAMLs {missing} not written under default {dataset_dir}"


@pytest.mark.unit
class TestDefaultLocalizationYamlLocation:
    """REQ-CFGLOC-05: localization YAML defaults to <R>/dataset_localization/<localization>.yaml.

    <R> is --output when given, otherwise the derivatives configs root.
    """

    def test_local_yaml_written_under_derivatives_configs(self, layout):
        _run_default(layout)
        expected = layout["configs_root"] / "dataset_localization" / "local.yaml"
        assert expected.is_file(), f"local.yaml not written at default {expected}"
        assert f"dataset_folder: {layout['dataset_parent']}" in expected.read_text()

    def test_localization_name_selects_file_under_derivatives_configs(self, layout):
        _run_default(layout, "--localization", "jean-zay")
        expected = layout["configs_root"] / "dataset_localization" / "jean-zay.yaml"
        assert expected.is_file(), f"jean-zay.yaml not written at default {expected}"

    def test_local_yaml_follows_output_when_given(self, layout, tmp_path):
        out = tmp_path / "out"
        _run_default(layout, "--output", str(out))
        expected = out / "dataset_localization" / "local.yaml"
        assert expected.is_file(), f"local.yaml not written under --output root {expected}"
        assert f"dataset_folder: {layout['dataset_parent']}" in expected.read_text()
        stray = layout["configs_root"] / "dataset_localization"
        assert not stray.exists(), f"localization YAML also written at derivatives default {stray}"


@pytest.mark.unit
class TestDefaultRunLeavesChampollionLocUntouched:
    """REQ-CFGLOC-03: a default run creates nothing under --champollion_loc."""

    def test_default_run_creates_nothing_under_champollion_loc(self, layout):
        champollion_loc: Path = layout["champollion_loc"]
        champollion_loc.mkdir()
        _run_default(layout)
        created = sorted(str(p.relative_to(champollion_loc)) for p in champollion_loc.rglob("*"))
        assert created == [], f"default run created paths under --champollion_loc: {created}"
