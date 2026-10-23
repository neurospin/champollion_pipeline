#!/usr/bin/env python3
"""
Tests for moving DatasetConfig, PipelineConfig and ConfigLoader out of main.py
into champollion_pipeline.config, with save_to_yaml serialising through
dataclasses.asdict (REQ-REFACTOR-BDRABCZUK-B104FC70DD5F, -D1E056EE6D11,
-2636D42F82B1, -88F94B3FF991, -47FAA83A09A4).

The golden YAML texts below were written by ConfigLoader.save_to_yaml of
main.py at commit cc8d67c (hand-built dict, before the move), plus one
'log: null' line after log_to_console (PipelineConfig.log, REQ-RUNLOG-BDRABCZUK-4C2F09533A2D);
the serialiser must reproduce them byte for byte (REQ-REFACTOR-BDRABCZUK-679F9A49A534,
superseding -2636D42F82B1) so existing config files stay compatible.
dataset.hf_token is set in the custom case and must stay absent.

main.py lives at the repo root (not inside the installed package), so it is
loaded by file path under a private module name.
"""

import dataclasses
import importlib
import importlib.util
from pathlib import Path

import pytest
import yaml

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"
CONFIG_MODULE = "champollion_pipeline.config"
CONFIG_NAMES = ("DatasetConfig", "PipelineConfig", "ConfigLoader")

# Golden output of save_to_yaml at cc8d67c (+ "log: null") for PipelineConfig() with root_path="/opt/champollion".
GOLDEN_DEFAULT_YAML = """\
root_path: /opt/champollion
data_path: ''
models_path: ''
outputs_path: ''
champollion_v1_path: ''
stages:
  generate_morphologist_graphs: false
  run_cortical_tiles: false
  generate_champollion_config: false
  generate_embeddings: true
  put_together_embeddings: false
  generate_snapshots: false
dependencies:
  generate_morphologist_graphs: []
  run_cortical_tiles:
  - generate_morphologist_graphs
  generate_champollion_config:
  - run_cortical_tiles
  generate_embeddings:
  - generate_champollion_config
  put_together_embeddings:
  - generate_embeddings
  generate_snapshots:
  - put_together_embeddings
mode: sequential
n_workers: 0
worker_timeout: 7200
stop_on_error: true
verbose: false
log_level: INFO
log_dir: ''
log_to_file: true
log_to_console: true
log: null
dataset:
  name: example_dataset
  dataset_localization: local
  datasets_root: ''
  datasets:
  - example
  labels:
  - Sex
  input_path: ''
  morphologist_graphs: ''
  cortical_tiles_output: ''
  crops_path: ''
  embeddings_path: ''
  njobs: 22
  path_to_graph: t1mri/default_acquisition/default_analysis/folds/3.3/base
  path_sk_with_hull: t1mri/default_acquisition/default_analysis/segmentation/mesh
  sk_qc_path: ''
  masks_version: canonical_25
  skel_seed: null
  regions: []
  classifier_name: svm
  overwrite: false
  embeddings_only: false
  checkpoint_epoch: 80
  models_cache: null
  subsets:
  - full
  epochs:
  - null
  split: random
  cv: 5
  splits_basedir: null
  idx_region_evaluation: null
  short_name: eval
  hf_enabled: false
  hf_repo_id: null
  config_path: null
  cpu: false
  bids: false
  snapshots_path: ''
  reference_data_path: ''
  champollion_data_root: null
"""

# Golden output of save_to_yaml at commit cc8d67c (+ "log: null") for _custom_config() below.
GOLDEN_CUSTOM_YAML = """\
root_path: /opt/champollion
data_path: /data
models_path: /models
outputs_path: /outputs
champollion_v1_path: /v1
stages:
  generate_embeddings: false
  generate_snapshots: true
dependencies:
  generate_snapshots:
  - generate_embeddings
mode: parallel
n_workers: 7
worker_timeout: 99
stop_on_error: false
verbose: true
log_level: DEBUG
log_dir: /logs
log_to_file: false
log_to_console: false
log: null
dataset:
  name: ukb
  dataset_localization: neurospin
  datasets_root: /data/ukb
  datasets:
  - a
  - b
  labels:
  - Age
  input_path: /in
  morphologist_graphs: /graphs
  cortical_tiles_output: /tiles
  crops_path: /crops
  embeddings_path: /emb
  njobs: 3
  path_to_graph: g/path
  path_sk_with_hull: sk/path
  sk_qc_path: /qc.tsv
  masks_version: canonical_28
  skel_seed: 42
  regions:
  - S.C.-sylv._left
  classifier_name: logistic
  overwrite: true
  embeddings_only: true
  checkpoint_epoch: 12
  models_cache: /cache
  subsets:
  - train
  - val
  epochs:
  - null
  - 5
  split: custom
  cv: 3
  splits_basedir: /splits
  idx_region_evaluation: 2
  short_name: short
  hf_enabled: true
  hf_repo_id: neurospin/champollion
  config_path: /cfg.yaml
  cpu: true
  bids: true
  snapshots_path: /snap
  reference_data_path: /ref
  champollion_data_root: /root
"""


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_config_module_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def config_module():
    return importlib.import_module(CONFIG_MODULE)


@pytest.fixture(scope="module")
def main_module():
    return _load_main_module()


def _default_config(config):
    pipeline_config = config.PipelineConfig()
    pipeline_config.root_path = "/opt/champollion"
    return pipeline_config


# Every DatasetConfig field set to a non-default value (hf_token included: it must not be written).
_CUSTOM_DATASET_VALUES = {
    "name": "ukb",
    "dataset_localization": "neurospin",
    "datasets_root": "/data/ukb",
    "datasets": ["a", "b"],
    "labels": ["Age"],
    "input_path": "/in",
    "morphologist_graphs": "/graphs",
    "cortical_tiles_output": "/tiles",
    "crops_path": "/crops",
    "embeddings_path": "/emb",
    "regions": ["S.C.-sylv._left"],
    "njobs": 3,
    "path_to_graph": "g/path",
    "path_sk_with_hull": "sk/path",
    "sk_qc_path": "/qc.tsv",
    "masks_version": "canonical_28",
    "skel_seed": 42,
    "classifier_name": "logistic",
    "overwrite": True,
    "embeddings_only": True,
    "checkpoint_epoch": 12,
    "models_cache": "/cache",
    "subsets": ["train", "val"],
    "epochs": [None, 5],
    "split": "custom",
    "cv": 3,
    "splits_basedir": "/splits",
    "idx_region_evaluation": 2,
    "short_name": "short",
    "hf_enabled": True,
    "hf_repo_id": "neurospin/champollion",
    "hf_token": "hf_SECRET",
    "config_path": "/cfg.yaml",
    "cpu": True,
    "bids": True,
    "snapshots_path": "/snap",
    "reference_data_path": "/ref",
    "champollion_data_root": "/root",
}


def _custom_config(config):
    dataset_fields = {f.name for f in dataclasses.fields(config.DatasetConfig)}
    assert set(_CUSTOM_DATASET_VALUES) == dataset_fields, (
        "golden custom case no longer sets every DatasetConfig field: "
        f"{sorted(set(_CUSTOM_DATASET_VALUES) ^ dataset_fields)}"
    )
    return config.PipelineConfig(
        root_path="/opt/champollion",
        data_path="/data",
        models_path="/models",
        outputs_path="/outputs",
        champollion_v1_path="/v1",
        stages={"generate_embeddings": False, "generate_snapshots": True},
        dependencies={"generate_snapshots": ["generate_embeddings"]},
        mode="parallel",
        n_workers=7,
        worker_timeout=99,
        stop_on_error=False,
        verbose=True,
        log_level="DEBUG",
        log_dir="/logs",
        log_to_file=False,
        log_to_console=False,
        dataset=config.DatasetConfig(**_CUSTOM_DATASET_VALUES),
    )


@pytest.mark.unit
def test_config_module_public_names(config_module):
    """REQ-REFACTOR-BDRABCZUK-B104FC70DD5F: champollion_pipeline.config exposes the three config names."""
    missing = [name for name in CONFIG_NAMES if not hasattr(config_module, name)]
    assert missing == [], f"{CONFIG_MODULE} lacks {missing}"


@pytest.mark.unit
def test_main_reexports_config_objects(config_module, main_module):
    """REQ-REFACTOR-BDRABCZUK-D1E056EE6D11: main.py binds the config names to champollion_pipeline.config's objects."""
    not_same = [
        name
        for name in CONFIG_NAMES
        if getattr(main_module, name, None) is None or getattr(main_module, name) is not getattr(config_module, name, 0)
    ]
    assert not_same == [], f"main.py names not bound to the {CONFIG_MODULE} objects: {not_same}"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("build", "expected"),
    [(_default_config, GOLDEN_DEFAULT_YAML), (_custom_config, GOLDEN_CUSTOM_YAML)],
    ids=["default", "custom"],
)
def test_save_to_yaml_matches_golden_bytes(config_module, tmp_path, build, expected):
    """REQ-REFACTOR-BDRABCZUK-679F9A49A534: save_to_yaml writes cc8d67c's bytes plus one 'log: null' line."""
    output = tmp_path / "saved.yaml"
    config_module.ConfigLoader.save_to_yaml(build(config_module), str(output))
    assert output.read_bytes() == expected.encode("utf-8"), "save_to_yaml output differs from the cc8d67c golden YAML"


@pytest.mark.unit
def test_save_to_yaml_writes_dataset_subclass_field(config_module, tmp_path):
    """REQ-REFACTOR-BDRABCZUK-88F94B3FF991: a DatasetConfig subclass field is written to the dataset mapping."""
    extended = dataclasses.make_dataclass(
        "ExtendedDatasetConfig",
        [("extra_setting", str, dataclasses.field(default="extra-value"))],
        bases=(config_module.DatasetConfig,),
    )
    pipeline_config = config_module.PipelineConfig(dataset=extended())
    output = tmp_path / "saved.yaml"

    config_module.ConfigLoader.save_to_yaml(pipeline_config, str(output))
    written = yaml.safe_load(output.read_text())

    assert written["dataset"].get("extra_setting") == "extra-value", (
        f"dataset mapping lacks the subclass field extra_setting: {sorted(written['dataset'])}"
    )


@pytest.mark.unit
def test_pipeline_config_default_root_path_is_main_py_directory(config_module):
    """REQ-REFACTOR-BDRABCZUK-47FAA83A09A4: PipelineConfig().root_path is the directory holding main.py."""
    root_path = config_module.PipelineConfig().root_path
    assert Path(root_path).resolve() == MAIN_PY.parent, (
        f"PipelineConfig().root_path is {root_path!r}, expected {str(MAIN_PY.parent)!r}"
    )
