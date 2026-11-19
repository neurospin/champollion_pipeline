"""The default masks_version lives in one place: DEFAULT_MASKS_VERSION."""

import json
from argparse import Namespace
from pathlib import Path

from champollion_pipeline.derivatives_layout import DEFAULT_MASKS_VERSION
from champollion_pipeline.utils.cortical_tiles_config import CorticalTilesConfigFactory

SRC = Path(__file__).resolve().parent.parent / "src" / "champollion_pipeline"
ROUTED = (
    "run_cortical_tiles.py",
    "utils/cortical_tiles_config.py",
    "generate_champollion_config.py",
    "purge_subject.py",
    "prune_failed_subjects.py",
)


def test_pipeline_loop_json_equals_constant():
    config = json.loads((SRC / "resources" / "pipeline_loop_2mm.json").read_text())
    assert config["masks_version"] == DEFAULT_MASKS_VERSION


def test_no_masks_literal_outside_constant():
    for name in ROUTED:
        text = (SRC / name).read_text()
        assert f'"{DEFAULT_MASKS_VERSION}"' not in text, name
        assert f"default {DEFAULT_MASKS_VERSION})" not in text, name


def test_config_factory_fallbacks_use_constant(tmp_path):
    assert CorticalTilesConfigFactory.from_args(Namespace(masks=None)).masks_version == DEFAULT_MASKS_VERSION
    path = tmp_path / "cfg.json"
    path.write_text("{}")
    assert CorticalTilesConfigFactory.from_pipeline_json(str(path)).masks_version == DEFAULT_MASKS_VERSION
