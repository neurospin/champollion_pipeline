#!/usr/bin/env python3
"""
Stage 6 of main.py without UMAP reference data (USER ANSWER 2026-11-05, "Skip UMAP only"):

- one WARNING record naming UMAP and --reference-data-path (REQ-INPUTKIND-BDRABCZUK-5B93C6EB8F51);
- the sulcal graph snapshots still render (REQ-INPUTKIND-BDRABCZUK-ECF94849D2F9);
- the cortical tiles snapshots still render (REQ-INPUTKIND-BDRABCZUK-895E39237C6F);
- no UMAP snapshot renders (REQ-INPUTKIND-BDRABCZUK-32FC84206D1F);
- the stage returns 0, not partial (REQ-INPUTKIND-BDRABCZUK-72BF928F4F71).

GenerateSnapshotsStage runs the real GenerateSnapshots script on a temporary tree (one Morphologist
subject, one completed crop region, one embeddings CSV, no reference data); only the Anatomist
renderers and the UMAP renderer are patched. main.py lives at the repo root, so it is loaded by file
path under a private module name.
"""

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.main_stage_helpers import load_main_module
from tests.test_generate_snapshots import (
    _RENDER_SULCAL,
    _RENDER_TILES,
    _embeddings_csv,
    _morphologist_subject,
    _write_snapshot,
)

_RENDER_UMAP = "champollion_pipeline.generate_snapshots.generate_umap_snapshot"
SUBJECT = "sub-01"


@pytest.fixture(scope="module")
def main_module():
    return load_main_module("_champollion_main_snapshots_skip_no_reference_under_test")


def _config(main_module, temp_dir):
    """PipelineConfig of a stage 6 run with sulcal, tiles and embeddings inputs on disk and no reference data."""
    morph = os.path.join(temp_dir, "morphologist")
    _morphologist_subject(morph, SUBJECT)
    tiles_root = Path(temp_dir) / "cortical_tiles"
    mask_dir = tiles_root / "crops" / "2mm" / "S.Or." / "mask"
    mask_dir.mkdir(parents=True)
    (mask_dir / "Lmask_skeleton.nii.gz").touch()
    embeddings = os.path.join(temp_dir, "embeddings")
    _embeddings_csv(embeddings, "S.Or.", "left", [(SUBJECT, 0.1, 0.2)])

    config = main_module.PipelineConfig(models_path="/models", outputs_path=os.path.join(temp_dir, "outputs"))
    dataset = config.dataset
    dataset.datasets_root = ""  # noqa: V101
    dataset.snapshots_path = os.path.join(temp_dir, "snapshots")  # noqa: V101
    dataset.morphologist_graphs = morph  # noqa: V101
    dataset.crops_path = str(tiles_root)  # noqa: V101
    dataset.embeddings_path = embeddings  # noqa: V101
    dataset.reference_data_path = ""  # noqa: V101
    return config


def _render_tiles(crops, out, *args, **kwargs):
    """Tiles render stand-in writing the left hemisphere file, the one hemisphere with a completed region."""
    return [_write_snapshot(crops, out[:-4] + "_left.png")]


class _Run:
    """Outcome of one stage 6 run: the StageResult and the three patched renderers."""

    def __init__(self, result, sulcal, tiles, umap):
        self.result = result
        self.sulcal = sulcal
        self.tiles = tiles
        self.umap = umap


def _run_stage(main_module, config):
    """Run GenerateSnapshotsStage.execute() on the champollion_pipeline logger with the renderers patched."""
    logger = logging.getLogger("champollion_pipeline")
    with (
        patch(_RENDER_SULCAL, side_effect=_write_snapshot) as sulcal,
        patch(_RENDER_TILES, side_effect=_render_tiles) as tiles,
        patch(_RENDER_UMAP, return_value=[]) as umap,
    ):
        stage = main_module.GenerateSnapshotsStage("generate_snapshots", config, logger)
        result = stage.execute()
    return _Run(result, sulcal, tiles, umap)


@pytest.mark.unit
def test_warning_names_reference_data_path(main_module, temp_dir, pipeline_caplog):
    """REQ-INPUTKIND-BDRABCZUK-5B93C6EB8F51: exactly one WARNING naming UMAP and --reference-data-path."""
    _run_stage(main_module, _config(main_module, temp_dir))

    naming = [
        record
        for record in pipeline_caplog.records
        if record.levelno == logging.WARNING
        and "UMAP" in record.getMessage()
        and "--reference-data-path" in record.getMessage()
    ]
    assert len(naming) == 1, [record.getMessage() for record in pipeline_caplog.records]
    assert naming[0].name.startswith("champollion_pipeline"), naming[0].name


@pytest.mark.unit
def test_sulcal_graph_snapshots_rendered(main_module, temp_dir):
    """REQ-INPUTKIND-BDRABCZUK-ECF94849D2F9: the left and right sulcal graphs of the subject are rendered."""
    run = _run_stage(main_module, _config(main_module, temp_dir))

    graphs = sorted(Path(call.args[0]).name for call in run.sulcal.call_args_list)
    assert graphs == [f"Lsulci_{SUBJECT}.arg", f"Rsulci_{SUBJECT}.arg"]
    for call in run.sulcal.call_args_list:
        assert Path(call.args[1]).is_file(), call.args[1]


@pytest.mark.unit
def test_cortical_tiles_snapshots_rendered(main_module, temp_dir):
    """REQ-INPUTKIND-BDRABCZUK-895E39237C6F: the tiles renderer runs on the crops of dataset.crops_path."""
    config = _config(main_module, temp_dir)
    run = _run_stage(main_module, config)

    run.tiles.assert_called_once()
    crops_dir = Path(run.tiles.call_args.args[0])
    assert crops_dir == Path(config.dataset.crops_path) / "crops" / "2mm"


@pytest.mark.unit
def test_no_umap_snapshot_rendered(main_module, temp_dir):
    """REQ-INPUTKIND-BDRABCZUK-32FC84206D1F: an embeddings CSV on disk, no reference data: no UMAP render."""
    config = _config(main_module, temp_dir)
    run = _run_stage(main_module, config)

    run.umap.assert_not_called()
    assert list(Path(config.dataset.snapshots_path).glob("umap*.png")) == []


@pytest.mark.unit
def test_stage_returns_0_not_partial(main_module, temp_dir):
    """REQ-INPUTKIND-BDRABCZUK-72BF928F4F71: the stage returns return_code 0 and partial false."""
    run = _run_stage(main_module, _config(main_module, temp_dir))

    assert run.result.return_code == 0, run.result.message
    assert run.result.success
    assert not run.result.partial
