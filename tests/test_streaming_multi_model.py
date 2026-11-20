"""Streaming mode over several models of one region (TASK-MULTIMODEL-BDRABCZUK-29169A15593C, epic -A02369CED8AE).

The resident consumer embeds every model of a ready scan and appends its row to the model's embeddings file, with the
layout rule of stage 4 applied per region: <region>/<model id>/full_embeddings.csv when the region holds two or more
models, or under DatasetConfig.per_model_folders for a non-empty model id; else <region>/full_embeddings.csv (single-
model files unchanged). The resume check skips a scan only when every model file holds its row. A model id named in
DatasetConfig.checkpoint_epochs ('<id>=N') embeds at epoch N; one matching no model is refused. Streaming combine
writes one combined folder per model id and its QC finds the combined files in the model id folders.

- REQ-MULTIMODEL-BDRABCZUK-C5F1844E557D test_several_models_write_model_id_files
- REQ-MULTIMODEL-BDRABCZUK-14895D3447FB test_per_model_folders_writes_model_id_file
- REQ-MULTIMODEL-BDRABCZUK-F46D1967C72F test_single_model_region_writes_region_file
- REQ-MULTIMODEL-BDRABCZUK-A6D7EFFFF9E8 test_empty_model_id_writes_region_file
- REQ-MULTIMODEL-BDRABCZUK-5B3E7E7514E7 test_mismatched_model_file_header_gives_failed
- REQ-MULTIMODEL-BDRABCZUK-E9E827D15E52 test_scan_in_each_model_file_skipped
- REQ-MULTIMODEL-BDRABCZUK-B7F8590BA2F5 test_missing_model_row_appended
- REQ-MULTIMODEL-BDRABCZUK-1555EF95C73A test_mapped_model_id_embeds_with_mapped_epoch
- REQ-MULTIMODEL-BDRABCZUK-B711454C6E28 test_unknown_epoch_model_id_raises
- REQ-MULTIMODEL-BDRABCZUK-ED9C4C1D3460 test_combine_qc_no_warning_for_model_id_folders
- REQ-MULTIMODEL-BDRABCZUK-AE440212A9C4 test_combine_writes_one_folder_per_model_id

The consumer runs a real EmbeddingEngine over tiny ConvNet models (as in tests/test_embedding_engine.py), so the tests
hold whatever the engine interface the consumer uses; only init_pipeline_process is replaced. Inputs are written before
the run and producers_done is set, so the first poll finds the scan ready and nothing sleeps.
DatasetConfig.per_model_folders and DatasetConfig.checkpoint_epochs are TASK-MULTIMODEL-BDRABCZUK-8AB92FCF3DDD's (T4).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from champollion_pipeline.config import DatasetConfig
from champollion_pipeline.derivatives_layout import compute_combined_embeddings_dir, compute_region_embeddings_dir
from champollion_pipeline.file_indexer import ScanId
from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings
from champollion_pipeline.streaming import combine, consumer, stages, worker
from champollion_pipeline.streaming.settings import StreamingSettings
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER
from tests.test_embedding_engine import HYDRA_CONFIG, _encoder_state
from tests.test_streaming_combine import records as combine_records  # noqa: F401 - a fixture

pytestmark = pytest.mark.unit  # noqa: V107

IN_SHAPE = (1, 8, 8, 8)
DIMS = 4
COLUMNS = ["ID", *(f"dim{i}" for i in range(1, DIMS + 1))]
EPOCH = 80
OTHER_EPOCH = 5
MASKS = "synthetic_masks"
MULTI = "SC-sylv_left"  # two models in the multi-model models path
FLAT = "FColl_right"  # one model
CROP_DIRS = {MULTI: "S.C.-sylv.", FLAT: "F.Coll."}
ALPHA, BETA, HASH = "run-alpha", "run-beta", "abc123"
SCAN = ScanId("subj01", "default_acquisition")


def _write_model(model_dir: Path, ckpts: dict[int, bool]) -> None:
    """A model directory: .hydra config and epoch checkpoints {epoch: all zero?}."""
    (model_dir / ".hydra").mkdir(parents=True)
    (model_dir / ".hydra" / "config.yaml").write_text(HYDRA_CONFIG)
    ckpt_dir = model_dir / "logs" / "lightning_logs" / "version_0" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    for epoch, zero in ckpts.items():
        torch.save({"state_dict": _encoder_state(zero)}, str(ckpt_dir / f"epoch={epoch}-step=1.ckpt"))


@pytest.fixture(scope="module")
def multi_models(tmp_path_factory) -> str:
    """MULTI holds ALPHA and BETA, both zero at EPOCH and random at OTHER_EPOCH; FLAT one model, empty model id."""
    root = tmp_path_factory.mktemp("streaming_multi_model") / "models"
    _write_model(root / MULTI / ALPHA, {EPOCH: True, OTHER_EPOCH: False})
    _write_model(root / MULTI / BETA, {EPOCH: True, OTHER_EPOCH: False})
    _write_model(root / FLAT, {EPOCH: False})
    return str(root)


@pytest.fixture(scope="module")
def nested_models(tmp_path_factory) -> str:
    """FLAT alone, one model with the non-empty model id HASH (HuggingFace-cache layout)."""
    root = tmp_path_factory.mktemp("streaming_nested_model") / "models"
    _write_model(root / FLAT / HASH, {EPOCH: False})
    return str(root)


@dataclasses.dataclass
class _Run:
    """One test's settings, dataset and producers_done event."""

    settings: StreamingSettings
    dataset: DatasetConfig
    done: threading.Event

    def write_inputs(self, scan: ScanId, regions=(MULTI, FLAT)) -> None:
        """The scan's per-region skeleton and subjects CSV, then the completion marker, as the producer writes them."""
        tiles = Path(stages.compute_scan_tiles_dir(self.dataset.cortical_tiles_output, scan))
        crops = tiles / DERIVATIVES_FOLDER / "crops" / MASKS / "2mm"
        for region in regions:
            side = "L" if region.endswith("_left") else "R"
            mask = crops / CROP_DIRS[region] / "mask"
            mask.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(1)
            np.save(mask / f"{side}skeleton.npy", rng.integers(0, 2, size=(1, *IN_SHAPE[1:], 1)).astype(np.int8))
            (mask / f"{side}skeleton_subject.csv").write_text(f"Subject\n{scan.subject}\n")
        (tiles / worker.TILES_COMPLETE_MARKER).write_text(json.dumps({"masks_version": MASKS}), encoding="utf-8")

    def path(self, region: str, model_id: str = "") -> Path:
        base = Path(compute_region_embeddings_dir(self.dataset.datasets_root, self.dataset.masks_version))
        return base / region / model_id / "full_embeddings.csv" if model_id else base / region / "full_embeddings.csv"

    def write_row(self, region: str, model_id: str = "", scan: ScanId = SCAN) -> None:
        path = self.path(region, model_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([[str(scan), *([0.5] * DIMS)]], columns=COLUMNS).to_csv(path, index=False)

    def ids(self, region: str, model_id: str = "") -> list[str]:
        path = self.path(region, model_id)
        return pd.read_csv(path, dtype={"ID": str})["ID"].tolist() if path.is_file() else []

    def run(self, models_path: str, dataset: DatasetConfig | None = None) -> str:
        self.done.set()
        results = consumer.run_consumer(self.settings, dataset or self.dataset, [SCAN], self.done, models_path)
        assert [r.scan for r in results] == [SCAN]
        return results[0].status.value


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Run:
    monkeypatch.setattr(consumer, "init_pipeline_process", lambda: logging.getLogger())
    settings = StreamingSettings(
        subjects_dir=str(tmp_path / "subjects"),
        output_dir=str(tmp_path / "out"),
        log_dir=str(tmp_path / "logs"),
        poll_interval=0.01,
        worker_timeout=5.0,
        consumer_stall_timeout=5.0,
        device="cpu",
    )
    dataset = DatasetConfig(
        name="synthetic",
        datasets_root=str(tmp_path / "dataset"),
        cortical_tiles_output=str(tmp_path / "ct_out"),
        masks_version=MASKS,
        checkpoint_epoch=EPOCH,
    )
    return _Run(settings, dataset, threading.Event())


def _dims(path: Path) -> np.ndarray:
    return pd.read_csv(path, dtype={"ID": str}).drop(columns="ID").to_numpy(dtype=float)


# --- layout ------------------------------------------------------------------------------------------------------


def test_several_models_write_model_id_files(run: _Run, multi_models: str) -> None:
    run.write_inputs(SCAN)
    assert run.run(multi_models) == "ok"
    assert run.ids(MULTI, ALPHA) == [str(SCAN)]
    assert run.ids(MULTI, BETA) == [str(SCAN)]


def test_per_model_folders_writes_model_id_file(run: _Run, nested_models: str) -> None:
    dataset = dataclasses.replace(run.dataset, per_model_folders=True)
    run.write_inputs(SCAN, regions=(FLAT,))
    assert run.run(nested_models, dataset) == "ok"
    assert run.ids(FLAT, HASH) == [str(SCAN)]


def test_single_model_region_writes_region_file(run: _Run, nested_models: str) -> None:
    run.write_inputs(SCAN, regions=(FLAT,))
    assert run.run(nested_models) == "ok"
    assert run.ids(FLAT) == [str(SCAN)]
    assert not run.path(FLAT, HASH).exists()


def test_empty_model_id_writes_region_file(run: _Run, multi_models: str) -> None:
    dataset = dataclasses.replace(run.dataset, per_model_folders=True)
    run.write_inputs(SCAN)
    assert run.run(multi_models, dataset) == "ok"
    assert run.ids(FLAT) == [str(SCAN)]


def test_mismatched_model_file_header_gives_failed(run: _Run, multi_models: str) -> None:
    beta = run.path(MULTI, BETA)
    beta.parent.mkdir(parents=True)
    beta.write_text("ID,other\n")
    run.write_inputs(SCAN)
    assert run.run(multi_models) == "failed"


# --- resume -------------------------------------------------------------------------------------------------------


def test_scan_in_each_model_file_skipped(run: _Run, multi_models: str) -> None:
    run.write_row(MULTI, ALPHA)
    run.write_row(MULTI, BETA)
    run.write_row(FLAT)
    run.write_inputs(SCAN)
    assert run.run(multi_models) == "skipped"


def test_missing_model_row_appended(run: _Run, multi_models: str) -> None:
    run.write_row(MULTI, ALPHA)
    run.write_row(FLAT)
    run.write_inputs(SCAN)
    assert run.run(multi_models) == "ok"
    assert run.ids(MULTI, BETA) == [str(SCAN)]
    assert run.ids(MULTI, ALPHA) == [str(SCAN)]


# --- per-model epoch ----------------------------------------------------------------------------------------------


def test_mapped_model_id_embeds_with_mapped_epoch(run: _Run, multi_models: str) -> None:
    """ALPHA and BETA are zero at EPOCH and random at OTHER_EPOCH: only ALPHA is mapped to OTHER_EPOCH."""
    dataset = dataclasses.replace(run.dataset, checkpoint_epochs=[f"{ALPHA}={OTHER_EPOCH}"])
    run.write_inputs(SCAN)
    assert run.run(multi_models, dataset) == "ok"
    assert not np.all(_dims(run.path(MULTI, ALPHA)) == 0)
    assert np.all(_dims(run.path(MULTI, BETA)) == 0)


def test_unknown_epoch_model_id_raises(run: _Run, multi_models: str) -> None:
    dataset = dataclasses.replace(run.dataset, checkpoint_epochs=[f"run-gamma={OTHER_EPOCH}"])
    run.write_inputs(SCAN)
    with pytest.raises(ValueError, match="run-gamma"):
        run.run(multi_models, dataset)


# --- combine --------------------------------------------------------------------------------------------------------


def _write_model_id_source(run: _Run) -> None:
    """The region embeddings directory as the consumer leaves it for MULTI: one file per model id."""
    for model_id in (ALPHA, BETA):
        run.write_row(MULTI, model_id)
        (run.path(MULTI, model_id).parent / "embedding_width.json").write_text(json.dumps({"width": DIMS}))


def test_combine_qc_no_warning_for_model_id_folders(
    run: _Run,
    monkeypatch: pytest.MonkeyPatch,
    combine_records: list[logging.LogRecord],  # noqa: F811 - imported fixture
) -> None:
    class ModelIdFoldersPutTogether(PutTogetherEmbeddings):
        """Writes the combined files of each model id in its own folder, as stage 5 does over several models."""

        def run(self) -> int:
            for model_id in (ALPHA, BETA):
                folder = os.path.join(self.args.output_path, model_id)
                os.makedirs(folder, exist_ok=True)
                with open(os.path.join(folder, f"{MULTI}_embeddings.csv"), "w") as handle:
                    handle.write(f"ID,dim1\n{SCAN},0.5\n")
            return 0

    monkeypatch.setattr(combine, "PutTogetherEmbeddings", ModelIdFoldersPutTogether)
    _write_model_id_source(run)
    combine.run_combine(run.settings, run.dataset)
    warnings = [r.getMessage() for r in combine_records if r.levelno == logging.WARNING]
    assert warnings == []


def test_combine_writes_one_folder_per_model_id(run: _Run) -> None:
    _write_model_id_source(run)
    combined = Path(combine.run_combine(run.settings, run.dataset))
    assert combined == Path(compute_combined_embeddings_dir(run.dataset.datasets_root, MASKS))
    for model_id in (ALPHA, BETA):
        assert sorted(p.name for p in (combined / model_id).glob("*_embeddings.csv")) == [f"{MULTI}_embeddings.csv"]
