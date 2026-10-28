"""EmbeddingEngine: resident in-process region encoders (TASK-STREAMING-BDRABCZUK-E56A3FEB14F0).

- REQ-STREAMING-BDRABCZUK-E9DFC740767F: regions lists the resolve_region_name names of the region model dirs, sorted.
- REQ-STREAMING-BDRABCZUK-F35283151590: a region encoder is loaded at the first embed call, then kept resident.
- REQ-STREAMING-BDRABCZUK-05BD1895FFAA: the weights are those evaluation_model_dir selects at the engine's epoch.
- REQ-STREAMING-BDRABCZUK-F2CD29B507E2: embed returns ID then dim1..dimK, one row per scan in input order.
- REQ-STREAMING-BDRABCZUK-0FE61AAE6E5F: a scan id count that differs from the scan count raises ValueError.
- REQ-STREAMING-BDRABCZUK-8E8199F8C2AF: an unknown region raises ValueError naming region and models path.
- REQ-STREAMING-BDRABCZUK-3B336DC8A034: the encoder runs on the given device, whatever CUDA availability says.
- REQ-STREAMING-BDRABCZUK-9BA9FF2DD264: loading writes nothing to standard output.
- REQ-STREAMING-BDRABCZUK-546A383B876B: one LOG-level record with region and load seconds per load.
- REQ-STREAMING-BDRABCZUK-1E4DA7E6C672: mini-batches never exceed the engine's batch size.
- REQ-STREAMING-BDRABCZUK-C92ADD556CC8: embedding starts no child process.
- REQ-STREAMING-BDRABCZUK-CC0CE23126AD: embed_crops reads the region side skeleton and subjects CSV.

Synthetic data only: tiny ConvNet region models (the champollion_V1 architecture, imported read-only), as in
tests/test_profile_embedding_path.py. Parity with evaluate.py is the parity task's (46371173BEC3), not this file's.
Weights are told apart by content: an all-zero encoder embeds every scan as zeros, a random one does not.
"""

import logging
import multiprocessing.process
import os
import re
import subprocess

import numpy as np
import pandas as pd
import pytest
import torch
import torch.utils.data._utils.pin_memory as pin_memory_utils
from champollion.backbones.convnet import ConvNet
from champollion_utils.logger import LOG

from champollion_pipeline.embedding_engine import EmbeddingEngine

IN_SHAPE = (1, 8, 8, 8)
DIMS = 4
EPOCH = 80
CONVNET_KWARGS = {
    "encoder_depth": 1,
    "block_depth": 2,
    "filters": [2],
    "initial_kernel_size": 3,
    "initial_stride": 1,
    "max_pool": False,
    "adaptive_pooling": None,
    "drop_rate": 0.0,
}
HYDRA_CONFIG = f"""backbone_name: convnet
data:
- input_size: (1, 8, 8, 8)
encoder_depth: 1
block_depth: 2
filters: [2]
initial_kernel_size: 3
initial_stride: 1
max_pool: false
backbone_output_size: {DIMS}
adaptive_pooling: null
drop_rate: 0.0
"""
LEFT, RIGHT, NESTED = "SC-sylv_left", "FColl_right", "Occ_left"
ENGINE_LOGGER = "champollion_pipeline.embedding_engine"


def _encoder_state(zero: bool) -> dict:
    """Lightning-style state_dict of a tiny ConvNet encoder, all zeros or random."""
    torch.manual_seed(0)
    encoder = ConvNet(in_channels=1, in_shape=IN_SHAPE, num_representation_features=DIMS, **CONVNET_KWARGS)
    state = encoder.state_dict()
    if zero:
        state = {k: torch.zeros_like(v) for k, v in state.items()}
    return {f"backbones.0.{k}": v for k, v in state.items()}


def _write_region_model(model_dir, ckpts: dict, best_weights_zero: bool | None = None) -> None:
    """A region model directory: .hydra config, epoch checkpoints {epoch: zero?}, optional best_model_weights.pt."""
    (model_dir / ".hydra").mkdir(parents=True)
    (model_dir / ".hydra" / "config.yaml").write_text(HYDRA_CONFIG)
    ckpt_dir = model_dir / "logs" / "lightning_logs" / "version_0" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    for epoch, zero in ckpts.items():
        torch.save({"state_dict": _encoder_state(zero)}, str(ckpt_dir / f"epoch={epoch}-step=1.ckpt"))
    if best_weights_zero is not None:
        torch.save(_encoder_state(best_weights_zero), str(model_dir / "logs" / "best_model_weights.pt"))


def _skeletons(n_scans: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n_scans, *IN_SHAPE[1:], 1)).astype(np.int8)


def _scan_ids(n_scans: int) -> list[str]:
    return [f"sub-{i:02d}" for i in range(n_scans)]


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """Models path: a flat left region (zero epoch-80, random epoch-5), a flat right region (random epoch-80),
    a nested left region (HuggingFace-cache layout), and a best-weights-only region (zero .pt, random epoch-3)."""
    root = tmp_path_factory.mktemp("embedding_engine") / "models"
    _write_region_model(root / LEFT, {EPOCH: True, 5: False})
    _write_region_model(root / RIGHT, {EPOCH: False})
    _write_region_model(root / NESTED / "abc123", {EPOCH: False})
    _write_region_model(root / "Best_right", {3: False}, best_weights_zero=True)
    return root


@pytest.fixture
def ckpt_loads(monkeypatch):
    """Paths of every .ckpt file passed to torch.load, in call order."""
    loads: list[str] = []
    real_load = torch.load

    def spy(f, *args, **kwargs):
        if str(f).endswith(".ckpt"):
            loads.append(str(f))
        return real_load(f, *args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    return loads


@pytest.fixture
def forwards(monkeypatch):
    """(batch size, input device, parameter device) of every ConvNet forward call."""
    calls: list[tuple[int, torch.device, torch.device]] = []
    real_forward = ConvNet.forward

    def spy(self, x):
        calls.append((x.shape[0], x.device, next(self.parameters()).device))
        return real_forward(self, x)

    monkeypatch.setattr(ConvNet, "forward", spy)
    return calls


def _loads_of(loads: list[str], region: str) -> int:
    return sum(f"{os.sep}{region}{os.sep}" in path for path in loads)


def test_regions_lists_region_names_sorted(models):
    engine = EmbeddingEngine(str(models), "cpu")
    assert engine.regions == ["Best_right", RIGHT, NESTED, LEFT]


def test_construction_loads_no_encoder(models, ckpt_loads):
    EmbeddingEngine(str(models), "cpu")
    assert ckpt_loads == []


def test_first_embed_loads_only_that_region(models, ckpt_loads):
    engine = EmbeddingEngine(str(models), "cpu")
    engine.embed(LEFT, _skeletons(2), _scan_ids(2))
    assert _loads_of(ckpt_loads, LEFT) == 1
    assert _loads_of(ckpt_loads, RIGHT) == 0
    assert _loads_of(ckpt_loads, NESTED) == 0


def test_encoder_loaded_once_per_region(models, ckpt_loads):
    engine = EmbeddingEngine(str(models), "cpu")
    engine.embed(LEFT, _skeletons(2), _scan_ids(2))
    engine.embed(RIGHT, _skeletons(2), _scan_ids(2))
    engine.embed(LEFT, _skeletons(3, seed=1), _scan_ids(3))
    engine.embed(RIGHT, _skeletons(1, seed=2), _scan_ids(1))
    assert _loads_of(ckpt_loads, LEFT) == 1
    assert _loads_of(ckpt_loads, RIGHT) == 1


@pytest.mark.parametrize(("epoch", "expect_zero"), [(EPOCH, True), (5, False)])
def test_epoch_checkpoint_selected(models, epoch, expect_zero):
    """LEFT holds a zero epoch-80 checkpoint and a random epoch-5 one: the engine epoch picks which."""
    engine = EmbeddingEngine(str(models), "cpu", epoch=epoch)
    values = engine.embed(LEFT, _skeletons(3), _scan_ids(3)).drop(columns="ID").to_numpy()
    assert bool(np.all(values == 0)) == expect_zero


def test_best_weights_stand_in_selected(models):
    """Best_right has no epoch-80 checkpoint: the zero best_model_weights.pt stands in, not the random epoch-3."""
    engine = EmbeddingEngine(str(models), "cpu", epoch=EPOCH)
    values = engine.embed("Best_right", _skeletons(3), _scan_ids(3)).drop(columns="ID").to_numpy()
    assert np.all(values == 0)


def test_embed_returns_id_then_dim_columns(models):
    engine = EmbeddingEngine(str(models), "cpu")
    scan_ids = ["sub-09", "sub-03", "sub-05"]
    table = engine.embed(RIGHT, _skeletons(3), scan_ids)
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ["ID", *(f"dim{i}" for i in range(1, DIMS + 1))]
    assert table["ID"].tolist() == scan_ids
    values = table.drop(columns="ID").to_numpy(dtype=float)
    assert values.shape == (3, DIMS)
    assert np.isfinite(values).all()


def test_scan_id_count_mismatch_raises(models):
    engine = EmbeddingEngine(str(models), "cpu")
    with pytest.raises(ValueError, match=r"(?s)(3.*7|7.*3)"):
        engine.embed(RIGHT, _skeletons(3), _scan_ids(7))


def test_unknown_region_raises(models):
    engine = EmbeddingEngine(str(models), "cpu")
    with pytest.raises(ValueError) as excinfo:
        engine.embed("Nowhere_left", _skeletons(1), _scan_ids(1))
    assert "Nowhere_left" in str(excinfo.value)
    assert str(models) in str(excinfo.value)


@pytest.mark.parametrize("cuda_available", [False, True])
def test_encoder_runs_on_given_device(models, forwards, monkeypatch, cuda_available):
    """A cpu engine stays on cpu even when torch reports a CUDA device (evaluate.main would pick cuda)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(pin_memory_utils, "pin_memory", lambda data, device=None: data)
    engine = EmbeddingEngine(str(models), torch.device("cpu"))
    engine.embed(RIGHT, _skeletons(3), _scan_ids(3))
    assert forwards
    assert {(inp.type, par.type) for _, inp, par in forwards} == {("cpu", "cpu")}


def test_load_writes_nothing_to_stdout(models, capsys):
    """evaluate.load_model prints every state_dict key and 'Model weights loaded'; none of it reaches stdout."""
    engine = EmbeddingEngine(str(models), "cpu")
    capsys.readouterr()
    engine.embed(LEFT, _skeletons(2), _scan_ids(2))
    out = capsys.readouterr().out
    assert "backbones.0." not in out
    assert "encoder." not in out
    assert "Model weights loaded" not in out


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def engine_records():
    """Records reaching the champollion_pipeline.embedding_engine logger (level and handlers restored)."""
    logger = logging.getLogger(ENGINE_LOGGER)
    handler = _ListHandler()
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def test_load_emits_log_record(models, engine_records):
    engine = EmbeddingEngine(str(models), "cpu")
    engine.embed(LEFT, _skeletons(2), _scan_ids(2))
    engine.embed(LEFT, _skeletons(2), _scan_ids(2))
    loads = [r for r in engine_records if r.levelno == LOG and LEFT in r.getMessage()]
    assert len(loads) == 1
    assert loads[0].name == ENGINE_LOGGER
    seconds = re.search(r"\d+(\.\d+)?\s*s\b", loads[0].getMessage().replace(LEFT, ""))
    assert seconds, loads[0].getMessage()


def test_minibatches_bounded_by_batch_size(models, forwards):
    engine = EmbeddingEngine(str(models), "cpu", batch_size=2)
    engine.embed(RIGHT, _skeletons(5), _scan_ids(5))
    sizes = [size for size, _, _ in forwards]
    assert sum(sizes) == 5
    assert max(sizes) <= 2


def test_embed_starts_no_child_process(models, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("EmbeddingEngine started a child process")

    monkeypatch.setattr(subprocess.Popen, "__init__", refuse)
    monkeypatch.setattr(os, "fork", refuse)
    monkeypatch.setattr(os, "posix_spawn", refuse)
    monkeypatch.setattr(multiprocessing.process.BaseProcess, "start", refuse)
    engine = EmbeddingEngine(str(models), "cpu")
    table = engine.embed(RIGHT, _skeletons(2), _scan_ids(2))
    assert len(table) == 2


def _write_crop(crops_2mm, crop_name: str, side: str, subjects: list[str], seed: int) -> None:
    mask_dir = crops_2mm / crop_name / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    np.save(mask_dir / f"{side}skeleton.npy", _skeletons(len(subjects), seed=seed))
    (mask_dir / f"{side}skeleton_subject.csv").write_text("Subject\n" + "".join(f"{s}\n" for s in subjects))


@pytest.fixture
def crops_2mm(tmp_path):
    """S.C.-sylv. holds a left skeleton of 3 scans and a right one of 2 (a decoy side); F.Coll. a right one of 4."""
    crops = tmp_path / "crops" / "2mm"
    _write_crop(crops, "S.C.-sylv.", "L", ["sub-a", "sub-b", "sub-c"], seed=1)
    _write_crop(crops, "S.C.-sylv.", "R", ["sub-x", "sub-y"], seed=2)
    _write_crop(crops, "F.Coll.", "R", ["sub-1", "sub-2", "sub-3", "sub-4"], seed=3)
    return crops


@pytest.mark.parametrize(
    ("region", "subjects"),
    [(LEFT, ["sub-a", "sub-b", "sub-c"]), (RIGHT, ["sub-1", "sub-2", "sub-3", "sub-4"])],
)
def test_embed_crops_reads_region_skeleton_and_subjects(models, crops_2mm, region, subjects):
    engine = EmbeddingEngine(str(models), "cpu")
    table = engine.embed_crops(region, str(crops_2mm))
    assert table["ID"].tolist() == subjects
    assert list(table.columns) == ["ID", *(f"dim{i}" for i in range(1, DIMS + 1))]
