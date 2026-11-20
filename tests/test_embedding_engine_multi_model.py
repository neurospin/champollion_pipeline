"""EmbeddingEngine over several models of one region (TASK-MULTIMODEL-BDRABCZUK-29169A15593C, epic -A02369CED8AE).

The engine keeps every model that find_region_models discovers, keyed by (region, model id), instead of ignoring a
second model of a region; each model loads at its own checkpoint epoch (a per-id epoch mapping, the default epoch for
the other ids); the encoder load record names the model label (<Region>/<id>).

- REQ-MULTIMODEL-BDRABCZUK-72CF83C93218 test_models_lists_region_and_model_id_pairs
- REQ-STREAMING-BDRABCZUK-6CD608E8C874 (supersedes -E9DFC740767F) test_regions_lists_each_region_once
- REQ-MULTIMODEL-BDRABCZUK-F0758449211D test_embed_uses_encoder_of_given_model_id
- REQ-MULTIMODEL-BDRABCZUK-CE4E51D63223 test_embed_crops_uses_encoder_of_given_model_id
- REQ-MULTIMODEL-BDRABCZUK-6D627CFC89CB test_unknown_model_id_raises
- REQ-MULTIMODEL-BDRABCZUK-291403EA3538 test_embed_without_model_id_uses_only_model
- REQ-MULTIMODEL-BDRABCZUK-1613C19FAECC test_embed_without_model_id_on_several_models_raises
- REQ-MULTIMODEL-BDRABCZUK-D40558480C99 test_mapped_model_id_loads_mapped_epoch
- REQ-MULTIMODEL-BDRABCZUK-CAFCF239F666 test_unmapped_model_id_loads_default_epoch
- REQ-MULTIMODEL-BDRABCZUK-4101F556274A test_encoder_load_record_names_model_label

Interface the tests expect: EmbeddingEngine(models_path, device, *, epoch=80, epochs=None, batch_size=32) where
``epochs`` maps a model id to its checkpoint epoch; EmbeddingEngine.models, a list of (region name, model id) tuples;
embed(region, skeletons, scan_ids, model_id=None) and embed_crops(region, crops_2mm, model_id=None).

Synthetic data only: tiny ConvNet region models, as in tests/test_embedding_engine.py. Weights are told apart by
content: an all-zero encoder embeds every scan as zeros, a random one does not.
"""

import numpy as np
import pytest
import torch
from champollion_utils.logger import LOG

from champollion_pipeline.embedding_engine import EmbeddingEngine
from tests.test_embedding_engine import (  # noqa: F401 - engine_records is a fixture
    HYDRA_CONFIG,
    _encoder_state,
    engine_records,
)

IN_SHAPE = (1, 8, 8, 8)
DIMS = 4
EPOCH = 80
OTHER_EPOCH = 5
MULTI = "SC-sylv_left"  # two models: ALPHA and BETA
FLAT = "FColl_right"  # one model, empty model id
NESTED = "Occ_left"  # one model, non-empty model id (HuggingFace-cache layout)
ALPHA, BETA, HASH = "run-alpha", "run-beta", "abc123"
ENGINE_LOGGER = "champollion_pipeline.embedding_engine"


def _write_model(model_dir, ckpts: dict[int, bool]) -> None:
    """A model directory: .hydra config and epoch checkpoints {epoch: all zero?}."""
    (model_dir / ".hydra").mkdir(parents=True)
    (model_dir / ".hydra" / "config.yaml").write_text(HYDRA_CONFIG)
    ckpt_dir = model_dir / "logs" / "lightning_logs" / "version_0" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    for epoch, zero in ckpts.items():
        torch.save({"state_dict": _encoder_state(zero)}, str(ckpt_dir / f"epoch={epoch}-step=1.ckpt"))


def _skeletons(n_scans: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n_scans, *IN_SHAPE[1:], 1)).astype(np.int8)


def _scan_ids(n_scans: int) -> list[str]:
    return [f"sub-{i:02d}" for i in range(n_scans)]


def _all_zero(table) -> bool:
    return bool(np.all(table.drop(columns="ID").to_numpy(dtype=float) == 0))


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """MULTI holds ALPHA (zero at EPOCH, random at OTHER_EPOCH) and BETA (random at EPOCH, zero at OTHER_EPOCH);
    FLAT one model in the region folder itself; NESTED one model in a hash folder."""
    root = tmp_path_factory.mktemp("engine_multi_model") / "models"
    _write_model(root / MULTI / ALPHA, {EPOCH: True, OTHER_EPOCH: False})
    _write_model(root / MULTI / BETA, {EPOCH: False, OTHER_EPOCH: True})
    _write_model(root / FLAT, {EPOCH: False})
    _write_model(root / NESTED / HASH, {EPOCH: False})
    return root


@pytest.fixture
def crops_2mm(tmp_path):
    """S.C.-sylv. holds a left skeleton of 3 scans."""
    mask_dir = tmp_path / "crops" / "2mm" / "S.C.-sylv." / "mask"
    mask_dir.mkdir(parents=True)
    np.save(mask_dir / "Lskeleton.npy", _skeletons(3, seed=1))
    (mask_dir / "Lskeleton_subject.csv").write_text("Subject\nsub-a\nsub-b\nsub-c\n")
    return tmp_path / "crops" / "2mm"


def test_models_lists_region_and_model_id_pairs(models):
    engine = EmbeddingEngine(str(models), "cpu")
    assert engine.models == [(FLAT, ""), (NESTED, HASH), (MULTI, ALPHA), (MULTI, BETA)]


def test_regions_lists_each_region_once(models):
    engine = EmbeddingEngine(str(models), "cpu")
    assert engine.regions == [FLAT, NESTED, MULTI]


def test_embed_uses_encoder_of_given_model_id(models):
    engine = EmbeddingEngine(str(models), "cpu", epoch=EPOCH)
    alpha = engine.embed(MULTI, _skeletons(3), _scan_ids(3), model_id=ALPHA)
    beta = engine.embed(MULTI, _skeletons(3), _scan_ids(3), model_id=BETA)
    assert _all_zero(alpha)
    assert not _all_zero(beta)


def test_embed_crops_uses_encoder_of_given_model_id(models, crops_2mm):
    engine = EmbeddingEngine(str(models), "cpu", epoch=EPOCH)
    alpha = engine.embed_crops(MULTI, str(crops_2mm), model_id=ALPHA)
    beta = engine.embed_crops(MULTI, str(crops_2mm), model_id=BETA)
    assert alpha["ID"].tolist() == ["sub-a", "sub-b", "sub-c"]
    assert _all_zero(alpha)
    assert not _all_zero(beta)


def test_unknown_model_id_raises(models):
    engine = EmbeddingEngine(str(models), "cpu")
    with pytest.raises(ValueError) as excinfo:
        engine.embed(MULTI, _skeletons(1), _scan_ids(1), model_id="run-gamma")
    assert MULTI in str(excinfo.value)
    assert "run-gamma" in str(excinfo.value)


def test_embed_without_model_id_uses_only_model(models):
    engine = EmbeddingEngine(str(models), "cpu", epoch=EPOCH)
    table = engine.embed(NESTED, _skeletons(2), _scan_ids(2))
    assert table["ID"].tolist() == _scan_ids(2)
    assert not _all_zero(table)


def test_embed_without_model_id_on_several_models_raises(models):
    engine = EmbeddingEngine(str(models), "cpu")
    with pytest.raises(ValueError) as excinfo:
        engine.embed(MULTI, _skeletons(1), _scan_ids(1))
    assert ALPHA in str(excinfo.value)
    assert BETA in str(excinfo.value)


def test_mapped_model_id_loads_mapped_epoch(models):
    """ALPHA is zero at EPOCH and random at OTHER_EPOCH: mapping ALPHA to OTHER_EPOCH gives non-zero embeddings."""
    engine = EmbeddingEngine(str(models), "cpu", epoch=EPOCH, epochs={ALPHA: OTHER_EPOCH})
    table = engine.embed(MULTI, _skeletons(3), _scan_ids(3), model_id=ALPHA)
    assert not _all_zero(table)


def test_unmapped_model_id_loads_default_epoch(models):
    """BETA is random at EPOCH and zero at OTHER_EPOCH: with only ALPHA mapped, BETA stays at the default EPOCH."""
    engine = EmbeddingEngine(str(models), "cpu", epoch=EPOCH, epochs={ALPHA: OTHER_EPOCH})
    table = engine.embed(MULTI, _skeletons(3), _scan_ids(3), model_id=BETA)
    assert not _all_zero(table)


def test_encoder_load_record_names_model_label(models, engine_records):  # noqa: F811 - imported fixture
    engine = EmbeddingEngine(str(models), "cpu")
    engine.embed(MULTI, _skeletons(2), _scan_ids(2), model_id=ALPHA)
    engine.embed(MULTI, _skeletons(2), _scan_ids(2), model_id=BETA)
    loads = [r.getMessage() for r in engine_records if r.levelno == LOG and r.name == ENGINE_LOGGER]
    assert any(f"{MULTI}/{ALPHA}" in message for message in loads), loads
    assert any(f"{MULTI}/{BETA}" in message for message in loads), loads
