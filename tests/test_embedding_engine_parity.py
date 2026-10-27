"""EmbeddingEngine parity with champollion_V1 evaluate.py, per region (TASK-STREAMING-BDRABCZUK-46371173BEC3).

- REQ-STREAMING-BDRABCZUK-B2F2577851E0: epoch-N checkpoint region: embed_crops equals the evaluate.py CSV.
- REQ-STREAMING-BDRABCZUK-13EBB548262B: best-weights stand-in region: embed_crops equals the evaluate.py CSV.
- REQ-STREAMING-BDRABCZUK-74EA68F55B3A: embed, given the skeletons and subjects evaluate.py reads, equals its CSV.

The reference is the embeddings stage itself: GenerateEmbeddings._run_per_region runs the real
external/champollion_V1/champollion/evaluate.py (read-only) as one subprocess per region, with the -m directory
evaluation_model_dir gives, and writes <out>/<region>/full_embeddings.csv. CUDA is hidden from that subprocess so
both sides run on CPU. Equal means: same ID values in the same order, same column names, and values within
numpy.allclose(engine, reference, rtol=1e-5, atol=1e-6) (float32 encoders; the CSV round-trip is the only loss).

Synthetic data only: tiny ConvNet region models (the champollion_V1 architecture), each with its own random
weights, so a region or weights mix-up cannot pass. Each best-weights region also holds a random checkpoint of
another epoch, which neither side may use. Both hemispheres are covered on both weights paths.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from champollion.backbones.convnet import ConvNet

from champollion_pipeline.embedding_engine import EmbeddingEngine
from champollion_pipeline.generate_embeddings import GenerateEmbeddings

pytestmark = pytest.mark.integration  # noqa: V107

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATE = REPO_ROOT / "external" / "champollion_V1" / "champollion" / "evaluate.py"
RTOL, ATOL = 1e-5, 1e-6
IN_SHAPE = (1, 8, 8, 8)
DIMS = 4
EPOCH = 80
OTHER_EPOCH = 3
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
# region model name -> (crop directory, skeleton side, subjects in skeleton order, encoder seed)
EPOCH_REGIONS = {
    "SC-sylv_left": ("S.C.-sylv.", "L", ["sub-09", "sub-03", "sub-05"], 11),
    "FColl_right": ("F.Coll.", "R", ["sub-1001", "sub-0042", "sub-0007", "sub-0100"], 12),
}
BEST_WEIGHTS_REGIONS = {
    "STs_left": ("S.T.s.", "L", ["sub-b", "sub-a", "sub-d", "sub-c", "sub-e"], 13),
    "SOr_right": ("S.Or.", "R", ["sub-x", "sub-y"], 14),
}
REGIONS = {**EPOCH_REGIONS, **BEST_WEIGHTS_REGIONS}


def _encoder_state(seed: int) -> dict:
    """Lightning-style state_dict (backbones.0. prefix) of a tiny ConvNet encoder with seeded random weights."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        encoder = ConvNet(in_channels=1, in_shape=IN_SHAPE, num_representation_features=DIMS, **CONVNET_KWARGS)
    return {f"backbones.0.{k}": v for k, v in encoder.state_dict().items()}


def _write_region_model(model_dir: Path, seed: int, best_weights: bool) -> None:
    """Epoch-80 checkpoint, or best_model_weights.pt beside an unused epoch-3 checkpoint of other weights."""
    (model_dir / ".hydra").mkdir(parents=True)
    (model_dir / ".hydra" / "config.yaml").write_text(HYDRA_CONFIG)
    ckpt_dir = model_dir / "logs" / "lightning_logs" / "version_0" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    if best_weights:
        torch.save(_encoder_state(seed), str(model_dir / "logs" / "best_model_weights.pt"))
        ckpt_epoch, ckpt_seed = OTHER_EPOCH, seed + 100
    else:
        ckpt_epoch, ckpt_seed = EPOCH, seed
    torch.save({"state_dict": _encoder_state(ckpt_seed)}, str(ckpt_dir / f"epoch={ckpt_epoch}-step=1.ckpt"))


def _write_crop(crops: Path, crop_name: str, side: str, subjects: list[str], seed: int) -> None:
    mask_dir = crops / crop_name / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    skeletons = rng.choice(np.array([0, 30, 60], dtype=np.int16), size=(len(subjects), *IN_SHAPE[1:], 1))
    np.save(mask_dir / f"{side}skeleton.npy", skeletons)
    pd.DataFrame({"Subject": subjects}).to_csv(mask_dir / f"{side}skeleton_subject.csv", index=False)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Models path and crops directory holding the four regions above."""
    root = tmp_path_factory.mktemp("embedding_engine_parity")
    models, crops = root / "models", root / "crops" / "2mm"
    for region, (crop_name, side, subjects, seed) in REGIONS.items():
        _write_region_model(models / region, seed, best_weights=region in BEST_WEIGHTS_REGIONS)
        _write_crop(crops, crop_name, side, subjects, seed)
    return {"root": root, "models": models, "crops": crops}


@pytest.fixture(scope="module")
def reference(dataset):
    """Region -> the full_embeddings.csv the embeddings stage writes by running evaluate.py, CPU only."""
    assert EVALUATE.is_file(), f"champollion_V1 evaluate.py not found at {EVALUATE}"
    out = dataset["root"] / "evaluate_out"
    script = GenerateEmbeddings()
    script.parse_args([str(dataset["models"]), str(dataset["root"] / "dataset")])
    script.args.checkpoint_epoch = EPOCH
    script.args.overwrite = True
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CUDA_VISIBLE_DEVICES", "")
        code = script._run_per_region(str(EVALUATE), str(dataset["crops"]), str(out))
    assert code == 0, f"evaluate.py failed for at least one region (exit {code})"
    tables = {}
    for region in REGIONS:
        csv = out / region / "full_embeddings.csv"
        assert csv.is_file(), f"evaluate.py wrote no CSV for {region}"
        table = pd.read_csv(csv, dtype={"ID": str})
        values = table.drop(columns="ID").to_numpy(dtype=float)
        # Guard against a trivial parity: random encoders give finite, non-constant embeddings.
        assert np.isfinite(values).all() and np.ptp(values) > 0, f"degenerate reference embeddings for {region}"
        tables[region] = table
    return tables


@pytest.fixture(scope="module")
def engine(dataset):
    return EmbeddingEngine(str(dataset["models"]), "cpu", epoch=EPOCH)


def _assert_parity(table: pd.DataFrame, reference: pd.DataFrame) -> None:
    assert table["ID"].astype(str).tolist() == reference["ID"].tolist()
    assert list(table.columns) == list(reference.columns)
    np.testing.assert_allclose(
        table.drop(columns="ID").to_numpy(dtype=float),
        reference.drop(columns="ID").to_numpy(dtype=float),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("region", sorted(EPOCH_REGIONS))
def test_embed_crops_matches_evaluate_epoch_checkpoint(engine, dataset, reference, region):
    _assert_parity(engine.embed_crops(region, str(dataset["crops"])), reference[region])


@pytest.mark.parametrize("region", sorted(BEST_WEIGHTS_REGIONS))
def test_embed_crops_matches_evaluate_best_weights(engine, dataset, reference, region):
    _assert_parity(engine.embed_crops(region, str(dataset["crops"])), reference[region])


@pytest.mark.parametrize("region", sorted(REGIONS))
def test_embed_matches_evaluate(engine, dataset, reference, region):
    crop_name, side, _, _ = REGIONS[region]
    mask_dir = dataset["crops"] / crop_name / "mask"
    skeletons = np.load(mask_dir / f"{side}skeleton.npy")
    subjects = pd.read_csv(mask_dir / f"{side}skeleton_subject.csv", dtype={"Subject": str})["Subject"].tolist()
    _assert_parity(engine.embed(region, skeletons, subjects), reference[region])
