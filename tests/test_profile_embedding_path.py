"""scripts/profile_embedding_path.py: plain measurements of the embedding path (TASK-STREAMING-BDRABCZUK-81AF48D80097).

- REQ-STREAMING-BDRABCZUK-9DE574D65560: one region entry per region model directory under the models path.
- REQ-STREAMING-BDRABCZUK-865CB5142097: checkpoint size in bytes per region.
- REQ-STREAMING-BDRABCZUK-501FA8BBE175: model load seconds per region.
- REQ-STREAMING-BDRABCZUK-DA12F2B0AECD: inference seconds per scan at the given batch size.
- REQ-STREAMING-BDRABCZUK-3052BB2AF538: peak CUDA memory in bytes, null without a CUDA device.
- REQ-STREAMING-BDRABCZUK-4BB8B7F9BBD0: path, shape, element type and bytes per scan of the skeleton input.
- REQ-STREAMING-BDRABCZUK-48AE20FF040C: seconds for a fresh interpreter to start and import torch.
- REQ-STREAMING-BDRABCZUK-EB7F5E25D3B4: seconds and exit code of each stage command per scan.

Synthetic data only: two tiny ConvNet region models (the champollion_V1 architecture, imported read-only) and
random binary skeletons laid out as the embeddings stage reads them. The script runs once as a subprocess, on CPU.
"""

import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from champollion.backbones.convnet import ConvNet

pytestmark = pytest.mark.integration  # noqa: V107

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "profile_embedding_path.py"
IN_SHAPE = (1, 8, 8, 8)
BATCH_SIZE = 2
SCANS = ("sub-01", "sub-02")
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
HYDRA_CONFIG = """backbone_name: convnet
data:
- input_size: (1, 8, 8, 8)
encoder_depth: 1
block_depth: 2
filters: [2]
initial_kernel_size: 3
initial_stride: 1
max_pool: false
backbone_output_size: 4
adaptive_pooling: null
drop_rate: 0.0
"""
# region model name -> (crop directory name, skeleton side, number of scans)
REGIONS = {
    "SC-sylv_left": ("S.C.-sylv.", "L", 3),
    "FColl_right": ("F.Coll.", "R", 2),
}


def _write_region_model(models: Path, region: str) -> Path:
    """A region model directory evaluate.py can load; returns its epoch-80 checkpoint."""
    model_dir = models / region
    (model_dir / ".hydra").mkdir(parents=True)
    (model_dir / ".hydra" / "config.yaml").write_text(HYDRA_CONFIG)
    ckpt_dir = model_dir / "logs" / "lightning_logs" / "version_0" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    encoder = ConvNet(in_channels=1, in_shape=IN_SHAPE, num_representation_features=4, **CONVNET_KWARGS)
    ckpt = ckpt_dir / "epoch=80-step=1.ckpt"
    torch.save({"state_dict": {f"backbones.0.{k}": v for k, v in encoder.state_dict().items()}}, str(ckpt))
    return ckpt


def _write_skeleton(crops: Path, crop_name: str, side: str, n_scans: int) -> Path:
    mask_dir = crops / crop_name / "mask"
    mask_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    skeleton = mask_dir / f"{side}skeleton.npy"
    np.save(skeleton, rng.integers(0, 2, size=(n_scans, 8, 8, 8, 1)).astype(np.int8))
    (mask_dir / f"{side}skeleton_subject.csv").write_text(
        "Subject\n" + "".join(f"sub-{i:02d}\n" for i in range(n_scans))
    )
    return skeleton


def _stage_command(code: str, *args: str) -> str:
    return " ".join(shlex.quote(part) for part in (sys.executable, "-c", code, *args))


def _run_script(tmp: Path, models: Path, crops: Path, *extra: str) -> dict:
    output = tmp / "report.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(models),
            str(crops),
            "--output",
            str(output),
            "--batch-size",
            str(BATCH_SIZE),
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    report = json.loads(output.read_text()) if output.is_file() else None
    return {"proc": proc, "report": report, "output": output}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("profile_embedding_path")
    models, crops, marks = tmp / "models", tmp / "crops" / "2mm", tmp / "marks"
    marks.mkdir()
    ckpts, skeletons = {}, {}
    for region, (crop_name, side, n_scans) in REGIONS.items():
        ckpts[region] = _write_region_model(models, region)
        skeletons[region] = _write_skeleton(crops, crop_name, side, n_scans)
    return {"tmp": tmp, "models": models, "crops": crops, "marks": marks, "ckpts": ckpts, "skeletons": skeletons}


@pytest.fixture(scope="module")
def cpu_run(dataset):
    marks = dataset["marks"]
    touch = _stage_command("import pathlib, sys; pathlib.Path(sys.argv[1]).touch()", f"{marks}/{{scan}}.done")
    fail = _stage_command("import sys; sys.exit(3)", "{scan}")
    scan_args = [arg for scan in SCANS for arg in ("--scan", scan)]
    return _run_script(
        dataset["tmp"],
        dataset["models"],
        dataset["crops"],
        "--cpu",
        "--stage",
        f"morphologist={touch}",
        "--stage",
        f"tiles={fail}",
        *scan_args,
    )


def _report(run: dict) -> dict:
    proc = run["proc"]
    assert run["report"] is not None, f"no report at {run['output']} (exit {proc.returncode}): {proc.stderr[-2000:]}"
    return run["report"]


def _entry(run: dict, region: str) -> dict:
    entries = {e["region"]: e for e in _report(run)["regions"]}
    assert region in entries, f"no entry for {region}: {sorted(entries)}"
    return entries[region]


def test_report_has_one_entry_per_region_model_dir(cpu_run):
    report = _report(cpu_run)
    assert cpu_run["proc"].returncode == 0, cpu_run["proc"].stderr[-2000:]
    assert sorted(e["region"] for e in report["regions"]) == sorted(REGIONS)


@pytest.mark.parametrize("region", sorted(REGIONS))
def test_region_entry_records_checkpoint_bytes(cpu_run, dataset, region):
    assert _entry(cpu_run, region)["checkpoint_bytes"] == dataset["ckpts"][region].stat().st_size


@pytest.mark.parametrize("region", sorted(REGIONS))
def test_region_entry_records_model_load_seconds(cpu_run, region):
    seconds = _entry(cpu_run, region)["model_load_seconds"]
    assert isinstance(seconds, float)
    assert seconds > 0


@pytest.mark.parametrize("region", sorted(REGIONS))
def test_region_entry_records_inference_seconds_per_scan_at_batch_size(cpu_run, region):
    entry = _entry(cpu_run, region)
    assert entry["batch_size"] == BATCH_SIZE
    assert entry["n_scans"] == REGIONS[region][2]
    assert isinstance(entry["inference_seconds_per_scan"], float)
    assert entry["inference_seconds_per_scan"] > 0


@pytest.mark.parametrize("region", sorted(REGIONS))
def test_region_entry_peak_gpu_memory_null_without_cuda(cpu_run, region):
    entry = _entry(cpu_run, region)
    assert "peak_gpu_memory_bytes" in entry
    assert entry["peak_gpu_memory_bytes"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_region_entry_peak_gpu_memory_bytes_with_cuda(dataset):
    run = _run_script(dataset["tmp"] / "cuda", dataset["models"], dataset["crops"])
    for region in REGIONS:
        peak = _entry(run, region)["peak_gpu_memory_bytes"]
        assert isinstance(peak, int)
        assert peak > 0


@pytest.mark.parametrize("region", sorted(REGIONS))
def test_region_entry_records_skeleton_input(cpu_run, dataset, region):
    _, _, n_scans = REGIONS[region]
    skeleton_input = _entry(cpu_run, region)["input"]
    assert Path(skeleton_input["path"]).resolve() == dataset["skeletons"][region].resolve()
    assert skeleton_input["shape"] == [n_scans, 8, 8, 8, 1]
    assert skeleton_input["dtype"] == "int8"
    assert skeleton_input["bytes_per_scan"] == 8 * 8 * 8


def test_report_records_python_torch_startup_seconds(cpu_run):
    seconds = _report(cpu_run)["python_torch_startup_seconds"]
    assert isinstance(seconds, float)
    assert seconds > 0


def test_report_records_stage_command_seconds_per_scan(cpu_run, dataset):
    stages = _report(cpu_run)["stages"]
    assert sorted(stages) == ["morphologist", "tiles"]
    for name, exit_code in (("morphologist", 0), ("tiles", 3)):
        rows = stages[name]
        assert [row["scan"] for row in rows] == list(SCANS)
        for row in rows:
            assert row["exit_code"] == exit_code
            assert isinstance(row["seconds"], float)
            assert row["seconds"] > 0
    for scan in SCANS:
        assert (dataset["marks"] / f"{scan}.done").is_file(), f"stage command not run on {scan}"
