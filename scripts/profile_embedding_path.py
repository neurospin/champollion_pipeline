#!/usr/bin/env python3
"""Profile the embedding path: per-region model size, load and inference time, GPU memory, inputs, stage cost.

Plain measurements, written as one JSON report:

- per region model directory under MODELS: checkpoint bytes, model load seconds, inference seconds per scan
  at --batch-size, peak CUDA memory bytes (null on CPU), and the skeleton input it reads under CROPS_2MM;
- the seconds a fresh interpreter takes to start and import torch (what each evaluate.py subprocess pays);
- for each --stage NAME=CMD, the seconds and exit code of CMD per --scan, with {scan} replaced by the scan id.

Models are loaded with champollion_V1's own load_model and run with its generate_embeddings (imported, not
modified), from the same epoch-N checkpoint the embeddings stage selects.

Usage:
    profile_embedding_path.py MODELS CROPS_2MM --output report.json [--batch-size 32] [--cpu]
        [--stage morphologist='cmd {scan}'] [--stage tiles='cmd {scan}'] [--scan sub-01]...
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import shlex
import subprocess
import sys
import time
from os.path import join
from pathlib import Path

import numpy as np
import torch
from champollion.evaluate import generate_embeddings, load_data, load_model
from champollion_utils.logger import OUTPUT, setup_logging

from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from champollion_pipeline.models.checkpoint import (
    DEFAULT_CHECKPOINT_EPOCH,
    evaluation_model_dir,
    find_epoch_checkpoint,
    find_region_model_dirs,
    resolve_region_name,
)

logger = logging.getLogger("champollion_pipeline.profile_embedding_path")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", help="Directory holding the region model directories.")
    parser.add_argument("crops_2mm", help="Crops directory: <crop>/mask/{L|R}skeleton.npy per region.")
    parser.add_argument("--output", required=True, help="JSON report to write.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size (default: 32).")
    parser.add_argument("--epoch", type=int, default=DEFAULT_CHECKPOINT_EPOCH, help="Checkpoint epoch to load.")
    parser.add_argument("--cpu", action="store_true", help="Run inference on CPU even when CUDA is available.")
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        metavar="NAME=CMD",
        help="Stage command to time per scan; {scan} in CMD is replaced by the scan id. Repeatable.",
    )
    parser.add_argument("--scan", action="append", default=[], help="Scan id the stage commands run on. Repeatable.")
    return parser.parse_args(argv)


def _python_torch_startup_seconds() -> float:
    """Seconds for a fresh interpreter to start and import torch."""
    start = time.perf_counter()
    subprocess.run([sys.executable, "-c", "import torch"], check=True)
    return time.perf_counter() - start


def _skeleton_input(skels_path: str) -> dict:
    """Path, shape, element type and bytes per scan of one skeleton array, read without loading it."""
    array = np.load(skels_path, mmap_mode="r")
    return {
        "path": skels_path,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes_per_scan": int(array.itemsize * np.prod(array.shape[1:])),
    }


def _profile_region(model_path: str, models: str, crops_2mm: str, args: argparse.Namespace) -> dict:
    """Load one region model and embed its skeletons, timing each step."""
    region = resolve_region_name(model_path, models)
    side = "L" if region.endswith("_left") else "R"
    # The embeddings stage's own crop lookup; it reads no instance state.
    crop_name = GenerateEmbeddings._find_crop_dir(None, crops_2mm, region)
    skels_path = join(crops_2mm, crop_name, "mask", f"{side}skeleton.npy")
    ckpt = find_epoch_checkpoint(model_path, args.epoch)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    dataloader, in_shape = load_data(skels_path, batch_size=args.batch_size)
    with evaluation_model_dir(model_path, args.epoch) as eval_model_path:
        start = time.perf_counter()
        # load_model prints every state_dict key; keep that out of the profile's own output.
        with contextlib.redirect_stdout(sys.stderr):
            model = load_model(eval_model_path, in_shape).to(device)
        load_seconds = time.perf_counter() - start

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    generate_embeddings(model, dataloader, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    n_scans = len(dataloader.dataset)
    inference_seconds_per_scan = (time.perf_counter() - start) / n_scans
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None

    logger.log(OUTPUT, f"{region}: load {load_seconds:.3f} s, {inference_seconds_per_scan:.4f} s/scan")
    return {
        "region": region,
        "checkpoint_bytes": ckpt.stat().st_size if ckpt is not None else None,
        "model_load_seconds": load_seconds,
        "batch_size": args.batch_size,
        "n_scans": n_scans,
        "inference_seconds_per_scan": inference_seconds_per_scan,
        "peak_gpu_memory_bytes": peak,
        "input": _skeleton_input(skels_path),
    }


def _time_stages(stages: list[str], scans: list[str]) -> dict:
    """Run every NAME=CMD once per scan, without a shell, and record seconds and exit code."""
    timings: dict[str, list[dict]] = {}
    for stage in stages:
        name, sep, command = stage.partition("=")
        if not sep or not name or not command:
            raise ValueError(f"--stage expects NAME=CMD, got {stage!r}")
        rows = timings.setdefault(name, [])
        for scan in scans:
            argv = [part.replace("{scan}", scan) for part in shlex.split(command)]
            start = time.perf_counter()
            exit_code = subprocess.run(argv).returncode
            rows.append({"scan": scan, "seconds": time.perf_counter() - start, "exit_code": exit_code})
            logger.log(OUTPUT, f"{name} {scan}: exit {exit_code}")
    return timings


def main(argv: list[str] | None = None) -> int:
    setup_logging(logger)
    args = _parse_args(argv)
    region_dirs = find_region_model_dirs(args.models)
    report = {
        "regions": [_profile_region(path, args.models, args.crops_2mm, args) for path in region_dirs],
        "python_torch_startup_seconds": _python_torch_startup_seconds(),
        "stages": _time_stages(args.stage, args.scan),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    logger.log(OUTPUT, f"Profile written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
