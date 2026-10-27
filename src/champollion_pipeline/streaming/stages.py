"""Per-scan producer runners of a streaming run: Morphologist graph and cortical_tiles for one scan.

compute_scan_tiles_dir and find_scan_crops_root are the only place the per-scan tiles layout
<cortical_tiles_output>/scans/<str(scan)> and its crops root are computed; the barrier, the dataset
config and the consumer import them.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Protocol

from champollion_pipeline.config import DatasetConfig
from champollion_pipeline.generate_morphologist_graphs import GenerateMorphologistGraphs
from champollion_pipeline.run_cortical_tiles import RunCorticalTiles
from champollion_pipeline.utils.lib import DERIVATIVES_FOLDER

from .settings import StreamingSettings

logger = logging.getLogger(__name__)

SCAN_TILES_DIRNAME = "scans"


class Scan(Protocol):
    """What the producer runners read from a scan identity (ScanId satisfies it)."""

    @property
    def path_prefix(self) -> str:
        """Subject, then '/' and the session when the session is set."""
        ...


def make_single_scan_dir(subjects_dir: str, path_prefix: str) -> tempfile.TemporaryDirectory:
    """Return a temporary directory whose only content links <path_prefix> to <subjects_dir>/<path_prefix>."""
    tmp = tempfile.TemporaryDirectory(prefix="champollion_scan_")
    link = os.path.join(tmp.name, path_prefix)
    try:
        os.makedirs(os.path.dirname(link), exist_ok=True)
        os.symlink(os.path.abspath(os.path.join(subjects_dir, path_prefix)), link, target_is_directory=True)
    except OSError:
        tmp.cleanup()
        raise
    return tmp


def compute_scan_tiles_dir(cortical_tiles_output: str, scan: Scan) -> str:
    """Return <cortical_tiles_output>/scans/<str(scan)>."""
    return os.path.join(cortical_tiles_output, SCAN_TILES_DIRNAME, str(scan))


def find_scan_crops_root(scan_tiles_dir: str, masks_version: str) -> str | None:
    """Return the 2mm crops directory of one scan's tiles directory, or None when it does not exist."""
    crops = os.path.join(scan_tiles_dir, DERIVATIVES_FOLDER, "crops", masks_version, "2mm")
    return crops if os.path.isdir(crops) else None


def _cortical_tiles_argv(
    input_dir: str, output_dir: str, settings: StreamingSettings, dataset: DatasetConfig
) -> list[str]:
    """RunCorticalTiles arguments of one scan, in main.py's order, with the per-worker job count."""
    argv = [
        input_dir,
        output_dir,
        f"--path_to_graph={dataset.path_to_graph}",
        f"--path_sk_with_hull={dataset.path_sk_with_hull}",
        f"--njobs={settings.njobs_per_worker}",
        f"--masks={dataset.masks_version}",
    ]
    if dataset.skel_seed is not None:
        argv.append(f"--skel-seed={dataset.skel_seed}")
    if dataset.sk_qc_path:
        argv.append(f"--sk_qc_path={dataset.sk_qc_path}")
    if dataset.bids:
        argv.append("--bids")
    if dataset.regions:
        argv += ["--regions", *dataset.regions]
    return argv


def run_cortical_tiles_for_scan(scan: Scan, settings: StreamingSettings, dataset: DatasetConfig) -> None:
    """Run cortical_tiles on one scan; a non-zero return code raises RuntimeError."""
    scan_input = os.path.join(settings.subjects_dir, scan.path_prefix)
    if not os.path.isdir(scan_input):
        raise FileNotFoundError(f"cortical_tiles input for scan {scan} not found: {scan_input}")
    out = compute_scan_tiles_dir(dataset.cortical_tiles_output, scan)
    os.makedirs(out, exist_ok=True)
    with make_single_scan_dir(settings.subjects_dir, scan.path_prefix) as input_dir:
        logger.trace(f"cortical_tiles: scan {scan}")
        script = RunCorticalTiles()
        script.parse_args(_cortical_tiles_argv(input_dir, out, settings, dataset))
        code = script.run()
    if code != 0:
        raise RuntimeError(f"cortical_tiles failed for scan {scan} with return code {code}")
    crops = find_scan_crops_root(out, dataset.masks_version)
    if crops is None:
        logger.warning(f"cortical_tiles QC: no {dataset.masks_version} 2mm crops for scan {scan} under {out}")
        return
    logger.output(f"cortical_tiles crops for scan {scan}: {crops}")


def run_morphologist_for_scan(raw_scan: str, settings: StreamingSettings, dataset: DatasetConfig) -> None:  # noqa: V107 - uniform producer runner signature
    """Run Morphologist on one raw scan file; a non-zero return code raises RuntimeError."""
    if not os.path.isfile(raw_scan):
        raise FileNotFoundError(f"Morphologist raw scan not found: {raw_scan}")
    name = os.path.basename(raw_scan)
    with tempfile.TemporaryDirectory(prefix="champollion_raw_") as input_dir:
        os.symlink(os.path.abspath(raw_scan), os.path.join(input_dir, name))
        argv = [input_dir, dataset.morphologist_graphs] + (["--bids"] if dataset.bids else [])
        logger.trace(f"Morphologist: raw scan {name}")
        script = GenerateMorphologistGraphs()
        script.parse_args(argv)
        code = script.run()
    if code != 0:
        raise RuntimeError(f"Morphologist failed for raw scan {name} with return code {code}")
    logger.output(f"Morphologist done for raw scan {name}")
