"""The crops each region model of the embeddings stage reads, and its per-region Mask version map (--region-masks).

The crop folder a region model name maps to, the parse-time refusals of the map, the 2mm crops folder of each region
model, the input QC rows over those folders and the provenance key of the map. Without a map every helper leaves the
stage as it was: no refusal, the --masks crops folder for every region, no provenance key.
"""

import argparse
import logging
import os
from collections.abc import Callable, Iterable
from os.path import exists, join

from champollion_pipeline.derivatives_layout import MASK_VERSIONS
from champollion_pipeline.models.embeddings_qc import (
    INPUT_QC_KINDS,
    compute_crops_by_side,
    compute_region_side,
    fetch_region_model_qc_rows,
)
from champollion_pipeline.models.model_run import ModelRun
from champollion_pipeline.models.version_options import find_folder_mask_version
from champollion_pipeline.utils.qc_report import CROP_SIDES, fetch_grouped_crop_qc_rows
from champollion_pipeline.utils.region_masks import REGION_MASKS_DEST, REGION_MASKS_OPTION

REGION_MASK_VERSIONS_KEY = REGION_MASKS_DEST  # provenance.json key of the --region-masks map

logger = logging.getLogger(__name__)


def find_crop_dir(crops_2mm_dir: str, region_model_name: str) -> str:
    """Map a model region name back to its crop directory name.

    Model dirs have dots stripped: crop 'SC-sylv.' → model 'SCsylv_left'.
    Search for a crop dir where dirname.replace('.', '') matches the base name.
    """
    base = region_model_name
    if base.endswith("_left"):
        base = base[:-5]
    elif base.endswith("_right"):
        base = base[:-6]

    if exists(crops_2mm_dir):
        for dname in sorted(os.listdir(crops_2mm_dir)):
            if os.path.isdir(join(crops_2mm_dir, dname)) and dname.replace(".", "") == base:
                return dname
    return base


def get_region_mask_versions(args: argparse.Namespace) -> dict[str, str] | None:
    """The --region-masks map (region model name -> Mask version), None without one (bare test namespaces too)."""
    return getattr(args, REGION_MASKS_DEST, None)


def refuse_legacy_with_map(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Exit 2 naming both options when --legacy is given beside --region-masks."""
    if get_region_mask_versions(args) is not None and getattr(args, "legacy", False):
        parser.error(
            f"--legacy cannot be given together with {REGION_MASKS_OPTION}: --legacy reads the unversioned "
            "deep_folding-2025 crops, the map names a Mask version per region model; drop one of them"
        )


def refuse_unnamed_models_folder(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Exit 2 with the --model-version-or-rename hint for a map run whose local models folder names no version."""
    if get_region_mask_versions(args) is None or args.masks is not None or args.model_version is not None:
        return
    if not os.path.isdir(args.models_path) or find_folder_mask_version(args.models_path) is not None:
        return
    message = (
        f"{REGION_MASKS_OPTION} given without --model-version and the local models folder {args.models_path!r} "
        f"is not named after a Mask version ({', '.join(MASK_VERSIONS)}): pass --model-version <version> to "
        "label the run, or rename the folder after a Mask version"
    )
    logger.error(message)
    parser.error(message)


def compute_map_provenance(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    """{region_mask_versions: the map, pair order} with a --region-masks map; {} without one."""
    versions = get_region_mask_versions(args)
    return {} if versions is None else {REGION_MASK_VERSIONS_KEY: dict(versions)}


def refuse_unmapped_regions(
    args: argparse.Namespace, parser: argparse.ArgumentParser, get_run: Callable[[], ModelRun]
) -> None:
    """Exit 2 naming the region model names of the run that the --region-masks map does not name.

    ``get_run`` is called only with a map, so a run without one resolves its models where it did before.
    """
    versions = get_region_mask_versions(args)
    if versions is None:
        return
    regions = dict.fromkeys(entry.region for entry in get_run().entries)
    unmapped = [region for region in regions if region not in versions]
    if unmapped:
        parser.error(
            f"{REGION_MASKS_OPTION} names no Mask version for the region model(s) {', '.join(unmapped)} found "
            f"under {args.models_path}: add a NAME=VERSION pair for each, or leave them out with --regions"
        )


def compute_version_crops_dir(datasets_root: str, derivatives_folder: str, version: str) -> str:
    """<datasets_root>/derivatives/<derivatives_folder>/crops/<version>/2mm."""
    return join(datasets_root.rstrip("/"), "derivatives", derivatives_folder, "crops", version, "2mm")


def compute_region_crops_dir(args: argparse.Namespace, derivatives_folder: str, crops_2mm_dir: str, region: str) -> str:
    """The 2mm crops folder of region model name ``region``: its mapped Mask version's, else crops_2mm_dir."""
    versions = get_region_mask_versions(args)
    if versions is None:
        return crops_2mm_dir
    return compute_version_crops_dir(args.datasets_root, derivatives_folder, versions[region])


def fetch_input_qc_rows(
    args: argparse.Namespace,
    derivatives_folder: str,
    crops_2mm_dir: str,
    regions: Iterable[str],
    find_crop: Callable[[str, str], str] = find_crop_dir,
) -> list[dict[str, object]]:
    """Input QC rows of ``regions``: over crops_2mm_dir without a map, else over each region's mapped crops folder.

    ``find_crop(crops_dir, region)`` names a region's crop folder in a crops folder.
    """
    regions = list(dict.fromkeys(regions))
    versions = get_region_mask_versions(args)
    if versions is None:
        crops = {region: find_crop(crops_2mm_dir, region) for region in regions}
        return fetch_region_model_qc_rows(crops_2mm_dir, compute_crops_by_side(crops))
    groups = []  # one group per (Mask version, side): versions sorted, sides in CROP_SIDES order
    for version in sorted({versions[region] for region in regions}):
        crops_dir = compute_version_crops_dir(args.datasets_root, derivatives_folder, version)
        for side in CROP_SIDES:
            crop_names = [
                find_crop(crops_dir, region)
                for region in regions
                if versions[region] == version and compute_region_side(region) == side
            ]
            if crop_names:
                groups.append((crops_dir, side, list(dict.fromkeys(crop_names))))
    return fetch_grouped_crop_qc_rows(groups, INPUT_QC_KINDS)
