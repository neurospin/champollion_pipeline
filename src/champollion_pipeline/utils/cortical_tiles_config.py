import json
import logging
import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from champollion_pipeline.derivatives_layout import (
    DEFAULT_MASKS_VERSION,
    HEMISPHERES,
    compute_region_model_name,
    compute_region_model_names,
)
from champollion_pipeline.utils.region_masks import REGION_MODEL_NAMES

CROPS_DIRNAME = "crops"
DEFAULT_VOXEL_SIZE = 2.0
INPUT_TYPES_DEFAULT = ("skeleton", "foldlabel")
CROP_MARKER_FILENAME = "crops_complete.marker"
CROP_PASS_SIDES = ("L", "R")  # side of each of HEMISPHERES, in its order
RESOURCES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources")
_MARKER_ORDER = {name: index for index, name in enumerate(REGION_MODEL_NAMES)}


@dataclass
class CorticalTilesConfig:
    masks_version: str
    out_voxel_size: float


class CorticalTilesConfigFactory:
    @classmethod
    def from_args(cls, args) -> "CorticalTilesConfig":
        masks_version = args.masks if args.masks else DEFAULT_MASKS_VERSION
        return CorticalTilesConfig(masks_version=masks_version, out_voxel_size=2.0)

    @classmethod
    def from_pipeline_json(cls, path: str) -> "CorticalTilesConfig | None":
        if not os.path.exists(path):
            return None
        with open(path) as f:
            config = json.load(f)
        masks_version = config.get("masks_version", DEFAULT_MASKS_VERSION)
        return CorticalTilesConfig(masks_version=masks_version, out_voxel_size=2.0)


def compute_versioned_crops_dir(
    derivatives_dir: str, masks_version: str, voxel_size: float = DEFAULT_VOXEL_SIZE
) -> str:
    """Return <derivatives_dir>/crops/<masks_version>/<int(voxel_size)>mm.

    Pure: no I/O, no existence check, no validation of masks_version. Complexity: O(1).
    """
    return os.path.join(derivatives_dir, CROPS_DIRNAME, masks_version, f"{int(voxel_size)}mm")


def versioned_crops_exist(output_dir: str, masks_version: str, voxel_size: float) -> bool:
    crops_path = compute_versioned_crops_dir(output_dir, masks_version, voxel_size)
    if not os.path.isdir(crops_path):
        return False
    return any(True for _ in os.scandir(crops_path))


@dataclass(frozen=True)
class CropPass:
    """One generate_sulcal_regions.py invocation: the Sulcal regions of one side cropped with one Mask version."""

    version: str
    side: str
    regions: tuple[str, ...]


def compute_crop_targets(
    versions: Mapping[str, str] | None, masks: str, regions: Sequence[str]
) -> dict[str, list[str]]:
    """Mask version -> region model names of ``regions`` (left then right, run order) the run crops with it.

    Without a map (``versions`` None) every name goes to ``masks``; with one, each name to its paired version.
    Keys sorted. Raises KeyError for a name the map does not pair.
    """
    targets: dict[str, list[str]] = {}
    for region in dict.fromkeys(regions):
        for hemisphere in HEMISPHERES:
            name = compute_region_model_name(region, hemisphere)
            targets.setdefault(masks if versions is None else versions[name], []).append(name)
    return dict(sorted(targets.items()))


def compute_missing_hemispheres(wanted: Sequence[str], covered: frozenset[str] | None, is_overwrite: bool) -> list[str]:
    """Names of ``wanted`` to crop: all under ``is_overwrite`` or with no marker, else those ``covered`` lacks."""
    is_all = is_overwrite or covered is None
    result = [name for name in wanted if is_all or name not in covered]
    assert len(result) <= len(wanted), (
        "crop_passes_by_mask_version.compute_missing_hemispheres post: len(result) <= len(wanted)"
    )
    return result


def plan_crop_passes(to_crop: Mapping[str, Sequence[str]], regions: Sequence[str]) -> list[CropPass]:
    """One CropPass per (Mask version, side) holding a region model name of ``to_crop``.

    Versions sorted, L before R, regions in ``regions`` order; a name of no region of ``regions`` is ignored.
    """
    result = []
    for version in sorted(to_crop):
        names = set(to_crop[version])
        for hemisphere, side in zip(HEMISPHERES, CROP_PASS_SIDES, strict=True):
            cropped = tuple(
                region for region in dict.fromkeys(regions) if compute_region_model_name(region, hemisphere) in names
            )
            if cropped:
                result.append(CropPass(version, side, cropped))
    assert len(result) <= 2 * len(to_crop), (
        "crop_passes_by_mask_version.plan_crop_passes post: len(result) <= 2 * len(to_crop)"
    )
    return result


def list_regions_holding(names: Iterable[str], regions: Sequence[str]) -> list[str]:
    """Regions of ``regions`` (in order, once) holding a hemisphere whose region model name is in ``names``."""
    wanted = set(names)
    return [
        region
        for region in dict.fromkeys(regions)
        if any(compute_region_model_name(region, hemisphere) in wanted for hemisphere in HEMISPHERES)
    ]


def plan_unmapped_crop(names: Iterable[str], regions: Sequence[str]) -> tuple[list[str] | None, list[str]]:
    """(-r regions, region model names cropped) of a run without a map that must crop ``names``.

    The regions holding a name are cropped, both hemispheres; -r regions is None (the run's own) when that is
    every region of ``regions``.
    """
    cropped = list_regions_holding(names, regions)
    is_partial = len(cropped) < len(dict.fromkeys(regions))
    return (cropped if is_partial else None), compute_region_model_names(cropped)


def check_version_crops(
    derivatives_dir: str,
    crops: CorticalTilesConfig,
    wanted: Sequence[str],
    is_overwrite: bool,
    stage_logger: logging.Logger,
) -> list[str]:
    """Names of ``wanted`` to crop into the ``crops`` folder (Mask version, voxel size): all when it holds nothing,
    else those its marker does not list (all under ``is_overwrite``); one WARNING when crops are already there."""
    version = crops.masks_version
    if not versioned_crops_exist(derivatives_dir, version, crops.out_voxel_size):
        return list(wanted)
    prior = read_marker_names(compute_versioned_crops_dir(derivatives_dir, version, crops.out_voxel_size))
    missing = compute_missing_hemispheres(wanted, prior, is_overwrite)
    if is_overwrite:
        stage_logger.warning(f"Overwriting existing crops for '{version}'.")
    elif not missing:
        stage_logger.warning(f"Crops for mask version '{version}' already exist. Use --overwrite to regenerate.")
    elif prior is None:
        stage_logger.warning(
            f"Crops for mask version '{version}' hold no {CROP_MARKER_FILENAME}: not complete, regenerating them."
        )
    else:
        stage_logger.warning(
            f"{CROP_MARKER_FILENAME} of mask version '{version}' does not list {', '.join(missing)}: "
            "not complete, cropping them."
        )
    return missing


def read_marker_names(crops_dir: str) -> frozenset[str] | None:
    """Region model names crops_dir's crops_complete.marker lists; None without a marker.

    A marker naming no region model name (written before markers listed names) counts as listing the 56;
    a line that is not a region model name is ignored.
    """
    try:
        with open(os.path.join(crops_dir, CROP_MARKER_FILENAME), encoding="utf-8") as marker:
            lines = marker.read().splitlines()
    except FileNotFoundError:
        return None
    names = frozenset(line.strip() for line in lines if line.strip() in _MARKER_ORDER)
    return names or frozenset(REGION_MODEL_NAMES)


def write_marker_names(crops_dir: str, names: Iterable[str]) -> str | None:
    """Write crops_dir's crops_complete.marker, one name per line in REGION_MODEL_NAMES order; return its path.

    Writes nothing (None) when ``names`` holds no region model name: an empty marker means every name.
    """
    listed = sorted({name for name in names if name in _MARKER_ORDER}, key=_MARKER_ORDER.__getitem__)
    if not listed:
        return None
    os.makedirs(crops_dir, exist_ok=True)
    path = os.path.join(crops_dir, CROP_MARKER_FILENAME)
    with open(path, "w", encoding="utf-8") as marker:
        marker.write("".join(f"{name}\n" for name in listed))
    return path


def remove_marker(crops_dir: str) -> frozenset[str]:
    """Delete crops_dir's crops_complete.marker; return the names it listed (none without a marker)."""
    prior = read_marker_names(crops_dir) or frozenset()
    with suppress(FileNotFoundError):
        os.remove(os.path.join(crops_dir, CROP_MARKER_FILENAME))
    return prior


def merge_skip_lists(merged: dict[str, list[Any]], skips: Mapping[str, Iterable[Any]]) -> None:
    """Append to ``merged`` each entry of ``skips`` its side does not hold yet, in file order."""
    for side, entries in skips.items():
        side_entries = merged.setdefault(side, [])
        side_entries.extend(entry for entry in entries if entry not in side_entries)


def migrate_legacy_crops(
    config_path: str, derivatives_abs: str, vox_str: str, is_overwrite: bool, stage_logger: logging.Logger
) -> bool:
    """Move legacy crops/<vox>/ under the masks version of the pipeline JSON at ``config_path``; False on an
    unresolved conflict. Under ``is_overwrite`` a conflict removes both folders. Records go to ``stage_logger``.
    """
    existing = CorticalTilesConfigFactory.from_pipeline_json(config_path)
    legacy_crops = os.path.join(derivatives_abs, CROPS_DIRNAME, vox_str)
    if existing is None or not os.path.isdir(legacy_crops):
        return True
    existing_masks_version = existing.masks_version
    target = os.path.join(derivatives_abs, CROPS_DIRNAME, existing_masks_version, vox_str)
    if not os.path.isdir(target):
        stage_logger.trace(f"Migrating legacy crops to versioned path: crops/{existing_masks_version}/...")
        shutil.move(legacy_crops, target)
        return True
    if not is_overwrite:
        stage_logger.error(
            f"Legacy crops found at crops/{vox_str}/ but versioned crops already exist "
            f"at crops/{existing_masks_version}/{vox_str}/. Please resolve manually or use --overwrite "
            "to remove both and regenerate."
        )
        return False
    stage_logger.warning(
        f"--overwrite: removing legacy crops/{vox_str}/ and existing crops/{existing_masks_version}/{vox_str}/..."
    )
    shutil.rmtree(legacy_crops)
    shutil.rmtree(target)
    return True


def write_crop_config(path: str, input_abs: str, derivatives_abs: str, args: Any, sk_qc_abs: str) -> None:
    """Set the run's paths and options in the pipeline JSON config at ``path``, when it exists.

    generate_sulcal_regions.py derives graphs_dir and output_dir from -d when they are "$local"; -d being the
    output (not the dataset root), both are set explicitly so outputs land in the derivatives folder.
    ``args.path_to_graph`` None (graphs handed over as a --graphs map) writes no graph folder path (JSON null).
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        config = json.load(f)
    config["graphs_dir"] = input_abs
    config["output_dir"] = derivatives_abs
    config["brain_regions_json"] = os.path.join(RESOURCES_DIR, "sulci_regions_champollion_V1.json")
    config["path_to_graph"] = args.path_to_graph
    config["path_to_skeleton_with_hull"] = args.path_sk_with_hull
    config["masks_version"] = args.masks if args.masks else DEFAULT_MASKS_VERSION
    config["skel_qc_path"] = sk_qc_abs
    if args.bids:
        config["bids"] = True
    config.pop("skip_distbottom", None)
    with open(path, "w") as f:
        json.dump(config, f, indent=3)
