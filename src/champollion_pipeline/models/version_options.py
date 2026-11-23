"""Parse-time checks of the embeddings stage's model version options, and the Mask version of each region model of a
local model set (from its model_set.json, else its PROVENANCE.csv, else its folder name)."""

import argparse
import csv
import logging
import os
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from os.path import join

from champollion_pipeline.derivatives_layout import (
    DEFAULT_MASKS_VERSION,
    DEFAULT_REGIONS,
    MASK_VERSIONS,
    compute_region_model_names,
)
from champollion_pipeline.models.cache import MODEL_SET_FILE_NAME, read_model_set
from champollion_pipeline.models.checkpoint import find_region_models
from champollion_pipeline.utils.region_masks import REGION_MASKS_DEST

MASK_SOURCE_MODEL_SET = MODEL_SET_FILE_NAME
MASK_SOURCE_FOLDER_NAME = "folder name"
PROVENANCE_FILE_NAME = "PROVENANCE.csv"
MASK_SOURCE_PROVENANCE = PROVENANCE_FILE_NAME
PROVENANCE_REGION_COLUMN = "region"
PROVENANCE_VERSION_COLUMN = "mask_version_trained_on"
PROVENANCE_REQUIRED_COLUMNS = (PROVENANCE_REGION_COLUMN, PROVENANCE_VERSION_COLUMN)
MASK_RESOLVED = "resolved"
MASK_UNRESOLVED = "unresolved"
MASK_CONFLICT = "conflict"

logger = logging.getLogger(__name__)


def resolve_model_version_alias(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Copy --masks-version into model_version with a FutureWarning; exit 2 if the two flags differ."""
    alias = args.masks_version
    if alias is None:
        return
    model_version = args.model_version
    if model_version is not None and model_version != alias:
        parser.error(
            f"--model-version {model_version!r} and --masks-version {alias!r} differ; --masks-version is a "
            "deprecated alias of --model-version, pass only --model-version"
        )
    warnings.warn(
        "--masks-version is deprecated; use --model-version instead "
        "(same value, the Hugging Face model-set subfolder).",
        FutureWarning,
        stacklevel=2,
    )
    args.model_version = alias


def find_folder_mask_version(models_path: str) -> str | None:
    """The last folder name of models_path when it is one of MASK_VERSIONS, else None."""
    folder_name = os.path.basename(os.path.normpath(models_path))
    return folder_name if folder_name in MASK_VERSIONS else None


@dataclass(frozen=True)
class RegionMaskVersion:
    """Mask version of one region model of a local model set.

    status: MASK_RESOLVED, MASK_UNRESOLVED or MASK_CONFLICT; version and source are set only when resolved;
    named: the distinct values the sources gave for the region, in precedence order (empty when no source names one);
    reason: the read error of an unreadable model_set.json, or the read error or missing column(s) of PROVENANCE.csv
    (then unresolved, named empty), else None.
    """

    status: str
    version: str | None
    source: str | None
    named: tuple[str, ...]
    reason: str | None = None


def _fetch_marker_masks(models_path: str) -> tuple[str | None, str | None]:
    """(masks, None) from models_path/model_set.json; (None, None) when absent; (None, read error) when unreadable."""
    if not os.path.exists(join(models_path, MODEL_SET_FILE_NAME)):
        return None, None
    try:
        return read_model_set(models_path).masks, None
    except (ValueError, OSError) as err:
        return None, str(err)


def _compute_region_mask_version(sources: tuple[tuple[str, str | None], ...]) -> RegionMaskVersion:
    """Decide one region's Mask version from its (source, value) pairs, in precedence order; value None: no value.

    Any value outside MASK_VERSIONS, or no value: unresolved; two distinct values: conflict; else resolved by the
    first source giving the value. O(len(sources)).
    """
    named = tuple(dict.fromkeys(value for _, value in sources if value is not None))
    if not named or any(value not in MASK_VERSIONS for value in named):
        return RegionMaskVersion(MASK_UNRESOLVED, None, None, named)
    if len(named) > 1:
        return RegionMaskVersion(MASK_CONFLICT, None, None, named)
    source = next(name for name, value in sources if value is not None)
    return RegionMaskVersion(MASK_RESOLVED, named[0], source, named)


def _compute_provenance_versions(rows: Iterable[Mapping[str, str]]) -> dict[str, tuple[str, ...]]:
    """Per exact region value, the sorted distinct non-empty mask_version_trained_on values; O(len(rows) log).

    A region value with only empty values has no key. Sorting makes the result independent of the row order.
    """
    values: dict[str, set[str]] = {}
    for row in rows:
        version = row[PROVENANCE_VERSION_COLUMN]
        if version:
            values.setdefault(row[PROVENANCE_REGION_COLUMN], set()).add(version)
    return {region: tuple(sorted(versions)) for region, versions in values.items()}


def _fetch_provenance_versions(models_path: str) -> tuple[dict[str, tuple[str, ...]], str | None]:
    """(region -> sorted Mask versions, None) from models_path/PROVENANCE.csv; ({}, None) when absent;
    ({}, reason naming the path) when it cannot be read or its header lacks a required column."""
    path = join(models_path, PROVENANCE_FILE_NAME)
    if not os.path.exists(path):
        return {}, None
    try:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, restval="")
            columns = reader.fieldnames or []
            missing = [column for column in PROVENANCE_REQUIRED_COLUMNS if column not in columns]
            if missing:
                return {}, f"{path}: header lacks the column(s) {', '.join(missing)}"
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as err:
        return {}, f"{path}: cannot be read ({err})"
    return _compute_provenance_versions(rows), None


def _compute_region_sources(
    provenance_values: tuple[str, ...], folder_version: str | None
) -> tuple[tuple[str, str | None], ...]:
    """The (source, value) pairs of one region model name: its PROVENANCE.csv values when it has any (the folder
    name is then not a source), else the folder name alone."""
    if provenance_values:
        return tuple((MASK_SOURCE_PROVENANCE, value) for value in provenance_values)
    return ((MASK_SOURCE_FOLDER_NAME, folder_version),)


def _compute_provenance_entries(
    models_path: str, names: tuple[str, ...], folder_version: str | None
) -> dict[str, RegionMaskVersion]:
    """The entries of a model set without model_set.json: PROVENANCE.csv rows, else the folder name, per name.

    An unreadable PROVENANCE.csv, or one lacking a required column, leaves every name unresolved with the reason
    and logs one ERROR record naming its path; no folder-name fallback then.
    """
    versions, reason = _fetch_provenance_versions(models_path)
    if reason is not None:
        logger.error(f"{reason}; every region model name of {models_path} is unresolved")
        entry = RegionMaskVersion(MASK_UNRESOLVED, None, None, (), reason)
        return dict.fromkeys(names, entry)
    return {
        name: _compute_region_mask_version(_compute_region_sources(versions.get(name, ()), folder_version))
        for name in names
    }


def resolve_region_mask_versions(models_path: str) -> dict[str, RegionMaskVersion]:
    """Mask version of each of the 56 region model names of the local model set at models_path, with its source.

    Args:
        models_path: a local model set directory (the caller checks it is one).

    Returns:
        region model name (compute_region_model_names(DEFAULT_REGIONS) order) -> RegionMaskVersion. Precedence:
        model_set.json masks (PROVENANCE.csv then not opened) > the PROVENANCE.csv rows of the name > the folder
        name (only for a name neither file names). Reads model_set.json and PROVENANCE.csv only; logs one ERROR
        record naming PROVENANCE.csv when it is unreadable or lacks a required column, nothing else; no exit, never
        raises for a directory. O(56 + rows).
    """
    names = tuple(compute_region_model_names(DEFAULT_REGIONS))
    marker_masks, marker_error = _fetch_marker_masks(models_path)
    folder_version = find_folder_mask_version(models_path)
    if marker_error is not None:
        result = dict.fromkeys(names, RegionMaskVersion(MASK_UNRESOLVED, None, None, (), marker_error))
    elif marker_masks is not None:
        entry = _compute_region_mask_version(
            ((MASK_SOURCE_MODEL_SET, marker_masks), (MASK_SOURCE_FOLDER_NAME, folder_version))
        )
        result = dict.fromkeys(names, entry)
    else:
        result = _compute_provenance_entries(models_path, names, folder_version)
    assert len(result) == 56, "region_mask_version_resolver.resolve_region_mask_versions post: len(result) == 56"
    return result


def resolve_masks_default(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Fill --masks when it was not given: the model version if set; else, for a local models directory, its folder
    name when it is a Mask version (exit 2 asking for --masks when it is not); else (models_path not a local
    directory) DEFAULT_MASKS_VERSION for both --masks and --model-version, so the downloaded model set matches."""
    if args.masks is not None:
        return
    if args.model_version is not None:
        args.masks = args.model_version
        return
    if not os.path.isdir(args.models_path):
        args.masks = args.model_version = DEFAULT_MASKS_VERSION
        return
    folder_version = find_folder_mask_version(args.models_path)
    if folder_version is None:
        message = (
            f"--masks not given and the local models folder {args.models_path!r} is not named after a mask version "
            f"({', '.join(MASK_VERSIONS)}); pass --masks <mask version of the crops to embed>"
        )
        logger.error(message)
        parser.error(message)
    args.masks = folder_version


def check_version_mismatch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Exit 2 when --masks and --model-version differ, unless --allow-version-mismatch (then log one WARNING)."""
    model_version, masks = args.model_version, args.masks
    if model_version is None or masks == model_version:
        return
    if not args.allow_version_mismatch:
        parser.error(
            f"--masks {masks!r} and --model-version {model_version!r} differ: --masks picks the input crops, "
            "--model-version the model set; pass --allow-version-mismatch to run with both"
        )
    logger.warning(
        f"--masks {masks!r} and --model-version {model_version!r} differ; running anyway "
        f"(--allow-version-mismatch): crops {masks!r}, models {model_version!r}"
    )


def find_run_region_names(args: argparse.Namespace) -> tuple[str, ...]:
    """The sorted distinct region model names of the run's models: the regions of find_region_models(models_path),
    only those named by --regions when it is given; () when find_region_models raises ValueError (resolve_model_run
    refuses that run when it starts). O(n log n) in the number of model directories."""
    try:
        entries = find_region_models(args.models_path)
    except ValueError:
        return ()
    regions = getattr(args, "regions", None)
    return tuple(sorted({entry.region for entry in entries if regions is None or entry.region in regions}))


def _read_local_marker_masks(models_dir: str, parser: argparse.ArgumentParser) -> str | None:
    """The masks of models_dir/model_set.json, None when absent; exit 2 naming it when read_model_set refuses it."""
    if not os.path.isfile(join(models_dir, MODEL_SET_FILE_NAME)):
        return None
    try:
        return read_model_set(models_dir).masks
    except ValueError as err:
        parser.error(str(err))


def _describe_region_mismatch(
    name: str, entry: RegionMaskVersion, mapped: str | None, masks: str | None, has_marker: bool
) -> str | None:
    """The mismatch text of one resolved or conflicting region model name, None when it agrees or is not checked.

    masks is --masks without a --region-masks map, None with one (mapped is then the map's value for name)."""
    if entry.status == MASK_CONFLICT:
        return f"{name}: the model set names {' and '.join(repr(value) for value in entry.named)}"
    version = entry.version
    if masks is None and mapped is not None and mapped != version:
        return f"{name}: --region-masks names {mapped!r}, the model set {version!r} ({entry.source})"
    if masks is not None and not has_marker and version != masks:
        return f"{name}: the model set names {version!r} ({entry.source}), --masks {masks!r}"
    return None


def _collect_region_findings(
    args: argparse.Namespace, region_map: Mapping[str, str] | None, has_marker: bool
) -> tuple[list[str], list[str]]:
    """(mismatch texts, unresolved region model names) over the run's region model names, in sorted order."""
    names = find_run_region_names(args)
    if not names:
        return [], []
    entries = resolve_region_mask_versions(args.models_path)
    masks = args.masks if region_map is None else None
    mismatches: list[str] = []
    unresolved: list[str] = []
    for name in names:
        entry = entries.get(name)
        if entry is None or entry.status == MASK_UNRESOLVED:
            unresolved.append(name)
            continue
        mapped = None if region_map is None else region_map.get(name)
        text = _describe_region_mismatch(name, entry, mapped, masks, has_marker)
        if text is not None:
            mismatches.append(text)
    return mismatches, unresolved


def _format_unresolved_record(
    models_dir: str, unresolved: list[str], region_map: Mapping[str, str] | None, masks: str
) -> str:
    """The one WARNING record text naming models_dir, the unresolved region model names and their crops version."""
    if region_map is None:
        names, tail = ", ".join(unresolved), f"from the --masks {masks!r} crops unchecked"
    else:
        names = ", ".join(
            f"{name} (--region-masks {region_map[name]!r})" if name in region_map else f"{name} (not in --region-masks)"
            for name in unresolved
        )
        tail = "from their mapped crops unchecked"
    return (
        f"{models_dir}: no known Mask version for the region model name(s) {names} in model_set.json, "
        f"PROVENANCE.csv or the folder name; embedding them {tail}"
    )


def check_local_model_set(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Check a local model set against the crops' Mask version: its model_set.json masks against --masks (without a
    --region-masks map), then each region model name of the run against --masks or its mapped Mask version.

    Not a local directory: nothing. An invalid model_set.json: exit 2 naming it. Mismatches: one exit 2 naming each,
    or with --allow-version-mismatch one WARNING record per mismatch. Unresolved region model names (and no exit):
    one WARNING record naming them. Never changes the namespace."""
    models_dir, masks = args.models_path, args.masks
    if not os.path.isdir(models_dir):
        return
    path = join(models_dir, MODEL_SET_FILE_NAME)
    marker_masks = _read_local_marker_masks(models_dir, parser)
    region_map = getattr(args, REGION_MASKS_DEST, None)
    mismatches, unresolved = _collect_region_findings(args, region_map, marker_masks is not None)
    is_marker_mismatch = region_map is None and marker_masks is not None and marker_masks != masks
    if (is_marker_mismatch or mismatches) and not args.allow_version_mismatch:
        parts: list[str] = []
        if is_marker_mismatch:
            parts.append(
                f"{path} masks {marker_masks!r} and --masks {masks!r} differ: the local model set was trained on "
                f"{marker_masks!r} crops; pass --masks {marker_masks!r}, or --allow-version-mismatch to run with both"
            )
        if mismatches:
            parts.append(
                f"Mask version mismatch under {models_dir}: {'; '.join(mismatches)}; pass crops of the model's Mask "
                "version, or --allow-version-mismatch to run with both"
            )
        parser.error("; ".join(parts))
    if is_marker_mismatch:
        logger.warning(
            f"{path} masks {marker_masks!r} and --masks {masks!r} differ; running anyway (--allow-version-mismatch): "
            f"crops {masks!r}, models trained on {marker_masks!r}"
        )
    for text in mismatches:
        logger.warning(f"{text}; running anyway (--allow-version-mismatch)")
    if unresolved:
        logger.warning(_format_unresolved_record(models_dir, unresolved, region_map, masks))


def find_marker_model_version(fetched_models_path: str) -> str | None:
    """The model_version of the fetched model set's model_set.json, or None when absent or unreadable."""
    path = join(fetched_models_path, MODEL_SET_FILE_NAME)
    if not os.path.isfile(path):
        return None
    try:
        return read_model_set(fetched_models_path).model_version
    except (ValueError, OSError) as err:
        warnings.warn(
            f"Could not read {path} for provenance.json ({err}); model_version falls back to --model-version",
            UserWarning,
            stacklevel=2,
        )
        return None
