"""The run's per-region Mask version map, resolved once before stage 1 (TASK-MODELVER-BDRABCZUK-6C6C79B0F6DE).

Precedence: a given map (champollion-pipeline --region-masks, or a Config file region_masks) > a given Mask version
(--masks-version, or a Config file masks_version without a map) > the map the per-region Mask version resolver reads
from the local model set > dataset.masks_version. A map run gives stages 2 to 4 --region-masks and no --masks; its run
label (the distinct Mask versions of the map, MASK_VERSIONS order, joined by RUN_LABEL_SEPARATOR) becomes
dataset.masks_version, so the stage 4 to 6 folders lie under champollion_V1/<label>.

Refusals (RegionMasksError, one ERROR record and exit 2 in cli.run()): a given map naming two or more Mask versions on
a Hugging Face stage 4 models source, whose subfolders each hold one Mask version; and, given no Mask version, a local
model set holding no model for a region model name the run processes (partial sets are deferred past 1.0.0).

Pure core: reads only the model set files (through the resolver), never exits, logs nothing of its own (the
resolver's one PROVENANCE.csv ERROR record stays).
"""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from champollion_pipeline.config import DatasetConfig
from champollion_pipeline.derivatives_layout import DEFAULT_REGIONS, MASK_VERSIONS, compute_region_model_names
from champollion_pipeline.models.checkpoint import find_region_models
from champollion_pipeline.models.fetch import HuggingFaceStrategy
from champollion_pipeline.models.version_options import (
    MASK_RESOLVED,
    RegionMaskVersion,
    resolve_region_mask_versions,
)
from champollion_pipeline.utils.region_masks import parse_region_masks

RUN_LABEL_SEPARATOR = "_and_"
ORIGIN_MAP = "map"
ORIGIN_VERSION = "version"
ORIGIN_RESOLVED = "resolved"
ORIGIN_DEFAULT = "default"
_ERROR_PREFIX = "region_masks: "
_MODELS_PATH_ERROR_PREFIX = "models_path: "


class RegionMasksError(ValueError):
    """A refusal of the run's Mask versions; run() logs str(error) as one ERROR record and exits 2.

    str(error) is "region_masks: " followed by parse_region_masks' message, which names the faulty value, or by the
    refusal of a given map naming two or more Mask versions on a Hugging Face stage 4 models source; or it is
    "models_path: " followed by the refusal of a local model set holding no model for region model names the run
    processes (given no Mask version), naming those names and a full model set.
    """


@dataclass(frozen=True)
class RunMasks:
    """The run's Mask versions.

    region_masks: region model name -> Mask version the stages get; None for a --masks run.
    label: the run's dataset.masks_version, naming the champollion_V1/<label> folders.
    origin: ORIGIN_MAP, ORIGIN_VERSION, ORIGIN_RESOLVED or ORIGIN_DEFAULT.
    source: the local model set the resolver read, None when it read none.
    entries: the resolver's entries over that set ({} on a given map or without a local set); read by the mismatch
        and unresolved checks of TASK-MODELVER-BDRABCZUK-3B2E4833B088.
    """

    region_masks: Mapping[str, str] | None
    label: str
    origin: str
    source: str | None
    entries: Mapping[str, RegionMaskVersion] = field(default_factory=dict)


def parse_given_region_masks(values: Sequence[str]) -> dict[str, str]:
    """Region model name -> Mask version of the given NAME=VERSION values, in input order; {} for no value.

    Raises:
        RegionMasksError: a value parse_region_masks refuses (its message, naming the value, after "region_masks: ").

    O(len(values)).
    """
    if not values:
        return {}
    try:
        return parse_region_masks(list(values))
    except ValueError as error:
        raise RegionMasksError(_ERROR_PREFIX + str(error)) from error


def compute_run_label(versions: Mapping[str, str]) -> str:
    """The distinct Mask versions of versions, in MASK_VERSIONS order, joined by RUN_LABEL_SEPARATOR.

    A uniform map gives its one Mask version. Pre: versions non-empty, each value in MASK_VERSIONS. O(len(versions)).
    """
    assert len(versions) > 0, "run_masks.compute_run_label pre: len(versions) > 0"
    named = set(versions.values())
    assert named <= set(MASK_VERSIONS), "run_masks.compute_run_label pre: values in MASK_VERSIONS"
    return RUN_LABEL_SEPARATOR.join(version for version in MASK_VERSIONS if version in named)


def compute_crops_versions(dataset: DatasetConfig) -> list[str]:
    """The Mask versions whose crops the run reads, in MASK_VERSIONS order: those of dataset.region_masks when it is
    non-empty, else [dataset.masks_version]. Pre: dataset.region_masks parses. O(56)."""
    if not dataset.region_masks:
        return [dataset.masks_version]
    named = set(parse_region_masks(dataset.region_masks).values())
    return [version for version in MASK_VERSIONS if version in named]


def find_missing_region_models(models_source: str, regions: Sequence[str]) -> list[str]:
    """The region model names the run processes that the local model set models_source holds no model for.

    The names the run processes are the two of each Sulcal region of regions (--regions or a Config file regions
    list), the 56 of DEFAULT_REGIONS when regions is empty. Returned in that order, never sorted. A set
    find_region_models refuses (two models of one region sharing a model id) gives [] (no check: stage 4 refuses it
    with its own error). Pre: models_source is a local directory. O(56 + model folders).
    """
    assert os.path.isdir(models_source), "run_masks.find_missing_region_models pre: models_source is a local directory"
    names = compute_region_model_names(regions or DEFAULT_REGIONS)
    try:
        held = {entry.region for entry in find_region_models(models_source)}
    except ValueError:
        return []
    return [name for name in names if name not in held]


def _is_hugging_face_source(dataset: DatasetConfig, models_source: str | None) -> bool:
    """Whether stage 4 reads its models from Hugging Face: hf_enabled, or a models source that is no local folder and
    that HuggingFaceStrategy (the strategy fetch_model_set tries first) handles. A local folder is never Hugging Face.
    """
    if dataset.hf_enabled:
        return True
    return (
        models_source is not None
        and not os.path.isdir(models_source)
        and HuggingFaceStrategy().can_handle(models_source)
    )


def resolve_run_masks(dataset: DatasetConfig, *, is_version_given: bool, models_source: str | None) -> RunMasks:
    """The run's RunMasks: given map > given Mask version > resolved map of a full local model set > masks_version.

    Args:
        dataset: the run's dataset settings; region_masks already checked by parse_given_region_masks.
        is_version_given: a Mask version is given (--masks-version, or a Config file masks_version without a map).
        models_source: the stage 4 models source; the resolver reads it only when it is a local directory and
            dataset.hf_enabled is false.

    Raises:
        RegionMasksError: a given map naming two or more Mask versions on a Hugging Face stage 4 models source
            ("region_masks: ..."); or, given no Mask version, a local model set holding no model for a region model
            name the run processes ("models_path: ...", checked before the resolver reads the set).

    Post: region_masks is None or label == compute_run_label(region_masks). Reads only the model set files; no exit,
    no logging of its own. O(56 + PROVENANCE.csv rows).
    """
    if dataset.region_masks:
        given = parse_region_masks(dataset.region_masks)
        versions = [version for version in MASK_VERSIONS if version in set(given.values())]
        if len(versions) >= 2 and _is_hugging_face_source(dataset, models_source):
            raise RegionMasksError(
                _ERROR_PREFIX + "a per-region Mask version map naming " + ", ".join(versions) + " needs a local model "
                "set; the stage 4 models source " + (models_source or "hf_enabled") + " is Hugging Face, whose "
                "subfolders each hold one Mask version: pass --models-path <local model set folder>"
            )
        result = RunMasks(given, compute_run_label(given), ORIGIN_MAP, None)
    else:
        result = _resolve_without_map(dataset, is_version_given, models_source)
    assert result.region_masks is None or result.label == compute_run_label(result.region_masks), (
        "run_masks.resolve_run_masks post: region_masks is None or label == compute_run_label(region_masks)"
    )
    return result


def _resolve_without_map(dataset: DatasetConfig, is_version_given: bool, models_source: str | None) -> RunMasks:
    """Steps 2 to 7 of resolve_run_masks: no map was given."""
    is_local = bool(models_source) and not dataset.hf_enabled and os.path.isdir(models_source)
    source = models_source if is_local else None
    if not is_version_given and source is not None:
        missing = find_missing_region_models(source, dataset.regions)
        if missing:
            raise RegionMasksError(
                _MODELS_PATH_ERROR_PREFIX + "the local model set " + source + " holds no model for region model names "
                "the run processes: " + ", ".join(missing) + "; pass a full model set, holding a model for each region "
                "model name the run processes"
            )
    entries = resolve_region_mask_versions(source) if source is not None else {}
    if is_version_given:
        return RunMasks(None, dataset.masks_version, ORIGIN_VERSION, source, entries)
    if entries and all(entry.status == MASK_RESOLVED for entry in entries.values()):
        resolved = {name: entry.version for name, entry in entries.items()}
        return RunMasks(resolved, compute_run_label(resolved), ORIGIN_RESOLVED, source, entries)
    return RunMasks(None, dataset.masks_version, ORIGIN_DEFAULT, source, entries)
