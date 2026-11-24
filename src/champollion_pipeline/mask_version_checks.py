"""The run's Mask versions checked against the local model set before stage 1 (TASK-MODELVER-BDRABCZUK-3B2E4833B088).

Given no Mask version (--masks-version, --region-masks or a Config file), a region model name the run processes that
the local model set gives no single Mask version stops the run. A given Mask version or per-region map differing from
the set's own Mask versions (or a region model name the set gives two Mask versions) stops it too, unless
--allow-version-mismatch: then each such region model name gets one WARNING text and the run goes on.

Pure core: reads RunMasks (run_masks) and the per-region Mask version resolver's entries, lists the local model set's
region model folders (find_region_models), never logs, never exits. cli.py logs the WARNING texts, maps
MaskVersionStop to one ERROR record and exit 2, and keeps stage 4 silent on an accepted mismatch run.
"""

import os
from collections.abc import Sequence

from champollion_pipeline.derivatives_layout import DEFAULT_REGIONS, compute_region_model_names
from champollion_pipeline.models.checkpoint import find_region_models
from champollion_pipeline.models.version_options import MASK_CONFLICT, MASK_RESOLVED, RegionMaskVersion
from champollion_pipeline.run_masks import ORIGIN_DEFAULT, ORIGIN_MAP, ORIGIN_VERSION, RunMasks

ACCEPTED_SUFFIX = "; running anyway (--allow-version-mismatch)"


class MaskVersionStop(ValueError):
    """A stop of the run on its Mask versions; str(error) is the one ERROR record text (exit 2 in cli.run()).

    "models_path: " followed by the region model names the local model set gives no single Mask version, asking for
    --masks-version; or "masks_version: " followed by every region model name whose given Mask version differs from
    the set's, with both Mask versions, asking for --allow-version-mismatch.
    """


def compute_run_region_names(source: str, regions: Sequence[str]) -> tuple[str, ...]:
    """The region model names the run processes that the local model set source holds a model for.

    The names the run processes are the two of each Sulcal region of regions (--regions or a Config file regions
    list), the 56 of DEFAULT_REGIONS when regions is empty; returned in that order, never sorted. A set
    find_region_models refuses (two models of one region sharing a model id) gives () (no check: stage 4 refuses it
    with its own error). Pre: source is a local directory. O(56 + model folders).
    """
    assert os.path.isdir(source), "mask_version_checks.compute_run_region_names pre: source is a local directory"
    names = compute_region_model_names(regions or DEFAULT_REGIONS)
    try:
        held = {entry.region for entry in find_region_models(source)}
    except ValueError:
        return ()
    return tuple(name for name in names if name in held)


def check_run_masks(run_masks: RunMasks, *, run_names: Sequence[str], allow_mismatch: bool) -> tuple[str, ...]:
    """The WARNING texts of the run's accepted Mask version mismatches, one per region model name, in run order.

    Args:
        run_masks: the run's Mask versions, resolved once (run_masks.resolve_run_masks).
        run_names: the region model names the run processes that the local model set holds (compute_run_region_names);
            empty for a Hugging Face source or no local set (no check).
        allow_mismatch: dataset.allow_version_mismatch (--allow-version-mismatch).

    Raises:
        MaskVersionStop: given no Mask version (origin default), a run name the set gives no single Mask version,
            whatever allow_mismatch; or, without allow_mismatch, a given Mask version or map differing from the set's
            for some run name (one text holding every such name).

    Pre: run_masks.source is not None when run_names is non-empty. Post: no logging, no exit. O(len(run_names)).
    """
    assert not run_names or run_masks.source is not None, (
        "mask_version_checks.check_run_masks pre: run_masks.source is not None when run_names is non-empty"
    )
    if not run_names:
        return ()
    if run_masks.origin == ORIGIN_DEFAULT:
        _check_single_versions(run_masks, run_names)
        return ()
    if run_masks.origin not in (ORIGIN_VERSION, ORIGIN_MAP):
        return ()
    findings = _collect_findings(run_masks, run_names)
    if findings and not allow_mismatch:
        raise MaskVersionStop(
            "masks_version: the given Mask version differs from the local model set "
            + str(run_masks.source)
            + ": "
            + "; ".join(name + " given " + given + ", " + found for name, given, found in _stop_parts(findings))
            + "; pass --allow-version-mismatch to run anyway"
        )
    return tuple(_warning_text(name, given, entry) for name, given, entry in findings)


def _check_single_versions(run_masks: RunMasks, run_names: Sequence[str]) -> None:
    """Raise MaskVersionStop naming the run names the set gives no single Mask version (unresolved or conflicting)."""
    unresolved = [name for name in run_names if _entry_status(run_masks, name) != MASK_RESOLVED]
    if unresolved:
        raise MaskVersionStop(
            "models_path: the local model set " + str(run_masks.source) + " gives no single Mask version for region "
            "model names the run processes: " + ", ".join(unresolved) + "; pass --masks-version <V>"
        )


def _entry_status(run_masks: RunMasks, name: str) -> str | None:
    entry = run_masks.entries.get(name)
    return entry.status if entry is not None else None


def _collect_findings(run_masks: RunMasks, run_names: Sequence[str]) -> list[tuple[str, str, RegionMaskVersion]]:
    """(name, given Mask version, resolver entry) of each run name whose given Mask version the set contradicts."""
    findings: list[tuple[str, str, RegionMaskVersion]] = []
    for name in run_names:
        given = _given_version(run_masks, name)
        entry = run_masks.entries.get(name)
        if given is None or entry is None:
            continue
        if (entry.status == MASK_RESOLVED and entry.version != given) or entry.status == MASK_CONFLICT:
            findings.append((name, given, entry))
    return findings


def _given_version(run_masks: RunMasks, name: str) -> str | None:
    """The Mask version given for name: the map's value (None when the map leaves name out), else the run label."""
    if run_masks.origin == ORIGIN_MAP:
        return (run_masks.region_masks or {}).get(name)
    return run_masks.label


def _found_text(entry: RegionMaskVersion) -> str:
    """The set's Mask version of a finding: "trained on K", or "conflicting between A and B"."""
    if entry.status == MASK_CONFLICT:
        return "conflicting between " + " and ".join(entry.named)
    return "trained on " + str(entry.version)


def _stop_parts(findings: list[tuple[str, str, RegionMaskVersion]]) -> list[tuple[str, str, str]]:
    return [(name, given, _found_text(entry)) for name, given, entry in findings]


def _warning_text(name: str, given: str, entry: RegionMaskVersion) -> str:
    """One accepted mismatch: the region model name, the given Mask version and the set's (or its two)."""
    is_conflict = entry.status == MASK_CONFLICT
    found = "conflicting " + " and ".join(entry.named) if is_conflict else str(entry.version)
    return name + ": given Mask version " + given + " differs from the local model set's " + found + ACCEPTED_SUFFIX
