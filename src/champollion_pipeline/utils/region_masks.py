"""The --region-masks option shared by stages 2, 3 and 4: one Mask version per region model name.

A stage adds the option with add_region_masks_option after declaring --masks; argparse turns the NAME=VERSION
pairs into args.region_mask_versions, a dict region model name -> Mask version in the order of the pairs (None
without the option), or exits 2 naming the faulty pair. refuse_masks_with_region_masks then refuses --masks
given beside the map. Pure: no I/O, no logging.
"""

import argparse
from collections.abc import Iterable, Mapping, Sequence

from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.derivatives_layout import (
    DEFAULT_REGIONS,
    HEMISPHERES,
    MASK_VERSIONS,
    compute_region_model_name,
    compute_region_model_names,
)

REGION_MASKS_OPTION = "--region-masks"
REGION_MASKS_DEST = "region_mask_versions"
REGION_MASKS_METAVAR = "NAME=VERSION"
MASKS_DEST = "masks"  # dest of every stage's --masks
# The 56 region model names a pair may give: both hemispheres of each of the 28 default Sulcal regions.
REGION_MODEL_NAMES: tuple[str, ...] = tuple(compute_region_model_names(DEFAULT_REGIONS))
_REGION_MODEL_PARTS: dict[str, tuple[str, str]] = {
    compute_region_model_name(region, hemisphere): (region, hemisphere)
    for region in DEFAULT_REGIONS
    for hemisphere in HEMISPHERES
}
_REGION_MASKS_HELP = (
    "Mask version of each region model, as NAME=VERSION pairs (e.g. SC-sylv_left=canonical_25); NAME is a Sulcal "
    f"region without dots, then _left or _right; VERSION one of {', '.join(MASK_VERSIONS)}. Not with --masks."
)
_MASKS_WITH_MAP_ERROR = (
    "--masks cannot be given together with --region-masks: the map names the Mask version of each region model "
    "name; drop --masks"
)


class DeclaredMasksDefault(str):
    """The --masks default a stage declares, marked: equal to the plain value, unlike a --masks typed by the user."""

    __slots__ = ()


def _check_pair(value: str, versions: Mapping[str, str]) -> tuple[str, str]:
    """(name, version) of one NAME=VERSION value; ValueError naming the value, name or version at fault."""
    name, has_equals, version = value.partition("=")
    if not has_equals:
        raise ValueError(f"{value!r} is not NAME=VERSION (no '=')")
    if not name or not version:
        raise ValueError(f"{value!r} is not NAME=VERSION (empty NAME or VERSION)")
    if name not in REGION_MODEL_NAMES:
        raise ValueError(
            f"unknown region model name {name!r} in {value!r} "
            "(a Sulcal region without dots, then _left or _right, e.g. SC-sylv_left)"
        )
    if name in versions:
        raise ValueError(f"region model name {name!r} given twice")
    if version not in MASK_VERSIONS:
        raise ValueError(f"unknown Mask version {version!r} in {value!r} (known: {', '.join(MASK_VERSIONS)})")
    return name, version


def parse_region_masks(values: Sequence[str]) -> dict[str, str]:
    """Region model name -> Mask version of NAME=VERSION values, in input order.

    Raises ValueError on the first value without '=', with an empty side, an unknown region model name, a name
    already given or a version outside MASK_VERSIONS.
    """
    assert len(values) > 0, "region_masks_option.parse_region_masks pre: len(values) > 0"
    result: dict[str, str] = {}
    for value in values:
        name, version = _check_pair(value, result)
        result[name] = version
    assert len(result) == len(values), "region_masks_option.parse_region_masks post: len(result) == len(values)"
    return result


class RegionMasksAction(argparse.Action):
    """Store the parsed map; a second occurrence of the option adds its pairs after the first one's."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[str] | None,
        option_string: str | None = None,
    ) -> None:
        del option_string  # argparse passes it; the messages name the option through self
        earlier = getattr(namespace, self.dest, None) or {}
        pairs = [values] if isinstance(values, str) else list(values or [])
        try:
            parsed = parse_region_masks(pairs)
        except ValueError as err:
            raise argparse.ArgumentError(self, str(err)) from err
        repeated = next((name for name in parsed if name in earlier), None)
        if repeated is not None:
            raise argparse.ArgumentError(self, f"region model name {repeated!r} given twice")
        setattr(namespace, self.dest, {**earlier, **parsed})


def add_region_masks_option(builder: ScriptBuilder) -> ScriptBuilder:
    """Add --region-masks NAME=VERSION [NAME=VERSION ...] to builder: args.region_mask_versions, None when not given.

    Call it after the stage declares --masks: a declared --masks default is marked (DeclaredMasksDefault), so
    refuse_masks_with_region_masks tells it from a --masks the user typed.
    """
    declared = builder.parser.get_default(MASKS_DEST)
    if declared is not None:
        builder.parser.set_defaults(**{MASKS_DEST: DeclaredMasksDefault(declared)})
    builder.add_argument(
        REGION_MASKS_OPTION,
        nargs="+",
        action=RegionMasksAction,
        dest=REGION_MASKS_DEST,
        default=None,
        metavar=REGION_MASKS_METAVAR,
        help=_REGION_MASKS_HELP,
    )
    return builder


def format_region_masks(versions: Mapping[str, str]) -> list[str]:
    """NAME=VERSION arguments of a region model name -> Mask version map, which the option parses back."""
    result = [f"{name}={version}" for name, version in versions.items()]
    assert len(result) == len(versions), "region_masks_option.format_region_masks post: len(result) == len(versions)"
    return result


def refuse_masks_with_region_masks(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Exit 2 through parser.error when --masks and --region-masks are both given.

    Afterwards args.masks is a plain str or None: the stage's declared default when --masks was not given.
    """
    masks = getattr(args, MASKS_DEST, None)
    is_masks_given = masks is not None and not isinstance(masks, DeclaredMasksDefault)
    if isinstance(masks, DeclaredMasksDefault):
        setattr(args, MASKS_DEST, str(masks))
    if is_masks_given and getattr(args, REGION_MASKS_DEST, None) is not None:
        parser.error(_MASKS_WITH_MAP_ERROR)


def find_unmapped_region_models(  # noqa: V103 - used by the stage Tasks of epic TASK-MODELVER-BDRABCZUK-1115EA860E09
    versions: Mapping[str, str], regions: Iterable[str]
) -> list[str]:
    """Region model names of ``regions`` (input order, left then right) that ``versions`` does not map."""
    return [name for name in compute_region_model_names(regions) if name not in versions]


def split_region_model_name(  # noqa: V103 - used by the stage Tasks of epic TASK-MODELVER-BDRABCZUK-1115EA860E09
    name: str,
) -> tuple[str, str]:
    """(Sulcal region as in DEFAULT_REGIONS, hemisphere) of a region model name; KeyError outside REGION_MODEL_NAMES."""
    return _REGION_MODEL_PARTS[name]
