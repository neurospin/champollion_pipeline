"""Parse-time checks of the embeddings stage's model version options, and the version a model set marker names."""

import argparse
import os
import warnings
from os.path import join

from champollion_pipeline.derivatives_layout import DEFAULT_MASKS_VERSION
from champollion_pipeline.models.cache import MODEL_SET_FILE_NAME, read_model_set


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


def resolve_masks_default(args: argparse.Namespace) -> None:
    """Fill --masks when it was not given: the model version if set, else DEFAULT_MASKS_VERSION."""
    if args.masks is None:
        model_version = args.model_version
        args.masks = model_version if model_version is not None else DEFAULT_MASKS_VERSION


def check_version_mismatch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Exit 2 when --masks and --model-version differ, unless --allow-version-mismatch (then warn)."""
    model_version, masks = args.model_version, args.masks
    if model_version is None or masks == model_version:
        return
    if not args.allow_version_mismatch:
        parser.error(
            f"--masks {masks!r} and --model-version {model_version!r} differ: --masks picks the input crops, "
            "--model-version the model set; pass --allow-version-mismatch to run with both"
        )
    warnings.warn(
        f"--masks {masks!r} and --model-version {model_version!r} differ; running anyway "
        f"(--allow-version-mismatch): crops {masks!r}, models {model_version!r}",
        UserWarning,
        stacklevel=2,
    )


def check_local_model_set(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Exit 2 when a local model set's model_set.json is invalid, or names other masks than --masks unless
    --allow-version-mismatch (then warn). A local directory without a marker: warn that it cannot be checked.
    Not a local directory: nothing."""
    models_dir = args.models_path
    if not os.path.isdir(models_dir):
        return
    if not os.path.isfile(join(models_dir, MODEL_SET_FILE_NAME)):
        masks = args.masks
        warnings.warn(
            f"{join(models_dir, MODEL_SET_FILE_NAME)} not found: the local model set cannot be checked against "
            f"--masks {masks!r}; running anyway with --masks {masks!r}",
            UserWarning,
            stacklevel=2,
        )
        return
    try:
        model_set = read_model_set(models_dir)
    except ValueError as err:
        parser.error(str(err))
    path, marker_masks, masks = join(models_dir, MODEL_SET_FILE_NAME), model_set.masks, args.masks
    if marker_masks == masks:
        return
    if not args.allow_version_mismatch:
        parser.error(
            f"{path} masks {marker_masks!r} and --masks {masks!r} differ: the local model set was trained on "
            f"{marker_masks!r} crops; pass --masks {marker_masks!r}, or --allow-version-mismatch to run with both"
        )
    warnings.warn(
        f"{path} masks {marker_masks!r} and --masks {masks!r} differ; running anyway (--allow-version-mismatch): "
        f"crops {masks!r}, models trained on {marker_masks!r}",
        UserWarning,
        stacklevel=2,
    )


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
