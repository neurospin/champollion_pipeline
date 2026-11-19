"""Parse-time checks of the embeddings stage's model version options, and the version a model set marker names."""

import argparse
import logging
import os
import warnings
from os.path import join

from champollion_pipeline.derivatives_layout import DEFAULT_MASKS_VERSION
from champollion_pipeline.models.cache import MODEL_SET_FILE_NAME, read_model_set

MASK_VERSIONS = ("canonical_25", "canonical_corrected_26_1")

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
