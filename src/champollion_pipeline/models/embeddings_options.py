"""Option declarations of the embeddings stage (stage 4) parser: GenerateEmbeddings declares them through
add_embeddings_arguments. No torch import, so the parser stays cheap to build."""

import argparse

from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.derivatives_layout import DEFAULT_MASKS_VERSION
from champollion_pipeline.models.checkpoint_defaults import DEFAULT_CHECKPOINT_EPOCH
from champollion_pipeline.models.epoch_spec import parse_epoch_specs
from champollion_pipeline.utils.lib import CORTICAL_TILES_VERSION
from champollion_pipeline.utils.region_masks import add_region_masks_option


class CheckpointEpochAction(argparse.Action):
    """Repeatable --checkpoint-epoch [<id>=]N: sets checkpoint_epoch (the default) and checkpoint_epochs_by_id.

    A value parse_epoch_specs refuses ends the parse with exit 2.
    """

    def __call__(self, parser, namespace, values, _option_string=None):
        held = [*getattr(namespace, "checkpoint_epoch_values", ()), values]
        try:
            default, by_id = parse_epoch_specs(held, DEFAULT_CHECKPOINT_EPOCH)
        except ValueError as err:
            raise argparse.ArgumentError(self, str(err)) from err
        namespace.checkpoint_epoch_values = held
        setattr(namespace, self.dest, default)
        namespace.checkpoint_epochs_by_id = by_id


def add_embeddings_arguments(builder: ScriptBuilder) -> ScriptBuilder:
    """Declare every option of the embeddings stage on builder, in the reference order; return builder."""
    _add_run_arguments(builder)
    _add_checkpoint_arguments(builder)
    _add_crops_arguments(builder)
    _add_version_arguments(builder)
    _add_selection_arguments(builder)
    return add_region_masks_option(builder)


def _add_run_arguments(builder: ScriptBuilder) -> ScriptBuilder:
    """Declare models_path, datasets_root, --overwrite, --cpu, --profiling, --run-cka and --no-cache on builder."""
    return (
        builder.add_argument("models_path", type=str, help="Path to the directory containing per-region model folders.")
        .add_argument(
            "datasets_root",
            type=str,
            help="Absolute path to the dataset root directory (e.g. /my/path/to/DATASET/).",
        )
        .add_flag("--overwrite", "Recompute embeddings that already exist on disk.")
        .add_flag("--cpu", "Force CPU usage (disable CUDA).")
        .add_flag("--profiling", "Enable Python profiling (cProfile).")
        .add_flag("--run-cka", "Run CKA coherence test after embeddings.")
        .add_flag("--no-cache", "Force re-extraction of archive (ignore cache).")
    )


def _add_checkpoint_arguments(builder: ScriptBuilder) -> ScriptBuilder:
    """Declare --checkpoint-epoch and --per-model-folders on builder."""
    return builder.add_argument(
        "--checkpoint-epoch",
        action=CheckpointEpochAction,
        default=DEFAULT_CHECKPOINT_EPOCH,
        metavar="[ID=]N",
        help="Training epoch N (0-based) of the Lightning checkpoint epoch=N-step=*.ckpt to evaluate each "
        f"model with. Default: {DEFAULT_CHECKPOINT_EPOCH}. Repeatable: a bare N is the default epoch (at most "
        "one); ID=N sets the epoch of the models with model id ID (the folder path between the region folder "
        "and the model folder).",
    ).add_flag(
        "--per-model-folders",
        "Write the embeddings of every model with a model id to <output>/<region>/<model id>/, even a "
        "single model of its region (by default only a region holding two or more models does).",
    )


def _add_crops_arguments(builder: ScriptBuilder) -> ScriptBuilder:
    """Declare --cortical_version and --legacy on builder."""
    return builder.add_optional_argument(
        "--cortical_version",
        f"Derivatives folder name (e.g. 'cortical_tiles-2027'). "
        f"Defaults to cortical_tiles-{CORTICAL_TILES_VERSION}. Crops are read from "
        f"<datasets_root>/derivatives/<cortical_version>/crops/<masks>/2mm.",
        default=f"cortical_tiles-{CORTICAL_TILES_VERSION}",
    ).add_flag(
        "--legacy",
        "Use deep_folding-2025 as the derivatives folder. Shorthand for --cortical_version deep_folding-2025.",
    )


def _add_version_arguments(builder: ScriptBuilder) -> ScriptBuilder:
    """Declare --model-version, --masks-version, --masks and --allow-version-mismatch on builder."""
    return (
        builder.add_optional_argument(
            "--model-version",
            "Model-set subfolder to download from the Hugging Face models repository (e.g. 'canonical_25'). "
            "Ignored when models_path is a local directory.",
            default=None,
        )
        .add_optional_argument(
            "--masks-version",
            "Deprecated alias of --model-version; same value and behaviour.",
            default=None,
        )
        .add_optional_argument(
            "--masks",
            "Cortical tiles mask version used as crops subdirectory (e.g. 'canonical_25'): crops are read from "
            "<datasets_root>/derivatives/<cortical_version>/crops/<masks>/2mm. "
            "Default: the --model-version value when given; else, when models_path is a local folder, "
            "the name of that folder if it is a known Mask version (otherwise exit 2 with an ERROR, pass --masks); "
            f"{DEFAULT_MASKS_VERSION} only when models_path is not a local directory.",
            default=None,
        )
        .add_flag(
            "--allow-version-mismatch",
            "Run even when --masks and --model-version differ (crops of one mask version evaluated with the "
            "model set of another); a warning is emitted.",
        )
    )


def _add_selection_arguments(builder: ScriptBuilder) -> ScriptBuilder:
    """Declare --regions, --output, --subjects and --models-cache on builder."""
    return (
        builder.add_argument(
            "--regions",
            type=str,
            nargs="+",
            default=None,
            help=(
                "Restrict embedding generation to specific region names "
                "(e.g. SCsylv_left FIPFIPPoCinf_right). "
                "By default all regions found in models_path are processed."
            ),
        )
        .add_optional_argument(
            "--output",
            "Output base directory override. "
            "Defaults to {datasets_root}/derivatives/champollion_V1/{masks}/region_embeddings/.",
            default=None,
        )
        .add_optional_argument(
            "--subjects",
            "Deprecated and ignored. Each region's subjects file is read from "
            "{crops}/<crop>/mask/{L|R}skeleton_subject.csv next to its skeleton.",
            default=None,
        )
        .add_optional_argument(
            "--models-cache",
            "Models cache directory; relative paths resolve against the cwd. "
            "Defaults to <datasets_root>/derivatives/champollion_V1/models_cache.",
            default=None,
        )
    )
