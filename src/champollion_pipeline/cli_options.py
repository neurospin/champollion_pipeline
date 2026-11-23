#!/usr/bin/env python3
"""
Config argument registry: one optional main.py argument per PipelineConfig and DatasetConfig field.

CONFIG_ARGUMENTS is derived from dataclasses.fields, so a new field gets its argument without an edit here, and a
field of an unknown type stops the import. Every argument defaults to None, which means "not given": a parse changes
only the fields it names.

PATH_ALIASES adds the other forms of two fields: main.py <input> <output> and --output-path. fold_path_arguments
writes the one value of each field back to its registry dest, or stops when its forms resolve to different paths.
"""

import argparse
import dataclasses
import re
import types
import typing
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from champollion_pipeline.config import DatasetConfig, PipelineConfig

EXEMPT_FIELDS = frozenset({"hf_token", "dependencies", "config_path", "root_path"})

# Fields whose flag and dest are not the field name: the names main.py already reads.
_FLAG_OVERRIDES = {"name": "--dataset-name", "profile_dir": "--profile"}
_DEST_OVERRIDES = {"name": "dataset_name", "profile_dir": "profile"}

# Flags a ScriptBuilder parser may already hold in the shape the registry needs.
_FLAGS_KEPT_FROM_PARSER = ("--log", "--profile")

_DECIMAL_INTEGER = re.compile(r"[+-]?[0-9]+")

# Help texts of the options main.py wrote by hand before the registry, kept word for word.
_LOG_HELP = (
    "Write a run log per stage: DIR/<stage>_<YYYYmmdd-HHMMSS>.log; bare --log: each stage's default log path "
    "(its derivatives .log folder) plus this run's own log at ./.log/champollion_pipeline_<YYYYmmdd-HHMMSS>.log. "
    "Overrides the config file's log."
)
_BIDS_HELP = (
    "Input database follows BIDS layout "
    "(subjects/sub-*/ses-*/...). "
    "Activates BIDS-aware scan enumeration in eligibility checks "
    "and passes --bids to cortical_tiles."
)
_ALLOW_PREFIX_IDS_HELP = (
    "TEMPORARY: let cortical_tiles run when a subject ID is a strict prefix "
    "of another (e.g. sub-1 and sub-10); its transforms may then be wrong."
)
_MODE_HELP = (
    "Execution mode: 'sequential' (default, stage-centric batch) or "
    "'streaming' (scan-centric parallel, one worker per scan). "
    "Streaming mode requires dataset.embeddings_only: true in the --config YAML."
)
_WORKER_TIMEOUT_HELP = (
    "Timeout in seconds of the whole producer-worker run of one scan; a longer run ends FAILED (default: 7200)."
)
# Help of the fields no stage and no champollion-train run reads; kept so existing YAML files still load.
_UNREAD_FIELD_HELP = (
    "Config file field, read by no stage and not by champollion-train; kept so existing YAML files load"
)
_UNREAD_FIELDS = (
    "datasets",
    "labels",
    "classifier_name",
    "subsets",
    "epochs",
    "split",
    "cv",
    "splits_basedir",
    "idx_region_evaluation",
    "short_name",
    "champollion_v1_path",
)


class ArgumentConflictError(ValueError):
    """Two forms of one config field resolve to different paths; a usage error, str(error) is the ERROR message."""


class ArgumentKind(Enum):
    """Value kind of a config field, which fixes the add_argument options of its argument."""

    TEXT = "text"
    INTEGER = "integer"
    FLAG = "flag"
    TEXT_LIST = "text_list"
    INTEGER_LIST = "integer_list"
    STAGE_SET = "stage_set"


@dataclass(frozen=True)
class ConfigArgument:
    """One row of the registry: a config field and its argument."""

    field_name: str
    is_dataset_field: bool
    flag: str
    dest: str
    kind: ArgumentKind

    def owner_of(self, config: PipelineConfig) -> PipelineConfig | DatasetConfig:
        """The object of config that holds this field."""
        return config.dataset if self.is_dataset_field else config


_HINT_KINDS = {
    str: ArgumentKind.TEXT,
    int: ArgumentKind.INTEGER,
    bool: ArgumentKind.FLAG,
    list[str]: ArgumentKind.TEXT_LIST,
    list[int | None]: ArgumentKind.INTEGER_LIST,
    dict[str, bool]: ArgumentKind.STAGE_SET,
}


def _parse_decimal_integer(text: str) -> int:
    """The int of a decimal integer; int() alone would also read 1_000 and surrounding spaces."""
    if _DECIMAL_INTEGER.fullmatch(text) is None:
        raise argparse.ArgumentTypeError(f"not a decimal integer: {text!r}")
    return int(text)


_KIND_OPTIONS = {
    ArgumentKind.TEXT: {},
    ArgumentKind.INTEGER: {"type": _parse_decimal_integer},
    ArgumentKind.FLAG: {"action": argparse.BooleanOptionalAction},
    ArgumentKind.TEXT_LIST: {"nargs": "+"},
    ArgumentKind.INTEGER_LIST: {"nargs": "+", "type": _parse_decimal_integer},
    ArgumentKind.STAGE_SET: {"nargs": "+", "choices": list(PipelineConfig().stages)},
}

# Options of single fields; they win over those of their kind. The levels are attributes of the logging module,
# which main.py reads: the champollion_utils level names are not, so they stay out.
_FIELD_OPTIONS = {
    "stages": {"help": "Specific stages to run (overrides config)"},
    "name": {"help": "Dataset name (overrides config)"},
    "models_path": {"help": "Path to models directory (overrides config)"},
    "verbose": {"help": "Enable verbose output"},
    "bids": {"help": _BIDS_HELP},
    "allow_prefix_ids": {"help": _ALLOW_PREFIX_IDS_HELP},
    "mode": {"choices": ("sequential", "streaming"), "help": _MODE_HELP},
    "n_workers": {
        "metavar": "N",
        "help": "Number of parallel scan workers (streaming mode only). Defaults to os.cpu_count().",
    },
    "worker_timeout": {"metavar": "SECONDS", "help": _WORKER_TIMEOUT_HELP},
    "log_level": {"choices": ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")},
    "log": {"nargs": "?", "const": "", "metavar": "DIR", "help": _LOG_HELP},
    **{field_name: {"help": _UNREAD_FIELD_HELP} for field_name in _UNREAD_FIELDS},
}


def _kind_of(field_name: str, hint: object) -> ArgumentKind:
    """Argument kind of a type hint, an optional type read as its inner type; TypeError on any other hint."""
    if isinstance(hint, types.UnionType):
        inner = [arg for arg in typing.get_args(hint) if arg is not types.NoneType]
        hint = inner[0] if len(inner) == 1 else hint
    if hint not in _HINT_KINDS:
        raise TypeError(f"config field {field_name} has a type with no argument kind: {hint}")
    return _HINT_KINDS[hint]


def _config_arguments() -> tuple[ConfigArgument, ...]:
    """One row per field of PipelineConfig then DatasetConfig, without the exempt and dataclass-typed fields."""
    rows = []
    for config_class in (PipelineConfig, DatasetConfig):
        hints = typing.get_type_hints(config_class)
        for config_field in dataclasses.fields(config_class):
            name, hint = config_field.name, hints[config_field.name]
            if name in EXEMPT_FIELDS or dataclasses.is_dataclass(hint):
                continue
            rows.append(
                ConfigArgument(
                    field_name=name,
                    is_dataset_field=config_class is DatasetConfig,
                    flag=_FLAG_OVERRIDES.get(name, "--" + name.replace("_", "-")),
                    dest=_DEST_OVERRIDES.get(name, name),
                    kind=_kind_of(name, hint),
                )
            )
    return tuple(rows)


CONFIG_ARGUMENTS = _config_arguments()


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add one optional argument per row of CONFIG_ARGUMENTS to parser.

    --log and --profile are kept when parser holds them; any other duplicate raises argparse.ArgumentError, so a
    hand-written flag left beside the registry shows up.
    """
    for row in CONFIG_ARGUMENTS:
        if row.flag in _FLAGS_KEPT_FROM_PARSER and row.flag in parser._option_string_actions:
            continue
        section = "dataset" if row.is_dataset_field else "pipeline"
        options = {
            "help": f"Config field {section}.{row.field_name}",
            **_KIND_OPTIONS[row.kind],
            **_FIELD_OPTIONS.get(row.field_name, {}),
        }
        parser.add_argument(row.flag, dest=row.dest, default=None, **options)


def apply_config_arguments(config: PipelineConfig, args: argparse.Namespace) -> None:
    """Set on config, in place, the value of each config argument given in args.

    --stages replaces the whole selection: the stages it does not name turn off. --verbose without --log-level sets
    log_level to DEBUG.
    """
    for row in CONFIG_ARGUMENTS:
        value = getattr(args, row.dest, None)
        if value is None:
            continue
        owner = row.owner_of(config)
        if row.kind is ArgumentKind.STAGE_SET:
            value = {**dict.fromkeys(getattr(owner, row.field_name), False), **dict.fromkeys(value, True)}
        setattr(owner, row.field_name, value)
    if getattr(args, "verbose", None) is True and getattr(args, "log_level", None) is None:
        config.log_level = "DEBUG"


@dataclass(frozen=True)
class PathAlias:
    """A config field that more than one main.py argument sets: each form as (name in messages, args dest).

    field_name is also the field's registry dest (it is not in _DEST_OVERRIDES).
    """

    field_name: str
    forms: tuple[tuple[str, str], ...]


# The forms of each field; the value kept is the first given form's.
PATH_ALIASES = (
    PathAlias("input_path", (("INPUT", "input_positional"), ("--input-path", "input_path"))),
    PathAlias(
        "datasets_root",
        (("OUTPUT", "output_positional"), ("--output-path", "output_path"), ("--datasets-root", "datasets_root")),
    ),
)


def add_path_arguments(parser: argparse.ArgumentParser) -> None:  # noqa: V103 - used by main.py
    """Add the INPUT and OUTPUT positionals and --output-path, all optional, all default None, all in --help."""
    parser.add_argument(
        "input_positional", nargs="?", default=None, metavar="INPUT", help="Input to process (same as --input-path)"
    )
    parser.add_argument(
        "output_positional",
        nargs="?",
        default=None,
        metavar="OUTPUT",
        help="Dataset root holding derivatives/ (same as --output-path)",
    )
    parser.add_argument(
        "--output-path", dest="output_path", default=None, metavar="DIR", help="Config field dataset.datasets_root"
    )


def _same_path_value(first: str, other: str) -> bool:
    """True when two forms name the same path: equal pathlib Path.resolve() (symlinks followed, from the cwd)."""
    return Path(first).resolve() == Path(other).resolve()


def fold_path_arguments(args: argparse.Namespace) -> None:  # noqa: V103 - used by main.py
    """Set, in args, each PATH_ALIASES field's registry dest to the value its given forms agree on.

    A field none of whose forms is given is left alone; the value kept is the first given form's, as given. Raises
    ArgumentConflictError naming every given form and its value when two of them resolve to different paths.
    """
    for alias in PATH_ALIASES:
        given = [(name, getattr(args, dest, None)) for name, dest in alias.forms]
        given = [(name, value) for name, value in given if value is not None]
        if not given:
            continue
        value = given[0][1]
        if not all(_same_path_value(value, other) for _, other in given[1:]):
            forms = ", ".join(f"{name} {other!r}" for name, other in given)
            raise ArgumentConflictError(f"dataset.{alias.field_name} given as different paths: {forms}; give one path")
        setattr(args, alias.field_name, value)
