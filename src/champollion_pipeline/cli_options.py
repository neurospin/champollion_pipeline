#!/usr/bin/env python3
"""
Config argument registry: one optional main.py argument per PipelineConfig and DatasetConfig field.

CONFIG_ARGUMENTS is derived from dataclasses.fields, so a new field gets its argument without an edit here, and a
field of an unknown type stops the import. Every argument defaults to None, which means "not given": a parse changes
only the fields it names.
"""

import argparse
import dataclasses
import re
import types
import typing
from dataclasses import dataclass
from enum import Enum

from champollion_pipeline.config import DatasetConfig, PipelineConfig

EXEMPT_FIELDS = frozenset({"hf_token", "dependencies", "config_path", "root_path"})

# Fields whose flag and dest are not the field name: the names main.py already reads.
_FLAG_OVERRIDES = {"name": "--dataset-name", "profile_dir": "--profile"}
_DEST_OVERRIDES = {"name": "dataset_name", "profile_dir": "profile"}

# Flags a ScriptBuilder parser may already hold in the shape the registry needs.
_FLAGS_KEPT_FROM_PARSER = ("--log", "--profile")

_DECIMAL_INTEGER = re.compile(r"[+-]?[0-9]+")


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

# Options of single fields, added to those of their kind. The levels are attributes of the logging module, which
# main.py reads: the champollion_utils level names are not, so they stay out.
_FIELD_OPTIONS = {
    "mode": {"choices": ("sequential", "streaming")},
    "log_level": {"choices": ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")},
    "log": {"nargs": "?", "const": ""},
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
        parser.add_argument(
            row.flag,
            dest=row.dest,
            default=None,
            help=f"Config field {section}.{row.field_name}",
            **_KIND_OPTIONS[row.kind],
            **_FIELD_OPTIONS.get(row.field_name, {}),
        )


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
