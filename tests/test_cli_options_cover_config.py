#!/usr/bin/env python3
"""
Config argument registry: each PipelineConfig and DatasetConfig field has an optional main.py argument.

The field list comes from dataclasses.fields, never from a hand-written copy: a new field without an
argument fails test_fields_without_argument_are_exactly_the_exempt_set and
test_config_field_has_an_optional_argument. A field of a value type this file does not know also fails
the second one, so that its argument gets a value test.

The parser under test is a ScriptBuilder parser built like main.py's: no --log from ScriptBuilder, because
main.py's --log takes a directory, and ScriptBuilder's --profile present.

Requirements: REQ-INPUTKIND-BDRABCZUK-0EE961FB28CE (each field has an argument),
REQ-INPUTKIND-BDRABCZUK-E9D883A4202B (exempt set),
REQ-INPUTKIND-BDRABCZUK-18BFEA3C804A (flat names),
REQ-INPUTKIND-BDRABCZUK-6255CE4BD52D (--dataset-name),
REQ-INPUTKIND-BDRABCZUK-A55CEBFEE1EA (text values),
REQ-INPUTKIND-BDRABCZUK-0EBB1F5FF3F6 (integer values),
REQ-INPUTKIND-BDRABCZUK-20BCDF5C1A48 (a non-integer value is refused),
REQ-INPUTKIND-BDRABCZUK-242ABF045697 (list values),
REQ-INPUTKIND-BDRABCZUK-FA4C04A8A273 (--epochs),
REQ-INPUTKIND-BDRABCZUK-9F0F6A3F67AB (default-false booleans),
REQ-INPUTKIND-BDRABCZUK-C77E22AF7FEA (absent arguments keep the config),
REQ-INPUTKIND-BDRABCZUK-54A6ABDCCB90 (bare --log),
REQ-INPUTKIND-BDRABCZUK-1E08AED65CA9 (--mode values),
REQ-INPUTKIND-BDRABCZUK-2EC7B50CD916 (--profile),
REQ-INPUTKIND-BDRABCZUK-DF1F5612AC4C (light import),
REQ-INPUTKIND-BDRABCZUK-8016CC4EE590 (ScriptBuilder parser),
REQ-INPUTKIND-BDRABCZUK-5C88088EDD37 (--no-<field>),
REQ-INPUTKIND-BDRABCZUK-47B44FCD32CD (default-true booleans),
REQ-INPUTKIND-BDRABCZUK-E11298657255 (--stages enables the named stages),
REQ-INPUTKIND-BDRABCZUK-9D565179DD11 (--stages disables the others),
REQ-INPUTKIND-BDRABCZUK-067041EC5A22 (--stages refuses an unknown name),
REQ-INPUTKIND-BDRABCZUK-57C752062264 (--verbose sets DEBUG),
REQ-INPUTKIND-BDRABCZUK-B7EC4F23D24F (explicit --log-level wins),
REQ-INPUTKIND-BDRABCZUK-F3EDB8D86DFD (--no-verbose keeps log_level),
REQ-INPUTKIND-BDRABCZUK-BB65B40C7E4C (no --profile-dir),
REQ-INPUTKIND-BDRABCZUK-9CE41FC67F18 (light config import),
REQ-INPUTKIND-BDRABCZUK-6B5E37BD469E (a bare list argument is refused),
REQ-INPUTKIND-BDRABCZUK-AD663D7A2843 (--epochs refuses a non-integer value),
REQ-INPUTKIND-BDRABCZUK-8E73D5D8CB1C (--log-level refuses an unknown level),
REQ-INPUTKIND-BDRABCZUK-9DC1BAD12F93 (--log-level takes the known levels).
"""

import argparse
import copy
import dataclasses
import json
import os
import subprocess
import sys
import types
import typing
from pathlib import Path

import pytest
from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.cli_options import add_config_arguments, apply_config_arguments
from champollion_pipeline.config import DatasetConfig, PipelineConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_MODULE = "champollion_pipeline.cli_options"
CONFIG_MODULE = "champollion_pipeline.config"
HEAVY_MODULES = ("torch", "soma.aims", "numpy", "pandas")

EXEMPT_FIELDS = {"hf_token", "dependencies", "config_path", "root_path"}
CONFIG_CLASSES = (PipelineConfig, DatasetConfig)

# Fields whose argument is not the flat name, with that argument.
NON_FLAT_FLAGS = {"name": ("--dataset-name",), "profile_dir": ("--profile",)}

# A value the argument accepts where a free text would not do.
TEXT_SAMPLES = {"mode": "streaming", "log_level": "DEBUG"}

PROBE_TEXT = "probe-value"
PROBE_INTEGER = 4242
PROBE_DIRECTORY = "/probe/directory"
EXPLICIT_LOG_LEVEL = "WARNING"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
NON_INTEGERS = ("not-an-integer", "1_000", "4.2")
KNOWN_VALUE_KINDS = (str, int, bool, list, dict)


def _value_kind(hint: object) -> type:
    """Value type of a field: str, int, bool, list or dict, with an optional type read as its inner type."""
    if isinstance(hint, types.UnionType):
        hint = next(arg for arg in typing.get_args(hint) if arg is not types.NoneType)
    return typing.get_origin(hint) or hint


def _config_fields(classes: tuple[type, ...] = CONFIG_CLASSES) -> dict[str, tuple[type, object]]:
    """Field name -> (owning dataclass, type hint), from dataclasses.fields, without the nested dataset field."""
    table = {}
    for config_class in classes:
        hints = typing.get_type_hints(config_class)
        for config_field in dataclasses.fields(config_class):
            hint = hints[config_field.name]
            if not dataclasses.is_dataclass(hint):
                table[config_field.name] = (config_class, hint)
    return table


def _argument_fields() -> list[str]:
    """Names of the config fields that have an argument."""
    return sorted(set(_config_fields()) - EXEMPT_FIELDS)


def _fields_of_kind(kind: type) -> list[str]:
    """Names of the fields with an argument whose value type is kind."""
    return [name for name in _argument_fields() if _value_kind(_config_fields()[name][1]) is kind]


def _default_of(name: str) -> object:
    """Default value of a config field."""
    owner = _config_fields()[name][0]
    return next(config_field.default for config_field in dataclasses.fields(owner) if config_field.name == name)


def _flat_flag(name: str) -> str:
    """Flat argument name of a field."""
    return "--" + name.replace("_", "-")


def _flags_of(name: str) -> tuple[str, ...]:
    """Argument strings that may carry a field."""
    return NON_FLAT_FLAGS.get(name, (_flat_flag(name),))


def _text_fields() -> list[str]:
    """Text fields with a value test here; --profile has its own test."""
    return [name for name in _fields_of_kind(str) if name != "profile_dir"]


def _text_list_fields() -> list[str]:
    """List fields of str elements."""
    return [name for name in _fields_of_kind(list) if typing.get_args(_config_fields()[name][1]) == (str,)]


def _default_false_bool_fields() -> list[str]:
    """Boolean fields whose default is False."""
    return [name for name in _fields_of_kind(bool) if _default_of(name) is False]


def _default_true_bool_fields() -> list[str]:
    """Boolean fields whose default is True."""
    return [name for name in _fields_of_kind(bool) if _default_of(name) is True]


class _PlainScript(ScriptBuilder):
    """A plain ScriptBuilder: it holds its own --log and --profile."""

    def run(self) -> int:
        """Never run: only the parser is used."""
        return 0


class _HostScript(_PlainScript):
    """A ScriptBuilder shaped like main.py's: its own --log comes from the registry, --profile stays."""

    def _add_log_option(self) -> None:  # noqa: V105 - called by ScriptBuilder.__init__
        """Add nothing, as main.py does."""


def _parser() -> argparse.ArgumentParser:
    """A host parser with the registry arguments."""
    parser = _HostScript("probe", "probe")
    add_config_arguments(parser.parser)
    return parser.parser


def _options(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    """Argument string -> action, of the optional arguments of a parser."""
    return {flag: action for action in parser._actions for flag in action.option_strings}


def _fields_without_argument(parser: argparse.ArgumentParser, classes: tuple[type, ...]) -> set[str]:
    """Names of the fields of classes with no argument on parser."""
    options = _options(parser)
    return {name for name in _config_fields(classes) if not set(_flags_of(name)) & set(options)}


def _applied(argv: list[str], config: PipelineConfig | None = None) -> PipelineConfig:
    """The config after parsing argv and applying it; a default config unless one is given."""
    config = config if config is not None else PipelineConfig()
    parsed = _parser().parse_args(argv)
    apply_config_arguments(config, parsed)
    return config


def _field_value(config: PipelineConfig, name: str) -> object:
    """Value of a field in its owning object."""
    owner = _config_fields()[name][0]
    return getattr(config.dataset if owner is DatasetConfig else config, name)


def _config_with(name: str, value: object) -> PipelineConfig:
    """A default config with one field set to value."""
    config = PipelineConfig()
    setattr(config.dataset if _config_fields()[name][0] is DatasetConfig else config, name, value)
    return config


def _heavy_modules_after_import(module: str) -> list[str]:
    """Heavy modules a new interpreter holds after importing module; the import must succeed."""
    code = f"import json, sys\nimport {module}\nprint(json.dumps(sorted(set({HEAVY_MODULES!r}) & set(sys.modules))))"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(path for path in (str(REPO_ROOT / "src"), env.get("PYTHONPATH")) if path)
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.splitlines()[-1])


def _non_default_config() -> PipelineConfig:
    """A config whose fields with an argument all hold a value that no default holds."""
    config = PipelineConfig()
    for name in _argument_fields():
        kind = _value_kind(_config_fields()[name][1])
        owner = config.dataset if _config_fields()[name][0] is DatasetConfig else config
        if kind is str:
            setattr(owner, name, PROBE_TEXT)
        elif kind is int:
            setattr(owner, name, PROBE_INTEGER)
        elif kind is bool:
            setattr(owner, name, not _default_of(name))
        elif kind is list:
            setattr(owner, name, [PROBE_TEXT])
    config.stages = dict.fromkeys(config.stages, True)
    return config


@pytest.mark.parametrize("name", _argument_fields())
def test_config_field_has_an_optional_argument(name: str) -> None:
    """Each config field outside the exempt set has an optional argument of a known value kind."""
    options = _options(_parser())
    carriers = [options[flag] for flag in _flags_of(name) if flag in options]
    assert carriers, f"no argument for config field {name}: expected one of {_flags_of(name)}"
    assert not carriers[0].required, f"the argument of config field {name} is required"
    assert _value_kind(_config_fields()[name][1]) in KNOWN_VALUE_KINDS, f"add a value test for config field {name}"


def test_field_added_without_argument_is_reported() -> None:
    """A dataclass field the registry does not know shows up as a field without an argument."""
    extended = dataclasses.make_dataclass(
        "ExtendedDatasetConfig", [("probe_new_field", str, "")], bases=(DatasetConfig,)
    )
    found = _fields_without_argument(_parser(), (PipelineConfig, extended))
    assert found == EXEMPT_FIELDS | {"probe_new_field"}


def test_fields_without_argument_are_exactly_the_exempt_set() -> None:
    """The config fields without an argument are the four exempt fields, no more and no fewer."""
    assert _fields_without_argument(_parser(), CONFIG_CLASSES) == EXEMPT_FIELDS


@pytest.mark.parametrize("name", sorted(set(_argument_fields()) - set(NON_FLAT_FLAGS)))
def test_argument_name_is_the_flat_field_name(name: str) -> None:
    """A field's argument is --<field> with hyphens."""
    assert _flat_flag(name) in _options(_parser())


def test_dataset_name_field_uses_dataset_name_argument() -> None:
    """DatasetConfig.name is --dataset-name; a flat --name does not exist."""
    options = _options(_parser())
    assert "--dataset-name" in options
    assert "--name" not in options


@pytest.mark.parametrize("name", _text_fields())
def test_text_field_takes_the_given_value(name: str) -> None:
    """A text argument sets its field, on the object that owns the field."""
    value = TEXT_SAMPLES.get(name, PROBE_TEXT)
    assert _field_value(_applied([_flags_of(name)[0], value]), name) == value


@pytest.mark.parametrize("name", _fields_of_kind(int))
def test_integer_field_takes_the_given_integer(name: str) -> None:
    """An integer argument sets its field to an int."""
    value = _field_value(_applied([_flat_flag(name), str(PROBE_INTEGER)]), name)
    assert type(value) is int
    assert value == PROBE_INTEGER


@pytest.mark.parametrize("value", NON_INTEGERS)
@pytest.mark.parametrize("name", _fields_of_kind(int))
def test_integer_field_refuses_a_non_integer_value(name: str, value: str) -> None:
    """A value that is not a decimal integer is a usage error, 1_000 included, which int() would read."""
    parser = _parser()
    with pytest.raises(SystemExit) as stopped:
        parser.parse_args([_flat_flag(name), value])
    assert stopped.value.code == 2


@pytest.mark.parametrize("name", _text_list_fields())
def test_text_list_field_takes_the_given_values(name: str) -> None:
    """A list argument takes several values and keeps their order."""
    assert _field_value(_applied([_flat_flag(name), "second", "first"]), name) == ["second", "first"]


def test_epochs_takes_the_given_integers() -> None:
    """--epochs takes several integers and keeps their order."""
    values = _applied(["--epochs", "20", "10"]).dataset.epochs
    assert values == [20, 10]
    assert all(type(value) is int for value in values)


@pytest.mark.parametrize("value", NON_INTEGERS)
def test_epochs_refuses_a_non_integer_value(value: str) -> None:
    """One value that is not a decimal integer refuses the whole --epochs argument."""
    parser = _parser()
    with pytest.raises(SystemExit) as stopped:
        parser.parse_args(["--epochs", "10", value])
    assert stopped.value.code == 2


@pytest.mark.parametrize("name", _fields_of_kind(list))
def test_list_field_refuses_a_bare_argument(name: str) -> None:
    """A list argument with no value is a usage error: an empty --regions would mean all regions."""
    parser = _parser()
    with pytest.raises(SystemExit) as stopped:
        parser.parse_args([_flat_flag(name)])
    assert stopped.value.code == 2


@pytest.mark.parametrize("value", ["probe", "debug"])
def test_log_level_refuses_an_unknown_level(value: str) -> None:
    """--log-level takes the named levels only, in capitals: main.py reads the level from the logging module."""
    parser = _parser()
    with pytest.raises(SystemExit) as stopped:
        parser.parse_args(["--log-level", value])
    assert stopped.value.code == 2


@pytest.mark.parametrize("level", LOG_LEVELS)
def test_log_level_takes_a_known_level(level: str) -> None:
    """--log-level sets each of the named levels."""
    assert _applied(["--log-level", level]).log_level == level


@pytest.mark.parametrize("name", _default_false_bool_fields())
def test_default_false_bool_field_is_set_by_its_flag(name: str) -> None:
    """The bare argument of a default-false boolean sets it to True."""
    assert _field_value(_applied([_flat_flag(name)]), name) is True


@pytest.mark.parametrize("name", _default_true_bool_fields())
def test_default_true_bool_field_is_set_by_its_flag(name: str) -> None:
    """The bare argument of a default-true boolean sets it to True over a False from a YAML file."""
    assert _field_value(_applied([_flat_flag(name)], _config_with(name, False)), name) is True


@pytest.mark.parametrize("name", _fields_of_kind(bool))
def test_no_form_turns_the_bool_field_off(name: str) -> None:
    """--no-<field> sets a boolean to False over a True from a YAML file."""
    no_flag = "--no-" + name.replace("_", "-")
    assert _field_value(_applied([no_flag], _config_with(name, True)), name) is False


def test_stages_enables_the_named_stages() -> None:
    """--stages sets the stages it names to True."""
    named = list(PipelineConfig().stages)[:2]
    all_off = _config_with("stages", dict.fromkeys(PipelineConfig().stages, False))
    stages = _applied(["--stages", *named], all_off).stages
    assert [stages[stage] for stage in named] == [True, True]


def test_stages_disables_the_other_stages() -> None:
    """--stages sets the stages it does not name to False."""
    named, *others = list(PipelineConfig().stages)
    all_on = _config_with("stages", dict.fromkeys(PipelineConfig().stages, True))
    stages = _applied(["--stages", named], all_on).stages
    assert [stages[stage] for stage in others] == [False] * len(others)


def test_stages_refuses_an_unknown_name() -> None:
    """--stages takes the keys of the default stages field only."""
    parser = _parser()
    with pytest.raises(SystemExit) as stopped:
        parser.parse_args(["--stages", "unknown_stage"])
    assert stopped.value.code == 2


def test_verbose_sets_debug_log_level() -> None:
    """--verbose alone also sets log_level to DEBUG."""
    assert _applied(["--verbose"]).log_level == "DEBUG"


@pytest.mark.parametrize(
    "argv",
    [["--verbose", "--log-level", EXPLICIT_LOG_LEVEL], ["--log-level", EXPLICIT_LOG_LEVEL, "--verbose"]],
    ids=["verbose_first", "log_level_first"],
)
def test_explicit_log_level_wins_over_verbose(argv: list[str]) -> None:
    """--log-level beats the DEBUG of --verbose, in either order."""
    assert _applied(argv).log_level == EXPLICIT_LOG_LEVEL


def test_no_verbose_keeps_the_log_level() -> None:
    """--no-verbose changes verbose only: the log_level of a YAML file stays."""
    assert _applied(["--no-verbose"], _config_with("log_level", EXPLICIT_LOG_LEVEL)).log_level == EXPLICIT_LOG_LEVEL


def test_profile_dir_has_no_flat_argument() -> None:
    """ScriptBuilder's --profile is the only argument of profile_dir."""
    assert "--profile-dir" not in _options(_parser())


def test_no_argument_leaves_the_config_unchanged() -> None:
    """Without arguments a config loaded from a YAML file keeps each of its values."""
    config = _non_default_config()
    expected = copy.deepcopy(config)
    assert _applied([], config) == expected


def test_bare_log_sets_the_empty_string() -> None:
    """--log without a directory is the "" of PipelineConfig.log: each stage's default run log."""
    assert _applied(["--log"]).log == ""


def test_mode_refuses_an_unknown_value() -> None:
    """--mode takes sequential or streaming only."""
    parser = _parser()
    with pytest.raises(SystemExit) as stopped:
        parser.parse_args(["--mode", "unknown"])
    assert stopped.value.code == 2


def test_profile_sets_profile_dir() -> None:
    """ScriptBuilder's --profile reaches PipelineConfig.profile_dir."""
    assert _applied(["--profile", PROBE_DIRECTORY]).profile_dir == PROBE_DIRECTORY


def test_import_loads_no_heavy_module() -> None:
    """A new interpreter importing the registry loads none of the heavy modules."""
    assert _heavy_modules_after_import(REGISTRY_MODULE) == []


def test_config_import_loads_no_heavy_module() -> None:
    """A new interpreter importing config.py loads none of the heavy modules."""
    assert _heavy_modules_after_import(CONFIG_MODULE) == []


@pytest.mark.parametrize("script_class", [_HostScript, _PlainScript], ids=["main_shape", "plain"])
def test_add_config_arguments_accepts_a_script_builder_parser(script_class: type[ScriptBuilder]) -> None:
    """The registry adds its arguments beside the --profile and --log a ScriptBuilder may hold, with no conflict."""
    host = script_class("probe", "probe")
    assert "--profile" in _options(host.parser)
    add_config_arguments(host.parser)
    assert {"--dataset-name", "--log", "--profile"} <= set(_options(host.parser))
