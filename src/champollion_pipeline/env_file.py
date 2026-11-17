"""Env sources: five config fields read from a .env file (main.py --env) and from the process environment.

hf_token, dependencies, config_path and root_path have no main.py argument (cli_options.EXEMPT_FIELDS); four
variables set them: HF_TOKEN, CHAMPOLLION_ROOT_PATH, CHAMPOLLION_CONFIG_PATH and CHAMPOLLION_DEPENDENCIES (a JSON
object of stage names to stage name lists). A fifth, CHAMPOLLION_DATA_ROOT, sets dataset.champollion_data_root, whose
--champollion-data-root argument wins over it. They are read from the --env file, in the python-dotenv syntax without
${VAR} expansion, then from the process environment, whose value wins; main.py applies them over the --config YAML
and under the command-line arguments.
A missing or unreadable file, an unknown CHAMPOLLION_ variable in the file or a bad CHAMPOLLION_DEPENDENCIES raises
EnvFileError; an unknown CHAMPOLLION_ variable in the process environment logs one WARNING and is ignored. Nothing is
written to os.environ, and no value, the token included, reaches a log record or an error message.

Inputs:
    the --env file path, or None when main.py runs without --env
    the process environment (os.environ unless a mapping is given)
    config.py: PipelineConfig, its default stage names and the fields set
Outputs:
    the values of the five variables, variable name to field value
    the PipelineConfig with those fields set, in place
    EnvFileError naming --env and the path, or the variable at fault
    one WARNING log record per unknown CHAMPOLLION_ variable of the process environment
    main.py: adds --env, reads the values before any stage, applies them to the configuration it runs

```mermaid
flowchart LR
    envfile[(--env file)] --> read[read_env_values]
    procenv[(process environment)] --> read
    read --> values[(variable values)]
    read --> error[/EnvFileError/]
    read --> warning[/WARNING per unknown variable/]
    values --> apply[apply_env_values]
    apply --> config[(PipelineConfig)]
    main[main.py] -.calls.-> read
    main -.calls.-> apply
```
"""

import argparse
import json
import logging
import os
from collections.abc import Iterable, Mapping

from dotenv import dotenv_values

from champollion_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

ENV_OPTION = "--env"
DEFAULT_ENV_FILE = ".env"
VARIABLE_PREFIX = "CHAMPOLLION_"
DEPENDENCIES_VARIABLE = "CHAMPOLLION_DEPENDENCIES"

# Env variables: name -> path of the config field it sets: the cli_options.EXEMPT_FIELDS, then champollion_data_root.
ENV_VARIABLES: dict[str, tuple[str, ...]] = {
    "HF_TOKEN": ("dataset", "hf_token"),
    "CHAMPOLLION_ROOT_PATH": ("root_path",),
    "CHAMPOLLION_CONFIG_PATH": ("dataset", "config_path"),
    DEPENDENCIES_VARIABLE: ("dependencies",),
    "CHAMPOLLION_DATA_ROOT": ("dataset", "champollion_data_root"),
}

_STAGE_NAMES = frozenset(PipelineConfig().stages)
_PROCESS_ENVIRONMENT = "the process environment"
_KNOWN_VARIABLES = ", ".join(ENV_VARIABLES)
_ENV_HELP = (
    "Read HF_TOKEN, CHAMPOLLION_ROOT_PATH, CHAMPOLLION_CONFIG_PATH, CHAMPOLLION_DEPENDENCIES (JSON) and "
    "CHAMPOLLION_DATA_ROOT from the .env file PATH (bare --env: ./.env; write --env=PATH, or put a bare --env after "
    "INPUT and OUTPUT). The same variables in the process environment win over the file; both win over the --config "
    "file, and --champollion-data-root wins over CHAMPOLLION_DATA_ROOT."
)


class EnvFileError(ValueError):
    """A --env file or a variable main.py cannot use; the message names the file or the variable, never a value."""


def add_env_argument(parser: argparse.ArgumentParser) -> None:  # noqa: V103 - used by main.py
    """Add --env [PATH] to parser: dest env, None without --env, DEFAULT_ENV_FILE for a bare --env."""
    parser.add_argument(
        ENV_OPTION, nargs="?", const=DEFAULT_ENV_FILE, default=None, metavar="PATH", dest="env", help=_ENV_HELP
    )


def read_env_values(  # noqa: V103 - used by main.py
    env_file: str | None, environ: Mapping[str, str] | None = None
) -> dict[str, object]:
    """Value of each env variable set: the --env file's, then the process environment's over it.

    Raises EnvFileError on an unreadable file, an unknown CHAMPOLLION_ variable in the file or a bad
    CHAMPOLLION_DEPENDENCIES in either source. Logs one WARNING per unknown CHAMPOLLION_ variable of environ.
    """
    environ = os.environ if environ is None else environ
    values = _read_env_file(env_file) if env_file is not None else {}
    values.update(_read_process_environment(environ))
    return values


def apply_env_values(config: PipelineConfig, values: Mapping[str, object]) -> None:  # noqa: V103 - used by main.py
    """Set, in place, the field of each variable in values to its value."""
    for variable, value in values.items():
        *owners, field_name = ENV_VARIABLES[variable]
        target: object = config
        for owner in owners:
            target = getattr(target, owner)
        setattr(target, field_name, value)


def _read_env_file(path: str) -> dict[str, object]:
    """The env variables of the file at path, python-dotenv syntax, no expansion; a key with no value is left out."""
    source = f"{ENV_OPTION} {path}"
    try:
        with open(path, encoding="utf-8") as stream:
            entries = dotenv_values(stream=stream, interpolate=False)
    except (OSError, UnicodeDecodeError):
        raise EnvFileError(f"{source}: no readable file at this path") from None
    unknown = _unknown_names(entries)
    if unknown:
        raise EnvFileError(f"{source}: unknown variable {', '.join(unknown)} (known: {_KNOWN_VARIABLES})")
    return {
        name: _parse_value(name, text, source)
        for name, text in entries.items()
        if name in ENV_VARIABLES and text is not None
    }


def _read_process_environment(environ: Mapping[str, str]) -> dict[str, object]:
    """The env variables of environ; one WARNING names each unknown CHAMPOLLION_ variable, which is ignored."""
    for name in _unknown_names(environ):
        logger.warning(f"{name} in {_PROCESS_ENVIRONMENT} is not a champollion variable; ignored ({_KNOWN_VARIABLES})")
    return {name: _parse_value(name, environ[name], _PROCESS_ENVIRONMENT) for name in ENV_VARIABLES if name in environ}


def _unknown_names(names: Iterable[str]) -> list[str]:
    """The CHAMPOLLION_ names among names that are not env variables, sorted."""
    return sorted(name for name in names if name.startswith(VARIABLE_PREFIX) and name not in ENV_VARIABLES)


def _parse_value(name: str, text: str, source: str) -> object:
    """The field value of a variable's text: the text itself, or the stage dependencies object it holds as JSON."""
    if name != DEPENDENCIES_VARIABLE:
        return text
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if not _is_stage_dependencies(value):
        raise EnvFileError(
            f"{DEPENDENCIES_VARIABLE} in {source} is not a JSON object of stage names to stage name lists "
            f"(stage names: {', '.join(sorted(_STAGE_NAMES))})"
        )
    return value


def _is_stage_dependencies(value: object) -> bool:
    """True for a dict whose keys are stage names and whose values are lists of stage names."""
    return isinstance(value, dict) and all(
        key in _STAGE_NAMES and isinstance(needs, list) and all(_is_stage_name(need) for need in needs)
        for key, needs in value.items()
    )


def _is_stage_name(value: object) -> bool:
    return isinstance(value, str) and value in _STAGE_NAMES
