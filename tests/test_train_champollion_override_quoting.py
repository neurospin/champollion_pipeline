#!/usr/bin/env python3
"""
Hydra override values train_champollion builds from paths (TASK-PATTERNS-BDRABCZUK-14C3A11B7192).

train_champollion passes train.py ``hydra.run.dir=``, ``hydra.searchpath=[...]``
and ``++dataset_folder=`` overrides whose values come from filesystem paths and
from the localization YAML. A comma, bracket, space, quote, ``#`` or backslash
in such a value must survive the Hydra 1.3 override grammar unchanged.

The oracle is Hydra's own ``OverridesParser``: each override train_champollion
builds is parsed the way ``train.py`` parses its argv, and the parsed value is
compared with the intended string (REQ-TRAINOVR-BDRABCZUK-AC4F872AA6CE,
REQ-TRAINOVR-BDRABCZUK-342BCF3B458A, REQ-TRAINOVR-BDRABCZUK-86E38B8BB795).
The end-to-end tests run a stub Hydra app in place of champollion_V1's
``train.py`` (helpers of test_train_champollion_config_precedence.py) and read
what Hydra actually used.

A path holding ``${`` cannot be used by Hydra 1.3 at all (``hydra.runtime``
re-resolves the run and config directories), so train_champollion refuses it
before starting train.py (REQ-TRAINOVR-BDRABCZUK-6BAA37D1A869,
REQ-TRAINOVR-BDRABCZUK-A244968151C6).
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from hydra.core.override_parser.overrides_parser import OverridesParser
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from champollion_pipeline import train_champollion
from tests.test_train_champollion import make_script
from tests.test_train_champollion_config_precedence import DATASET, LOCALIZATION, REGION, _make_checkout, _write

_PARSER = OverridesParser.create()

# Characters that the Hydra override grammar treats specially, plus ordinary ones.
_TOKENS = [*"aZ0 ,[]{}()'\"\\#$=:~*?!@%^&+;|<>`", "é", "\t"]

# One path segment: no "/" (a separator) and no "${" (refused, see module docstring).
_SEGMENTS = (
    st.lists(st.sampled_from(_TOKENS), min_size=1, max_size=16)
    .map("".join)
    .filter(lambda s: "${" not in s and s not in (".", ".."))
)

# A dataset_folder string: anything YAML can hold on one line, "${...}" included.
_DATASET_FOLDERS = st.lists(st.sampled_from([*_TOKENS, "/", "${x}", "${oc.env:HOME}"]), min_size=1, max_size=20).map(
    "".join
)

_HYPOTHESIS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])

# One directory name holding every special character the audit lists (no "${").
_HOSTILE = "a,b [c] {d} 'e' \"f\" #g \\h =i"

_E2E_DUMP_ENV = "TRAINOVR_DUMP_PATH"

_E2E_TRAIN_PY = textwrap.dedent(
    """\
    import json
    import os

    import hydra
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf


    @hydra.main(config_name="config", version_base="1.1", config_path="configs")
    def main(cfg):
        hydra_cfg = HydraConfig.get()
        dump = {{
            "dataset_folder": OmegaConf.to_container(cfg, resolve=False).get("dataset_folder"),
            "run_dir": hydra_cfg.run.dir,
            "searchpath": list(hydra_cfg.searchpath),
        }}
        with open(os.environ["{dump_env}"], "w") as f:
            json.dump(dump, f)


    if __name__ == "__main__":
        main()
    """
).format(dump_env=_E2E_DUMP_ENV)

_E2E_DATASET_FOLDER = "/d/a,b #c 'q' [r] ${oc.env:HOME}"


def _train_argv(script, tmp_dir):
    """Run _run_training with chdir and train.py mocked; return the override argv elements."""
    captured = {}

    def fake_execute(cmd, shell=False):
        captured["cmd"] = cmd
        return 0

    script.execute_command = fake_execute
    with patch.object(train_champollion.os, "chdir"):
        script._run_training(str(tmp_dir))
    cmd = captured["cmd"]
    assert cmd[:2] == ["python3", "train.py"], f"unexpected argv {cmd}"
    end = cmd.index("--config-path") if "--config-path" in cmd else len(cmd)
    return cmd[2:end]


def _parsed_value(overrides, prefix):
    """Parse the one override starting with prefix with Hydra's grammar; return its value."""
    matching = [o for o in overrides if o.startswith(prefix)]
    assert len(matching) == 1, f"expected exactly one {prefix!r} override, got {matching}"
    return _PARSER.parse_overrides([matching[0]])[0].value()


def _script_args(config_dir, output_dir):
    return [
        "--dataset",
        DATASET,
        "--region",
        REGION,
        "--config-dir",
        str(config_dir),
        "--output_dir",
        str(output_dir),
        "--localization",
        LOCALIZATION,
    ]


def _write_localization(config_dir, text):
    _write(Path(config_dir) / "dataset_localization" / f"{LOCALIZATION}.yaml", text)


# --- REQ-TRAINOVR-BDRABCZUK-AC4F872AA6CE: hydra.run.dir -------------------------------------


@_HYPOTHESIS
@given(segment=_SEGMENTS)
def test_run_dir_override_parses_to_output_dir(segment):
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = os.path.abspath(os.path.join(tmp, segment, "out"))
        script = make_script(_script_args(os.path.join(tmp, "configs"), output_dir))

        value = _parsed_value(_train_argv(script, tmp), "hydra.run.dir=")

        assert value == output_dir, (
            f"REQ-TRAINOVR-BDRABCZUK-AC4F872AA6CE: hydra.run.dir parses to {value!r}, not the output directory"
        )


# --- REQ-TRAINOVR-BDRABCZUK-342BCF3B458A: hydra.searchpath ----------------------------------


@_HYPOTHESIS
@given(segment=_SEGMENTS)
def test_searchpath_override_parses_to_config_dir(segment):
    with tempfile.TemporaryDirectory() as tmp:
        config_dir = os.path.abspath(os.path.join(tmp, segment, "configs"))
        script = make_script(_script_args(config_dir, os.path.join(tmp, "out")))

        value = _parsed_value(_train_argv(script, tmp), "hydra.searchpath=")

        assert value == [f"file://{config_dir}"], (
            f"REQ-TRAINOVR-BDRABCZUK-342BCF3B458A: hydra.searchpath parses to {value!r}, "
            "not a one-element list holding the file:// URL of --config-dir"
        )


# --- REQ-TRAINOVR-BDRABCZUK-86E38B8BB795: ++dataset_folder ----------------------------------


@_HYPOTHESIS
@given(dataset_folder=_DATASET_FOLDERS)
def test_dataset_folder_override_parses_to_yaml_value(dataset_folder):
    with tempfile.TemporaryDirectory() as tmp:
        config_dir = os.path.join(tmp, "configs")
        text = "# @package _global_\n" + yaml.safe_dump({"dataset_folder": dataset_folder}, allow_unicode=True)
        _write_localization(config_dir, text)
        expected = yaml.safe_load(text)["dataset_folder"]
        script = make_script(_script_args(config_dir, os.path.join(tmp, "out")))

        value = _parsed_value(_train_argv(script, tmp), "++dataset_folder=")

        assert value == expected, (
            f"REQ-TRAINOVR-BDRABCZUK-86E38B8BB795: ++dataset_folder parses to {value!r}, "
            f"not the YAML value {expected!r}"
        )


def test_dataset_folder_quoted_yaml_with_comment(tmp_path):
    config_dir = tmp_path / "configs"
    _write_localization(config_dir, "# @package _global_\ndataset_folder: '/data/a,b root'  # shared mount\n")
    script = make_script(_script_args(config_dir, tmp_path / "out"))

    value = _parsed_value(_train_argv(script, tmp_path), "++dataset_folder=")

    assert value == "/data/a,b root", (
        f"REQ-TRAINOVR-BDRABCZUK-86E38B8BB795: ++dataset_folder parses to {value!r}; "
        "the YAML quotes and trailing comment must not reach the value"
    )


# --- REQ-TRAINOVR-BDRABCZUK-6BAA37D1A869 / -A244968151C6: "${" refused ----------------------


def _assert_refused(tmp_path, config_dir, output_dir, path, req):
    _write(Path(config_dir) / "dataset" / DATASET / f"{REGION}.yaml", "origin: config_dir\n")
    script = make_script(_script_args(config_dir, output_dir))
    script.execute_command = lambda cmd, shell=False: pytest.fail(f"{req}: train.py started with {cmd}")
    with patch.object(train_champollion.os, "chdir"), pytest.raises(ValueError) as excinfo:
        script.run()
    assert path in str(excinfo.value), f"{req}: ValueError message {excinfo.value!s} does not name {path}"


def test_output_dir_with_interpolation_is_refused(tmp_path):
    output_dir = os.path.abspath(str(tmp_path / "run_${x}" / "out"))
    _assert_refused(tmp_path, tmp_path / "configs", output_dir, output_dir, "REQ-TRAINOVR-BDRABCZUK-6BAA37D1A869")


def test_config_dir_with_interpolation_is_refused(tmp_path):
    config_dir = os.path.abspath(str(tmp_path / "cfg_${x}" / "configs"))
    _assert_refused(tmp_path, config_dir, tmp_path / "out", config_dir, "REQ-TRAINOVR-BDRABCZUK-A244968151C6")


# --- End to end through a stub Hydra train.py ----------------------------------------------


@pytest.fixture(scope="module")
def hostile_run(tmp_path_factory):
    """Run train_champollion once with hostile config and output paths; return Hydra's view."""
    root = tmp_path_factory.mktemp("trainovr")
    pkg = _make_checkout(root)
    (pkg / "train.py").write_text(_E2E_TRAIN_PY)
    config_dir = root / _HOSTILE / "configs"
    output_dir = root / _HOSTILE / "out"
    _write(config_dir / "dataset" / DATASET / f"{REGION}.yaml", "origin: config_dir\n")
    _write_localization(
        config_dir, f'# @package _global_\ndataset_folder: "{_E2E_DATASET_FOLDER}"  # localization comment\n'
    )
    dump = root / "hydra_view.json"
    result = {"config_dir": str(config_dir), "output_dir": str(output_dir), "stderr": "", "dump": None}

    def real_execute(cmd, **_kwargs):
        env = dict(os.environ, **{_E2E_DUMP_ENV: str(dump)})
        proc = subprocess.run([sys.executable, *cmd[1:]], cwd=os.getcwd(), env=env, capture_output=True, text=True)
        result["stderr"] = proc.stderr[-3000:]
        return proc.returncode

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(train_champollion, "_CONTRASTIVE_DIR", str(pkg))
        script = make_script([*_script_args(config_dir, output_dir), "--cpu"])
        script.execute_command = real_execute
        result["status"] = script.run()
    if dump.is_file():
        result["dump"] = json.loads(dump.read_text())
    return result


def _hydra_view(hostile_run, req):
    assert hostile_run["status"] == 0 and hostile_run["dump"] is not None, (
        f"{req}: stub train.py failed for paths under {_HOSTILE!r}:\n{hostile_run['stderr']}"
    )
    return hostile_run["dump"]


@pytest.mark.integration
def test_run_dir_reaches_hydra_end_to_end(hostile_run):
    view = _hydra_view(hostile_run, "REQ-TRAINOVR-BDRABCZUK-AC4F872AA6CE")
    assert view["run_dir"] == hostile_run["output_dir"], (
        f"REQ-TRAINOVR-BDRABCZUK-AC4F872AA6CE: Hydra ran in {view['run_dir']!r}, not the output directory"
    )


@pytest.mark.integration
def test_searchpath_reaches_hydra_end_to_end(hostile_run):
    view = _hydra_view(hostile_run, "REQ-TRAINOVR-BDRABCZUK-342BCF3B458A")
    assert view["searchpath"] == [f"file://{hostile_run['config_dir']}"], (
        f"REQ-TRAINOVR-BDRABCZUK-342BCF3B458A: Hydra searchpath is {view['searchpath']!r}"
    )


@pytest.mark.integration
def test_dataset_folder_reaches_hydra_end_to_end(hostile_run):
    view = _hydra_view(hostile_run, "REQ-TRAINOVR-BDRABCZUK-86E38B8BB795")
    assert view["dataset_folder"] == _E2E_DATASET_FOLDER, (
        f"REQ-TRAINOVR-BDRABCZUK-86E38B8BB795: Hydra composed dataset_folder {view['dataset_folder']!r}"
    )
