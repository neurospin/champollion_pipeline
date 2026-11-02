#!/usr/bin/env python3
"""
main.py profiling propagation: PipelineConfig.profile_dir, --profile overriding it, the per-stage
profiling segments and the --profile option forwarded to each stage script.

Requirements:
- REQ-PROFILING-BDRABCZUK-101290CAC6F3: PipelineConfig.profile_dir defaults to None.
- REQ-PROFILING-BDRABCZUK-63D5938DBD65: a --config YAML profile_dir is loaded.
- REQ-PROFILING-BDRABCZUK-47106DC69083: main.py --profile DIR replaces the config profile_dir.
- REQ-PROFILING-BDRABCZUK-3A09F7FBE41A: a config profile_dir alone turns profiling on.
- REQ-PROFILING-BDRABCZUK-F1F2589C89E5: each stage runs in a segment tagged with the stage name.
- REQ-PROFILING-BDRABCZUK-B09796DF6E14: each stage script receives --profile <absolute profile_dir>.
- REQ-PROFILING-BDRABCZUK-8E8CF75E4A51: no --profile reaches a stage script when profile_dir is None.
- REQ-PROFILING-BDRABCZUK-063F1345231D: profiling stays off by default (no CHAMPOLLION_PROFILE* in a stage).

Profiling is process-global (one session per interpreter, exit hooks, PYTHONPATH changes), so every
test that may start it runs main.py in a fresh interpreter; the argv tests stay in-process because
profiling never starts there.
"""

import importlib.util
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from champollion_pipeline.config import ConfigLoader, PipelineConfig
from champollion_pipeline.generate_champollion_config import GenerateChampollionConfig
from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from champollion_pipeline.generate_morphologist_graphs import GenerateMorphologistGraphs
from champollion_pipeline.generate_snapshots import GenerateSnapshots
from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings
from champollion_pipeline.run_cortical_tiles import RunCorticalTiles

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_PY = REPO_ROOT / "main.py"
PROFILE_ENV_PREFIX = "CHAMPOLLION_PROFILE"
STAGE_NAMES = (
    "generate_morphologist_graphs",
    "run_cortical_tiles",
    "generate_champollion_config",
    "generate_embeddings",
    "put_together_embeddings",
    "generate_snapshots",
)
STAGE_SCRIPTS = [
    ("GenerateMorphologistGraphsStage", GenerateMorphologistGraphs),
    ("RunCorticalTilesStage", RunCorticalTiles),
    ("GenerateChampollionConfigStage", GenerateChampollionConfig),
    ("GenerateEmbeddingsStage", GenerateEmbeddings),
    ("PutTogetherEmbeddingsStage", PutTogetherEmbeddings),
    ("GenerateSnapshotsStage", GenerateSnapshots),
]

# Runs ChampollionPipeline().main() in a fresh interpreter with sys.argv[2:] as its argv. Each of the six
# registered stages is replaced by a stub that records the CHAMPOLLION_PROFILE* variables it sees; with
# CAPTURE_CONFIG=1 the orchestrator itself is replaced and only the config it receives is reported.
_DRIVER = r"""
import importlib.util, json, os, sys

spec = importlib.util.spec_from_file_location("champollion_main_profile_driver", sys.argv[1])
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)

import champollion_utils.script_builder as script_builder

main.check_for_updates = lambda *args, **kwargs: None
script_builder.check_for_updates = lambda *args, **kwargs: None
report = {"stage_env": {}, "profile_dir": "<unset>"}


class RecordingStage:
    def __init__(self, name, config, logger):
        self.name = name

    def validate(self):
        return True

    def subject_input_dir(self):
        return None

    def output_dir(self):
        return None

    def execute(self):
        report["stage_env"][self.name] = sorted(k for k in os.environ if k.startswith("CHAMPOLLION_PROFILE"))
        return main.StageResult(stage_name=self.name, success=True, message="ok")


main.PipelineOrchestrator.STAGE_REGISTRY = {name: RecordingStage for name in main.PipelineOrchestrator.STAGE_REGISTRY}

if os.environ.get("CAPTURE_CONFIG") == "1":
    class CapturingOrchestrator:
        STAGE_REGISTRY = main.PipelineOrchestrator.STAGE_REGISTRY

        def __init__(self, config):
            report["profile_dir"] = getattr(config, "profile_dir", "<missing>")

        def run(self):
            return 0

    main.PipelineOrchestrator = CapturingOrchestrator

sys.argv = ["main.py", *sys.argv[2:]]
code = main.ChampollionPipeline().main()
print("REPORT=" + json.dumps(report))
sys.exit(code)
"""


def _clean_env(**extra: str) -> dict[str, str]:
    """os.environ without any CHAMPOLLION_PROFILE* key, with src/ and the repo root on PYTHONPATH."""
    env = {key: value for key, value in os.environ.items() if not key.startswith(PROFILE_ENV_PREFIX)}
    paths = [str(REPO_ROOT / "src"), str(REPO_ROOT), *filter(None, env.get("PYTHONPATH", "").split(os.pathsep))]
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.update(extra)
    return env


def _run_main(argv: list[str], cwd: Path, **extra_env: str) -> tuple[subprocess.CompletedProcess, dict | None]:
    """Run the driver; return the completed process and its REPORT (None when it printed none)."""
    completed = subprocess.run(
        [sys.executable, "-c", _DRIVER, str(MAIN_PY), *argv],
        cwd=str(cwd),
        env=_clean_env(**extra_env),
        capture_output=True,
        text=True,
        timeout=300,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("REPORT=")]
    report = json.loads(lines[-1][len("REPORT=") :]) if lines else None
    return completed, report


def _write_config(path: Path, outputs: Path, *, enable_stages: bool, profile_dir: Path | None = None) -> Path:
    """A --config YAML: outputs in ``outputs``, no log file, every stage on or off, optional profile_dir."""
    lines = [f"outputs_path: {outputs}", "log_to_file: false", "stages:"]
    lines += [f"  {name}: {str(enable_stages).lower()}" for name in STAGE_NAMES]
    if profile_dir is not None:
        lines.append(f"profile_dir: {profile_dir}")
    path.write_text("\n".join(lines) + "\n")
    return path


def _sidecars(profile_dir: Path) -> list[dict]:
    """Every process sidecar written under <profile_dir>/<run_id>/proc/."""
    return [json.loads(path.read_text()) for path in sorted(profile_dir.glob("*/proc/*.json"))]


def _output(completed: subprocess.CompletedProcess) -> str:
    return f"exit {completed.returncode}\nstdout:\n{completed.stdout[-3000:]}\nstderr:\n{completed.stderr[-3000:]}"


# ------------------------------------------------------------------ config


@pytest.mark.unit
def test_pipeline_config_profile_dir_defaults_to_none():
    """REQ-PROFILING-BDRABCZUK-101290CAC6F3: PipelineConfig().profile_dir is None."""
    assert PipelineConfig().profile_dir is None


@pytest.mark.unit
def test_load_from_yaml_reads_profile_dir(tmp_path):
    """REQ-PROFILING-BDRABCZUK-63D5938DBD65: a YAML profile_dir comes back as PipelineConfig.profile_dir."""
    config_file = tmp_path / "pipeline.yaml"
    config_file.write_text(f"outputs_path: {tmp_path / 'out'}\nprofile_dir: {tmp_path / 'prof'}\n")
    config = ConfigLoader.load_from_yaml(str(config_file))
    assert config.profile_dir == str(tmp_path / "prof")


# ------------------------------------------------------------------ main.py runs (fresh interpreter)


@pytest.mark.unit
def test_profile_option_overrides_config_profile_dir(tmp_path):
    """REQ-PROFILING-BDRABCZUK-47106DC69083: --profile DIR replaces the --config file's profile_dir."""
    cli_dir = tmp_path / "cli_prof"
    cli_dir.mkdir()
    config_file = _write_config(
        tmp_path / "pipeline.yaml", tmp_path / "out", enable_stages=False, profile_dir=tmp_path / "config_prof"
    )
    completed, report = _run_main(
        ["--config", str(config_file), "--profile", str(cli_dir), "--profile-mode", "light"],
        tmp_path,
        CAPTURE_CONFIG="1",
    )
    assert report is not None, _output(completed)
    assert report["profile_dir"] is not None, _output(completed)
    assert Path(report["profile_dir"]).absolute() == cli_dir, _output(completed)


@pytest.mark.unit
def test_config_profile_dir_writes_process_sidecar(tmp_path):
    """REQ-PROFILING-BDRABCZUK-3A09F7FBE41A: a config profile_dir, without --profile, writes a process sidecar."""
    profile_dir = tmp_path / "prof"
    profile_dir.mkdir()
    config_file = _write_config(
        tmp_path / "pipeline.yaml", tmp_path / "out", enable_stages=False, profile_dir=profile_dir
    )
    completed, _ = _run_main(["--config", str(config_file)], tmp_path, CHAMPOLLION_PROFILE_MODE="light")
    assert completed.returncode == 0, _output(completed)
    assert _sidecars(profile_dir), f"no sidecar under {profile_dir}/<run_id>/proc/\n{_output(completed)}"


@pytest.mark.unit
def test_each_stage_runs_in_its_own_segment(tmp_path):
    """REQ-PROFILING-BDRABCZUK-F1F2589C89E5: with --profile, a sidecar per stage carries stage == stage name."""
    profile_dir = tmp_path / "prof"
    profile_dir.mkdir()
    config_file = _write_config(tmp_path / "pipeline.yaml", tmp_path / "out", enable_stages=True)
    completed, report = _run_main(
        ["--config", str(config_file), "--profile", str(profile_dir), "--profile-mode", "light"], tmp_path
    )
    assert completed.returncode == 0, _output(completed)
    assert report is not None, _output(completed)
    assert sorted(report["stage_env"]) == sorted(STAGE_NAMES), _output(completed)
    stages = {sidecar["stage"] for sidecar in _sidecars(profile_dir)}
    missing = [name for name in STAGE_NAMES if name not in stages]
    assert not missing, f"no segment sidecar for stages {missing}; sidecar stages: {sorted(stages)}"


@pytest.mark.unit
def test_profiling_off_by_default(tmp_path):
    """REQ-PROFILING-BDRABCZUK-063F1345231D: no --profile, no profile_dir, scrubbed env -> stages see no
    CHAMPOLLION_PROFILE* variable (Contract 0, test 0b)."""
    config_file = _write_config(tmp_path / "pipeline.yaml", tmp_path / "out", enable_stages=True)
    completed, report = _run_main(["--config", str(config_file)], tmp_path)
    assert completed.returncode == 0, _output(completed)
    assert report is not None, _output(completed)
    assert sorted(report["stage_env"]) == sorted(STAGE_NAMES), _output(completed)
    leaked = {name: keys for name, keys in report["stage_env"].items() if keys}
    assert not leaked, f"profiling variables set by default: {leaked}"


# ------------------------------------------------------------------ stage argv (in-process)


def _load_main_module():
    spec = importlib.util.spec_from_file_location("_champollion_main_profile_under_test", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def main_module():
    return _load_main_module()


def _base_config(main_module, profile_dir: str | None):
    """PipelineConfig with every path the six stages read set, and profile_dir as given."""
    config = main_module.PipelineConfig(models_path="/models", outputs_path="/outputs")
    dataset = config.dataset
    dataset.input_path = "/raw"  # noqa: V101
    dataset.morphologist_graphs = "/graphs"  # noqa: V101
    dataset.cortical_tiles_output = "/tiles"  # noqa: V101
    dataset.crops_path = "/crops"  # noqa: V101
    dataset.datasets_root = "/data/MYDATASET"  # noqa: V101
    dataset.snapshots_path = "/snapshots"  # noqa: V101
    config.profile_dir = profile_dir  # noqa: V101
    return config


def _stage_namespace(main_module, stage_class_name, script_cls, config):
    """Run the stage with its script mocked; parse the captured argv with the script's real parser."""
    mocked = MagicMock()
    mocked.return_value.run.return_value = 0
    with patch.object(main_module, script_cls.__name__, mocked, create=True):
        stage = getattr(main_module, stage_class_name)(
            stage_class_name, config, logging.getLogger("test_main_profile_propagation")
        )
        result = stage.execute()
    assert result.success, result.message
    argv = list(mocked.return_value.parse_args.call_args[0][0])
    try:
        return argv, script_cls().parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"{script_cls.__name__} parser rejected {argv!r} (SystemExit {exc.code})")


@pytest.mark.unit
@pytest.mark.parametrize(("stage_class_name", "script_cls"), STAGE_SCRIPTS, ids=[name for name, _ in STAGE_SCRIPTS])
def test_stage_argv_forwards_profile(main_module, stage_class_name, script_cls, tmp_path):
    """REQ-PROFILING-BDRABCZUK-B09796DF6E14: profile_dir set -> the script parses profile == profile_dir."""
    profile_dir = str(tmp_path / "prof")
    argv, namespace = _stage_namespace(
        main_module, stage_class_name, script_cls, _base_config(main_module, profile_dir)
    )
    assert namespace.profile == profile_dir, f"argv was {argv!r}"


@pytest.mark.unit
def test_stage_argv_forwards_absolute_path_for_relative_profile_dir(main_module, tmp_path, monkeypatch):
    """REQ-PROFILING-BDRABCZUK-B09796DF6E14: a relative profile_dir reaches the script as its absolute path."""
    monkeypatch.chdir(tmp_path)
    argv, namespace = _stage_namespace(
        main_module, "GenerateEmbeddingsStage", GenerateEmbeddings, _base_config(main_module, "prof")
    )
    assert namespace.profile == str(tmp_path / "prof"), f"argv was {argv!r}"


@pytest.mark.unit
@pytest.mark.parametrize(("stage_class_name", "script_cls"), STAGE_SCRIPTS, ids=[name for name, _ in STAGE_SCRIPTS])
def test_stage_argv_has_no_profile_by_default(main_module, stage_class_name, script_cls):
    """REQ-PROFILING-BDRABCZUK-8E8CF75E4A51: profile_dir None -> no --profile in the stage argv."""
    argv, namespace = _stage_namespace(main_module, stage_class_name, script_cls, _base_config(main_module, None))
    assert not any(arg == "--profile" or arg.startswith("--profile=") for arg in argv), f"argv was {argv!r}"
    assert namespace.profile is None
