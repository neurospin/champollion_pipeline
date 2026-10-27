"""Tests for the once-per-dataset Champollion config of a streaming run (TASK-STREAMING-BDRABCZUK-B75262C2ED8E).

Requirements:
- REQ-STREAMING-BDRABCZUK-84007FD84E13 script arguments from DatasetConfig
- REQ-STREAMING-BDRABCZUK-DF21D63C715E output and external_config are the configs root
- REQ-STREAMING-BDRABCZUK-EB1A1B920683 returns the configs root
- REQ-STREAMING-BDRABCZUK-4F5CADBDFDE4 skipped on resume (localization YAML exists)
- REQ-STREAMING-BDRABCZUK-1D8F048F111B generated once per dataset
- REQ-STREAMING-BDRABCZUK-AE65AC0C1ED5 non-zero script code raises RuntimeError
- REQ-STREAMING-BDRABCZUK-CF986A52F725 dry run skips the script
- REQ-STREAMING-BDRABCZUK-AB218919A186 missing crops path raises FileNotFoundError
- REQ-STREAMING-BDRABCZUK-FAAAA3B2850B configs root inside champollion_V1 raises ValueError

GenerateChampollionConfig is replaced, in the module under test, by a subclass that keeps the real argument
parser and records the parsed arguments instead of writing configs; on success it writes only the
dataset_localization YAML the real script writes last. Synthetic paths only: nothing is read or written inside
external/champollion_V1.
"""

import re
from dataclasses import dataclass, field
from os.path import abspath, join
from pathlib import Path

import pytest

from champollion_pipeline.config import DatasetConfig
from champollion_pipeline.derivatives_layout import compute_champollion_configs_root
from champollion_pipeline.generate_champollion_config import GenerateChampollionConfig
from champollion_pipeline.streaming import dataset_config
from champollion_pipeline.streaming.dataset_config import ensure_dataset_config
from champollion_pipeline.streaming.settings import StreamingSettings

DATASET = "SYNTHETIC_STREAMING_DS"
MASKS = "synthetic_masks_7"
LOCALIZATION = "synthetic_site"
FAILURE_CODE = 37


@dataclass
class ScriptRecorder:
    """What the fake GenerateChampollionConfig saw: parsed arguments per run, and the code run() returns."""

    code: int = 0
    runs: list[dict[str, object]] = field(default_factory=list)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> ScriptRecorder:
    rec = ScriptRecorder()

    class FakeGenerateChampollionConfig(GenerateChampollionConfig):
        def run(self) -> int:
            rec.runs.append(dict(vars(self.args)))
            if rec.code == 0:
                yaml = Path(self.args.external_config) / "dataset_localization" / f"{self.args.localization}.yaml"
                yaml.parent.mkdir(parents=True, exist_ok=True)
                yaml.write_text("# @package _global_\n")
            return rec.code

    monkeypatch.setattr(dataset_config, "GenerateChampollionConfig", FakeGenerateChampollionConfig)
    return rec


def _crops_path(parent: Path) -> str:
    return str(parent / DATASET / "derivatives" / "cortical_tiles-2026" / "crops" / MASKS / "2mm")


def _dataset(crops_path: str) -> DatasetConfig:
    return DatasetConfig(name=DATASET, crops_path=crops_path, masks_version=MASKS, dataset_localization=LOCALIZATION)


def _settings(tmp_path: Path, dry_run: bool = False) -> StreamingSettings:
    return StreamingSettings(
        subjects_dir=str(tmp_path / "subjects"),
        output_dir=str(tmp_path / "out"),
        log_dir=str(tmp_path / "logs"),
        dry_run=dry_run,
    )


@pytest.fixture
def crops(tmp_path: Path) -> str:
    """An existing, empty synthetic crops directory of DATASET under tmp_path/data."""
    path = _crops_path(tmp_path / "data")
    Path(path).mkdir(parents=True)
    return path


def _configs_root(tmp_path: Path) -> str:
    return compute_champollion_configs_root(str(tmp_path / "data"), DATASET)


def test_script_arguments_come_from_dataset_config(tmp_path: Path, crops: str, recorder: ScriptRecorder) -> None:
    ensure_dataset_config(_settings(tmp_path), _dataset(crops))

    assert len(recorder.runs) == 1
    args = recorder.runs[0]
    assert args["crop_path"] == crops
    assert args["dataset"] == DATASET
    assert args["masks"] == MASKS
    assert args["localization"] == LOCALIZATION


def test_output_and_external_config_are_the_configs_root(tmp_path: Path, crops: str, recorder: ScriptRecorder) -> None:
    ensure_dataset_config(_settings(tmp_path), _dataset(crops))

    assert len(recorder.runs) == 1
    args = recorder.runs[0]
    assert args["output"] is not None
    assert args["external_config"] is not None
    assert abspath(str(args["output"])) == _configs_root(tmp_path)
    assert abspath(str(args["external_config"])) == _configs_root(tmp_path)


def test_returns_the_configs_root(tmp_path: Path, crops: str, recorder: ScriptRecorder) -> None:
    assert ensure_dataset_config(_settings(tmp_path), _dataset(crops)) == _configs_root(tmp_path)


def test_existing_localization_yaml_skips_the_script(tmp_path: Path, crops: str, recorder: ScriptRecorder) -> None:
    existing = Path(_configs_root(tmp_path)) / "dataset_localization" / f"{LOCALIZATION}.yaml"
    existing.parent.mkdir(parents=True)
    existing.write_text("# @package _global_\n")

    ensure_dataset_config(_settings(tmp_path), _dataset(crops))

    assert recorder.runs == []


def test_second_call_for_one_dataset_runs_the_script_once(tmp_path: Path, crops: str, recorder: ScriptRecorder) -> None:
    settings, dataset = _settings(tmp_path), _dataset(crops)

    ensure_dataset_config(settings, dataset)
    ensure_dataset_config(settings, dataset)

    assert len(recorder.runs) == 1


def test_non_zero_script_code_raises_runtime_error(tmp_path: Path, crops: str, recorder: ScriptRecorder) -> None:
    recorder.code = FAILURE_CODE

    with pytest.raises(RuntimeError) as excinfo:
        ensure_dataset_config(_settings(tmp_path), _dataset(crops))

    assert len(recorder.runs) == 1, "the error must come from the script's exit code"
    assert DATASET in str(excinfo.value)
    assert str(FAILURE_CODE) in str(excinfo.value)


def test_dry_run_skips_the_script(tmp_path: Path, recorder: ScriptRecorder) -> None:
    # In a dry run no producer has run yet, so the crops directory does not exist.
    dataset = _dataset(_crops_path(tmp_path / "data"))

    result = ensure_dataset_config(_settings(tmp_path, dry_run=True), dataset)

    assert result == _configs_root(tmp_path)
    assert recorder.runs == []


def test_missing_crops_path_raises_file_not_found(tmp_path: Path, recorder: ScriptRecorder) -> None:
    missing = _crops_path(tmp_path / "data")

    with pytest.raises(FileNotFoundError, match=re.escape(missing)):
        ensure_dataset_config(_settings(tmp_path), _dataset(missing))

    assert recorder.runs == []


@pytest.mark.parametrize("dry_run", [False, True])
def test_configs_root_inside_champollion_v1_raises_value_error(
    tmp_path: Path, recorder: ScriptRecorder, dry_run: bool
) -> None:
    # Path computation only: the crops path under the checkout does not exist and nothing is written there.
    checkout = GenerateChampollionConfig().parse_args(["/unused", "--dataset", DATASET]).champollion_loc
    crops_inside_checkout = _crops_path(Path(join(abspath(checkout), "synthetic_parent")))

    with pytest.raises(ValueError):
        ensure_dataset_config(_settings(tmp_path, dry_run=dry_run), _dataset(crops_inside_checkout))

    assert recorder.runs == []
