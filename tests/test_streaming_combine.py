"""Tests for the streaming combine (TASK-STREAMING-BDRABCZUK-877D715C6863).

Synthetic data only: the per-region embeddings are tiny CSVs written under tmp_path in the layout the
resident consumer is expected to append to (<region_embeddings>/<region>/full_embeddings.csv, ID = str(scan)).
PutTogetherEmbeddings is replaced by a recording subclass, so no real combine runs.

Requirements:
- REQ-STREAMING-BDRABCZUK-E7C9DB9515C2 default embeddings source
- REQ-STREAMING-BDRABCZUK-A09C5407F5CA explicit embeddings source
- REQ-STREAMING-BDRABCZUK-F195634E69BE versioned output path
- REQ-STREAMING-BDRABCZUK-BEF0DBCD9467 output path from cortical_tiles_output
- REQ-STREAMING-BDRABCZUK-6F70C33E6FAE output path from the run output directory
- REQ-STREAMING-BDRABCZUK-797D374B26D4 unresolvable embeddings source refused
- REQ-STREAMING-BDRABCZUK-4F189A26E1BB dry run
- REQ-STREAMING-BDRABCZUK-A5E42783ECF1 missing embeddings source
- REQ-STREAMING-BDRABCZUK-E0525347B645 non-zero exit code raises
- REQ-STREAMING-BDRABCZUK-32015336B60A returns the combined directory
- REQ-STREAMING-BDRABCZUK-33596ED06B9F start log record
- REQ-STREAMING-BDRABCZUK-81B4A92FCE08 QC warning on empty output
- REQ-STREAMING-BDRABCZUK-233188A22608 no fold progress text
"""

import logging
import os
import re
from os.path import join
from types import SimpleNamespace

import pytest

from champollion_pipeline.config import DatasetConfig
from champollion_pipeline.derivatives_layout import compute_combined_embeddings_dir, compute_region_embeddings_dir
from champollion_pipeline.put_together_embeddings import PutTogetherEmbeddings
from champollion_pipeline.streaming import combine
from champollion_pipeline.streaming.settings import StreamingSettings

LOGGER_NAME = "champollion_pipeline.streaming.combine"
LOG_LEVEL = 22
MASKS = "test_masks"
REGION = "SC-sylv_left"
FOLD_PROGRESS = re.compile(r"[Ff]old\s+\d+\s*(?:/|of)\s*\d+")


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def records():
    """Records of the combine module logger, captured on the logger itself (no propagation assumed)."""
    handler = _ListHandler()
    module_logger = logging.getLogger(LOGGER_NAME)
    previous_level = module_logger.level
    module_logger.addHandler(handler)
    module_logger.setLevel(logging.DEBUG)
    yield handler.records
    module_logger.removeHandler(handler)
    module_logger.setLevel(previous_level)


@pytest.fixture
def recorder(monkeypatch, records):
    """Replace PutTogetherEmbeddings in the combine module with a recording subclass."""
    state = SimpleNamespace(constructed=0, runs=[], code=0, write_csv=True, messages_at_run=None)

    class RecordingPutTogetherEmbeddings(PutTogetherEmbeddings):
        def __init__(self) -> None:
            super().__init__()
            state.constructed += 1

        def run(self) -> int:
            state.runs.append(self.args)
            state.messages_at_run = [(r.levelno, r.getMessage()) for r in records]
            if state.write_csv:
                os.makedirs(self.args.output_path, exist_ok=True)
                with open(join(self.args.output_path, f"{REGION}_embeddings.csv"), "w") as handle:
                    handle.write("ID,dim1\nsub-01_ses-1,0.5\n")
            return state.code

    monkeypatch.setattr(combine, "PutTogetherEmbeddings", RecordingPutTogetherEmbeddings)
    return state


def _settings(tmp_path, **overrides: object) -> StreamingSettings:
    paths = {
        "subjects_dir": str(tmp_path / "subjects"),
        "output_dir": str(tmp_path / "out"),
        "log_dir": str(tmp_path / "logs"),
    }
    return StreamingSettings(**{**paths, **overrides})


def _write_region_embeddings(source: str) -> str:
    """Write one per-region full_embeddings.csv with one scan row under source; return source."""
    region_dir = join(source, REGION)
    os.makedirs(region_dir, exist_ok=True)
    with open(join(region_dir, "full_embeddings.csv"), "w") as handle:
        handle.write("ID,dim1\nsub-01_ses-1,0.5\n")
    return source


def _versioned_dataset(tmp_path, *, with_source: bool = True) -> DatasetConfig:
    root = str(tmp_path / "dataset")
    if with_source:
        _write_region_embeddings(compute_region_embeddings_dir(root, MASKS))
    return DatasetConfig(datasets_root=root, masks_version=MASKS)


def _explicit_source_dataset(tmp_path, **overrides: object) -> DatasetConfig:
    source = _write_region_embeddings(str(tmp_path / "explicit_embeddings"))
    return DatasetConfig(**{"datasets_root": "", "embeddings_path": source, "masks_version": MASKS, **overrides})


class TestEmbeddingsSource:
    def test_default_source(self, tmp_path, recorder):
        dataset = _versioned_dataset(tmp_path)
        combine.run_combine(_settings(tmp_path), dataset)
        assert [args.embeddings_source for args in recorder.runs] == [
            compute_region_embeddings_dir(dataset.datasets_root, MASKS)
        ]

    def test_explicit_source(self, tmp_path, recorder):
        dataset = _explicit_source_dataset(tmp_path, datasets_root=str(tmp_path / "dataset"))
        combine.run_combine(_settings(tmp_path), dataset)
        assert [args.embeddings_source for args in recorder.runs] == [dataset.embeddings_path]


class TestOutputPath:
    def test_versioned_output_path(self, tmp_path, recorder):
        dataset = _versioned_dataset(tmp_path)
        combine.run_combine(_settings(tmp_path), dataset)
        assert [args.output_path for args in recorder.runs] == [
            compute_combined_embeddings_dir(dataset.datasets_root, MASKS)
        ]

    def test_output_path_from_cortical_tiles_output(self, tmp_path, recorder):
        dataset = _explicit_source_dataset(tmp_path, cortical_tiles_output=str(tmp_path / "tiles"))
        combine.run_combine(_settings(tmp_path), dataset)
        assert [args.output_path for args in recorder.runs] == [str(tmp_path / "tiles")]

    def test_output_path_from_run_output_dir(self, tmp_path, recorder):
        settings = _settings(tmp_path)
        dataset = _explicit_source_dataset(tmp_path, cortical_tiles_output="")
        combine.run_combine(settings, dataset)
        assert [args.output_path for args in recorder.runs] == [settings.output_dir]


class TestUnresolvableSource:
    @pytest.mark.parametrize("dry_run", [False, True])
    def test_raises_before_constructing(self, tmp_path, recorder, dry_run):
        dataset = DatasetConfig(datasets_root="", embeddings_path="", masks_version=MASKS)
        with pytest.raises(ValueError):
            combine.run_combine(_settings(tmp_path, dry_run=dry_run), dataset)
        assert recorder.constructed == 0


class TestDryRun:
    def test_returns_combined_dir_without_running(self, tmp_path, recorder):
        dataset = _versioned_dataset(tmp_path, with_source=False)
        result = combine.run_combine(_settings(tmp_path, dry_run=True), dataset)
        assert result == compute_combined_embeddings_dir(dataset.datasets_root, MASKS)
        assert recorder.runs == []


class TestMissingSource:
    def test_raises_with_source_path(self, tmp_path, recorder):
        dataset = _versioned_dataset(tmp_path, with_source=False)
        with pytest.raises(FileNotFoundError) as excinfo:
            combine.run_combine(_settings(tmp_path), dataset)
        assert compute_region_embeddings_dir(dataset.datasets_root, MASKS) in str(excinfo.value)
        assert recorder.runs == []


class TestExitCode:
    def test_non_zero_code_raises(self, tmp_path, recorder):
        recorder.code = 7
        with pytest.raises(RuntimeError, match="7"):
            combine.run_combine(_settings(tmp_path), _versioned_dataset(tmp_path))

    def test_returns_output_path(self, tmp_path, recorder):
        result = combine.run_combine(_settings(tmp_path), _versioned_dataset(tmp_path))
        assert [result] == [args.output_path for args in recorder.runs]


class TestLogging:
    def test_start_record(self, tmp_path, recorder):
        dataset = _versioned_dataset(tmp_path)
        source = compute_region_embeddings_dir(dataset.datasets_root, MASKS)
        combined = compute_combined_embeddings_dir(dataset.datasets_root, MASKS)
        combine.run_combine(_settings(tmp_path), dataset)
        assert recorder.messages_at_run is not None
        start = [m for level, m in recorder.messages_at_run if level == LOG_LEVEL and source in m and combined in m]
        assert len(start) == 1

    def test_no_fold_progress_text(self, tmp_path, recorder, records):
        combine.run_combine(_settings(tmp_path), _versioned_dataset(tmp_path))
        recorder.write_csv = False
        combine.run_combine(_settings(tmp_path / "qc"), _versioned_dataset(tmp_path / "qc"))
        combine.run_combine(_settings(tmp_path / "dry", dry_run=True), _versioned_dataset(tmp_path / "dry"))
        assert len(recorder.runs) == 2
        assert [r.getMessage() for r in records if FOLD_PROGRESS.search(r.getMessage())] == []


class TestQc:
    def test_warning_on_empty_output(self, tmp_path, recorder, records):
        recorder.write_csv = False
        dataset = _versioned_dataset(tmp_path)
        combined = compute_combined_embeddings_dir(dataset.datasets_root, MASKS)
        result = combine.run_combine(_settings(tmp_path), dataset)
        assert result == combined
        warnings = [r.getMessage() for r in records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert combined in warnings[0]

    def test_no_warning_when_output_holds_csv(self, tmp_path, recorder, records):
        combine.run_combine(_settings(tmp_path), _versioned_dataset(tmp_path))
        assert len(recorder.runs) == 1
        assert [r for r in records if r.levelno == logging.WARNING] == []
