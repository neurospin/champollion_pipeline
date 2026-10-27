"""Tests for StreamingSettings (TASK-STREAMING-BDRABCZUK-7958B66DBEE1).

Requirements:
- REQ-STREAMING-BDRABCZUK-1E1A0B41908A field set
- REQ-STREAMING-BDRABCZUK-B02DFB146F6A required paths
- REQ-STREAMING-BDRABCZUK-8DF1FCEF5D6B immutability
- REQ-STREAMING-BDRABCZUK-E7E4609E178F defaults
- REQ-STREAMING-BDRABCZUK-46D0E63B85A5 producer workers from n_workers
- REQ-STREAMING-BDRABCZUK-0D1C7078026B producer workers from os.cpu_count()
- REQ-STREAMING-BDRABCZUK-2ABBB13D308D jobs per producer worker
- REQ-STREAMING-BDRABCZUK-93CFA868B070 unknown CPU count
- REQ-STREAMING-BDRABCZUK-9D98B17FF5D0 non-positive durations rejected
"""

import dataclasses
import os

import pytest

from champollion_pipeline.streaming.settings import StreamingSettings

PATHS = {"subjects_dir": "/data/subjects", "output_dir": "/data/out", "log_dir": "/data/out/logs"}
EXPECTED_FIELDS = (
    "subjects_dir",
    "output_dir",
    "log_dir",
    "n_workers",
    "poll_interval",
    "worker_timeout",
    "consumer_stall_timeout",
    "device",
    "dry_run",
)


def _settings(**overrides: object) -> StreamingSettings:
    return StreamingSettings(**{**PATHS, **overrides})


def _fake_cpu_count(monkeypatch: pytest.MonkeyPatch, value: int | None) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: value)


class TestFields:
    def test_field_names(self):
        assert tuple(f.name for f in dataclasses.fields(StreamingSettings)) == EXPECTED_FIELDS

    @pytest.mark.parametrize("missing", sorted(PATHS))
    def test_missing_path_raises(self, missing):
        kwargs = {name: value for name, value in PATHS.items() if name != missing}
        with pytest.raises(TypeError):
            StreamingSettings(**kwargs)


class TestFrozen:
    @pytest.mark.parametrize("name", EXPECTED_FIELDS)
    def test_assignment_raises(self, name):
        settings = _settings()
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(settings, name, getattr(settings, name))


class TestDefaults:
    def test_defaults(self):
        settings = _settings()
        assert (settings.n_workers, settings.poll_interval, settings.worker_timeout, settings.dry_run) == (
            0,
            10,
            7200,
            False,
        )


class TestProducerWorkers:
    def test_positive_n_workers_kept(self, monkeypatch):
        _fake_cpu_count(monkeypatch, 16)
        assert _settings(n_workers=3).producer_workers == 3

    @pytest.mark.parametrize("n_workers", [0, -1, -8])
    def test_non_positive_n_workers_uses_cpu_count(self, monkeypatch, n_workers):
        _fake_cpu_count(monkeypatch, 12)
        assert _settings(n_workers=n_workers).producer_workers == 12


class TestNjobsPerWorker:
    @pytest.mark.parametrize(
        ("cpus", "n_workers", "expected"),
        [(16, 4, 4), (16, 3, 5), (16, 0, 1), (4, 8, 1), (8, 1, 8)],
    )
    def test_quotient_with_floor_of_one(self, monkeypatch, cpus, n_workers, expected):
        _fake_cpu_count(monkeypatch, cpus)
        assert _settings(n_workers=n_workers).njobs_per_worker == expected


class TestUnknownCpuCount:
    def test_producer_workers_is_one(self, monkeypatch):
        _fake_cpu_count(monkeypatch, None)
        assert _settings(n_workers=0).producer_workers == 1

    def test_njobs_per_worker_is_one(self, monkeypatch):
        _fake_cpu_count(monkeypatch, None)
        assert _settings(n_workers=2).njobs_per_worker == 1


class TestDurationValidation:
    @pytest.mark.parametrize("name", ["poll_interval", "worker_timeout", "consumer_stall_timeout"])
    @pytest.mark.parametrize("value", [0, -5])
    def test_non_positive_duration_raises(self, name, value):
        with pytest.raises(ValueError, match=name):
            _settings(**{name: value})

    def test_positive_durations_accepted(self):
        settings = _settings(poll_interval=1, worker_timeout=60, consumer_stall_timeout=30)
        assert (settings.poll_interval, settings.worker_timeout, settings.consumer_stall_timeout) == (1, 60, 30)
