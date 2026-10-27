"""Settings of one streaming run, shared by the orchestrator, the producer workers and the consumer."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_POLL_INTERVAL = 10
DEFAULT_WORKER_TIMEOUT = 7200
# The longest normal gap between two scans reaching the consumer is one per-scan
# Morphologist + tiles run, which worker_timeout already bounds.
DEFAULT_CONSUMER_STALL_TIMEOUT = 7200
DEFAULT_DEVICE = "cuda"


def _cpu_count() -> int:
    """CPUs on this machine, read on every call (never cached)."""
    return os.cpu_count() or 1


@dataclass(frozen=True, kw_only=True)
class StreamingSettings:
    """Run-wide settings of a streaming run; durations are in seconds and must be greater than 0."""

    subjects_dir: str  # noqa: V107 - read by the streaming stages
    output_dir: str  # noqa: V107 - read by the streaming stages
    log_dir: str  # noqa: V107 - read by the streaming stages
    n_workers: int = 0
    poll_interval: float = DEFAULT_POLL_INTERVAL
    worker_timeout: float = DEFAULT_WORKER_TIMEOUT
    consumer_stall_timeout: float = DEFAULT_CONSUMER_STALL_TIMEOUT
    device: str = DEFAULT_DEVICE  # noqa: V107 - read by the consumer
    dry_run: bool = False

    def __post_init__(self) -> None:
        for name in ("poll_interval", "worker_timeout", "consumer_stall_timeout"):
            value = getattr(self, name)
            if not value > 0:
                raise ValueError(f"{name} must be greater than 0, got {value!r}")

    @property
    def producer_workers(self) -> int:
        """n_workers when set (> 0), else one producer worker per CPU."""
        return self.n_workers if self.n_workers > 0 else _cpu_count()

    @property
    def njobs_per_worker(self) -> int:
        """CPUs left to each producer worker, at least 1."""
        return max(1, _cpu_count() // self.producer_workers)
