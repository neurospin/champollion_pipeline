"""Red tests for the GPU columns of compute_profile_report (PAGG, design Contract 7c).

Task TASK-TASK-BDRABCZUK-F20A7E4D5472; design COMP-ARCH-BDRABCZUK-7B43010AD32E.
No GPU is needed: the sidecar ``gpu`` blocks and ``tree/*.gpu.jsonl`` lines are
synthetic (tests/profile_run_fixtures.py), one run per sampler backend.
"""

import pytest

from tests.profile_run_fixtures import (
    EXPECTED_GPU,
    EXPECTED_GPU_TIME_BOUNDS,
    EXPECTED_GPU_TIME_SHARE,
    GPU_NAME,
    HOST_A,
    build_cpu_run,
    build_gpu_run,
    find_process,
    group_rows,
    load_profile_report,
)


def _compute(run_dir) -> dict:
    return load_profile_report().compute_profile_report(run_dir, "stage", 20)


@pytest.fixture
def nvml_run(tmp_path):
    """GPU run sampled by the nvml backend (per-pid SM utilisation present)."""
    return build_gpu_run(tmp_path, "nvml")


@pytest.fixture
def smi_run(tmp_path):
    """GPU run sampled by the nvidia-smi backend (no per-pid SM utilisation)."""
    return build_gpu_run(tmp_path, "nvidia-smi")


def test_gpu_peak_memory_is_largest_torch_reserved_or_sampler_used(nvml_run):
    """REQ: gpu_peak_mem_mb = largest torch max_reserved_mb or sampler used_mb among the group's processes."""
    rows = group_rows(_compute(nvml_run))
    for stage, (peak, _mean, _max, _foreign) in EXPECTED_GPU.items():
        assert rows[stage]["gpu_peak_mem_mb"] == pytest.approx(peak), stage


def test_gpu_utilisation_counts_samples_with_a_group_process_on_the_device(nvml_run):
    """REQ: gpu_util_mean_pct / gpu_util_max_pct from device samples taken while a group process was on it."""
    rows = group_rows(_compute(nvml_run))
    for stage, (_peak, mean, maximum, _foreign) in EXPECTED_GPU.items():
        assert rows[stage]["gpu_util_mean_pct"] == pytest.approx(mean), stage
        assert rows[stage]["gpu_util_max_pct"] == pytest.approx(maximum), stage


def test_gpu_foreign_pids_flag_follows_counted_device_samples(nvml_run):
    """REQ: gpu_foreign_pids is true when a counted device sample has foreign_pids true."""
    rows = group_rows(_compute(nvml_run))
    for stage, (_peak, _mean, _max, foreign) in EXPECTED_GPU.items():
        assert rows[stage]["gpu_foreign_pids"] is foreign, stage


def test_gpu_time_from_per_pid_sm_utilisation(nvml_run):
    """REQ: gpu_time_s = sum over the group's GPU process samples of sm_util_pct / 100 times the interval."""
    rows = group_rows(_compute(nvml_run))
    for stage, (low, high) in EXPECTED_GPU_TIME_BOUNDS.items():
        assert low <= rows[stage]["gpu_time_s"] <= high, (stage, rows[stage]["gpu_time_s"])


def test_gpu_time_is_null_on_nvidia_smi_backend(smi_run):
    """REQ: gpu_time_s is null when the run's GPU samples come from the nvidia-smi backend."""
    rows = group_rows(_compute(smi_run))
    for stage in EXPECTED_GPU:
        assert "gpu_time_s" in rows[stage], stage
        assert rows[stage]["gpu_time_s"] is None, stage


def test_gpu_time_share_is_group_time_over_run_total(nvml_run):
    """REQ: gpu_time_share = group gpu_time_s / sum of gpu_time_s over the run's group rows."""
    rows = group_rows(_compute(nvml_run))
    for stage, share in EXPECTED_GPU_TIME_SHARE.items():
        assert rows[stage]["gpu_time_share"] == pytest.approx(share), stage
    assert sum(rows[stage]["gpu_time_share"] for stage in EXPECTED_GPU_TIME_SHARE) == pytest.approx(1.0)


def test_no_gpu_data_gives_gpu_used_false_and_no_gpu_fields(tmp_path):
    """REQ: with no GPU data, gpu_used is false and no group row carries a gpu_ prefixed field."""
    report = _compute(build_cpu_run(tmp_path))
    assert report["gpu_used"] is False
    for row in report["groups"]:
        assert not [key for key in row if key.startswith("gpu_")], row["group"]


def test_gpu_data_sets_gpu_used_true(nvml_run):
    """Counterpart of the no-GPU case: a run with GPU data reports gpu_used true."""
    assert _compute(nvml_run)["gpu_used"] is True


def test_process_entry_gpu_fields(nvml_run):
    """REQ: a process with GPU data gets device name, torch peak allocated/reserved MB and sampler peak used MB."""
    report = _compute(nvml_run)
    (torch_process,) = find_process(report, HOST_A, 300)
    assert torch_process["gpu_device_name"] == GPU_NAME
    assert torch_process["gpu_max_allocated_mb"] == pytest.approx(1000.0)
    assert torch_process["gpu_max_reserved_mb"] == pytest.approx(1500.0)
    assert torch_process["gpu_used_mb"] == pytest.approx(1200.0)
    (sampled_process,) = find_process(report, HOST_A, 400)
    assert sampled_process["gpu_device_name"] == GPU_NAME
    assert sampled_process["gpu_used_mb"] == pytest.approx(800.0)
