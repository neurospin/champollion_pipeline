"""Synthetic profiling run directories for the profile_report tests (PAGG).

The layout and the file formats follow the cross-process-profiling design
(COMP-ARCH-BDRABCZUK-7B43010AD32E, "File layout and sidecar schema" and
Contract 7b)::

    <run_dir>/
      run.json                          # schema, run_id, root argv, mode, hosts, gpu_backend
      proc/<host>-<pid>-<seq>.json      # sidecar, schema 1
      proc/<host>-<pid>-<seq>.prof      # cProfile dump
      tree/<host>-<sampler_pid>.jsonl   # one line per (sample time, pid)
      tree/<host>-<sampler_pid>.gpu.jsonl  # GPU device/proc lines, only on GPU runs

The per-process profiling hook (PHOOK) that writes these files does not exist
yet, so every file is built here by hand; nothing from that hook is imported.
The ``.prof`` files are real cProfile dumps of small local functions, so
``pstats`` can load and merge them. Data is synthetic: the only subject ids are
public IXI ids.

Report document keys the tests rely on (chosen by the red stage, since the
design names only the GPU keys)::

    {"schema", "run_id", "group_by", "gpu_used",
     "groups": [{"group", "process_count", "wall_s", "cpu_s", "peak_rss_kb",
                 "incomplete_count", "top_functions": [{"function", "cumulative_s", "calls"}],
                 gpu_peak_mem_mb, gpu_util_mean_pct, gpu_util_max_pct, gpu_foreign_pids,
                 gpu_time_s, gpu_time_share   # GPU runs only
                }],
     "processes": [{"host", "pid", "argv", "group", "wall_s", "cpu_s", "peak_rss_kb",
                    "exit_path", "complete",
                    gpu_device_name, gpu_max_allocated_mb, gpu_max_reserved_mb, gpu_used_mb  # GPU runs only
                   }]}

A process entry is one sidecar segment, or one process seen only in the tree
samples.
"""

import cProfile
import importlib
import json
from pathlib import Path

MODULE_UNDER_TEST = "champollion_pipeline.profile_report"

RUN_ID = "run-20261028-synthetic"
HOST_A = "nodeA"
HOST_B = "nodeB"
SAMPLER_PID = 100

# Public IXI subject ids only (CC BY-SA 3.0 dataset).
SUBJECT_1 = "IXI002"
SUBJECT_2 = "IXI012"

GPU_NAME = "Tesla V100-SXM2-32GB"

# Number of distinct Python functions profiled in the "wide" segment, above the default top of 20.
WIDE_FUNCTION_COUNT = 25


def load_profile_report():
    """Import the module under test at call time, so a missing module fails each test, not collection."""
    return importlib.import_module(MODULE_UNDER_TEST)


# ---------------------------------------------------------------------------
# Profiled workloads (their names appear in the .prof files)
# ---------------------------------------------------------------------------


def morphologist_work():
    """Workload profiled in the morphologist segment."""
    return sum(i * i for i in range(2000))


def tiles_parent_work():
    """Workload profiled in the main process's cortical_tiles segment."""
    return sorted(range(3000), key=lambda i: -i)[0]


def tiles_worker_work():
    """Workload profiled in the forked cortical_tiles worker."""
    return max(i % 7 for i in range(3000))


def embeddings_work():
    """Workload profiled in the embeddings segment on the second host."""
    return len([i for i in range(4000) if i % 3 == 0])


def _wide_functions() -> list:
    """``WIDE_FUNCTION_COUNT`` distinct functions, each with its own code object and name."""
    namespace: dict = {}
    for index in range(WIDE_FUNCTION_COUNT):
        source = f"def wide_fn_{index:02d}():\n    return sum(range({(index + 1) * 200}))\n"
        exec(compile(source, f"<wide_fn_{index:02d}>", "exec"), namespace)
    return [namespace[f"wide_fn_{index:02d}"] for index in range(WIDE_FUNCTION_COUNT)]


def _write_prof(path: Path, workloads) -> None:
    """Profile ``workloads`` with cProfile and dump the stats to ``path``."""
    profiler = cProfile.Profile()
    profiler.enable()
    for workload in workloads:
        workload()
    profiler.disable()
    profiler.dump_stats(str(path))


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------


def _write_json(path: Path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def _segment_name(host: str, pid: int, seq: int) -> str:
    return f"{host}-{pid}-{seq}"


def _sidecar(segment: dict) -> dict:
    """A schema-1 sidecar for ``segment`` (keys: host, pid, ppid, seq, stage, t_start, t_end, ...)."""
    host, pid, seq = segment["host"], segment["pid"], segment["seq"]
    name = _segment_name(host, pid, seq)
    complete = segment.get("complete", True)
    t_end = segment.get("t_end")
    return {
        "schema": 1,
        "run_id": RUN_ID,
        "host": host,
        "pid": pid,
        "ppid": segment["ppid"],
        "seq": seq,
        "parent_segment": segment.get("parent_segment"),
        "origin": segment.get("origin", "exec"),
        "executable": "/opt/pixi/envs/default/bin/python3.11",
        "python_version": "3.11.0",
        "argv": segment["argv"],
        "cwd": "/tmp/run",
        "stage": segment["stage"],
        "tags": segment.get("tags", {}),
        "mode": "cpu",
        "t_start": segment["t_start"],
        "t_end": t_end,
        "wall_s": None if t_end is None else t_end - segment["t_start"],
        "cpu_user_s": segment.get("cpu_user_s"),
        "cpu_sys_s": segment.get("cpu_sys_s"),
        "children_cpu_user_s": segment.get("children_cpu_user_s", 0.0),
        "children_cpu_sys_s": segment.get("children_cpu_sys_s", 0.0),
        "maxrss_kb": segment.get("maxrss_kb"),
        "complete": complete,
        "exit_path": segment.get("exit_path", "segment-end" if complete else None),
        "profile_file": f"{name}.prof",
        "mem_file": None,
        "gpu": segment.get("gpu"),
    }


def _write_run(run_dir: Path, segments: list[dict], tree_lines: list[dict], gpu_backend: str) -> None:
    """Write run.json, one sidecar (+ .prof when workloads are given) per segment and the tree samples."""
    hosts = sorted({segment["host"] for segment in segments})
    _write_json(
        run_dir / "run.json",
        {
            "schema": 1,
            "run_id": RUN_ID,
            "argv": ["main.py", "--profile", str(run_dir.parent)],
            "t_start": 0.0,
            "t_end": 300.0,
            "mode": "cpu",
            "hosts": hosts,
            "gpu_backend": gpu_backend,
        },
    )
    for segment in segments:
        name = _segment_name(segment["host"], segment["pid"], segment["seq"])
        _write_json(run_dir / "proc" / f"{name}.json", _sidecar(segment))
        if segment.get("workloads"):
            _write_prof(run_dir / "proc" / f"{name}.prof", segment["workloads"])
    _write_jsonl(run_dir / "tree" / f"{HOST_A}-{SAMPLER_PID}.jsonl", tree_lines)


def _tree_line(t: float, pid: int, ppid: int, create_time: float, **fields) -> dict:
    return {
        "t": t,
        "host": HOST_A,
        "pid": pid,
        "ppid": ppid,
        "create_time": create_time,
        "name": fields.get("name", "python3.11"),
        "cmdline": fields.get("cmdline", ["python3.11"]),
        "cpu_user_s": fields["cpu_user_s"],
        "cpu_sys_s": fields["cpu_sys_s"],
        "rss_kb": fields["rss_kb"],
        "num_threads": 1,
        "status": "running",
    }


# ---------------------------------------------------------------------------
# CPU-only run
# ---------------------------------------------------------------------------

# Expected per-stage figures of the CPU run (see build_cpu_run for the inputs). Each value is the
# tuple process_count, wall_s, cpu_s, peak_rss_kb, incomplete_count.
EXPECTED_STAGES = {
    "morphologist": (2, 50.0, 18.0, 100_000, 0),
    "cortical_tiles": (4, 70.0, 61.0, 400_000, 0),
    "embeddings": (2, 75.0, 44.0, 300_000, 1),
}

# Function names profiled per stage, for the merged .prof checks.
EXPECTED_FUNCTIONS = {
    "morphologist": {"morphologist_work"},
    "cortical_tiles": {"tiles_parent_work", "tiles_worker_work"},
    "embeddings": {"embeddings_work"},
}

# Number of process entries: five sidecar segments plus three tree-only processes.
EXPECTED_PROCESS_COUNT = 8


def cpu_segments(run_dir: Path) -> list[dict]:
    """Sidecar segments of the CPU run.

    nodeA pid 100 is main.py with two segments (morphologist, then cortical_tiles);
    nodeA pid 101 is a forked cortical_tiles worker; nodeB pid 101 runs embeddings
    on another host with the same PID; nodeA pid 102 is an embeddings child that
    was killed (complete false, no end, no CPU, no .prof file).
    """
    main_argv = ["main.py", "--profile", str(run_dir.parent)]
    return [
        {
            "host": HOST_A,
            "pid": 100,
            "ppid": 1,
            "seq": 0,
            "stage": "morphologist",
            "argv": main_argv,
            "t_start": 0.0,
            "t_end": 50.0,
            "cpu_user_s": 10.0,
            "cpu_sys_s": 2.0,
            "maxrss_kb": 100_000,
            "workloads": [morphologist_work],
        },
        {
            "host": HOST_A,
            "pid": 100,
            "ppid": 1,
            "seq": 1,
            "stage": "cortical_tiles",
            "argv": main_argv,
            "parent_segment": _segment_name(HOST_A, 100, 0),
            "t_start": 50.0,
            "t_end": 120.0,
            "cpu_user_s": 30.0,
            "cpu_sys_s": 5.0,
            "maxrss_kb": 150_000,
            "workloads": [tiles_parent_work],
        },
        {
            "host": HOST_A,
            "pid": 101,
            "ppid": 100,
            "seq": 0,
            "stage": "cortical_tiles",
            "origin": "fork",
            "argv": main_argv,
            "parent_segment": _segment_name(HOST_A, 100, 1),
            "tags": {"subject": SUBJECT_1},
            "t_start": 60.0,
            "t_end": 110.0,
            "cpu_user_s": 20.0,
            "cpu_sys_s": 1.0,
            "maxrss_kb": 90_000,
            "exit_path": "mp-finalize",
            "workloads": [tiles_worker_work, *_wide_functions()],
        },
        {
            "host": HOST_B,
            "pid": 101,
            "ppid": 1,
            "seq": 0,
            "stage": "embeddings",
            "argv": ["python", "generate_embeddings.py", "--subject", SUBJECT_1],
            "tags": {"subject": SUBJECT_1},
            "t_start": 130.0,
            "t_end": 200.0,
            "cpu_user_s": 40.0,
            "cpu_sys_s": 4.0,
            "maxrss_kb": 300_000,
            "exit_path": "atexit",
            "workloads": [embeddings_work],
        },
        {
            "host": HOST_A,
            "pid": 102,
            "ppid": 100,
            "seq": 0,
            "stage": "embeddings",
            "argv": ["python", "generate_embeddings.py", "--subject", SUBJECT_2],
            "parent_segment": _segment_name(HOST_A, 100, 1),
            "tags": {"subject": SUBJECT_2},
            "t_start": 125.0,
            "t_end": None,
            "complete": False,
        },
    ]


def cpu_tree_lines() -> list[dict]:
    """Tree samples of the CPU run (sampler on nodeA).

    pid 100 and 101 have sidecars: their samples must not be counted twice.
    pid 200 (AimsThreshold, child of 101, created at 70) and pid 201 (sh, child
    of 200, created at 75) belong to the cortical_tiles worker segment; pid 202
    (morphologist-cli, child of 100, created at 20) belongs to the morphologist
    segment of pid 100, not to its later cortical_tiles segment.
    """
    aims = {"name": "AimsThreshold", "cmdline": ["AimsThreshold", "-i", "in.nii.gz"]}
    shell = {"name": "sh", "cmdline": ["sh", "-c", "true"]}
    morpho = {"name": "morphologist-cli", "cmdline": ["morphologist-cli", "t1.nii.gz"]}
    return [
        _tree_line(30.0, 100, 1, 0.0, cpu_user_s=42.0, cpu_sys_s=8.0, rss_kb=99_000),
        _tree_line(30.0, 202, 100, 20.0, cpu_user_s=2.0, cpu_sys_s=0.5, rss_kb=50_000, **morpho),
        _tree_line(40.0, 202, 100, 20.0, cpu_user_s=5.0, cpu_sys_s=1.0, rss_kb=70_000, **morpho),
        _tree_line(80.0, 101, 100, 60.0, cpu_user_s=18.0, cpu_sys_s=1.0, rss_kb=80_000),
        _tree_line(80.0, 200, 101, 70.0, cpu_user_s=1.0, cpu_sys_s=0.5, rss_kb=50_000, **aims),
        _tree_line(80.0, 201, 200, 75.0, cpu_user_s=0.5, cpu_sys_s=0.5, rss_kb=1_000, **shell),
        _tree_line(90.0, 200, 101, 70.0, cpu_user_s=3.0, cpu_sys_s=1.0, rss_kb=400_000, **aims),
        _tree_line(90.0, 100, 1, 0.0, cpu_user_s=45.0, cpu_sys_s=9.0, rss_kb=98_000),
    ]


def build_cpu_run(base: Path) -> Path:
    """Build the CPU-only run directory under ``base`` and return it (no GPU data at all)."""
    run_dir = base / "profile" / RUN_ID
    _write_run(run_dir, cpu_segments(run_dir), cpu_tree_lines(), gpu_backend="none")
    return run_dir


# ---------------------------------------------------------------------------
# GPU run
# ---------------------------------------------------------------------------

# Expected GPU figures per stage (nvml backend). Each value is the tuple
# gpu_peak_mem_mb, gpu_util_mean_pct, gpu_util_max_pct, gpu_foreign_pids.
EXPECTED_GPU = {
    "embeddings": (1500.0, 80.0, 100.0, False),
    "snapshots": (800.0, 20.0, 30.0, True),
}

# GPU time bounds per stage: three samples two seconds apart, at 75 % and 25 % SM utilisation.
# Two to three intervals of 2 s depending on how the first or last sample is weighted.
EXPECTED_GPU_TIME_BOUNDS = {"embeddings": (3.0, 4.5), "snapshots": (1.0, 1.5)}
EXPECTED_GPU_TIME_SHARE = {"embeddings": 0.75, "snapshots": 0.25}


def _gpu_lines(backend: str) -> list[dict]:
    """Device and proc lines: pid 300 (embeddings) at t=100..104, pid 400 (snapshots) at t=200..204.

    A device line at t=150 has no tracked process on the device: it counts for no group.
    """
    sm_util = {300: 75.0, 400: 25.0} if backend == "nvml" else {300: None, 400: None}
    device_util = {100.0: 60.0, 102.0: 80.0, 104.0: 100.0, 150.0: 5.0, 200.0: 10.0, 202.0: 20.0, 204.0: 30.0}
    foreign = {202.0}
    lines = []
    for t, util in device_util.items():
        lines.append(
            {
                "kind": "device",
                "t": t,
                "host": HOST_A,
                "backend": backend,
                "gpu_index": 0,
                "uuid": "GPU-00000000-0000-0000-0000-000000000000",
                "name": GPU_NAME,
                "util_pct": util,
                "mem_used_mb": 4000.0,
                "mem_total_mb": 32768.0,
                "foreign_pids": t in foreign,
            }
        )
    for pid, times, used in ((300, (100.0, 102.0, 104.0), 1200.0), (400, (200.0, 202.0, 204.0), 800.0)):
        for t in times:
            lines.append(
                {
                    "kind": "proc",
                    "t": t,
                    "host": HOST_A,
                    "backend": backend,
                    "pid": pid,
                    "gpu_index": 0,
                    "used_mb": used if t == times[1] else used / 2,
                    "sm_util_pct": sm_util[pid],
                }
            )
    return lines


def build_gpu_run(base: Path, backend: str = "nvml") -> Path:
    """Build a run with GPU data: pid 300 (embeddings, torch peaks) and pid 400 (snapshots, sampler only)."""
    run_dir = base / "profile" / RUN_ID
    torch_peaks = {
        "source": "torch",
        "devices": [{"index": 0, "name": GPU_NAME, "max_allocated_mb": 1000.0, "max_reserved_mb": 1500.0}],
    }
    segments = [
        {
            "host": HOST_A,
            "pid": 300,
            "ppid": 1,
            "seq": 0,
            "stage": "embeddings",
            "argv": ["python", "evaluate.py"],
            "t_start": 90.0,
            "t_end": 110.0,
            "cpu_user_s": 15.0,
            "cpu_sys_s": 1.0,
            "maxrss_kb": 200_000,
            "gpu": torch_peaks,
            "workloads": [embeddings_work],
        },
        {
            "host": HOST_A,
            "pid": 400,
            "ppid": 1,
            "seq": 0,
            "stage": "snapshots",
            "argv": ["python", "generate_snapshots.py"],
            "t_start": 190.0,
            "t_end": 210.0,
            "cpu_user_s": 5.0,
            "cpu_sys_s": 1.0,
            "maxrss_kb": 120_000,
            "workloads": [morphologist_work],
        },
    ]
    tree = [
        _tree_line(100.0, 300, 1, 90.0, cpu_user_s=5.0, cpu_sys_s=0.5, rss_kb=150_000),
        _tree_line(200.0, 400, 1, 190.0, cpu_user_s=2.0, cpu_sys_s=0.5, rss_kb=100_000),
    ]
    _write_run(run_dir, segments, tree, gpu_backend=backend)
    _write_jsonl(run_dir / "tree" / f"{HOST_A}-{SAMPLER_PID}.gpu.jsonl", _gpu_lines(backend))
    return run_dir


# ---------------------------------------------------------------------------
# Report lookups
# ---------------------------------------------------------------------------


def group_rows(report: dict) -> dict:
    """Group rows of ``report`` keyed by group name (fails on a duplicate name)."""
    rows = {}
    for row in report["groups"]:
        assert row["group"] not in rows, f"duplicate group row {row['group']!r}"
        rows[row["group"]] = row
    return rows


def find_process(report: dict, host: str, pid: int) -> list[dict]:
    """Process entries of ``report`` for (host, pid)."""
    return [entry for entry in report["processes"] if entry["host"] == host and entry["pid"] == pid]
