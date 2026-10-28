#!/usr/bin/env python3
"""
Script to aggregate a cross-process profiling run directory into a report.

Reads the sidecars, cProfile dumps and tree samples that ``--profile`` writes
under ``<run_dir>`` and produces ``report.json`` plus one ``merged-<group>.prof``
per group. All figures come from the pure function ``compute_profile_report``.
"""

import json
import logging
import pstats
import re
from pathlib import Path

from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.process_setup import init_pipeline_process

logger = logging.getLogger(__name__)

REPORT_SCHEMA = 1
REPORT_FILE = "report.json"
UNTAGGED = "(untagged)"
# Sampler poll interval when profiling is on; GPU polls run at max(interval, 2 s).
DEFAULT_INTERVAL_S = 1.0
GPU_DT_CAP_S = max(DEFAULT_INTERVAL_S, 2.0)
NVIDIA_SMI_BACKEND = "nvidia-smi"
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._()-]")


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _read_json(path: Path):
    """JSON document of ``path``, or None when it is unreadable or malformed."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        logger.warning(f"profile_report: skipping {path}: {error}")
        return None


def _read_jsonl(path: Path) -> list[dict]:
    """Object lines of ``path``; malformed lines are skipped with a warning."""
    lines = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        logger.warning(f"profile_report: skipping {path}: {error}")
        return lines
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            document = json.loads(line)
        except ValueError:
            logger.warning(f"profile_report: {path}:{number}: malformed line skipped")
            continue
        if isinstance(document, dict):
            lines.append(document)
    return lines


def _read_sidecars(proc_dir: Path) -> list[dict]:
    """Sidecar documents of ``proc_dir`` that carry a host, pid and seq."""
    sidecars = []
    for path in sorted(proc_dir.glob("*.json")):
        sidecar = _read_json(path)
        if not isinstance(sidecar, dict) or not {"host", "pid", "seq"} <= set(sidecar):
            continue
        profile_file = sidecar.get("profile_file")
        sidecar["_profile_path"] = proc_dir / Path(profile_file).name if profile_file else None
        sidecars.append(sidecar)
    return sidecars


def _read_tree(tree_dir: Path) -> tuple[list[dict], list[dict]]:
    """Process samples and GPU lines of ``tree_dir`` (both empty when it is absent)."""
    samples, gpu_lines = [], []
    if not tree_dir.is_dir():
        return samples, gpu_lines
    for path in sorted(tree_dir.glob("*.jsonl")):
        if path.name.endswith(".gpu.jsonl"):
            gpu_lines.extend(_read_jsonl(path))
        else:
            samples.extend(_read_jsonl(path))
    return samples, gpu_lines


# ---------------------------------------------------------------------------
# Process entries
# ---------------------------------------------------------------------------


def _cpu_of(record: dict):
    """user + system CPU seconds of a sidecar or a tree sample, None when either is missing."""
    user, system = record.get("cpu_user_s"), record.get("cpu_sys_s")
    if user is None or system is None:
        return None
    return user + system


def _segment_entry(sidecar: dict, last_sample: dict | None) -> dict:
    """Process entry of one sidecar segment (null CPU/RSS fall back to the last tree sample)."""
    t_start, t_end = sidecar.get("t_start"), sidecar.get("t_end")
    cpu_s = _cpu_of(sidecar)
    if cpu_s is None:
        cpu_s = _cpu_of(last_sample) if last_sample else None
    peak_rss_kb = sidecar.get("maxrss_kb")
    if peak_rss_kb is None and last_sample:
        peak_rss_kb = last_sample.get("rss_kb")
    return {
        "host": sidecar["host"],
        "pid": sidecar["pid"],
        "argv": sidecar.get("argv") or [],
        "group": None,
        "wall_s": None if t_start is None or t_end is None else t_end - t_start,
        "cpu_s": cpu_s or 0.0,
        "peak_rss_kb": peak_rss_kb,
        "exit_path": sidecar.get("exit_path"),
        "complete": bool(sidecar.get("complete", False)),
        "_seq": sidecar["seq"],
        "_t_start": t_start,
        "_t_end": t_end,
        "_stage": sidecar.get("stage"),
        "_tags": sidecar.get("tags") or {},
        "_prof": sidecar["_profile_path"],
        "_gpu": sidecar.get("gpu"),
    }


def _tree_only_entry(samples: list[dict]) -> dict:
    """Process entry of a process seen only in tree samples (time-ordered ``samples``)."""
    last = samples[-1]
    create_time = last.get("create_time")
    rss = [sample["rss_kb"] for sample in samples if sample.get("rss_kb") is not None]
    return {
        "host": last["host"],
        "pid": last["pid"],
        "argv": last.get("cmdline") or [],
        "group": None,
        "wall_s": None if create_time is None else last["t"] - create_time,
        "cpu_s": _cpu_of(last) or 0.0,
        "peak_rss_kb": max(rss) if rss else None,
        "exit_path": None,
        "complete": True,
        "_seq": None,
        "_t_start": create_time,
        "_t_end": None,
        "_ppid": last.get("ppid"),
        "_create_time": create_time,
        "_stage": None,
        "_tags": {},
        "_prof": None,
        "_gpu": None,
    }


def _contains(segment: dict, t) -> bool:
    """True when ``t`` lies in the segment's [t_start, t_end] (an open end holds any later time)."""
    if t is None or segment["_t_start"] is None or t < segment["_t_start"]:
        return False
    return segment["_t_end"] is None or t <= segment["_t_end"]


def _segment_at(segments: list[dict], t) -> dict | None:
    """Segment of one (host, pid) whose interval holds ``t``, the latest one on overlap."""
    holding = [segment for segment in segments if _contains(segment, t)]
    return holding[-1] if holding else None


def _attribute_tree_only(entry: dict, segments_by_pid: dict, tree_only_by_pid: dict) -> dict | None:
    """Nearest ancestor segment whose interval holds the entry's create time, through tree-only ancestors."""
    host, ppid, seen = entry["host"], entry["_ppid"], set()
    while ppid is not None and (host, ppid) not in seen:
        seen.add((host, ppid))
        segment = _segment_at(segments_by_pid.get((host, ppid), []), entry["_create_time"])
        if segment is not None:
            return segment
        parent = tree_only_by_pid.get((host, ppid))
        ppid = parent["_ppid"] if parent else None
    return None


def _build_entries(sidecars: list[dict], samples: list[dict]) -> list[dict]:
    """Segment entries plus attributed tree-only entries; stage and tags are inherited by tree-only ones."""
    samples_by_pid: dict = {}
    for sample in sorted(samples, key=lambda sample: sample.get("t", 0.0)):
        if "host" in sample and "pid" in sample and "t" in sample:
            samples_by_pid.setdefault((sample["host"], sample["pid"]), []).append(sample)

    segments_by_pid: dict = {}
    for sidecar in sorted(sidecars, key=lambda sidecar: (sidecar["host"], sidecar["pid"], sidecar["seq"])):
        key = (sidecar["host"], sidecar["pid"])
        last_sample = samples_by_pid.get(key, [None])[-1]
        segments_by_pid.setdefault(key, []).append(_segment_entry(sidecar, last_sample))

    tree_only_by_pid = {
        key: _tree_only_entry(pid_samples) for key, pid_samples in samples_by_pid.items() if key not in segments_by_pid
    }
    entries = [segment for segments in segments_by_pid.values() for segment in segments]
    for entry in tree_only_by_pid.values():
        owner = _attribute_tree_only(entry, segments_by_pid, tree_only_by_pid)
        if owner is None:
            logger.warning(f"profile_report: {entry['host']} pid {entry['pid']} has no ancestor segment; skipped")
            continue
        entry["_stage"], entry["_tags"] = owner["_stage"], owner["_tags"]
        entries.append(entry)
    entries.sort(key=lambda entry: (entry["_t_start"] or 0.0, entry["host"], entry["pid"], entry["_seq"] or 0))
    return entries


def _group_name(entry: dict, group_by: str) -> str:
    """Name of the group ``entry`` belongs to under ``group_by``."""
    if group_by == "process":
        if entry["_seq"] is None:
            return f"{entry['host']}-{entry['pid']}"
        return f"{entry['host']}-{entry['pid']}-{entry['_seq']}"
    if group_by.startswith("tag:"):
        value = entry["_tags"].get(group_by[len("tag:") :])
        return UNTAGGED if value is None else str(value)
    return str(entry["_stage"])


# ---------------------------------------------------------------------------
# Top functions
# ---------------------------------------------------------------------------


def _load_stats(prof_paths: list[Path]) -> pstats.Stats | None:
    """Merged stats of the dumps that exist and load; None when there is none."""
    stats = None
    for path in prof_paths:
        if not path.is_file():
            continue
        try:
            if stats is None:
                stats = pstats.Stats(str(path))
            else:
                stats.add(str(path))
        except (OSError, EOFError, ValueError, TypeError) as error:
            logger.warning(f"profile_report: skipping dump {path}: {error}")
    return stats


def _top_functions(stats: pstats.Stats | None, top: int) -> list[dict]:
    """At most ``top`` functions of ``stats`` by decreasing cumulative time."""
    if stats is None:
        return []
    rows = [
        {"function": pstats.func_std_string(func), "cumulative_s": ct, "calls": nc}
        for func, (_cc, nc, _tt, ct, _callers) in stats.stats.items()
    ]
    rows.sort(key=lambda row: (-row["cumulative_s"], row["function"]))
    return rows[: max(top, 0)]


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


def _assign_gpu_samples(entries: list[dict], gpu_lines: list[dict]) -> None:
    """Attach each ``proc`` GPU line to the process entry of its (host, pid) at its time."""
    by_pid: dict = {}
    for entry in entries:
        entry["_gpu_samples"] = []
        by_pid.setdefault((entry["host"], entry["pid"]), []).append(entry)
    for line in sorted(gpu_lines, key=lambda line: line.get("t", 0.0)):
        if line.get("kind") != "proc":
            continue
        candidates = by_pid.get((line.get("host"), line.get("pid")), [])
        owner = _segment_at(candidates, line.get("t")) if len(candidates) > 1 else (candidates or [None])[0]
        if owner is not None:
            owner["_gpu_samples"].append(line)


def _sample_gpu_time(samples: list[dict]):
    """sum sm_util_pct / 100 * dt over time-ordered samples of one process, None without SM figures."""
    total, previous_t = 0.0, None
    for sample in samples:
        sm = sample.get("sm_util_pct")
        if sample.get("backend") == NVIDIA_SMI_BACKEND or sm is None:
            return None
        dt = GPU_DT_CAP_S if previous_t is None else min(sample["t"] - previous_t, GPU_DT_CAP_S)
        total += sm / 100.0 * dt
        previous_t = sample["t"]
    return total


def _device_name(entry: dict, device_lines: list[dict]) -> str | None:
    """Device name from the torch peaks, else from the sampler's device lines."""
    for device in (entry["_gpu"] or {}).get("devices") or []:
        if device.get("name"):
            return device["name"]
    for sample in entry["_gpu_samples"]:
        for line in device_lines:
            if line.get("host") == sample.get("host") and line.get("gpu_index") == sample.get("gpu_index"):
                return line.get("name")
    return None


def _process_gpu_fields(entry: dict, device_lines: list[dict]) -> dict:
    """GPU fields of a process entry with GPU data (torch peaks and/or sampler lines)."""
    devices = (entry["_gpu"] or {}).get("devices") or []
    allocated = [device["max_allocated_mb"] for device in devices if device.get("max_allocated_mb") is not None]
    reserved = [device["max_reserved_mb"] for device in devices if device.get("max_reserved_mb") is not None]
    used = [sample["used_mb"] for sample in entry["_gpu_samples"] if sample.get("used_mb") is not None]
    return {
        "gpu_device_name": _device_name(entry, device_lines),
        "gpu_max_allocated_mb": max(allocated) if allocated else None,
        "gpu_max_reserved_mb": max(reserved) if reserved else None,
        "gpu_used_mb": max(used) if used else None,
    }


def _group_gpu_fields(members: list[dict], device_lines: list[dict]) -> dict:
    """GPU fields of one group row (gpu_time_share is filled once every row is known)."""
    peaks = []
    for entry in members:
        fields = entry.get("_gpu_fields") or {}
        peaks += [fields[key] for key in ("gpu_max_reserved_mb", "gpu_used_mb") if fields.get(key) is not None]
    on_device = {
        (sample.get("t"), sample.get("host"), sample.get("gpu_index"))
        for entry in members
        for sample in entry["_gpu_samples"]
    }
    counted = [line for line in device_lines if (line.get("t"), line.get("host"), line.get("gpu_index")) in on_device]
    utils = [line["util_pct"] for line in counted if line.get("util_pct") is not None]
    with_samples = [entry for entry in members if entry["_gpu_samples"]]
    times = [_sample_gpu_time(entry["_gpu_samples"]) for entry in with_samples]
    return {
        "gpu_peak_mem_mb": max(peaks) if peaks else None,
        "gpu_util_mean_pct": sum(utils) / len(utils) if utils else None,
        "gpu_util_max_pct": max(utils) if utils else None,
        "gpu_foreign_pids": any(bool(line.get("foreign_pids")) for line in counted),
        "gpu_time_s": sum(times) if times and None not in times else None,
        "gpu_time_share": None,
    }


def _fill_gpu_time_share(groups: list[dict]) -> None:
    """gpu_time_share = group gpu_time_s / total over the rows (None when the total is unknown or zero)."""
    total = sum(row["gpu_time_s"] for row in groups if row["gpu_time_s"] is not None)
    for row in groups:
        if row["gpu_time_s"] is not None and total > 0:
            row["gpu_time_share"] = row["gpu_time_s"] / total


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _public(entry: dict) -> dict:
    """Process entry without its private (underscore) keys, GPU fields included when present."""
    document = {key: value for key, value in entry.items() if not key.startswith("_")}
    document.update(entry.get("_gpu_fields") or {})
    return document


def _run_id(run_dir: Path, sidecars: list[dict]) -> str | None:
    """run_id from run.json, else from the first sidecar that names one."""
    run_json = run_dir / "run.json"
    if run_json.is_file():
        document = _read_json(run_json)
        if isinstance(document, dict) and document.get("run_id"):
            return document["run_id"]
    return next((sidecar["run_id"] for sidecar in sidecars if sidecar.get("run_id")), None)


def _build_report(run_dir: Path, group_by: str, top: int) -> tuple[dict, dict]:
    """The report document and, per group, the paths of its cProfile dumps."""
    run_dir = Path(run_dir)
    sidecars = _read_sidecars(run_dir / "proc")
    samples, gpu_lines = _read_tree(run_dir / "tree")
    entries = _build_entries(sidecars, samples)
    device_lines = [line for line in gpu_lines if line.get("kind") == "device"]
    _assign_gpu_samples(entries, gpu_lines)
    gpu_used = bool(gpu_lines) or any(entry["_gpu"] for entry in entries)

    members_by_group: dict = {}
    for entry in entries:
        entry["group"] = _group_name(entry, group_by)
        if gpu_used and (entry["_gpu"] or entry["_gpu_samples"]):
            entry["_gpu_fields"] = _process_gpu_fields(entry, device_lines)
        members_by_group.setdefault(entry["group"], []).append(entry)

    groups, prof_paths = [], {}
    for name, members in members_by_group.items():
        segments = [entry for entry in members if entry["_seq"] is not None]
        starts = [entry["_t_start"] for entry in segments if entry["_t_start"] is not None]
        ends = [entry["_t_end"] for entry in segments if entry["_t_end"] is not None]
        rss = [entry["peak_rss_kb"] for entry in members if entry["peak_rss_kb"] is not None]
        prof_paths[name] = [entry["_prof"] for entry in members if entry["_prof"] is not None]
        row = {
            "group": name,
            "process_count": len(members),
            "wall_s": max(ends) - min(starts) if starts and ends else 0.0,
            "cpu_s": sum(entry["cpu_s"] for entry in members),
            "peak_rss_kb": max(rss) if rss else None,
            "incomplete_count": sum(1 for entry in members if not entry["complete"]),
            "top_functions": _top_functions(_load_stats(prof_paths[name]), top),
        }
        if gpu_used:
            row.update(_group_gpu_fields(members, device_lines))
        groups.append(row)
    if gpu_used:
        _fill_gpu_time_share(groups)

    document = {
        "schema": REPORT_SCHEMA,
        "run_id": _run_id(run_dir, sidecars),
        "group_by": group_by,
        "gpu_used": gpu_used,
        "groups": groups,
        "processes": [_public(entry) for entry in entries],
    }
    return document, prof_paths


def compute_profile_report(run_dir: Path, group_by: str, top: int) -> dict:
    """Report document of the profiling run in ``run_dir`` (reads files only, writes nothing)."""
    return _build_report(Path(run_dir), group_by, top)[0]


def _format_figure(value, unit: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.1f}{unit}"
    return f"{value}{unit}"


def _text_line(row: dict, gpu_used: bool) -> str:
    """One text line for a group row, starting with the group name."""
    line = (
        f"{row['group']}: processes={row['process_count']} wall={_format_figure(row['wall_s'], 's')}"
        f" cpu={_format_figure(row['cpu_s'], 's')} peak_rss={_format_figure(row['peak_rss_kb'], 'kB')}"
        f" incomplete={row['incomplete_count']}"
    )
    if gpu_used:
        line += (
            f" gpu_peak_mem={_format_figure(row['gpu_peak_mem_mb'], 'MB')}"
            f" gpu_util_mean={_format_figure(row['gpu_util_mean_pct'], '%')}"
            f" gpu_time={_format_figure(row['gpu_time_s'], 's')}"
        )
        if row["gpu_foreign_pids"]:
            line += " (GPU shared with other users)"
    return line


class ProfileReport(ScriptBuilder):
    """Aggregate a --profile run directory into report.json and merged .prof files."""

    def __init__(self):
        super().__init__(
            script_name="profile_report",
            description="Aggregate a cross-process profiling run directory into report.json and merged .prof files.",
        )
        (
            self.add_argument("run_dir", type=str, help="Profiling run directory (holds proc/ and tree/).")
            .add_optional_argument(
                "--group-by",
                "Grouping: stage, process or tag:<key>.",
                default="stage",
            )
            .add_optional_argument("--top", "Number of functions listed per group.", default=20, type_=int)
            .add_argument(
                "--format",
                choices=("text", "json"),
                default="text",
                help="Standard output form: text lines or the report.json document.",
            )
            .add_optional_argument("--output", "Output directory (default: <run_dir>/report).")
        )

    def run(self):
        """Write report.json and merged-<group>.prof files, then print the report."""
        run_dir = Path(self.args.run_dir)
        group_by = self.args.group_by
        if group_by not in ("stage", "process") and not (group_by.startswith("tag:") and len(group_by) > 4):
            logger.error(f"--group-by must be stage, process or tag:<key>, got {group_by!r}")
            return 1
        if not (run_dir / "proc").is_dir():
            logger.error(f"{run_dir} is not a profiling run directory: {run_dir / 'proc'} is missing")
            return 1
        output = Path(self.args.output) if self.args.output else run_dir / "report"
        logger.trace(f"profile_report/run_dir: {run_dir}")
        logger.trace(f"profile_report/output: {output}")

        document, prof_paths = _build_report(run_dir, group_by, self.args.top)
        output.mkdir(parents=True, exist_ok=True)
        (output / REPORT_FILE).write_text(json.dumps(document, indent=2), encoding="utf-8")
        for name, paths in prof_paths.items():
            stats = _load_stats(paths)
            if stats is not None:
                stats.dump_stats(str(output / f"merged-{_UNSAFE_NAME_CHARS.sub('_', name)}.prof"))

        if self.args.format == "json":
            print(json.dumps(document, indent=2))
            return 0
        for row in document["groups"]:
            logger.output(_text_line(row, document["gpu_used"]))
        if not document["gpu_used"]:
            logger.output("no GPU used")
        logger.output(f"report written to {output / REPORT_FILE}")
        return 0


def main():
    """Main entry point."""
    init_pipeline_process()
    script = ProfileReport()
    return script.main()


if __name__ == "__main__":
    exit(main())
