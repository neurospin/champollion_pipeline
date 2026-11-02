#!/usr/bin/env python3
"""Checker of one finished e2e run directory: its facts and its hard failures.

Usage: ``python3 scripts/e2e/check_e2e.py <run dir>``. Writes ``<run dir>/report.json``
(per-stage exit code, subject counts, QC counts, failed subjects, region-side coverage,
hard failures) and prints each hard failure on stderr as ``check_e2e: ...``. Exit 0 when
there is no hard failure, 1 otherwise, 2 on a usage error (nothing written).
The run-directory layout is written by scripts/e2e/run_e2e.sh; E2E_RUN_PROFILE_DIRNAME
comes from e2e.env. Standard library only, so it runs without aims, on the host or in the SIF.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_env

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

STAGES_TSV = "stages.tsv"
STAGES_HEADER = ["stage", "exit_code"]
MORPHOLOGIST_STAGE = "generate_morphologist_graphs"
SNAPSHOTS_STAGE = "generate_snapshots"
MORPHOLOGIST_QC = Path("morphologist/derivatives/morphologist-6.0/input_qc.tsv")
CHAMPOLLION = Path("derivatives/champollion_V1")
TILES = Path("derivatives/cortical_tiles")
SNAPSHOTS_QC_GLOB = "*/snapshots/input_qc.tsv"
COMBINED_GLOB = "*/embeddings/*_embeddings.csv"
COMBINED_SUFFIX = "_embeddings.csv"


class Check:
    """Hard failures of one run, printed on stderr as they are found."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def fail(self, message: str) -> None:
        line = f"check_e2e: {message}"
        sys.stderr.write(line + "\n")
        self.failures.append(line)


def read_stages(run: Path, check: Check) -> list[tuple[str, int]]:
    """Rows of stages.tsv as (stage, exit_code), in file order; malformed rows are failures."""
    path = run / STAGES_TSV
    try:
        lines = path.read_bytes().splitlines()
    except OSError:
        check.fail(f"missing stages.tsv: {path}")
        return []
    rows = []
    for number, raw in enumerate(lines or [b""], start=1):
        try:
            fields = raw.decode().split("\t")
            if number == 1:
                if fields != STAGES_HEADER:
                    raise ValueError(raw)
                continue
            if len(fields) != len(STAGES_HEADER):
                raise ValueError(raw)
            rows.append((fields[0], int(fields[1])))
        except ValueError:  # UnicodeDecodeError included
            check.fail(f"malformed stages.tsv: {path}: line {number}")
    return rows


def read_dicts(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    """Rows of a CSV/TSV with a header; an unreadable file has no rows."""
    try:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


def distinct(paths: list[Path], column: str) -> int:
    """Number of distinct values of ``column`` over all ``paths``."""
    return len({row[column] for path in paths for row in read_dicts(path) if row.get(column) is not None})


def has_non_finite(path: Path) -> bool:
    """True when a data cell after the first column parses to NaN or +-inf."""
    try:
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle))
    except (OSError, csv.Error, UnicodeDecodeError):
        return False
    for row in rows[1:]:
        for cell in row[1:]:
            try:
                if not math.isfinite(float(cell)):
                    return True
            except ValueError:
                continue
    return False


def qc_facts(paths: list[Path]) -> dict:
    """QC counts and sorted failed subjects over the rows of ``paths``, concatenated."""
    rows = [row for path in paths for row in read_dicts(path, delimiter="\t")]
    qc = [(row.get("participant_id") or "", (row.get("qc") or "").strip()) for row in rows]
    return {
        "qc": {"pass": sum(value == "1" for _, value in qc), "fail": sum(value == "0" for _, value in qc)},
        "failed_subjects": sorted(subject for subject, value in qc if value == "0"),
    }


def sidecar_stages(proc: Path) -> set[str]:
    """``stage`` values of the profiling sidecars in ``proc``; unreadable files are skipped."""
    stages = set()
    for path in sorted(proc.glob("*.json")):
        try:
            sidecar = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(sidecar, dict) and isinstance(sidecar.get("stage"), str):
            stages.add(sidecar["stage"])
    return stages


def check_run(run: Path) -> dict:
    """Run every check on ``run`` (absolute) and return the report object."""
    check = Check()
    rows = read_stages(run, check)
    ran = [stage for stage, _ in rows]
    for stage, code in rows:
        if code != 0:
            check.fail(f"stage {stage} exited {code}")

    full = sorted((run / CHAMPOLLION).glob("**/full_embeddings.csv"))
    combined = sorted((run / CHAMPOLLION).glob(COMBINED_GLOB))
    for path in sorted(set(full) | set(combined)):
        if has_non_finite(path):
            check.fail(f"non-finite embedding value in {path}")

    morphologist_qc = run / MORPHOLOGIST_QC
    if MORPHOLOGIST_STAGE in ran and not morphologist_qc.is_file():
        check.fail(f"missing QC TSV for {MORPHOLOGIST_STAGE}: {morphologist_qc}")
    snapshots_qc = sorted((run / CHAMPOLLION).glob(SNAPSHOTS_QC_GLOB))
    if SNAPSHOTS_STAGE in ran and not snapshots_qc:
        check.fail(f"missing QC TSV for {SNAPSHOTS_STAGE}: {run / CHAMPOLLION / SNAPSHOTS_QC_GLOB}")

    try:
        proc = run / e2e_env.load_settings()["E2E_RUN_PROFILE_DIRNAME"] / "proc"
    except (OSError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        check.fail(f"cannot read e2e.env: {error}")
    else:
        profiled = sidecar_stages(proc)
        for stage in ran:
            if stage not in profiled:
                check.fail(f"no profile sidecar for stage {stage} in {proc}")

    subject_files = {
        "run_cortical_tiles": (sorted((run / TILES).glob("**/[LR]skeleton_subject.csv")), "Subject"),
        "generate_embeddings": (full, "ID"),
        "put_together_embeddings": (combined, "ID"),
    }
    stages: dict[str, dict] = {}
    for stage, code in rows:
        stages[stage] = {"exit_code": code}
        if stage in subject_files:
            stages[stage]["subjects"] = distinct(*subject_files[stage])
    qc_files = {
        MORPHOLOGIST_STAGE: [morphologist_qc] if morphologist_qc.is_file() else [],
        SNAPSHOTS_STAGE: snapshots_qc,
    }
    for stage, paths in qc_files.items():
        if paths:
            stages.setdefault(stage, {}).update(qc_facts(paths))

    return {
        "run_dir": str(run),
        "stages": stages,
        "region_sides": sorted(path.name.removesuffix(COMBINED_SUFFIX) for path in combined),
        "hard_failures": check.failures,
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or not Path(args[0]).is_dir():
        sys.stderr.write("usage: check_e2e.py <run dir>\n")
        return EXIT_USAGE
    run = Path(args[0]).resolve()
    report = check_run(run)
    (run / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return EXIT_FAILED if report["hard_failures"] else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
