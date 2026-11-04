#!/usr/bin/env python3
"""Checker of one finished e2e run directory: its facts and its hard failures.

Usage: ``python3 scripts/e2e/check_e2e.py <run dir> [--tier T --layout L [--baselines-dir DIR]
[--accept-baseline]]``. Writes ``<run dir>/report.json`` (per-stage exit code, subject counts,
QC counts, failed subjects, region-side coverage, partial stages, hard failures) and prints
each hard failure on stderr as ``check_e2e: ...``. With ``--tier``/``--layout`` it also compares
those facts with the shared baseline ``<E2E_BASELINES_DIR>/<tier>-<layout>.json``; any difference
is a hard failure. Only ``--accept-baseline`` on a run with no hard failure and no partial stage
writes that baseline. Exit 0 on a clean run, 3 on a partial stage without hard failure, 1 on a
hard failure, 2 on a usage error (nothing written).
The run-directory layout is written by scripts/e2e/run_e2e.sh; E2E_RUN_PROFILE_DIRNAME
comes from e2e.env. Standard library only, so it runs without aims, on the host or in the SIF.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_env

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_PARTIAL = 3

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
USAGE = "usage: check_e2e.py <run dir> [--tier <tier> --layout <layout> [--baselines-dir <dir>] [--accept-baseline]]"
ENV_ERRORS = (OSError, KeyError, RuntimeError, subprocess.SubprocessError)
BASELINE_FACTS = ("exit_code", "subjects", "qc", "failed_subjects")
_ABSENT = object()  # a key missing on one side of the baseline comparison


class UsageError(Exception):
    """A command-line error: one usage line, exit 2, nothing written."""


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


def partial_stages(rows: list[tuple[str, int]]) -> list[str]:
    """Stages of the rows that exited EXIT_PARTIAL, in file order."""
    return [stage for stage, code in rows if code == EXIT_PARTIAL]


def exit_status(report: dict) -> int:
    """Exit code of a finished report; names the partial stages on stderr when the run exits EXIT_PARTIAL."""
    if report["hard_failures"]:
        return EXIT_FAILED
    for stage in report["partial_stages"]:
        sys.stderr.write(f"check_e2e: partial stage {stage}\n")
    return EXIT_PARTIAL if report["partial_stages"] else EXIT_OK


def check_run(run: Path, settings: dict[str, str] | None = None, check: Check | None = None) -> dict:
    """Run every check on ``run`` (absolute) and return the report object.

    ``settings`` are the e2e.env values read once by tier mode; plain mode reads them here.
    """
    check = check or Check()
    rows = read_stages(run, check)
    ran = [stage for stage, _ in rows]
    for stage, code in rows:
        if code not in (EXIT_OK, EXIT_PARTIAL):
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
        proc = run / (settings or e2e_env.load_settings())["E2E_RUN_PROFILE_DIRNAME"] / "proc"
    except ENV_ERRORS as error:
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
        "partial_stages": partial_stages(rows),
        "hard_failures": check.failures,
    }


def baseline_facts(report: dict) -> dict:
    """The BASELINE_FACTS of ``report``: per-stage facts in report order, then region_sides."""
    stages = {
        stage: {fact: facts[fact] for fact in BASELINE_FACTS if fact in facts}
        for stage, facts in report["stages"].items()
    }
    return {"stages": stages, "region_sides": report["region_sides"]}


def compare_baseline(facts: dict, baseline: dict) -> list[str]:
    """Difference lines between the run's facts and a baseline object, in contract order."""
    base_stages = baseline.get("stages")
    base_stages = base_stages if isinstance(base_stages, dict) else {}
    lines = []

    def differ(key: str, run_value: object, base_value: object) -> None:
        if run_value != base_value:
            shown = [
                "absent" if value is _ABSENT else json.dumps(value, sort_keys=True) for value in (run_value, base_value)
            ]
            lines.append(f"check_e2e: baseline difference {key}: run {shown[0]}, baseline {shown[1]}")

    run_stages = facts["stages"]
    for stage in [*run_stages, *(name for name in base_stages if name not in run_stages)]:
        run_entry = run_stages.get(stage, {})
        base_entry = base_stages.get(stage, {})
        base_entry = base_entry if isinstance(base_entry, dict) else {}
        for fact in BASELINE_FACTS:
            differ(f"stages.{stage}.{fact}", run_entry.get(fact, _ABSENT), base_entry.get(fact, _ABSENT))
    differ("region_sides", facts["region_sides"], baseline.get("region_sides", _ABSENT))
    return lines


def write_baseline(path: Path, facts: dict) -> None:
    """Write ``facts`` to ``path`` atomically (temp file in the same directory, then os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as out:
            out.write(json.dumps(facts, indent=2) + "\n")
        os.replace(temp, path)
    except OSError:
        if os.path.exists(temp):
            os.remove(temp)
        raise


def baseline_step(report: dict, check: Check, args: argparse.Namespace, settings: dict[str, str]) -> dict:
    """Accept, refuse or compare the tier-layout baseline (modes B1-B6); returns report["baseline"]."""
    directory = args.baselines_dir or settings.get("E2E_BASELINES_DIR")
    result = {"tier": args.tier, "layout": args.layout, "file": None, "status": None, "differences": []}
    if not directory:
        check.fail("E2E_BASELINES_DIR not set in e2e.env")
        result["status"] = "unconfigured"
        return result
    path = Path(os.path.abspath(os.path.join(directory, f"{args.tier}-{args.layout}.json")))
    result["file"] = str(path)
    if args.accept_baseline:
        if check.failures:
            sys.stderr.write(f"check_e2e: baseline not written (run has hard failures): {path}\n")
            result["status"] = "refused"
            return result
        if report["partial_stages"]:
            sys.stderr.write(f"check_e2e: baseline not written (run is partial): {path}\n")
            result["status"] = "refused"
            return result
        try:
            write_baseline(path, baseline_facts(report))
        except OSError as error:
            check.fail(f"cannot write baseline {path}: {error}")
            result["status"] = "unwritable"
            return result
        sys.stderr.write(f"check_e2e: baseline written: {path}\n")
        result["status"] = "accepted"
        return result
    if not path.exists():
        sys.stderr.write(f"check_e2e: no baseline: {path}\n")
        result["status"] = "missing"
        return result
    try:
        baseline = json.loads(path.read_text())
        if not isinstance(baseline, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as error:  # UnicodeDecodeError and JSONDecodeError included
        check.fail(f"unreadable baseline {path}: {error}")
        result["status"] = "unreadable"
        return result
    for line in compare_baseline(baseline_facts(report), baseline):
        sys.stderr.write(line + "\n")
        check.failures.append(line)
        result["differences"].append(line)
    result["status"] = "compared"
    return result


class _Parser(argparse.ArgumentParser):
    """argparse whose errors raise UsageError instead of printing and exiting."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise UsageError(message)


class _Once(argparse.Action):
    """Store an option value, refusing a second occurrence."""

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest) is not None:
            raise UsageError(f"{option_string} given twice")
        setattr(namespace, self.dest, values)


class _OnceFlag(argparse.Action):
    """Set a flag, refusing a second occurrence."""

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, default=False, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest):
            raise UsageError(f"{option_string} given twice")
        setattr(namespace, self.dest, True)


def parse_args(args: list[str]) -> argparse.Namespace:
    """Parse the command line; any error raises UsageError."""
    parser = _Parser(add_help=False, allow_abbrev=False)
    parser.add_argument("run_dir", nargs="?")
    parser.add_argument("--tier", action=_Once)
    parser.add_argument("--layout", action=_Once)
    parser.add_argument("--baselines-dir", action=_Once)
    parser.add_argument("--accept-baseline", action=_OnceFlag)
    parsed = parser.parse_args(args)
    if parsed.run_dir is None or not Path(parsed.run_dir).is_dir():
        raise UsageError("<run dir> must be one existing directory")
    if (parsed.tier is None) != (parsed.layout is None):
        raise UsageError("--tier and --layout go together")
    if parsed.tier is None and (parsed.baselines_dir is not None or parsed.accept_baseline):
        raise UsageError("--baselines-dir and --accept-baseline need --tier and --layout")
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    args = sys.argv[1:] if argv is None else argv
    try:
        parsed = parse_args(args)
    except UsageError as error:
        sys.stderr.write(f"{USAGE}: {error}\n")
        return EXIT_USAGE
    run = Path(parsed.run_dir).resolve()
    if parsed.tier is None:
        report = check_run(run)
    else:
        try:
            settings = e2e_env.load_settings()
        except ENV_ERRORS as error:
            sys.stderr.write(f"check_e2e: cannot read e2e.env: {error}\n")
            return EXIT_FAILED
        if parsed.tier not in settings.get("E2E_TIERS", "").split():
            sys.stderr.write(f"{USAGE}: unknown tier {parsed.tier!r}\n")
            return EXIT_USAGE
        if parsed.layout not in settings.get(f"E2E_TIER_{parsed.tier.upper()}_LAYOUTS", "").split():
            sys.stderr.write(f"{USAGE}: unknown layout {parsed.layout!r} for tier {parsed.tier}\n")
            return EXIT_USAGE
        check = Check()
        report = check_run(run, settings, check)
        baseline = baseline_step(report, check, parsed, settings)
        hard_failures = report.pop("hard_failures")
        report["baseline"] = baseline
        report["hard_failures"] = hard_failures
    (run / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return exit_status(report)


if __name__ == "__main__":
    sys.exit(main())
