#!/usr/bin/env python3
"""Champollion pipeline install health check.

Runs on bare system Python 3.8+ (no pixi env required).
Rich is used for pretty output when available; plain ASCII fallback otherwise.

Usage
-----
  python3 scripts/install_health.py              # report only
  python3 scripts/install_health.py --fix        # auto-fix stale editable installs
  python3 scripts/install_health.py --pre-update # fetch remote diff, warn before update
  python3 scripts/install_health.py --report     # save full diagnostic report to file
  python3 scripts/install_health.py --output /path/to/report.txt  # explicit output path

  pixi run check-install
  pixi run check-install-fix
  pixi run pre-update
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UTILS_DIR = PROJECT_ROOT.parent / "champollion_utils"

PACKAGES = {
    "champollion_pipeline": {
        "src": PROJECT_ROOT,
        "fix_cmd": [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "--no-build-isolation"],
        "fix_cwd": PROJECT_ROOT,
    },
    "champollion": {
        "src": PROJECT_ROOT / "external" / "champollion_V1",
        "fix_cmd": [sys.executable, "-m", "pip", "install", "-e",
                    "external/champollion_V1", "--no-deps", "--no-build-isolation"],
        "fix_cwd": PROJECT_ROOT,
    },
    "cortical_tiles": {
        "src": PROJECT_ROOT / "external" / "cortical_tiles",
        "fix_cmd": [sys.executable, "-m", "pip", "install", "-e",
                    "external/cortical_tiles", "--no-deps", "--no-build-isolation"],
        "fix_cwd": PROJECT_ROOT,
        "env_extra": {"SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL": "True"},
    },
    "champollion_utils": {
        "src": UTILS_DIR,
        "fix_cmd": [sys.executable, "-m", "pip", "install", "-e",
                    str(UTILS_DIR), "--no-deps", "--no-build-isolation"],
        "fix_cwd": PROJECT_ROOT,
    },
}

SUBMODULES = {
    "external/champollion_V1": PROJECT_ROOT / "external" / "champollion_V1",
    "external/cortical_tiles": PROJECT_ROOT / "external" / "cortical_tiles",
}

# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    name: str
    ok: bool
    status: str       # human-readable one-liner
    fixable: bool = False
    fix_attempted: bool = False
    fix_ok: bool = False
    fix_output: str = ""
    manual_cmd: str = ""


@dataclass
class State:
    checks: list[CheckResult] = field(default_factory=list)
    fix_log: list[str] = field(default_factory=list)
    unfixable: list[str] = field(default_factory=list)
    manual_cmds: list[str] = field(default_factory=list)
    remote_info: list[str] = field(default_factory=list)
    sys_info: dict = field(default_factory=dict)
    git_info: dict = field(default_factory=dict)

# ── Output helpers ────────────────────────────────────────────────────────────


def _out(msg: str) -> None:
    print(msg, flush=True)


def _tag(tag: str, msg: str) -> str:
    return f"[{tag}] {msg}"


def log_check(result: CheckResult) -> None:
    icon = "OK" if result.ok else "FAIL"
    _out(_tag("CHECK", f"{result.name:<30} {icon}  {result.status}"))


def log_fix(msg: str) -> None:
    _out(_tag("FIX", msg))


def log_warn(msg: str) -> None:
    _out(_tag("WARN", msg))


def log_remote(msg: str) -> None:
    _out(_tag("REMOTE", msg))

# ── System info ───────────────────────────────────────────────────────────────


def collect_sys_info() -> dict:
    info: dict = {
        "OS": platform.platform(),
        "python": f"{sys.version.split()[0]} (at {sys.executable})",
    }
    for tool in ("pixi", "git", "pip"):
        path = shutil.which(tool)
        if path:
            try:
                ver = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5)
                info[tool] = ver.stdout.strip().splitlines()[0] if ver.returncode == 0 else path
            except Exception:
                info[tool] = path
        else:
            info[tool] = "NOT FOUND"
    return info


def collect_git_info() -> dict:
    info: dict = {}
    try:
        for key, cmd in [
            ("branch", ["git", "branch", "--show-current"]),
            ("HEAD", ["git", "rev-parse", "--short", "HEAD"]),
            ("remote_url", ["git", "remote", "get-url", "origin"]),
        ]:
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=5)
            info[key] = r.stdout.strip() if r.returncode == 0 else "unknown"
        # commits behind
        r = subprocess.run(
            ["git", "rev-list", "--count", "HEAD..origin/main"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=5,
        )
        info["commits_behind"] = r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        pass
    return info

# ── Package checks ────────────────────────────────────────────────────────────


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def _pkg_importable(name: str) -> bool:
    spec = importlib.util.find_spec(name)
    return spec is not None


def check_package(name: str, meta: dict, state: State, fix: bool) -> CheckResult:
    importable = _pkg_importable(name)
    version = _pkg_version(name)
    src_path: Path = meta["src"]
    src_exists = src_path.is_dir() and any(src_path.iterdir()) if src_path.exists() else False

    if importable:
        result = CheckResult(
            name=name, ok=True,
            status=f"OK  version={version or 'unknown'}  src={src_path}",
        )
    elif not src_exists:
        msg = f"NOT importable, source missing: {src_path}"
        if name == "champollion_utils":
            manual = f"git clone https://github.com/neurospin/champollion_utils.git {UTILS_DIR}"
        else:
            manual = f"git submodule update --init {meta.get('submodule', '')}"
        result = CheckResult(name=name, ok=False, status=msg, fixable=False, manual_cmd=manual)
        state.unfixable.append(f"{name}: {msg}")
        state.manual_cmds.append(manual)
        log_warn(f"Cannot auto-fix {name}: source directory missing.\n        Run: {manual}")
    else:
        result = CheckResult(name=name, ok=False, status="NOT importable (source exists)",
                             fixable=True)
        if fix:
            result = _run_fix(name, meta, result, state)
        else:
            manual = "  ".join(str(c) for c in meta["fix_cmd"])
            result.manual_cmd = manual
            state.manual_cmds.append(manual)
            log_warn(f"{name} not installed. Run: {manual}")

    log_check(result)
    state.checks.append(result)
    return result


def _run_fix(name: str, meta: dict, result: CheckResult, state: State) -> CheckResult:
    cmd = meta["fix_cmd"]
    cwd = meta.get("fix_cwd", PROJECT_ROOT)
    env = os.environ.copy()
    env.update(meta.get("env_extra", {}))
    cmd_str = " ".join(str(c) for c in cmd)
    log_fix(f"Running: {cmd_str}")
    state.fix_log.append(f"[ATTEMPTED] {cmd_str}")
    try:
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            result.fix_attempted = True
            result.fix_ok = True
            result.ok = True
            result.status = f"FIXED by pip install"
            log_fix(f"OK  {name} installed successfully")
            state.fix_log.append(f"[OK] {name} installed")
        else:
            result.fix_attempted = True
            result.fix_ok = False
            output = (r.stderr or r.stdout or "")[:2000]
            result.fix_output = output
            state.fix_log.append(f"[FAILED] return code {r.returncode}")
            state.fix_log.append(f"[STDERR] {output}")
            state.unfixable.append(f"{name}: pip install failed (rc={r.returncode})")
            state.manual_cmds.append(" ".join(str(c) for c in cmd))
            log_fix(f"FAILED {name}  rc={r.returncode}")
            log_warn(output[:500])
    except Exception as exc:
        result.fix_attempted = True
        result.fix_ok = False
        state.unfixable.append(f"{name}: fix raised {exc}")
        log_fix(f"ERROR: {exc}")
    return result

# ── Submodule checks ──────────────────────────────────────────────────────────


def check_submodule(rel_path: str, abs_path: Path, state: State) -> CheckResult:
    initialized = abs_path.exists() and abs_path.is_dir() and any(abs_path.iterdir())
    if initialized:
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, cwd=abs_path, timeout=5,
            )
            commit = r.stdout.strip() if r.returncode == 0 else "unknown"
        except Exception:
            commit = "unknown"
        result = CheckResult(name=rel_path, ok=True, status=f"initialized  commit={commit}")
    else:
        manual = f"git submodule update --init {rel_path}"
        result = CheckResult(
            name=rel_path, ok=False,
            status="NOT initialized",
            fixable=False,
            manual_cmd=manual,
        )
        state.unfixable.append(f"{rel_path}: not initialized")
        state.manual_cmds.append(manual)
        log_warn(f"Submodule {rel_path} not initialized.\n        Run: {manual}")
    log_check(result)
    state.checks.append(result)
    return result

# ── Remote diff (--pre-update) ────────────────────────────────────────────────


def pre_update_checks(state: State) -> None:
    log_remote("Fetching origin …")
    try:
        subprocess.run(
            ["git", "fetch", "--quiet"], cwd=PROJECT_ROOT, check=True, timeout=30,
        )
    except Exception as exc:
        log_warn(f"git fetch failed: {exc}")
        state.remote_info.append(f"WARN: git fetch failed: {exc}")
        return

    # Commits ahead
    try:
        r = subprocess.run(
            ["git", "log", "HEAD..origin/main", "--oneline"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=10,
        )
        commits = [l for l in r.stdout.strip().splitlines() if l]
        if commits:
            log_remote(f"{len(commits)} new commit(s) since your HEAD:")
            for c in commits:
                _out(f"    {c}")
            state.remote_info.extend(commits)
        else:
            log_remote("Already up to date with origin/main")
    except Exception as exc:
        log_warn(f"Could not list remote commits: {exc}")

    # pixi.toml changed?
    try:
        r = subprocess.run(
            ["git", "diff", "HEAD", "origin/main", "--", "pixi.toml"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=10,
        )
        if r.stdout.strip():
            msg = "pixi.toml changed upstream → run 'pixi install' after update"
            log_remote(msg)
            state.remote_info.append(msg)
    except Exception:
        pass

    # pixi.lock conflict?
    try:
        r = subprocess.run(
            ["git", "merge-tree", "HEAD", "origin/main", "--", "pixi.lock"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=10,
        )
        if "<<<<<<" in r.stdout:
            msg = "pixi.lock WILL conflict → safe to reset: 'git checkout -- pixi.lock' before merge"
            log_remote(msg)
            state.remote_info.append(msg)
    except Exception:
        # merge-tree with path args is git >= 2.38; fallback: check if lock differs
        try:
            r = subprocess.run(
                ["git", "diff", "HEAD", "origin/main", "--name-only"],
                capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=10,
            )
            if "pixi.lock" in r.stdout:
                msg = "pixi.lock differs from remote → may conflict; reset with: 'git checkout -- pixi.lock'"
                log_remote(msg)
                state.remote_info.append(msg)
        except Exception:
            pass

    # Submodule pointer changed?
    try:
        r = subprocess.run(
            ["git", "diff", "HEAD", "origin/main", "--", ".gitmodules"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=10,
        )
        if r.stdout.strip():
            msg = "submodule pointer changed upstream → run 'git submodule update --init --remote --force' after update"
            log_remote(msg)
            state.remote_info.append(msg)
    except Exception:
        pass

# ── Report generation ─────────────────────────────────────────────────────────


def generate_report(state: State) -> str:
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    sep = "─" * 65
    lines: list[str] = [
        sep,
        f" CHAMPOLLION INSTALL REPORT  —  {ts}",
        sep,
        "",
        "SYSTEM",
    ]
    for k, v in state.sys_info.items():
        lines.append(f"  {k:<12} {v}")

    lines += ["", "REPOSITORY"]
    for k, v in state.git_info.items():
        lines.append(f"  {k:<16} {v}")
    lines.append(f"  {'path':<16} {PROJECT_ROOT}")

    lines += ["", "PACKAGE STATUS"]
    pkg_names = set(PACKAGES.keys())
    for chk in state.checks:
        if chk.name in pkg_names:
            icon = "OK" if chk.ok else "FAIL"
            lines.append(f"  {chk.name:<28} {icon}  {chk.status}")

    lines += ["", "SUBMODULES"]
    for chk in state.checks:
        if chk.name in SUBMODULES:
            icon = "OK" if chk.ok else "FAIL"
            lines.append(f"  {chk.name:<38} {icon}  {chk.status}")

    if state.fix_log:
        lines += ["", "AUTO-FIX LOG"]
        for entry in state.fix_log:
            lines.append(f"  {entry}")

    if state.remote_info:
        lines += ["", "REMOTE STATE"]
        for entry in state.remote_info:
            lines.append(f"  {entry}")

    if state.unfixable:
        lines += ["", "WHAT I CANNOT FIX"]
        for item in state.unfixable:
            lines.append(f"  - {item}")

    if state.manual_cmds:
        lines += ["", "COMMANDS TO RUN MANUALLY"]
        for i, cmd in enumerate(state.manual_cmds, 1):
            lines.append(f"  {i}. {cmd}")

    lines += [
        "",
        sep,
        "Send this report to Barthélémy (champollion maintainer) or open a GitHub issue.",
        sep,
    ]
    return "\n".join(lines)

# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Champollion pipeline install health check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fix", action="store_true", help="Auto-fix stale editable installs")
    parser.add_argument("--pre-update", action="store_true", dest="pre_update",
                        help="Fetch remote diff and warn about what update will change")
    parser.add_argument("--report", action="store_true",
                        help="Save full diagnostic report to a file")
    parser.add_argument("--output", metavar="FILE",
                        help="Path for the report file (default: install_report_TIMESTAMP.txt)")
    args = parser.parse_args()

    state = State()
    _out("")
    _out("=== Champollion install health check ===")
    _out(f"    project: {PROJECT_ROOT}")
    _out(f"    python:  {sys.executable}")
    _out("")

    state.sys_info = collect_sys_info()
    state.git_info = collect_git_info()

    if args.pre_update:
        _out("--- Remote diff ---")
        pre_update_checks(state)
        _out("")

    _out("--- Submodules ---")
    for rel, abs_path in SUBMODULES.items():
        check_submodule(rel, abs_path, state)
    _out("")

    _out("--- Packages ---")
    for name, meta in PACKAGES.items():
        check_package(name, meta, state, fix=args.fix)
    _out("")

    all_ok = all(c.ok for c in state.checks)

    if all_ok:
        _out("[OK] All checks passed.")
    else:
        failed = [c for c in state.checks if not c.ok]
        _out(f"[FAIL] {len(failed)} check(s) failed:")
        for c in failed:
            _out(f"  - {c.name}: {c.status}")
        _out("")
        if state.manual_cmds:
            _out("COMMANDS TO RUN MANUALLY:")
            for i, cmd in enumerate(state.manual_cmds, 1):
                _out(f"  {i}. {cmd}")
            _out("")

    if args.report or state.unfixable or (args.fix and any(c.fix_attempted and not c.fix_ok for c in state.checks)):
        report_text = generate_report(state)
        if args.output:
            out_path = Path(args.output)
        else:
            ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            out_path = PROJECT_ROOT / f"install_report_{ts}.txt"

        out_path.write_text(report_text, encoding="utf-8")
        _out(f"Report saved: {out_path}")
        _out("Send this file to Barthélémy or open a GitHub issue.")
        _out("")
        print(report_text)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
