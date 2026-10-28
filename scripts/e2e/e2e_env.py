#!/usr/bin/env python3
"""Reader of scripts/e2e/e2e.env and the open-data allowlist check.

Usage: ``python3 scripts/e2e/e2e_env.py check-path PATH``. Exit 0 when PATH is
allowed, 1 when it is refused (reason on stderr). A path is allowed when it lies
inside one of E2E_ALLOWED_DATA_ROOTS and matches none of E2E_BANNED_DATA_PATTERNS.
Every value comes from e2e.env; this file restates none of them.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

ENV_FILE = Path(__file__).resolve().with_name("e2e.env")
EXIT_ALLOWED = 0
EXIT_REFUSED = 1
EXIT_USAGE = 2


def load_settings(env_file: Path = ENV_FILE, home: str | None = None) -> dict[str, str]:
    """Return the E2E_* settings defined by sourcing ``env_file`` in bash."""
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("bash is required to read e2e.env")
    env = {"HOME": home or os.path.expanduser("~"), "PATH": os.defpath}
    result = subprocess.run(
        [bash, "-euc", 'set -a; source "$1"; set +a; env -0', "_", str(env_file)],
        env=env,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sourcing {env_file} failed: {result.stderr.decode(errors='replace').strip()}")
    settings = {}
    for item in result.stdout.split(b"\0"):
        name, _, value = item.decode().partition("=")
        if name.startswith("E2E_"):
            settings[name] = value
    return settings


def check_data_path(path: str, settings: Mapping[str, str]) -> str | None:
    """Return None when ``path`` is allowed, else the refusal reason."""
    norm = os.path.normpath(path)
    banned = [p for p in settings.get("E2E_BANNED_DATA_PATTERNS", "").split() if re.search(p, norm, re.IGNORECASE)]
    if banned:
        return f"{path}: matches banned data pattern(s) {' '.join(banned)}"
    roots = [os.path.normpath(r) for r in settings.get("E2E_ALLOWED_DATA_ROOTS", "").split()]
    if not roots:
        return "E2E_ALLOWED_DATA_ROOTS is empty"
    if any(norm == root or norm.startswith(root + os.sep) for root in roots):
        return None
    return f"{path}: outside the allowed data roots {' '.join(roots)}"


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or args[0] != "check-path":
        sys.stderr.write("usage: e2e_env.py check-path PATH\n")
        return EXIT_USAGE
    try:
        settings = load_settings()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        sys.stderr.write(f"e2e_env.py: cannot read {ENV_FILE}: {error}\n")
        return EXIT_REFUSED
    reason = check_data_path(args[1], settings)
    if reason is not None:
        sys.stderr.write(f"refused: {reason}\n")
        return EXIT_REFUSED
    return EXIT_ALLOWED


if __name__ == "__main__":
    sys.exit(main())
