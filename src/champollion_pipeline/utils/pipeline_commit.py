"""The commit of the code that runs, for the pipeline_commit key of provenance.json.

The code directory is the directory holding src/champollion_pipeline. A commit comes from the repository
rooted there or from the COMMIT file there, never from a repository rooted elsewhere.
"""

import re
import subprocess
from pathlib import Path

ONE_COMMIT = r"[0-9a-f]{40}"


def _head_of(code_dir: Path) -> str | None:
    """HEAD of the repository rooted at code_dir, or None; --git-dir stops git climbing to an enclosing repository."""
    try:
        result = subprocess.run(
            ["git", f"--git-dir={code_dir / '.git'}", "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and re.fullmatch(ONE_COMMIT, commit) else None


def _commit_file_of(code_dir: Path) -> str | None:
    """Commit held by the COMMIT file of code_dir, or None when that file holds anything else or cannot be read."""
    try:
        content = (code_dir / "COMMIT").read_bytes().strip()
    except OSError:
        return None
    return content.decode() if re.fullmatch(ONE_COMMIT.encode(), content) else None


def read_pipeline_commit() -> str | None:
    """Return the 40-hex commit of the code that runs: HEAD, else the COMMIT file, else None."""
    package = Path(__file__).resolve().parents[1]
    if package.parent.name != "src":
        return None
    code_dir = package.parents[1]
    return _head_of(code_dir) or _commit_file_of(code_dir)
