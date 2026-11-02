"""The root .dockerignore keeps data, reports and tool state out of the image.

REQ-E2ECI-BDRABCZUK-95DA92B191D8: the data, outputs, htmlcov, .alm, .omc,
.codegraph and .claude directories at the repository root are excluded.
REQ-E2ECI-BDRABCZUK-5B7F244FA2A5: files ending in .nii, .nii.gz, .prof, .csv
or .ipynb are excluded at any directory depth.
REQ-E2ECI-BDRABCZUK-757A1C5D9BAF: comparison_report.json is excluded at any
directory depth.
REQ-E2ECI-BDRABCZUK-B2023BE9C1CD: Python sources under src/ and external/
stay in the build context.

No Docker daemon is used. ``.dockerignore`` is parsed and matched with the
semantics of moby/patternmatcher (the library ``docker build`` uses):

* each pattern is ``filepath.Clean``-ed (trailing ``/`` dropped, leading
  ``/`` stripped) and anchored at the context root;
* ``*`` and ``?`` never cross ``/``; ``**`` matches any number of
  directories, including none;
* a path is excluded when the pattern matches the path or one of its parent
  directories; the last matching pattern wins and ``!`` re-includes.
"""

import os
import posixpath
import re
from pathlib import Path

import pytest

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
DOCKERIGNORE = PIPELINE_ROOT / ".dockerignore"

pytestmark = pytest.mark.unit  # noqa: V107


def _compile(pattern: str) -> re.Pattern[str]:
    """Translate one cleaned dockerignore pattern into an anchored regex."""
    regex = "^"
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "*":
            if i + 1 < len(pattern) and pattern[i + 1] == "*":
                i += 2
                if i < len(pattern) and pattern[i] == "/":
                    i += 1
                regex += ".*" if i >= len(pattern) else "(.*/)?"
                continue
            regex += "[^/]*"
        elif ch == "?":
            regex += "[^/]"
        elif ch == "\\" and i + 1 < len(pattern):
            i += 1
            regex += re.escape(pattern[i])
        else:
            regex += re.escape(ch)
        i += 1
    return re.compile(regex + "$")


def _load_patterns(path: Path) -> list[tuple[bool, re.Pattern[str]]]:
    """Read .dockerignore into (is_exclusion, regex) pairs, in file order."""
    patterns = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        if negate:
            line = line[1:].strip()
        cleaned = posixpath.normpath(line)
        if cleaned.startswith("/") and len(cleaned) > 1:
            cleaned = cleaned[1:]
        patterns.append((negate, _compile(cleaned)))
    return patterns


def _is_excluded(rel_path: str, patterns: list[tuple[bool, re.Pattern[str]]]) -> bool:
    """Docker's MatchesOrParentMatches: the last matching pattern wins."""
    parts = rel_path.split("/")
    candidates = [rel_path] + ["/".join(parts[: i + 1]) for i in range(len(parts) - 1)]
    excluded = False
    for negate, regex in patterns:
        if any(regex.match(candidate) for candidate in candidates):
            excluded = not negate
    return excluded


@pytest.fixture(scope="module")
def patterns() -> list[tuple[bool, re.Pattern[str]]]:
    assert DOCKERIGNORE.is_file(), f"{DOCKERIGNORE} does not exist"
    return _load_patterns(DOCKERIGNORE)


@pytest.mark.parametrize(
    "rel_path",
    [
        "data/ixi/participants.tsv",
        "outputs/run_01/embeddings.pt",
        "htmlcov/index.html",
        ".alm/TASKS.jsonl",
        ".omc/state/mode-state.json",
        ".codegraph/graph.db",
        ".claude/settings.local.json",
    ],
)
def test_root_directories_excluded(patterns, rel_path):
    """REQ-E2ECI-BDRABCZUK-95DA92B191D8."""
    assert _is_excluded(rel_path, patterns), f"{rel_path} is in the Docker build context"


@pytest.mark.parametrize(
    "rel_path",
    [
        "sub-01_T1w.nii",
        "scans/sub-01/t1mri/default_acquisition/sub-01.nii.gz",
        "external/cortical_tiles/data/reference/mask/R/S.C._right.nii.gz",
        "run.prof",
        "src/profiles/stage1.prof",
        "participants.csv",
        "results/run_01/subjects.csv",
        "analysis.ipynb",
        "notebooks/qc/embeddings.ipynb",
    ],
)
def test_files_by_suffix_excluded(patterns, rel_path):
    """REQ-E2ECI-BDRABCZUK-5B7F244FA2A5."""
    assert _is_excluded(rel_path, patterns), f"{rel_path} is in the Docker build context"


@pytest.mark.parametrize(
    "rel_path",
    ["comparison_report.json", "results/run_01/comparison_report.json"],
)
def test_comparison_report_excluded(patterns, rel_path):
    """REQ-E2ECI-BDRABCZUK-757A1C5D9BAF."""
    assert _is_excluded(rel_path, patterns), f"{rel_path} is in the Docker build context"


def test_python_sources_kept(patterns):
    """REQ-E2ECI-BDRABCZUK-B2023BE9C1CD."""
    sources = []
    for top in ("src", "external"):
        for dirpath, dirnames, filenames in os.walk(PIPELINE_ROOT / top):
            dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
            rel_dir = Path(dirpath).relative_to(PIPELINE_ROOT).as_posix()
            sources += [f"{rel_dir}/{name}" for name in filenames if name.endswith(".py")]
    assert sources, "no Python source found under src/ or external/"
    dropped = [path for path in sources if _is_excluded(path, patterns)]
    assert not dropped, f"Python sources left out of the Docker build context: {dropped[:10]}"
