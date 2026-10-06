"""Each submodule's own root .gitignore must ignore the local CodeGraph index.

REQ-SUBMOD-04 (external/champollion_V1) and REQ-SUBMOD-05
(external/cortical_tiles): the submodule's committed root ``.gitignore``
shall contain a pattern that makes git ignore ``.codegraph/`` at the
submodule root.

``git check-ignore --no-index`` still honours ``$GIT_DIR/info/exclude``
and ``core.excludesFile``, so checking the live submodule would pass on
local-only excludes. Instead, the submodule's root ``.gitignore`` alone
is copied into a scratch repository (global excludes disabled) and real
git pattern matching is run there.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

PIPELINE_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.unit  # noqa: V107


def _assert_root_gitignore_ignores_codegraph(submodule: str, tmp_path: Path) -> None:
    sub_root = PIPELINE_ROOT / "external" / submodule
    if not (sub_root / ".git").exists():
        pytest.skip(f"submodule external/{submodule} is not initialised")
    if shutil.which("git") is None:
        pytest.skip("git executable not available")

    gitignore = sub_root / ".gitignore"
    assert gitignore.is_file(), f"external/{submodule} has no root .gitignore"

    scratch = tmp_path / "repo"
    scratch.mkdir()
    subprocess.run(["git", "init", "-q", str(scratch)], check=True)
    shutil.copyfile(gitignore, scratch / ".gitignore")
    (scratch / ".codegraph").mkdir()
    (scratch / ".codegraph" / "index.db").write_text("")

    result = subprocess.run(
        [
            "git",
            "-c",
            f"core.excludesFile={tmp_path / 'no-global-excludes'}",
            "-C",
            str(scratch),
            "check-ignore",
            "-q",
            ".codegraph/",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"external/{submodule}/.gitignore has no pattern ignoring the "
        f".codegraph/ directory (git check-ignore rc={result.returncode}, "
        f"stderr={result.stderr.strip()!r})"
    )


def test_champollion_v1_gitignore_ignores_codegraph_dir(tmp_path):
    """REQ-SUBMOD-04: champollion_V1 root .gitignore ignores .codegraph/."""
    _assert_root_gitignore_ignores_codegraph("champollion_V1", tmp_path)


def test_cortical_tiles_gitignore_ignores_codegraph_dir(tmp_path):
    """REQ-SUBMOD-05: cortical_tiles root .gitignore ignores .codegraph/."""
    _assert_root_gitignore_ignores_codegraph("cortical_tiles", tmp_path)
