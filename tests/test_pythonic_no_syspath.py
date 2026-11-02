"""generate_embeddings and generate_masks import champollion_V1 and cortical_tiles as packages.

Both submodules are editable-installed into the pixi environments (``install-champollion``,
``install-cortical-tiles``), so neither module needs to edit ``sys.path`` to import them.

- REQ-PYTHONIC-BDRABCZUK-7892342E598D: the source of champollion_pipeline.generate_embeddings
  contains zero references to sys.path.
- REQ-PYTHONIC-BDRABCZUK-27F204B72703: the source of champollion_pipeline.generate_masks contains
  zero references to sys.path.
- REQ-PYTHONIC-BDRABCZUK-A12E32025A65: importing champollion_pipeline.generate_embeddings in a
  new Python process succeeds with no sys.path entry under the repository's external directory.

Task: TASK-PYTHONIC-BDRABCZUK-349B41038EC4 (head TASK-PYTHONIC-BDRABCZUK-0BA769629273).
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
EXTERNAL_DIR = REPO_ROOT / "external"
PACKAGE_DIR = SRC_DIR / "champollion_pipeline"

_IMPORT_PROBE = (
    "import json, sys\n"
    "import champollion_pipeline.generate_embeddings as module\n"
    "print(json.dumps({'file': module.__file__, 'path': sys.path}))\n"
)


def _sys_path_reference_lines(source_file: Path) -> list[int]:
    """Line numbers of every ``sys.path`` attribute access and ``from sys import path`` in source_file."""
    tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
    lines = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "path"
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
        ):
            lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module == "sys":
            lines.extend(node.lineno for alias in node.names if alias.name == "path")
    return sorted(lines)


def _is_under(entry: str, directory: Path) -> bool:
    """True when the sys.path entry resolves to directory or to a path inside it."""
    if not entry:
        return False
    resolved = Path(entry).resolve()
    return resolved == directory or directory in resolved.parents


def test_generate_embeddings_source_has_no_sys_path_reference():
    """generate_embeddings.py never reads or edits sys.path."""
    assert _sys_path_reference_lines(PACKAGE_DIR / "generate_embeddings.py") == []


def test_generate_masks_source_has_no_sys_path_reference():
    """generate_masks.py never reads or edits sys.path, brainvisa_dir included."""
    assert _sys_path_reference_lines(PACKAGE_DIR / "generate_masks.py") == []


def test_generate_embeddings_imports_without_external_sys_path_entry():
    """A fresh interpreter imports generate_embeddings with champollion_V1 resolved as an installed package."""
    env = {**os.environ, "PYTHONPATH": str(SRC_DIR)}
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert Path(report["file"]).resolve().is_relative_to(PACKAGE_DIR), report["file"]
    external_entries = [entry for entry in report["path"] if _is_under(entry, EXTERNAL_DIR)]
    assert external_entries == []
