"""compare_sulcal and generate_umap_reference import champollion_V1 and cortical_tiles as packages.

Both submodules are editable-installed into the pixi environments (``install-champollion``,
``install-cortical-tiles``): ``cortical_tiles.brainvisa`` and ``champollion.utils`` import as
packages, so neither compare_sulcal nor generate_umap_reference needs to edit ``sys.path``.
Same approach and helpers as ``tests/test_pythonic_no_syspath.py``.

- REQ-PYTHONIC-BDRABCZUK-EB180CC06E9E: the source of compare_sulcal.cli contains zero references
  to sys.path.
- REQ-PYTHONIC-BDRABCZUK-D25CE4F68A29: the source of compare_sulcal.databases contains zero
  references to sys.path.
- REQ-PYTHONIC-BDRABCZUK-768577DC9A7B: the source of generate_umap_reference contains zero
  references to sys.path.
- REQ-PYTHONIC-BDRABCZUK-8B002982947E: importing generate_umap_reference in a new Python process
  succeeds with no sys.path entry under the repository's external directory.

Task: TASK-PYTHONIC-BDRABCZUK-5C088134FBDB (head TASK-PYTHONIC-BDRABCZUK-59A64A6FE550).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from tests.test_pythonic_no_syspath import EXTERNAL_DIR, REPO_ROOT, SRC_DIR, _is_under, _sys_path_reference_lines

COMPARE_DIR = SRC_DIR / "compare_sulcal"

_IMPORT_PROBE = (
    "import json, sys\n"
    "import generate_umap_reference as module\n"
    "print(json.dumps({'file': module.__file__, 'path': sys.path}))\n"
)


def test_compare_sulcal_cli_source_has_no_sys_path_reference():
    """compare_sulcal/cli.py never reads or edits sys.path, brainvisa_dir included."""
    assert _sys_path_reference_lines(COMPARE_DIR / "cli.py") == []


def test_compare_sulcal_databases_source_has_no_sys_path_reference():
    """compare_sulcal/databases.py never reads or edits sys.path in its joblib workers."""
    assert _sys_path_reference_lines(COMPARE_DIR / "databases.py") == []


def test_generate_umap_reference_source_has_no_sys_path_reference():
    """generate_umap_reference.py never reads or edits sys.path."""
    assert _sys_path_reference_lines(SRC_DIR / "generate_umap_reference.py") == []


def test_generate_umap_reference_imports_without_external_sys_path_entry():
    """A fresh interpreter imports generate_umap_reference with champollion_V1 resolved as an installed package."""
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
    assert Path(report["file"]).resolve() == (SRC_DIR / "generate_umap_reference.py").resolve(), report["file"]
    external_entries = [entry for entry in report["path"] if _is_under(entry, EXTERNAL_DIR)]
    assert external_entries == []
