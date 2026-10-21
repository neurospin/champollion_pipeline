"""Split of ``generate_snapshots.py`` into ``snapshot_types/`` (TASK-REFACTOR-BDRABCZUK-266C44E23373).

- REQ-SNAPSPLIT-BDRABCZUK-8A493EC51F88: ``generate_sulcal_graph_snapshot``,
  ``generate_tiles_snapshot`` and ``generate_umap_snapshot`` of ``generate_snapshots``
  are the functions defined in ``snapshot_types.sulcal``, ``.tiles`` and ``.umap``.
- REQ-SNAPSPLIT-BDRABCZUK-5AA8DEF7F507: ``generate_snapshots`` keeps its public names.
- REQ-SNAPSPLIT-BDRABCZUK-E0973E5BBDA8: every ``generate_snapshots*`` / ``snapshot_types*``
  Python file under ``src/champollion_pipeline/`` has at most 500 lines.
- REQ-SNAPSPLIT-BDRABCZUK-C99D5DB146E0: ``snapshot_types.sulcal``, ``.tiles`` and ``.umap``
  import without ``anatomist.headless``, ``soma.aims`` and ``cortical_tiles.config``.
"""

import importlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import champollion_pipeline.generate_snapshots as gs

pytestmark = pytest.mark.smoke  # noqa: V107

PACKAGE_DIR = Path(gs.__file__).resolve().parent
SNAPSHOT_TYPES = "champollion_pipeline.snapshot_types"
MAX_LINES = 500
GENERATORS = {
    "sulcal": "generate_sulcal_graph_snapshot",
    "tiles": "generate_tiles_snapshot",
    "umap": "generate_umap_snapshot",
}
PUBLIC_NAMES = (
    "GenerateSnapshots",
    "main",
    "list_acquisitions",
    "find_sulcal_graphs",
    "find_white_mesh",
    "find_completed_regions",
    "discover_umap_pairs",
    "generate_sulcal_graph_snapshot",
    "generate_tiles_snapshot",
    "generate_umap_snapshot",
    "REGION_GRAPH_ROOT_HINT",
    "ICBM_MESH_DIR_FALLBACK",
)
BLOCKED_MODULES = (
    "anatomist",
    "anatomist.headless",
    "soma",
    "soma.aims",
    "cortical_tiles",
    "cortical_tiles.config",
)


@pytest.mark.parametrize("type_module", sorted(GENERATORS))
def test_generator_defined_in_snapshot_types(type_module):
    """REQ-SNAPSPLIT-BDRABCZUK-8A493EC51F88."""
    name = GENERATORS[type_module]
    module_name = f"{SNAPSHOT_TYPES}.{type_module}"
    module = importlib.import_module(module_name)
    defined = getattr(module, name)
    assert defined.__module__ == module_name, (
        f"REQ-SNAPSPLIT-BDRABCZUK-8A493EC51F88: {name} is defined in {defined.__module__}, not {module_name}"
    )
    assert getattr(gs, name) is defined, (
        f"REQ-SNAPSPLIT-BDRABCZUK-8A493EC51F88: generate_snapshots.{name} is not {module_name}.{name}"
    )


def test_generate_snapshots_public_names():
    """REQ-SNAPSPLIT-BDRABCZUK-5AA8DEF7F507."""
    missing = [name for name in PUBLIC_NAMES if not hasattr(gs, name)]
    assert not missing, f"REQ-SNAPSPLIT-BDRABCZUK-5AA8DEF7F507: generate_snapshots lacks {missing}"


def test_snapshot_source_files_at_most_500_lines():
    """REQ-SNAPSPLIT-BDRABCZUK-E0973E5BBDA8."""
    files = sorted(PACKAGE_DIR.glob("generate_snapshots*.py")) + sorted(PACKAGE_DIR.glob("snapshot_types*/**/*.py"))
    assert PACKAGE_DIR / "generate_snapshots.py" in files
    too_long = {}
    for path in files:
        with path.open(encoding="utf-8") as handle:
            count = sum(1 for _ in handle)
        if count > MAX_LINES:
            too_long[str(path.relative_to(PACKAGE_DIR))] = count
    assert not too_long, f"REQ-SNAPSPLIT-BDRABCZUK-E0973E5BBDA8: files over {MAX_LINES} lines: {too_long}"


def test_snapshot_types_import_without_brainvisa(tmp_path):
    """REQ-SNAPSPLIT-BDRABCZUK-C99D5DB146E0."""
    script = textwrap.dedent(
        """
        import importlib
        import json
        import sys

        for name in {blocked!r}:
            sys.modules[name] = None

        errors = {{}}
        for type_module in {modules!r}:
            try:
                importlib.import_module({package!r} + "." + type_module)
            except Exception as exc:
                errors[type_module] = f"{{type(exc).__name__}}: {{exc}}"
        print(json.dumps(errors))
        """
    ).format(blocked=BLOCKED_MODULES, modules=sorted(GENERATORS), package=SNAPSHOT_TYPES)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"child interpreter failed:\n{proc.stdout}{proc.stderr}"
    errors = json.loads(proc.stdout.strip().splitlines()[-1])
    assert not errors, f"REQ-SNAPSPLIT-BDRABCZUK-C99D5DB146E0: import failures without BrainVISA: {errors}"
