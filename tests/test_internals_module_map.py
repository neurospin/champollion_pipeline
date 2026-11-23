"""Guard: the Module map of ``docs/internals.md`` names the console scripts of ``pyproject.toml``.

REQ-DOCS-BDRABCZUK-9BA3DE89DEA3: the Entry point column of the Module map table names exactly the
console scripts declared under ``[project.scripts]``, one script per row, beside the module whose
``main`` function that script targets. Reads text only: no filesystem writes, no network.
"""

from __future__ import annotations

from pathlib import Path

import tomllib

from tests.docs_test_helpers import MD_SEPARATOR_CELL_RE, md_cell_text, md_section_lines, read_text

REPO_ROOT = Path(__file__).resolve().parent.parent
INTERNALS_MD = REPO_ROOT / "docs" / "internals.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_PREFIX = "champollion_pipeline."


def _declared_scripts() -> dict[str, str]:
    """``{console script: module}`` of ``[project.scripts]``, the module without the package prefix."""
    scripts = tomllib.loads(read_text(PYPROJECT))["project"]["scripts"]
    assert scripts, "pyproject.toml declares no [project.scripts] (test parser sanity check)"
    declared = {}
    for name, target in scripts.items():
        module, _, function = target.partition(":")
        assert module.startswith(PACKAGE_PREFIX) and function == "main", (
            f"[project.scripts] {name} = {target!r} does not target champollion_pipeline.<module>:main"
        )
        declared[name] = module.removeprefix(PACKAGE_PREFIX)
    return declared


def _module_map_rows() -> list[dict[str, str]]:
    """Body rows of the Module map table, as ``{column header: cell text}``."""
    table = [
        [md_cell_text(c) for c in line.strip().strip("|").split("|")]
        for line in md_section_lines(read_text(INTERNALS_MD), "Module map")
        if line.strip().startswith("|")
    ]
    body = [cells for cells in table[1:] if not all(MD_SEPARATOR_CELL_RE.match(c) for c in cells if c)]
    assert table and body, "the Module map section holds no Markdown table with rows (test parser sanity check)"
    header = table[0]
    assert {"Module", "Entry point"} <= set(header), f"Module map header lacks Module or Entry point: {header}"
    return [dict(zip(header, cells, strict=True)) for cells in body]


def test_module_map_names_no_undeclared_entry_point() -> None:
    """No Entry point cell names a command that ``[project.scripts]`` does not declare."""
    declared = _declared_scripts()
    undeclared = [row["Entry point"] for row in _module_map_rows() if row["Entry point"] not in declared]
    assert not undeclared, f"Module map Entry point cells not declared in [project.scripts]: {undeclared}"


def test_module_map_lists_each_console_script_once_beside_its_module() -> None:
    """Each declared console script sits in exactly one row, whose Module cell is the script's target module."""
    rows = _module_map_rows()
    wrong = []
    for script, module in _declared_scripts().items():
        modules = [row["Module"] for row in rows if row["Entry point"] == script]
        if modules != [module]:
            wrong.append(f"{script}: expected one row with module {module!r}, found modules {modules}")
    assert not wrong, "Module map rows disagree with [project.scripts]:\n" + "\n".join(wrong)
