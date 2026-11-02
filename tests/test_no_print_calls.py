"""Guard: no call to the built-in ``print`` in ``main.py`` or ``src/`` outside PRINT_ALLOWLIST.

REQ-LOGGING-BDRABCZUK-1B44822D6DA4: each call to the built-in print function in main.py
or in a Python file under src/ lies in a file and function that a PRINT_ALLOWLIST entry names.
REQ-LOGGING-BDRABCZUK-417F79493369: each PRINT_ALLOWLIST entry names an existing file path,
a function qualified name and a non-empty reason.
REQ-LOGGING-BDRABCZUK-67F2EF89794A: each PRINT_ALLOWLIST entry names a file and function
that contain at least one call to the built-in print function.

The scan is an AST walk: only ``ast.Call`` nodes whose ``func`` is ``ast.Name(id="print")``
count, so attribute calls such as ``report.print()`` are not print calls (the
pipeline-logging-convention rule). Program output goes through the champollion_utils logger;
an entry here is for a print whose stdout text is the program's machine-read output.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_LEVEL = "<module>"

# "<repo-relative path>:<function qualified name>" -> reason the print stays a print.
PRINT_ALLOWLIST: dict[str, str] = {
    "src/champollion_pipeline/profile_report.py:ProfileReport.run": (
        "--format json writes the report document to stdout as the command's machine-read output; "
        "a log record would add a level and source prefix and break the JSON."
    ),
}


def _scanned_files() -> list[Path]:
    """``main.py`` and every Python file under ``src/``, sorted."""
    return [REPO_ROOT / "main.py", *sorted((REPO_ROOT / "src").rglob("*.py"))]


def print_sites(source: str) -> list[tuple[int, str]]:
    """(line, enclosing function qualified name) of each built-in print call in ``source``."""
    sites: list[tuple[int, str]] = []

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            inner = scope
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                inner = (*scope, child.name)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "print":
                sites.append((child.lineno, ".".join(scope) or MODULE_LEVEL))
            visit(child, inner)

    visit(ast.parse(source), ())
    return sorted(sites)


def _repo_print_sites() -> dict[str, list[int]]:
    """``"<path>:<qualname>"`` -> lines of its print calls, over the scanned files."""
    found: dict[str, list[int]] = {}
    for path in _scanned_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        for line, qualname in print_sites(path.read_text(encoding="utf-8")):
            found.setdefault(f"{relative}:{qualname}", []).append(line)
    return found


def test_print_sites_lists_builtin_print_calls_only():
    source = (
        "print('module')\n"
        "class Report:\n"
        "    def run(self):\n"
        "        def inner():\n"
        "            print('nested')\n"
        "        self.report.print()\n"
        "        print('method')\n"
        "async def main():\n"
        "    print('async')\n"
    )
    assert print_sites(source) == [
        (1, MODULE_LEVEL),
        (5, "Report.run.inner"),
        (7, "Report.run"),
        (9, "main"),
    ]


def test_no_print_call_outside_allowlist():
    offenders = [
        f"{site} (line {line})"
        for site, lines in sorted(_repo_print_sites().items())
        if site not in PRINT_ALLOWLIST
        for line in lines
    ]
    assert offenders == [], "print() outside PRINT_ALLOWLIST; log it instead:\n" + "\n".join(offenders)


def test_allowlist_entry_names_file_function_and_reason():
    problems = []
    for entry, reason in PRINT_ALLOWLIST.items():
        path, _, qualname = entry.partition(":")
        if not (REPO_ROOT / path).is_file():
            problems.append(f"{entry}: no file {path}")
        if not qualname:
            problems.append(f"{entry}: no function qualified name after ':'")
        if not reason.strip():
            problems.append(f"{entry}: empty reason")
    assert problems == []


def test_allowlist_has_no_stale_entry():
    sites = _repo_print_sites()
    stale = [entry for entry in PRINT_ALLOWLIST if entry not in sites]
    assert stale == [], f"PRINT_ALLOWLIST entries with no print call left: {stale}"
