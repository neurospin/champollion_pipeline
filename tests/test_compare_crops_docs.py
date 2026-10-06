"""Tests for REQ-COMPARE-31..37 — documentation of ``compare.py crops`` (TASK-195).

The artifact under test is the module docstring of ``src/compare.py``, which
is where the comparison script's subcommands are documented today (its
``Subcommands`` list and ``Usage`` examples).

Every expectation is derived from ``src/compare.py`` itself by static AST
inspection (no import, so neither PyAIMS nor ``champollion_utils`` is needed):
the crops subparser's option strings, ``_CROPS_CSV_COLUMNS``, the keys of the
dict ``_summarise_region`` returns, the top-level ``summary.json`` keys built
in ``Compare._run_crops``, the skipped-entry keys and the skip reason codes.
Adding a flag, a column, a summary field or a reason code without documenting
it therefore turns these tests red.

The "crops section" is the part of the module docstring that starts at a
dash-underlined heading containing the word ``crops`` and ends at the next
dash-underlined heading (or the end of the docstring).
"""

import ast
import re
from pathlib import Path

import pytest

COMPARE_PY = Path(__file__).resolve().parents[1] / "src" / "compare.py"


# --------------------------------------------------------------------------- #
# Static extraction helpers
# --------------------------------------------------------------------------- #


def _tree() -> ast.Module:
    return ast.parse(COMPARE_PY.read_text(encoding="utf-8"))


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{COMPARE_PY} defines no function {name!r}")


def _crops_option_strings(tree: ast.Module) -> list[str]:
    """Option strings passed to ``crops_p.add_argument(...)``."""
    options: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "crops_p"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("-"):
                    options.append(arg.value)
    return options


def _crops_csv_columns(tree: ast.Module) -> list[str]:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_CROPS_CSV_COLUMNS" for t in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise AssertionError(f"{COMPARE_PY} defines no _CROPS_CSV_COLUMNS")


def _returned_dict_keys(func: ast.FunctionDef) -> list[str]:
    keys: list[str] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            keys += [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
    return keys


def _region_entry_keys(tree: ast.Module) -> list[str]:
    return _returned_dict_keys(_function(tree, "_summarise_region"))


def _summary_top_level_keys(tree: ast.Module) -> list[str]:
    run_crops = _function(tree, "_run_crops")
    for node in ast.walk(run_crops):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "summary" for t in node.targets)
            and isinstance(node.value, ast.Dict)
        ):
            return [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
    raise AssertionError("Compare._run_crops builds no `summary = {...}` dict")


def _skipped_entry_keys(tree: ast.Module) -> list[str]:
    skip = _function(_function(tree, "_compare_region_side"), "_skip")
    keys: list[str] = []
    for node in ast.walk(skip):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "entry" and isinstance(node.value, ast.Dict):
                    keys += [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "entry"
                    and isinstance(target.slice, ast.Constant)
                ):
                    keys.append(target.slice.value)
    return keys


def _skip_reason_codes(tree: ast.Module) -> list[str]:
    """Reason literals from ``_compute_crop_offset`` returns and ``_skip(...)`` calls."""
    reasons: list[str] = []
    for node in ast.walk(_function(tree, "_compute_crop_offset")):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple) and len(node.value.elts) == 2:
            second = node.value.elts[1]
            if isinstance(second, ast.Constant) and isinstance(second.value, str):
                reasons.append(second.value)
    for node in ast.walk(_function(tree, "_compare_region_side")):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_skip"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            reasons.append(node.args[0].value)
    return sorted(set(reasons))


def _module_docstring(tree: ast.Module) -> str:
    return ast.get_docstring(tree) or ""


_UNDERLINE = re.compile(r"^-{3,}$")


def _headings(lines: list[str]) -> list[int]:
    return [i for i in range(len(lines) - 1) if lines[i].strip() and _UNDERLINE.match(lines[i + 1].strip())]


def _crops_section(docstring: str) -> str:
    lines = docstring.splitlines()
    heads = _headings(lines)
    for n, i in enumerate(heads):
        if re.search(r"\bcrops\b", lines[i]):
            end = heads[n + 1] if n + 1 < len(heads) else len(lines)
            return "\n".join(lines[i:end])
    return ""


def _subcommands_section(docstring: str) -> str:
    lines = docstring.splitlines()
    heads = _headings(lines)
    for n, i in enumerate(heads):
        if lines[i].strip() == "Subcommands":
            end = heads[n + 1] if n + 1 < len(heads) else len(lines)
            return "\n".join(lines[i + 2 : end])
    return ""


def _names(text: str, token: str) -> bool:
    """True when ``token`` appears in ``text`` as a standalone name."""
    return re.search(rf"(?<![\w\-.]){re.escape(token)}(?![\w])", text) is not None


# --------------------------------------------------------------------------- #
# Fixtures and expectation lists (computed at collection time from the code)
# --------------------------------------------------------------------------- #

_TREE = _tree()
CROPS_OPTIONS = _crops_option_strings(_TREE)
CROPS_CSV_COLUMNS = _crops_csv_columns(_TREE)
SUMMARY_KEYS = (
    [("top-level", k) for k in _summary_top_level_keys(_TREE)]
    + [("regions entry", k) for k in _region_entry_keys(_TREE)]
    + [("skipped entry", k) for k in _skipped_entry_keys(_TREE)]
)
SKIP_REASONS = _skip_reason_codes(_TREE)


@pytest.fixture(scope="module")
def docstring() -> str:
    return _module_docstring(_TREE)


@pytest.fixture(scope="module")
def crops_section(docstring) -> str:
    return _crops_section(docstring)


def test_expectations_are_derived_from_code():
    """Guard: the AST extraction found the crops CLI, CSV, summary and reasons."""
    assert {"--set_a", "--set_b", "--output"} <= set(CROPS_OPTIONS), CROPS_OPTIONS
    assert "pct_lost" in CROPS_CSV_COLUMNS, CROPS_CSV_COLUMNS
    assert ("regions entry", "alignment_offset_vox") in SUMMARY_KEYS, SUMMARY_KEYS
    assert ("skipped entry", "reason") in SUMMARY_KEYS, SUMMARY_KEYS
    assert {"no_mask_cropped", "missing_in_a"} <= set(SKIP_REASONS), SKIP_REASONS


# --------------------------------------------------------------------------- #
# REQ-COMPARE-31 — Subcommands list names crops
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_subcommands_list_names_crops(docstring):
    """REQ-COMPARE-31: the docstring's Subcommands list names the crops subcommand."""
    subcommands = _subcommands_section(docstring)
    assert subcommands, "src/compare.py module docstring has no 'Subcommands' section"
    assert re.search(r"^\s*crops\b", subcommands, re.MULTILINE), (
        f"'Subcommands' list does not name crops:\n{subcommands}"
    )


# --------------------------------------------------------------------------- #
# REQ-COMPARE-32 — crops section with example invocation
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_crops_section_has_example_invocation(crops_section):
    """REQ-COMPARE-32: a dash-underlined 'crops' section shows `python compare.py crops ...`."""
    assert crops_section, (
        "src/compare.py module docstring has no section headed by a dash-underlined line containing 'crops'"
    )
    assert "python compare.py crops" in crops_section, (
        f"crops section has no example invocation beginning 'python compare.py crops':\n{crops_section}"
    )


# --------------------------------------------------------------------------- #
# REQ-COMPARE-33 — every crops option string is named
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize("option", CROPS_OPTIONS)
def test_crops_section_names_option(crops_section, option):
    """REQ-COMPARE-33: each option registered on the crops subparser is named in the crops section."""
    assert _names(crops_section, option), f"crops section does not name option {option!r}"


# --------------------------------------------------------------------------- #
# REQ-COMPARE-34 — per_subject.csv and its columns
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_crops_section_names_per_subject_csv(crops_section):
    """REQ-COMPARE-34: the crops section names the per_subject.csv output file."""
    assert _names(crops_section, "per_subject.csv"), "crops section does not name per_subject.csv"


@pytest.mark.smoke
@pytest.mark.parametrize("column", CROPS_CSV_COLUMNS)
def test_crops_section_names_csv_column(crops_section, column):
    """REQ-COMPARE-34: each _CROPS_CSV_COLUMNS column is named in the crops section."""
    assert _names(crops_section, column), f"crops section does not name per_subject.csv column {column!r}"


# --------------------------------------------------------------------------- #
# REQ-COMPARE-35 — summary.json structure
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_crops_section_names_summary_json(crops_section):
    """REQ-COMPARE-35: the crops section names the summary.json output file."""
    assert _names(crops_section, "summary.json"), "crops section does not name summary.json"


@pytest.mark.smoke
@pytest.mark.parametrize(("level", "key"), SUMMARY_KEYS, ids=[f"{lv}:{k}" for lv, k in SUMMARY_KEYS])
def test_crops_section_names_summary_key(crops_section, level, key):
    """REQ-COMPARE-35: each summary.json key (top-level, regions entry, skipped entry) is named."""
    assert _names(crops_section, key), f"crops section does not name summary.json {level} key {key!r}"


# --------------------------------------------------------------------------- #
# REQ-COMPARE-36 — skip reason codes
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize("reason", SKIP_REASONS)
def test_crops_section_names_skip_reason(crops_section, reason):
    """REQ-COMPARE-36: each skip reason code recordable in summary.json is named."""
    assert _names(crops_section, reason), f"crops section does not name skip reason code {reason!r}"


# --------------------------------------------------------------------------- #
# REQ-COMPARE-37 — PyAIMS prerequisite and mask_cropped.nii.gz headers
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_crops_section_names_pyaims_prerequisite(crops_section):
    """REQ-COMPARE-37: the crops section names PyAIMS (soma.aims) as a prerequisite."""
    assert "PyAIMS" in crops_section, "crops section does not name PyAIMS"
    assert "soma.aims" in crops_section, "crops section does not name soma.aims"


@pytest.mark.smoke
def test_crops_section_names_mask_cropped_header_file(crops_section):
    """REQ-COMPARE-37: the crops section names mask_cropped.nii.gz as the header file crops reads."""
    assert "mask_cropped.nii.gz" in crops_section, "crops section does not name mask_cropped.nii.gz"
