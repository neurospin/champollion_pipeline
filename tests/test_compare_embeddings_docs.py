"""Static checks for ``compare.py embeddings`` (TASK-204).

REQ-COMPARE-71 / 72: documentation of the subcommand in the module docstring
of ``src/compare.py``. REQ-COMPARE-73: no import statement inside the pinned
embeddings functions (project rule: all imports at the top of the file).

Everything is derived from ``src/compare.py`` by static AST inspection (no
import): the option strings of the subparser created by
``subparsers.add_parser("embeddings", ...)``, whatever variable it is bound
to. Adding an embeddings flag without documenting it turns REQ-COMPARE-72 red.

The "embeddings section" is the part of the module docstring that starts at the
dash-underlined heading ``Embeddings comparison (embeddings)`` and ends at the
next dash-underlined heading (or the end of the docstring).
"""

import ast
import re
from pathlib import Path

import pytest

COMPARE_PY = Path(__file__).resolve().parents[1] / "src" / "compare.py"
SECTION_TITLE = "Embeddings comparison (embeddings)"
_UNDERLINE = re.compile(r"^-{3,}$")

# Module-level helpers and method pinned by tests/test_compare_embeddings.py.
EMBEDDINGS_FUNCTIONS = ["linear_cka", "knn_overlap", "procrustes_align", "load_umap_model", "_run_embeddings"]


def _tree() -> ast.Module:
    return ast.parse(COMPARE_PY.read_text(encoding="utf-8"))


def _embeddings_parser_names(tree: ast.Module) -> set[str]:
    """Names bound to ``<x>.add_parser("embeddings", ...)``."""
    names = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "add_parser"
            and node.value.args
            and isinstance(node.value.args[0], ast.Constant)
            and node.value.args[0].value == "embeddings"
        ):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return names


def _embeddings_option_strings(tree: ast.Module) -> list[str]:
    parsers = _embeddings_parser_names(tree)
    options: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in parsers
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("-"):
                    options.append(arg.value)
    return options


def _headings(lines: list[str]) -> list[int]:
    return [i for i in range(len(lines) - 1) if lines[i].strip() and _UNDERLINE.match(lines[i + 1].strip())]


def _embeddings_section(docstring: str) -> str:
    lines = docstring.splitlines()
    heads = _headings(lines)
    for n, i in enumerate(heads):
        if lines[i].strip() == SECTION_TITLE:
            end = heads[n + 1] if n + 1 < len(heads) else len(lines)
            return "\n".join(lines[i:end])
    return ""


def _names(text: str, token: str) -> bool:
    return re.search(rf"(?<![\w\-.]){re.escape(token)}(?![\w])", text) is not None


def _function(tree: ast.Module, name: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


_TREE = _tree()
EMBEDDINGS_OPTIONS = _embeddings_option_strings(_TREE)


@pytest.fixture(scope="module")
def section() -> str:
    return _embeddings_section(ast.get_docstring(_TREE) or "")


# --------------------------------------------------------------------------- #
# REQ-COMPARE-71  section + non-comparability of raw UMAP coordinates
# --------------------------------------------------------------------------- #


class TestEmbeddingsDocsSection:
    """REQ-COMPARE-71."""

    def test_section_heading_present(self, section):
        assert section, f"no dash-underlined '{SECTION_TITLE}' heading in the compare.py docstring"

    def test_states_raw_umap_coordinates_not_comparable_without_alignment(self, section):
        flat = " ".join(section.split()).lower()
        assert re.search(r"raw umap coordinates.{0,80}not comparable.{0,40}without alignment", flat), section


# --------------------------------------------------------------------------- #
# REQ-COMPARE-72  each option named
# --------------------------------------------------------------------------- #


class TestEmbeddingsDocsOptions:
    """REQ-COMPARE-72."""

    def test_embeddings_subparser_found(self):
        """Guard: the AST extraction found the embeddings subparser and core options."""
        expected = {"--set_a", "--set_b", "--output", "--regions", "--side", "--k", "--njobs"}
        assert expected <= set(EMBEDDINGS_OPTIONS), EMBEDDINGS_OPTIONS

    @pytest.mark.parametrize("option", EMBEDDINGS_OPTIONS or ["<no embeddings subparser>"])
    def test_section_names_option(self, section, option):
        assert _names(section, option), f"{option} not documented in the embeddings section"


# --------------------------------------------------------------------------- #
# REQ-COMPARE-73  no import statement inside the embeddings functions
# --------------------------------------------------------------------------- #


class TestEmbeddingsNoFunctionImports:
    """REQ-COMPARE-73."""

    @pytest.mark.parametrize("name", EMBEDDINGS_FUNCTIONS)
    def test_function_has_no_import_statement(self, name):
        func = _function(_TREE, name)
        assert func is not None, f"src/compare.py defines no function {name!r}"
        imports = [
            (node.lineno, ast.unparse(node))
            for node in ast.walk(func)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert imports == []
