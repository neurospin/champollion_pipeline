"""Tests for REQ-TEST-HYGIENE-01 — class-scoped fixtures are never instance methods.

pytest 9.1 emits ``PytestRemovedIn10Warning`` ("Class-scoped fixture defined
as instance method is deprecated") whenever a ``scope="class"`` fixture lives
in a test class as a plain instance method, and pytest 10 drops the support.
pytest only stays silent when the fixture is bound to the class itself
(``@classmethod``) or not bound at all (``@staticmethod``).

The check is a static AST scan of every ``*.py`` file under ``tests/``: fast,
deterministic, and independent of whether the offending fixtures' tests would
skip in the current environment (both known offenders need pixi + Sphinx).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
ALLOWED_DECORATORS = {"classmethod", "staticmethod"}


def _decorator_name(node: ast.expr) -> str:
    """Return the trailing dotted name of a decorator (``pytest.fixture`` -> ``fixture``)."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _is_class_scoped_fixture(node: ast.expr) -> bool:
    """True if ``node`` is ``@fixture(..., scope="class", ...)`` in any import form."""
    if not isinstance(node, ast.Call) or _decorator_name(node) != "fixture":
        return False
    return any(
        kw.arg == "scope" and isinstance(kw.value, ast.Constant) and kw.value.value == "class" for kw in node.keywords
    )


def find_instance_method_class_fixtures(source: str, label: str) -> list[str]:
    """List ``label::Class.method`` for every class-scoped fixture that is a plain instance method."""
    offenders: list[str] = []
    for cls in ast.walk(ast.parse(source)):
        if not isinstance(cls, ast.ClassDef):
            continue
        for item in cls.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = item.decorator_list
            if not any(_is_class_scoped_fixture(d) for d in decorators):
                continue
            if any(_decorator_name(d) in ALLOWED_DECORATORS for d in decorators):
                continue
            offenders.append(f"{label}::{cls.name}.{item.name} (line {item.lineno})")
    return offenders


@pytest.mark.smoke
class TestClassScopedFixtureHygiene:
    """REQ-TEST-HYGIENE-01: no ``scope="class"`` fixture is a plain instance method."""

    def test_no_class_scoped_fixture_is_an_instance_method(self):
        offenders: list[str] = []
        for path in sorted(TESTS_DIR.rglob("*.py")):
            label = str(path.relative_to(TESTS_DIR.parent))
            offenders.extend(find_instance_method_class_fixtures(path.read_text(), label))
        assert not offenders, (
            "Class-scoped fixtures defined as plain instance methods "
            "(PytestRemovedIn10Warning; use @classmethod or @staticmethod):\n  " + "\n  ".join(offenders)
        )

    def test_scanner_flags_instance_method_and_spares_classmethod_staticmethod(self):
        source = (
            "import pytest\n"
            "from pytest import fixture\n"
            "class TestX:\n"
            "    @pytest.fixture(scope='class')\n"
            "    def bad(self, tmp_path_factory):\n"
            "        return 1\n"
            "    @fixture(scope='class')\n"
            "    def bad_bare(self):\n"
            "        return 1\n"
            "    @pytest.fixture(scope='class')\n"
            "    @classmethod\n"
            "    def ok_cls(cls):\n"
            "        return 1\n"
            "    @pytest.fixture(scope='class')\n"
            "    @staticmethod\n"
            "    def ok_static(tmp_path_factory):\n"
            "        return 1\n"
            "    @pytest.fixture\n"
            "    def ok_function_scope(self):\n"
            "        return 1\n"
            "    @pytest.fixture(scope='module')\n"
            "    def ok_module_scope(self):\n"
            "        return 1\n"
        )
        found = find_instance_method_class_fixtures(source, "x.py")
        assert found == ["x.py::TestX.bad (line 5)", "x.py::TestX.bad_bare (line 8)"]
