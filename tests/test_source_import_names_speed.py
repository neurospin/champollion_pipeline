"""Tests for REQ-TEST-SPEED-01 — the deep_folding import guard scans in linear time.

The REQ-IMPORT-01 guard (``tests/test_source_import_names.py``) is static: it
parses each ``src/`` module with ``ast`` and never imports it. It was still the
slowest file in the suite (~28 s) because its scan re-split the whole source
for every AST node, which is quadratic in module size.

These tests call the guard's own test function on generated modules, so they
pin its observable behaviour — elapsed time and verdict — without depending on
how its helpers are structured.
"""

from __future__ import annotations

import time

import pytest

import tests.test_source_import_names as import_guard

STATEMENT_COUNT = 400
TIME_BUDGET_S = 1.0


def _clean_module_source(statement_count: int) -> str:
    """A valid module with one stdlib import and ``statement_count`` assignments."""
    body = "".join(f"value_{i} = os.path.join('root', str({i}))\n" for i in range(statement_count))
    return "import os\n" + body


@pytest.mark.smoke
class TestImportGuardSpeed:
    """REQ-TEST-SPEED-01."""

    def test_guard_returns_within_one_second_on_400_statement_module(self, tmp_path):
        """A 400-statement clean module is classified in under 1 s."""
        module = tmp_path / "large_clean_module.py"
        module.write_text(_clean_module_source(STATEMENT_COUNT), encoding="utf-8")

        start = time.perf_counter()
        import_guard.test_no_source_module_imports_the_renamed_deep_folding_package(module)
        elapsed = time.perf_counter() - start

        assert elapsed < TIME_BUDGET_S, (
            f"import guard took {elapsed:.2f} s on a {STATEMENT_COUNT}-statement module "
            f"(budget {TIME_BUDGET_S:.0f} s): the scan is super-linear in module size"
        )

    def test_guard_still_flags_deep_folding_import_without_executing_module(self, tmp_path):
        """Regression guard: a speed fix must keep REQ-IMPORT-01's verdict and stay static.

        The module ends with a top-level ``raise``, so any import/exec of it
        would surface as ``RuntimeError`` instead of the guard's own
        ``AssertionError`` naming the offending line.
        """
        source = (
            _clean_module_source(50)
            + "from deep_folding.brainvisa import utils\n"
            + "raise RuntimeError('module was executed')\n"
        )
        module = tmp_path / "offending_module.py"
        module.write_text(source, encoding="utf-8")
        offending_line = 52  # "import os" + 50 assignments, then the import

        with pytest.raises(AssertionError) as excinfo:
            import_guard.test_no_source_module_imports_the_renamed_deep_folding_package(module)

        assert f"offending_module.py:{offending_line}" in str(excinfo.value)
