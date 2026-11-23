#!/usr/bin/env python3
"""
Import-time isolation of the modules the pipeline command loads before any stage runs.

One rule over a set of modules: champollion_pipeline.config, champollion_pipeline.cli,
champollion_pipeline.pipeline_stages, main.py and the champollion_pipeline module that assigns
DEFAULT_CHECKPOINT_EPOCH. Each is imported alone in a
new Python interpreter (conftest.py stubs ``soma`` in this process), and the test lists the modules
of the toolchain packages (soma, cortical_tiles, champollion) that import added to ``sys.modules``.

main.py sits at the repo root, which the probe puts on its path, so it is imported as ``main``:
the ``if __name__ == "__main__"`` block does not run.

The set is finite and each member costs one interpreter, so the property is checked on every
member by parametrisation rather than on a Hypothesis sample of it.

Requirement: the invariant REQ-LAZYIMPORT-BDRABCZUK-20C492F675F0, generalising
REQ-LAZYIMPORT-BDRABCZUK-BB2CC03C17D6, REQ-CLICMD-BDRABCZUK-8D525167B449, REQ-LAZYIMPORT-BDRABCZUK-C85B2829CDA2 and
REQ-LAZYIMPORT-BDRABCZUK-82B7DF9D66CC (candidate INV-2E5420ADFA59); pipeline_stages joined the set with
TASK-CODE-BDRABCZUK-D457E2861B14 (cli.py split; REQ-CLISPLIT-BDRABCZUK-38095C750F40 states it for that member).
"""

import pytest

from tests.test_lazy_import_checkpoint_epoch import _constant_module_name, _toolchain_modules_added_by_import

# Member of the set -> how to get its importable module name (the constant's module is found by parsing).
STARTUP_MODULES = {
    "config": lambda: "champollion_pipeline.config",
    "cli": lambda: "champollion_pipeline.cli",
    "pipeline_stages": lambda: "champollion_pipeline.pipeline_stages",
    "main.py": lambda: "main",
    "checkpoint_constant_module": _constant_module_name,
}


@pytest.mark.parametrize("member", sorted(STARTUP_MODULES))
def test_startup_module_imports_no_toolchain(member):
    """Importing the member alone adds no soma, cortical_tiles or champollion module to sys.modules."""
    module_name = STARTUP_MODULES[member]()
    assert _toolchain_modules_added_by_import(module_name) == [], module_name
