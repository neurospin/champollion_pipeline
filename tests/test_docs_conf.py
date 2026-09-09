"""Tests for REQ-DOCS-02 — Sphinx conf.py.

The artifact under test is ``docs/conf.py``: the requirement states that it
shall configure Sphinx with the ``furo`` theme and the ``myst_parser``,
``sphinx.ext.autodoc``, ``sphinx.ext.napoleon``, ``sphinx.ext.viewcode`` and
``sphinx.ext.intersphinx`` extensions, and shall add ``src/`` to ``sys.path``
so autodoc can import ``champollion_pipeline``.

Two complementary angles are covered:

* the *declared* configuration — ``conf.py`` is executed in an isolated
  namespace and its module-level settings are asserted directly, which pins
  down each named extension and the theme without needing Sphinx installed;
* the *effective* configuration — ``sphinx-build`` is run end to end through
  the ``docs`` pixi environment (the only one carrying the Sphinx toolchain)
  and must exit 0 with no ``ERROR`` lines.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
CONF_PY = DOCS_DIR / "conf.py"
SRC_DIR = PROJECT_ROOT / "src"
BUILD_DIR = DOCS_DIR / "_build" / "html"

REQUIRED_EXTENSIONS = (
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
)


def _missing_suffix(namespace: dict) -> str:
    """Explain a missing ``conf.py`` inline in an assertion message."""
    return f" ({CONF_PY} does not exist)" if namespace.get("_conf_py_missing") else ""


@pytest.fixture(scope="module")
def conf_namespace() -> dict:
    """Module-level namespace produced by executing ``docs/conf.py``.

    ``conf.py`` is a plain Python script, so executing it is the only faithful
    way to read its settings. It mutates ``sys.path`` by design, so the
    original path is restored afterwards to keep the rest of the suite clean.
    """
    if not CONF_PY.is_file():
        # Returning an empty namespace (rather than erroring out here) keeps
        # each test's own assertion as the reported failure, so a missing
        # conf.py reads as a plain red test rather than a fixture error.
        return {"_conf_py_missing": True, "_sys_path_after_exec": []}

    original_sys_path = list(sys.path)
    namespace: dict = {"__file__": str(CONF_PY), "__name__": "docs_conf"}
    try:
        exec(compile(CONF_PY.read_text(), str(CONF_PY), "exec"), namespace)  # noqa: S102
        namespace["_sys_path_after_exec"] = list(sys.path)
    finally:
        sys.path[:] = original_sys_path
    return namespace


@pytest.mark.smoke
class TestDocsConfDeclaration:
    """REQ-DOCS-02: docs/conf.py declares the required theme, extensions and path."""

    def test_conf_py_exists(self):
        """``docs/conf.py`` is present at the documented location."""
        assert CONF_PY.is_file(), f"{CONF_PY} does not exist"

    def test_html_theme_is_furo(self, conf_namespace):
        """The HTML theme is ``furo``."""
        assert conf_namespace.get("html_theme") == "furo", (
            f"expected html_theme == 'furo', found "
            f"{conf_namespace.get('html_theme')!r}{_missing_suffix(conf_namespace)}"
        )

    @pytest.mark.parametrize("extension", REQUIRED_EXTENSIONS)
    def test_extension_is_enabled(self, conf_namespace, extension):
        """Each required Sphinx extension is listed in ``extensions``."""
        extensions = conf_namespace.get("extensions", [])
        assert extension in extensions, (
            f"conf.py does not enable {extension!r}; found {sorted(extensions)}{_missing_suffix(conf_namespace)}"
        )

    def test_src_dir_added_to_sys_path(self, conf_namespace):
        """Executing conf.py puts the project's ``src/`` on ``sys.path``."""
        paths_after_exec = {Path(entry).resolve() for entry in conf_namespace.get("_sys_path_after_exec", [])}
        assert SRC_DIR.resolve() in paths_after_exec, (
            f"conf.py did not add {SRC_DIR} to sys.path, so autodoc cannot "
            f"import champollion_pipeline{_missing_suffix(conf_namespace)}"
        )


@pytest.mark.smoke
class TestDocsConfBuild:
    """REQ-DOCS-02: the configuration actually drives a clean Sphinx build."""

    @pytest.fixture(scope="class")
    def sphinx_build_result(self) -> subprocess.CompletedProcess:
        """Run ``sphinx-build`` end to end inside the ``docs`` pixi environment.

        The Sphinx toolchain (``sphinx``, ``furo``, ``myst-parser``) lives only
        in the ``docs`` environment declared by REQ-DOCS-01, so a bare
        ``sphinx-build`` would pick up an unrelated interpreter lacking the
        theme and fail for the wrong reason.
        """
        if shutil.which("pixi") is None:
            pytest.skip("pixi is not on PATH; cannot reach the docs environment")

        # pixi < 0.77.0 cannot parse the platforms table syntax introduced in PR#7
        pixi_ver_str = subprocess.run(["pixi", "--version"], capture_output=True, text=True).stdout.strip()
        pixi_ver_tuple = tuple(int(x) for x in pixi_ver_str.split()[-1].split(".") if x.isdigit())
        if pixi_ver_tuple < (0, 77, 0):
            pytest.skip(f"{pixi_ver_str} < 0.77.0: platforms table syntax not supported")

        if BUILD_DIR.exists():
            shutil.rmtree(BUILD_DIR)

        # Sphinx localises its diagnostics; on a French locale the word
        # "ERROR" never appears and the error-line scan below would silently
        # pass on a broken build. Pin the subprocess to the C locale.
        env = {**os.environ, "LC_ALL": "C", "LANG": "C", "LANGUAGE": "en"}

        return subprocess.run(
            ["pixi", "run", "-e", "docs", "sphinx-build", "-b", "html", "docs", "docs/_build/html"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            env=env,
        )

    def test_sphinx_build_exits_zero(self, sphinx_build_result):
        """``sphinx-build`` completes successfully."""
        assert sphinx_build_result.returncode == 0, (
            f"sphinx-build exited {sphinx_build_result.returncode}\n"
            f"--- stdout ---\n{sphinx_build_result.stdout}\n"
            f"--- stderr ---\n{sphinx_build_result.stderr}"
        )

    def test_sphinx_build_reports_no_errors(self, sphinx_build_result):
        """No error line appears in the build output.

        Sphinx signals problems either as ``ERROR:``-prefixed lines during the
        build or as a ``Configuration error`` banner when ``conf.py`` cannot be
        loaded at all; both count.
        """
        combined = sphinx_build_result.stdout + sphinx_build_result.stderr
        error_lines = [line for line in combined.splitlines() if "ERROR" in line or "Configuration error" in line]
        assert not error_lines, "sphinx-build reported errors:\n" + "\n".join(error_lines)

    def test_build_produces_index_html(self, sphinx_build_result):
        """The build writes ``docs/_build/html/index.html``."""
        index_html = BUILD_DIR / "index.html"
        assert index_html.is_file(), (
            f"{index_html} was not produced by sphinx-build\n--- stdout ---\n{sphinx_build_result.stdout}"
        )
