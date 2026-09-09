"""Tests for REQ-DOCS-03 — all doc pages.

The artifacts under test are the documentation pages themselves: the
requirement states that ``docs/index.md`` shall define a root toctree with a
"User Guide" section (``installation``, ``usage``, ``troubleshooting``) and a
"Reference" section (``internals``, ``api``), and that each of the five
referenced pages shall exist and render without broken references in the
Sphinx build.

Three complementary angles are covered:

* the *sources* — each page exists, ``index.md`` wires it into a toctree, and
  ``api.md`` carries at least one ``automodule::`` directive (without one, the
  Reference page has no autodoc content to render);
* the *strict build* — ``sphinx-build -W --keep-going`` turns every warning
  into an error, so an orphaned page or a toctree entry pointing at a missing
  document fails the build rather than passing silently;
* the *rendered output* — ``docs/_build/html/api.html`` must mention
  ``champollion_pipeline``, which is only true if autodoc actually imported
  and documented the package.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
INDEX_MD = DOCS_DIR / "index.md"
API_MD = DOCS_DIR / "api.md"
BUILD_DIR = DOCS_DIR / "_build" / "html"
API_HTML = BUILD_DIR / "api.html"

USER_GUIDE_PAGES = ("installation", "usage", "troubleshooting")
REFERENCE_PAGES = ("internals", "api")
REQUIRED_PAGES = USER_GUIDE_PAGES + REFERENCE_PAGES


def _read(path: Path) -> str:
    """Contents of ``path``, or an empty string when it does not exist yet.

    Returning empty (rather than raising) keeps each test's own assertion as
    the reported failure, so a missing page reads as a plain red test rather
    than an unrelated ``FileNotFoundError``.
    """
    return path.read_text(encoding="utf-8") if path.is_file() else ""


@pytest.mark.smoke
class TestDocPagesExist:
    """REQ-DOCS-03: every page referenced by the root toctree is present."""

    @pytest.mark.parametrize("page", REQUIRED_PAGES)
    def test_page_file_exists(self, page):
        """``docs/<page>.md`` exists at the documented location."""
        page_path = DOCS_DIR / f"{page}.md"
        assert page_path.is_file(), f"{page_path} does not exist"


@pytest.mark.smoke
class TestIndexToctree:
    """REQ-DOCS-03: docs/index.md declares the root toctree and its sections."""

    def test_index_md_exists(self):
        """``docs/index.md`` is present."""
        assert INDEX_MD.is_file(), f"{INDEX_MD} does not exist"

    def test_index_declares_a_toctree(self):
        """``index.md`` contains a MyST ``toctree`` directive block."""
        assert "{toctree}" in _read(INDEX_MD), f"{INDEX_MD} declares no ```{{toctree}}``` block"

    def test_index_declares_user_guide_section(self):
        """A toctree section is captioned "User Guide"."""
        assert "User Guide" in _read(INDEX_MD), f'{INDEX_MD} has no toctree section captioned "User Guide"'

    def test_index_declares_reference_section(self):
        """A toctree section is captioned "Reference"."""
        assert "Reference" in _read(INDEX_MD), f'{INDEX_MD} has no toctree section captioned "Reference"'

    @pytest.mark.parametrize("page", REQUIRED_PAGES)
    def test_index_toctree_lists_page(self, page):
        """Each of the five pages is listed as a toctree entry."""
        content = _read(INDEX_MD)
        entries = {line.strip() for line in content.splitlines()}
        assert page in entries, f"{INDEX_MD} has no toctree entry for {page!r}"


@pytest.mark.smoke
class TestApiPageUsesAutodoc:
    """REQ-DOCS-03: the api page pulls its content from the package itself."""

    def test_api_md_contains_automodule_directive(self):
        """``docs/api.md`` carries at least one ``automodule::`` directive."""
        assert "automodule::" in _read(API_MD), (
            f"{API_MD} contains no 'automodule::' directive, so the Reference page has no autodoc content"
        )


@pytest.mark.smoke
class TestDocsStrictBuild:
    """REQ-DOCS-03: the page set builds strictly and renders autodoc output."""

    @pytest.fixture(scope="class")
    def strict_build_result(self) -> subprocess.CompletedProcess:
        """Run the requirement's verify command inside the ``docs`` pixi environment.

        The Sphinx toolchain lives only in the ``docs`` environment declared by
        REQ-DOCS-01, so a bare ``sphinx-build`` would resolve to an interpreter
        lacking ``furo`` and ``myst_parser`` and fail for the wrong reason.
        ``-W`` promotes warnings — including "document isn't included in any
        toctree" and unresolved references — to errors; ``--keep-going`` makes
        the build report all of them instead of stopping at the first.
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

        # Sphinx localises its diagnostics; pin the subprocess to the C locale
        # so the captured output stays readable in failure messages.
        env = {**os.environ, "LC_ALL": "C", "LANG": "C", "LANGUAGE": "en"}

        return subprocess.run(
            [
                "pixi",
                "run",
                "-e",
                "docs",
                "sphinx-build",
                "-W",
                "--keep-going",
                "-b",
                "html",
                "docs",
                "docs/_build/html",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            env=env,
        )

    def test_strict_build_exits_zero(self, strict_build_result):
        """The strict build completes with no warnings or broken references."""
        assert strict_build_result.returncode == 0, (
            f"sphinx-build -W --keep-going exited {strict_build_result.returncode}\n"
            f"--- stdout ---\n{strict_build_result.stdout}\n"
            f"--- stderr ---\n{strict_build_result.stderr}"
        )

    def test_build_produces_api_html(self, strict_build_result):
        """The build writes ``docs/_build/html/api.html``."""
        assert API_HTML.is_file(), (
            f"{API_HTML} was not produced by sphinx-build\n--- stdout ---\n{strict_build_result.stdout}"
        )

    def test_api_html_contains_autodoc_output(self, strict_build_result):
        """``api.html`` mentions ``champollion_pipeline`` — proof autodoc ran.

        Autodoc only emits the package name into the page after successfully
        importing it, so its presence distinguishes a rendered API reference
        from a page that merely declared the directive.
        """
        rendered = _read(API_HTML)
        assert "champollion_pipeline" in rendered, (
            f"{API_HTML} does not mention 'champollion_pipeline'; autodoc produced no output\n"
            f"--- stdout ---\n{strict_build_result.stdout}"
        )


# TASK-011
@pytest.mark.smoke
class TestSetupInToctree:
    """REQ-WIZARD-05: setup page appears before installation in User Guide toctree."""

    def test_setup_page_in_toctree(self):
        index_content = (DOCS_DIR / "index.md").read_text()
        assert "setup" in index_content, "setup not found in docs/index.md"

    def test_setup_before_installation_in_toctree(self):
        index_content = (DOCS_DIR / "index.md").read_text()
        setup_pos = index_content.find("setup")
        install_pos = index_content.find("installation")
        assert setup_pos != -1, "setup not found in index.md"
        assert install_pos != -1, "installation not found in index.md"
        assert setup_pos < install_pos, f"setup ({setup_pos}) must appear before installation ({install_pos})"
