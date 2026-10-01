"""Tests for REQ-DOCS-13 — the strict documentation build needs no network.

``docs/conf.py`` used to fetch ``https://docs.python.org/3/objects.inv``
through ``sphinx.ext.intersphinx`` on every build; when that host answered
503 (2026-10-01) the fetch warning became an error under ``-W`` and the
whole suite went red for a reason unrelated to the documentation (TASK-134).

Network loss is simulated deterministically, independent of the host's real
connectivity: the build subprocess gets every proxy variable pointed at a
127.0.0.1 port that this test holds *bound but not listening*, so each
connection attempt is refused. Sphinx fetches through ``requests``, which
honours these variables. No test-only opt-in switch is set — the docs must
build offline on their own.

The build runs against a temporary copy of ``docs/`` (with ``src`` symlinked
beside it, since ``conf.py`` locates the package via ``../src``) so the real
``docs/_build`` tree is never touched, and so the regression guard can inject
a broken reference without editing a tracked file.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
SRC_DIR = PROJECT_ROOT / "src"

PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
NO_PROXY_VARS = ("NO_PROXY", "no_proxy")

BROKEN_TARGET = "req-docs-13-nonexistent-page"


def _require_docs_toolchain() -> None:
    """Skip unless the ``docs`` pixi environment is reachable (mirrors REQ-DOCS-03's test)."""
    if shutil.which("pixi") is None:
        pytest.skip("pixi is not on PATH; cannot reach the docs environment")
    pixi_ver_str = subprocess.run(["pixi", "--version"], capture_output=True, text=True).stdout.strip()
    pixi_ver_tuple = tuple(int(x) for x in pixi_ver_str.split()[-1].split(".") if x.isdigit())
    if pixi_ver_tuple < (0, 77, 0):
        pytest.skip(f"{pixi_ver_str} < 0.77.0: platforms table syntax not supported")


@pytest.fixture(scope="module")
def dead_proxy_url() -> Iterator[str]:
    """URL of a local port that refuses every connection for the module's lifetime.

    The socket is bound (so no other process can take the port) but never
    ``listen()``-ed, so the kernel answers each connect with ECONNREFUSED.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        sock.close()


@pytest.fixture(scope="module")
def offline_env(dead_proxy_url: str) -> dict[str, str]:
    """Process environment in which no outbound HTTP(S) request can succeed."""
    env = {k: v for k, v in os.environ.items() if k not in NO_PROXY_VARS}
    env.update({var: dead_proxy_url for var in PROXY_VARS})
    # Sphinx localises its diagnostics; keep failure messages readable.
    env.update({"LC_ALL": "C", "LANG": "C", "LANGUAGE": "en"})
    return env


def _copy_docs(dest_root: Path) -> Path:
    """Copy ``docs/`` (minus build output) under ``dest_root``; symlink ``src`` beside it."""
    docs_copy = dest_root / "docs"
    shutil.copytree(DOCS_DIR, docs_copy, ignore=shutil.ignore_patterns("_build"))
    (dest_root / "src").symlink_to(SRC_DIR, target_is_directory=True)
    return docs_copy


def _strict_build(docs_src: Path, out_dir: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    """Run the REQ-DOCS-13 strict build command inside the ``docs`` pixi environment."""
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
            str(docs_src),
            str(out_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env=env,
    )


def _report(result: subprocess.CompletedProcess) -> str:
    return f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"


@pytest.mark.smoke
class TestDocsStrictBuildOffline:
    """REQ-DOCS-13: ``sphinx-build -W`` succeeds when every HTTP(S) connection fails."""

    def test_strict_build_exits_zero_without_network(self, tmp_path, offline_env):
        """The unmodified docs build strictly with no reachable network."""
        _require_docs_toolchain()
        docs_copy = _copy_docs(tmp_path)

        result = _strict_build(docs_copy, tmp_path / "html", offline_env)

        assert result.returncode == 0, (
            f"strict docs build exited {result.returncode} with all HTTP(S) connections refused "
            f"(proxy {offline_env['HTTPS_PROXY']}); the build must not depend on network access\n"
            f"{_report(result)}"
        )

    def test_strict_build_still_fails_on_broken_reference_without_network(self, tmp_path, offline_env):
        """Regression guard (REQ-DOCS-03): offline success must not come from muting reference warnings."""
        _require_docs_toolchain()
        docs_copy = _copy_docs(tmp_path)
        index_md = docs_copy / "index.md"
        index_md.write_text(
            index_md.read_text(encoding="utf-8") + f"\n\nSee {{doc}}`{BROKEN_TARGET}`.\n",
            encoding="utf-8",
        )

        result = _strict_build(docs_copy, tmp_path / "html", offline_env)

        assert result.returncode != 0, (
            f"strict docs build exited 0 despite a {{doc}} reference to missing page {BROKEN_TARGET!r}\n"
            f"{_report(result)}"
        )
        assert BROKEN_TARGET in result.stdout + result.stderr, (
            f"strict docs build failed, but not because of the broken reference to {BROKEN_TARGET!r}\n{_report(result)}"
        )
