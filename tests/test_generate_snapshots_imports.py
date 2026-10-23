"""Top-level imports in ``generate_snapshots.py`` (TASK-REFACTOR-BDRABCZUK-33C4A004C981).

- REQ-IMPORTS-BDRABCZUK-A2208E509D18: PLC0415, applied without the ``ruff.toml``
  per-file ignores, reports zero findings in ``generate_snapshots.py``.
- REQ-IMPORTS-BDRABCZUK-C3EB4D1AE34F: ruff, run with the repository ``ruff.toml``,
  reports PLC0415 for a function-level import placed in ``generate_snapshots.py``.
- REQ-IMPORTS-BDRABCZUK-C291DC0AF950: the module imports without error when
  ``anatomist.headless``, ``soma.aims`` and ``cortical_tiles.config`` are not importable.
- REQ-IMPORTS-BDRABCZUK-9667C41C7E31: without ``anatomist.headless``,
  ``generate_sulcal_graph_snapshot`` called with no Anatomist instance raises ImportError.
- REQ-IMPORTS-BDRABCZUK-5D10A3B8F60C: without ``anatomist.headless``,
  ``generate_tiles_snapshot`` raises ImportError.

The BrainVISA cases run in a child interpreter whose ``sys.modules`` blocks those
modules before ``generate_snapshots`` is imported, so guarded top-level imports
are exercised exactly as on a machine without BrainVISA.
"""

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke  # noqa: V107

REPO_ROOT = Path(__file__).resolve().parents[1]
RUFF_TOML = REPO_ROOT / "ruff.toml"
RULE = "PLC0415"
TARGET = "src/champollion_pipeline/generate_snapshots.py"
PROBE_SOURCE = 'def _probe():\n    import json\n\n    return json.dumps("")\n'
BLOCKED_MODULES = (
    "anatomist",
    "anatomist.headless",
    "soma",
    "soma.aims",
    "cortical_tiles",
    "cortical_tiles.config",
)


def _ruff() -> list[str]:
    """Return the argv prefix that invokes ruff, or skip when unavailable."""
    binary = shutil.which("ruff")
    if binary:
        return [binary]
    probe = subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("ruff not available")
    return [sys.executable, "-m", "ruff"]


def _run_without_brainvisa(body: str, tmp_path: Path) -> dict:
    """Run ``body`` in a child interpreter with the BrainVISA modules blocked.

    ``body`` sees ``gs`` (the imported module, or ``None``) and fills ``result``.
    Returns the ``result`` dict printed by the child as JSON.
    """
    script = textwrap.dedent(
        """
        import json
        import sys

        for name in {blocked!r}:
            sys.modules[name] = None

        result = {{}}
        try:
            import champollion_pipeline.generate_snapshots as gs
        except Exception as exc:
            gs = None
            result["import_error"] = f"{{type(exc).__name__}}: {{exc}}"
        """
    ).format(blocked=BLOCKED_MODULES)
    script += textwrap.dedent(body)
    script += "\nprint(json.dumps(result))\n"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"child interpreter failed:\n{proc.stdout}{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestSnapshotsImportsTopLevel:
    """REQ-IMPORTS-BDRABCZUK-A2208E509D18, -C3EB4D1AE34F."""

    def test_no_import_outside_top_level(self):
        ruff = _ruff()
        result = subprocess.run(
            [*ruff, "check", "--isolated", "--no-cache", "--select", RULE, "--output-format", "concise", TARGET],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        findings = [line for line in result.stdout.splitlines() if f" {RULE} " in line]
        assert not findings, (
            f"REQ-IMPORTS-BDRABCZUK-A2208E509D18: {len(findings)} {RULE} finding(s) in {TARGET}:\n"
            + "\n".join(findings)
        )
        assert result.returncode == 0, f"ruff exited {result.returncode}:\n{result.stdout}{result.stderr}"

    def test_ruff_config_flags_function_level_import(self):
        ruff = _ruff()
        result = subprocess.run(
            [
                *ruff,
                "check",
                "--no-cache",
                "--config",
                str(RUFF_TOML),
                "--output-format",
                "json",
                "--stdin-filename",
                TARGET,
                "-",
            ],
            cwd=REPO_ROOT,
            input=PROBE_SOURCE,
            capture_output=True,
            text=True,
        )
        codes = {finding["code"] for finding in json.loads(result.stdout or "[]")}
        assert RULE in codes, (
            f"REQ-IMPORTS-BDRABCZUK-C3EB4D1AE34F: ruff.toml does not report {RULE} for a "
            f"function-level import in {TARGET} (codes reported: {sorted(codes)})"
        )


class TestSnapshotsWithoutBrainvisa:
    """REQ-IMPORTS-BDRABCZUK-C291DC0AF950, -9667C41C7E31, -5D10A3B8F60C."""

    def test_module_imports_without_brainvisa(self, tmp_path):
        result = _run_without_brainvisa("", tmp_path)
        assert "import_error" not in result, (
            f"REQ-IMPORTS-BDRABCZUK-C291DC0AF950: importing generate_snapshots without BrainVISA failed: "
            f"{result['import_error']}"
        )

    def test_sulcal_snapshot_raises_import_error(self, tmp_path):
        body = """
        if gs is not None:
            try:
                gs.generate_sulcal_graph_snapshot("missing.arg", "out.png")
                result["raised"] = None
            except ImportError:
                result["raised"] = "ImportError"
            except Exception as exc:
                result["raised"] = f"{type(exc).__name__}: {exc}"
        """
        result = _run_without_brainvisa(body, tmp_path)
        assert "import_error" not in result, f"module import failed: {result['import_error']}"
        assert result["raised"] == "ImportError", (
            f"REQ-IMPORTS-BDRABCZUK-9667C41C7E31: expected ImportError, got {result['raised']}"
        )

    def test_tiles_snapshot_raises_import_error(self, tmp_path):
        body = """
        if gs is not None:
            try:
                gs.generate_tiles_snapshot("missing_crops", "out.png")
                result["raised"] = None
            except ImportError:
                result["raised"] = "ImportError"
            except Exception as exc:
                result["raised"] = f"{type(exc).__name__}: {exc}"
        """
        result = _run_without_brainvisa(body, tmp_path)
        assert "import_error" not in result, f"module import failed: {result['import_error']}"
        assert result["raised"] == "ImportError", (
            f"REQ-IMPORTS-BDRABCZUK-5D10A3B8F60C: expected ImportError, got {result['raised']}"
        )
