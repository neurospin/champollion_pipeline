"""Tests for ``scripts/e2e/env_key.sh``, the key that names the e2e env SIF.

TASK-E2ECI-BDRABCZUK-DF3FEC570B69 (epic TASK-E2ECI-BDRABCZUK-F3B0138F58F3).
The env SIF (the pixi environment image, no pipeline code) is rebuilt only when
its key changes, so the key must cover exactly what shapes that environment:

- ``pixi.lock``;
- the pixi version pin, ``E2E_PIXI_VERSION`` in ``scripts/e2e/e2e.env``;
- the champollion_utils pin, the ``champollion-utils`` dependency specifier in
  ``pyproject.toml`` (champollion_utils is not in ``pixi.lock``: the pixi tasks
  clone it as an unpinned sibling, so the pinned git reference is the input);
- the submodule dependency manifests: ``external/champollion_V1/setup.py``,
  ``external/cortical_tiles/setup.py`` and
  ``external/cortical_tiles/cortical_tiles/info.py`` (the submodules have no
  pyproject.toml; cortical_tiles declares REQUIRES in info.py);
- the env image recipe, ``.github/Dockerfile.base`` (user decision 2026-10-28:
  a base image or system package change must rebuild the env SIF).

CLI contract: ``bash scripts/e2e/env_key.sh [--sif-name] [ROOT]``. Every test
builds a fake repository tree under ``tmp_path`` and passes it as ROOT (or copies
the script into it), so no test reads the real pixi.lock or needs the
submodules checked out.

Requirements (one test or more each):
- REQ-E2ECI-BDRABCZUK-3A892AE6F832 one line of 12 lowercase hex characters, exit 0
- REQ-E2ECI-BDRABCZUK-4DC4A1BE0402 the key depends only on the seven listed inputs
  (supersedes REQ-E2ECI-BDRABCZUK-D490B93DA736, six inputs)
- REQ-E2ECI-BDRABCZUK-5D8812A38665 a pixi.lock change changes the key
- REQ-E2ECI-BDRABCZUK-CCC13BA1EA59 an E2E_PIXI_VERSION change changes the key
- REQ-E2ECI-BDRABCZUK-164DB7DC9CA3 a champollion-utils pin change changes the key
- REQ-E2ECI-BDRABCZUK-9D9C0E6FF09D a submodule manifest change changes the key
- REQ-E2ECI-BDRABCZUK-435366BF2FB2 a .github/Dockerfile.base change changes the key
- REQ-E2ECI-BDRABCZUK-D768F564C30D an absent input is refused (non-zero, no key, named)
- REQ-E2ECI-BDRABCZUK-0A1EE8FE1ABF omitted ROOT means the script's own repository
- REQ-E2ECI-BDRABCZUK-14583CF7AF14 --sif-name fills E2E_ENV_SIF_NAME from ROOT's e2e.env
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "e2e" / "env_key.sh"
BASH = shutil.which("bash")

KEY_LINE = re.compile(r"[0-9a-f]{12}\n")
UTILS_PIN = "champollion-utils @ git+https://github.com/neurospin/champollion_utils.git@v0.3.0"

PIXI_LOCK = "version: 6\nenvironments:\n  default:\n    packages:\n      - conda: numpy-2.1.0.conda\n"
PYPROJECT = f"""[project]
name = "champollion-pipeline"
version = "0.2.32"
dependencies = [
    "{UTILS_PIN}",
    "scipy>=1.10",
]

[tool.ruff]
line-length = 120
"""
E2E_ENV = """# fake settings file
E2E_ROSETTE_ROOT=/fake/rosette
E2E_ENV_SIF_NAME="champollion-env-{key12}.sif"
E2E_ENV_SIF_KEEP=3
E2E_PIXI_VERSION=0.80.0
"""
V1_SETUP = "from setuptools import setup\nsetup(name='champollion_V1', install_requires=['torch'])\n"
CT_SETUP = "from setuptools import setup\nsetup(name='cortical_tiles')\n"
CT_INFO = "__version__ = '1.0.0'\nREQUIRES = ['six', 'numpy']\n"
DOCKERFILE_BASE = "FROM ubuntu:22.04\nRUN apt-get install -y curl git\nARG PIXI_VERSION\n"

FILES = {
    "pixi.lock": PIXI_LOCK,
    "pixi.toml": '[workspace]\nname = "champollion"\nrequires-pixi = ">=0.80.0"\n',
    "pyproject.toml": PYPROJECT,
    "scripts/e2e/e2e.env": E2E_ENV,
    "external/champollion_V1/setup.py": V1_SETUP,
    "external/cortical_tiles/setup.py": CT_SETUP,
    "external/cortical_tiles/cortical_tiles/info.py": CT_INFO,
    ".github/Dockerfile.base": DOCKERFILE_BASE,
    "external/champollion_V1/champollion/model.py": "LAYERS = 4\n",
    "external/cortical_tiles/cortical_tiles/crop.py": "SIZE = 2\n",
    "src/champollion_pipeline/stage.py": "VALUE = 1\n",
    "README.md": "# fake\n",
}

pytestmark = pytest.mark.skipif(BASH is None, reason="bash executable not available")  # noqa: V107


def _make_root(base: Path) -> Path:
    """Write the fake repository tree under ``base`` and return it."""
    for rel, text in FILES.items():
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return base


def _clean_env(home: Path) -> dict[str, str]:
    """Only HOME and a PATH that finds bash, coreutils and this test's python3."""
    path = os.pathsep.join([str(Path(sys.executable).parent), "/usr/bin", "/bin"])
    return {"HOME": str(home), "PATH": path}


def _run(*args: str, script: Path = SCRIPT, cwd: Path | None = None) -> subprocess.CompletedProcess:
    assert SCRIPT.is_file(), f"{SCRIPT.relative_to(REPO)} is missing"
    return subprocess.run(
        [BASH, str(script), *args],
        capture_output=True,
        text=True,
        env=_clean_env(Path(tempfile.gettempdir())),
        cwd=cwd,
        timeout=60,
    )


def _key(root: Path) -> str:
    """The key printed for ``root``; fails the test on a non-zero exit."""
    result = _run(str(root))
    assert result.returncode == 0, f"exit {result.returncode}: {result.stderr}"
    return result.stdout.strip()


def _edit(root: Path, rel: str, old: str, new: str) -> None:
    path = root / rel
    text = path.read_text()
    assert text.count(old) == 1, f"{old!r} not unique in {rel}"
    path.write_text(text.replace(old, new))


# --- REQ-E2ECI-BDRABCZUK-3A892AE6F832: output format -----------------------------


def test_key_is_one_line_of_12_lowercase_hex(tmp_path):
    root = _make_root(tmp_path / "repo")
    result = _run(str(root))
    assert result.returncode == 0, result.stderr
    assert KEY_LINE.fullmatch(result.stdout), f"stdout is not one line of 12 lowercase hex: {result.stdout!r}"


# --- REQ-E2ECI-BDRABCZUK-4DC4A1BE0402: the key depends only on the inputs --------


def test_identical_inputs_give_identical_key(tmp_path):
    first = _make_root(tmp_path / "first")
    second = _make_root(tmp_path / "elsewhere" / "second")
    key = _key(first)
    assert _key(first) == key, "two runs on one tree gave different keys"
    assert _key(second) == key, "the same inputs under another directory gave a different key"


def _bump_version(root: Path) -> None:
    _edit(root, "pyproject.toml", 'version = "0.2.32"', 'version = "0.2.33"')


def _other_dependency(root: Path) -> None:
    _edit(root, "pyproject.toml", "scipy>=1.10", "scipy>=1.11")


def _ruff_setting(root: Path) -> None:
    _edit(root, "pyproject.toml", "line-length = 120", "line-length = 100")


def _pixi_toml(root: Path) -> None:
    (root / "pixi.toml").write_text((root / "pixi.toml").read_text() + '\n[tasks]\nnew = "echo"\n')


def _other_setting(root: Path) -> None:
    _edit(root, "scripts/e2e/e2e.env", "E2E_ENV_SIF_KEEP=3", "E2E_ENV_SIF_KEEP=4")


def _pipeline_source(root: Path) -> None:
    _edit(root, "src/champollion_pipeline/stage.py", "VALUE = 1", "VALUE = 2")


def _submodule_source(root: Path) -> None:
    _edit(root, "external/champollion_V1/champollion/model.py", "LAYERS = 4", "LAYERS = 5")
    _edit(root, "external/cortical_tiles/cortical_tiles/crop.py", "SIZE = 2", "SIZE = 3")


def _new_file(root: Path) -> None:
    (root / "src" / "champollion_pipeline" / "added.py").write_text("ADDED = True\n")


def _touch_inputs(root: Path) -> None:
    for rel in ("pixi.lock", "external/champollion_V1/setup.py"):
        os.utime(root / rel, (1_000_000_000, 1_000_000_000))


OUTSIDE_CHANGES: dict[str, Callable[[Path], None]] = {
    "pyproject-version": _bump_version,
    "pyproject-other-dependency": _other_dependency,
    "pyproject-tool-setting": _ruff_setting,
    "pixi-toml-task": _pixi_toml,
    "e2e-env-other-key": _other_setting,
    "pipeline-source": _pipeline_source,
    "submodule-source": _submodule_source,
    "new-source-file": _new_file,
    "input-mtime-only": _touch_inputs,
}


@pytest.mark.parametrize("change", list(OUTSIDE_CHANGES), ids=list(OUTSIDE_CHANGES))
def test_key_ignores_change_outside_inputs(tmp_path, change):
    root = _make_root(tmp_path / "repo")
    before = _key(root)
    OUTSIDE_CHANGES[change](root)
    assert _key(root) == before, f"a change outside the env key inputs ({change}) changed the key"


# --- REQ-E2ECI-BDRABCZUK-5D8812A38665: pixi.lock ---------------------------------


def test_pixi_lock_change_changes_key(tmp_path):
    root = _make_root(tmp_path / "repo")
    before = _key(root)
    _edit(root, "pixi.lock", "numpy-2.1.0", "numpy-2.1.1")
    assert _key(root) != before, "a pixi.lock change left the key unchanged"


RAW_INPUTS = (
    "external/champollion_V1/setup.py",
    "external/cortical_tiles/setup.py",
    "external/cortical_tiles/cortical_tiles/info.py",
    ".github/Dockerfile.base",
)


@pytest.mark.parametrize(
    "other", RAW_INPUTS, ids=["champollion_V1-setup", "cortical_tiles-setup", "cortical_tiles-info", "dockerfile-base"]
)
@pytest.mark.parametrize("edge", ["tail-to-head", "head-to-tail"])
def test_bytes_moved_between_inputs_change_key(tmp_path, other, edge):
    """Moving one pixi.lock line into another input changes the key, whatever their hashing order."""
    root = _make_root(tmp_path / "repo")
    before = _key(root)
    lines = PIXI_LOCK.splitlines(keepends=True)
    target = root / other
    if edge == "tail-to-head":
        (root / "pixi.lock").write_text("".join(lines[:-1]))
        target.write_text(lines[-1] + target.read_text())
    else:
        (root / "pixi.lock").write_text("".join(lines[1:]))
        target.write_text(target.read_text() + lines[0])
    assert _key(root) != before, f"a pixi.lock line moved into {other} ({edge}) left the key unchanged"


# --- REQ-E2ECI-BDRABCZUK-CCC13BA1EA59: pixi version pin -------------------------


def test_pixi_version_pin_change_changes_key(tmp_path):
    root = _make_root(tmp_path / "repo")
    before = _key(root)
    _edit(root, "scripts/e2e/e2e.env", "E2E_PIXI_VERSION=0.80.0", "E2E_PIXI_VERSION=0.81.0")
    assert _key(root) != before, "an E2E_PIXI_VERSION change left the key unchanged"


# --- REQ-E2ECI-BDRABCZUK-164DB7DC9CA3: champollion_utils pin --------------------


@pytest.mark.parametrize(
    "new_ref",
    ["v0.3.1", "4f2c9a1e0b7d3c5a6e8f9012ab34cd56ef7890ab"],
    ids=["tag", "commit"],
)
def test_utils_pin_change_changes_key(tmp_path, new_ref):
    root = _make_root(tmp_path / "repo")
    before = _key(root)
    _edit(root, "pyproject.toml", "champollion_utils.git@v0.3.0", f"champollion_utils.git@{new_ref}")
    assert _key(root) != before, f"pinning champollion-utils to {new_ref} left the key unchanged"


# --- REQ-E2ECI-BDRABCZUK-9D9C0E6FF09D: submodule dependency manifests -----------

MANIFEST_EDITS = {
    "champollion_V1-setup": ("external/champollion_V1/setup.py", "['torch']", "['torch', 'pandas']"),
    "cortical_tiles-setup": ("external/cortical_tiles/setup.py", "name='cortical_tiles'", "name='cortical-tiles'"),
    "cortical_tiles-info": ("external/cortical_tiles/cortical_tiles/info.py", "'numpy']", "'numpy', 'scipy']"),
}


@pytest.mark.parametrize("manifest", list(MANIFEST_EDITS), ids=list(MANIFEST_EDITS))
def test_submodule_manifest_change_changes_key(tmp_path, manifest):
    root = _make_root(tmp_path / "repo")
    before = _key(root)
    _edit(root, *MANIFEST_EDITS[manifest])
    assert _key(root) != before, f"a change in {MANIFEST_EDITS[manifest][0]} left the key unchanged"


# --- REQ-E2ECI-BDRABCZUK-435366BF2FB2: env image recipe --------------------------


def test_dockerfile_base_change_changes_key(tmp_path):
    root = _make_root(tmp_path / "repo")
    before = _key(root)
    _edit(root, ".github/Dockerfile.base", "FROM ubuntu:22.04", "FROM ubuntu:24.04")
    assert _key(root) != before, "a .github/Dockerfile.base change left the key unchanged"


# --- REQ-E2ECI-BDRABCZUK-D768F564C30D: absent input -----------------------------


def _drop_pixi_version(root: Path) -> None:
    _edit(root, "scripts/e2e/e2e.env", "E2E_PIXI_VERSION=0.80.0\n", "")


def _drop_utils_pin(root: Path) -> None:
    _edit(root, "pyproject.toml", f'    "{UTILS_PIN}",\n', "")


def _remover(rel: str) -> Callable[[Path], None]:
    return lambda root: (root / rel).unlink()


ABSENT_INPUTS: dict[str, tuple[Callable[[Path], None], str]] = {
    "pixi.lock": (_remover("pixi.lock"), "pixi.lock"),
    "e2e.env": (_remover("scripts/e2e/e2e.env"), "e2e.env"),
    "E2E_PIXI_VERSION": (_drop_pixi_version, "E2E_PIXI_VERSION"),
    "pyproject.toml": (_remover("pyproject.toml"), "pyproject.toml"),
    "champollion-utils": (_drop_utils_pin, "champollion-utils"),
    "champollion_V1-setup": (_remover("external/champollion_V1/setup.py"), "external/champollion_V1/setup.py"),
    "cortical_tiles-setup": (_remover("external/cortical_tiles/setup.py"), "external/cortical_tiles/setup.py"),
    "cortical_tiles-info": (
        _remover("external/cortical_tiles/cortical_tiles/info.py"),
        "external/cortical_tiles/cortical_tiles/info.py",
    ),
    "Dockerfile.base": (_remover(".github/Dockerfile.base"), ".github/Dockerfile.base"),
}


@pytest.mark.parametrize("absent", list(ABSENT_INPUTS), ids=list(ABSENT_INPUTS))
def test_absent_input_is_refused(tmp_path, absent):
    root = _make_root(tmp_path / "repo")
    remove, name = ABSENT_INPUTS[absent]
    remove(root)
    result = _run(str(root))
    assert result.returncode != 0, f"exit 0 with {absent} absent; stdout {result.stdout!r}"
    assert result.stdout == "", f"a key was printed with {absent} absent: {result.stdout!r}"
    assert name in result.stderr, f"stderr does not name {name!r}: {result.stderr!r}"


# --- REQ-E2ECI-BDRABCZUK-0A1EE8FE1ABF: omitted ROOT -----------------------------


def test_omitted_root_uses_repository_of_script(tmp_path):
    root = _make_root(tmp_path / "repo")
    expected = _key(root)
    copy = root / "scripts" / "e2e" / "env_key.sh"
    shutil.copy2(SCRIPT, copy)
    elsewhere = tmp_path / "elsewhere"
    _make_root(elsewhere)
    _edit(elsewhere, "pixi.lock", "numpy-2.1.0", "numpy-9.9.9")
    result = _run(script=copy, cwd=elsewhere)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected, "without ROOT the key is not that of the script's repository"


# --- REQ-E2ECI-BDRABCZUK-14583CF7AF14: --sif-name -------------------------------


def test_sif_name_uses_settings_template(tmp_path):
    root = _make_root(tmp_path / "repo")
    key = _key(root)
    result = _run("--sif-name", str(root))
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"champollion-env-{key}.sif\n"


def test_sif_name_follows_template_in_root(tmp_path):
    root = _make_root(tmp_path / "repo")
    _edit(root, "scripts/e2e/e2e.env", '"champollion-env-{key12}.sif"', '"other-env-{key12}.img"')
    key = _key(root)
    result = _run("--sif-name", str(root))
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"other-env-{key}.img\n"
