"""Tests for ``scripts/e2e/pack_code.sh``, the per-commit code directory of the e2e runs.

TASK-E2ECI-BDRABCZUK-40360AB17808 (epic TASK-E2ECI-BDRABCZUK-F3B0138F58F3).
The env SIF holds no pipeline code: each run binds one code directory read-only.
``pack_code.sh <commit>`` exports, from the local repositories only (so an
unpushed commit can be packed, with no network):

- the repository's files tracked at ``<commit>``;
- external/champollion_V1 and external/cortical_tiles at the gitlinks recorded
  in ``<commit>`` (not what the submodules check out);
- champollion_utils at the reference after ``@`` in the ``champollion-utils``
  pin of ``pyproject.toml`` at ``<commit>``, read from the champollion_utils
  clone beside the repository root (where the pixi install-utils task clones it);
- a stub git repository at external/cortical_tiles (cortical_tiles logs.py asks
  gitpython for a HEAD), a COMMIT file and a sha256 MANIFEST;

into ``$E2E_CODE_DIR/<E2E_CODE_DIR_NAME>`` and prints that path. No CSV,
notebook, ``.git`` (except the stub) or cortical_tiles fixture data
(data/source, data/reference, data/target) is packed.

Every test builds throwaway git repositories under ``tmp_path``: a fake
pipeline repository holding a copy of the script and of ``e2e.env``, two
upstream submodule repositories and a champollion_utils clone beside it. The copied
``e2e.env`` has its ``E2E_CODE_DIR`` line rewritten to a directory under
``tmp_path`` (the real value is a rosette path), and HOME points into
``tmp_path``. No network, no data.

Requirements (one test or more each):
- REQ-E2ECI-BDRABCZUK-15CEB2C80F3E pipeline files packed at the exact commit
- REQ-E2ECI-BDRABCZUK-0512C0B79719 no untracked file in the packed pipeline tree
- REQ-E2ECI-BDRABCZUK-883D3EDDE452 champollion_V1 packed at its recorded gitlink
- REQ-E2ECI-BDRABCZUK-F9D7A4A65074 cortical_tiles packed at its recorded gitlink
- REQ-E2ECI-BDRABCZUK-0F4ADF05CF59 champollion_utils packed at its pinned reference
- REQ-E2ECI-BDRABCZUK-5A82654637E4 code directory location printed (supersedes B084A49D69E2)
- REQ-E2ECI-BDRABCZUK-69B19C613F82 COMMIT file
- REQ-E2ECI-BDRABCZUK-6A9091157F53 code directory MANIFEST
- REQ-E2ECI-BDRABCZUK-2990533B4239 cortical_tiles stub git repository
- REQ-E2ECI-BDRABCZUK-64105AF4EB36 no .git besides the cortical_tiles stub
- REQ-E2ECI-BDRABCZUK-F492C1BA6242 no CSV or notebook in the code directory
- REQ-E2ECI-BDRABCZUK-F739E440AD84 no cortical_tiles fixture data
- REQ-E2ECI-BDRABCZUK-CAF7D528AF36 banned data path refused
- REQ-E2ECI-BDRABCZUK-A3647BB89083 unknown commit refused (supersedes DCB90F0A86A0)
- REQ-E2ECI-BDRABCZUK-6B6D8FF0CBF0 absent or empty E2E_CODE_DIR refused
- REQ-E2ECI-BDRABCZUK-F297568C9014 missing submodule commit refused
- REQ-E2ECI-BDRABCZUK-1B3F8705020E missing champollion_utils pin refused
- REQ-E2ECI-BDRABCZUK-DD7C7BFA9970 packing leaves the sources unchanged
- REQ-E2ECI-BDRABCZUK-D4790ED256EC re-run verifies an intact code directory
- REQ-E2ECI-BDRABCZUK-CDA29E4B9E73 re-run refuses a drifted code directory
- REQ-E2ECI-BDRABCZUK-DD1754E42AE0 usage
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "e2e" / "pack_code.sh"
E2E_ENV = REPO / "scripts" / "e2e" / "e2e.env"
BASH = shutil.which("bash")
GIT = shutil.which("git")

SUBMODULES = ("external/champollion_V1", "external/cortical_tiles")
CT_FIXTURE_DIRS = ("data/source", "data/reference", "data/target")
UTILS_URL = "git+https://github.com/neurospin/champollion_utils.git"

pytestmark = pytest.mark.skipif(BASH is None or GIT is None, reason="bash or git not available")  # noqa: V107


# --- throwaway repositories ------------------------------------------------------


def _git_env(home: Path) -> dict[str, str]:
    """Environment of the tests' own git calls: an identity and file:// submodules."""
    return {
        "HOME": str(home),
        "PATH": os.pathsep.join(["/usr/bin", "/bin"]),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.org",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.org",
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "protocol.file.allow",
        "GIT_CONFIG_VALUE_0": "always",
        "GIT_CONFIG_KEY_1": "init.defaultBranch",
        "GIT_CONFIG_VALUE_1": "main",
    }


def _script_env(home: Path) -> dict[str, str]:
    """Environment of the script: HOME and a PATH, no git identity, no E2E_ value."""
    path = os.pathsep.join([str(Path(sys.executable).parent), "/usr/bin", "/bin"])
    return {"HOME": str(home), "PATH": path}


def _write(root: Path, files: dict[str, str | bytes]) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)


def _pyproject(pin: str) -> str:
    return f"""[project]
name = "champollion-pipeline"
version = "0.2.32"
dependencies = [
    "{pin}",
    "scipy>=1.10",
]
"""


V1_A = {
    "setup.py": "from setuptools import setup\nsetup(name='champollion_V1')\n",
    "champollion/__init__.py": "",
    "champollion/model.py": "LAYERS = 4\n",
    "champollion/old.py": "OLD = True\n",
    "champollion/configs/config.yaml": "seed: 1\n",
    "results/table.csv": "subject,value\nsub-001,1\n",
    "notebooks/explore.ipynb": '{"cells": []}\n',
}
CT_A = {
    "setup.py": "from setuptools import setup\nsetup(name='cortical_tiles')\n",
    "cortical_tiles/__init__.py": "",
    "cortical_tiles/info.py": "__version__ = '1.0.0'\n",
    "cortical_tiles/brainvisa/utils/logs.py": "import git\n",
    "cortical_tiles/brainvisa/masks.ipynb": '{"cells": []}\n',
    "data/README.rst": "Masks and fixtures\n",
    "data/sulci_regions_gridsearch.json": "{}\n",
    "data/mask/canonical_25/2mm/L/S.C._left.nii.gz": b"\x1f\x8bMASK-L\x00\x01",
    "data/mask/canonical_25/2mm/R/S.C._right.nii.gz": b"\x1f\x8bMASK-R\x00\x02",
    "data/source/supervised/sujet01/Lskeleton_sujet01.nii.gz": b"\x1f\x8bSKEL\x00",
    "data/reference/benchmark/skeleton_100206_suppr.nii.gz": b"\x1f\x8bREF\x00",
    "data/target/Lcrops/100206_normalized.nii.gz": b"\x1f\x8bTGT\x00",
    "tests/data/benchmark/abnormality.csv": "id,score\n1,0.5\n",
}
UTILS_TAGGED = {
    "pyproject.toml": '[project]\nname = "champollion-utils"\nversion = "0.3.0"\n',
    "src/champollion_utils/__init__.py": 'VERSION = "0.3.0"\n',
    "src/champollion_utils/script_builder.py": "class ScriptBuilder:\n    pass\n",
    "tests/fixtures/participants.csv": "participant_id\nsub-001\n",
}
PIPELINE_C1 = {
    "main.py": "STAGES = 6\n",
    "src/champollion_pipeline/__init__.py": "",
    "src/champollion_pipeline/stage.py": "VALUE = 1\n",
    "docs/Notes.CSV": "a,b\n1,2\n",
    "notebooks/analysis.ipynb": '{"cells": []}\n',
    "config/pipeline.yaml": "stages: [1, 2, 3, 4, 5, 6]\n",
    ".gitignore": "data/\n",
}


@dataclass
class World:
    """The throwaway repositories of one test."""

    base: Path
    home: Path
    upstream_v1: Path
    upstream_ct: Path
    utils: Path
    pipeline: Path
    v1_a: str
    v1_b: str
    ct_a: str
    ct_b: str
    utils_tag: str
    c1: str
    c2: str

    @property
    def script(self) -> Path:
        return self.pipeline / "scripts" / "e2e" / "pack_code.sh"

    def git(self, cwd: Path, *args: str) -> str:
        result = subprocess.run(
            [GIT, *args], cwd=cwd, env=_git_env(self.home), capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, f"git {' '.join(args)} in {cwd}: {result.stderr}"
        return result.stdout.strip()

    def commit_all(self, cwd: Path, message: str) -> str:
        self.git(cwd, "add", "-A")
        self.git(cwd, "commit", "-q", "-m", message)
        return self.git(cwd, "rev-parse", "HEAD")

    def run(self, *args: str) -> subprocess.CompletedProcess:
        """Run the pipeline repository's copy of pack_code.sh."""
        return subprocess.run(
            [BASH, str(self.script), *args],
            cwd=self.base,
            env=_script_env(self.home),
            capture_output=True,
            text=True,
            timeout=120,
        )

    def setting(self, name: str) -> str:
        """One value of the copied e2e.env, sourced with this test's HOME."""
        env_file = self.pipeline / "scripts" / "e2e" / "e2e.env"
        result = subprocess.run(
            [BASH, "-c", f'set -a; source "{env_file}" >/dev/null 2>&1; printf "%s" "${{{name}-}}"'],
            env=_script_env(self.home),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout

    @property
    def code_parent(self) -> Path:
        return Path(self.setting("E2E_CODE_DIR"))

    def code_dir(self, commit: str) -> Path:
        name = self.setting("E2E_CODE_DIR_NAME").replace("{commit12}", commit[:12])
        return self.code_parent / name

    def tree(self, cwd: Path, commit: str) -> dict[str, bytes]:
        """Blob paths and contents of ``commit`` in ``cwd`` (gitlinks left out)."""
        listing = subprocess.run(
            [GIT, "ls-tree", "-r", "-z", commit],
            cwd=cwd,
            env=_git_env(self.home),
            capture_output=True,
            timeout=60,
            check=True,
        ).stdout.decode()
        files = {}
        for entry in filter(None, listing.split("\0")):
            meta, rel = entry.split("\t", 1)
            _mode, kind, sha = meta.split()
            if kind != "blob":
                continue
            files[rel] = subprocess.run(
                [GIT, "cat-file", "blob", sha], cwd=cwd, env=_git_env(self.home), capture_output=True, check=True
            ).stdout
        return files


CODE_DIR_LINE = re.compile(r"^E2E_CODE_DIR=.*$", re.MULTILINE)


def _settings(code_dir_line: str) -> str:
    """The real e2e.env with its one E2E_CODE_DIR line replaced (empty string: removed)."""
    text, count = CODE_DIR_LINE.subn(code_dir_line, E2E_ENV.read_text())
    assert count == 1, f"expected one E2E_CODE_DIR line in e2e.env, found {count}"
    return text


def _init_repo(world_git, path: Path, files: dict[str, str | bytes]) -> None:
    path.mkdir(parents=True)
    world_git(path, "init", "-q")
    _write(path, files)


def _build_world(base: Path) -> World:
    home = base / "home"
    home.mkdir(parents=True)
    w = World(base, home, base / "upstream" / "v1", base / "upstream" / "ct", base / "work" / "champollion_utils",
              base / "work" / "champollion_pipeline", "", "", "", "", "", "", "")  # fmt: skip

    # Upstream submodule repositories: commit A is the one the pipeline records, B moves on.
    _init_repo(w.git, w.upstream_v1, V1_A)
    w.v1_a = w.commit_all(w.upstream_v1, "V1 A")
    (w.upstream_v1 / "champollion" / "old.py").unlink()
    _write(w.upstream_v1, {"champollion/model.py": "LAYERS = 5\n", "champollion/new.py": "NEW = True\n"})
    w.v1_b = w.commit_all(w.upstream_v1, "V1 B")

    _init_repo(w.git, w.upstream_ct, CT_A)
    w.ct_a = w.commit_all(w.upstream_ct, "CT A")
    _write(w.upstream_ct, {"cortical_tiles/info.py": "__version__ = '1.1.0'\n", "cortical_tiles/later.py": "X = 1\n"})
    w.ct_b = w.commit_all(w.upstream_ct, "CT B")

    # champollion_utils clone beside the pipeline: tag v0.3.0, then a newer HEAD.
    _init_repo(w.git, w.utils, UTILS_TAGGED)
    w.utils_tag = w.commit_all(w.utils, "utils 0.3.0")
    w.git(w.utils, "tag", "v0.3.0")
    _write(w.utils, {"src/champollion_utils/__init__.py": 'VERSION = "0.4.0.dev0"\n', "src/champollion_utils/x.py": ""})
    w.commit_all(w.utils, "utils dev")

    # Pipeline repository with the script under test and the shared settings file.
    files: dict[str, str | bytes] = dict(PIPELINE_C1)
    files["pyproject.toml"] = _pyproject(f"champollion-utils @ {UTILS_URL}@v0.3.0")
    files["scripts/e2e/e2e.env"] = _settings(code_dir_line=f"E2E_CODE_DIR={base / 'codedirs'}")
    _init_repo(w.git, w.pipeline, files)
    if SCRIPT.is_file():
        shutil.copy2(SCRIPT, w.script)
    for rel, upstream, commit in ((SUBMODULES[0], w.upstream_v1, w.v1_a), (SUBMODULES[1], w.upstream_ct, w.ct_a)):
        w.git(w.pipeline, "submodule", "add", "-q", str(upstream), rel)
        w.git(w.pipeline / rel, "checkout", "-q", commit)
        w.git(w.pipeline, "add", rel)
    w.c1 = w.commit_all(w.pipeline, "C1")

    # A later local commit (HEAD), then submodule checkouts moved away from the gitlinks,
    # an uncommitted edit and an untracked DEMO01 image in the working tree.
    _write(w.pipeline, {"main.py": "STAGES = 7\n", "src/champollion_pipeline/new_in_c2.py": "C2 = True\n"})
    w.c2 = w.commit_all(w.pipeline, "C2")
    w.git(w.pipeline / SUBMODULES[0], "checkout", "-q", w.v1_b)
    w.git(w.pipeline / SUBMODULES[1], "checkout", "-q", w.ct_b)
    _write(w.pipeline, {"main.py": "STAGES = 99  # uncommitted\n", "data/DEMO01/sub-01_T1w.nii.gz": b"\x1f\x8bDEMO"})
    _write(w.pipeline, {"scratch/untracked.py": "UNTRACKED = True\n"})
    return w


@pytest.fixture
def world(tmp_path: Path) -> World:
    return _build_world(tmp_path)


@dataclass
class Packed:
    world: World
    result: subprocess.CompletedProcess
    code: Path


@pytest.fixture(scope="module")
def packed(tmp_path_factory: pytest.TempPathFactory) -> Packed:
    """One run of pack_code.sh on C1 (not HEAD), shared by the read-only checks."""
    w = _build_world(tmp_path_factory.mktemp("packed"))
    return Packed(w, w.run(w.c1), w.code_dir(w.c1))


def _assert_success(p: Packed) -> None:
    assert p.world.script.is_file(), f"{SCRIPT.relative_to(REPO)} is missing"
    assert p.result.returncode == 0, f"exit {p.result.returncode}: {p.result.stderr}"
    assert p.code.is_dir(), f"no code directory at {p.code}"


def _files(root: Path) -> dict[str, Path]:
    """Regular files under ``root`` by path relative to it (symbolic links not followed)."""
    found = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            found[path.relative_to(root).as_posix()] = path
    return found


def _packable(rel: str) -> bool:
    return not rel.lower().endswith((".csv", ".ipynb"))


def _assert_holds_tree(code: Path, prefix: str, expected: dict[str, bytes]) -> None:
    for rel, content in expected.items():
        path = code / prefix / rel if prefix else code / rel
        assert path.is_file(), f"{(Path(prefix) / rel).as_posix()} missing from the code directory"
        assert path.read_bytes() == content, f"{(Path(prefix) / rel).as_posix()} differs from the recorded commit"


def _assert_refused(w: World, result: subprocess.CompletedProcess, named: str, commit: str | None = None) -> None:
    assert w.script.is_file(), f"{SCRIPT.relative_to(REPO)} is missing"
    assert result.returncode != 0, f"exit 0 although refused; stdout: {result.stdout}"
    assert result.returncode != 127, f"script not runnable: {result.stderr}"
    assert named in result.stderr, f"{named!r} not named on stderr: {result.stderr!r}"
    if commit is not None:
        assert not w.code_dir(commit).exists(), "a code directory was left behind"


# --- REQ-E2ECI-BDRABCZUK-15CEB2C80F3E: pipeline files at the exact commit ---------


def test_packs_tracked_files_at_commit(packed):
    _assert_success(packed)
    w = packed.world
    expected = {
        rel: content
        for rel, content in w.tree(w.pipeline, w.c1).items()
        if _packable(rel) and not rel.startswith(tuple(s + "/" for s in SUBMODULES))
    }
    assert "pyproject.toml" in expected and ".gitmodules" in expected
    _assert_holds_tree(packed.code, "", expected)


def test_packs_committed_content_not_working_tree_edit(packed):
    _assert_success(packed)
    assert (packed.world.pipeline / "main.py").read_text() != "STAGES = 6\n"
    assert (packed.code / "main.py").read_text() == "STAGES = 6\n", "main.py is not the content recorded at C1"


def test_packs_unpushed_local_commit_not_head(packed):
    _assert_success(packed)
    assert not (packed.code / "src" / "champollion_pipeline" / "new_in_c2.py").exists(), "a file of HEAD (C2) packed"
    assert (packed.code / "COMMIT").read_text().strip() == packed.world.c1


# --- REQ-E2ECI-BDRABCZUK-0512C0B79719: nothing untracked ----------------------------


def test_untracked_files_not_packed(packed):
    _assert_success(packed)
    w = packed.world
    tracked = set(w.tree(w.pipeline, w.c1))
    own = ("COMMIT", "MANIFEST")
    skipped = ("champollion_utils/", *(s + "/" for s in SUBMODULES))
    extra = [
        rel for rel in _files(packed.code) if rel not in own and not rel.startswith(skipped) and rel not in tracked
    ]
    assert extra == [], f"files not tracked at C1 in the code directory: {extra}"
    assert not (packed.code / "data").exists(), "the untracked data/ directory (DEMO01) was packed"


# --- REQ-E2ECI-BDRABCZUK-883D3EDDE452 / F9D7A4A65074: submodules at their gitlinks -


def test_champollion_v1_packed_at_gitlink(packed):
    _assert_success(packed)
    w = packed.world
    expected = {rel: c for rel, c in w.tree(w.upstream_v1, w.v1_a).items() if _packable(rel)}
    _assert_holds_tree(packed.code, SUBMODULES[0], expected)
    assert not (packed.code / SUBMODULES[0] / "champollion" / "new.py").exists(), "a file of the checked-out V1 commit"


def test_cortical_tiles_packed_at_gitlink(packed):
    _assert_success(packed)
    w = packed.world
    expected = {
        rel: c
        for rel, c in w.tree(w.upstream_ct, w.ct_a).items()
        if _packable(rel) and not rel.startswith(tuple(d + "/" for d in CT_FIXTURE_DIRS))
    }
    assert "data/mask/canonical_25/2mm/L/S.C._left.nii.gz" in expected
    _assert_holds_tree(packed.code, SUBMODULES[1], expected)
    assert not (packed.code / SUBMODULES[1] / "cortical_tiles" / "later.py").exists(), "a file of the checked-out CT"


# --- REQ-E2ECI-BDRABCZUK-0F4ADF05CF59: champollion_utils at its pin ----------------


def test_utils_packed_at_pinned_ref(packed):
    _assert_success(packed)
    w = packed.world
    expected = {rel: c for rel, c in w.tree(w.utils, w.utils_tag).items() if _packable(rel)}
    _assert_holds_tree(packed.code, "champollion_utils", expected)
    assert not (packed.code / "champollion_utils" / "src" / "champollion_utils" / "x.py").exists()


# --- REQ-E2ECI-BDRABCZUK-5A82654637E4: location printed ----------------------------


def test_code_dir_path_printed(packed):
    _assert_success(packed)
    w = packed.world
    assert w.code_parent == w.base / "codedirs", "the copied e2e.env does not point E2E_CODE_DIR into tmp_path"
    assert packed.code.parent == w.code_parent
    assert packed.code.name == w.c1[:12]
    last = packed.result.stdout.rstrip("\n").splitlines()[-1]
    assert Path(last) == packed.code, f"last stdout line {last!r} is not {packed.code}"


def test_short_commit_argument_resolves_to_full_hash(world):
    result = world.run(world.c1[:7])
    assert world.script.is_file(), f"{SCRIPT.relative_to(REPO)} is missing"
    assert result.returncode == 0, result.stderr
    code = world.code_dir(world.c1)
    assert Path(result.stdout.rstrip("\n").splitlines()[-1]) == code
    assert (code / "COMMIT").read_text() == world.c1 + "\n"


# --- REQ-E2ECI-BDRABCZUK-69B19C613F82: COMMIT --------------------------------------


def test_commit_file(packed):
    _assert_success(packed)
    text = (packed.code / "COMMIT").read_text()
    assert re.fullmatch(r"[0-9a-f]{40}\n", text), f"COMMIT is not one 40-hex line: {text!r}"
    assert text == packed.world.c1 + "\n"


# --- REQ-E2ECI-BDRABCZUK-6A9091157F53: MANIFEST ------------------------------------


def test_manifest_lists_each_other_file_sorted(packed):
    _assert_success(packed)
    lines = (packed.code / "MANIFEST").read_bytes().decode().splitlines()
    entries = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        assert match, f"not a sha256sum line: {line!r}"
        entries.append((match.group(2), match.group(1)))
    paths = [rel for rel, _ in entries]
    assert paths == sorted(paths, key=lambda rel: rel.encode()), "MANIFEST is not sorted by byte order"
    files = _files(packed.code)
    files.pop("MANIFEST")
    assert set(paths) == set(files), f"MANIFEST and files differ: {sorted(set(paths) ^ set(files))[:10]}"
    for rel, digest in entries:
        assert hashlib.sha256(files[rel].read_bytes()).hexdigest() == digest, f"wrong checksum for {rel}"


# --- REQ-E2ECI-BDRABCZUK-2990533B4239: cortical_tiles stub repository --------------


def test_cortical_tiles_stub_repo_head(packed):
    _assert_success(packed)
    w = packed.world
    stub = packed.code / SUBMODULES[1]
    assert (stub / ".git").exists(), "no .git in the packed external/cortical_tiles"
    assert Path(w.git(stub, "rev-parse", "--show-toplevel")).resolve() == stub.resolve()
    head = w.git(stub, "rev-parse", "--verify", "HEAD^{commit}")
    assert re.fullmatch(r"[0-9a-f]{40}", head)


# --- REQ-E2ECI-BDRABCZUK-64105AF4EB36: no .git besides the stub --------------------


def test_no_git_entry_except_stub(packed):
    _assert_success(packed)
    found = [
        Path(dirpath, name).relative_to(packed.code).as_posix()
        for dirpath, dirnames, filenames in os.walk(packed.code)
        for name in (*dirnames, *filenames)
        if name == ".git"
    ]
    assert [rel for rel in found if rel != "external/cortical_tiles/.git"] == []


# --- REQ-E2ECI-BDRABCZUK-F492C1BA6242: no CSV or notebook --------------------------


def test_no_csv_or_notebook(packed):
    _assert_success(packed)
    found = [rel for rel in _files(packed.code) if not _packable(rel)]
    assert found == [], f"CSV or notebook files packed: {found}"


# --- REQ-E2ECI-BDRABCZUK-F739E440AD84: no cortical_tiles fixture data --------------


def test_no_cortical_tiles_fixture_data(packed):
    _assert_success(packed)
    for rel in CT_FIXTURE_DIRS:
        assert not (packed.code / SUBMODULES[1] / rel).exists(), f"external/cortical_tiles/{rel} packed"


# --- REQ-E2ECI-BDRABCZUK-CAF7D528AF36: banned data path refused --------------------


def test_banned_path_refused(world):
    _write(world.pipeline, {"docs/demo01_subjects.txt": "sub-01\n"})
    world.git(world.pipeline, "add", "docs/demo01_subjects.txt")
    world.git(world.pipeline, "commit", "-q", "-m", "banned")
    commit = world.git(world.pipeline, "rev-parse", "HEAD")
    _assert_refused(world, world.run(commit), "docs/demo01_subjects.txt", commit)


# --- REQ-E2ECI-BDRABCZUK-A3647BB89083: unknown commit refused ----------------------


@pytest.mark.parametrize("argument", ["0123456789abcdef0123456789abcdef01234567", "no-such-branch"])
def test_unknown_commit_refused(world, argument):
    _assert_refused(world, world.run(argument), argument)
    assert not world.code_parent.exists() or not any(world.code_parent.iterdir())


def test_non_commit_object_refused(world):
    tree = world.git(world.pipeline, "rev-parse", f"{world.c1}^{{tree}}")
    _assert_refused(world, world.run(tree), tree)
    assert not world.code_parent.exists() or not any(world.code_parent.iterdir())


# --- REQ-E2ECI-BDRABCZUK-6B6D8FF0CBF0: E2E_CODE_DIR required ---------------------


@pytest.mark.parametrize("line", ["", "E2E_CODE_DIR="], ids=["unset", "empty"])
def test_code_dir_setting_required(world, line):
    (world.pipeline / "scripts" / "e2e" / "e2e.env").write_text(_settings(code_dir_line=line))
    before = sorted(_files(world.base))
    _assert_refused(world, world.run(world.c1), "E2E_CODE_DIR")
    assert sorted(_files(world.base)) == before, "files were written although E2E_CODE_DIR is missing"
    assert not Path("/", world.c1[:12]).exists()


# --- REQ-E2ECI-BDRABCZUK-F297568C9014: missing submodule commit refused ------------


@pytest.mark.parametrize("submodule", SUBMODULES)
def test_missing_submodule_commit_refused(world, submodule):
    absent = "0123456789abcdef0123456789abcdef01234567"
    world.git(world.pipeline, "update-index", "--cacheinfo", f"160000,{absent},{submodule}")
    world.git(world.pipeline, "commit", "-q", "-m", "gitlink to an absent commit")
    commit = world.git(world.pipeline, "rev-parse", "HEAD")
    _assert_refused(world, world.run(commit), submodule, commit)


# --- REQ-E2ECI-BDRABCZUK-1B3F8705020E: missing champollion_utils pin refused -------


def _commit_pin(world: World, pin: str) -> str:
    _write(world.pipeline, {"pyproject.toml": _pyproject(pin)})
    world.git(world.pipeline, "add", "pyproject.toml")
    world.git(world.pipeline, "commit", "-q", "-m", "pin")
    return world.git(world.pipeline, "rev-parse", "HEAD")


def test_unpinned_utils_refused(world):
    commit = _commit_pin(world, "champollion-utils>=0.3")
    _assert_refused(world, world.run(commit), "champollion-utils", commit)


def test_unknown_utils_ref_refused(world):
    commit = _commit_pin(world, f"champollion-utils @ {UTILS_URL}@v9.9.9")
    _assert_refused(world, world.run(commit), "champollion-utils", commit)


# --- REQ-E2ECI-BDRABCZUK-DD7C7BFA9970: sources unchanged ---------------------------


def _snapshot(world: World, repo: Path) -> dict[str, object]:
    files = {}
    for rel, path in _files(repo).items():
        if rel.split("/", 1)[0] != ".git" and "/.git/" not in f"/{rel}" and not rel.endswith("/.git"):
            files[rel] = path.read_bytes()
    return {
        "head": world.git(repo, "rev-parse", "HEAD"),
        "index": world.git(repo, "ls-files", "-s"),
        "files": files,
    }


def test_sources_unchanged(world):
    repos = [world.pipeline, *(world.pipeline / s for s in SUBMODULES), world.utils]
    before = [_snapshot(world, repo) for repo in repos]
    result = world.run(world.c1)
    assert world.script.is_file(), f"{SCRIPT.relative_to(REPO)} is missing"
    assert result.returncode == 0, result.stderr
    after = [_snapshot(world, repo) for repo in repos]
    for repo, old, new in zip(repos, before, after, strict=True):
        for key in old:
            assert old[key] == new[key], f"{key} of {repo.relative_to(world.base)} changed"


# --- REQ-E2ECI-BDRABCZUK-D4790ED256EC: re-run on an intact directory ---------------


def _stat_all(root: Path) -> dict[str, tuple]:
    found = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in (*dirnames, *filenames):
            path = Path(dirpath, name)
            st = path.lstat()
            content = path.read_bytes() if path.is_file() and not path.is_symlink() else None
            found[path.relative_to(root).as_posix()] = (st.st_mode, st.st_mtime_ns, content)
    return found


def test_rerun_keeps_intact_code_dir(world):
    first = world.run(world.c1)
    assert world.script.is_file(), f"{SCRIPT.relative_to(REPO)} is missing"
    assert first.returncode == 0, first.stderr
    code = world.code_dir(world.c1)
    before = _stat_all(code)
    second = world.run(world.c1)
    assert second.returncode == 0, second.stderr
    assert _stat_all(code) == before, "the re-run changed the code directory"


# --- REQ-E2ECI-BDRABCZUK-CDA29E4B9E73: re-run on a drifted directory ---------------


def _packed_once(world: World) -> Path:
    first = world.run(world.c1)
    assert world.script.is_file(), f"{SCRIPT.relative_to(REPO)} is missing"
    assert first.returncode == 0, first.stderr
    code = world.code_dir(world.c1)
    for dirpath, _dirnames, _filenames in os.walk(code):
        os.chmod(dirpath, 0o755)
    return code


def test_rerun_refuses_altered_file(world):
    code = _packed_once(world)
    target = code / "src" / "champollion_pipeline" / "stage.py"
    target.chmod(0o644)
    target.write_text("VALUE = 2\n")
    _assert_refused(world, world.run(world.c1), "src/champollion_pipeline/stage.py")


def test_rerun_refuses_missing_file(world):
    code = _packed_once(world)
    (code / "main.py").unlink()
    _assert_refused(world, world.run(world.c1), "main.py")


def test_rerun_refuses_unlisted_file(world):
    code = _packed_once(world)
    (code / "extra_unlisted.py").write_text("X = 1\n")
    _assert_refused(world, world.run(world.c1), "extra_unlisted.py")


# --- REQ-E2ECI-BDRABCZUK-DD1754E42AE0: usage ---------------------------------------


def test_usage_without_argument(world):
    result = world.run()
    assert world.script.is_file(), f"{SCRIPT.relative_to(REPO)} is missing"
    assert result.returncode == 2, f"exit {result.returncode}: {result.stderr}"
    assert "usage" in result.stderr.lower()


def test_usage_with_two_arguments(world):
    result = world.run(world.c1, world.c2)
    assert world.script.is_file(), f"{SCRIPT.relative_to(REPO)} is missing"
    assert result.returncode == 2, f"exit {result.returncode}: {result.stderr}"
    assert "usage" in result.stderr.lower()
