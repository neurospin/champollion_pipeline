"""Tests for ``scripts/e2e/stage_cohort_view.sh``: the read-only cohort view of the e2e runs.

TASK-E2ECI-BDRABCZUK-4D4599ABA667 (epic TASK-E2ECI-BDRABCZUK-F3B0138F58F3).

The script builds ``E2E_DATA_DIR/tier_cohort``: a view of every raw IXI T1 under
``E2E_IXI_RAWDATA`` (open data) in the two flat input layouts stage 1 accepts, made of
symbolic links to the absolute source paths (no copy of the data), plus an
ATTRIBUTION file and a MANIFEST of the links and their targets:

- ``tier_cohort/bids/<source file name>`` -> the source T1;
- ``tier_cohort/classic/<subject>.nii.gz`` -> the source T1.

The tests stay off rosette the way ``tests/test_e2e_stage_data.py`` does: each test
copies the script and ``e2e_env.py`` into a temporary ``scripts/e2e/`` whose
``e2e.env`` is the real file plus an appended block re-pointing ``E2E_IXI_RAWDATA``,
``E2E_IXI_MORPHOLOGIST_DIR``, ``E2E_ALLOWED_DATA_ROOTS`` and ``E2E_DATA_DIR`` at a fake
IXI tree under ``tmp_path``. No real data, no network.

Requirements (one test or more each):
- REQ-E2ECI-BDRABCZUK-E4F3F1D57D22 BIDS view: symbolic links named as the source file
- REQ-E2ECI-BDRABCZUK-A45C6F5ADA24 classic view: symbolic links named ``<subject>.nii.gz``
- REQ-E2ECI-BDRABCZUK-E919DB2E66B0 a subject with two T1 is refused
- REQ-E2ECI-BDRABCZUK-D08C6A452163 an empty cohort is refused
- REQ-E2ECI-BDRABCZUK-1BF64CA1DF2E check-path on the resolved real path of each source
- REQ-E2ECI-BDRABCZUK-0172495B613D the IXI source tree is left unchanged
- REQ-E2ECI-BDRABCZUK-8984BAA3113A ATTRIBUTION (IXI, CC BY-SA 3.0)
- REQ-E2ECI-BDRABCZUK-59F66D6C14F5 MANIFEST of link paths and targets
- REQ-E2ECI-BDRABCZUK-B92BE9C40C2D re-run on an intact view changes nothing
- REQ-E2ECI-BDRABCZUK-79296D0569BE re-run refuses a drifted view
- REQ-E2ECI-BDRABCZUK-4D3750D2901F the view is read-only
- REQ-E2ECI-BDRABCZUK-7BB6FC9C1B80 failed run leaves E2E_DATA_DIR unchanged
- REQ-E2ECI-BDRABCZUK-82469329998E settings come from the sibling e2e.env
- REQ-E2ECI-BDRABCZUK-60B2E493BF3B (SET) check-path on each source, refusal exits non-zero
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
E2E_DIR = REPO / "scripts" / "e2e"
SCRIPT = E2E_DIR / "stage_cohort_view.sh"
CHECKER = E2E_DIR / "e2e_env.py"
ENV_FILE = E2E_DIR / "e2e.env"
BASH = shutil.which("bash")

TIER_DIRNAME = "tier_cohort"
LINK_SEPARATOR = " -> "
IXI_URL = "https://brain-development.org/ixi-dataset/"
LICENCE_URL = "https://creativecommons.org/licenses/by-sa/3.0/"
COHORT = ("sub-002", "sub-012", "sub-031", "sub-047")
T2_ONLY_SUBJECT = "sub-050"  # a subject directory with no T1: no link expected
RAW_ROOT_FILE = "rawdata_v-20260312.tsv"  # a non-subject file at the raw root, as on rosette

pytestmark = pytest.mark.skipif(BASH is None, reason="bash executable not available")  # noqa: V107


def _real_settings(home: Path) -> dict[str, str]:
    """E2E_* values of the real scripts/e2e/e2e.env, read with e2e_env.load_settings."""
    spec = importlib.util.spec_from_file_location("_e2e_env_under_test", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_settings(ENV_FILE, home=str(home))


def _t1_relpath(subject: str) -> str:
    """Relative path of a fake IXI raw T1, shaped like the real IXI tree."""
    return f"{subject}/ses-V1/anat/{subject}_ses-V1_T1w.nii.gz"


class Sandbox:
    """A temporary scripts/e2e/ copy, a fake IXI raw tree and a fake data directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.home = root / "home"
        self.home.mkdir()
        self.settings = _real_settings(self.home)
        self.subjects = list(COHORT)
        self.e2e_dir = root / "repo" / "scripts" / "e2e"
        self.raw = root / "src" / "ixi" / "rawdata"
        self.morpho = root / "src" / "ixi" / "morphologist"
        self.data = root / "dest" / "data"
        self.tier = self.data / TIER_DIRNAME
        self.overrides: dict[str, str] = {}

    # -- setup ---------------------------------------------------------------

    def build(self, subjects: tuple[str, ...] = COHORT, **overrides: str) -> Sandbox:
        """Copy the scripts, write the e2e.env copy and the fake IXI tree."""
        assert SCRIPT.is_file(), f"{SCRIPT.relative_to(REPO)} is missing"
        self.e2e_dir.mkdir(parents=True)
        shutil.copy2(SCRIPT, self.e2e_dir / SCRIPT.name)
        shutil.copy2(CHECKER, self.e2e_dir / CHECKER.name)
        self.morpho.mkdir(parents=True)
        self.raw.mkdir(parents=True)
        (self.raw / RAW_ROOT_FILE).write_text("participant_id\n")
        self.subjects = list(subjects)
        for subject in self.subjects:
            self.add_source(_t1_relpath(subject), f"fake T1 of {subject}")
        for subject in [*self.subjects, T2_ONLY_SUBJECT]:
            self.add_source(f"{subject}/ses-V1/anat/{subject}_ses-V1_T2w.nii.gz", f"fake T2 of {subject}")
        self.overrides = {
            "E2E_IXI_RAWDATA": str(self.raw),
            "E2E_IXI_MORPHOLOGIST_DIR": str(self.morpho),
            "E2E_ALLOWED_DATA_ROOTS": f"{self.raw} {self.morpho}",
            "E2E_DATA_DIR": str(self.data),
        }
        self.overrides.update(overrides)
        self.write_env()
        glob = self.settings["E2E_IXI_T1_GLOB"]
        matched = sorted(str(p.relative_to(self.raw)) for p in self.raw.glob(glob))
        assert matched == sorted(_t1_relpath(s) for s in self.subjects), f"test premise: {glob} vs fake tree"
        return self

    def write_env(self) -> None:
        block = "".join(f'{key}="{value}"\n' for key, value in self.overrides.items())
        text = ENV_FILE.read_text() + "\n# --- test sandbox overrides ---\n" + block
        (self.e2e_dir / ENV_FILE.name).write_text(text)

    def add_source(self, relpath: str, text: str) -> Path:
        path = self.raw / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode() + os.urandom(64))
        return path

    # -- run -----------------------------------------------------------------

    def run(self, extra_env: dict[str, str] | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        env = {"HOME": str(self.home), "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.defpath}"}
        env.update(extra_env or {})
        return subprocess.run(
            [str(self.e2e_dir / SCRIPT.name)],
            cwd=cwd or self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def stage(self) -> subprocess.CompletedProcess[str]:
        result = self.run()
        assert result.returncode == 0, f"stage_cohort_view.sh failed ({result.returncode}): {result.stderr}"
        return result

    # -- inspection ----------------------------------------------------------

    def source_of(self, subject: str) -> Path:
        return self.raw / _t1_relpath(subject)

    def links(self) -> dict[str, str]:
        """Relative path under tier_cohort -> raw target of every symbolic link there."""
        return {str(p.relative_to(self.tier)): os.readlink(p) for p in sorted(self.tier.rglob("*")) if p.is_symlink()}

    def tree_paths(self) -> set[str]:
        """Every path under E2E_DATA_DIR, relative to it (empty when absent)."""
        if not self.data.exists():
            return set()
        return {str(p.relative_to(self.data)) for p in self.data.rglob("*")}


def _snapshot(root: Path) -> dict[str, tuple[str, int, int]]:
    """(sha256, link target or 'dir', mode, mtime_ns) of every path under ``root``, links not followed."""
    state = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        if path.is_symlink():
            digest = "link:" + os.readlink(path)
        elif path.is_dir():
            digest = "dir"
        else:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        state[str(path.relative_to(root))] = (digest, info.st_mode, info.st_mtime_ns)
    return state


def _make_writable(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
        for name in [*dirnames, *filenames]:
            path = Path(dirpath, name)
            if not path.is_symlink():
                path.chmod(path.stat().st_mode | stat.S_IWUSR)
    if root.exists():
        root.chmod(root.stat().st_mode | stat.S_IWUSR)


@pytest.fixture
def sandbox(tmp_path: Path):
    box = Sandbox(tmp_path)
    yield box
    _make_writable(tmp_path)


@contextlib.contextmanager
def _writable(path: Path) -> Iterator[None]:
    """Make ``path`` owner-writable for a test edit, then restore its mode."""
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IWUSR)
    try:
        yield
    finally:
        path.chmod(mode)


# REQ-E2ECI-BDRABCZUK-E4F3F1D57D22 --------------------------------------------------


def test_bids_view_links_each_source_by_name(sandbox: Sandbox) -> None:
    sandbox.build().stage()
    bids = sandbox.tier / "bids"
    entries = sorted(bids.iterdir())
    assert [p.name for p in entries] == sorted(sandbox.source_of(s).name for s in sandbox.subjects)
    for entry in entries:
        assert entry.is_symlink(), f"{entry} is not a symbolic link"
        subject = entry.name.split("_", 1)[0]
        assert os.readlink(entry) == str(sandbox.source_of(subject)), f"{entry} targets {os.readlink(entry)}"


# REQ-E2ECI-BDRABCZUK-A45C6F5ADA24 --------------------------------------------------


def test_classic_view_links_each_subject(sandbox: Sandbox) -> None:
    sandbox.build().stage()
    classic = sandbox.tier / "classic"
    entries = sorted(classic.iterdir())
    assert [p.name for p in entries] == sorted(f"{s}.nii.gz" for s in sandbox.subjects)
    for entry in entries:
        assert entry.is_symlink(), f"{entry} is not a symbolic link"
        subject = entry.name.removesuffix(".nii.gz")
        assert os.readlink(entry) == str(sandbox.source_of(subject)), f"{entry} targets {os.readlink(entry)}"


# REQ-E2ECI-BDRABCZUK-E919DB2E66B0 --------------------------------------------------


def test_subject_with_two_t1_is_refused(sandbox: Sandbox) -> None:
    sandbox.build()
    doubled = sandbox.subjects[1]
    sandbox.add_source(f"{doubled}/ses-V1/anat/{doubled}_ses-V1_run-2_T1w.nii.gz", "second T1")
    result = sandbox.run()
    assert result.returncode != 0
    assert doubled in result.stderr, result.stderr


# REQ-E2ECI-BDRABCZUK-D08C6A452163 --------------------------------------------------


def test_empty_cohort_is_refused(sandbox: Sandbox) -> None:
    sandbox.build(subjects=())
    result = sandbox.run()
    assert result.returncode != 0
    assert str(sandbox.raw) in result.stderr, result.stderr


# REQ-E2ECI-BDRABCZUK-60B2E493BF3B --------------------------------------------------


def test_source_outside_allowed_roots_is_refused(sandbox: Sandbox) -> None:
    sandbox.build(E2E_ALLOWED_DATA_ROOTS=str(sandbox.morpho))
    result = sandbox.run()
    assert result.returncode != 0
    assert "refused" in result.stderr, result.stderr
    assert str(sandbox.raw) in result.stderr, result.stderr


@pytest.mark.parametrize("marker", ["DEMO01", "UkBioBank"])
def test_banned_dataset_source_is_refused(sandbox: Sandbox, marker: str) -> None:
    sandbox.raw = sandbox.root / "src" / marker / "rawdata"
    sandbox.build()
    result = sandbox.run()
    assert result.returncode != 0
    assert "banned data pattern" in result.stderr, result.stderr


# REQ-E2ECI-BDRABCZUK-1BF64CA1DF2E --------------------------------------------------


def test_symlink_to_outside_root_is_refused(sandbox: Sandbox) -> None:
    sandbox.build()
    outside = sandbox.root / "elsewhere" / "t1.nii.gz"
    outside.parent.mkdir()
    outside.write_bytes(b"not open data")
    source = sandbox.source_of(sandbox.subjects[0])
    source.unlink()
    source.symlink_to(outside)
    result = sandbox.run()
    assert result.returncode != 0
    assert "refused" in result.stderr, result.stderr
    assert os.path.realpath(outside) in result.stderr, result.stderr


# REQ-E2ECI-BDRABCZUK-0172495B613D --------------------------------------------------


def test_source_tree_is_unchanged(sandbox: Sandbox) -> None:
    sandbox.build()
    before = _snapshot(sandbox.raw)
    root_mode = sandbox.raw.lstat()
    sandbox.stage()
    assert _snapshot(sandbox.raw) == before
    after = sandbox.raw.lstat()
    assert (after.st_mode, after.st_mtime_ns) == (root_mode.st_mode, root_mode.st_mtime_ns)


# REQ-E2ECI-BDRABCZUK-8984BAA3113A --------------------------------------------------


def test_attribution_names_dataset_and_licence(sandbox: Sandbox) -> None:
    sandbox.build().stage()
    text = (sandbox.tier / "ATTRIBUTION").read_text()
    licence = sandbox.settings["E2E_DATA_LICENSE"]
    for needle in ("IXI", IXI_URL, licence, LICENCE_URL):
        assert needle in text, f"ATTRIBUTION lacks {needle!r}"


# REQ-E2ECI-BDRABCZUK-59F66D6C14F5 --------------------------------------------------


def test_manifest_lists_each_link_and_target(sandbox: Sandbox) -> None:
    sandbox.build().stage()
    lines = (sandbox.tier / "MANIFEST").read_text().splitlines()
    bad = [line for line in lines if line.count(LINK_SEPARATOR) != 1]
    assert not bad, f"lines not shaped '<link>{LINK_SEPARATOR}<target>': {bad}"
    listed = dict(line.split(LINK_SEPARATOR) for line in lines)
    assert len(listed) == len(lines), "a link is listed twice"
    assert listed == sandbox.links()
    paths = [line.split(LINK_SEPARATOR)[0] for line in lines]
    assert paths == sorted(paths), "MANIFEST is not sorted by link path"


# REQ-E2ECI-BDRABCZUK-B92BE9C40C2D --------------------------------------------------


def test_rerun_on_intact_view_changes_nothing(sandbox: Sandbox) -> None:
    sandbox.build().stage()
    before = _snapshot(sandbox.data)
    result = sandbox.run()
    assert result.returncode == 0, result.stderr
    assert _snapshot(sandbox.data) == before


# REQ-E2ECI-BDRABCZUK-79296D0569BE --------------------------------------------------


def test_rerun_refuses_retargeted_link(sandbox: Sandbox) -> None:
    sandbox.build().stage()
    link = sandbox.tier / "classic" / f"{sandbox.subjects[0]}.nii.gz"
    with _writable(link.parent):
        link.unlink()
        link.symlink_to(sandbox.source_of(sandbox.subjects[1]))
    result = sandbox.run()
    assert result.returncode != 0
    assert link.name in result.stderr, result.stderr


def test_rerun_refuses_missing_link(sandbox: Sandbox) -> None:
    sandbox.build().stage()
    link = sandbox.tier / "bids" / sandbox.source_of(sandbox.subjects[0]).name
    with _writable(link.parent):
        link.unlink()
    result = sandbox.run()
    assert result.returncode != 0
    assert link.name in result.stderr, result.stderr


def test_rerun_refuses_unlisted_link(sandbox: Sandbox) -> None:
    sandbox.build().stage()
    extra = sandbox.tier / "bids" / "sub-077_ses-V1_T1w.nii.gz"
    with _writable(extra.parent):
        extra.symlink_to(sandbox.source_of(sandbox.subjects[0]))
    result = sandbox.run()
    assert result.returncode != 0
    assert extra.name in result.stderr, result.stderr


# REQ-E2ECI-BDRABCZUK-4D3750D2901F --------------------------------------------------


def test_view_has_no_write_bit(sandbox: Sandbox) -> None:
    sandbox.build().stage()
    paths = [sandbox.tier, *sandbox.tier.rglob("*")]
    writable = [str(p) for p in paths if not p.is_symlink() and p.lstat().st_mode & 0o222]
    assert not writable, f"paths with a write bit: {writable}"


# REQ-E2ECI-BDRABCZUK-7BB6FC9C1B80 --------------------------------------------------


def _break_outside_root(box: Sandbox) -> None:
    box.overrides["E2E_ALLOWED_DATA_ROOTS"] = str(box.morpho)
    box.write_env()


def _break_two_t1(box: Sandbox) -> None:
    subject = box.subjects[-1]
    box.add_source(f"{subject}/ses-V1/anat/{subject}_ses-V1_run-2_T1w.nii.gz", "second T1")


def _break_symlink_outside(box: Sandbox) -> None:
    outside = box.root / "elsewhere.nii.gz"
    outside.write_bytes(b"not open data")
    source = box.source_of(box.subjects[-1])
    source.unlink()
    source.symlink_to(outside)


@pytest.mark.parametrize(
    ("breaker", "reason"),
    [
        (_break_outside_root, "refused"),
        (_break_two_t1, "{last_subject}"),
        (_break_symlink_outside, "refused"),
    ],
    ids=["outside-root", "two-t1", "symlink-outside"],
)
def test_failed_run_leaves_data_dir_unchanged(sandbox: Sandbox, breaker, reason: str) -> None:
    sandbox.build()
    sandbox.data.mkdir(parents=True)
    (sandbox.data / "unrelated.txt").write_text("kept as is")
    breaker(sandbox)
    before = sandbox.tree_paths()
    result = sandbox.run()
    assert result.returncode != 0, "test premise: the broken source must make the run fail"
    assert reason.format(last_subject=sandbox.subjects[-1]) in result.stderr, result.stderr
    assert sandbox.tree_paths() == before


def test_failed_run_creates_no_data_dir(sandbox: Sandbox) -> None:
    sandbox.build()
    _break_two_t1(sandbox)
    result = sandbox.run()
    assert result.returncode != 0, "test premise: the second T1 must make the run fail"
    assert sandbox.subjects[-1] in result.stderr, result.stderr
    assert sandbox.tree_paths() == set()


# REQ-E2ECI-BDRABCZUK-82469329998E --------------------------------------------------


def test_settings_come_from_sibling_env_file(sandbox: Sandbox) -> None:
    sandbox.build()
    decoy_data = sandbox.root / "decoy" / "data"
    decoy_raw = sandbox.root / "decoy" / "rawdata"
    result = sandbox.run(
        extra_env={
            "E2E_DATA_DIR": str(decoy_data),
            "E2E_IXI_RAWDATA": str(decoy_raw),
            "E2E_ALLOWED_DATA_ROOTS": str(decoy_raw),
        }
    )
    assert result.returncode == 0, result.stderr
    assert (sandbox.tier / "MANIFEST").is_file()
    assert not decoy_data.exists()


def test_script_works_from_another_working_directory(sandbox: Sandbox, tmp_path: Path) -> None:
    sandbox.build()
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    result = sandbox.run(cwd=elsewhere)
    assert result.returncode == 0, result.stderr
    assert (sandbox.tier / "MANIFEST").is_file()
