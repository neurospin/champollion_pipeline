"""Tests for ``scripts/e2e/stage_data.sh``: the read-only QUICK data directory of the e2e runs.

TASK-E2ECI-BDRABCZUK-B59B0E64514D (epic TASK-E2ECI-BDRABCZUK-F3B0138F58F3).

The script stages the raw T1 of each ``E2E_TIER_QUICK_SUBJECTS`` subject (IXI, open
data) from ``E2E_IXI_RAWDATA`` into ``E2E_DATA_DIR/tier_quick`` in the two input
layouts stage 1 (``generate_morphologist_graphs.py``) accepts. Stage 1 lists only
the top-level files of its input directory, so both layouts are flat:

- ``tier_quick/bids/<source file name>`` (``--bids`` run, e.g. ``sub-002_ses-V1_T1w.nii.gz``);
- ``tier_quick/classic/<subject>.nii.gz`` (default ``morphologist-auto-nonoverlap-1.0`` run).

How the tests stay off rosette without weakening the allowlist: the script takes
every ``E2E_*`` value from the ``e2e.env`` next to it and calls the ``e2e_env.py``
next to it. Each test copies ``stage_data.sh`` and ``e2e_env.py`` into a temporary
``scripts/e2e/`` with an ``e2e.env`` made of the real file plus an appended block
that re-points ``E2E_IXI_RAWDATA``, ``E2E_ALLOWED_DATA_ROOTS`` and ``E2E_DATA_DIR`` at
a fake IXI tree under ``tmp_path``. The production script has no override switch;
the copy's own ``check-path`` still enforces the (fake) allowlist and the real
banned patterns.

Requirements (one test or more each):
- REQ-E2ECI-BDRABCZUK-F5F94B156571 BIDS layout: flat copies keeping the source name
- REQ-E2ECI-BDRABCZUK-5ED381AE7B34 classic layout: flat ``<subject>.nii.gz`` copies
- REQ-E2ECI-BDRABCZUK-5130638141E5 regular byte-identical copies, no symbolic link
- REQ-E2ECI-BDRABCZUK-B73FC1941888 exactly one T1 per QUICK subject
- REQ-E2ECI-BDRABCZUK-606D4E09C743 MANIFEST in sha256sum format
- REQ-E2ECI-BDRABCZUK-EBE4C3122CD3 ATTRIBUTION (IXI, CC BY-SA 3.0)
- REQ-E2ECI-BDRABCZUK-0D0675AC02C7 staged tree is read-only
- REQ-E2ECI-BDRABCZUK-C1AE8AC30BF8 re-run on an intact directory changes nothing
- REQ-E2ECI-BDRABCZUK-FBFF4297BCC9 re-run refuses a drifted directory
- REQ-E2ECI-BDRABCZUK-5D83BC5FDA7B failed run leaves E2E_DATA_DIR unchanged
- REQ-E2ECI-BDRABCZUK-604DDAFCB20D check-path on the resolved real path of each source
- REQ-E2ECI-BDRABCZUK-B35CE67DC2BB settings come from the sibling e2e.env
- REQ-E2ECI-BDRABCZUK-60B2E493BF3B (SET) check-path on each source, refusal exits non-zero
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
E2E_DIR = REPO / "scripts" / "e2e"
SCRIPT = E2E_DIR / "stage_data.sh"
CHECKER = E2E_DIR / "e2e_env.py"
ENV_FILE = E2E_DIR / "e2e.env"
BASH = shutil.which("bash")

TIER_DIRNAME = "tier_quick"
IXI_URL = "https://brain-development.org/ixi-dataset/"
LICENCE_URL = "https://creativecommons.org/licenses/by-sa/3.0/"
MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  (\S.*)$")
OTHER_SUBJECT = "sub-099"

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
        self.subjects = self.settings["E2E_TIER_QUICK_SUBJECTS"].split()
        self.e2e_dir = root / "repo" / "scripts" / "e2e"
        self.raw = root / "src" / "ixi" / "rawdata"
        self.morpho = root / "src" / "ixi" / "morphologist"
        self.data = root / "dest" / "data"
        self.tier = self.data / TIER_DIRNAME
        self.overrides: dict[str, str] = {}
        self.sources: dict[str, Path] = {}

    # -- setup ---------------------------------------------------------------

    def build(self, subjects: list[str] | None = None, **overrides: str) -> Sandbox:
        """Copy the scripts, write the e2e.env copy and the fake IXI tree."""
        assert SCRIPT.is_file(), f"{SCRIPT.relative_to(REPO)} is missing"
        self.e2e_dir.mkdir(parents=True)
        shutil.copy2(SCRIPT, self.e2e_dir / SCRIPT.name)
        shutil.copy2(CHECKER, self.e2e_dir / CHECKER.name)
        self.morpho.mkdir(parents=True)
        for subject in [*(subjects or self.subjects), OTHER_SUBJECT]:
            self.add_source(_t1_relpath(subject), f"fake T1 of {subject}")
            self.add_source(f"{subject}/ses-V1/anat/{subject}_ses-V1_T2w.nii.gz", f"fake T2 of {subject}")
        self.overrides = {
            "E2E_IXI_RAWDATA": str(self.raw),
            "E2E_IXI_MORPHOLOGIST_DIR": str(self.morpho),
            "E2E_ALLOWED_DATA_ROOTS": f"{self.raw} {self.morpho}",
            "E2E_DATA_DIR": str(self.data),
        }
        if subjects is not None:
            self.overrides["E2E_TIER_QUICK_SUBJECTS"] = " ".join(subjects)
            self.subjects = list(subjects)
        self.overrides.update(overrides)
        self.write_env()
        glob = self.settings["E2E_IXI_T1_GLOB"]
        for subject in self.subjects:
            matches = [p for p in self.raw.glob(glob) if p.relative_to(self.raw).parts[0] == subject]
            assert matches == [self.raw / _t1_relpath(subject)], f"test premise: {glob} misses the fake T1"
        return self

    def write_env(self) -> None:
        block = "".join(f'{key}="{value}"\n' for key, value in self.overrides.items())
        text = ENV_FILE.read_text() + "\n# --- test sandbox overrides ---\n" + block
        (self.e2e_dir / ENV_FILE.name).write_text(text)

    def add_source(self, relpath: str, text: str) -> Path:
        path = self.raw / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode() + os.urandom(256))
        self.sources[relpath] = path
        return path

    # -- run -----------------------------------------------------------------

    def run(self, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        env = {"HOME": str(self.home), "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.defpath}"}
        env.update(extra_env or {})
        return subprocess.run(
            [str(self.e2e_dir / SCRIPT.name)],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def stage(self) -> subprocess.CompletedProcess[str]:
        result = self.run()
        assert result.returncode == 0, f"stage_data.sh failed ({result.returncode}): {result.stderr}"
        return result

    # -- inspection ----------------------------------------------------------

    def source_of(self, subject: str) -> Path:
        return self.raw / _t1_relpath(subject)

    def tree_paths(self) -> set[str]:
        """Every path under E2E_DATA_DIR, relative to it (empty when absent)."""
        if not self.data.exists():
            return set()
        return {str(p.relative_to(self.data)) for p in self.data.rglob("*")}

    def snapshot(self) -> dict[str, tuple[str, int, int]]:
        """(sha256 or 'dir', mode, mtime_ns) of every path under E2E_DATA_DIR."""
        state = {}
        for path in sorted(self.data.rglob("*")):
            info = path.lstat()
            digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "dir"
            state[str(path.relative_to(self.data))] = (digest, info.st_mode, info.st_mtime_ns)
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


# REQ-E2ECI-BDRABCZUK-F5F94B156571 --------------------------------------------------


def test_bids_layout_holds_source_names(sandbox: Sandbox) -> None:
    sandbox.build().stage()
    bids = sandbox.tier / "bids"
    expected = sorted(sandbox.source_of(subject).name for subject in sandbox.subjects)
    assert sorted(p.name for p in bids.iterdir()) == expected


# REQ-E2ECI-BDRABCZUK-5ED381AE7B34 --------------------------------------------------


def test_classic_layout_holds_subject_names(sandbox: Sandbox) -> None:
    sandbox.build().stage()
    classic = sandbox.tier / "classic"
    assert sorted(p.name for p in classic.iterdir()) == sorted(f"{s}.nii.gz" for s in sandbox.subjects)


# REQ-E2ECI-BDRABCZUK-F5F94B156571 / REQ-E2ECI-BDRABCZUK-5ED381AE7B34 --------------


def test_staged_subjects_follow_the_settings(sandbox: Sandbox) -> None:
    sandbox.build(subjects=["sub-031", "sub-047", "sub-102"]).stage()
    assert sorted(p.name for p in (sandbox.tier / "classic").iterdir()) == [
        "sub-031.nii.gz",
        "sub-047.nii.gz",
        "sub-102.nii.gz",
    ]
    assert sorted(p.name for p in (sandbox.tier / "bids").iterdir()) == [
        "sub-031_ses-V1_T1w.nii.gz",
        "sub-047_ses-V1_T1w.nii.gz",
        "sub-102_ses-V1_T1w.nii.gz",
    ]


# REQ-E2ECI-BDRABCZUK-5130638141E5 --------------------------------------------------


def test_staged_images_are_regular_copies(sandbox: Sandbox) -> None:
    sandbox.build().stage()
    for subject in sandbox.subjects:
        source = sandbox.source_of(subject)
        for staged in (sandbox.tier / "bids" / source.name, sandbox.tier / "classic" / f"{subject}.nii.gz"):
            assert not staged.is_symlink(), f"{staged} is a symbolic link"
            assert stat.S_ISREG(staged.lstat().st_mode), f"{staged} is not a regular file"
            assert staged.read_bytes() == source.read_bytes(), f"{staged} differs from {source}"


# REQ-E2ECI-BDRABCZUK-B73FC1941888 --------------------------------------------------


def test_subject_without_t1_is_refused(sandbox: Sandbox) -> None:
    sandbox.build()
    missing = sandbox.subjects[-1]
    sandbox.source_of(missing).unlink()
    result = sandbox.run()
    assert result.returncode != 0
    assert missing in result.stderr, result.stderr


def test_subject_with_two_t1_is_refused(sandbox: Sandbox) -> None:
    sandbox.build()
    doubled = sandbox.subjects[0]
    sandbox.add_source(f"{doubled}/ses-V1/anat/{doubled}_ses-V1_run-2_T1w.nii.gz", "second T1")
    result = sandbox.run()
    assert result.returncode != 0
    assert doubled in result.stderr, result.stderr


# REQ-E2ECI-BDRABCZUK-606D4E09C743 --------------------------------------------------


def test_manifest_lists_checksums_of_other_files(sandbox: Sandbox) -> None:
    sandbox.build().stage()
    manifest = sandbox.tier / "MANIFEST"
    lines = manifest.read_text().splitlines()
    parsed = [MANIFEST_LINE.match(line) for line in lines]
    bad = [line for line in lines if not MANIFEST_LINE.match(line)]
    assert not bad, f"lines not in sha256sum format: {bad}"
    listed = {m.group(2): m.group(1) for m in parsed}
    assert len(listed) == len(lines), "a path is listed twice"
    others = {
        str(p.relative_to(sandbox.tier)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sandbox.tier.rglob("*")
        if p.is_file() and p != manifest
    }
    assert listed == others
    paths = [m.group(2) for m in parsed]
    assert paths == sorted(paths), "MANIFEST is not sorted by path"


def test_manifest_passes_sha256sum_check(sandbox: Sandbox) -> None:
    sandbox.build().stage()
    sha256sum = shutil.which("sha256sum")
    if sha256sum is None:
        pytest.skip("sha256sum not available")
    result = subprocess.run(
        [sha256sum, "--check", "--strict", "MANIFEST"],
        cwd=sandbox.tier,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# REQ-E2ECI-BDRABCZUK-EBE4C3122CD3 --------------------------------------------------


def test_attribution_names_dataset_and_licence(sandbox: Sandbox) -> None:
    sandbox.build().stage()
    text = (sandbox.tier / "ATTRIBUTION").read_text()
    licence = sandbox.settings["E2E_DATA_LICENSE"]
    for needle in ("IXI", IXI_URL, licence, LICENCE_URL):
        assert needle in text, f"ATTRIBUTION lacks {needle!r}"


# REQ-E2ECI-BDRABCZUK-0D0675AC02C7 --------------------------------------------------


def test_staged_tree_has_no_write_bit(sandbox: Sandbox) -> None:
    sandbox.build().stage()
    paths = [sandbox.tier, *sandbox.tier.rglob("*")]
    writable = [str(p) for p in paths if p.lstat().st_mode & 0o222]
    assert not writable, f"paths with a write bit: {writable}"


# REQ-E2ECI-BDRABCZUK-C1AE8AC30BF8 --------------------------------------------------


def test_rerun_on_intact_directory_changes_nothing(sandbox: Sandbox) -> None:
    sandbox.build().stage()
    before = sandbox.snapshot()
    result = sandbox.run()
    assert result.returncode == 0, result.stderr
    assert sandbox.snapshot() == before


# REQ-E2ECI-BDRABCZUK-FBFF4297BCC9 --------------------------------------------------


def test_rerun_refuses_altered_file(sandbox: Sandbox) -> None:
    sandbox.build().stage()
    target = sandbox.tier / "bids" / sandbox.source_of(sandbox.subjects[0]).name
    with _writable(target):
        target.write_bytes(target.read_bytes() + b"drift")
    result = sandbox.run()
    assert result.returncode != 0
    assert target.name in result.stderr, result.stderr


def test_rerun_refuses_missing_file(sandbox: Sandbox) -> None:
    sandbox.build().stage()
    target = sandbox.tier / "classic" / f"{sandbox.subjects[0]}.nii.gz"
    with _writable(target.parent):
        target.unlink()
    result = sandbox.run()
    assert result.returncode != 0
    assert target.name in result.stderr, result.stderr


def test_rerun_refuses_unlisted_file(sandbox: Sandbox) -> None:
    sandbox.build().stage()
    extra = sandbox.tier / "bids" / "sub-077_ses-V1_T1w.nii.gz"
    with _writable(extra.parent):
        extra.write_bytes(b"not staged")
    result = sandbox.run()
    assert result.returncode != 0
    assert extra.name in result.stderr, result.stderr


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


# REQ-E2ECI-BDRABCZUK-604DDAFCB20D --------------------------------------------------


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


# REQ-E2ECI-BDRABCZUK-5D83BC5FDA7B --------------------------------------------------


def _break_outside_root(box: Sandbox) -> None:
    box.overrides["E2E_ALLOWED_DATA_ROOTS"] = str(box.morpho)
    box.write_env()


def _break_missing_t1(box: Sandbox) -> None:
    box.source_of(box.subjects[-1]).unlink()


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
        (_break_missing_t1, "{last_subject}"),
        (_break_two_t1, "{last_subject}"),
        (_break_symlink_outside, "refused"),
    ],
    ids=["outside-root", "missing-t1", "two-t1", "symlink-outside"],
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
    _break_missing_t1(sandbox)
    result = sandbox.run()
    assert result.returncode != 0, "test premise: the missing T1 must make the run fail"
    assert sandbox.subjects[-1] in result.stderr, result.stderr
    assert sandbox.tree_paths() == set()


# REQ-E2ECI-BDRABCZUK-B35CE67DC2BB --------------------------------------------------


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
    env = {"HOME": str(sandbox.home), "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.defpath}"}
    result = subprocess.run(
        [str(sandbox.e2e_dir / SCRIPT.name)], cwd=elsewhere, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert (sandbox.tier / "MANIFEST").is_file()
