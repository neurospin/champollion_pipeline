"""Tests for the e2e shared settings file ``scripts/e2e/e2e.env`` and its allowlist check.

TASK-E2ECI-BDRABCZUK-EF4E486F027F (SET, epic TASK-E2ECI-BDRABCZUK-F3B0138F58F3).
``scripts/e2e/e2e.env`` is the single source of the e2e paths, names, tier table
and open-data allowlist; ``scripts/e2e/e2e_env.py check-path PATH`` is the one
allowlist check every data-touching e2e step calls.

The settings are read the way the e2e shell scripts read them: the file is sourced
by ``bash -eu`` with ``set -a`` in a clean environment (only ``HOME`` and ``PATH``),
then every exported ``E2E_*`` variable is collected. ``HOME`` points at a temporary
directory so ``$HOME``-relative values are checked exactly.

Requirements (one test or more each):
- REQ-E2ECI-BDRABCZUK-0DD4B9A9BA05 file format (comments, blanks, E2E_ assignments, no command substitution)
- REQ-E2ECI-BDRABCZUK-10CCB449125E rosette root and its sifs/data/runs/reports/code subdirectories
- REQ-E2ECI-BDRABCZUK-746496CB0463 local state dir, last_quick_gated file, local report store
- REQ-E2ECI-BDRABCZUK-A3BF58F4DCD0 env SIF and code directory name templates
- REQ-E2ECI-BDRABCZUK-713C0EA881BC report-store layout template
- REQ-E2ECI-BDRABCZUK-83EEA6228C09 cc26.1 models dir, Hugging Face disabled
- REQ-E2ECI-BDRABCZUK-AFC4FAEA5DEC env SIF retention (3)
- REQ-E2ECI-BDRABCZUK-A3C7B8A7CA48 WIP branch name
- REQ-E2ECI-BDRABCZUK-44984A6D62C4 allowlist is the two IXI roots
- REQ-E2ECI-BDRABCZUK-A1A489677E2B IXI T1 glob and licence
- REQ-E2ECI-BDRABCZUK-08D729E7A2AE check-path accepts allowlisted paths, refuses others
- REQ-E2ECI-BDRABCZUK-2D5D903725CD check-path refuses DEMO01 and UK Biobank by pattern
- REQ-E2ECI-BDRABCZUK-04C9E57D13AE no banned dataset path among the values
- REQ-E2ECI-BDRABCZUK-CCFDACC7DA2A no secret in the file
- REQ-E2ECI-BDRABCZUK-61675FBA45CE tier list and main-gate tiers
- REQ-E2ECI-BDRABCZUK-B0668039D74C QUICK tier row
- REQ-E2ECI-BDRABCZUK-FAFDA738F26C NIGHTLY tier row
- REQ-E2ECI-BDRABCZUK-41E4A0CB54BA WEEKLY tier row
- REQ-E2ECI-BDRABCZUK-7D703A43E1C5 no other scripts/e2e file restates the owned paths
- REQ-E2ECI-BDRABCZUK-60B2E493BF3B data-touching e2e scripts call check-path
- REQ-E2ECI-BDRABCZUK-20D1499D1431 named IXI data roots (supersedes -4FE68FD50A5E: morphologist-2021 root)
- REQ-E2ECI-BDRABCZUK-5EACD169E3D9 profile subdirectory name inside each run directory
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
E2E_DIR = REPO / "scripts" / "e2e"
ENV_FILE = E2E_DIR / "e2e.env"
CHECKER = E2E_DIR / "e2e_env.py"
BASH = shutil.which("bash")

ROSETTE_ROOT = "/neurospin/dico/bdrabczuk/champollion_e2e"
MODELS_DIR = "/neurospin/dico/bdrabczuk/models/Champollion_V1/canonical_corrected_26_1"
IXI_RAWDATA = "/neurospin/psy/ixi/rawdata"
IXI_MORPHOLOGIST = "/neurospin/psy/ixi/derivatives/morphologist-2021/subjects"
# The former root, which holds no sulcal graph (user decision 2026-10-28), is no longer allowed.
FORMER_IXI_MORPHOLOGIST = "/neurospin/dico/data/human/openbhb/ixi/derivatives/morphologist-6.0/subjects"

ASSIGNMENT = re.compile(r"^E2E_[A-Z0-9_]+=")
COMMENT_OR_BLANK = re.compile(r"^\s*(#.*)?$")
COMMAND_SUBSTITUTION = ("$(", "`", "<(", ">(")
SECRET_NAME = re.compile(r"TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.IGNORECASE)
SECRET_VALUE = re.compile(r"hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----")
# Canonical spellings of the banned datasets (independent of the file's own patterns).
BANNED_REFERENCE = re.compile(r"demo01|uk[\W_]*bio[\W_]*bank|ukb", re.IGNORECASE)
BANNED_SAMPLES = ("DEMO01", "demo01", "UkBioBank", "UKBioBank", "ukbiobank", "uk_biobank", "UK-Biobank", "ukb")
DATA_KEYS = ("E2E_DATA_DIR", "E2E_IXI_RAWDATA", "E2E_IXI_MORPHOLOGIST_DIR", "E2E_ALLOWED_DATA_ROOTS")

pytestmark = pytest.mark.skipif(BASH is None, reason="bash executable not available")  # noqa: V107


def _env_text() -> str:
    assert ENV_FILE.is_file(), f"{ENV_FILE.relative_to(REPO)} is missing"
    return ENV_FILE.read_text()


def _assignments() -> list[tuple[str, str]]:
    """(name, raw value) for each assignment line of the file."""
    pairs = []
    for line in _env_text().splitlines():
        if ASSIGNMENT.match(line):
            name, value = line.split("=", 1)
            pairs.append((name, value))
    return pairs


def _clean_env(home: Path) -> dict[str, str]:
    return {"HOME": str(home), "PATH": os.defpath}


def _source(home: Path) -> dict[str, str]:
    """Source the file with ``bash -eu`` and ``set -a``; return the exported E2E_* variables."""
    _env_text()
    result = subprocess.run(
        [BASH, "-euc", 'set -a; source "$1"; set +a; env -0', "_", str(ENV_FILE)],
        env=_clean_env(home),
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    settings = {}
    for item in result.stdout.split(b"\0"):
        name, _, value = item.decode().partition("=")
        if name.startswith("E2E_"):
            settings[name] = value
    return settings


@pytest.fixture
def home(tmp_path: Path) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    return path


def _check_path(path: str, home: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), "check-path", path],
        cwd=REPO,
        env=_clean_env(home),
        capture_output=True,
        text=True,
        timeout=60,
    )


# REQ-E2ECI-BDRABCZUK-0DD4B9A9BA05 -------------------------------------------------


def test_env_file_lines_are_comments_blanks_or_assignments() -> None:
    bad = [line for line in _env_text().splitlines() if not (COMMENT_OR_BLANK.match(line) or ASSIGNMENT.match(line))]
    assert not bad, f"lines that are neither comment, blank nor E2E_KEY=value: {bad}"


def test_env_file_has_no_command_substitution() -> None:
    bad = [(name, value) for name, value in _assignments() if any(token in value for token in COMMAND_SUBSTITUTION)]
    assert not bad, f"values with command or process substitution: {bad}"


def test_env_file_sources_cleanly_in_bash(home: Path) -> None:
    _env_text()
    result = subprocess.run(
        [BASH, "-euc", 'source "$1"', "_", str(ENV_FILE)],
        env=_clean_env(home),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == "", result.stderr
    assert result.stdout == "", result.stdout


# REQ-E2ECI-BDRABCZUK-10CCB449125E -------------------------------------------------


def test_rosette_root(home: Path) -> None:
    settings = _source(home)
    assert settings.get("E2E_ROSETTE_ROOT") == ROSETTE_ROOT


@pytest.mark.parametrize(
    ("key", "subdir"),
    [
        ("E2E_SIFS_DIR", "sifs"),
        ("E2E_DATA_DIR", "data"),
        ("E2E_RUNS_DIR", "runs"),
        ("E2E_REPORTS_DIR", "reports"),
        ("E2E_CODE_DIR", "code"),
    ],
)
def test_rosette_subdirectories(home: Path, key: str, subdir: str) -> None:
    settings = _source(home)
    assert settings.get(key) == f"{ROSETTE_ROOT}/{subdir}"


# REQ-E2ECI-BDRABCZUK-746496CB0463 -------------------------------------------------


def test_local_state_dir(home: Path) -> None:
    settings = _source(home)
    assert settings.get("E2E_STATE_DIR") == f"{home}/.local/state/champollion_e2e"


def test_last_quick_gated_file(home: Path) -> None:
    settings = _source(home)
    assert settings.get("E2E_LAST_QUICK_GATED_FILE") == f"{home}/.local/state/champollion_e2e/last_quick_gated"


def test_local_reports_dir(home: Path) -> None:
    settings = _source(home)
    assert settings.get("E2E_LOCAL_REPORTS_DIR") == f"{home}/.local/state/champollion_e2e/reports"


# REQ-E2ECI-BDRABCZUK-A3BF58F4DCD0 -------------------------------------------------


def test_env_sif_name_template(home: Path) -> None:
    settings = _source(home)
    assert settings.get("E2E_ENV_SIF_NAME") == "champollion-env-{key12}.sif"


def test_code_dir_name_template(home: Path) -> None:
    settings = _source(home)
    assert settings.get("E2E_CODE_DIR_NAME") == "{commit12}"


# REQ-E2ECI-BDRABCZUK-713C0EA881BC -------------------------------------------------


def test_report_relpath_template(home: Path) -> None:
    settings = _source(home)
    template = settings.get("E2E_REPORT_RELPATH")
    assert template == "{commit12}/{tier}-{layout}/report.json"
    assert template.format(commit12="0123456789ab", tier="quick", layout="bids") == (
        "0123456789ab/quick-bids/report.json"
    )


# REQ-E2ECI-BDRABCZUK-83EEA6228C09 -------------------------------------------------


def test_models_dir(home: Path) -> None:
    settings = _source(home)
    assert settings.get("E2E_MODELS_DIR") == MODELS_DIR


def test_hf_disabled(home: Path) -> None:
    settings = _source(home)
    assert settings.get("E2E_HF_ENABLED") == "false"


# REQ-E2ECI-BDRABCZUK-AFC4FAEA5DEC / REQ-E2ECI-BDRABCZUK-A3C7B8A7CA48 --------------


def test_env_sif_retention(home: Path) -> None:
    settings = _source(home)
    assert settings.get("E2E_ENV_SIF_KEEP") == "3"


def test_wip_branch(home: Path) -> None:
    settings = _source(home)
    assert settings.get("E2E_WIP_BRANCH") == "WIP"


# REQ-E2ECI-BDRABCZUK-5EACD169E3D9 -------------------------------------------------


def test_run_profile_dirname(home: Path) -> None:
    settings = _source(home)
    assert settings.get("E2E_RUN_PROFILE_DIRNAME") == "profile"


# REQ-E2ECI-BDRABCZUK-44984A6D62C4 / REQ-E2ECI-BDRABCZUK-20D1499D1431 --------------


def test_allowed_data_roots_are_ixi_only(home: Path) -> None:
    settings = _source(home)
    roots = settings.get("E2E_ALLOWED_DATA_ROOTS", "").split()
    assert sorted(roots) == sorted([IXI_RAWDATA, IXI_MORPHOLOGIST])


def test_ixi_named_roots(home: Path) -> None:
    settings = _source(home)
    assert settings.get("E2E_IXI_RAWDATA") == IXI_RAWDATA
    assert settings.get("E2E_IXI_MORPHOLOGIST_DIR") == IXI_MORPHOLOGIST


# REQ-E2ECI-BDRABCZUK-A1A489677E2B -------------------------------------------------


def test_ixi_t1_glob(home: Path) -> None:
    settings = _source(home)
    assert settings.get("E2E_IXI_T1_GLOB") == "sub-*/ses-V1/anat/*_T1w.nii.gz"


def test_data_license(home: Path) -> None:
    settings = _source(home)
    assert settings.get("E2E_DATA_LICENSE") == "CC BY-SA 3.0"


# REQ-E2ECI-BDRABCZUK-08D729E7A2AE -------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        f"{IXI_RAWDATA}/sub-002/ses-V1/anat/sub-002_ses-V1_T1w.nii.gz",
        f"{IXI_RAWDATA}/sub-012/ses-V1/anat",
        f"{IXI_MORPHOLOGIST}/sub-012",
        f"{IXI_RAWDATA}/sub-002/../sub-012/ses-V1",
    ],
)
def test_check_path_accepts_ixi_paths(home: Path, path: str) -> None:
    result = _check_path(path, home)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/sub-002_T1w.nii.gz",
        f"{IXI_RAWDATA}_other/sub-002",
        "/neurospin/dico/data/human/openbhb/ixi/derivatives/morphologist-5.2/subjects/sub-002",
        "/neurospin/psy/ixi",
        f"{FORMER_IXI_MORPHOLOGIST}/sub-002",
    ],
)
def test_check_path_refuses_outside_allowlist(home: Path, path: str) -> None:
    result = _check_path(path, home)
    assert result.returncode == 1, result.stderr
    assert "Traceback" not in result.stderr, result.stderr


@pytest.mark.parametrize(
    "path",
    [
        f"{IXI_RAWDATA}/../../../../etc/passwd",
        f"{IXI_MORPHOLOGIST}/../../morphologist-5.2/subjects/sub-002",
    ],
)
def test_check_path_refuses_dotdot_escape(home: Path, path: str) -> None:
    result = _check_path(path, home)
    assert result.returncode == 1, result.stderr
    assert "Traceback" not in result.stderr, result.stderr


# REQ-E2ECI-BDRABCZUK-2D5D903725CD -------------------------------------------------


@pytest.mark.parametrize("sample", BANNED_SAMPLES)
def test_banned_patterns_cover_demo01_and_uk_biobank(home: Path, sample: str) -> None:
    settings = _source(home)
    patterns = settings.get("E2E_BANNED_DATA_PATTERNS", "").split()
    assert any(re.search(pattern, sample, re.IGNORECASE) for pattern in patterns), (
        f"no E2E_BANNED_DATA_PATTERNS entry matches {sample!r}: {patterns}"
    )


@pytest.mark.parametrize(
    "path",
    [
        f"{IXI_RAWDATA}/DEMO01/sub-1_T1w.nii.gz",
        f"{IXI_MORPHOLOGIST}/UkBioBank/sub-1000",
        f"{IXI_RAWDATA}/uk_biobank/sub-1000/ses-V1/anat",
        "/volatile/someone/champollion_pipeline/data/DEMO01/t1.nii.gz",
        "/neurospin/dico/data/human/UkBioBank/sub-1000/t1mri",
    ],
)
def test_check_path_refuses_banned_names(home: Path, path: str) -> None:
    settings = _source(home)
    patterns = settings.get("E2E_BANNED_DATA_PATTERNS", "").split()
    matched = [pattern for pattern in patterns if re.search(pattern, path, re.IGNORECASE)]
    assert matched, f"test premise: no E2E_BANNED_DATA_PATTERNS entry matches {path}"
    result = _check_path(path, home)
    assert result.returncode == 1, result.stderr
    assert any(pattern in result.stderr for pattern in matched), (
        f"stderr names none of the matched patterns {matched}: {result.stderr!r}"
    )


# REQ-E2ECI-BDRABCZUK-04C9E57D13AE -------------------------------------------------


def test_env_file_holds_no_banned_path(home: Path) -> None:
    settings = _source(home)
    patterns = settings.get("E2E_BANNED_DATA_PATTERNS", "").split()
    offending = {
        name: value
        for name, value in settings.items()
        if name != "E2E_BANNED_DATA_PATTERNS"
        and (BANNED_REFERENCE.search(value) or any(re.search(p, value, re.IGNORECASE) for p in patterns))
    }
    assert not offending, f"values naming a banned dataset: {offending}"


# REQ-E2ECI-BDRABCZUK-CCFDACC7DA2A -------------------------------------------------


def test_env_file_has_no_secret_names() -> None:
    names = [name for name, _ in _assignments() if SECRET_NAME.search(name)]
    assert not names, f"secret-like variable names: {names}"


def test_env_file_has_no_secret_values() -> None:
    hits = SECRET_VALUE.findall(_env_text())
    assert not hits, f"secret-shaped values: {len(hits)} found"


# REQ-E2ECI-BDRABCZUK-61675FBA45CE -------------------------------------------------


def test_tier_list(home: Path) -> None:
    settings = _source(home)
    assert settings.get("E2E_TIERS", "").split() == ["quick", "nightly", "weekly"]


def test_main_gate_tiers(home: Path) -> None:
    settings = _source(home)
    assert settings.get("E2E_MAIN_GATE_TIERS", "").split() == ["quick", "nightly"]


# REQ-E2ECI-BDRABCZUK-B0668039D74C / -FAFDA738F26C / -41E4A0CB54BA -----------------


def _row(settings: dict[str, str], tier: str, fields: tuple[str, ...]) -> dict[str, str]:
    prefix = f"E2E_TIER_{tier.upper()}_"
    return {field: settings.get(prefix + field) for field in fields}


def _words(row: dict[str, str], *fields: str) -> dict[str, list[str] | None]:
    return {field: row[field].split() if row[field] is not None else None for field in fields}


def test_quick_tier_row(home: Path) -> None:
    settings = _source(home)
    row = _row(
        settings, "quick", ("TRIGGER", "SUBJECTS", "LAYOUTS", "STAGES", "INPUT", "MODE", "RESUMABLE", "TIMEOUT_S")
    )
    assert _words(row, "SUBJECTS", "LAYOUTS", "STAGES") == {
        "SUBJECTS": ["sub-002", "sub-012"],
        "LAYOUTS": ["bids", "classic"],
        "STAGES": ["1", "2", "3", "4", "5", "6"],
    }
    assert {k: row[k] for k in ("TRIGGER", "INPUT", "MODE", "RESUMABLE")} == {
        "TRIGGER": "pre-push",
        "INPUT": "raw",
        "MODE": "foreground",
        "RESUMABLE": "no",
    }
    timeout = row["TIMEOUT_S"] or ""
    assert timeout.isdigit() and int(timeout) > 0, f"E2E_TIER_QUICK_TIMEOUT_S={timeout!r}"


def test_nightly_tier_row(home: Path) -> None:
    settings = _source(home)
    row = _row(settings, "nightly", ("TRIGGER", "SUBJECTS", "LAYOUTS", "STAGES", "INPUT", "MODE"))
    assert _words(row, "LAYOUTS", "STAGES") == {"LAYOUTS": ["bids", "classic"], "STAGES": ["2", "3", "4", "5", "6"]}
    assert {k: row[k] for k in ("TRIGGER", "SUBJECTS", "INPUT", "MODE")} == {
        "TRIGGER": "nightly",
        "SUBJECTS": "cohort",
        "INPUT": "existing-morphologist",
        "MODE": "detached",
    }


def test_weekly_tier_row(home: Path) -> None:
    settings = _source(home)
    row = _row(settings, "weekly", ("TRIGGER", "SUBJECTS", "LAYOUTS", "STAGES", "INPUT", "MODE", "RESUMABLE"))
    assert _words(row, "LAYOUTS", "STAGES") == {
        "LAYOUTS": ["bids", "classic"],
        "STAGES": ["1", "2", "3", "4", "5", "6"],
    }
    assert {k: row[k] for k in ("TRIGGER", "SUBJECTS", "INPUT", "MODE", "RESUMABLE")} == {
        "TRIGGER": "weekly",
        "SUBJECTS": "cohort",
        "INPUT": "raw",
        "MODE": "detached",
        "RESUMABLE": "yes",
    }


# REQ-E2ECI-BDRABCZUK-7D703A43E1C5 -------------------------------------------------


def _other_e2e_files() -> list[Path]:
    return [
        path
        for path in sorted(E2E_DIR.rglob("*"))
        if path.is_file() and path != ENV_FILE and "__pycache__" not in path.parts
    ]


def test_no_other_e2e_file_restates_owned_paths(home: Path) -> None:
    settings = _source(home)
    owned = [settings.get("E2E_ROSETTE_ROOT"), settings.get("E2E_MODELS_DIR")]
    owned += settings.get("E2E_ALLOWED_DATA_ROOTS", "").split()
    assert all(owned), f"owned values missing from e2e.env: {owned}"
    offenders = [
        f"{path.relative_to(REPO)}: {value}"
        for path in _other_e2e_files()
        for value in owned
        if value in path.read_text(errors="ignore")
    ]
    assert not offenders, f"files restating e2e.env values: {offenders}"


# REQ-E2ECI-BDRABCZUK-60B2E493BF3B -------------------------------------------------


def test_data_touching_scripts_call_check_path(home: Path) -> None:
    """A shell script referencing a data key of e2e.env must call ``e2e_env.py check-path``.

    Static half of the rule: each data task (stage_data, stage_cohort_view, run_e2e) proves
    the refusal behaviour in its own tests.
    """
    settings = _source(home)
    missing = [key for key in DATA_KEYS if key not in settings]
    assert not missing, f"data keys missing from e2e.env: {missing}"
    offenders = [
        str(path.relative_to(REPO))
        for path in sorted(E2E_DIR.rglob("*.sh"))
        if any(key in path.read_text() for key in DATA_KEYS) and "e2e_env.py check-path" not in path.read_text()
    ]
    assert not offenders, f"data-touching scripts without the allowlist check: {offenders}"
