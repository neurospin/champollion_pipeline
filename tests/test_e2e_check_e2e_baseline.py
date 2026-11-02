"""Tests for the baseline comparison of ``scripts/e2e/check_e2e.py`` (BASEL).

TASK-E2ECI-BDRABCZUK-8367BEF3BD15 (epic TASK-E2ECI-BDRABCZUK-F3B0138F58F3).
``python3 scripts/e2e/check_e2e.py <run dir> --tier <tier> --layout <layout>`` compares
the run's facts in report.json with the baseline file ``<dir>/<tier>-<layout>.json``,
a file with the same key paths as report.json. ``<dir>`` is the ``--baselines-dir``
value, else E2E_BASELINES_DIR of scripts/e2e/e2e.env: a shared rosette directory
outside the per-commit code copy and outside git (user decision 2026-11-02). Any
difference in per-stage exit codes, subject counts, QC counts, failed-subject ids or
region-side coverage, an improvement included, fails the run and names the key and
both values. Only ``--accept-baseline`` on a run with no hard failure writes the
baseline; a missing baseline is reported as "no baseline". Without
``--tier``/``--layout`` the checker behaves as in tests/test_e2e_check_e2e.py.

Each test runs a copy of scripts/e2e/{check_e2e.py,e2e_env.py,e2e.env} under tmp_path
and points the baselines directory at tmp_path (``--baselines-dir``, or an
E2E_BASELINES_DIR line appended to the copied e2e.env), so no rosette path and no
repository file is read or written. Data is synthetic; the only subject ids are the
public IXI ids of the QUICK tier.

Requirements (one test or more each):
- REQ-E2ECI-BDRABCZUK-0AAC5075D401 default baseline directory from e2e.env
- REQ-E2ECI-BDRABCZUK-BD06C3231956 --baselines-dir overrides the directory
- REQ-E2ECI-BDRABCZUK-77D9FFFDFBAF E2E_BASELINES_DIR shared, outside the code copy
- REQ-E2ECI-BDRABCZUK-744F14261317 any baseline fact deviation fails
- REQ-E2ECI-BDRABCZUK-4D2888B44DA5 unreadable baseline file fails
- REQ-E2ECI-BDRABCZUK-C01572FFFB22 baseline file unchanged without --accept-baseline
- REQ-E2ECI-BDRABCZUK-EFFF10F1F399 report.json key baseline.differences
- REQ-E2ECI-BDRABCZUK-12443B519CC1 missing baseline is reported, not a hard failure
- REQ-E2ECI-BDRABCZUK-8B651F44BF51 --accept-baseline writes the baseline from a clean run
- REQ-E2ECI-BDRABCZUK-23B256F2E0C6 a failing run never writes the baseline
- REQ-E2ECI-BDRABCZUK-BE3EEE3CE81F --accept-baseline without tier and layout is a usage error
- REQ-E2ECI-BDRABCZUK-C15554596494 unknown tier is a usage error
- REQ-E2ECI-BDRABCZUK-5A73313FCF52 unknown layout is a usage error
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_e2e_check_e2e import (
    COMBINED_DIR,
    E2E_DIR,
    LEFT,
    MORPHOLOGIST_QC,
    REPO,
    SNAPSHOTS_QC,
    STAGES,
    SUB_A,
    SUB_B,
    _make_run,
    _write_qc,
    _write_stages,
)

TIER = "quick"
LAYOUT = "bids"
COPIED = ("check_e2e.py", "e2e_env.py", "e2e.env")
BASELINE_FACTS = ("exit_code", "subjects", "qc", "failed_subjects")


def _e2e_copy(tmp_path: Path) -> Path:
    """A private copy of scripts/e2e/ holding the checker, its settings reader and e2e.env."""
    e2e_dir = tmp_path / "repo" / "scripts" / "e2e"
    e2e_dir.mkdir(parents=True)
    for name in COPIED:
        shutil.copy2(E2E_DIR / name, e2e_dir / name)
    return e2e_dir


def _baselines(e2e_dir: Path) -> Path:
    """The test's shared baselines directory, outside the copied code (tmp_path/shared/baselines)."""
    return e2e_dir.parents[2] / "shared" / "baselines"


def _baseline_path(e2e_dir: Path) -> Path:
    return _baselines(e2e_dir) / f"{TIER}-{LAYOUT}.json"


def _set_default_baselines_dir(e2e_dir: Path, directory: Path) -> None:
    """Make ``directory`` the E2E_BASELINES_DIR of the copied e2e.env (a later assignment wins)."""
    with (e2e_dir / "e2e.env").open("a") as handle:
        handle.write(f"\nE2E_BASELINES_DIR={directory}\n")


def _check(e2e_dir: Path, run: Path, *options: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(e2e_dir / "check_e2e.py"), str(run), *options],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _check_tier(
    e2e_dir: Path, run: Path, *extra: str, tier: str = TIER, layout: str = LAYOUT
) -> subprocess.CompletedProcess:
    """A check with --tier/--layout against the test's baselines directory."""
    options = ("--tier", tier, "--layout", layout, "--baselines-dir", str(_baselines(e2e_dir)))
    return _check(e2e_dir, run, *options, *extra)


def _baseline_facts(document: dict) -> dict:
    """The baseline facts of a report.json-shaped dict: per-stage facts and region_sides."""
    stages = {
        stage: {key: value for key, value in facts.items() if key in BASELINE_FACTS}
        for stage, facts in document.get("stages", {}).items()
    }
    return {"stages": stages, "region_sides": document.get("region_sides")}


def _facts(e2e_dir: Path, run: Path) -> dict:
    """The run's baseline facts, from report.json of a plain (no --tier) check."""
    _check(e2e_dir, run)
    return _baseline_facts(_report(run))


def _write_baseline(e2e_dir: Path, baseline: dict, path: Path | None = None) -> Path:
    path = path or _baseline_path(e2e_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2) + "\n")
    return path


def _setup(tmp_path: Path) -> tuple[Path, Path, dict]:
    """A clean run, a private scripts/e2e/ copy and the run's own facts as a baseline dict."""
    e2e_dir = _e2e_copy(tmp_path)
    run = _make_run(tmp_path / "run")
    return e2e_dir, run, _facts(e2e_dir, run)


def _line_naming(key: str, stderr: str) -> str:
    """The first stderr line that names ``key``; fails the test when there is none."""
    lines = [line for line in stderr.splitlines() if key in line]
    assert lines, f"no stderr line names {key}: {stderr!r}"
    return lines[0]


def _report(run: Path) -> dict:
    return json.loads((run / "report.json").read_text())


def _repo_settings() -> dict[str, str]:
    """The settings of the repository's scripts/e2e/e2e.env, read through scripts/e2e/e2e_env.py."""
    spec = importlib.util.spec_from_file_location("e2e_env_for_baseline_tests", E2E_DIR / "e2e_env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_settings()


# REQ-E2ECI-BDRABCZUK-0AAC5075D401 ---------------------------------------------------


def test_default_baseline_dir_from_e2e_env(tmp_path: Path) -> None:
    e2e_dir, run, baseline = _setup(tmp_path)
    default_dir = tmp_path / "rosette" / "baselines"
    _set_default_baselines_dir(e2e_dir, default_dir)
    baseline["stages"]["run_cortical_tiles"]["subjects"] = 3  # the run has 2
    _write_baseline(e2e_dir, baseline, default_dir / f"{TIER}-{LAYOUT}.json")
    result = _check(e2e_dir, run, "--tier", TIER, "--layout", LAYOUT)
    assert result.returncode == 1, result.stderr
    _line_naming("stages.run_cortical_tiles.subjects", result.stderr)
    assert not (e2e_dir / "baselines").exists(), "a baseline directory was created next to the code"


def test_accept_baseline_default_dir(tmp_path: Path) -> None:
    e2e_dir, run, baseline = _setup(tmp_path)
    default_dir = tmp_path / "rosette" / "baselines"
    _set_default_baselines_dir(e2e_dir, default_dir)
    result = _check(e2e_dir, run, "--tier", TIER, "--layout", LAYOUT, "--accept-baseline")
    assert result.returncode != 2, f"--accept-baseline refused as a usage error: {result.stderr}"
    written = default_dir / f"{TIER}-{LAYOUT}.json"
    assert written.is_file(), result.stderr
    assert _baseline_facts(json.loads(written.read_text())) == baseline
    assert not (e2e_dir / "baselines").exists(), "a baseline directory was created next to the code"


# REQ-E2ECI-BDRABCZUK-BD06C3231956 ---------------------------------------------------


def test_baselines_dir_option_overrides_default(tmp_path: Path) -> None:
    e2e_dir, run, baseline = _setup(tmp_path)
    default_dir = tmp_path / "rosette" / "baselines"
    _set_default_baselines_dir(e2e_dir, default_dir)
    _write_baseline(e2e_dir, baseline, default_dir / f"{TIER}-{LAYOUT}.json")  # matches the run
    deviating = json.loads(json.dumps(baseline))
    deviating["stages"]["generate_embeddings"]["subjects"] = 2  # the run has 1
    _write_baseline(e2e_dir, deviating)  # under --baselines-dir
    result = _check_tier(e2e_dir, run)
    assert result.returncode == 1, result.stderr
    _line_naming("stages.generate_embeddings.subjects", result.stderr)


# REQ-E2ECI-BDRABCZUK-77D9FFFDFBAF ---------------------------------------------------


def test_e2e_env_baselines_dir_is_shared() -> None:
    settings = _repo_settings()
    assert "E2E_BASELINES_DIR" in settings, "scripts/e2e/e2e.env does not set E2E_BASELINES_DIR"
    baselines = Path(settings["E2E_BASELINES_DIR"])
    assert baselines.is_relative_to(Path(settings["E2E_ROSETTE_ROOT"])), baselines
    assert baselines != Path(settings["E2E_ROSETTE_ROOT"]), baselines
    assert not baselines.is_relative_to(Path(settings["E2E_CODE_DIR"])), baselines
    assert not baselines.resolve().is_relative_to(REPO), baselines


# REQ-E2ECI-BDRABCZUK-744F14261317 ---------------------------------------------------


def test_one_more_failed_subject_fails(tmp_path: Path) -> None:
    e2e_dir, run, baseline = _setup(tmp_path)
    _write_baseline(e2e_dir, baseline)  # snapshots: no failed subject
    _write_qc(run / SNAPSHOTS_QC, [(SUB_A, 1, ""), (SUB_B, 0, "missing mesh")])
    result = _check_tier(e2e_dir, run)
    assert result.returncode == 1, result.stderr
    line = _line_naming("stages.generate_snapshots.failed_subjects", result.stderr)
    assert SUB_B in line, line


def test_one_fewer_failed_subject_fails(tmp_path: Path) -> None:
    """An improvement is a difference too: it fails until the baseline is accepted again."""
    e2e_dir, run, baseline = _setup(tmp_path)
    assert baseline["stages"]["generate_morphologist_graphs"]["failed_subjects"] == [SUB_B]
    _write_baseline(e2e_dir, baseline)
    _write_qc(run / MORPHOLOGIST_QC, [(SUB_A, 1, ""), (SUB_B, 1, "")])
    result = _check_tier(e2e_dir, run)
    assert result.returncode == 1, result.stderr
    line = _line_naming("stages.generate_morphologist_graphs.failed_subjects", result.stderr)
    assert SUB_B in line, line


# REQ-E2ECI-BDRABCZUK-744F14261317 ---------------------------------------------------


def test_subject_count_deviation_fails(tmp_path: Path) -> None:
    e2e_dir, run, baseline = _setup(tmp_path)
    assert baseline["stages"]["run_cortical_tiles"]["subjects"] == 2
    baseline["stages"]["run_cortical_tiles"]["subjects"] = 3
    _write_baseline(e2e_dir, baseline)
    result = _check_tier(e2e_dir, run)
    assert result.returncode == 1, result.stderr
    _line_naming("stages.run_cortical_tiles.subjects", result.stderr)


def test_stage_absent_from_run_fails(tmp_path: Path) -> None:
    e2e_dir, run, baseline = _setup(tmp_path)
    _write_baseline(e2e_dir, baseline)
    # The run lists every stage but run_cortical_tiles: its subjects key is absent from report.json.
    _write_stages(run, dict.fromkeys((stage for stage in STAGES if stage != "run_cortical_tiles"), 0))
    result = _check_tier(e2e_dir, run)
    assert result.returncode == 1, result.stderr
    _line_naming("stages.run_cortical_tiles.subjects", result.stderr)


# REQ-E2ECI-BDRABCZUK-744F14261317 ---------------------------------------------------


def test_qc_count_deviation_fails(tmp_path: Path) -> None:
    e2e_dir, run, baseline = _setup(tmp_path)
    assert baseline["stages"]["generate_morphologist_graphs"]["qc"] == {"pass": 1, "fail": 1}
    baseline["stages"]["generate_morphologist_graphs"]["qc"] = {"pass": 0, "fail": 2}
    _write_baseline(e2e_dir, baseline)
    result = _check_tier(e2e_dir, run)
    assert result.returncode == 1, result.stderr
    _line_naming("stages.generate_morphologist_graphs.qc", result.stderr)


def test_qc_count_improvement_fails(tmp_path: Path) -> None:
    """One more QC pass than the baseline is a difference too."""
    e2e_dir, run, baseline = _setup(tmp_path)
    _write_baseline(e2e_dir, baseline)  # morphologist QC: pass 1, fail 1
    _write_qc(run / MORPHOLOGIST_QC, [(SUB_A, 1, ""), (SUB_B, 1, "")])
    result = _check_tier(e2e_dir, run)
    assert result.returncode == 1, result.stderr
    _line_naming("stages.generate_morphologist_graphs.qc", result.stderr)


# REQ-E2ECI-BDRABCZUK-744F14261317 ---------------------------------------------------


def test_region_side_deviation_fails(tmp_path: Path) -> None:
    e2e_dir, run, baseline = _setup(tmp_path)
    _write_baseline(e2e_dir, baseline)
    (run / COMBINED_DIR / f"{LEFT}_embeddings.csv").unlink()  # one region side fewer
    result = _check_tier(e2e_dir, run)
    assert result.returncode == 1, result.stderr
    line = _line_naming("region_sides", result.stderr)
    assert LEFT in line, line


# REQ-E2ECI-BDRABCZUK-744F14261317 ---------------------------------------------------


def test_exit_code_deviation_fails(tmp_path: Path) -> None:
    e2e_dir, run, baseline = _setup(tmp_path)
    assert baseline["stages"]["generate_champollion_config"] == {"exit_code": 0}
    baseline["stages"]["generate_champollion_config"]["exit_code"] = 1  # the run has 0
    _write_baseline(e2e_dir, baseline)
    result = _check_tier(e2e_dir, run)
    assert result.returncode == 1, result.stderr
    _line_naming("stages.generate_champollion_config.exit_code", result.stderr)


def test_extra_stage_in_run_fails(tmp_path: Path) -> None:
    e2e_dir, run, baseline = _setup(tmp_path)
    del baseline["stages"]["generate_champollion_config"]  # the run has this stage, the baseline not
    _write_baseline(e2e_dir, baseline)
    result = _check_tier(e2e_dir, run)
    assert result.returncode == 1, result.stderr
    _line_naming("stages.generate_champollion_config.exit_code", result.stderr)


# REQ-E2ECI-BDRABCZUK-4D2888B44DA5 ---------------------------------------------------


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", ""])
def test_unreadable_baseline_fails(tmp_path: Path, content: str) -> None:
    e2e_dir, run, _ = _setup(tmp_path)
    path = _baseline_path(e2e_dir)
    path.parent.mkdir(parents=True)
    path.write_text(content)
    result = _check_tier(e2e_dir, run)
    assert "Traceback" not in result.stderr, result.stderr
    assert result.returncode == 1, result.stderr
    assert str(path) in result.stderr, result.stderr


# REQ-E2ECI-BDRABCZUK-C01572FFFB22 ---------------------------------------------------


def test_deviating_run_leaves_baseline_unchanged(tmp_path: Path) -> None:
    e2e_dir, run, baseline = _setup(tmp_path)
    baseline["stages"]["generate_embeddings"]["subjects"] = 2  # the run has 1
    path = _write_baseline(e2e_dir, baseline)
    before = path.read_bytes()
    result = _check_tier(e2e_dir, run)
    assert result.returncode != 2, f"--tier/--layout refused as a usage error: {result.stderr}"
    assert path.read_bytes() == before
    assert sorted(path.parent.iterdir()) == [path]


def test_first_run_creates_no_baseline(tmp_path: Path) -> None:
    e2e_dir, run, _ = _setup(tmp_path)
    result = _check_tier(e2e_dir, run)
    assert result.returncode != 2, f"--tier/--layout refused as a usage error: {result.stderr}"
    assert not _baseline_path(e2e_dir).exists(), "a baseline was written without --accept-baseline"


# REQ-E2ECI-BDRABCZUK-EFFF10F1F399 ---------------------------------------------------


def test_report_lists_baseline_differences(tmp_path: Path) -> None:
    e2e_dir, run, baseline = _setup(tmp_path)
    baseline["stages"]["put_together_embeddings"]["subjects"] = 1  # the run has 2
    _write_baseline(e2e_dir, baseline)
    result = _check_tier(e2e_dir, run)
    differences = _report(run).get("baseline", {}).get("differences")
    assert isinstance(differences, list) and differences, _report(run)
    stderr_lines = result.stderr.splitlines()
    assert all(line in stderr_lines for line in differences), (differences, stderr_lines)
    assert any("stages.put_together_embeddings.subjects" in line for line in differences), differences


def test_matching_baseline_passes(tmp_path: Path) -> None:
    e2e_dir, run, baseline = _setup(tmp_path)
    _write_baseline(e2e_dir, baseline)
    result = _check_tier(e2e_dir, run)
    assert result.returncode == 0, result.stderr
    report = _report(run)
    assert report.get("baseline", {}).get("differences") == [], report
    assert report["hard_failures"] == []


# REQ-E2ECI-BDRABCZUK-12443B519CC1 ---------------------------------------------------


def test_missing_baseline_is_reported(tmp_path: Path) -> None:
    e2e_dir, run, _ = _setup(tmp_path)
    path = _baseline_path(e2e_dir)
    result = _check_tier(e2e_dir, run)
    line = _line_naming("no baseline", result.stderr)
    assert str(path) in line.split("no baseline", 1)[1], line
    assert result.returncode == 0, result.stderr
    assert _report(run)["hard_failures"] == []


# REQ-E2ECI-BDRABCZUK-8B651F44BF51 ---------------------------------------------------


def test_accept_baseline_creates_baseline(tmp_path: Path) -> None:
    e2e_dir, run, baseline = _setup(tmp_path)
    result = _check_tier(e2e_dir, run, "--accept-baseline")
    assert result.returncode != 2, f"--accept-baseline refused as a usage error: {result.stderr}"
    path = _baseline_path(e2e_dir)
    assert path.is_file(), result.stderr
    assert _baseline_facts(json.loads(path.read_text())) == baseline
    again = _check_tier(e2e_dir, run)
    assert again.returncode == 0, again.stderr
    assert _report(run).get("baseline", {}).get("differences") == [], _report(run)


def test_accept_baseline_overwrites_baseline(tmp_path: Path) -> None:
    e2e_dir, run, baseline = _setup(tmp_path)
    older = json.loads(json.dumps(baseline))
    older["stages"]["run_cortical_tiles"]["subjects"] = 5
    older["region_sides"] = ["SC-sylv_right"]
    path = _write_baseline(e2e_dir, older)
    result = _check_tier(e2e_dir, run, "--accept-baseline")
    assert result.returncode != 2, f"--accept-baseline refused as a usage error: {result.stderr}"
    assert _baseline_facts(json.loads(path.read_text())) == baseline
    again = _check_tier(e2e_dir, run)
    assert again.returncode == 0, again.stderr


# REQ-E2ECI-BDRABCZUK-23B256F2E0C6 ---------------------------------------------------


@pytest.mark.parametrize("existing", [False, True], ids=["no-baseline", "existing-baseline"])
def test_failing_run_never_writes_baseline(tmp_path: Path, existing: bool) -> None:
    e2e_dir, run, baseline = _setup(tmp_path)
    path = _baseline_path(e2e_dir)
    before = _write_baseline(e2e_dir, baseline).read_bytes() if existing else None
    _write_stages(run, {**dict.fromkeys(STAGES, 0), "generate_snapshots": 3})  # a hard failure
    result = _check_tier(e2e_dir, run, "--accept-baseline")
    assert result.returncode == 1, result.stderr
    if existing:
        assert path.read_bytes() == before
    else:
        assert not path.exists(), "a failing run wrote a baseline"


# REQ-E2ECI-BDRABCZUK-BE3EEE3CE81F ---------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [(), ("--tier", TIER), ("--layout", LAYOUT)],
    ids=["neither", "tier-only", "layout-only"],
)
def test_accept_baseline_without_tier_is_usage_error(tmp_path: Path, options: tuple[str, ...]) -> None:
    e2e_dir, run, _ = _setup(tmp_path)
    accepted = _check_tier(e2e_dir, run, "--accept-baseline")
    assert accepted.returncode != 2, f"--accept-baseline refused as a usage error: {accepted.stderr}"
    shutil.rmtree(_baselines(e2e_dir), ignore_errors=True)
    (run / "report.json").unlink()
    baselines = ("--baselines-dir", str(_baselines(e2e_dir)))
    result = _check(e2e_dir, run, *options, *baselines, "--accept-baseline")
    assert result.returncode == 2, result.stderr
    assert len(result.stderr.splitlines()) == 1 and "usage" in result.stderr, result.stderr
    assert not (run / "report.json").exists()
    assert not _baselines(e2e_dir).exists()


# REQ-E2ECI-BDRABCZUK-C15554596494 ---------------------------------------------------


@pytest.mark.parametrize("tier", ["daily", "../../quick", ""])
def test_unknown_tier_is_usage_error(tmp_path: Path, tier: str) -> None:
    e2e_dir, run, baseline = _setup(tmp_path)
    _write_baseline(e2e_dir, baseline)
    accepted = _check_tier(e2e_dir, run)
    assert accepted.returncode != 2, f"a known tier is refused as a usage error: {accepted.stderr}"
    (run / "report.json").unlink()
    result = _check_tier(e2e_dir, run, tier=tier)
    assert result.returncode == 2, result.stderr
    assert len(result.stderr.splitlines()) == 1 and "usage" in result.stderr, result.stderr
    assert not (run / "report.json").exists()
    assert sorted(_baselines(e2e_dir).iterdir()) == [_baseline_path(e2e_dir)]


# REQ-E2ECI-BDRABCZUK-5A73313FCF52 ---------------------------------------------------


@pytest.mark.parametrize("layout", ["flat", "../bids", ""])
def test_unknown_layout_is_usage_error(tmp_path: Path, layout: str) -> None:
    e2e_dir, run, baseline = _setup(tmp_path)
    _write_baseline(e2e_dir, baseline)
    accepted = _check_tier(e2e_dir, run)
    assert accepted.returncode != 2, f"a known layout is refused as a usage error: {accepted.stderr}"
    (run / "report.json").unlink()
    result = _check_tier(e2e_dir, run, layout=layout)
    assert result.returncode == 2, result.stderr
    assert len(result.stderr.splitlines()) == 1 and "usage" in result.stderr, result.stderr
    assert not (run / "report.json").exists()
    assert sorted(_baselines(e2e_dir).iterdir()) == [_baseline_path(e2e_dir)]
