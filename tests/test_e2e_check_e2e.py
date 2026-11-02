"""Tests for the e2e run checker ``scripts/e2e/check_e2e.py`` (facts and hard failures).

TASK-E2ECI-BDRABCZUK-E5909F411253 (CHECK, epic TASK-E2ECI-BDRABCZUK-F3B0138F58F3).
``python3 scripts/e2e/check_e2e.py <run dir>`` reads one finished e2e run directory,
writes ``<run dir>/report.json`` with the run's facts (per-stage exit code and subject
count, failed-subject ids, region-side coverage, QC counts) and exits 1 on a hard
failure. The baseline comparison is a later task (BASEL).

Run directory contract, as read here (the run script, task RUN, writes it)::

    <run dir>/
      stages.tsv                                   # header stage<TAB>exit_code, one row per stage that ran
      morphologist/derivatives/morphologist-6.0/input_qc.tsv     # stage 1 QC TSV
      derivatives/cortical_tiles/<crop>/mask/{L,R}skeleton_subject.csv
      derivatives/champollion_V1/<region-side>/full_embeddings.csv
      derivatives/champollion_V1/<masks>/embeddings/<region-side>_embeddings.csv
      derivatives/champollion_V1/<masks>/snapshots/input_qc.tsv  # stage 6 QC TSV
      <E2E_RUN_PROFILE_DIRNAME>/proc/*.json         # profiling sidecars, key "stage"

Stage names are the main.py stage names. Data is synthetic; the only subject ids are
the public IXI ids of the QUICK tier.

Requirements (one test or more each):
- REQ-E2ECI-BDRABCZUK-EB3C7977AC9D report.json, one JSON object, in the run directory
- REQ-E2ECI-BDRABCZUK-66CA4843E64B per-stage exit codes from stages.tsv
- REQ-E2ECI-BDRABCZUK-28A839ED2D51 missing stages.tsv fails
- REQ-E2ECI-BDRABCZUK-D268B7A1152A non-zero stage exit fails
- REQ-E2ECI-BDRABCZUK-CFB5E11FD27A NaN or infinite embedding value fails
- REQ-E2ECI-BDRABCZUK-EC82B0EAB260 missing Morphologist input QC TSV fails
- REQ-E2ECI-BDRABCZUK-650CA7A769EB missing snapshots input QC TSV fails
- REQ-E2ECI-BDRABCZUK-1CA495C91305 missing profile sidecar for a stage that ran fails
- REQ-E2ECI-BDRABCZUK-ED6AE6327C24 hard_failures lists the stderr failure lines
- REQ-E2ECI-BDRABCZUK-2A0AF7B30782 exit 0 when hard_failures is empty
- REQ-E2ECI-BDRABCZUK-4EBFB2BB5892 region-side coverage
- REQ-E2ECI-BDRABCZUK-42B77BA11567 cortical tiles subject count
- REQ-E2ECI-BDRABCZUK-AA0A046CA8C8 region embeddings subject count
- REQ-E2ECI-BDRABCZUK-50F2F3CFD1C0 combined embeddings subject count
- REQ-E2ECI-BDRABCZUK-C30DA144BD1E QC counts per stage QC TSV
- REQ-E2ECI-BDRABCZUK-295F078E8277 failed-subject ids per stage QC TSV
- REQ-E2ECI-BDRABCZUK-59670620F1D3 standard library imports only (runs without aims)
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
E2E_DIR = REPO / "scripts" / "e2e"
CHECK_E2E = E2E_DIR / "check_e2e.py"
E2E_ENV_PY = E2E_DIR / "e2e_env.py"

STAGES = (
    "generate_morphologist_graphs",
    "run_cortical_tiles",
    "generate_champollion_config",
    "generate_embeddings",
    "put_together_embeddings",
    "generate_snapshots",
)
NIGHTLY_STAGES = STAGES[1:]

# Public IXI subject ids (CC BY-SA 3.0), the QUICK tier pair.
SUB_A = "sub-002"
SUB_B = "sub-012"

MASKS = "canonical_25"
LEFT = "SC-sylv_left"
RIGHT = "SC-sylv_right"
CROP = "S.C.-sylv."

MORPHOLOGIST_QC = Path("morphologist/derivatives/morphologist-6.0/input_qc.tsv")
CHAMPOLLION = Path("derivatives/champollion_V1")
SNAPSHOTS_QC = CHAMPOLLION / MASKS / "snapshots" / "input_qc.tsv"
COMBINED_DIR = CHAMPOLLION / MASKS / "embeddings"
TILES_MASK_DIR = Path("derivatives/cortical_tiles") / CROP / "mask"


def _profile_dirname() -> str:
    """E2E_RUN_PROFILE_DIRNAME read from e2e.env through scripts/e2e/e2e_env.py."""
    spec = importlib.util.spec_from_file_location("e2e_env_for_check_tests", E2E_ENV_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_settings()["E2E_RUN_PROFILE_DIRNAME"]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _write_stages(run: Path, exit_codes: dict[str, int]) -> None:
    rows = "".join(f"{stage}\t{code}\n" for stage, code in exit_codes.items())
    _write(run / "stages.tsv", "stage\texit_code\n" + rows)


def _write_qc(path: Path, rows: list[tuple[str, int, str]]) -> None:
    body = "".join(f"{subject}\t{qc}\t{reason}\n" for subject, qc, reason in rows)
    _write(path, "participant_id\tqc\treason\n" + body)


def _write_embeddings(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    body = "".join(f"{subject},{dim1},{dim2}\n" for subject, dim1, dim2 in rows)
    return _write(path, "ID,dim1,dim2\n" + body)


def _write_sidecar(run: Path, stage: str, seq: int) -> None:
    sidecar = {"host": "nodeA", "pid": 4242, "seq": seq, "stage": stage, "tags": {}}
    _write(run / _profile_dirname() / "proc" / f"nodeA-4242-{seq}.json", json.dumps(sidecar))


def _make_run(run: Path, stages: tuple[str, ...] = STAGES) -> Path:
    """A clean run directory: every listed stage exited 0 and left its outputs."""
    _write_stages(run, dict.fromkeys(stages, 0))
    if "generate_morphologist_graphs" in stages:
        _write_qc(run / MORPHOLOGIST_QC, [(SUB_A, 1, ""), (SUB_B, 0, "unreadable NIfTI")])
    _write(run / TILES_MASK_DIR / "Lskeleton_subject.csv", f"Subject\n{SUB_A}\n")
    _write(run / TILES_MASK_DIR / "Rskeleton_subject.csv", f"Subject\n{SUB_A}\n{SUB_B}\n")
    _write_embeddings(run / CHAMPOLLION / LEFT / "full_embeddings.csv", [(SUB_A, "0.1", "0.2")])
    _write_embeddings(run / CHAMPOLLION / RIGHT / "full_embeddings.csv", [(SUB_A, "0.3", "0.4")])
    _write_embeddings(run / COMBINED_DIR / f"{RIGHT}_embeddings.csv", [(SUB_A, "0.3", "0.4"), (SUB_B, "0.5", "0.6")])
    _write_embeddings(run / COMBINED_DIR / f"{LEFT}_embeddings.csv", [(SUB_A, "0.1", "0.2")])
    _write_qc(run / SNAPSHOTS_QC, [(SUB_A, 1, ""), (SUB_B, 1, "")])
    for seq, stage in enumerate(stages):
        _write_sidecar(run, stage, seq)
    return run


def _run_check(run: Path) -> subprocess.CompletedProcess:
    assert CHECK_E2E.is_file(), f"{CHECK_E2E.relative_to(REPO)} does not exist yet"
    return subprocess.run(
        [sys.executable, str(CHECK_E2E), str(run)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _report(run: Path) -> dict:
    return json.loads((run / "report.json").read_text())


def _names(path: Path, run: Path, stderr: str) -> bool:
    """True when stderr names ``path``, absolute or relative to the run directory."""
    return str(path) in stderr or str(path.relative_to(run)) in stderr


# REQ-E2ECI-BDRABCZUK-EB3C7977AC9D ---------------------------------------------------


def test_writes_report_json_object_in_run_dir(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    _run_check(run)
    assert (run / "report.json").is_file(), "report.json not written into the run directory"
    assert isinstance(_report(run), dict)


# REQ-E2ECI-BDRABCZUK-66CA4843E64B ---------------------------------------------------


def test_stage_exit_codes_copied_from_stages_tsv(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    _write_stages(run, {**dict.fromkeys(STAGES, 0), "generate_snapshots": 3})
    _run_check(run)
    stages = _report(run)["stages"]
    expected = {**dict.fromkeys(STAGES, 0), "generate_snapshots": 3}
    assert {stage: stages[stage]["exit_code"] for stage in STAGES} == expected
    assert all(type(stages[stage]["exit_code"]) is int for stage in STAGES)


# REQ-E2ECI-BDRABCZUK-28A839ED2D51 ---------------------------------------------------


def test_missing_stages_tsv_fails(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    (run / "stages.tsv").unlink()
    result = _run_check(run)
    assert result.returncode == 1, result.stderr
    assert "stages.tsv" in result.stderr


# REQ-E2ECI-BDRABCZUK-D268B7A1152A ---------------------------------------------------


def test_non_zero_stage_exit_fails(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    _write_stages(run, {**dict.fromkeys(STAGES, 0), "run_cortical_tiles": 1})
    result = _run_check(run)
    assert result.returncode == 1, result.stderr
    assert "run_cortical_tiles" in result.stderr


# REQ-E2ECI-BDRABCZUK-CFB5E11FD27A ---------------------------------------------------


@pytest.mark.parametrize(
    ("value", "relpath"),
    [
        ("nan", CHAMPOLLION / LEFT / "full_embeddings.csv"),
        ("inf", COMBINED_DIR / f"{RIGHT}_embeddings.csv"),
        ("-Infinity", CHAMPOLLION / RIGHT / "full_embeddings.csv"),
        ("NaN", COMBINED_DIR / f"{LEFT}_embeddings.csv"),
    ],
)
def test_non_finite_embedding_fails(tmp_path: Path, value: str, relpath: Path) -> None:
    run = _make_run(tmp_path / "run")
    bad = _write_embeddings(run / relpath, [(SUB_A, "0.1", value)])
    result = _run_check(run)
    assert result.returncode == 1, result.stderr
    assert _names(bad, run, result.stderr), result.stderr


# REQ-E2ECI-BDRABCZUK-EC82B0EAB260 ---------------------------------------------------


def test_missing_morphologist_qc_tsv_fails(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    (run / MORPHOLOGIST_QC).unlink()
    result = _run_check(run)
    assert result.returncode == 1, result.stderr
    assert _names(run / MORPHOLOGIST_QC, run, result.stderr), result.stderr


def test_morphologist_qc_tsv_not_required_without_stage_1(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run", NIGHTLY_STAGES)
    result = _run_check(run)
    assert result.returncode == 0, result.stderr
    assert not _names(run / MORPHOLOGIST_QC, run, result.stderr)


# REQ-E2ECI-BDRABCZUK-650CA7A769EB ---------------------------------------------------


def test_missing_snapshots_qc_tsv_fails(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    (run / SNAPSHOTS_QC).unlink()
    result = _run_check(run)
    assert result.returncode == 1, result.stderr
    assert "generate_snapshots" in result.stderr


# REQ-E2ECI-BDRABCZUK-1CA495C91305 ---------------------------------------------------


def test_missing_profile_sidecar_fails(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    proc = run / _profile_dirname() / "proc"
    for path in proc.glob("*.json"):
        if json.loads(path.read_text())["stage"] == "generate_embeddings":
            path.unlink()
    result = _run_check(run)
    assert result.returncode == 1, result.stderr
    assert "generate_embeddings" in result.stderr


# REQ-E2ECI-BDRABCZUK-ED6AE6327C24 ---------------------------------------------------


def test_hard_failures_lists_stderr_failure_lines(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    _write_stages(run, {**dict.fromkeys(STAGES, 0), "generate_embeddings": 2})
    (run / SNAPSHOTS_QC).unlink()
    result = _run_check(run)
    failures = _report(run)["hard_failures"]
    assert isinstance(failures, list)
    assert len(failures) >= 2, failures
    stderr_lines = result.stderr.splitlines()
    assert all(line in stderr_lines for line in failures), (failures, stderr_lines)
    assert any("generate_embeddings" in line for line in failures)
    assert any("generate_snapshots" in line for line in failures)


def test_hard_failures_empty_on_clean_run(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    _run_check(run)
    assert _report(run)["hard_failures"] == []


# REQ-E2ECI-BDRABCZUK-2A0AF7B30782 ---------------------------------------------------


def test_clean_run_exits_zero(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    result = _run_check(run)
    assert _report(run)["hard_failures"] == []
    assert result.returncode == 0, result.stderr


# REQ-E2ECI-BDRABCZUK-4EBFB2BB5892 ---------------------------------------------------


def test_region_sides_listed(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    _run_check(run)
    assert _report(run)["region_sides"] == [LEFT, RIGHT]


# REQ-E2ECI-BDRABCZUK-42B77BA11567 ---------------------------------------------------


def test_cortical_tiles_subject_count(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    _run_check(run)
    # Lskeleton has sub-002; Rskeleton has sub-002 and sub-012: two distinct subjects.
    assert _report(run)["stages"]["run_cortical_tiles"]["subjects"] == 2


# REQ-E2ECI-BDRABCZUK-AA0A046CA8C8 ---------------------------------------------------


def test_region_embeddings_subject_count(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    _run_check(run)
    # Both full_embeddings.csv files hold sub-002 only: one distinct subject.
    assert _report(run)["stages"]["generate_embeddings"]["subjects"] == 1


# REQ-E2ECI-BDRABCZUK-50F2F3CFD1C0 ---------------------------------------------------


def test_combined_embeddings_subject_count(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    _run_check(run)
    # Combined files hold sub-002 (left, right) and sub-012 (right): two distinct subjects.
    assert _report(run)["stages"]["put_together_embeddings"]["subjects"] == 2


# REQ-E2ECI-BDRABCZUK-C30DA144BD1E ---------------------------------------------------


def test_qc_counts_per_stage(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    _run_check(run)
    stages = _report(run)["stages"]
    assert stages["generate_morphologist_graphs"]["qc"] == {"pass": 1, "fail": 1}
    assert stages["generate_snapshots"]["qc"] == {"pass": 2, "fail": 0}


# REQ-E2ECI-BDRABCZUK-295F078E8277 ---------------------------------------------------


def test_failed_subject_ids_per_stage(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    _write_qc(run / SNAPSHOTS_QC, [(SUB_B, 0, "missing mesh"), (SUB_A, 0, "missing mesh")])
    _run_check(run)
    stages = _report(run)["stages"]
    assert stages["generate_morphologist_graphs"]["failed_subjects"] == [SUB_B]
    assert stages["generate_snapshots"]["failed_subjects"] == [SUB_A, SUB_B]


# REQ-E2ECI-BDRABCZUK-59670620F1D3 ---------------------------------------------------


def test_imports_standard_library_only() -> None:
    assert CHECK_E2E.is_file(), f"{CHECK_E2E.relative_to(REPO)} does not exist yet"
    tree = ast.parse(CHECK_E2E.read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    outside = sorted(modules - set(sys.stdlib_module_names) - {"e2e_env"})
    assert not outside, f"non-standard-library imports: {outside}"


# Contract: an unreadable input is a failure line, never a traceback --------------------


def test_non_string_sidecar_stage_is_skipped(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    _write(run / _profile_dirname() / "proc" / "nodeA-4242-99.json", json.dumps({"stage": ["generate_embeddings"]}))
    result = _run_check(run)
    assert "Traceback" not in result.stderr, result.stderr
    assert result.returncode == 0, result.stderr
    assert _report(run)["hard_failures"] == []


def test_non_utf8_stages_tsv_is_a_failure_line(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    (run / "stages.tsv").write_bytes(b"stage\texit_code\n\xff\xfe\t0\n")
    result = _run_check(run)
    assert "Traceback" not in result.stderr, result.stderr
    assert result.returncode == 1, result.stderr
    assert "malformed stages.tsv" in result.stderr
    assert (run / "report.json").is_file()
