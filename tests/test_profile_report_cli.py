"""Red tests for the profile_report script and its pixi task (PAGG: outputs, options, task).

Task TASK-TASK-BDRABCZUK-F20A7E4D5472; design COMP-ARCH-BDRABCZUK-7B43010AD32E, Contract 5.
User decisions 2026-10-27: outputs are the merged .prof files plus report.json,
no TSV; report.json grouped by stage is what the METRICS profiling source reads.
"""

import json
import logging
import pstats
from pathlib import Path

import pytest
import tomllib

from tests.profile_run_fixtures import (
    EXPECTED_FUNCTIONS,
    EXPECTED_STAGES,
    build_cpu_run,
    group_rows,
    load_profile_report,
)

PIXI_TOML = Path(__file__).resolve().parents[1] / "pixi.toml"
TASK_NAME = "profile-report"
SCRIPT_PATH = "src/champollion_pipeline/profile_report.py"
DEFAULT_TOP = 20


@pytest.fixture
def cpu_run(tmp_path):
    """Synthetic CPU-only run directory."""
    return build_cpu_run(tmp_path)


def _run_script(argv: list[str]) -> int:
    """Parse ``argv`` and run the ProfileReport script in-process (no logging setup, no update check)."""
    script = load_profile_report().ProfileReport()
    script.parse_args(argv)
    return script.run()


def _read_report(output_dir: Path) -> dict:
    return json.loads((output_dir / "report.json").read_text(encoding="utf-8"))


def _function_names(prof_path: Path) -> set[str]:
    return {funcname for (_file, _line, funcname) in pstats.Stats(str(prof_path)).stats}


# ---------------------------------------------------------------------------
# report.json
# ---------------------------------------------------------------------------


def test_report_json_defaults_to_report_subdirectory(cpu_run):
    """REQ: report.json goes to --output, which defaults to <run_dir>/report."""
    assert _run_script([str(cpu_run)]) == 0
    assert (cpu_run / "report" / "report.json").is_file()


def test_report_json_goes_to_given_output_directory(cpu_run, tmp_path):
    """REQ: with --output DIR, report.json is written into DIR."""
    output = tmp_path / "elsewhere"
    assert _run_script([str(cpu_run), "--output", str(output)]) == 0
    assert (output / "report.json").is_file()
    assert not (cpu_run / "report" / "report.json").exists()


@pytest.mark.parametrize(("group_by", "top"), [("stage", 5), ("process", 2)])
def test_report_json_holds_the_computed_document(cpu_run, tmp_path, group_by, top):
    """REQ: report.json holds the document compute_profile_report returns for the same run, grouping and top."""
    expected = json.loads(json.dumps(load_profile_report().compute_profile_report(cpu_run, group_by, top)))
    output = tmp_path / "out"
    assert _run_script([str(cpu_run), "--group-by", group_by, "--top", str(top), "--output", str(output)]) == 0
    assert _read_report(output) == expected


def test_default_grouping_is_by_stage_with_per_stage_figures(cpu_run):
    """REQ: without --group-by the script groups by stage (the per-stage rows the METRICS source reads)."""
    assert _run_script([str(cpu_run)]) == 0
    report = _read_report(cpu_run / "report")
    assert report["group_by"] == "stage"
    rows = group_rows(report)
    assert set(rows) == set(EXPECTED_STAGES)
    for stage, (_count, wall_s, cpu_s, peak_rss_kb, _incomplete) in EXPECTED_STAGES.items():
        assert rows[stage]["wall_s"] == pytest.approx(wall_s), stage
        assert rows[stage]["cpu_s"] == pytest.approx(cpu_s), stage
        assert rows[stage]["peak_rss_kb"] == peak_rss_kb, stage


def test_default_top_lists_twenty_functions(cpu_run):
    """REQ: without --top, report.json lists at most 20 functions per group (cortical_tiles has more)."""
    assert _run_script([str(cpu_run)]) == 0
    rows = group_rows(_read_report(cpu_run / "report"))
    assert len(rows["cortical_tiles"]["top_functions"]) == DEFAULT_TOP
    assert all(len(row["top_functions"]) <= DEFAULT_TOP for row in rows.values())


# ---------------------------------------------------------------------------
# merged .prof files and the output file set
# ---------------------------------------------------------------------------


def test_merged_prof_per_group_combines_the_group_dumps(cpu_run):
    """REQ: one merged-<group>.prof per group, loadable by pstats, combining the group's segment dumps."""
    assert _run_script([str(cpu_run)]) == 0
    output = cpu_run / "report"
    for stage, functions in EXPECTED_FUNCTIONS.items():
        merged = output / f"merged-{stage}.prof"
        assert merged.is_file(), stage
        assert functions <= _function_names(merged), stage


def test_output_directory_holds_only_report_json_and_merged_prof(cpu_run):
    """REQ: the script writes only report.json and merged-<group>.prof files (no TSV, user decision Q18)."""
    assert _run_script([str(cpu_run)]) == 0
    names = {path.name for path in (cpu_run / "report").iterdir()}
    merged = {name for name in names if name.startswith("merged-") and name.endswith(".prof")}
    unexpected = names - merged - {"report.json"}
    assert not unexpected, sorted(unexpected)
    assert "report.json" in names


# ---------------------------------------------------------------------------
# --format
# ---------------------------------------------------------------------------


def test_format_json_prints_the_report_document(cpu_run, capsys):
    """REQ: with --format json, standard output is the report.json document."""
    assert _run_script([str(cpu_run), "--format", "json"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == _read_report(cpu_run / "report")


def _text_output_lines(capsys, caplog) -> list[str]:
    """Lines the script showed: stdout plus logged messages (the text form may go through the logger)."""
    lines = capsys.readouterr().out.splitlines()
    for record in caplog.records:
        lines.extend(record.getMessage().splitlines())
    return [line.strip() for line in lines]


def test_format_text_prints_one_line_per_group(cpu_run, capsys, caplog):
    """REQ: with --format text, one line per group row, starting with the group name."""
    caplog.set_level(logging.DEBUG)
    assert _run_script([str(cpu_run), "--format", "text"]) == 0
    lines = _text_output_lines(capsys, caplog)
    for stage in EXPECTED_STAGES:
        assert any(line.startswith(stage) for line in lines), (stage, lines)


def test_text_output_says_no_gpu_used_without_gpu_data(cpu_run, capsys, caplog):
    """REQ: with no GPU data, the text output includes the line 'no GPU used'."""
    caplog.set_level(logging.DEBUG)
    assert _run_script([str(cpu_run), "--format", "text"]) == 0
    assert "no GPU used" in _text_output_lines(capsys, caplog)


# ---------------------------------------------------------------------------
# pixi task
# ---------------------------------------------------------------------------


def _tasks_visible_in(config: dict, env: str) -> dict:
    """Tasks pixi exposes in ``env``: workspace ``[tasks]`` plus its features' tasks."""
    env_spec = config["environments"][env]
    features = env_spec["features"] if isinstance(env_spec, dict) else env_spec
    tasks = dict(config.get("tasks", {}))
    for feature in features:
        tasks.update(config.get("feature", {}).get(feature, {}).get("tasks", {}))
    return tasks


def test_pixi_defines_profile_report_task_in_default_environment():
    """REQ: pixi.toml defines a profile-report task, visible in the default environment, running the script."""
    with PIXI_TOML.open("rb") as handle:
        config = tomllib.load(handle)
    tasks = _tasks_visible_in(config, "default")
    assert TASK_NAME in tasks, f"pixi.toml defines no {TASK_NAME!r} task visible in the default environment"
    task = tasks[TASK_NAME]
    command = task if isinstance(task, str) else task.get("cmd", "")
    assert SCRIPT_PATH in command
