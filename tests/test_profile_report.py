"""Red tests for compute_profile_report, the pure part of profile_report (PAGG, CPU figures).

Task TASK-TASK-BDRABCZUK-F20A7E4D5472; design COMP-ARCH-BDRABCZUK-7B43010AD32E, Contract 5.
The run directory is synthetic (tests/profile_run_fixtures.py): it follows the
design's sidecar and tree-sample formats, since the per-process hook that
writes them does not exist yet.
"""

import pytest

from tests.profile_run_fixtures import (
    EXPECTED_PROCESS_COUNT,
    EXPECTED_STAGES,
    HOST_A,
    HOST_B,
    SUBJECT_1,
    SUBJECT_2,
    build_cpu_run,
    find_process,
    group_rows,
    load_profile_report,
)

PROCESS_ENTRY_KEYS = {"host", "pid", "argv", "group", "wall_s", "cpu_s", "peak_rss_kb", "exit_path", "complete"}


@pytest.fixture
def cpu_run(tmp_path):
    """Synthetic CPU-only run directory."""
    return build_cpu_run(tmp_path)


def _compute(run_dir, group_by="stage", top=20) -> dict:
    return load_profile_report().compute_profile_report(run_dir, group_by, top)


def _stage_index(field: str) -> int:
    return ("process_count", "wall_s", "cpu_s", "peak_rss_kb", "incomplete_count").index(field)


# ---------------------------------------------------------------------------
# Group rows (stage grouping)
# ---------------------------------------------------------------------------


def test_stage_grouping_gives_one_row_per_stage_with_process_count(cpu_run):
    """REQ: one group row per distinct sidecar stage, holding the number of processes of that stage."""
    report = _compute(cpu_run, "stage")
    rows = group_rows(report)
    assert report["group_by"] == "stage"
    assert set(rows) == set(EXPECTED_STAGES)
    for stage, expected in EXPECTED_STAGES.items():
        assert rows[stage]["process_count"] == expected[_stage_index("process_count")], stage


def test_group_wall_seconds_span_from_first_start_to_last_end(cpu_run):
    """REQ: wall seconds = latest segment end minus earliest segment start among the group's sidecars."""
    rows = group_rows(_compute(cpu_run))
    for stage, expected in EXPECTED_STAGES.items():
        assert rows[stage]["wall_s"] == pytest.approx(expected[_stage_index("wall_s")]), stage


def test_group_cpu_seconds_sum_user_plus_system(cpu_run):
    """REQ: CPU seconds = sum of user plus system CPU over the group's processes, tree-only ones counted once."""
    rows = group_rows(_compute(cpu_run))
    for stage, expected in EXPECTED_STAGES.items():
        assert rows[stage]["cpu_s"] == pytest.approx(expected[_stage_index("cpu_s")]), stage


def test_group_peak_rss_includes_tree_sampled_processes(cpu_run):
    """REQ: peak RSS (kB) = largest peak RSS among the group's processes, tree-sampled ones included."""
    rows = group_rows(_compute(cpu_run))
    for stage, expected in EXPECTED_STAGES.items():
        assert rows[stage]["peak_rss_kb"] == expected[_stage_index("peak_rss_kb")], stage


def test_tree_only_processes_attributed_to_ancestor_segment_by_create_time(cpu_run):
    """REQ: a tree-only process goes to the nearest ancestor segment whose interval holds its create time."""
    report = _compute(cpu_run)
    groups = {pid: [entry["group"] for entry in find_process(report, HOST_A, pid)] for pid in (200, 201, 202)}
    # 200: child of the cortical_tiles worker; 201: grandchild through 200 (tree-only itself).
    assert groups[200] == ["cortical_tiles"]
    assert groups[201] == ["cortical_tiles"]
    # 202: child of main.py created during its morphologist segment, not its later cortical_tiles one.
    assert groups[202] == ["morphologist"]


def test_incomplete_sidecar_is_kept_and_counted(cpu_run):
    """REQ: a complete:false sidecar stays as a process entry flagged incomplete, counted in its group row."""
    report = _compute(cpu_run)
    killed = find_process(report, HOST_A, 102)
    assert len(killed) == 1
    assert killed[0]["complete"] is False
    assert killed[0]["group"] == "embeddings"
    rows = group_rows(report)
    for stage, expected in EXPECTED_STAGES.items():
        assert rows[stage]["incomplete_count"] == expected[_stage_index("incomplete_count")], stage


def test_missing_profile_file_does_not_stop_the_report(cpu_run):
    """REQ: with a sidecar naming an absent .prof, the group rows come from the remaining files, no exception."""
    (cpu_run / "proc" / f"{HOST_A}-100-0.prof").unlink()
    rows = group_rows(_compute(cpu_run))
    assert set(rows) == set(EXPECTED_STAGES)
    assert rows["morphologist"]["cpu_s"] == pytest.approx(EXPECTED_STAGES["morphologist"][_stage_index("cpu_s")])


def test_same_pid_on_two_hosts_gives_two_process_entries(cpu_run):
    """REQ: sidecars from different hosts sharing one PID are two distinct process entries."""
    report = _compute(cpu_run)
    on_a = find_process(report, HOST_A, 101)
    on_b = find_process(report, HOST_B, 101)
    assert len(on_a) == 1
    assert len(on_b) == 1
    assert on_a[0]["group"] == "cortical_tiles"
    assert on_b[0]["group"] == "embeddings"


def test_process_entry_fields(cpu_run):
    """REQ: each process entry carries host, PID, argv, group, wall, CPU, peak RSS (kB), exit path, completeness."""
    report = _compute(cpu_run)
    assert len(report["processes"]) == EXPECTED_PROCESS_COUNT
    for entry in report["processes"]:
        assert set(entry) >= PROCESS_ENTRY_KEYS, sorted(PROCESS_ENTRY_KEYS - set(entry))
    (entry,) = find_process(report, HOST_B, 101)
    assert entry["argv"][:2] == ["python", "generate_embeddings.py"]
    assert entry["group"] == "embeddings"
    assert entry["wall_s"] == pytest.approx(70.0)
    assert entry["cpu_s"] == pytest.approx(44.0)
    assert entry["peak_rss_kb"] == 300_000
    assert entry["exit_path"] == "atexit"
    assert entry["complete"] is True


# ---------------------------------------------------------------------------
# Top functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("top", [1, 3])
def test_top_functions_bounded_and_ordered_by_cumulative_time(cpu_run, top):
    """REQ: per group, at most `top` functions from the merged dumps, by decreasing cumulative time."""
    rows = group_rows(_compute(cpu_run, "stage", top))
    for stage, row in rows.items():
        functions = row["top_functions"]
        assert len(functions) <= top, stage
        cumulative = [function["cumulative_s"] for function in functions]
        assert cumulative == sorted(cumulative, reverse=True), stage
    # cortical_tiles has two dumps with more than `top` functions between them: the list is full.
    assert len(rows["cortical_tiles"]["top_functions"]) == top
    # embeddings: the killed child on nodeA has no dump; the nodeB dump alone still gives a list.
    assert rows["embeddings"]["top_functions"]


# ---------------------------------------------------------------------------
# Purity and other groupings
# ---------------------------------------------------------------------------


def _snapshot(directory) -> dict:
    return {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def test_compute_leaves_run_directory_unchanged(cpu_run):
    """REQ: compute_profile_report leaves the file listing and contents of the run directory unchanged."""
    before = _snapshot(cpu_run)
    _compute(cpu_run)
    assert _snapshot(cpu_run) == before


def test_group_by_process_gives_one_row_per_process_entry(cpu_run):
    """REQ: grouped by process, one group row per process entry."""
    report = _compute(cpu_run, "process")
    rows = group_rows(report)
    assert len(rows) == len(report["processes"]) == EXPECTED_PROCESS_COUNT
    assert all(row["process_count"] == 1 for row in rows.values())


def test_group_by_tag_gives_one_row_per_tag_value(cpu_run):
    """REQ: grouped by tag:<key>, one group row per distinct value of that key among the sidecar tags."""
    report = _compute(cpu_run, "tag:subject")
    rows = group_rows(report)  # fails on a duplicate row
    assert report["group_by"] == "tag:subject"
    assert {SUBJECT_1, SUBJECT_2} <= set(rows)
    # IXI002: the cortical_tiles worker on nodeA and the embeddings segment on nodeB.
    assert rows[SUBJECT_1]["process_count"] >= 2
    assert rows[SUBJECT_2]["process_count"] >= 1
