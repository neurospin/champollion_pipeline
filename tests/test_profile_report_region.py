"""Red test for the report region of champollion_V1 evaluate.py processes (PNONPY).

Task TASK-TASK-BDRABCZUK-8DF4AED957EE; design COMP-ARCH-BDRABCZUK-EC857620B0EB
(cross-process-profiling), Contract 4: champollion_V1 stays frozen, so the region of an
evaluate.py child is derived at report time from its ``-s <saving path>`` argument
(stage 4 writes ``<output_base>/<region>/full_embeddings.csv``). Requirement
"Report region of an evaluate.py process" (area PROFREPORT).
"""

import pytest

from tests.profile_run_fixtures import (
    HOST_A,
    _tree_line,
    _write_run,
    find_process,
    load_profile_report,
)

EVALUATE = "/opt/champollion_pipeline/external/champollion_V1/champollion/evaluate.py"
EMBEDDINGS_PID = 500
EVALUATE_PID = 501
REGION = "S.C.-sylv._left"


def _evaluate_argv(region: str) -> list[str]:
    return [
        EVALUATE,
        "-m",
        f"/models/{region}",
        "-sk",
        f"/crops/{region}/mask/Lskeleton.npy",
        "-i",
        f"/crops/{region}/mask/Lskeleton_subject.csv",
        "-s",
        f"/out/embeddings/{region}/full_embeddings.csv",
    ]


def _embeddings_segment() -> dict:
    return {
        "host": HOST_A,
        "pid": EMBEDDINGS_PID,
        "ppid": 1,
        "seq": 0,
        "stage": "generate_embeddings",
        "argv": ["generate_embeddings.py"],
        "t_start": 0.0,
        "t_end": 100.0,
        "cpu_user_s": 2.0,
        "cpu_sys_s": 0.5,
        "maxrss_kb": 50_000,
    }


def _run_with_sidecar(run_dir):
    """The evaluate.py process wrote its own sidecar (profiled through the boot)."""
    evaluate = {
        "host": HOST_A,
        "pid": EVALUATE_PID,
        "ppid": EMBEDDINGS_PID,
        "seq": 0,
        "stage": "generate_embeddings",
        "argv": _evaluate_argv(REGION),
        "t_start": 10.0,
        "t_end": 60.0,
        "cpu_user_s": 40.0,
        "cpu_sys_s": 2.0,
        "maxrss_kb": 900_000,
    }
    _write_run(run_dir, [_embeddings_segment(), evaluate], [], "none")


def _run_tree_only(run_dir):
    """The evaluate.py process is seen only by the tree sampler."""
    cmdline = ["python3.11", *_evaluate_argv(REGION)]
    lines = [
        _tree_line(t, EVALUATE_PID, EMBEDDINGS_PID, 10.0, cmdline=cmdline, cpu_user_s=t, cpu_sys_s=1.0, rss_kb=800_000)
        for t in (20.0, 40.0)
    ]
    _write_run(run_dir, [_embeddings_segment()], lines, "none")


@pytest.mark.parametrize("build", [_run_with_sidecar, _run_tree_only], ids=["sidecar", "tree-only"])
def test_group_by_region_places_evaluate_process_in_saving_path_region(tmp_path, build):
    """Report region of an evaluate.py process (group named after the parent directory of -s)."""
    run_dir = tmp_path / "run"
    build(run_dir)
    report = load_profile_report().compute_profile_report(run_dir, "tag:region", 20)
    entries = find_process(report, HOST_A, EVALUATE_PID)
    assert entries, f"no process entry for evaluate.py pid {EVALUATE_PID}: {report['processes']}"
    assert {entry["group"] for entry in entries} == {REGION}
    assert REGION in {row["group"] for row in report["groups"]}
