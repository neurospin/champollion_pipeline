#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CI workflow coverage of the pip-extras install route.

Run with: pixi run test-specific tests/test_ci_pip_extras_job.py

REQ-INSTALL-09: `.github/workflows/ci.yml` must contain a job step whose
`run` command executes `pip install -e .[embeddings]` (the `embeddings`
extra added by REQ-INSTALL-04). Today CI's only install path is
`pixi run --environment default install-all`, so the pip-extras route has
no CI coverage. This pins the job *definition* only; whether CI can run
green is blocked separately (TASK-052, org billing lock).
"""

import re
from pathlib import Path

import pytest
import yaml

CI_WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"

# `pip install -e .[embeddings]`, with `.[embeddings]` optionally wrapped in
# single or double quotes (both are equivalent in the runner's bash shell).
PIP_EXTRAS_INSTALL = re.compile(r"""\bpip\s+install\s+-e\s+(?P<q>["']?)\.\[embeddings\](?P=q)(?:\s|$)""")


def _workflow() -> dict:
    with CI_WORKFLOW.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _run_commands(workflow: dict) -> list[tuple[str, str]]:
    """Return (job_id, run_command) for every step with a `run` key."""
    commands = []
    for job_id, job in (workflow.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            if isinstance(step, dict) and "run" in step:
                commands.append((job_id, str(step["run"])))
    return commands


@pytest.mark.smoke
class TestCiPipExtrasInstallStep:
    """REQ-INSTALL-09: ci.yml exercises `pip install -e .[embeddings]`."""

    def test_ci_workflow_parses_with_jobs(self):
        """Anchor: ci.yml exists, is valid YAML, and declares at least one job."""
        workflow = _workflow()
        assert isinstance(workflow, dict)
        assert workflow.get("jobs"), "ci.yml declares no jobs"

    def test_some_step_runs_pip_install_embeddings_extra(self):
        """A step's `run` command executes `pip install -e .[embeddings]`."""
        commands = _run_commands(_workflow())
        matches = [(job_id, cmd) for job_id, cmd in commands if PIP_EXTRAS_INSTALL.search(cmd)]
        assert matches, (
            "No step in .github/workflows/ci.yml runs "
            "`pip install -e .[embeddings]`; found run commands: "
            f"{[cmd.strip().splitlines()[0] for _, cmd in commands]}"
        )
