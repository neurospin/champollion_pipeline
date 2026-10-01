#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-region failure handling of GenerateEmbeddings._run_per_region (TASK-089).

REQ-EMBED-ANY-FAILURE-01..06. Every region's evaluate.py invocation goes
through ``execute_command``, which (champollion_utils ScriptBuilder) never
raises and reports failure only as a non-zero return code, so it is replaced
here by a fake returning a scripted code per region. No real evaluate.py runs.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings

pytestmark = pytest.mark.unit  # noqa: V107

MARKER = "<<fake-evaluate-done>>"


def make_models(tmp_path: Path, regions: list[str]) -> Path:
    """Create a flat models dir with one <region>/logs/ per region."""
    models = tmp_path / "models"
    for region in regions:
        (models / region / "logs").mkdir(parents=True)
    return models


def make_script(argv: list[str]) -> GenerateEmbeddings:
    script = GenerateEmbeddings()
    script.parse_args(argv)
    return script


def region_of(cmd: list[str]) -> str:
    """Region name from the evaluate command's -m model path."""
    return Path(cmd[cmd.index("-m") + 1]).name


class FakeEvaluate:
    """Stand-in for execute_command returning a scripted code per region.

    On success it writes ``content`` to the command's ``-s`` path, the way
    evaluate.py would; on failure it writes nothing.
    """

    def __init__(self, codes: dict[str, int], content: str = "new\n"):
        self.codes = codes
        self.content = content
        self.calls: list[str] = []

    def __call__(self, cmd, **_kwargs):
        region = region_of(cmd)
        self.calls.append(region)
        code = self.codes.get(region, 0)
        if code == 0:
            out = Path(cmd[cmd.index("-s") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(self.content)
        print(MARKER)
        return code


def run_stage(tmp_path: Path, regions: list[str], codes: dict[str, int]) -> int:
    """Run GenerateEmbeddings.run() end to end with mocked fetch/evaluate."""
    models = make_models(tmp_path, regions)
    out = tmp_path / "out"
    script = make_script([str(models), str(tmp_path / "dataset"), "--output", str(out)])
    script.execute_command = FakeEvaluate(codes)
    with patch.object(GenerateEmbeddings, "fetch_models", return_value=str(models)):
        return script.run()


REGIONS = ["Aaa_left", "Bbb_left", "Ccc_right"]  # processed in this (sorted) order


class TestRunReturnsNonZeroOnAnyRegionFailure:
    """REQ-EMBED-ANY-FAILURE-01."""

    def test_first_region_fails_last_succeeds(self, tmp_path):
        assert run_stage(tmp_path, REGIONS, {"Aaa_left": 1}) != 0

    def test_middle_region_fails(self, tmp_path):
        assert run_stage(tmp_path, REGIONS, {"Bbb_left": 2}) != 0

    def test_last_region_fails(self, tmp_path):
        assert run_stage(tmp_path, REGIONS, {"Ccc_right": 1}) != 0

    def test_all_regions_succeed_returns_zero(self, tmp_path):
        assert run_stage(tmp_path, REGIONS, {}) == 0


class TestFailedRegionsReported:
    """REQ-EMBED-ANY-FAILURE-02."""

    def test_failed_region_names_printed_after_last_region(self, tmp_path, capsys):
        models = make_models(tmp_path, REGIONS)
        script = make_script([str(models), "/d"])
        script.execute_command = FakeEvaluate({"Aaa_left": 1, "Bbb_left": 1})
        script._run_per_region("evaluate.py", str(tmp_path), str(tmp_path / "out"))

        stdout = capsys.readouterr().out
        after_last_region = stdout.rsplit(MARKER, 1)[1]
        assert "Aaa_left" in after_last_region
        assert "Bbb_left" in after_last_region


class TestProcessingContinuesAfterFailure:
    """REQ-EMBED-ANY-FAILURE-03."""

    def test_regions_after_failure_still_evaluated(self, tmp_path):
        models = make_models(tmp_path, REGIONS)
        script = make_script([str(models), "/d"])
        fake = FakeEvaluate({"Aaa_left": 1})
        script.execute_command = fake
        script._run_per_region("evaluate.py", str(tmp_path), str(tmp_path / "out"))
        assert fake.calls == REGIONS


class TestNoStaleOutputAfterFailedRecompute:
    """REQ-EMBED-ANY-FAILURE-04."""

    def test_overwrite_failure_removes_previous_output(self, tmp_path):
        models = make_models(tmp_path, ["Aaa_left"])
        out = tmp_path / "out"
        stale = out / "Aaa_left" / "full_embeddings.csv"
        stale.parent.mkdir(parents=True)
        stale.write_text("stale\n")

        script = make_script([str(models), "/d", "--overwrite"])
        script.execute_command = FakeEvaluate({"Aaa_left": 1})
        script._run_per_region("evaluate.py", str(tmp_path), str(out))

        assert not stale.exists()

    def test_missing_output_failure_leaves_no_file(self, tmp_path):
        models = make_models(tmp_path, ["Aaa_left"])
        out = tmp_path / "out"
        script = make_script([str(models), "/d"])
        script.execute_command = FakeEvaluate({"Aaa_left": 1})
        script._run_per_region("evaluate.py", str(tmp_path), str(out))

        assert not (out / "Aaa_left" / "full_embeddings.csv").exists()


class TestSkippedOutputUntouched:
    """REQ-EMBED-ANY-FAILURE-05."""

    def test_skipped_output_unchanged_when_other_region_fails(self, tmp_path):
        models = make_models(tmp_path, REGIONS)
        out = tmp_path / "out"
        existing = out / "Bbb_left" / "full_embeddings.csv"
        existing.parent.mkdir(parents=True)
        existing.write_text("kept\n")

        script = make_script([str(models), "/d"])
        fake = FakeEvaluate({"Aaa_left": 1, "Ccc_right": 1})
        script.execute_command = fake
        script._run_per_region("evaluate.py", str(tmp_path), str(out))

        assert "Bbb_left" not in fake.calls
        assert existing.read_text() == "kept\n"


class TestSuccessfulOutputKept:
    """REQ-EMBED-ANY-FAILURE-06."""

    def test_overwrite_success_keeps_new_output(self, tmp_path):
        models = make_models(tmp_path, ["Aaa_left"])
        out = tmp_path / "out"
        saving = out / "Aaa_left" / "full_embeddings.csv"
        saving.parent.mkdir(parents=True)
        saving.write_text("old\n")

        script = make_script([str(models), "/d", "--overwrite"])
        script.execute_command = FakeEvaluate({}, content="fresh\n")
        script._run_per_region("evaluate.py", str(tmp_path), str(out))

        assert saving.read_text() == "fresh\n"
