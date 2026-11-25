#!/usr/bin/env python3
"""
Stage log records of generate_embeddings (LOGGING convention, TASK-LOGGING-BDRABCZUK-7132B94434A5).

Split from tests/test_generate_embeddings.py (TASK-CODE-BDRABCZUK-73BD87657EA2).
"""

import ast
import inspect
import logging
from unittest.mock import patch

import pytest

from champollion_pipeline import generate_embeddings as generate_embeddings_module
from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from tests.embeddings_parse_helpers import MASKS_ARGS
from tests.log_assertions import FAIL, LOG, OUTPUT, print_call_lines
from tests.region_model_dirs import make_region_logs

LOGGER_NAME = "champollion_pipeline.generate_embeddings"


def _region_script(tmp_path, regions, *extra_args):
    """A GenerateEmbeddings over tmp_path/models holding one epoch=80 region model directory per region."""
    models_dir = tmp_path / "models"
    for region in regions:
        make_region_logs(models_dir / region)
    script = GenerateEmbeddings()
    script.args = script.parse_args([str(models_dir), str(tmp_path), *MASKS_ARGS, *extra_args])
    return script


def _run_regions(script, tmp_path, exit_code=0):
    """Run _run_per_region with evaluate.py stubbed to return exit_code; return the stage status."""
    with patch.object(script, "execute_command", return_value=exit_code):
        return script._run_per_region(
            evaluate_script="/eval.py",
            crops_2mm_dir=str(tmp_path / "crops"),
            output_base=str(tmp_path / "out"),
        )


class _FakeStats:
    """pstats.Stats stand-in whose print_stats writes a fixed statistics header to its stream."""

    HEADER = "   Ordered by: cumulative time"

    def __init__(self, *_profilers, stream=None):
        self.stream = stream

    def sort_stats(self, *_keys):
        return self

    def dump_stats(self, _filename):
        return None

    def print_stats(self, *_restrictions):
        self.stream.write(f"{self.HEADER}\n")
        return self


@pytest.mark.unit
class TestEmbeddingsLogging:
    """Stage messages go through the champollion_pipeline.generate_embeddings logger at the mapped levels.

    REQ-LOGGING-BDRABCZUK-3244BE9D5C49: Fold <i>/<n> per region model directory at LOG.
    REQ-LOGGING-BDRABCZUK-DF78FDBC9C1B: run start with the output directory at LOG.
    REQ-LOGGING-BDRABCZUK-34DD57C23B5C: [SKIP] <region> at LOG.
    REQ-LOGGING-BDRABCZUK-2364137F954D: weights file path of a region at LOG.
    REQ-LOGGING-BDRABCZUK-82A9C5E4546A: best_model_weights.pt stand-in at WARNING.
    REQ-LOGGING-BDRABCZUK-A32D0E63055B: failed region names at FAIL.
    REQ-LOGGING-BDRABCZUK-7DB89B98D5DD: provenance.json path at OUTPUT.
    REQ-LOGGING-BDRABCZUK-EF548A46DAEC: --cpu (CUDA_VISIBLE_DEVICES) at LOG.
    REQ-LOGGING-BDRABCZUK-8A3FD0F22242: embeddings_profile.prof at OUTPUT.
    REQ-LOGGING-BDRABCZUK-F40F6E961CA2: profiling statistics at OUTPUT.
    REQ-LOGGING-BDRABCZUK-66706D8B409A: CKA output directory at LOG.
    REQ-LOGGING-BDRABCZUK-5D9C1F31AEA1: CKA completion at OUTPUT.
    REQ-LOGGING-BDRABCZUK-130243633649: CKA failure at WARNING.
    REQ-LOGGING-BDRABCZUK-1952DEF1166A: no print() call in the module.
    """

    def test_fold_progress_logged_per_region_model(self, tmp_path, pipeline_caplog):
        # champollion_sulcal_mcp reads "Fold N/M" from the stage output (runner.FOLD_RE).
        script = _region_script(tmp_path, ["SC-sylv_left", "SC-sylv_right"])

        _run_regions(script, tmp_path)

        for fold in (1, 2):
            record = pipeline_caplog.assert_fold_progress(fold, 2)
            assert (record.name, record.levelno) == (LOGGER_NAME, LOG)

    def test_fold_progress_logged_for_skipped_region(self, tmp_path, pipeline_caplog):
        script = _region_script(tmp_path, ["SC-sylv_left", "SC-sylv_right"])
        for region in ("SC-sylv_left", "SC-sylv_right"):
            (tmp_path / "out" / region).mkdir(parents=True)
            (tmp_path / "out" / region / "full_embeddings.csv").write_text("existing")

        _run_regions(script, tmp_path)

        for fold in (1, 2):
            record = pipeline_caplog.assert_fold_progress(fold, 2)
            assert (record.name, record.levelno) == (LOGGER_NAME, LOG)

    def test_run_start_logged_with_output_directory(self, tmp_path, pipeline_caplog):
        script = _region_script(tmp_path, ["SC-sylv_left"])

        _run_regions(script, tmp_path)

        record = pipeline_caplog.assert_logged(LOG, str(tmp_path / "out"))
        assert record.name == LOGGER_NAME

    def test_skipped_region_logged(self, tmp_path, pipeline_caplog):
        script = _region_script(tmp_path, ["SC-sylv_left"])
        (tmp_path / "out" / "SC-sylv_left").mkdir(parents=True)
        (tmp_path / "out" / "SC-sylv_left" / "full_embeddings.csv").write_text("existing")

        _run_regions(script, tmp_path)

        record = pipeline_caplog.assert_logged(LOG, "[SKIP] SC-sylv_left")
        assert record.name == LOGGER_NAME

    def test_region_weights_logged(self, tmp_path, pipeline_caplog):
        script = _region_script(tmp_path, ["SC-sylv_left"])
        checkpoints = tmp_path / "models" / "SC-sylv_left" / "logs" / "lightning_logs" / "version_0" / "checkpoints"

        _run_regions(script, tmp_path)

        record = pipeline_caplog.assert_logged(LOG, str(checkpoints / "epoch=80-step=1.ckpt"))
        assert record.name == LOGGER_NAME

    def test_best_weights_stand_in_logged_as_warning(self, tmp_path, pipeline_caplog):
        script = _region_script(tmp_path, ["SC-sylv_left"])

        with patch("champollion_pipeline.generate_embeddings.uses_best_weights_stand_in", return_value=True):
            _run_regions(script, tmp_path)

        records = [r for r in pipeline_caplog.records if (r.name, r.levelno) == (LOGGER_NAME, logging.WARNING)]
        assert any("SC-sylv_left" in r.getMessage() for r in records), [r.getMessage() for r in records]

    def test_failed_region_logged_at_fail(self, tmp_path, pipeline_caplog):
        script = _region_script(tmp_path, ["SC-sylv_left"])

        assert _run_regions(script, tmp_path, exit_code=3) == 3

        record = pipeline_caplog.assert_logged(FAIL, "SC-sylv_left")
        assert record.name == LOGGER_NAME

    def test_provenance_path_logged_at_output(self, tmp_path, pipeline_caplog):
        script = GenerateEmbeddings()

        path = script._write_provenance(str(tmp_path / "out"), {"masks": "canonical_25"})

        record = pipeline_caplog.assert_logged(OUTPUT, path)
        assert path == str(tmp_path / "out" / "provenance.json")
        assert record.name == LOGGER_NAME

    def test_cpu_mode_logged(self, tmp_path, monkeypatch, pipeline_caplog):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        script = GenerateEmbeddings()
        script.parse_args([str(tmp_path), str(tmp_path), *MASKS_ARGS, "--cpu"])

        with patch.object(script, "_run_pipeline", return_value=0):
            script._run_normal()

        record = pipeline_caplog.assert_logged(LOG, "CUDA_VISIBLE_DEVICES")
        assert record.name == LOGGER_NAME

    @staticmethod
    def _run_profiled(temp_dir) -> None:
        """Parse --profiling and run _run_with_profiling with the profiler and _run_normal mocked."""
        script = GenerateEmbeddings()
        script.parse_args([temp_dir, temp_dir, *MASKS_ARGS, "--profiling"])

        with (
            patch("champollion_pipeline.generate_embeddings.cProfile.Profile"),
            patch("champollion_pipeline.generate_embeddings.pstats.Stats", _FakeStats),
            patch.object(script, "_run_normal", return_value=0),
        ):
            script._run_with_profiling()

    def test_profile_file_logged_at_output(self, temp_dir, pipeline_caplog):
        self._run_profiled(temp_dir)
        record = pipeline_caplog.assert_logged(OUTPUT, "embeddings_profile.prof")
        assert record.name == LOGGER_NAME

    def test_profile_statistics_logged_at_output(self, temp_dir, pipeline_caplog):
        self._run_profiled(temp_dir)
        record = pipeline_caplog.assert_logged(OUTPUT, _FakeStats.HEADER.strip())
        assert record.name == LOGGER_NAME

    def test_cka_output_directory_logged(self, tmp_path, pipeline_caplog):
        script = GenerateEmbeddings()

        with patch("champollion_pipeline.generate_embeddings.test_models_coherence_from_directory"):
            script._run_cka_test(str(tmp_path / "embeddings"))

        record = pipeline_caplog.assert_logged(LOG, str(tmp_path / "embeddings" / "cka_results"))
        assert record.name == LOGGER_NAME

    def test_cka_completion_logged_at_output(self, tmp_path, pipeline_caplog):
        script = GenerateEmbeddings()

        with patch("champollion_pipeline.generate_embeddings.test_models_coherence_from_directory"):
            script._run_cka_test(str(tmp_path / "embeddings"))

        record = pipeline_caplog.assert_logged(OUTPUT, "CKA coherence test completed")
        assert record.name == LOGGER_NAME

    def test_cka_failure_logged_at_warning(self, tmp_path, pipeline_caplog):
        script = GenerateEmbeddings()

        with patch(
            "champollion_pipeline.generate_embeddings.test_models_coherence_from_directory",
            side_effect=RuntimeError("CKA matrix is singular"),
        ):
            script._run_cka_test(str(tmp_path / "embeddings"))

        record = pipeline_caplog.assert_logged(logging.WARNING, "CKA matrix is singular")
        assert record.name == LOGGER_NAME

    def test_module_has_no_print_call(self):
        tree = ast.parse(inspect.getsource(generate_embeddings_module))

        assert print_call_lines(tree) == []
